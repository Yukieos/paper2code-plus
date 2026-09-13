from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from state import PipelineState, initial_state
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


def set_api_keys_inline() -> None:
    """Load API credentials from the environment / a local .env file.

    Never overwrites a key the caller already set (e.g. via the Streamlit UI
    or a real shell export) — it only fills in gaps and reports what's live.
    """
    if load_dotenv is not None:
        load_dotenv(override=False)

    os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
    os.environ.setdefault("OPENAI_API_BASE", os.environ["OPENAI_BASE_URL"])

    def _mask(v: str | None) -> str:
        return ("****" + v[-4:]) if v and len(v) >= 4 else "(unset)"

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, put it in a .env file, "
            "or enter it in the Streamlit sidebar."
        )

    print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL", "(unset)"))
    print("OPENAI_API_KEY:", _mask(os.environ.get("OPENAI_API_KEY")))


def configure_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "agent_pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UPS-IR LangChain/LangGraph pipeline runner")
    parser.add_argument("md_path", type=Path, help="Path to the source Markdown file")
    parser.add_argument(
        "--agents",
        type=str,
        default="reader,extractor,structurer,verifier,synthesizer",
        help="Comma-separated list indicating which agents to run in order.",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip the picture description augmentation step in ReaderAgent.",
    )
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Execute the pipeline via LangGraph's StateGraph instead of sequential calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name for ExtractorAgent (LangChain ChatOpenAI).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for ExtractorAgent LLM.",
    )

    return parser.parse_args(list(argv))


def create_agents(agent_names: list[str], args: argparse.Namespace) -> Dict[str, object]:
    agents: Dict[str, object] = {}

    if "reader" in agent_names:
        from agents.reader import ReaderAgent, ReaderConfig  # lazy import

        agents["reader"] = ReaderAgent(
            ReaderConfig(
                md_path=args.md_path,
                annotate_images=not args.skip_annotation,
                annotated_output=Path("output") / "paper_with_images.md",
            )
        )

    if "extractor" in agent_names or args.use_graph:
        from agents.extractor import ExtractorAgent, ExtractorConfig  # lazy import

        agents["extractor"] = ExtractorAgent(
            ExtractorConfig(
                output_path=Path("output") / "extracted_info.json",
                model_name=args.model,
                temperature=args.temperature,
            )
        )

    if "structurer" in agent_names or args.use_graph:
        from agents.structurer import StructurerAgent, StructurerConfig  # lazy import

        agents["structurer"] = StructurerAgent(
            StructurerConfig(output_path=Path("UPS-IR.json"))
        )

    if "verifier" in agent_names or args.use_graph:
        from agents.verifier import VerifierAgent  # lazy import

        agents["verifier"] = VerifierAgent()

    if "synthesizer" in agent_names or args.use_graph:
        from agents.synthesizer import SynthesizerAgent, SynthesizerConfig  # lazy import

        agents["synthesizer"] = SynthesizerAgent(
            SynthesizerConfig(output_dir=Path("UPS-IR_Output"))
        )

    return agents


def parse_agent_sequence(agent_string: str) -> List[str]:
    sequence = [item.strip().lower() for item in agent_string.split(",") if item.strip()]
    valid_order = ["reader", "extractor", "structurer", "verifier", "synthesizer"]

    for agent_name in sequence:
        if agent_name not in valid_order:
            raise ValueError(f"Unsupported agent '{agent_name}'. Valid options: {', '.join(valid_order)}")

    # Preserve global order but allow omissions or reordering within the valid list.
    return sequence


MAX_VERIFICATION_RETRIES = 2


def run_sequential(agent_names: List[str], agents: Dict[str, object]) -> PipelineState:
    """Runs the extraction agents in order — except this isn't purely
    fixed-sequence: on a VerifierAgent rejection, this backtracks to the
    extractor with the specific validation error as feedback instead of
    just halting (bounded to MAX_VERIFICATION_RETRIES). Two other actions
    were considered and deliberately left out: retrying verification itself
    (pointless — it's deterministic given the same UPS-IR) and backtracking
    past the extractor to the reader (there's nothing wrong with the raw
    text on a schema/integrity failure, only with how it was structured).
    """
    from eval.instrumentation import default_run_id, run_traced
    from eval.trace_store import TraceStore

    state: PipelineState = initial_state()
    run_id = default_run_id()
    store = TraceStore()
    can_retry = "extractor" in agent_names
    retries_used = 0

    i = 0
    while i < len(agent_names):
        name = agent_names[i]
        agent = agents[name]

        if name == "verifier":
            try:
                verified = run_traced(agent, state, run_id=run_id, stage="extraction",
                                       agent_name=name, store=store)  # type: ignore[attr-defined]
                verifier_error = None
            except ValueError as exc:
                verified = False
                verifier_error = str(exc)

            state.setdefault("artifacts", {})
            state["artifacts"]["verified"] = str(bool(verified))

            if verified:
                i += 1
                continue

            if can_retry and retries_used < MAX_VERIFICATION_RETRIES:
                retries_used += 1
                logging.warning(
                    "Verification failed (%s); backtracking to extractor with feedback "
                    "(attempt %d/%d) [ReAct: RETRY_WITH_FEEDBACK]",
                    verifier_error, retries_used, MAX_VERIFICATION_RETRIES,
                )
                state["verifier_feedback"] = verifier_error
                i = agent_names.index("extractor")
                continue

            logging.error("Verification failed after %d retries. Halting pipeline before synthesis.",
                           retries_used)
            break

        state = run_traced(agent, state, run_id=run_id, stage="extraction",
                            agent_name=name, store=store)  # type: ignore[attr-defined]
        i += 1

    if "ups_ir" not in state:
        logging.warning("UPS-IR structure not present in state after sequential run.")

    return state


def run_via_graph(agents: Dict[str, object]) -> PipelineState:
    """LangGraph version of the same ReAct backtrack in run_sequential,
    expressed the idiomatic LangGraph way: a conditional edge out of
    "verifier" that routes back to "extractor" (with feedback) on
    rejection, forward to "synthesizer" on success, or to END once the
    retry budget is spent — a real cycle in the graph, not a fixed chain.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Install it or run without --use-graph.") from exc

    builder: StateGraph = StateGraph(PipelineState)

    builder.add_node("reader", lambda s: agents["reader"].run(s))  # type: ignore[attr-defined]
    builder.add_node("extractor", lambda s: agents["extractor"].run(s))  # type: ignore[attr-defined]
    builder.add_node("structurer", lambda s: agents["structurer"].run(s))  # type: ignore[attr-defined]

    def verifier_wrapper(state: PipelineState) -> PipelineState:
        try:
            verified = agents["verifier"].run(state)  # type: ignore[attr-defined]
            state["artifacts"] = {**state.get("artifacts", {}), "verified": str(bool(verified))}
        except ValueError as exc:
            state["artifacts"] = {**state.get("artifacts", {}), "verified": "False"}
            state["verifier_feedback"] = str(exc)
        return state

    def route_after_verifier(state: PipelineState) -> str:
        if state.get("artifacts", {}).get("verified") == "True":
            return "synthesizer"

        retries = state.get("verification_retries", 0)
        if retries < MAX_VERIFICATION_RETRIES:
            state["verification_retries"] = retries + 1
            logging.warning(
                "Verification failed (%s); backtracking to extractor with feedback "
                "(attempt %d/%d) [ReAct: RETRY_WITH_FEEDBACK]",
                state.get("verifier_feedback"), retries + 1, MAX_VERIFICATION_RETRIES,
            )
            return "extractor"

        logging.error("Verification failed after %d retries. Halting pipeline before synthesis.",
                       retries)
        return END

    builder.add_node("verifier", verifier_wrapper)
    builder.add_node("synthesizer", lambda s: agents["synthesizer"].run(s))  # type: ignore[attr-defined]

    builder.set_entry_point("reader")
    builder.add_edge("reader", "extractor")
    builder.add_edge("extractor", "structurer")
    builder.add_edge("structurer", "verifier")
    builder.add_conditional_edges("verifier", route_after_verifier,
                                   {"synthesizer": "synthesizer", "extractor": "extractor", END: END})
    builder.add_edge("synthesizer", END)

    graph = builder.compile()
    return graph.invoke(initial_state())


def main(argv: Iterable[str] | None = None) -> PipelineState:
    # Inline keys for local testing (user requested). Remove for production use.
    set_api_keys_inline()
    configure_logging()
    set_llm_cache(InMemoryCache())

    args = parse_args(argv or sys.argv[1:])

    if not args.md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {args.md_path}")

    # Decide which agents to import/build before importing heavy deps
    agent_sequence = parse_agent_sequence(args.agents)
    agents = create_agents(agent_sequence, args)

    if args.use_graph:
        logging.info("Running pipeline via LangGraph.")
        state = run_via_graph(agents)
    else:
        logging.info("Running pipeline sequentially with agents: %s", ", ".join(agent_sequence))
        state = run_sequential(agent_sequence, agents)

    logging.info("Pipeline execution completed.")
    return state


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        logging.exception("Pipeline execution failed: %s", exc)
        sys.exit(1)
