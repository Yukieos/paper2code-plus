"""Root-cause diagnosis for a failure mode, given the runs that exhibit it.

Swappable `llm_call` (same convention as eval/judge.py) so this can be unit
tested without hitting a real API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from eval.mine_failures import RunResult
from eval.taxonomy import FailureMode, TAXONOMY_BY_MODE
from improve.prompt_registry import registered_agents

logger = logging.getLogger(__name__)

DIAGNOSIS_PROMPT_TEMPLATE = """You are diagnosing a recurring failure in the Paper2Code multi-agent pipeline.

Failure mode: {mode} — {description}

Evidence from {n} run(s) exhibiting this failure:
{evidence}

Task: identify the most likely root cause — which agent's prompt is most
responsible, and specifically what about its current instructions leads to
this failure mode. Be concrete: point at what's missing, ambiguous, or wrong
in the prompt, not just a restatement of the symptom.

`responsible_agent` MUST be EXACTLY one of these tunable agent names (or null
if no single one of them owns this failure) — do not invent a name, and do not
describe a role in your own words:
{agent_names}

Respond with ONLY a JSON object of this exact shape:
{{
  "responsible_agent": "<one of the agent names listed above, verbatim, or null>",
  "root_cause": "<2-4 sentences, concrete and specific>",
  "confidence": <float 0-1>
}}
"""


@dataclass
class Diagnosis:
    mode: FailureMode
    responsible_agent: str | None
    root_cause: str
    confidence: float
    run_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "responsible_agent": self.responsible_agent,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "run_ids": self.run_ids,
        }


def _evidence_block(mode: FailureMode, results: list[RunResult], limit: int = 4000) -> str:
    lines = []
    for result in results:
        matching = [s for s in result.grader_signals.signals if s.mode == mode]
        for signal in matching:
            lines.append(f"- run {result.run_id}: {signal.detail} ({signal.evidence[:300]})")
        if result.judge_verdict and result.judge_verdict.primary_failure_mode == mode:
            lines.append(f"- run {result.run_id} (LLM judge): {result.judge_verdict.rationale}")
    text = "\n".join(lines) or "(no detailed evidence captured)"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _default_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    return llm.invoke(prompt).content


def _extract_json(text: str) -> str:
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in diagnosis output: {text[:200]!r}")
    return text[start:end + 1]


def diagnose(
    mode: FailureMode,
    results: list[RunResult],
    llm_call: Callable[[str], str] | None = None,
    model: str = "gpt-4o-mini",
) -> Diagnosis:
    call = llm_call or (lambda prompt: _default_llm_call(prompt, model=model))
    spec = TAXONOMY_BY_MODE[mode]
    matching_results = [r for r in results if mode in r.all_modes]
    valid_agents = registered_agents()

    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
        mode=mode.value,
        description=spec.description,
        n=len(matching_results),
        evidence=_evidence_block(mode, matching_results),
        agent_names="\n".join(f"- {name}" for name in valid_agents),
    )

    try:
        data = json.loads(_extract_json(call(prompt)), strict=False)  # tolerate literal newlines in root_cause
    except Exception as exc:  # pragma: no cover - network/model dependent
        logger.warning("Diagnosis call failed (%s).", exc)
        return Diagnosis(mode, None, f"Diagnosis failed: {exc}", 0.0, [r.run_id for r in matching_results])

    # The model can still return a name that isn't a registered agent (e.g. a
    # made-up "code_generator"). propose_change can only act on a registered
    # one, so anything else becomes None here — the loop then stops cleanly
    # instead of silently no-opping one step later with a confusing warning.
    responsible = data.get("responsible_agent") or None
    if responsible and responsible not in valid_agents:
        logger.warning("Diagnosis named unregistered agent %r; treating as no specific agent. "
                       "Valid agents: %s", responsible, ", ".join(valid_agents))
        responsible = None

    return Diagnosis(
        mode=mode,
        responsible_agent=responsible,
        root_cause=str(data.get("root_cause", "")),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        run_ids=[r.run_id for r in matching_results],
    )
