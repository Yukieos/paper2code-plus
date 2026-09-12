"""LLM-as-judge: classify a run's remaining ambiguity into the taxonomy.

Deterministic graders (graders.py) catch the mechanical stuff (syntax
errors, a rejected verification, a timed-out tool call). What's left —
"did the extractor actually understand the paper", "does the plan match
what the paper asked for" — needs a model to read the trace and judge it.
This is deliberately a thin, swappable piece: pass any callable matching
`Callable[[str], str]` as `llm_call` for testing without hitting a real API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from eval.graders import GraderReport
from eval.taxonomy import FailureMode, render_rubric
from eval.trace_schema import TraceRecord

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are auditing one run of the Paper2Code multi-agent pipeline for failures.

Below is the taxonomy of known failure modes:
{rubric}

Deterministic checks already flagged these signals for this run (may be empty):
{grader_signals}

Here is a condensed trace of what each agent did (may be truncated):
{trace_summary}

Task: decide whether this run has a failure not already fully explained by the
deterministic signals above (e.g. the extractor misreading the paper, or the
plan drifting from the UPS-IR). If the deterministic signals already fully
explain the run's failures, or the run looks healthy, say so.

Respond with ONLY a JSON object of this exact shape:
{{
  "primary_failure_mode": "<one of the taxonomy ids above, or null if none>",
  "secondary_failure_modes": ["<taxonomy id>", ...],
  "confidence": <float 0-1>,
  "rationale": "<one or two sentences>"
}}
"""


@dataclass
class JudgeVerdict:
    primary_failure_mode: FailureMode | None
    secondary_failure_modes: list[FailureMode]
    confidence: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "primary_failure_mode": self.primary_failure_mode.value if self.primary_failure_mode else None,
            "secondary_failure_modes": [m.value for m in self.secondary_failure_modes],
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def _summarize_traces(traces: list[TraceRecord], limit: int = 6000) -> str:
    lines = []
    for record in traces:
        status = "OK" if record.success else f"FAILED ({record.error})"
        lines.append(f"- [{record.stage}/{record.agent_name}] {status}; output: {record.output_excerpt or ''}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _default_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    return llm.invoke(prompt).content


class LLMJudge:
    def __init__(self, llm_call: Callable[[str], str] | None = None, model: str = "gpt-4o-mini"):
        self.model = model
        self.llm_call = llm_call or (lambda prompt: _default_llm_call(prompt, model=model))

    def classify(self, traces: list[TraceRecord], grader_report: GraderReport) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            rubric=render_rubric(),
            grader_signals=json.dumps(grader_report.to_dict(), indent=2) if grader_report.signals else "(none)",
            trace_summary=_summarize_traces(traces),
        )

        try:
            raw = self.llm_call(prompt)
            data = json.loads(_extract_json(raw))
        except Exception as exc:  # pragma: no cover - network/model dependent
            logger.warning("LLM judge call failed or returned unparseable output (%s); leaving run unclassified.", exc)
            return JudgeVerdict(None, [], 0.0, f"Judge call failed: {exc}")

        primary_raw = data.get("primary_failure_mode")
        primary = _safe_mode(primary_raw)
        secondary = [m for m in (_safe_mode(m) for m in data.get("secondary_failure_modes", []) or []) if m]

        return JudgeVerdict(
            primary_failure_mode=primary,
            secondary_failure_modes=secondary,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            rationale=str(data.get("rationale", "")),
        )


def _safe_mode(value: str | None) -> FailureMode | None:
    if not value:
        return None
    try:
        return FailureMode(value)
    except ValueError:
        logger.warning("Judge returned unknown failure mode id: %r", value)
        return None


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")
    return text[start:end + 1]
