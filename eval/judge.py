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

For primary_failure_mode:
- use one of the taxonomy ids above when the failure clearly matches it;
- use "other" if there IS a real failure but NONE of the ids above fit it
  (do not force a bad match into a named mode);
- use null ONLY if the run looks healthy / has no failure.

Respond with ONLY a JSON object of this exact shape:
{{
  "primary_failure_mode": "<a taxonomy id above, or \\"other\\", or null>",
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
    def __init__(self, llm_call: Callable[[str], str] | None = None, model: str = "gpt-4o-mini",
                 votes: int = 3):
        self.model = model
        self.llm_call = llm_call or (lambda prompt: _default_llm_call(prompt, model=model))
        # A single LLM judgment is noisy. Voting `votes` times and aggregating
        # (majority for whether-it-failed and the mode, median for confidence)
        # is what makes the classification stable enough to drive an automated
        # change. votes=1 recovers the old single-shot behavior.
        self.votes = max(1, votes)

    def _classify_once(self, prompt: str) -> JudgeVerdict:
        try:
            raw = self.llm_call(prompt)
            data = json.loads(_extract_json(raw), strict=False)  # tolerate literal newlines in rationale
        except Exception as exc:  # pragma: no cover - network/model dependent
            logger.warning("LLM judge call failed or returned unparseable output (%s); leaving run unclassified.", exc)
            return JudgeVerdict(None, [], 0.0, f"Judge call failed: {exc}")

        primary = _safe_mode(data.get("primary_failure_mode"))
        secondary = [m for m in (_safe_mode(m) for m in data.get("secondary_failure_modes", []) or []) if m]
        return JudgeVerdict(primary, secondary, float(data.get("confidence", 0.0) or 0.0),
                            str(data.get("rationale", "")))

    def classify(self, traces: list[TraceRecord], grader_report: GraderReport) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            rubric=render_rubric(),
            grader_signals=json.dumps(grader_report.to_dict(), indent=2) if grader_report.signals else "(none)",
            trace_summary=_summarize_traces(traces),
        )
        verdicts = [self._classify_once(prompt) for _ in range(self.votes)]
        return _aggregate_verdicts(verdicts, self.votes)


def _aggregate_verdicts(verdicts: list[JudgeVerdict], votes: int) -> JudgeVerdict:
    """Combine N judge votes: majority decides whether it failed and which mode;
    confidence is the median across votes; a mode is 'secondary' if it appears
    in a majority of votes."""
    import statistics

    n = len(verdicts)
    majority = n // 2 + 1
    confidence = statistics.median([v.confidence for v in verdicts]) if verdicts else 0.0

    # Majority vote on whether this is a failure at all.
    primaries = [v.primary_failure_mode for v in verdicts if v.primary_failure_mode]
    if len(primaries) < majority:
        return JudgeVerdict(None, [], confidence,
                            f"{n - len(primaries)}/{n} votes saw no unexplained failure.")

    # Mode = plurality of the non-null primary votes (ties -> most_common order).
    from collections import Counter
    primary = Counter(primaries).most_common(1)[0][0]

    sec_counts = Counter(m for v in verdicts for m in set(v.secondary_failure_modes))
    secondary = [m for m, c in sec_counts.items() if c >= majority and m != primary]

    rationale = next((v.rationale for v in verdicts if v.primary_failure_mode == primary), "")
    agree = sum(1 for p in primaries if p == primary)
    return JudgeVerdict(primary, secondary, confidence,
                        f"[{agree}/{n} votes] {rationale}")


def _safe_mode(value: str | None) -> FailureMode | None:
    if not value:
        return None
    try:
        return FailureMode(value)
    except ValueError:
        # The judge named a failure but not one of our ids -> that's exactly the
        # OTHER (taxonomy-expansion) signal, not something to silently drop.
        logger.info("Judge returned unrecognized failure id %r; recording as OTHER.", value)
        return FailureMode.OTHER


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
