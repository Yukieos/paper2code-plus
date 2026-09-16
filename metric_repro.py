"""Reproduction fidelity: does the generated repo actually hit the numbers the
paper claims?

Three deterministic pieces (no LLM):
  1. claimed_metrics_from_ups_ir  — the paper's claimed headline numbers are
     already structured in UPS-IR experiments[*].metrics, so read them straight.
  2. parse_achieved_metrics       — scan a completed run's output/logs for the
     metric it actually produced (analogous to smoke_run's loss parser).
  3. compare_metrics              — match claimed vs achieved by normalized name,
     put both on a common scale, and decide within-tolerance.

This is the layer that answers "复现得像不像作者数字". It is depth-agnostic: a
short run yields a partial number, a full run the real one — the comparison is
the same. Actually producing a trustworthy achieved number needs a real (often
full) training run, which is why this is separate from the smoke layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Metrics that live on a [0,1] (or 0-100%) scale, so a "%" or a >1 value can be
# normalized to a fraction for comparison. Others (BLEU, PSNR dB, RMSE, R2, ...)
# are left on their own scale.
_RATIO_METRICS = {"accuracy", "f1", "precision", "recall", "auc", "auroc", "iou", "dice", "map"}

_SYNONYMS = {
    "acc": "accuracy", "top1": "accuracy", "top-1": "accuracy", "top1accuracy": "accuracy",
    "f1score": "f1", "f1-score": "f1", "aucroc": "auc", "roc": "auc", "psnrdb": "psnr",
}
_QUALIFIERS = ("test", "val", "validation", "train", "training", "eval", "evaluation",
               "overall", "mean", "average", "avg", "final", "best", "top-1", "top1")


def _norm_name(name: str) -> str:
    n = name.strip().lower()
    for q in _QUALIFIERS:
        if n.startswith(q + " ") or n.startswith(q + "_"):
            n = n[len(q):].strip(" _")
    n = re.sub(r"[\s_]+", "", n)
    return _SYNONYMS.get(n, n)


@dataclass
class ClaimedMetric:
    name: str
    value: float
    unit: str = ""
    dataset: str = ""
    source: str = ""


@dataclass
class AchievedMetric:
    name: str
    value: float
    unit: str = ""


@dataclass
class MetricComparison:
    name: str
    claimed: float
    achieved: float | None
    unit: str
    matched: bool          # did we find an achieved metric of this name at all
    gap: float | None      # |achieved - claimed| on the common scale
    rel_gap: float | None  # gap / |claimed|
    within_tolerance: bool

    def summary(self) -> str:
        if not self.matched:
            return f"{self.name}: claimed {self.claimed}{self.unit} — NOT PRODUCED by the run"
        verdict = "MATCH" if self.within_tolerance else "OFF"
        return (f"{self.name}: claimed {self.claimed} vs achieved {self.achieved} "
                f"(rel gap {self.rel_gap:.1%}) -> {verdict}")


def claimed_metrics_from_ups_ir(ups_ir: dict[str, Any]) -> list[ClaimedMetric]:
    """Headline claimed numbers, straight from UPS-IR experiments[*].metrics."""
    out: list[ClaimedMetric] = []
    for exp in ups_ir.get("experiments", []) or []:
        for m in exp.get("metrics", []) or []:
            name, value = m.get("name"), m.get("value")
            if name is None or value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            out.append(ClaimedMetric(name=str(name), value=value, unit=str(m.get("unit", "") or ""),
                                     dataset=str(exp.get("dataset", "") or ""),
                                     source=str(exp.get("source_reference", "") or "")))
    return out


# "accuracy: 0.9834", "test acc = 98.3%", "BLEU 28.4", "F1-score: 0.71"
_METRIC_RE = re.compile(
    r"\b(top-?1 accuracy|test accuracy|val(?:idation)? accuracy|accuracy|acc|f1[- ]?score|f1|"
    r"precision|recall|auroc|auc|bleu|rouge|psnr|ssim|rmse|mae|mse|r2|r\^2|iou|dice|map)\b"
    r"\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*(%?)",
    re.IGNORECASE,
)


def parse_achieved_metrics(text: str) -> list[AchievedMetric]:
    """Every metric value mentioned in the run's output/logs, in order."""
    out: list[AchievedMetric] = []
    for m in _METRIC_RE.finditer(text):
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        out.append(AchievedMetric(name=m.group(1), value=value, unit=m.group(3) or ""))
    return out


def _to_common_scale(name: str, value: float, unit: str) -> float:
    """Ratio metrics given as a percentage (unit '%' or a value > 1) become a
    fraction so claimed and achieved compare on one scale."""
    if _norm_name(name) in _RATIO_METRICS and (unit == "%" or value > 1.0):
        return value / 100.0
    return value


def compare_metrics(
    claimed: list[ClaimedMetric],
    achieved: list[AchievedMetric],
    rel_tol: float = 0.05,
    abs_tol: float | None = None,
) -> list[MetricComparison]:
    """Match each claimed metric to the LAST achieved metric of the same
    normalized name (the last is usually the final/best reported value)."""
    by_name: dict[str, AchievedMetric] = {}
    for a in achieved:
        by_name[_norm_name(a.name)] = a  # last wins

    comparisons: list[MetricComparison] = []
    for c in claimed:
        key = _norm_name(c.name)
        a = by_name.get(key)
        c_scaled = _to_common_scale(c.name, c.value, c.unit)
        if a is None:
            comparisons.append(MetricComparison(c.name, c_scaled, None, c.unit, False, None, None, False))
            continue
        a_scaled = _to_common_scale(a.name, a.value, a.unit)
        gap = abs(a_scaled - c_scaled)
        rel_gap = gap / abs(c_scaled) if c_scaled else float("inf")
        within = rel_gap <= rel_tol or (abs_tol is not None and gap <= abs_tol)
        comparisons.append(MetricComparison(c.name, c_scaled, a_scaled, c.unit, True, gap, rel_gap, within))
    return comparisons


@dataclass
class ReproFidelity:
    comparisons: list[MetricComparison] = field(default_factory=list)

    @property
    def reproduced(self) -> bool:
        """Every claimed metric was produced AND landed within tolerance."""
        return bool(self.comparisons) and all(c.within_tolerance for c in self.comparisons)

    @property
    def matched_any(self) -> bool:
        return any(c.matched for c in self.comparisons)

    def summary(self) -> str:
        if not self.comparisons:
            return "no claimed metrics found in UPS-IR to check against"
        head = "REPRODUCED" if self.reproduced else ("PARTIAL" if self.matched_any else "NO METRICS PRODUCED")
        return head + " | " + "; ".join(c.summary() for c in self.comparisons)


def assess_fidelity(ups_ir: dict[str, Any], run_output: str,
                    rel_tol: float = 0.05, abs_tol: float | None = None) -> ReproFidelity:
    """Top-level: claimed (from UPS-IR) vs achieved (parsed from a completed
    run's combined output/logs), compared with tolerance."""
    claimed = claimed_metrics_from_ups_ir(ups_ir)
    achieved = parse_achieved_metrics(run_output)
    return ReproFidelity(compare_metrics(claimed, achieved, rel_tol=rel_tol, abs_tol=abs_tol))


def collect_run_output(repo_dir: Path, stdout: str = "", stderr: str = "") -> str:
    """Combine a run's stdout/stderr with any *.log it wrote in the repo dir —
    generated code often logs metrics to app.log rather than stdout."""
    from smoke_run import _collect_log_text
    return "\n".join([stdout, stderr, _collect_log_text(Path(repo_dir))])


def run_for_metrics(repo_dir: Path, entry: str = "main.py", args: list[str] | None = None,
                    timeout: int = 1800, mem_mb: int = 4096, env: dict | None = None) -> tuple[str, bool]:
    """Run the repo to completion (train + eval) and return (combined full
    output, timed_out). Longer/heavier than a smoke run on purpose — a real
    achieved metric needs a real run. Full text (not tails) so metric parsing
    sees everything."""
    import subprocess
    import sys

    from smoke_run import _limit_memory

    repo_dir = Path(repo_dir)
    cmd = [sys.executable, entry] + (args if args is not None else ["--config", "config.json", "--seed", "0"])
    timed_out = False
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True,
                              timeout=timeout, preexec_fn=_limit_memory(mem_mb))
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    return collect_run_output(repo_dir, stdout, stderr), timed_out
