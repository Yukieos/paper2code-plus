"""Independent reproduction of a mined failure, and targeted verification of a
fix — the two safety layers the design puts around every harness change.

- reproduce_failure: before a production trace is allowed to drive a change, the
  same failure mode must recur when we re-run the pipeline on that run's own
  input paper. A failure that doesn't reproduce (transient dependency error,
  sampling noise, one-off bad external data) must NOT trigger a prompt change.

- verify_target_fixed: after a fix is applied, re-run the triggering input and
  confirm the specific failure mode is actually gone — not just that held-out
  and regression fixtures didn't regress.

Both re-use eval_gate.run_pipeline_on_text, so a reproduce/verify run is graded
by exactly the same path as the eval gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from eval.taxonomy import FailureMode
from improve.eval_gate import FixtureResult, gate_env, run_pipeline_on_text

logger = logging.getLogger(__name__)


def load_run_input(runs_dir: Path, run_id: str) -> str | None:
    """The original paper Markdown for a mined run, written by the pipeline as
    runs/<run_id>/input.md. None if it wasn't captured (older runs / traces
    only) — such a run can't be reproduced and shouldn't drive a change."""
    input_path = Path(runs_dir) / run_id / "input.md"
    if not input_path.exists():
        return None
    text = input_path.read_text(encoding="utf-8").strip()
    return text or None


@dataclass
class ReproResult:
    mode: FailureMode
    run_id: str
    attempts: int
    recurrences: int  # how many attempts re-exhibited `mode`
    threshold: int
    results: list[FixtureResult] = field(default_factory=list)

    @property
    def reproduced(self) -> bool:
        return self.recurrences >= self.threshold

    def summary(self) -> str:
        return (f"{self.mode.value} on {self.run_id}: reproduced {self.recurrences}/{self.attempts} "
                f"(threshold {self.threshold}) -> {'REPRODUCED' if self.reproduced else 'NOT reproduced'}")


def _mode_present(result: FixtureResult, mode: FailureMode) -> bool:
    return mode in result.grader_report.modes


def reproduce_failure(
    input_text: str,
    mode: FailureMode,
    run_id: str,
    workdir_root: Path,
    model: str = "gpt-4o-mini",
    attempts: int = 2,
    threshold: int = 2,
    extra_env: dict | None = None,
) -> ReproResult:
    """Re-run the pipeline `attempts` times on `input_text`; count how many
    re-exhibit `mode`. Reproduced iff that count >= threshold."""
    env = gate_env(extra_env)
    results: list[FixtureResult] = []
    recurrences = 0
    for i in range(attempts):
        workdir = Path(workdir_root) / "reproduce" / run_id / f"attempt{i}"
        result = run_pipeline_on_text(f"{run_id}#repro{i}", input_text, "reproduce", workdir, model, env)
        results.append(result)
        if _mode_present(result, mode):
            recurrences += 1
        # Short-circuit once the threshold is met — no need to burn more runs.
        if recurrences >= threshold:
            logger.info("Reproduce: %s met threshold %d after %d attempt(s).", mode.value, threshold, i + 1)
            break
    return ReproResult(mode, run_id, len(results), recurrences, threshold, results)


@dataclass
class TargetedVerifyResult:
    mode: FailureMode
    run_id: str
    attempts: int
    still_failing: int  # attempts where `mode` was STILL present after the fix
    results: list[FixtureResult] = field(default_factory=list)

    @property
    def fixed(self) -> bool:
        # The whole point is the triggering failure is gone: no attempt may
        # still show the mode.
        return self.still_failing == 0 and self.attempts > 0

    def summary(self) -> str:
        return (f"targeted {self.mode.value} on {self.run_id}: still present in "
                f"{self.still_failing}/{self.attempts} post-fix run(s) -> "
                f"{'FIXED' if self.fixed else 'NOT fixed'}")


def verify_target_fixed(
    input_text: str,
    mode: FailureMode,
    run_id: str,
    workdir_root: Path,
    model: str = "gpt-4o-mini",
    attempts: int = 1,
    extra_env: dict | None = None,
) -> TargetedVerifyResult:
    """Re-run the triggering input after a fix and confirm `mode` is gone."""
    env = gate_env(extra_env)
    results: list[FixtureResult] = []
    still = 0
    for i in range(attempts):
        workdir = Path(workdir_root) / "targeted" / run_id / f"attempt{i}"
        result = run_pipeline_on_text(f"{run_id}#verify{i}", input_text, "targeted", workdir, model, env)
        results.append(result)
        if _mode_present(result, mode):
            still += 1
    return TargetedVerifyResult(mode, run_id, len(results), still, results)
