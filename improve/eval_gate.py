"""Run the pipeline over the fixture set and grade the results.

Used twice per improvement cycle — once against the current code
(baseline) and once against a branch with a proposed prompt change
(proposed) — so run_cycle.py can compare the two and only proceed to a PR
if the change doesn't regress anything.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval.graders import CodeGrader, GraderReport

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FixtureResult:
    fixture: str
    group: str  # "heldout" | "regression"
    extraction_ok: bool
    codegen_ok: bool
    grader_report: GraderReport

    @property
    def passed(self) -> bool:
        return self.extraction_ok and self.codegen_ok and not self.grader_report.has_failures


@dataclass
class GateReport:
    label: str
    results: list[FixtureResult] = field(default_factory=list)

    @property
    def total_signals(self) -> int:
        return sum(len(r.grader_report.signals) for r in self.results)

    @property
    def failing_fixtures(self) -> list[str]:
        return [r.fixture for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [f"[{self.label}] {len(self.results)} fixture(s), "
                 f"{len(self.failing_fixtures)} failing, {self.total_signals} total signal(s)."]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            modes = ", ".join(s.mode.value for s in r.grader_report.signals) or "-"
            lines.append(f"  [{status}] {r.group}/{r.fixture}: {modes}")
        return "\n".join(lines)


def _run_fixture(fixture_path: Path, group: str, workdir: Path, model: str, env: dict) -> FixtureResult:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "input.md").write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    extraction = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "input.md", "--model", model, "--skip-annotation"],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=600,
    )
    extraction_ok = extraction.returncode == 0
    if not extraction_ok:
        logger.warning("Extraction failed for %s: %s", fixture_path.name, extraction.stdout[-2000:])

    codegen_ok = False
    report = GraderReport()
    if extraction_ok:
        codegen = subprocess.run(
            [sys.executable, str(REPO_ROOT / "codegen_pipeline.py"), "UPS-IR.json", "--model", model],
            cwd=workdir, env=env, capture_output=True, text=True, timeout=1800,
        )
        codegen_ok = codegen.returncode == 0
        if not codegen_ok:
            logger.warning("Codegen failed for %s: %s", fixture_path.name, codegen.stdout[-2000:])
        report = CodeGrader(workdir / "generated_repo").grade()

    return FixtureResult(fixture_path.stem, group, extraction_ok, codegen_ok, report)


def run_gate(
    label: str,
    fixtures_dir: Path,
    workdir_root: Path,
    model: str = "gpt-4o-mini",
    extra_env: dict | None = None,
) -> GateReport:
    env = os.environ.copy()
    env["PAPER2CODE_TRACE_ENABLED"] = "0"  # gate runs are not what we're mining for failures
    env.update(extra_env or {})

    report = GateReport(label=label)
    for group in ("heldout", "regression"):
        group_dir = fixtures_dir / group
        if not group_dir.exists():
            continue
        for fixture_path in sorted(group_dir.glob("*.md")):
            workdir = workdir_root / label / group / fixture_path.stem
            result = _run_fixture(fixture_path, group, workdir, model, env)
            report.results.append(result)

    return report


def gate_passes(baseline: GateReport, proposed: GateReport) -> tuple[bool, str]:
    """Simple, conservative rule: the change must not make anything worse.

    No regression-set fixture may newly fail, and the total signal count
    across held-out fixtures must not increase. Anything more nuanced
    (weighting failure modes, partial credit) is left to the human reviewer
    reading the attached before/after report in the PR.
    """
    if len(baseline.results) == 0:
        return False, "No fixtures found under eval/fixtures/ — populate heldout/ and regression/ first."

    baseline_by_key = {(r.group, r.fixture): r for r in baseline.results}
    newly_broken = []
    for r in proposed.results:
        base = baseline_by_key.get((r.group, r.fixture))
        if base and base.passed and not r.passed:
            newly_broken.append(f"{r.group}/{r.fixture}")

    if newly_broken:
        return False, f"Proposed change newly breaks: {', '.join(newly_broken)}"

    if proposed.total_signals > baseline.total_signals:
        return False, (f"Proposed change increases total grader signals "
                        f"({baseline.total_signals} -> {proposed.total_signals}).")

    return True, "No regressions detected."
