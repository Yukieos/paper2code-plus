"""Unit tests for the reproduce/verify counting logic and the append-only
changelog — all offline (the real pipeline run is faked)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.graders import GraderReport, GraderSignal
from eval.taxonomy import FailureMode
from improve import reproduce as repro_mod
from improve.changelog import ChangelogEntry, append_entry
from improve.eval_gate import FixtureResult
from improve.reproduce import reproduce_failure, verify_target_fixed

MODE = FailureMode.LINT_VIOLATION


def _result_with(mode_present: bool) -> FixtureResult:
    report = GraderReport(signals=[GraderSignal(MODE, "flake8")] if mode_present else [])
    return FixtureResult("x", "reproduce", True, True, report)


class ReproduceTest(unittest.TestCase):
    def test_reproduced_when_threshold_met(self):
        # mode present every attempt -> reproduced, and short-circuits at threshold
        with mock.patch.object(repro_mod, "run_pipeline_on_text",
                               side_effect=lambda *a, **k: _result_with(True)) as run:
            res = reproduce_failure("paper", MODE, "run1", Path(tempfile.gettempdir()),
                                    attempts=3, threshold=2)
        self.assertTrue(res.reproduced)
        self.assertEqual(res.recurrences, 2)
        self.assertEqual(run.call_count, 2, "should short-circuit once threshold is met")

    def test_not_reproduced_when_transient(self):
        # mode present only once out of two -> below threshold 2 -> not reproduced
        seq = [_result_with(True), _result_with(False)]
        with mock.patch.object(repro_mod, "run_pipeline_on_text", side_effect=lambda *a, **k: seq.pop(0)):
            res = reproduce_failure("paper", MODE, "run1", Path(tempfile.gettempdir()),
                                    attempts=2, threshold=2)
        self.assertFalse(res.reproduced)
        self.assertEqual(res.recurrences, 1)

    def test_verify_fixed_when_mode_absent(self):
        with mock.patch.object(repro_mod, "run_pipeline_on_text",
                               side_effect=lambda *a, **k: _result_with(False)):
            res = verify_target_fixed("paper", MODE, "run1", Path(tempfile.gettempdir()), attempts=1)
        self.assertTrue(res.fixed)

    def test_verify_not_fixed_when_mode_persists(self):
        with mock.patch.object(repro_mod, "run_pipeline_on_text",
                               side_effect=lambda *a, **k: _result_with(True)):
            res = verify_target_fixed("paper", MODE, "run1", Path(tempfile.gettempdir()), attempts=1)
        self.assertFalse(res.fixed)


class ChangelogTest(unittest.TestCase):
    def _entry(self, commit: str, supersedes: str = "none") -> ChangelogEntry:
        return ChangelogEntry(
            mode="lint_violation", triggering_runs=["run1"], reproduced="2/2 (threshold 2)",
            observed_behavior="unused imports", smoking_gun="F401", hypotheses=["h1"],
            changes=["FILE_ANALYSIS_PROMPT in codegen_pipeline.py — add lint rule"],
            files_changed=["codegen_pipeline.py"], targeted_result="FIXED",
            heldout_result="3/3 passed", regression_result="3/3 passed", commit=commit,
            supersedes=supersedes,
        )

    def test_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "HARNESS_CHANGELOG.md"
            append_entry(self._entry("aaa111"), path=path)
            first = path.read_text(encoding="utf-8")
            self.assertIn("# HARNESS_CHANGELOG", first)
            self.assertIn("aaa111", first)

            append_entry(self._entry("bbb222", supersedes="aaa111"), path=path)
            second = path.read_text(encoding="utf-8")
            # header appears once; both entries present; first entry text untouched
            self.assertEqual(second.count("# HARNESS_CHANGELOG"), 1)
            self.assertTrue(second.startswith(first), "existing content must be preserved verbatim (append-only)")
            self.assertIn("aaa111", second)
            self.assertIn("bbb222", second)
            self.assertIn("Supersedes:** aaa111", second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
