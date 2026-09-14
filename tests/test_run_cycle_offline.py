"""Offline, hermetic integration test for the full improve.run_cycle control flow.

Drives the REAL `run_cycle.main()` end to end, faking only the paid/side-effecting
boundaries — trace mining, LLM judge/diagnose/propose, the fixture eval gate, and
every git/gh subprocess. Everything in between runs for real: target-mode
selection (`pick_target_mode`), the pass/fail gate rule (`gate_passes`), branch
naming, PR-body assembly, and the discard-branch-on-failure cleanup.

This is what makes "the loop runs end to end" a repeatable check instead of a
one-off manual run that costs API credits. Run it with:

    PYTHONPATH=. .venv/bin/python tests/test_run_cycle_offline.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.graders import GraderReport, GraderSignal
from eval.mine_failures import RunResult
from eval.taxonomy import FailureMode
from improve import run_cycle
from improve.diagnose import Diagnosis
from improve.eval_gate import FixtureResult, GateReport
from improve.propose_change import ProposedChange
from improve.prompt_registry import get_prompt_text, refs_for_agent
from improve.reproduce import ReproResult, TargetedVerifyResult

TARGET_MODE = FailureMode.LINT_VIOLATION
RESPONSIBLE_AGENT = "execution_agent"  # has real registered prompts


def _fake_repro(reproduced: bool = True) -> ReproResult:
    n = 2
    return ReproResult(TARGET_MODE, "seed-1", n, n if reproduced else 0, 2, [])


def _fake_verify(fixed: bool = True) -> TargetedVerifyResult:
    return TargetedVerifyResult(TARGET_MODE, "seed-1", 1, 0 if fixed else 1, [])


def _fake_run_result(run_id: str) -> RunResult:
    report = GraderReport(signals=[GraderSignal(TARGET_MODE, f"{run_id}: flake8 F-codes", "F401 unused import")])
    return RunResult(run_id=run_id, trace_count=3, grader_signals=report, judge_verdict=None)


def _fake_diagnosis() -> Diagnosis:
    return Diagnosis(
        mode=TARGET_MODE,
        responsible_agent=RESPONSIBLE_AGENT,
        root_cause="Prompt does not tell the model to remove unused imports.",
        confidence=0.8,
        run_ids=["seed-1"],
    )


def _real_proposal() -> ProposedChange:
    """A genuine, placeholder-preserving proposal built from the live prompt, so
    set_prompt_text (which we stub in the test) would have real, valid input."""
    ref = refs_for_agent(RESPONSIBLE_AGENT)[0]
    old = get_prompt_text(ref)
    new = old + "\nAdditionally, do not leave unused imports in the file."
    return ProposedChange(ref, _fake_diagnosis(), old, new, "Add an unused-import instruction.")


def _gate(label: str, *, break_one: bool = False) -> GateReport:
    results = [
        FixtureResult("mlp_regression", "heldout", True, True, GraderReport()),
        FixtureResult("cnn_classifier", "regression", True, True, GraderReport()),
    ]
    if break_one:
        # proposed run newly breaks a regression fixture
        results[1] = FixtureResult(
            "cnn_classifier", "regression", True, True,
            GraderReport(signals=[GraderSignal(FailureMode.SYNTAX_ERROR, "broke it")]),
        )
    return GateReport(label=label, results=results)


class RunCycleOfflineTest(unittest.TestCase):
    def _patches(self, gate_side_effect, *, repro_ok=True, verify_ok=True, appended=None):
        """Common boundary fakes; `gate_side_effect` decides baseline/proposed.
        `appended` (a list) captures ChangelogEntry objects instead of writing
        the real HARNESS_CHANGELOG file."""
        store = mock.Mock()
        store.list_run_ids.return_value = ["seed-1", "seed-2", "seed-3"]

        def fake_append(entry, *a, **k):
            if appended is not None:
                appended.append(entry)

        return [
            mock.patch.object(run_cycle, "TraceStore", return_value=store),
            mock.patch.object(run_cycle, "LLMJudge", return_value=mock.Mock()),
            mock.patch.object(run_cycle, "mine_run", side_effect=lambda rid, *a, **k: _fake_run_result(rid)),
            mock.patch.object(run_cycle, "load_run_input", return_value="# A paper\n\nfake input text"),
            mock.patch.object(run_cycle, "reproduce_failure", return_value=_fake_repro(repro_ok)),
            mock.patch.object(run_cycle, "verify_target_fixed", return_value=_fake_verify(verify_ok)),
            mock.patch.object(run_cycle, "diagnose", return_value=_fake_diagnosis()),
            mock.patch.object(run_cycle, "propose_changes", return_value=[_real_proposal()]),
            mock.patch.object(run_cycle, "set_prompt_text"),  # don't mutate the tracked file
            mock.patch.object(run_cycle, "append_entry", side_effect=fake_append),  # don't write real changelog
            mock.patch.object(run_cycle, "run_gate", side_effect=gate_side_effect),
        ]

    def test_dry_run_stops_before_git(self):
        calls = []
        with mock.patch.object(run_cycle, "reproduce_failure") as repro:
            with mock.patch.object(run_cycle, "_run", side_effect=lambda cmd, **k: calls.append(cmd)):
                with self._ctx(self._patches(lambda *a, **k: _gate("x"))):
                    rc = run_cycle.main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [], "dry-run must not touch git/gh")
        repro.assert_not_called()  # dry-run must not run the paid reproduce pipeline

    def test_not_reproduced_stops_before_diagnose(self):
        appended = []
        with mock.patch.object(run_cycle, "_run") as run, \
                mock.patch.object(run_cycle, "diagnose") as diag:
            with self._ctx(self._patches(lambda *a, **k: _gate("x"), repro_ok=False, appended=appended)):
                rc = run_cycle.main([])
        self.assertEqual(rc, 0, "a failure that doesn't reproduce >=2x must not drive a change")
        diag.assert_not_called()
        run.assert_not_called()
        self.assertEqual(appended, [])

    def test_passing_gate_opens_pr_and_logs(self):
        seen = {"gate_calls": 0}

        def gate(label, *a, **k):
            seen["gate_calls"] += 1
            return _gate(label)  # baseline and proposed both clean -> no regression

        calls, appended = [], []
        with mock.patch.object(run_cycle, "_run", side_effect=lambda cmd, **k: calls.append(cmd) or mock.Mock()):
            with self._ctx(self._patches(gate, appended=appended)):
                rc = run_cycle.main([])
        self.assertEqual(rc, 0)
        joined = [" ".join(str(x) for x in c) for c in calls]
        self.assertTrue(any(c.startswith("git checkout -b improve/") for c in joined), joined)
        self.assertTrue(any("HARNESS_CHANGELOG" in c for c in joined), joined)
        self.assertTrue(any(c.startswith("git push") for c in joined), joined)
        self.assertTrue(any(c.startswith("gh pr create") for c in joined), joined)
        self.assertEqual(seen["gate_calls"], 2, "should run baseline + proposed gates")
        self.assertEqual(len(appended), 1, "must log exactly one changelog entry")
        self.assertEqual(appended[0].mode, TARGET_MODE.value)

    def test_failing_gate_discards_branch_no_pr(self):
        def gate(label, *a, **k):
            return _gate(label, break_one=(label == "proposed"))

        calls, appended = [], []
        with mock.patch.object(run_cycle, "_run", side_effect=lambda cmd, **k: calls.append(cmd) or mock.Mock()):
            with self._ctx(self._patches(gate, appended=appended)):
                rc = run_cycle.main([])
        self.assertEqual(rc, 1)
        joined = [" ".join(str(x) for x in c) for c in calls]
        self.assertTrue(any(c.startswith("git branch -D improve/") for c in joined), joined)
        self.assertFalse(any(c.startswith("gh pr create") for c in joined), "must NOT open a PR on a failing gate")
        self.assertEqual(appended, [], "no changelog entry on a failing gate")

    def test_targeted_verify_failure_blocks_pr(self):
        calls, appended = [], []
        with mock.patch.object(run_cycle, "_run", side_effect=lambda cmd, **k: calls.append(cmd) or mock.Mock()):
            with self._ctx(self._patches(lambda label, *a, **k: _gate(label), verify_ok=False, appended=appended)):
                rc = run_cycle.main([])
        self.assertEqual(rc, 1, "gate passing but targeted failure NOT fixed must block the PR")
        joined = [" ".join(str(x) for x in c) for c in calls]
        self.assertTrue(any(c.startswith("git branch -D improve/") for c in joined), joined)
        self.assertFalse(any(c.startswith("gh pr create") for c in joined))
        self.assertEqual(appended, [], "no changelog entry when the fix didn't actually fix the target")

    @staticmethod
    def _ctx(patches):
        class _Multi:
            def __enter__(self):
                for p in patches:
                    p.start()
                return self

            def __exit__(self, *exc):
                for p in reversed(patches):
                    p.stop()
                return False

        return _Multi()


if __name__ == "__main__":
    unittest.main(verbosity=2)
