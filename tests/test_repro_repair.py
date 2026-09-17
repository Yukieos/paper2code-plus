"""Offline tests for the reproduction-repair escalation ladder. The actions are
scripted fakes returning canned SmokeResults — no torch, no LLM."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro_repair import repair_reproduction
from smoke_run import SmokeResult


def ok():
    return SmokeResult(ok=True, reason="ran to completion")


def fail():
    return SmokeResult(ok=False, reason="runtime crash", crashed=True, traceback="Traceback ...")


class FakeActions:
    def __init__(self, smoke_seq, patch_ok=True, replan_ok=True, reclass_ok=True):
        self._seq = list(smoke_seq)
        self.patch_ok, self.replan_ok, self.reclass_ok = patch_ok, replan_ok, reclass_ok
        self.calls = []

    def smoke(self):
        return self._seq.pop(0)

    def diagnose_and_patch(self, result):
        self.calls.append("diagnose_patch")
        return self.patch_ok

    def replan_and_regenerate(self):
        self.calls.append("replan")
        return self.replan_ok

    def reclassify_and_regenerate(self):
        self.calls.append("reclassify")
        return self.reclass_ok


class RepairLadderTest(unittest.TestCase):
    def test_passes_first_try_no_actions(self):
        out = repair_reproduction(FakeActions([ok()]))
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, [])

    def test_diagnose_patch_fixes(self):
        a = FakeActions([fail(), ok()])
        out = repair_reproduction(a)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["diagnose_patch"])

    def test_patch_exhausted_then_replan(self):
        a = FakeActions([fail(), fail(), fail(), ok()])  # 2 patch tries don't fix, replan does
        out = repair_reproduction(a, patch_attempts=2, replan_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["diagnose_patch", "diagnose_patch", "replan"])

    def test_diagnosis_escalation_skips_to_replan(self):
        # diagnose_and_patch returns False (escalated) -> straight to replan, no wasted re-smoke
        a = FakeActions([fail(), ok()], patch_ok=False)
        out = repair_reproduction(a, patch_attempts=2, replan_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["diagnose_patch", "replan"])

    def test_full_ladder_then_reclassify(self):
        a = FakeActions([fail(), fail(), fail(), ok()])
        out = repair_reproduction(a, patch_attempts=1, replan_attempts=1, reclassify_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["diagnose_patch", "replan", "reclassify"])

    def test_gives_up_when_budget_exhausted(self):
        a = FakeActions([fail(), fail(), fail(), fail()])
        out = repair_reproduction(a, patch_attempts=1, replan_attempts=1, reclassify_attempts=1)
        self.assertFalse(out.repaired)
        self.assertEqual(out.actions_taken, ["diagnose_patch", "replan", "reclassify"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
