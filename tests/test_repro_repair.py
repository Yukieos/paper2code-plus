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


def crash(file="model.py"):
    return SmokeResult(ok=False, reason="runtime crash", crashed=True,
                       offending_file=file, traceback="Traceback ... in forward")


def structural():
    return SmokeResult(ok=False, reason="timed out with no measurable progress", timed_out=True)


class FakeActions:
    def __init__(self, smoke_seq, regen_ok=True, replan_ok=True, reclass_ok=True):
        self._seq = list(smoke_seq)
        self.regen_ok, self.replan_ok, self.reclass_ok = regen_ok, replan_ok, reclass_ok
        self.calls = []

    def smoke(self):
        return self._seq.pop(0)

    def regenerate_file(self, rel_path, traceback):
        self.calls.append(("regenerate_file", rel_path))
        return self.regen_ok

    def replan_and_regenerate(self):
        self.calls.append(("replan", None))
        return self.replan_ok

    def reclassify_and_regenerate(self):
        self.calls.append(("reclassify", None))
        return self.reclass_ok


class RepairLadderTest(unittest.TestCase):
    def test_passes_first_try_no_actions(self):
        out = repair_reproduction(FakeActions([ok()]))
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, [])

    def test_single_file_crash_regenerated(self):
        a = FakeActions([crash("model.py"), ok()])
        out = repair_reproduction(a)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["regenerate_file"])
        self.assertEqual(a.calls[0], ("regenerate_file", "model.py"))

    def test_regen_exhausted_then_replan(self):
        a = FakeActions([crash(), crash(), crash(), ok()])  # 2 regen tries fail, replan fixes
        out = repair_reproduction(a, file_attempts=2, replan_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["regenerate_file", "regenerate_file", "replan"])

    def test_structural_failure_skips_file_regen(self):
        a = FakeActions([structural(), ok()])
        out = repair_reproduction(a)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["replan"])  # no offending file -> straight to replan

    def test_full_ladder_then_reclassify(self):
        a = FakeActions([structural(), structural(), ok()])
        out = repair_reproduction(a, replan_attempts=1, reclassify_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["replan", "reclassify"])

    def test_gives_up_when_budget_exhausted(self):
        a = FakeActions([structural(), structural(), structural()], )
        out = repair_reproduction(a, file_attempts=2, replan_attempts=1, reclassify_attempts=1)
        self.assertFalse(out.repaired)
        self.assertEqual(out.actions_taken, ["replan", "reclassify"])

    def test_failed_action_escalates(self):
        # regenerate_file can't apply -> drop its budget, escalate to replan
        a = FakeActions([crash("m.py"), ok()], regen_ok=False)
        out = repair_reproduction(a, file_attempts=1, replan_attempts=1)
        self.assertTrue(out.repaired)
        self.assertEqual(out.actions_taken, ["regenerate_file", "replan"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
