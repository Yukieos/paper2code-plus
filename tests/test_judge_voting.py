"""Unit tests for multi-vote judge aggregation and the OTHER bucket. Offline:
the LLM call is a stub cycling through canned JSON responses."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.graders import GraderReport
from eval.judge import LLMJudge, _safe_mode
from eval.taxonomy import FailureMode


def _vote(mode, conf, secondary=None, rationale="r"):
    sec = secondary or []
    import json
    return json.dumps({
        "primary_failure_mode": mode, "secondary_failure_modes": sec,
        "confidence": conf, "rationale": rationale,
    })


def _judge_returning(responses):
    it = iter(responses)
    return LLMJudge(llm_call=lambda prompt: next(it), votes=len(responses))


class JudgeVotingTest(unittest.TestCase):
    def test_majority_mode_and_median_confidence(self):
        judge = _judge_returning([
            _vote("lint_violation", 0.9),
            _vote("lint_violation", 0.7),
            _vote("syntax_error", 0.2),
        ])
        v = judge.classify([], GraderReport())
        self.assertEqual(v.primary_failure_mode, FailureMode.LINT_VIOLATION)  # 2/3 plurality
        self.assertAlmostEqual(v.confidence, 0.7)  # median of [0.9,0.7,0.2]

    def test_minority_failure_votes_mean_healthy(self):
        # only 1 of 3 votes sees a failure -> below majority -> None
        judge = _judge_returning([
            _vote("lint_violation", 0.9),
            _vote(None, 0.0),
            _vote(None, 0.0),
        ])
        v = judge.classify([], GraderReport())
        self.assertIsNone(v.primary_failure_mode)

    def test_secondary_needs_majority(self):
        judge = _judge_returning([
            _vote("lint_violation", 0.8, secondary=["import_dependency_error"]),
            _vote("lint_violation", 0.8, secondary=["import_dependency_error"]),
            _vote("lint_violation", 0.8, secondary=["syntax_error"]),
        ])
        v = judge.classify([], GraderReport())
        self.assertEqual(v.secondary_failure_modes, [FailureMode.IMPORT_DEPENDENCY_ERROR])

    def test_unknown_mode_becomes_other(self):
        self.assertEqual(_safe_mode("some_new_unnamed_failure"), FailureMode.OTHER)
        self.assertEqual(_safe_mode("other"), FailureMode.OTHER)
        self.assertIsNone(_safe_mode(None))
        self.assertEqual(_safe_mode("lint_violation"), FailureMode.LINT_VIOLATION)

    def test_other_survives_voting(self):
        judge = _judge_returning([_vote("other", 0.6), _vote("made_up_mode", 0.5)])
        v = judge.classify([], GraderReport())
        self.assertEqual(v.primary_failure_mode, FailureMode.OTHER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
