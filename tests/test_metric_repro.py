"""Offline tests for the reproduction-fidelity layer: claimed-vs-achieved metric
comparison. No run needed — achieved metrics come from sample text."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metric_repro import (assess_fidelity, claimed_metrics_from_ups_ir, compare_metrics,
                          parse_achieved_metrics)

CNN_UPS_IR = {
    "experiments": [{
        "method": "m1", "dataset": "d1", "setup": "Trained TinyConvNet.",
        "metrics": [{"name": "test accuracy", "value": 98.3, "unit": "%"}],
        "source_reference": "5. Evaluation",
    }]
}


class ClaimedTest(unittest.TestCase):
    def test_from_ups_ir(self):
        claimed = claimed_metrics_from_ups_ir(CNN_UPS_IR)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].name, "test accuracy")
        self.assertEqual(claimed[0].value, 98.3)
        self.assertEqual(claimed[0].unit, "%")

    def test_empty_when_no_experiments(self):
        self.assertEqual(claimed_metrics_from_ups_ir({}), [])


class ParseAchievedTest(unittest.TestCase):
    def test_various_formats(self):
        text = "Epoch done. Accuracy: 0.9834\nBLEU 28.4\ntest acc = 97%\nF1-score: 0.71"
        got = {(a.name.lower(), a.value, a.unit) for a in parse_achieved_metrics(text)}
        self.assertIn(("accuracy", 0.9834, ""), got)
        self.assertIn(("bleu", 28.4, ""), got)
        self.assertIn(("f1-score", 0.71, ""), got)
        self.assertTrue(any(v == 97.0 and u == "%" for (_, v, u) in got))


class CompareTest(unittest.TestCase):
    def test_accuracy_within_tolerance_percent_vs_fraction(self):
        # claimed 98.3% -> 0.983 ; achieved 0.9834 -> match
        fid = assess_fidelity(CNN_UPS_IR, "Accuracy: 0.9834")
        self.assertTrue(fid.reproduced, fid.summary())
        c = fid.comparisons[0]
        self.assertTrue(c.matched and c.within_tolerance)
        self.assertLess(c.rel_gap, 0.05)

    def test_accuracy_off_beyond_tolerance(self):
        fid = assess_fidelity(CNN_UPS_IR, "Accuracy: 0.80")  # 0.80 vs 0.983
        c = fid.comparisons[0]
        self.assertTrue(c.matched)
        self.assertFalse(c.within_tolerance)
        self.assertFalse(fid.reproduced)

    def test_metric_not_produced(self):
        fid = assess_fidelity(CNN_UPS_IR, "training done, no metric printed")
        c = fid.comparisons[0]
        self.assertFalse(c.matched)
        self.assertFalse(fid.reproduced)
        self.assertIn("NOT PRODUCED", c.summary())

    def test_name_synonym_matching(self):
        # claimed "test accuracy" should match achieved "acc"
        claimed = claimed_metrics_from_ups_ir(CNN_UPS_IR)
        from metric_repro import AchievedMetric
        comps = compare_metrics(claimed, [AchievedMetric("acc", 0.983, "")])
        self.assertTrue(comps[0].matched and comps[0].within_tolerance)

    def test_bleu_not_scaled(self):
        ups = {"experiments": [{"dataset": "wmt", "metrics": [{"name": "BLEU", "value": 28.4, "unit": ""}]}]}
        # achieved 28.5 -> within; 26 -> off (rel ~8%)
        self.assertTrue(assess_fidelity(ups, "BLEU 28.5").reproduced)
        self.assertFalse(assess_fidelity(ups, "BLEU 26.0").reproduced)


class RunForMetricsTest(unittest.TestCase):
    def test_captures_metric_from_stdout_and_logfile(self):
        import tempfile
        import textwrap

        from metric_repro import run_for_metrics
        d = Path(tempfile.mkdtemp())
        (d / "main.py").write_text(textwrap.dedent("""
            import argparse, logging
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            logging.basicConfig(filename='app.log', level=logging.INFO)
            logging.info('Accuracy: 0.9834')
            print('done')
        """), encoding="utf-8")
        output, timed_out = run_for_metrics(d, timeout=30)
        self.assertFalse(timed_out)
        self.assertIn("Accuracy: 0.9834", output)  # pulled from app.log
        self.assertIn("done", output)              # and from stdout
        fid = assess_fidelity(CNN_UPS_IR, output)
        self.assertTrue(fid.reproduced, fid.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
