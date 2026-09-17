"""Offline tests for the whole-repo, source-grounded diagnosis: prompt assembly,
plan parsing, and multi-file patch application. Fake llm_call, no torch."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro_diagnose import (apply_repair_plan, build_prompt, diagnose_reproduction,
                            paper_text_from_ups_ir, read_repo_files)

UPS = {"sections": [{"title": "3. Model", "text": "TinyConvNet takes a 28x28 image."}],
       "experiments": [{"metrics": [{"name": "test accuracy", "value": 98.3, "unit": "%"}]}]}


class PromptTest(unittest.TestCase):
    def test_prompt_includes_all_grounding(self):
        files = {"main.py": "import model\nmodel.TinyConvNet()\n", "model.py": "class TinyConvNet:\n    def __init__(self, input_shape): ...\n"}
        prompt = build_prompt(files, "runtime crash", "TypeError: missing input_shape",
                              UPS, plan={"files": [{"path": "main.py", "purpose": "entry"}]},
                              code_reviews={"main.py": "looks fine"}, trace_text="extractor ok")
        for needle in ["RUNTIME FAILURE", "ALL GENERATED FILES", "PAPER (ground truth)",
                       "SIGNATURE INDEX", "main.py", "model.py", "TinyConvNet", "input_shape",
                       "TinyConvNet takes a 28x28", "extractor ok"]:
            self.assertIn(needle, prompt, needle)

    def test_paper_text_prefers_sections(self):
        self.assertIn("TinyConvNet takes a 28x28", paper_text_from_ups_ir(UPS))


class DiagnoseParseTest(unittest.TestCase):
    def test_multi_file_edits_parsed(self):
        resp = json.dumps({
            "root_cause": "main.py calls TinyConvNet() but __init__ needs input_shape",
            "faithfulness": "faithful", "escalate": False,
            "edits": [
                {"path": "main.py", "new_content": "import model\nmodel.TinyConvNet((1,28,28))\n", "rationale": "pass shape"},
                {"path": "model.py", "new_content": "class TinyConvNet:\n    def __init__(self, input_shape):\n        self.s = input_shape\n", "rationale": "keep sig"},
            ],
        })
        plan = diagnose_reproduction("prompt", lambda p: resp)
        self.assertFalse(plan.escalate)
        self.assertEqual([e.path for e in plan.edits], ["main.py", "model.py"])
        self.assertIn("TinyConvNet", plan.edits[0].new_content)

    def test_escalate_flag_and_empty_edits(self):
        resp = json.dumps({"root_cause": "plan is wrong, missing a data module", "escalate": True, "edits": []})
        plan = diagnose_reproduction("p", lambda p: resp)
        self.assertTrue(plan.escalate)
        self.assertEqual(plan.edits, [])

    def test_no_edits_forces_escalate(self):
        resp = json.dumps({"root_cause": "x", "escalate": False, "edits": []})
        self.assertTrue(diagnose_reproduction("p", lambda p: resp).escalate)

    def test_unparseable_escalates(self):
        plan = diagnose_reproduction("p", lambda p: "not json at all")
        self.assertTrue(plan.escalate)

    def test_fenced_code_in_edit_stripped(self):
        resp = json.dumps({"root_cause": "x", "escalate": False,
                           "edits": [{"path": "a.py", "new_content": "```python\nprint('hello world')\n```"}]})
        plan = diagnose_reproduction("p", lambda p: resp)
        self.assertEqual(plan.edits[0].new_content, "print('hello world')")


class ApplyTest(unittest.TestCase):
    def test_applies_to_existing_only_and_blocks_escapes(self):
        from repro_diagnose import FileEdit, RepairPlan
        d = Path(tempfile.mkdtemp())
        (d / "main.py").write_text("old", encoding="utf-8")
        plan = RepairPlan(edits=[
            FileEdit("main.py", "new content here", "fix"),          # existing -> applied
            FileEdit("brand_new.py", "should be skipped xx", "add"),  # non-existent -> skipped
            FileEdit("../escape.py", "evil path attempt here", "x"),  # escapes repo -> skipped
        ])
        changed = apply_repair_plan(d, plan)
        self.assertEqual(changed, ["main.py"])
        self.assertEqual((d / "main.py").read_text(), "new content here")
        self.assertFalse((d / "brand_new.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
