"""Guard-path tests for PipelineRepairActions that need no LLM/torch — the
chain is lazy, so construction and the early-return paths run offline."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codegen_pipeline import CodeWriterConfig, PipelineRepairActions


class RepairActionsGuardTest(unittest.TestCase):
    def _actions(self, state=None):
        cfg = CodeWriterConfig(output_dir=Path(tempfile.mkdtemp()), model_name="gpt-4o-mini",
                               save_intermediate=False)
        return PipelineRepairActions(state or {"plan": {"files": []}, "generated_files": []}, cfg)

    def test_entry_default_and_override(self):
        a = self._actions()
        self.assertEqual(a._entry(), "main.py")
        a.state["entry_point_path"] = "run.py"
        self.assertEqual(a._entry(), "run.py")

    def test_diagnose_patch_on_empty_repo_returns_false_without_llm(self):
        from smoke_run import SmokeResult
        a = self._actions()  # output_dir has no .py files
        self.assertFalse(a.diagnose_and_patch(SmokeResult(ok=False, reason="crash", crashed=True)))
        self.assertIsNone(a._llm)  # never built the LLM -> no credentials needed

    def test_replan_without_planner_returns_false(self):
        a = self._actions()  # no planner_agent injected
        self.assertFalse(a.replan_and_regenerate())


if __name__ == "__main__":
    unittest.main(verbosity=2)
