"""Offline tests for smoke_run: real subprocesses, but tiny stdlib-only fake
'generated repos' so no torch/data is needed."""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke_run import offending_repo_file, parse_losses, run_smoke


def _mkrepo(files: dict[str, str]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, body in files.items():
        (d / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return d


class ParseTest(unittest.TestCase):
    def test_parse_losses_formats(self):
        text = "Epoch 1 loss: 0.90\nloss = 1.2e-1\ntrain_loss 0.5\nLoss: nan"
        vals = parse_losses(text)
        self.assertEqual(vals[:3], [0.90, 0.12, 0.5])
        self.assertTrue(vals[-1] != vals[-1])  # nan

    def test_offending_file_deepest_in_repo(self):
        repo = _mkrepo({})
        tb = (f'File "{repo}/main.py", line 3, in <module>\n'
              f'File "/usr/lib/python3/logging.py", line 9, in emit\n'
              f'File "{repo}/model.py", line 12, in forward')
        # deepest IN-REPO frame wins, stdlib frame ignored
        self.assertEqual(offending_repo_file(tb, repo), "model.py")


class SmokeRunTest(unittest.TestCase):
    def test_decreasing_loss_stdout_passes(self):
        repo = _mkrepo({"main.py": """
            import argparse
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            for i, l in enumerate([0.9, 0.6, 0.3]):
                print(f'epoch {i} loss: {l}')
        """})
        res = run_smoke(repo, timeout=30)
        self.assertTrue(res.ok, res.summary())
        self.assertTrue(res.progressed)
        self.assertEqual(res.loss_values, [0.9, 0.6, 0.3])

    def test_loss_only_in_logfile_still_parsed(self):
        # mimics generated code doing logging.basicConfig(filename='app.log')
        repo = _mkrepo({"main.py": """
            import argparse, logging
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            logging.basicConfig(filename='app.log', level=logging.INFO)
            for i, l in enumerate([1.0, 0.5]):
                logging.info(f'Epoch {i} Loss: {l:.4f}')
        """})
        res = run_smoke(repo, timeout=30)
        self.assertTrue(res.ok, res.summary())
        self.assertEqual(res.loss_values, [1.0, 0.5])
        self.assertTrue(res.progressed)

    def test_crash_reports_offending_file(self):
        repo = _mkrepo({
            "main.py": """
                import argparse
                p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
                import model
                model.build()
            """,
            "model.py": """
                def build():
                    raise ValueError('bad layer shape')
            """,
        })
        res = run_smoke(repo, timeout=30)
        self.assertFalse(res.ok)
        self.assertTrue(res.crashed)
        self.assertEqual(res.offending_file, "model.py")
        self.assertIn("bad layer shape", res.traceback)

    def test_nan_loss_is_divergence(self):
        repo = _mkrepo({"main.py": """
            import argparse
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            print('loss: 0.5'); print('loss: nan')
        """})
        res = run_smoke(repo, timeout=30)
        self.assertFalse(res.ok)
        self.assertTrue(res.diverged)

    def test_timeout_with_progress_is_ok(self):
        repo = _mkrepo({"main.py": """
            import argparse, time
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            print('loss: 0.9', flush=True); print('loss: 0.4', flush=True)
            time.sleep(30)
        """})
        res = run_smoke(repo, timeout=2)
        self.assertTrue(res.timed_out)
        self.assertTrue(res.ok, res.summary())  # was clearly training when killed

    def test_timeout_no_progress_fails(self):
        repo = _mkrepo({"main.py": """
            import argparse, time
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            time.sleep(30)
        """})
        res = run_smoke(repo, timeout=2)
        self.assertTrue(res.timed_out)
        self.assertFalse(res.ok)

    def test_require_progress_fails_on_clean_no_loss_run(self):
        # exits 0 but logs no loss/metric — a no-op that must NOT pass when we
        # expect training progress (mimics generated code swallowing its error)
        repo = _mkrepo({"main.py": """
            import argparse
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            print('setup done, trained nothing')
        """})
        self.assertTrue(run_smoke(repo, timeout=30).ok)  # lenient default: clean exit passes
        strict = run_smoke(repo, timeout=30, require_progress=True)
        self.assertFalse(strict.ok, strict.summary())
        self.assertIn("no loss", strict.reason.lower())

    def test_require_progress_passes_when_loss_present(self):
        repo = _mkrepo({"main.py": """
            import argparse
            p = argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed'); p.parse_args()
            print('loss: 0.9'); print('loss: 0.4')
        """})
        self.assertTrue(run_smoke(repo, timeout=30, require_progress=True).ok)

    def test_missing_entry_point(self):
        repo = _mkrepo({"notmain.py": "print('hi')"})
        res = run_smoke(repo, timeout=10)
        self.assertFalse(res.ok)
        self.assertIn("not found", res.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
