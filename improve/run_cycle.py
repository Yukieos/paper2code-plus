"""Orchestrates one continuous-improvement cycle end to end.

    python -m improve.run_cycle                      # mine + target the most common failure mode
    python -m improve.run_cycle --failure-mode syntax_error
    python -m improve.run_cycle --dry-run             # diagnose + propose only; no git/gh calls

Never merges anything itself: on a passing gate it pushes a branch and opens
a PR (`gh pr create`) for a human to review and merge. That PR is the
deployment gate, not this script.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from eval.judge import LLMJudge
from eval.mine_failures import mine_run
from eval.taxonomy import FailureMode
from eval.trace_store import TraceStore
from improve.diagnose import diagnose
from improve.eval_gate import gate_passes, run_gate
from improve.propose_change import propose_changes
from improve.prompt_registry import set_prompt_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "eval" / "fixtures"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, **kwargs)


def pick_target_mode(results, explicit: str | None) -> FailureMode | None:
    if explicit:
        return FailureMode(explicit)
    counts = Counter(m for r in results for m in r.all_modes)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one continuous-improvement cycle.")
    parser.add_argument("--failure-mode", type=str, default=None,
                         help="Target a specific FailureMode id instead of the most common one.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true",
                         help="Diagnose + propose only; skip git branch/commit/push and PR creation.")
    parser.add_argument("--base-branch", type=str, default="main")
    args = parser.parse_args(argv)

    if not any((FIXTURES_DIR / "heldout").glob("*.md")):
        logger.error("No fixtures under %s/heldout/*.md — see eval/fixtures/README.md before running this.",
                      FIXTURES_DIR)
        return 1

    store = TraceStore()
    run_ids = store.list_run_ids()
    if not run_ids:
        logger.error("No traces found. Set PAPER2CODE_TRACE_ENABLED=1 during pipeline runs to collect them.")
        return 1

    logger.info("Mining %d run(s)...", len(run_ids))
    judge = LLMJudge(model=args.model)
    results = [mine_run(run_id, store, REPO_ROOT / "runs", use_judge=True, judge=judge) for run_id in run_ids]

    mode = pick_target_mode(results, args.failure_mode)
    if mode is None:
        logger.info("No failures found across %d run(s). Nothing to improve.", len(run_ids))
        return 0
    logger.info("Targeting failure mode: %s", mode.value)

    diagnosis = diagnose(mode, results, model=args.model)
    logger.info("Diagnosis: responsible_agent=%s confidence=%.2f\n%s",
                diagnosis.responsible_agent, diagnosis.confidence, diagnosis.root_cause)

    if not diagnosis.responsible_agent:
        logger.info("Diagnosis didn't point at a specific agent's prompt; stopping (nothing safe to change).")
        return 0

    proposals = propose_changes(diagnosis, model=args.model)
    if not proposals:
        logger.info("No valid prompt change proposed; stopping.")
        return 0

    for change in proposals:
        logger.info("Proposed change to %s: %s", change.ref.constant_name, change.explanation)

    if args.dry_run:
        logger.info("--dry-run set: stopping before baseline gate / git operations.")
        return 0

    with tempfile.TemporaryDirectory(prefix="p2c-gate-") as tmp:
        workdir_root = Path(tmp)

        logger.info("Running baseline gate on %s...", args.base_branch)
        baseline = run_gate("baseline", FIXTURES_DIR, workdir_root, model=args.model)
        logger.info("\n%s", baseline.summary())

        branch = f"improve/{mode.value}-{diagnosis.run_ids[0] if diagnosis.run_ids else 'auto'}"
        _run(["git", "checkout", "-b", branch])

        applied: list[str] = []
        try:
            for change in proposals:
                set_prompt_text(change.ref, change.new_prompt)
                applied.append(change.ref.constant_name)

            _run(["git", "add"] + [str(REPO_ROOT / c.ref.file_path) for c in proposals])
            commit_msg = (
                f"Propose prompt fix for {mode.value}\n\n"
                f"Diagnosis (confidence {diagnosis.confidence:.2f}): {diagnosis.root_cause}\n\n"
                + "\n".join(f"- {c.ref.constant_name}: {c.explanation}" for c in proposals)
            )
            _run(["git", "commit", "-m", commit_msg])

            logger.info("Running proposed gate on %s...", branch)
            proposed = run_gate("proposed", FIXTURES_DIR, workdir_root, model=args.model)
            logger.info("\n%s", proposed.summary())

            passed, reason = gate_passes(baseline, proposed)
            logger.info("Gate result: %s (%s)", "PASS" if passed else "FAIL", reason)

            if not passed:
                _run(["git", "checkout", args.base_branch])
                _run(["git", "branch", "-D", branch])
                logger.info("Discarded branch %s; no PR opened.", branch)
                return 1

            _run(["git", "push", "-u", "origin", branch])
            pr_body = (
                f"## Diagnosis\n{diagnosis.root_cause}\n\n"
                f"**Confidence:** {diagnosis.confidence:.2f}\n\n"
                f"## Changes\n" + "\n".join(f"- `{c.ref.constant_name}`: {c.explanation}" for c in proposals) +
                f"\n\n## Gate results\n```\n{baseline.summary()}\n\n{proposed.summary()}\n```\n\n"
                f"Automated proposal from `improve.run_cycle` — held-out + regression gate passed "
                f"({reason}). Requires human review before merge."
            )
            _run(["gh", "pr", "create", "--base", args.base_branch, "--head", branch,
                  "--title", f"Fix {mode.value}: {diagnosis.responsible_agent} prompt",
                  "--body", pr_body])
            logger.info("Opened PR for branch %s.", branch)
            return 0
        finally:
            _run(["git", "checkout", args.base_branch], check=False)


if __name__ == "__main__":
    sys.exit(main())
