"""Orchestrates one continuous-improvement cycle end to end.

    python -m improve.run_cycle                      # mine + target the most common failure mode
    python -m improve.run_cycle --failure-mode syntax_error
    python -m improve.run_cycle --dry-run             # diagnose + propose only; no reproduce/gate/git

Full (non-dry) flow, matching the design's safety layers:
    mine -> pick target mode
    -> REPRODUCE the failure >=Nx on its own input (transient failures stop here)
    -> diagnose (root cause + responsible, registry-valid agent)
    -> propose a scoped prompt change
    -> baseline gate  -> apply change -> proposed gate (held-out + regression)
    -> TARGETED VERIFY: re-run the triggering input; the target failure must be gone
    -> append an entry to HARNESS_CHANGELOG
    -> push branch + open PR

Never merges anything itself: on passing reproduce + gate + targeted verify it
pushes a branch and opens a PR (`gh pr create`) for a human to review and
merge. That PR is the deployment gate, not this script.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from eval.cluster_failures import cluster_failures
from eval.judge import LLMJudge
from eval.mine_failures import mine_run
from eval.taxonomy import FailureMode
from eval.trace_store import TraceStore
from improve.changelog import CHANGELOG_PATH, ChangelogEntry, append_entry
from improve.diagnose import diagnose
from improve.eval_gate import gate_passes, run_gate
from improve.propose_change import propose_changes
from improve.prompt_registry import set_prompt_text
from improve.reproduce import load_run_input, reproduce_failure, verify_target_fixed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "eval" / "fixtures"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, **kwargs)


def pick_target_mode(results, explicit: str | None) -> FailureMode | None:
    """Fallback selection: the most common mode by raw count, excluding OTHER
    (OTHER isn't a fixable mode — it's a signal to expand the taxonomy)."""
    if explicit:
        return FailureMode(explicit)
    counts = Counter(m for r in results for m in r.all_modes if m != FailureMode.OTHER)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def select_target(results, args) -> tuple[FailureMode | None, str]:
    """Pick which failure to work on, and explain how. Prefers the largest
    recurring CLUSTER's dominant mode (a pattern that recurs, not just a mode
    that's individually common) over the raw per-mode count. Falls back to the
    count if clustering is off, unavailable, or finds no multi-instance group.
    Returns (mode, human-readable selection note)."""
    if args.failure_mode:
        return FailureMode(args.failure_mode), f"explicit --failure-mode {args.failure_mode}"

    # Surface an OTHER pileup: it can't be auto-fixed, but a human should see it.
    other_count = sum(1 for r in results for m in r.all_modes if m == FailureMode.OTHER)
    if other_count:
        logger.info("%d failure(s) classified OTHER — unmapped modes that may need a new taxonomy entry "
                    "(not auto-fixable).", other_count)

    if not args.no_cluster_target:
        try:
            clusters = cluster_failures(results, eps=args.cluster_eps)  # largest first
        except Exception as exc:  # embeddings unavailable / no key / sklearn issue
            logger.warning("Clustering failed (%s); falling back to raw mode counts.", exc)
            clusters = []
        for cluster in clusters:
            modes = [m for m in (i.mode for i in cluster.instances) if m and m != FailureMode.OTHER]
            if cluster.size >= 2 and modes:
                dominant = Counter(modes).most_common(1)[0][0]
                return dominant, f"largest recurring cluster '{cluster.label}' (size {cluster.size})"

    mode = pick_target_mode(results, None)
    return mode, "most common failure mode by raw count"


def changelog_path() -> Path:
    return CHANGELOG_PATH


def _gate_group_summary(report, group: str) -> str:
    group_results = [r for r in report.results if r.group == group]
    if not group_results:
        return "no fixtures"
    passed = sum(1 for r in group_results if r.passed)
    return f"{passed}/{len(group_results)} passed"


def _smoking_gun(results, mode: FailureMode, limit: int = 3) -> str:
    """Concrete evidence lines for the target mode, pulled from the deterministic
    grader signals across the mining results."""
    lines = []
    for r in results:
        for signal in r.grader_signals.signals:
            if signal.mode == mode:
                detail = signal.detail
                if signal.evidence:
                    detail += f" — {signal.evidence[:200]}"
                lines.append(f"[{r.run_id}] {detail}")
    return "; ".join(lines[:limit]) if lines else ""


def _record_changelog(mode, diagnosis, proposals, results, repro_summary,
                      targeted_summary, baseline, proposed, commit_sha) -> None:
    entry = ChangelogEntry(
        mode=mode.value,
        triggering_runs=diagnosis.run_ids,
        reproduced=repro_summary,
        observed_behavior=diagnosis.root_cause,
        smoking_gun=diagnosis.smoking_gun or _smoking_gun(results, mode),
        hypotheses=(diagnosis.hypotheses
                    or [f"{diagnosis.responsible_agent}: {diagnosis.root_cause} "
                        f"(confidence {diagnosis.confidence:.2f})"]),
        changes=[f"{c.ref.constant_name} in {c.ref.file_path} — {c.explanation}" for c in proposals],
        files_changed=sorted({c.ref.file_path for c in proposals}),
        targeted_result=targeted_summary,
        heldout_result=_gate_group_summary(proposed, "heldout"),
        regression_result=_gate_group_summary(proposed, "regression"),
        cost_delta=f"grader signals {baseline.total_signals} -> {proposed.total_signals} (token/$ not tracked)",
        risks="Prompt-only change; behavior on papers unlike the fixtures is unverified.",
        commit=commit_sha,
    )
    append_entry(entry)


def find_triggering_input(results, mode: FailureMode, runs_dir: Path):
    """The first run exhibiting `mode` whose original input paper we still have
    (runs/<run_id>/input.md), so it can be re-run for reproduction/verification.
    Returns (run_id, input_text) or (None, None)."""
    for r in results:
        if mode in r.all_modes:
            text = load_run_input(runs_dir, r.run_id)
            if text:
                return r.run_id, text
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one continuous-improvement cycle.")
    parser.add_argument("--failure-mode", type=str, default=None,
                         help="Target a specific FailureMode id instead of the most common one.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true",
                         help="Diagnose + propose only; skip reproduce, gate, and all git/gh calls.")
    parser.add_argument("--base-branch", type=str, default="main")
    parser.add_argument("--no-reproduce", action="store_true",
                         help="Skip the reproduce->=Nx gate before diagnosis (NOT recommended: a "
                              "transient one-off failure could then drive a change).")
    parser.add_argument("--reproduce-attempts", type=int, default=2,
                         help="How many times to re-run the triggering input to check reproduction (default 2).")
    parser.add_argument("--reproduce-threshold", type=int, default=2,
                         help="Minimum re-runs that must re-exhibit the failure to count as reproduced (default 2).")
    parser.add_argument("--verify-attempts", type=int, default=1,
                         help="Post-fix re-runs of the triggering input to confirm the failure is gone (default 1).")
    parser.add_argument("--judge-votes", type=int, default=3,
                         help="How many times the LLM judge votes per run; aggregated by majority/median (default 3).")
    parser.add_argument("--no-cluster-target", action="store_true",
                         help="Pick the target by raw per-mode count instead of the largest recurring cluster.")
    parser.add_argument("--cluster-eps", type=float, default=0.35,
                         help="DBSCAN cosine-distance threshold for clustering target selection (default 0.35).")
    args = parser.parse_args(argv)
    runs_dir = REPO_ROOT / "runs"

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
    judge = LLMJudge(model=args.model, votes=args.judge_votes)
    results = [mine_run(run_id, store, runs_dir, use_judge=True, judge=judge) for run_id in run_ids]

    mode, selection_note = select_target(results, args)
    if mode is None:
        logger.info("No auto-fixable failures found across %d run(s). Nothing to improve.", len(run_ids))
        return 0
    logger.info("Targeting failure mode: %s (%s)", mode.value, selection_note)

    # The triggering run's original input — needed to reproduce the failure and,
    # later, to verify the fix. Reproduction/verification are impossible without
    # it, so a mode we can't tie to a re-runnable input is treated conservatively.
    trigger_run_id, trigger_input = find_triggering_input(results, mode, runs_dir)

    # ---- Reproduce >=Nx: a production failure must recur on its own input
    # before it's allowed to drive a change (guards against transient/one-off
    # failures). Skipped under --dry-run (it runs the real, paid pipeline) and
    # --no-reproduce. Recorded for the changelog either way.
    repro_summary = "skipped"
    repro_result = None
    if not args.dry_run and not args.no_reproduce:
        if not trigger_input:
            logger.error("Cannot reproduce %s: no run exhibiting it still has runs/<id>/input.md. "
                         "Refusing to change the harness off an unreproducible failure.", mode.value)
            return 1
        with tempfile.TemporaryDirectory(prefix="p2c-repro-") as tmp:
            repro_result = reproduce_failure(
                trigger_input, mode, trigger_run_id, Path(tmp), model=args.model,
                attempts=args.reproduce_attempts, threshold=args.reproduce_threshold,
            )
        logger.info("Reproduce: %s", repro_result.summary())
        repro_summary = f"{repro_result.recurrences}/{repro_result.attempts} (threshold {repro_result.threshold})"
        if not repro_result.reproduced:
            logger.info("Failure %s did not reproduce >=%dx; treating as transient, stopping.",
                        mode.value, args.reproduce_threshold)
            return 0

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

            # ---- Targeted verification: the held-out/regression gate proves the
            # fix didn't break anything else; this proves it actually fixed the
            # thing it was for. Re-run the triggering input (on this branch, so
            # the fix is live) and require the target mode to be gone.
            targeted_summary = "skipped (no triggering input)"
            if trigger_input:
                verify = verify_target_fixed(
                    trigger_input, mode, trigger_run_id, workdir_root, model=args.model,
                    attempts=args.verify_attempts,
                )
                logger.info("Targeted verify: %s", verify.summary())
                targeted_summary = verify.summary()
                if not verify.fixed:
                    _run(["git", "checkout", args.base_branch])
                    _run(["git", "branch", "-D", branch])
                    logger.info("Targeted verification failed (%s still present); discarded branch %s, no PR.",
                                mode.value, branch)
                    return 1

            commit_sha = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
            _record_changelog(mode, diagnosis, proposals, results, repro_summary,
                              targeted_summary, baseline, proposed, commit_sha)
            _run(["git", "add", str(changelog_path())])
            _run(["git", "commit", "-m", f"Log harness change for {mode.value} to HARNESS_CHANGELOG"])

            _run(["git", "push", "-u", "origin", branch])
            pr_body = (
                f"## Diagnosis\n{diagnosis.root_cause}\n\n"
                f"**Confidence:** {diagnosis.confidence:.2f}\n\n"
                f"## Reproduced\n{repro_summary}\n\n"
                f"## Changes\n" + "\n".join(f"- `{c.ref.constant_name}`: {c.explanation}" for c in proposals) +
                f"\n\n## Targeted verification\n{targeted_summary}\n\n"
                f"## Gate results\n```\n{baseline.summary()}\n\n{proposed.summary()}\n```\n\n"
                f"Automated proposal from `improve.run_cycle` — reproduced {repro_summary}, targeted failure "
                f"fixed, held-out + regression gate passed ({reason}). Logged to HARNESS_CHANGELOG. "
                f"Requires human review before merge."
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
