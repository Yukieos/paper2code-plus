"""CLI: mine execution traces for failures and produce a report.

    python -m eval.mine_failures                       # mine every run found
    python -m eval.mine_failures --run-ids 20260912-... # mine specific runs
    python -m eval.mine_failures --no-judge             # skip the LLM judge (grader signals only)
    python -m eval.mine_failures --cluster              # also cluster failures within/across modes
    python -m eval.mine_failures --output report.json

Looks for each run's generated_repo/ under runs/<run_id>/generated_repo (where
the Streamlit UI puts it) — pass --runs-dir to point elsewhere.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval.graders import CodeGrader, GraderReport, TraceGrader
from eval.judge import JudgeVerdict, LLMJudge
from eval.taxonomy import FailureMode
from eval.trace_store import TraceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_env() -> None:
    """Best-effort load of the repo's .env (by absolute path, so it works from
    any cwd) — the judge and embeddings read OPENAI_API_KEY from the environment
    and there's no reason to make callers wrap this by hand."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class RunResult:
    run_id: str
    trace_count: int
    grader_signals: GraderReport
    judge_verdict: JudgeVerdict | None

    @property
    def all_modes(self) -> list[FailureMode]:
        modes = list(self.grader_signals.modes)
        if self.judge_verdict and self.judge_verdict.primary_failure_mode:
            modes.append(self.judge_verdict.primary_failure_mode)
        if self.judge_verdict:
            modes.extend(self.judge_verdict.secondary_failure_modes)
        return modes

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "trace_count": self.trace_count,
            "grader_signals": self.grader_signals.to_dict(),
            "judge_verdict": self.judge_verdict.to_dict() if self.judge_verdict else None,
            "failure_modes": [m.value for m in self.all_modes],
        }


def mine_run(run_id: str, store: TraceStore, runs_dir: Path, use_judge: bool, judge: LLMJudge | None) -> RunResult:
    traces = store.get_traces(run_id)

    trace_report = TraceGrader().grade(traces)
    repo_dir = runs_dir / run_id / "generated_repo"
    code_report = CodeGrader(repo_dir).grade()

    combined = GraderReport(signals=trace_report.signals + code_report.signals)

    verdict = None
    if use_judge and judge is not None:
        verdict = judge.classify(traces, combined)

    return RunResult(run_id=run_id, trace_count=len(traces), grader_signals=combined, judge_verdict=verdict)


def build_report(results: list[RunResult], clusters: "list | None" = None) -> dict:
    mode_counts = Counter()
    for result in results:
        mode_counts.update(m.value for m in result.all_modes)

    healthy_runs = [r.run_id for r in results if not r.all_modes]
    failing_runs = [r.run_id for r in results if r.all_modes]

    report = {
        "runs_mined": len(results),
        "healthy_runs": healthy_runs,
        "failing_runs": failing_runs,
        "failure_mode_counts": dict(mode_counts.most_common()),
        "runs": [r.to_dict() for r in results],
    }
    if clusters is not None:
        report["failure_clusters"] = [c.to_dict() for c in clusters]
    return report


def print_summary(report: dict) -> None:
    print(f"\nMined {report['runs_mined']} run(s): "
          f"{len(report['healthy_runs'])} healthy, {len(report['failing_runs'])} with failures.\n")
    if report["failure_mode_counts"]:
        print("Failure mode breakdown (most common first):")
        for mode, count in report["failure_mode_counts"].items():
            print(f"  {count:>3}  {mode}")
    else:
        print("No failures detected.")

    clusters = report.get("failure_clusters")
    if clusters:
        print("\nFailure clusters (recurring patterns within/across modes, largest first):")
        for cluster in clusters:
            if cluster["size"] < 2:
                continue
            print(f"  [{cluster['size']}] {cluster['label']}  (modes: {', '.join(cluster['modes']) or 'unclassified'})")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Mine Paper2Code agent traces for recurring failures.")
    parser.add_argument("--run-ids", type=str, default=None,
                         help="Comma-separated run ids to mine. Default: every run in the trace store.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                         help="Where to look for <run_id>/generated_repo (default: ./runs).")
    parser.add_argument("--bucket", type=str, default=None, help="Override the S3 trace bucket.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge classification step.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for the LLM judge.")
    parser.add_argument("--cluster", action="store_true",
                         help="Also embed+cluster failures to surface recurring patterns within/across the "
                              "fixed taxonomy modes (e.g. several differently-worded failures sharing one "
                              "root cause). Costs one embeddings call plus one label call per cluster.")
    parser.add_argument("--cluster-eps", type=float, default=0.35,
                         help="DBSCAN cosine-distance threshold — lower = stricter clustering (default: 0.35).")
    parser.add_argument("--judge-votes", type=int, default=3,
                         help="LLM-judge votes per run, aggregated by majority/median (default 3).")
    parser.add_argument("--output", type=Path, default=None,
                         help="Where to write the JSON report (default: output/failure_report_<timestamp>.json).")
    args = parser.parse_args(argv)
    _load_env()  # so the judge + embeddings get OPENAI_API_KEY without a manual wrapper

    store = TraceStore(bucket=args.bucket)
    run_ids = args.run_ids.split(",") if args.run_ids else store.list_run_ids()

    if not run_ids:
        logger.warning("No runs found to mine. Did you set PAPER2CODE_TRACE_ENABLED=1 during pipeline runs?")

    judge = None if args.no_judge else LLMJudge(model=args.model, votes=args.judge_votes)

    results = [mine_run(run_id.strip(), store, args.runs_dir, not args.no_judge, judge) for run_id in run_ids if run_id.strip()]

    clusters = None
    if args.cluster:
        from eval.cluster_failures import cluster_failures

        logger.info("Clustering failures...")
        try:
            clusters = cluster_failures(results, eps=args.cluster_eps)
        except Exception as exc:
            # Clustering needs embeddings (an API key). Degrade to a
            # cluster-less report instead of crashing the whole mine.
            logger.warning("Clustering unavailable (%s); writing report without clusters.", exc)
            clusters = None

    report = build_report(results, clusters=clusters)

    output_path = args.output
    if output_path is None:
        from datetime import datetime
        output_path = Path("output") / f"failure_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote failure report to %s", output_path)

    print_summary(report)
    return report


if __name__ == "__main__":
    main()
