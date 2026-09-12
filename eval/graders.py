"""Deterministic graders that turn raw traces / generated code into failure
signals. Anything a grader can pin down directly is tagged with its
FailureMode right away; only ambiguous cases need the LLM judge (judge.py).
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval.taxonomy import FailureMode
from eval.trace_schema import TraceRecord

INCOMPLETE_MARKERS = re.compile(r"\bTODO\b|\bFIXME\b|raise\s+NotImplementedError|\.\.\.\s*$")

# Common ML/scientific-stack packages whose import name differs from (or is a
# reasonable superset of) their PyPI name, or that generated repos routinely
# rely on without always being listed verbatim in requirements.txt. Kept
# deliberately generous: a false "hallucinated import" signal is noisier than
# a missed one.
COMMON_EXTERNAL_PACKAGES = {
    "torch", "torchvision", "torchaudio", "numpy", "pandas", "sklearn",
    "scipy", "PIL", "cv2", "matplotlib", "seaborn", "tqdm", "yaml",
    "requests", "transformers", "datasets", "tensorboard", "wandb",
    "einops", "yacs", "omegaconf", "pytest", "setuptools", "pkg_resources",
}


@dataclass
class GraderSignal:
    mode: FailureMode
    detail: str
    evidence: str = ""


@dataclass
class GraderReport:
    signals: list[GraderSignal] = field(default_factory=list)

    @property
    def modes(self) -> list[FailureMode]:
        return [s.mode for s in self.signals]

    @property
    def has_failures(self) -> bool:
        return bool(self.signals)

    def add(self, mode: FailureMode, detail: str, evidence: str = "") -> None:
        self.signals.append(GraderSignal(mode, detail, evidence))

    def to_dict(self) -> dict:
        return {"signals": [{"mode": s.mode.value, "detail": s.detail, "evidence": s.evidence}
                             for s in self.signals]}


class TraceGrader:
    """Heuristics over a run's TraceRecords — the "tool use" / "reasoning" side."""

    def grade(self, traces: list[TraceRecord]) -> GraderReport:
        report = GraderReport()
        if not traces:
            report.add(FailureMode.MISSING_DEPENDENCY_INPUT, "No traces found for this run.")
            return report

        for record in traces:
            if record.success:
                continue

            error = (record.error or "").lower()
            if record.agent_name == "verifier":
                report.add(FailureMode.VERIFICATION_REJECTED,
                            "VerifierAgent rejected the UPS-IR structure.", record.error or "")
            elif "json" in error and ("decode" in error or "parse" in error):
                report.add(FailureMode.MALFORMED_JSON_OUTPUT,
                            f"{record.agent_name} failed to produce parseable JSON.", record.error or "")
            elif "validationerror" in error or "validation error" in error:
                report.add(FailureMode.SCHEMA_VALIDATION_FAILURE,
                            f"{record.agent_name} output failed schema validation.", record.error or "")
            elif "timeout" in error or "timed out" in error:
                report.add(FailureMode.TOOL_CALL_TIMEOUT,
                            f"{record.agent_name} timed out.", record.error or "")
            elif "filenotfound" in error or "no such file" in error:
                report.add(FailureMode.MISSING_DEPENDENCY_INPUT,
                            f"{record.agent_name} was missing a required input file.", record.error or "")
            else:
                report.add(FailureMode.RETRY_EXHAUSTED,
                            f"{record.agent_name} failed: {record.error}", record.error or "")

            if record.metadata.get("verified") is False:
                report.add(FailureMode.VERIFICATION_REJECTED, "VerifierAgent flagged the UPS-IR as invalid.")

        classifier_trace = next((t for t in traces if t.agent_name == "paper_type_classifier"), None)
        if classifier_trace and classifier_trace.metadata.get("low_confidence"):
            report.add(FailureMode.PAPER_TYPE_MISCLASSIFICATION,
                        "Paper-type classifier flagged low confidence.", classifier_trace.output_excerpt or "")

        return report


class CodeGrader:
    """Static checks over generated_repo/ — the "generated code" side."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = Path(repo_dir)

    def grade(self) -> GraderReport:
        report = GraderReport()
        if not self.repo_dir.exists():
            return report

        py_files = sorted(self.repo_dir.rglob("*.py"))
        if not py_files:
            return report

        self._check_syntax(py_files, report)
        self._check_incomplete(py_files, report)
        self._check_local_imports(py_files, report)
        self._run_flake8(report)
        self._run_mypy(report)
        return report

    def _check_syntax(self, py_files: list[Path], report: GraderReport) -> None:
        for path in py_files:
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                report.add(FailureMode.SYNTAX_ERROR, f"{path.relative_to(self.repo_dir)}: {exc.msg} (line {exc.lineno})")

    def _check_incomplete(self, py_files: list[Path], report: GraderReport) -> None:
        for path in py_files:
            text = path.read_text(encoding="utf-8")
            if INCOMPLETE_MARKERS.search(text):
                report.add(FailureMode.INCOMPLETE_IMPLEMENTATION,
                            f"{path.relative_to(self.repo_dir)} contains a TODO/NotImplementedError/stub marker.")

    def _known_external_packages(self) -> set[str]:
        names = set(COMMON_EXTERNAL_PACKAGES)
        for req_name in ("requirements.txt",):
            req_path = self.repo_dir / req_name
            if not req_path.exists():
                continue
            for line in req_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg = re.split(r"[<>=!\[; ]", line)[0].strip()
                if pkg:
                    names.add(pkg)
                    names.add(pkg.replace("-", "_"))
        return names

    def _check_local_imports(self, py_files: list[Path], report: GraderReport) -> None:
        module_names = {p.relative_to(self.repo_dir).with_suffix("").as_posix().replace("/", ".") for p in py_files}
        package_roots = {name.split(".")[0] for name in module_names}
        stdlib_names = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names)
        known_external = self._known_external_packages()

        def check(module: str, path: Path) -> None:
            root = module.split(".")[0]
            if root in stdlib_names or root in known_external:
                return
            if root in package_roots:
                if module in module_names or any(m.startswith(module + ".") for m in module_names):
                    return
                report.add(FailureMode.IMPORT_DEPENDENCY_ERROR,
                            f"{path.relative_to(self.repo_dir)} imports '{module}', which doesn't "
                            "resolve to any file in the repo.")
                return
            report.add(FailureMode.IMPORT_DEPENDENCY_ERROR,
                        f"{path.relative_to(self.repo_dir)} imports '{module}', which is neither a "
                        "local file, a stdlib module, nor listed in requirements.txt.")

        for path in py_files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue  # already reported by _check_syntax
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    check(node.module, path)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        check(alias.name, path)

    def _run_flake8(self, report: GraderReport) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "--select=F", str(self.repo_dir)],
                capture_output=True, text=True, timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        issues = [line for line in result.stdout.splitlines() if line.strip()]
        if issues:
            report.add(FailureMode.LINT_VIOLATION,
                        f"flake8 found {len(issues)} undefined-name/unused-import style issues.",
                        "\n".join(issues[:20]))

    def _run_mypy(self, report: GraderReport) -> None:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--ignore-missing-imports", "--no-error-summary", str(self.repo_dir)],
                capture_output=True, text=True, timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        issues = [line for line in result.stdout.splitlines() if ": error:" in line]
        if issues:
            report.add(FailureMode.STATIC_TYPE_ERROR,
                        f"mypy found {len(issues)} type errors.", "\n".join(issues[:20]))
