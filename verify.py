"""Standalone verification report for a generated_repo/ directory.

Self-contained (no dependency on the private eval/ framework) so this repo
can be public on its own: syntax check, unresolved-import check, and a
TODO/incomplete-implementation scan, reused for both the pre-baked sample
reports and any live "regenerate with your own key" run.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

INCOMPLETE_MARKERS = re.compile(r"\bTODO\b|\bFIXME\b|raise\s+NotImplementedError")

COMMON_EXTERNAL_PACKAGES = {
    "torch", "torchvision", "torchaudio", "numpy", "pandas", "sklearn",
    "scipy", "PIL", "cv2", "matplotlib", "seaborn", "tqdm", "yaml",
    "requests", "transformers", "datasets", "tensorboard", "wandb",
    "einops", "yacs", "omegaconf", "pytest", "setuptools", "pkg_resources",
}


@dataclass
class Finding:
    category: str  # "syntax_error" | "unresolved_import" | "incomplete_implementation" | "empty_main_block"
    file: str
    detail: str


@dataclass
class VerificationReport:
    files_checked: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {"files_checked": self.files_checked, "passed": self.passed,
                "findings": [asdict(f) for f in self.findings]}


def _known_external_packages(repo_dir: Path) -> set[str]:
    names = set(COMMON_EXTERNAL_PACKAGES)
    req_path = repo_dir / "requirements.txt"
    if req_path.exists():
        for line in req_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = re.split(r"[<>=!\[; ]", line)[0].strip()
            if pkg:
                names.add(pkg)
                names.add(pkg.replace("-", "_"))
    return names


def _has_empty_main_block(tree: ast.Module) -> bool:
    for node in tree.body:
        is_main_guard = (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        )
        if is_main_guard and all(isinstance(stmt, ast.Expr) for stmt in node.body):
            return True
    return False


def verify(repo_dir: Path) -> VerificationReport:
    report = VerificationReport()
    py_files = sorted(repo_dir.rglob("*.py")) if repo_dir.exists() else []
    report.files_checked = len(py_files)
    if not py_files:
        return report

    module_names = {p.relative_to(repo_dir).with_suffix("").as_posix().replace("/", ".") for p in py_files}
    package_roots = {name.split(".")[0] for name in module_names}
    stdlib_names = set(getattr(sys, "stdlib_module_names", ())) or set(sys.builtin_module_names)
    known_external = _known_external_packages(repo_dir)

    for path in py_files:
        rel = str(path.relative_to(repo_dir))
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            report.findings.append(Finding("syntax_error", rel, f"{exc.msg} (line {exc.lineno})"))
            continue

        if INCOMPLETE_MARKERS.search(source):
            report.findings.append(Finding("incomplete_implementation", rel,
                                            "Contains a TODO/FIXME/NotImplementedError marker."))

        if _has_empty_main_block(tree):
            report.findings.append(Finding("empty_main_block", rel,
                                            "`if __name__ == \"__main__\":` block has no executable statement "
                                            "(only comments) — a real run would raise IndentationError."))

        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                root = module.split(".")[0]
                if root in stdlib_names or root in known_external or root in package_roots:
                    continue
                report.findings.append(Finding("unresolved_import", rel,
                                                f"imports '{module}', which isn't stdlib, a listed "
                                                "dependency, or a local file."))

    return report


def run_flake8(repo_dir: Path) -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "flake8", "--select=F", str(repo_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("generated_repo")
    rpt = verify(target)
    print(json.dumps(rpt.to_dict(), indent=2))
