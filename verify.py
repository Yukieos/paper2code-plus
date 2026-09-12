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
        if not is_main_guard or not node.body:
            continue
        # A no-op block is one where every statement is a bare constant
        # expression (a string used as a comment, or `...`) — NOT any
        # ast.Expr, since a normal `main()` call is *also* an Expr statement
        # and must not be flagged here.
        if all(isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) for stmt in node.body):
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

    report.findings.extend(_flake8_findings(repo_dir))
    return report


def _flake8_findings(repo_dir: Path) -> list[Finding]:
    """flake8 --select=F catches things AST-walking above doesn't: undefined
    names (F821 — e.g. a type hint referencing an unimported symbol), unused
    imports/variables, redefinitions, etc.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "flake8", "--select=F", str(repo_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    findings = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # "<path>:<line>:<col>: <code> <message>"
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue
        path_str, line_no, _col, message = parts
        try:
            rel = str(Path(path_str).resolve().relative_to(repo_dir.resolve()))
        except ValueError:
            rel = path_str
        findings.append(Finding("lint_violation", rel, f"line {line_no}:{message.strip()}"))
    return findings


SEMANTIC_REVIEW_PROMPT = """You are auditing a PyTorch repository generated by an automated \
paper-to-code pipeline, looking for real bugs a linter cannot catch — the kind that make this \
fail or silently do nothing when actually run:

- The same function/class defined in more than one file with incompatible signatures, where a
  caller might be using either one.
- A function called with arguments (names, count, order) that don't match its real definition.
- Stub/no-op logic dressed up to look complete (e.g. a method that never uses its inputs, a
  loop that does nothing, a "training" function that never calls backward()/step()).
- A generated entry point that doesn't actually do what its own docstring/name claims.
- An obviously wrong assumption about the data (e.g. assuming a local file exists that nothing
  in this repo creates or downloads).

Repository files:
{files_blob}

Only report issues you are concretely confident about — cite the exact file and function/line \
involved and what would actually go wrong. Do not report style preferences, missing type hints, \
or unused imports (a separate linter already covers those).

Respond with ONLY a JSON object of this exact shape:
{{"issues": [{{"file": "...", "severity": "critical|moderate|minor", "detail": "..."}}]}}
Return {{"issues": []}} if you find nothing you're confident about.
"""


def _default_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    return llm.invoke(prompt).content


def _extract_json_object(text: str) -> str:
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in semantic review output: {text[:200]!r}")
    return text[start:end + 1]


def semantic_review(repo_dir: Path, llm_call=None, model: str = "gpt-4o-mini",
                     max_chars: int = 20000) -> list[Finding]:
    """LLM-as-judge pass over the actual code, for the class of bug that's
    semantic rather than mechanical — duplicate/inconsistent definitions,
    wrong call signatures, stub logic — which AST/flake8 checks structurally
    cannot see (a stub method and a real one are equally valid Python).

    Swappable `llm_call` (same convention as eval/judge.py in the private
    companion repo) so this is unit-testable without hitting a real API.
    Requires an OpenAI-compatible API key to be configured when no
    `llm_call` is supplied — callers should treat this as opt-in, not part
    of the always-on, free static checks in `verify()`.
    """
    py_files = sorted(repo_dir.rglob("*.py")) if repo_dir.exists() else []
    if not py_files:
        return []

    blob_parts = []
    for path in py_files:
        rel = path.relative_to(repo_dir)
        blob_parts.append(f"### {rel}\n```python\n{path.read_text(encoding='utf-8')}\n```")
    files_blob = "\n\n".join(blob_parts)
    if len(files_blob) > max_chars:
        files_blob = files_blob[:max_chars] + "\n...<truncated>"

    call = llm_call or (lambda prompt: _default_llm_call(prompt, model=model))
    prompt = SEMANTIC_REVIEW_PROMPT.format(files_blob=files_blob)

    try:
        data = json.loads(_extract_json_object(call(prompt)))
    except Exception as exc:  # pragma: no cover - network/model dependent
        return [Finding("semantic_review_failed", "-", f"LLM semantic review call failed: {exc}")]

    return [
        Finding("semantic_issue", issue.get("file", "-"),
                f"[{issue.get('severity', 'unknown')}] {issue.get('detail', '')}")
        for issue in data.get("issues", [])
    ]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("generated_repo")
    rpt = verify(target)
    if "--llm" in sys.argv:
        rpt.findings.extend(semantic_review(target))
    print(json.dumps(rpt.to_dict(), indent=2))
