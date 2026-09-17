"""Whole-codebase, source-grounded diagnosis for a failed reproduction.

The shallow "regenerate the one file the traceback names" approach can't fix a
cross-file mismatch (it just oscillates between the two files) and has no way to
tell whether the code is even faithful to the paper. This diagnoses the WHOLE
generated repo at once, grounded in four things:

  1. the runtime failure (full traceback / no-progress signal),
  2. every generated file (so a mismatch is fixed on both sides together),
  3. WHY the agents wrote it this way (plan, flow reasoning, code reviews, and —
     when available — the agent execution trace),
  4. the paper itself (UPS-IR sections + the original text) as ground truth.

It returns a coherent MULTI-FILE patch (or a decision to escalate to re-plan).
Pure orchestration over an injected llm_call, so it's tested offline with a fake
model (see tests/test_repro_diagnose.py); no torch, no codegen_pipeline import
(keeps it dependency-light and free of import cycles).
"""
from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FileEdit:
    path: str          # repo-relative
    new_content: str
    rationale: str = ""


@dataclass
class RepairPlan:
    root_cause: str = ""
    faithfulness: str = ""          # how/whether the code diverges from the paper
    escalate: bool = False          # a coherent edit set can't fix it -> re-plan
    edits: list[FileEdit] = field(default_factory=list)

    def summary(self) -> str:
        if self.escalate and not self.edits:
            return f"escalate (no coherent multi-file fix): {self.root_cause[:160]}"
        return f"{len(self.edits)} file edit(s): {self.root_cause[:160]}"


def read_repo_files(repo_dir: Path, per_file_chars: int = 8000) -> dict[str, str]:
    """Every .py in the repo, repo-relative path -> content (truncated)."""
    repo_dir = Path(repo_dir)
    files: dict[str, str] = {}
    for path in sorted(repo_dir.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = str(path.relative_to(repo_dir))
        files[rel] = text if len(text) <= per_file_chars else text[:per_file_chars] + "\n# ...<truncated>"
    return files


def _signatures(files: dict[str, str]) -> str:
    """A compact cross-file signature index, so the model sees every file's real
    top-level defs at a glance even where full bodies were truncated."""
    lines = []
    for rel, text in files.items():
        module = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
        try:
            tree = ast.parse(text)
        except SyntaxError:
            lines.append(f"- `{module}`: (syntax error — cannot introspect)")
            continue
        defs = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(f"def {node.name}({', '.join(a.arg for a in node.args.args)})")
            elif isinstance(node, ast.ClassDef):
                methods = [f"{n.name}({', '.join(a.arg for a in n.args.args)})"
                           for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                defs.append(f"class {node.name}: [{', '.join(methods)}]")
        if defs:
            lines.append(f"- `{module}`: " + "; ".join(defs))
    return "\n".join(lines) or "(no definitions)"


def paper_text_from_ups_ir(ups_ir: dict[str, Any], max_chars: int = 4000) -> str:
    """Paper ground truth: prefer UPS-IR sections (with their source refs); fall
    back to a dump of the spec. This is what faithfulness is judged against."""
    sections = ups_ir.get("sections") or []
    if sections:
        parts = []
        for s in sections:
            title = s.get("title") or s.get("name") or s.get("source_reference") or ""
            body = s.get("text") or s.get("content") or s.get("summary") or ""
            parts.append(f"## {title}\n{body}".strip())
        text = "\n\n".join(p for p in parts if p.strip())
        if text.strip():
            return text if len(text) <= max_chars else text[:max_chars] + "\n...<truncated>"
    dump = json.dumps(ups_ir, ensure_ascii=False)[:max_chars]
    return dump + ("...<truncated>" if len(dump) >= max_chars else "")


def _plan_why(plan: dict[str, Any] | None) -> str:
    if not plan:
        return "(plan unavailable)"
    lines = []
    for f in plan.get("files", []):
        skels = ", ".join(s.get("name", "") for s in f.get("function_skeletons", []))
        lines.append(f"- {f.get('path')}: {f.get('purpose','')}" + (f" [functions: {skels}]" if skels else ""))
    return "\n".join(lines) or "(plan has no files)"


def _reviews_why(code_reviews: dict[str, str] | None) -> str:
    if not code_reviews:
        return "(no code reviews captured)"
    out = []
    for path, review in code_reviews.items():
        out.append(f"- {path}: {str(review)[:300]}")
    return "\n".join(out)


def build_prompt(
    repo_files: dict[str, str],
    smoke_reason: str,
    traceback: str,
    ups_ir: dict[str, Any],
    plan: dict[str, Any] | None = None,
    code_reviews: dict[str, str] | None = None,
    flow_reasoning: dict[str, Any] | None = None,
    trace_text: str = "",
    paper_text: str = "",
) -> str:
    files_block = "\n\n".join(f"### FILE: {rel}\n```python\n{content}\n```" for rel, content in repo_files.items())
    paper = paper_text.strip() or paper_text_from_ups_ir(ups_ir)
    flow = json.dumps(flow_reasoning, ensure_ascii=False)[:1500] if flow_reasoning else "(none)"
    trace = trace_text.strip() or "(agent execution trace unavailable for this run)"

    return (
        "You are diagnosing why a GENERATED code repository, meant to reproduce a paper, fails to run.\n"
        "Diagnose the WHOLE repo together — many failures are cross-file (a caller and callee whose "
        "signatures disagree), so fixing one file in isolation just moves the crash. Ground your "
        "diagnosis in the paper (is the code even faithful to what the paper describes?) and in why the "
        "agents generated it this way.\n\n"
        f"=== RUNTIME FAILURE ===\n{smoke_reason}\n\nTraceback / error:\n{traceback[:3000]}\n\n"
        f"=== CROSS-FILE SIGNATURE INDEX ===\n{_signatures(repo_files)}\n\n"
        f"=== ALL GENERATED FILES ===\n{files_block}\n\n"
        f"=== WHY THE AGENTS GENERATED THIS (plan) ===\n{_plan_why(plan)}\n\n"
        f"=== CODE REVIEW NOTES ===\n{_reviews_why(code_reviews)}\n\n"
        f"=== FLOW REASONING ===\n{flow}\n\n"
        f"=== AGENT EXECUTION TRACE ===\n{trace}\n\n"
        f"=== PAPER (ground truth) ===\n{paper}\n\n"
        "Decide the smallest COHERENT set of file edits that makes the repo run AND stay faithful to the "
        "paper. If a caller/callee disagree, edit BOTH so they match. If no edit set can fix it (the plan "
        "itself is wrong — missing files, wrong decomposition), set escalate=true with an empty edits list.\n\n"
        "Respond with ONLY a JSON object of this exact shape:\n"
        '{\n'
        '  "root_cause": "<the cross-file root cause, concrete>",\n'
        '  "faithfulness": "<how the code diverges from the paper, or \\"faithful\\">",\n'
        '  "escalate": <true|false>,\n'
        '  "edits": [ {"path": "<repo-relative>", "new_content": "<full new file content>", "rationale": "<why>"} ]\n'
        '}\n'
        "Each edit's new_content MUST be the COMPLETE file, no markdown fences."
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in diagnosis output: {text[:200]!r}")
    return text[start:end + 1]


def _strip_fences(code: str) -> str:
    c = code.strip()
    if c.startswith("```"):
        lines = c.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        c = "\n".join(lines).strip()
    return c


def diagnose_reproduction(prompt: str, llm_call: Callable[[str], str]) -> RepairPlan:
    """Run the diagnosis LLM call and parse a RepairPlan. Any failure to produce
    a usable plan becomes escalate=True (let the caller fall back to re-plan)."""
    try:
        raw = llm_call(prompt)
        data = json.loads(_extract_json(raw), strict=False)  # code bodies have literal newlines
    except Exception as exc:  # pragma: no cover - network/model dependent
        logger.warning("Reproduction diagnosis failed to parse (%s); escalating.", exc)
        return RepairPlan(root_cause=f"diagnosis unparseable: {exc}", escalate=True)

    edits = []
    for e in data.get("edits") or []:
        path, content = e.get("path"), e.get("new_content")
        if not path or not content:
            continue
        content = _strip_fences(str(content))
        if len(content) < 20:
            continue
        edits.append(FileEdit(path=str(path), new_content=content, rationale=str(e.get("rationale", ""))))

    return RepairPlan(
        root_cause=str(data.get("root_cause", "")),
        faithfulness=str(data.get("faithfulness", "")),
        escalate=bool(data.get("escalate", False)) or not edits,
        edits=edits,
    )


def apply_repair_plan(repo_dir: Path, plan: RepairPlan) -> list[str]:
    """Write each edit's new content. Refuses paths that escape the repo or don't
    already exist (the diagnosis edits existing files, it doesn't add new ones —
    adding files is a re-plan's job). Returns the repo-relative paths changed."""
    repo_dir = Path(repo_dir).resolve()
    changed: list[str] = []
    for edit in plan.edits:
        target = (repo_dir / edit.path).resolve()
        try:
            target.relative_to(repo_dir)
        except ValueError:
            logger.warning("Repair edit path escapes repo, skipping: %s", edit.path)
            continue
        if not target.exists():
            logger.warning("Repair edit targets a non-existent file, skipping: %s", edit.path)
            continue
        target.write_text(edit.new_content, encoding="utf-8")
        changed.append(edit.path)
    return changed
