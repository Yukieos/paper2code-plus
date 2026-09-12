"""Registry of every prompt an automated change is allowed to touch.

Scope is deliberately narrow: only the module-level PROMPT string
constants actually wired into the live pipeline (verified by reading
main.py / codegen_pipeline.py's agent wiring, not just grepping for
"PROMPT") are listed here. Everything else — control flow, parsing,
grader logic, the pipeline's structure — is out of bounds for automated
change proposals.

Note: codegen_pipeline.py defines a `CODE_REVIEW_PROMPT` twice (once for
the unused CodeWriterAgent, once for the live CodeReviewDebugAgent) — the
second definition shadows the first at import time, and only the second
is ever actually used. This registry intentionally points at that live
one via its line-based lookup, not the dead one earlier in the file.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PromptRef:
    agent_name: str
    file_path: str  # relative to repo root
    constant_name: str
    occurrence: int = 0  # which top-level assignment to this name, 0-indexed (for shadowed dupes)


# agent_name matches the `agent_name=` used in eval/instrumentation.run_traced()
# call sites in main.py / codegen_pipeline.py.
PROMPT_REGISTRY: dict[str, list[PromptRef]] = {
    "extractor": [
        PromptRef("extractor", "agents/extractor.py", "PROMPT_TEMPLATE"),
        PromptRef("extractor", "agents/extractor.py", "RETRY_PROMPT_TEMPLATE"),
    ],
    "planner": [
        PromptRef("planner", "codegen_pipeline.py", "PLAN_PROMPT"),
    ],
    "plan_reviewer": [
        PromptRef("plan_reviewer", "codegen_pipeline.py", "PLAN_REVIEW_PROMPT"),
        PromptRef("plan_reviewer", "codegen_pipeline.py", "PLAN_REVISE_PROMPT"),
    ],
    "flow_reasoner": [
        PromptRef("flow_reasoner", "codegen_pipeline.py", "FLOW_REASONER_PROMPT"),
    ],
    "dependency_analyzer": [
        PromptRef("dependency_analyzer", "codegen_pipeline.py", "DEPENDENCY_ANALYSIS_PROMPT"),
    ],
    "file_analyzer": [
        PromptRef("file_analyzer", "codegen_pipeline.py", "FILE_ANALYSIS_PROMPT"),
    ],
    "dataset_agent": [
        PromptRef("dataset_agent", "codegen_pipeline.py", "DATASET_PROMPT"),
    ],
    "execution_agent": [
        PromptRef("execution_agent", "codegen_pipeline.py", "EXECUTION_PROMPT_ML"),
        PromptRef("execution_agent", "codegen_pipeline.py", "EXECUTION_PROMPT_ALGORITHM"),
    ],
    "evaluation_agent": [
        PromptRef("evaluation_agent", "codegen_pipeline.py", "EVALUATION_PROMPT"),
    ],
    "entry_point_agent": [
        PromptRef("entry_point_agent", "codegen_pipeline.py", "ENTRY_POINT_PROMPT"),
    ],
    "code_review_debug": [
        # occurrence=1: the second (live) definition — see module docstring.
        PromptRef("code_review_debug", "codegen_pipeline.py", "CODE_REVIEW_PROMPT", occurrence=1),
        PromptRef("code_review_debug", "codegen_pipeline.py", "CODE_FIX_PROMPT"),
    ],
}


def _find_assignment(tree: ast.Module, constant_name: str, occurrence: int) -> ast.Assign:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == constant_name
    ]
    if occurrence >= len(matches):
        raise ValueError(f"Expected occurrence {occurrence} of '{constant_name}', found {len(matches)}.")
    return matches[occurrence]


def get_prompt_text(ref: PromptRef) -> str:
    source = (REPO_ROOT / ref.file_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=ref.file_path)
    assign = _find_assignment(tree, ref.constant_name, ref.occurrence)
    value = assign.value
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        raise ValueError(f"{ref.constant_name} in {ref.file_path} is not a plain string constant; "
                          "refusing to treat it as an editable prompt.")
    return value.value


def placeholders(text: str) -> set[str]:
    """The {name} template variables a prompt string relies on."""
    return set(re.findall(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})", text))


def _format_literal(text: str) -> str:
    if '"""' not in text and not text.endswith('"') and not text.endswith("\\"):
        return f'"""{text}"""'
    return repr(text)


def set_prompt_text(ref: PromptRef, new_text: str) -> None:
    """Rewrite the prompt constant in place, preserving the rest of the file.

    Refuses the change if it would alter the set of {placeholder} variables
    the prompt template relies on — that would break the ChatPromptTemplate
    call site (missing/extra input variable) rather than just changing wording.
    """
    old_text = get_prompt_text(ref)
    old_vars, new_vars = placeholders(old_text), placeholders(new_text)
    if old_vars != new_vars:
        raise ValueError(
            f"Refusing to update {ref.constant_name} in {ref.file_path}: template variables changed "
            f"(had {sorted(old_vars)}, proposal has {sorted(new_vars)}). This would break the call site."
        )

    path = REPO_ROOT / ref.file_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=ref.file_path)
    assign = _find_assignment(tree, ref.constant_name, ref.occurrence)
    value = assign.value

    lines = source.splitlines(keepends=True)
    start_line, start_col = value.lineno - 1, value.col_offset
    end_line, end_col = value.end_lineno - 1, value.end_col_offset

    if start_line == end_line:
        line = lines[start_line]
        lines[start_line] = line[:start_col] + _format_literal(new_text) + line[end_col:]
    else:
        first = lines[start_line][:start_col]
        last = lines[end_line][end_col:]
        lines[start_line:end_line + 1] = [first + _format_literal(new_text) + last]

    new_source = "".join(lines)
    ast.parse(new_source, filename=ref.file_path)  # fail loudly before writing anything broken
    path.write_text(new_source, encoding="utf-8")


def refs_for_agent(agent_name: str) -> list[PromptRef]:
    return PROMPT_REGISTRY.get(agent_name, [])
