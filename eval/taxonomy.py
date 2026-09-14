"""The 18-mode agent failure taxonomy.

Grouped into the three areas the pipeline actually breaks in: tool use
(structured I/O with the LLM and the filesystem), reasoning/planning
(misreading the paper or its own plan), and generated code (what ends up in
generated_repo/). Grader signals (eval/graders.py) can point directly at a
mode; anything ambiguous goes to the LLM judge (eval/judge.py) to classify.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    # --- Tool use -----------------------------------------------------
    MALFORMED_JSON_OUTPUT = "malformed_json_output"
    SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"
    TOOL_CALL_TIMEOUT = "tool_call_timeout"
    RETRY_EXHAUSTED = "retry_exhausted"
    FILE_WRITE_FAILURE = "file_write_failure"
    MISSING_DEPENDENCY_INPUT = "missing_dependency_input"

    # --- Reasoning / planning ------------------------------------------
    PAPER_MISUNDERSTANDING = "paper_misunderstanding"
    VERIFICATION_REJECTED = "verification_rejected"
    PLAN_CODE_MISMATCH = "plan_code_mismatch"
    PAPER_TYPE_MISCLASSIFICATION = "paper_type_misclassification"
    PARAMETER_HALLUCINATION = "parameter_hallucination"
    UNRESOLVED_MISSING_PARAMETER = "unresolved_missing_parameter"

    # --- Generated code --------------------------------------------------
    SYNTAX_ERROR = "syntax_error"
    STATIC_TYPE_ERROR = "static_type_error"
    LINT_VIOLATION = "lint_violation"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    IMPORT_DEPENDENCY_ERROR = "import_dependency_error"
    RUNTIME_EXECUTION_FAILURE = "runtime_execution_failure"

    # --- Meta ------------------------------------------------------------
    # A real failure the judge identified but could NOT map to any of the 18
    # modes above. Deliberately kept out of TAXONOMY/the rubric: it's not a
    # 19th category to classify into, it's a taxonomy-EXPANSION signal. A
    # rising count of OTHER means the pipeline is failing in a way we haven't
    # named yet and a human should add a mode for it.
    OTHER = "other"


@dataclass(frozen=True)
class FailureModeSpec:
    mode: FailureMode
    category: str
    description: str


TAXONOMY: list[FailureModeSpec] = [
    FailureModeSpec(FailureMode.MALFORMED_JSON_OUTPUT, "tool_use",
                     "LLM output could not be parsed as JSON after all retry/repair attempts."),
    FailureModeSpec(FailureMode.SCHEMA_VALIDATION_FAILURE, "tool_use",
                     "Output parsed but failed Pydantic/JSON-schema validation against the expected shape."),
    FailureModeSpec(FailureMode.TOOL_CALL_TIMEOUT, "tool_use",
                     "An agent step exceeded its time budget or otherwise appears to have hung."),
    FailureModeSpec(FailureMode.RETRY_EXHAUSTED, "tool_use",
                     "Agent hit its max-retry limit without ever producing valid output."),
    FailureModeSpec(FailureMode.FILE_WRITE_FAILURE, "tool_use",
                     "An expected output artifact (plan, config, source file, ...) was never written to disk."),
    FailureModeSpec(FailureMode.MISSING_DEPENDENCY_INPUT, "tool_use",
                     "An agent ran without a required upstream artifact (e.g. UPS-IR.json) present or non-empty."),
    FailureModeSpec(FailureMode.PAPER_MISUNDERSTANDING, "reasoning",
                     "Extracted UPS-IR contradicts or omits material content from the source paper."),
    FailureModeSpec(FailureMode.VERIFICATION_REJECTED, "reasoning",
                     "VerifierAgent rejected the UPS-IR structure, halting the pipeline before synthesis."),
    FailureModeSpec(FailureMode.PLAN_CODE_MISMATCH, "reasoning",
                     "The code plan or generated code diverges from what the UPS-IR / plan review actually specified."),
    FailureModeSpec(FailureMode.PAPER_TYPE_MISCLASSIFICATION, "reasoning",
                     "PaperTypeClassifier picked the wrong paper type, mismatching downstream prompts/templates."),
    FailureModeSpec(FailureMode.PARAMETER_HALLUCINATION, "reasoning",
                     "A parameter value was invented with no source_reference, or contradicts the paper's stated value."),
    FailureModeSpec(FailureMode.UNRESOLVED_MISSING_PARAMETER, "reasoning",
                     "A high-risk parameter was left unfilled without being flagged to the user."),
    FailureModeSpec(FailureMode.SYNTAX_ERROR, "generated_code",
                     "A generated file fails to compile (invalid Python syntax)."),
    FailureModeSpec(FailureMode.STATIC_TYPE_ERROR, "generated_code",
                     "mypy flags a real type error in generated code."),
    FailureModeSpec(FailureMode.LINT_VIOLATION, "generated_code",
                     "flake8 flags significant issues: undefined names, unused imports/vars, etc."),
    FailureModeSpec(FailureMode.INCOMPLETE_IMPLEMENTATION, "generated_code",
                     "Generated code contains TODOs, bare `pass`, or NotImplementedError instead of real logic."),
    FailureModeSpec(FailureMode.IMPORT_DEPENDENCY_ERROR, "generated_code",
                     "A generated file imports a local module/file that doesn't exist anywhere in the repo."),
    FailureModeSpec(FailureMode.RUNTIME_EXECUTION_FAILURE, "generated_code",
                     "The generated repo raises an exception when actually executed (smoke run)."),
]

OTHER_SPEC = FailureModeSpec(
    FailureMode.OTHER, "meta",
    "A real failure the judge could not map to any of the 18 known modes — a "
    "taxonomy-expansion signal, not an auto-fixable category.",
)

# TAXONOMY stays the 18 named modes (and drives the rubric). BY_MODE also
# includes OTHER so lookups (e.g. in diagnose) never KeyError on it.
TAXONOMY_BY_MODE: dict[FailureMode, FailureModeSpec] = {spec.mode: spec for spec in TAXONOMY}
TAXONOMY_BY_MODE[FailureMode.OTHER] = OTHER_SPEC

assert len(TAXONOMY) == 18, f"Expected 18 failure modes, found {len(TAXONOMY)}"


def render_rubric() -> str:
    """Render the taxonomy as a numbered rubric for the LLM-as-judge prompt."""
    lines = []
    for spec in TAXONOMY:
        lines.append(f"- `{spec.mode.value}` ({spec.category}): {spec.description}")
    return "\n".join(lines)
