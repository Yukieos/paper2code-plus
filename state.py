from __future__ import annotations

from typing import Any, Dict, TypedDict


class PipelineState(TypedDict, total=False):
    text: str
    info: Dict[str, Any]
    ups_ir: Dict[str, Any]
    artifacts: Dict[str, str]
    # ReAct backtrack (verifier -> extractor): see main.py's run_sequential /
    # run_via_graph. verifier_feedback carries the specific rejection reason
    # into the extractor's retry prompt; verification_retries bounds the loop.
    verifier_feedback: str
    verification_retries: int


def initial_state() -> PipelineState:
    return PipelineState()
