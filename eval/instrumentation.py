"""Thin wrapper around `agent.run(state)` that emits a TraceRecord.

Usage at a call site (see main.py / codegen_pipeline.py):

    state = run_traced(agent, state, run_id=run_id, stage="extraction",
                        agent_name="extractor", store=store)

instead of:

    state = agent.run(state)

When tracing is disabled (the default — see trace_store.tracing_enabled),
this is a zero-cost passthrough to agent.run(state).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from eval.trace_schema import TraceRecord
from eval.trace_store import TraceStore, tracing_enabled

logger = logging.getLogger(__name__)


def default_run_id() -> str:
    """PAPER2CODE_RUN_ID if set, else the cwd's folder name.

    The Streamlit UI runs each stage's subprocess with cwd=runs/<timestamp>/,
    so Stage 1 and Stage 2 traces for the same UI run naturally share a
    run_id without any extra plumbing between the two subprocess calls.
    """
    return os.environ.get("PAPER2CODE_RUN_ID") or Path.cwd().name


def run_traced(
    agent: Any,
    state: dict,
    *,
    run_id: str,
    stage: str,
    agent_name: str,
    store: TraceStore | None = None,
    model: str | None = None,
    metadata: dict | None = None,
) -> dict:
    if not tracing_enabled():
        return agent.run(state)

    store = store or TraceStore()
    record = TraceRecord(
        run_id=run_id,
        stage=stage,
        agent_name=agent_name,
        model=model,
        input_state_keys=sorted(state.keys()),
        input_excerpt=TraceRecord.make_excerpt(state),
        metadata=metadata or {},
    )

    try:
        result = agent.run(state)
    except Exception as exc:
        record.finish(success=False, error=f"{type(exc).__name__}: {exc}")
        store.put_trace(record)
        raise

    # Agents that return a bool (e.g. VerifierAgent) don't produce a new
    # state; the caller keeps using the original `state` in that case.
    output_state = result if isinstance(result, dict) else state
    success = True
    extra_metadata = dict(record.metadata)
    if isinstance(result, bool):
        success = result
        extra_metadata["verified"] = result
    record.metadata = extra_metadata

    record.finish(
        success=success,
        output_state_keys=sorted(output_state.keys()) if isinstance(output_state, dict) else None,
        output_excerpt=result,
    )
    store.put_trace(record)
    return result
