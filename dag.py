"""Render the FlowReasonerAgent's execution_order as a Mermaid task DAG.

Real edges come from each step's `prerequisites` list when the model
populates it; in practice it often leaves that empty and only fills
`sequence_index`, so this falls back to chaining steps in sequence_index
order — still a faithful "what happens after what" DAG, just without
branching/parallel edges the model didn't call out explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path


def _sanitize(text: str, limit: int = 60) -> str:
    text = text.replace('"', "'").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def flow_reasoning_to_mermaid(flow_reasoning: dict) -> str:
    steps = sorted(flow_reasoning.get("execution_order", []), key=lambda s: s.get("sequence_index", 0))
    if not steps:
        return "flowchart TD\n  empty[\"No execution steps found\"]"

    lines = ["flowchart TD"]
    for step in steps:
        label = _sanitize(step.get("description", step["step_id"]))
        lines.append(f'  {step["step_id"]}["{step["step_id"]}: {label}"]')

    has_explicit_edges = any(step.get("prerequisites") for step in steps)
    if has_explicit_edges:
        for step in steps:
            for prereq in step.get("prerequisites", []):
                lines.append(f'  {prereq} --> {step["step_id"]}')
    else:
        for prev, cur in zip(steps, steps[1:]):
            lines.append(f'  {prev["step_id"]} --> {cur["step_id"]}')

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/flow_reasoning.json")
    print(flow_reasoning_to_mermaid(json.loads(path.read_text(encoding="utf-8"))))
