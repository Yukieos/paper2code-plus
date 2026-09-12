"""Data shape for one agent execution trace."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _excerpt(value: Any, limit: int = 2000) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


@dataclass
class TraceRecord:
    run_id: str
    stage: str  # "extraction" | "codegen"
    agent_name: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(default_factory=_now_iso)
    ended_at: str | None = None
    duration_ms: float | None = None
    success: bool = True
    error: str | None = None
    model: str | None = None
    input_state_keys: list[str] = field(default_factory=list)
    output_state_keys: list[str] = field(default_factory=list)
    input_excerpt: str | None = None
    output_excerpt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, *, success: bool, error: str | None = None,
               output_state_keys: list[str] | None = None,
               output_excerpt: Any = None) -> None:
        ended = datetime.now(timezone.utc)
        started = datetime.fromisoformat(self.started_at)
        self.ended_at = ended.isoformat()
        self.duration_ms = (ended - started).total_seconds() * 1000
        self.success = success
        self.error = error
        if output_state_keys is not None:
            self.output_state_keys = output_state_keys
        if output_excerpt is not None:
            self.output_excerpt = _excerpt(output_excerpt)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def make_excerpt(value: Any, limit: int = 2000) -> str:
        return _excerpt(value, limit)
