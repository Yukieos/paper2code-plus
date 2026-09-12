"""S3-backed storage for agent execution traces.

Tracing is observability, not core functionality: every method here is
best-effort and swallows its own errors (missing credentials, no network,
bucket typo, ...) after logging a warning, so a tracing problem can never
take down an actual pipeline run. It's also off by default — set
PAPER2CODE_TRACE_ENABLED=1 to turn it on, so plain CLI/local usage never
needs AWS credentials at all.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

from eval.trace_schema import TraceRecord

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return os.environ.get("PAPER2CODE_TRACE_ENABLED", "").lower() in ("1", "true", "yes")


class TraceStore:
    """Writes/reads TraceRecords under s3://<bucket>/<prefix>/<run_id>/...

    Falls back to a local JSON-lines-per-file directory (`runs/_traces_local/`)
    when boto3 isn't installed, credentials aren't configured, or an S3 call
    fails — so the framework degrades gracefully instead of hard-failing.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str = "traces",
        local_fallback_dir: Path | None = None,
    ) -> None:
        self.bucket = bucket or os.environ.get("PAPER2CODE_TRACE_BUCKET", "paper2code-agent-trace")
        self.prefix = prefix.strip("/")
        self.local_fallback_dir = local_fallback_dir or Path("runs") / "_traces_local"
        self._client = None
        self._s3_available = self._init_client()

    def _init_client(self) -> bool:
        try:
            import boto3  # noqa: local import so boto3 stays optional at import time

            self._client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("S3 client unavailable (%s); traces will be written locally to %s", exc, self.local_fallback_dir)
            return False

    def _key(self, record: TraceRecord) -> str:
        return f"{self.prefix}/{record.run_id}/{record.stage}__{record.agent_name}__{record.trace_id}.json"

    def put_trace(self, record: TraceRecord) -> None:
        if not tracing_enabled():
            return
        body = json.dumps(record.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        if self._s3_available:
            try:
                self._client.put_object(Bucket=self.bucket, Key=self._key(record), Body=body,
                                         ContentType="application/json")
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to upload trace %s to s3://%s/%s (%s); falling back to local disk.",
                               record.trace_id, self.bucket, self._key(record), exc)
        self._write_local(record, body)

    def _write_local(self, record: TraceRecord, body: bytes) -> None:
        try:
            local_path = self.local_fallback_dir / record.run_id
            local_path.mkdir(parents=True, exist_ok=True)
            (local_path / f"{record.stage}__{record.agent_name}__{record.trace_id}.json").write_bytes(body)
        except Exception:  # pragma: no cover - best-effort only
            logger.exception("Failed to write trace locally either; dropping trace %s.", record.trace_id)

    def list_run_ids(self) -> list[str]:
        run_ids: set[str] = set()
        if self._s3_available:
            try:
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/", Delimiter="/"):
                    for common_prefix in page.get("CommonPrefixes", []):
                        run_id = common_prefix["Prefix"][len(self.prefix) + 1:].rstrip("/")
                        if run_id:
                            run_ids.add(run_id)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to list runs from S3 (%s); falling back to local disk.", exc)
        if self.local_fallback_dir.exists():
            run_ids.update(p.name for p in self.local_fallback_dir.iterdir() if p.is_dir())
        return sorted(run_ids)

    def get_traces(self, run_id: str) -> list[TraceRecord]:
        records: list[TraceRecord] = []
        seen_trace_ids: set[str] = set()

        if self._s3_available:
            try:
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/{run_id}/"):
                    for obj in page.get("Contents", []):
                        body = self._client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                        record = TraceRecord.from_dict(json.loads(body))
                        records.append(record)
                        seen_trace_ids.add(record.trace_id)
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to fetch traces for run %s from S3 (%s).", run_id, exc)

        local_dir = self.local_fallback_dir / run_id
        if local_dir.exists():
            for path in local_dir.glob("*.json"):
                record = TraceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if record.trace_id not in seen_trace_ids:
                    records.append(record)

        records.sort(key=lambda r: r.started_at)
        return records

    def iter_all_traces(self) -> Iterator[tuple[str, list[TraceRecord]]]:
        for run_id in self.list_run_ids():
            yield run_id, self.get_traces(run_id)
