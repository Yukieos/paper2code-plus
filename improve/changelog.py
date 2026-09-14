"""Append-only HARNESS_CHANGELOG: an auditable record of every behavior change
the improvement loop proposes.

Every proposed harness change (opened as a PR by run_cycle) appends one entry
here. Entries are NEVER edited or deleted — a later change that reverses an
earlier one appends a fresh entry naming the superseded commit in `Supersedes`.
That makes the harness's behavior evolution auditable, attributable, and
reversible, exactly as the design requires.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = REPO_ROOT / "HARNESS_CHANGELOG.md"

_HEADER = (
    "# HARNESS_CHANGELOG\n\n"
    "Append-only log of every behavior-controlling change proposed by the "
    "continuous-improvement loop (`improve/run_cycle.py`). Newest entries are "
    "appended at the bottom. Do not edit or remove past entries — reverse a "
    "change by appending a new entry that names the superseded commit.\n"
)


@dataclass
class ChangelogEntry:
    mode: str  # the failure mode / metric that triggered this
    triggering_runs: list[str]
    reproduced: str  # e.g. "2/2 (threshold 2)"
    observed_behavior: str  # diagnosis root cause
    smoking_gun: str  # concrete evidence
    hypotheses: list[str]  # root-cause hypotheses considered
    changes: list[str]  # e.g. ["EXECUTION_PROMPT_ML in codegen_pipeline.py — add ..."]
    files_changed: list[str]
    targeted_result: str
    heldout_result: str
    regression_result: str
    cost_delta: str = "not tracked"
    risks: str = ""
    commit: str = "pending"
    supersedes: str = "none"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def render(self) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"  - {i}" for i in items) if items else "  - (none)"

        return (
            f"\n## {self.timestamp} — {self.mode} — {self.commit}\n\n"
            f"- **Triggering runs:** {', '.join(self.triggering_runs) or '(none)'}\n"
            f"- **Failure mode / metric:** {self.mode}\n"
            f"- **Reproduced:** {self.reproduced}\n"
            f"- **Observed behavior:** {self.observed_behavior}\n"
            f"- **Smoking gun / evidence:** {self.smoking_gun or '(none captured)'}\n"
            f"- **Hypotheses considered:**\n{bullets(self.hypotheses)}\n"
            f"- **Change(s):**\n{bullets(self.changes)}\n"
            f"- **Files changed:**\n{bullets(self.files_changed)}\n"
            f"- **Targeted verification:** {self.targeted_result}\n"
            f"- **Held-out gate:** {self.heldout_result}\n"
            f"- **Regression gate:** {self.regression_result}\n"
            f"- **Cost / token delta:** {self.cost_delta}\n"
            f"- **Risks:** {self.risks or '(none noted)'}\n"
            f"- **Commit:** {self.commit}\n"
            f"- **Supersedes:** {self.supersedes}\n"
        )


def append_entry(entry: ChangelogEntry, path: Path | None = None) -> Path:
    """Append one entry to the changelog, creating it with a header if needed.
    Append-only: this never rewrites existing content."""
    path = path or CHANGELOG_PATH
    if not path.exists():
        path.write_text(_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as fp:
        fp.write(entry.render())
    logger.info("Appended changelog entry for %s to %s", entry.mode, path)
    return path
