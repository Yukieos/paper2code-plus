"""Smoke-run a generated repository: actually execute its entry point briefly
and judge whether the reproduction *runs*, rather than only whether it looks
runnable (static analysis).

Deliberately smoke-level, not full training: run the entry point under a hard
wall-clock timeout (and a best-effort memory cap), then decide from three
signals — did it crash, did the loss diverge (NaN/inf), and did it make any
measurable progress. This produces the runtime signal the within-run
reproduction-repair controller (see repro_repair.py) reacts to: a crash gives a
traceback and the offending in-repo file to regenerate; a hang or divergence is
a softer failure.

No torch dependency here — this module only launches a subprocess and parses
its output/logs, so it (and its tests) run without the generated repo's own
heavy deps installed.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# "loss: 0.1234", "loss = 1.2e-3", "train_loss 0.5", "val_loss=0.1" — the
# optional \w* prefix catches train_/val_/test_loss, which generated code
# logs far more often than a bare "loss". Case-insensitive.
_LOSS_RE = re.compile(r"\b\w*loss\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf)", re.IGNORECASE)
# A traceback frame: '  File "…/repo/model.py", line 12, in forward'
_FRAME_RE = re.compile(r'File "([^"]+)", line \d+')


@dataclass
class SmokeResult:
    ok: bool
    reason: str
    returncode: int | None = None
    timed_out: bool = False
    crashed: bool = False
    diverged: bool = False
    progressed: bool = False
    offending_file: str | None = None  # repo-relative .py from the deepest in-repo frame
    traceback: str = ""
    loss_values: list[float] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""

    def summary(self) -> str:
        bits = [f"ok={self.ok}", self.reason]
        if self.loss_values:
            bits.append(f"loss[{self.loss_values[0]:.4g}->{self.loss_values[-1]:.4g}]")
        if self.offending_file:
            bits.append(f"offending={self.offending_file}")
        return " | ".join(bits)


def parse_losses(text: str) -> list[float]:
    """Every numeric loss value mentioned, in order. NaN/inf are captured too
    (as float('nan')/inf) so divergence can be detected."""
    values: list[float] = []
    for m in _LOSS_RE.finditer(text):
        raw = m.group(1).lower()
        try:
            values.append(float(raw))  # float("nan")/float("inf") work
        except ValueError:
            continue
    return values


def offending_repo_file(traceback_text: str, repo_dir: Path) -> str | None:
    """The deepest traceback frame whose file lives inside repo_dir, as a
    repo-relative path — i.e. which generated file to regenerate. Frames in
    stdlib / site-packages are ignored."""
    repo_dir = repo_dir.resolve()
    deepest = None
    for m in _FRAME_RE.finditer(traceback_text):
        try:
            path = Path(m.group(1)).resolve()
            rel = path.relative_to(repo_dir)
        except (ValueError, OSError):
            continue
        deepest = str(rel)  # later frames are deeper in the call stack
    return deepest


def _collect_log_text(repo_dir: Path) -> str:
    """Generated code often routes loss/errors to a log FILE
    (logging.basicConfig(filename='app.log')) rather than stdout, so scan any
    *.log the run wrote in the repo dir too."""
    chunks = []
    for log in sorted(repo_dir.glob("*.log")):
        try:
            chunks.append(log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _limit_memory(mem_mb: int):
    """Best-effort per-process address-space cap (POSIX). Returns a preexec_fn
    or None. Silently degrades where RLIMIT_AS isn't honored (e.g. macOS)."""
    try:
        import resource
    except ImportError:
        return None

    def _set():
        try:
            soft = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass  # not honored on this platform; timeout remains the guard

    return _set


def _grade(returncode, timed_out, stdout, stderr, log_text, repo_dir) -> SmokeResult:
    combined = "\n".join([stdout, stderr, log_text])
    losses = [v for v in parse_losses(combined)]
    finite = [v for v in losses if math.isfinite(v)]
    diverged = any(not math.isfinite(v) for v in losses)
    progressed = len(finite) >= 2 and finite[-1] < finite[0]

    tail = lambda s: s[-2000:] if s else ""
    res = SmokeResult(
        ok=False, reason="", returncode=returncode, timed_out=timed_out,
        diverged=diverged, progressed=progressed, loss_values=losses,
        stdout_tail=tail(stdout), stderr_tail=tail(stderr),
    )

    if diverged:
        res.reason = "loss diverged (NaN/inf)"
        return res

    if timed_out:
        # A timeout is only OK if it was clearly making progress when killed.
        res.ok = progressed
        res.reason = "training in progress at timeout" if progressed else "timed out with no measurable progress"
        return res

    if returncode not in (0, None):
        res.crashed = True
        res.traceback = tail(stderr) or tail(log_text)
        res.offending_file = offending_repo_file(stderr + "\n" + log_text, repo_dir)
        res.reason = f"runtime crash (exit {returncode})"
        return res

    # Exited cleanly and didn't diverge. Any finite loss is a bonus signal; its
    # absence isn't a hard failure at smoke level (some entry points don't print
    # one), but completing without crashing is the bar.
    res.ok = True
    res.reason = "ran to completion" + (", loss decreasing" if progressed else
                                        (", loss seen" if finite else ", no loss detected"))
    return res


def run_smoke(
    repo_dir: Path,
    entry: str = "main.py",
    args: list[str] | None = None,
    timeout: int = 120,
    mem_mb: int = 2048,
    env: dict | None = None,
) -> SmokeResult:
    """Execute repo_dir/<entry> as a subprocess (cwd=repo_dir) under a wall-clock
    timeout and best-effort memory cap, then grade the run."""
    repo_dir = Path(repo_dir)
    entry_path = repo_dir / entry
    if not entry_path.exists():
        return SmokeResult(ok=False, reason=f"entry point {entry} not found in generated repo")

    cmd = [sys.executable, entry] + (args if args is not None else ["--config", "config.json", "--seed", "0"])
    try:
        proc = subprocess.run(
            cmd, cwd=repo_dir, env=env, capture_output=True, text=True,
            timeout=timeout, preexec_fn=_limit_memory(mem_mb),
        )
        returncode, timed_out, stdout, stderr = proc.returncode, False, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, timed_out = None, True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")

    return _grade(returncode, timed_out, stdout, stderr, _collect_log_text(repo_dir), repo_dir)
