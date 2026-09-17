"""Within-run reproduction repair: when a freshly generated repo fails its
smoke run, re-do the REPRODUCTION (re-invoke / reorder / regenerate agents) —
never edit the pipeline's own source. Our agents are fixed tools; a bad run
means the reproduction steps for THIS paper went wrong, so we redo them.

This is the run-internal counterpart to the cross-run, human-gated harness
loop in improve/ (which is the only thing allowed to change our own prompts).

Escalation ladder (cheapest first), each bounded:
  1. diagnose_and_patch  — diagnose the WHOLE repo at once (all files + why the
     agents wrote them + the paper as ground truth) and apply a coherent
     MULTI-FILE patch. Not single-file: a caller/callee signature mismatch has
     to be fixed on both sides together, or the crash just moves.
  2. re-plan + re-split files      — the plan/decomposition itself is wrong
  3. re-classify the paper type    — whole approach/file-set looks wrong
Exhausting the ladder is itself a signal — an unrepairable reproduction is
exactly the kind of recurring failure the cross-run improver should later mine.

The controller is pure orchestration over injected callables, so it's tested
without running torch or calling an LLM (see tests/test_repro_repair.py); the
real actions are wired in codegen_pipeline via PipelineRepairActions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from smoke_run import SmokeResult

logger = logging.getLogger(__name__)


class RepairActions(Protocol):
    """The reproduction-redo actions the controller can take. Each mutates the
    generated repo in place; none touches the pipeline's own code."""

    def smoke(self) -> SmokeResult: ...
    def diagnose_and_patch(self, result: SmokeResult) -> bool: ...
    def replan_and_regenerate(self) -> bool: ...
    def reclassify_and_regenerate(self) -> bool: ...


@dataclass
class RepairStep:
    action: str
    detail: str
    result: SmokeResult


@dataclass
class RepairOutcome:
    repaired: bool
    final: SmokeResult
    steps: list[RepairStep] = field(default_factory=list)

    @property
    def actions_taken(self) -> list[str]:
        return [s.action for s in self.steps if s.action != "initial"]

    def summary(self) -> str:
        chain = " -> ".join(f"{s.action}({'ok' if s.result.ok else 'fail'})" for s in self.steps)
        return f"{'REPAIRED' if self.repaired else 'UNREPAIRED'} via [{chain}]"


def repair_reproduction(
    actions: RepairActions,
    patch_attempts: int = 2,
    replan_attempts: int = 1,
    reclassify_attempts: int = 1,
) -> RepairOutcome:
    """Smoke-run; on failure walk the escalation ladder until it passes or the
    budgets are exhausted."""
    result = actions.smoke()
    steps = [RepairStep("initial", result.reason, result)]
    if result.ok:
        logger.info("Reproduction smoke passed on first run: %s", result.summary())
        return RepairOutcome(True, result, steps)

    patch_budget, replan_budget, reclassify_budget = patch_attempts, replan_attempts, reclassify_attempts

    while not result.ok:
        action, detail = _choose_action(patch_budget, replan_budget, reclassify_budget)
        if action is None:
            logger.info("Repair ladder exhausted; reproduction still failing: %s", result.reason)
            break

        logger.info("Reproduction failed (%s); repair action: %s", result.reason, action)
        if action == "diagnose_patch":
            patch_budget -= 1
            applied = actions.diagnose_and_patch(result)
        elif action == "replan":
            replan_budget -= 1
            applied = actions.replan_and_regenerate()
        else:  # reclassify
            reclassify_budget -= 1
            applied = actions.reclassify_and_regenerate()

        if not applied:
            # The action couldn't apply (diagnosis escalated / regeneration
            # failed). Record it and zero that tier's budget so the ladder
            # escalates on the next pass instead of retrying a dead end.
            steps.append(RepairStep(action, f"{detail} [no change applied]", result))
            if action == "diagnose_patch":
                patch_budget = 0
            elif action == "replan":
                replan_budget = 0
            else:
                reclassify_budget = 0
            continue

        result = actions.smoke()
        steps.append(RepairStep(action, detail, result))

    return RepairOutcome(result.ok, result, steps)


def _choose_action(patch_budget: int, replan_budget: int, reclassify_budget: int):
    """Cheapest still-budgeted tier: whole-repo diagnose+patch, then re-plan,
    then re-classify. The diagnosis handles any failure shape (crash, hang,
    divergence, cross-file mismatch), so there's no per-failure gating here."""
    if patch_budget > 0:
        return "diagnose_patch", "diagnose whole repo and apply a coherent multi-file patch"
    if replan_budget > 0:
        return "replan", "re-plan and re-split files"
    if reclassify_budget > 0:
        return "reclassify", "re-classify paper type and regenerate"
    return None, ""
