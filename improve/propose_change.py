"""Given a Diagnosis, draft a revised prompt for the implicated agent.

The only thing this is allowed to change is prompt wording — see
improve/prompt_registry.py for the enforced allowlist and the
{placeholder}-preservation check that rejects anything else.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from improve.diagnose import Diagnosis
from improve.prompt_registry import PromptRef, get_prompt_text, placeholders, refs_for_agent

logger = logging.getLogger(__name__)

PROPOSE_PROMPT_TEMPLATE = """You are fixing a prompt in the Paper2Code multi-agent pipeline.

Root-cause diagnosis: {root_cause}

Current prompt (`{constant_name}`) — you MUST preserve every `{{placeholder}}`
variable exactly as written (they're substituted by code at runtime; renaming,
adding, or removing one will break the pipeline):
---
{current_prompt}
---

Task: rewrite this prompt to address the diagnosed root cause, changing as
little else as possible. Keep the same {{placeholder}} variables, the same
general structure, and the same output format the prompt asks for — only
sharpen/add/correct the specific instruction the diagnosis points at.

Respond with ONLY a JSON object of this exact shape:
{{
  "new_prompt": "<the full revised prompt text>",
  "explanation": "<1-2 sentences on what changed and why>"
}}
"""


@dataclass
class ProposedChange:
    ref: PromptRef
    diagnosis: Diagnosis
    old_prompt: str
    new_prompt: str
    explanation: str

    def is_valid(self) -> bool:
        return placeholders(self.old_prompt) == placeholders(self.new_prompt) and \
            self.new_prompt.strip() and self.new_prompt.strip() != self.old_prompt.strip()


def _default_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0.2)
    return llm.invoke(prompt).content


def _extract_json(text: str) -> str:
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in propose_change output: {text[:200]!r}")
    return text[start:end + 1]


def propose_changes(
    diagnosis: Diagnosis,
    llm_call: Callable[[str], str] | None = None,
    model: str = "gpt-4o-mini",
) -> list[ProposedChange]:
    if not diagnosis.responsible_agent:
        logger.info("Diagnosis for %s named no responsible agent; nothing to propose.", diagnosis.mode.value)
        return []

    refs = refs_for_agent(diagnosis.responsible_agent)
    if not refs:
        logger.warning("No registered prompts for agent '%s'; nothing to propose.", diagnosis.responsible_agent)
        return []

    call = llm_call or (lambda prompt: _default_llm_call(prompt, model=model))
    proposals: list[ProposedChange] = []

    for ref in refs:
        old_prompt = get_prompt_text(ref)
        prompt = PROPOSE_PROMPT_TEMPLATE.format(
            root_cause=diagnosis.root_cause,
            constant_name=ref.constant_name,
            current_prompt=old_prompt,
        )
        try:
            # strict=False: the new_prompt value is a full multi-line prompt,
            # and the model routinely emits literal newlines/tabs inside that
            # JSON string rather than escaping them — which strict parsing
            # rejects as an "invalid control character".
            data = json.loads(_extract_json(call(prompt)), strict=False)
            new_prompt = str(data["new_prompt"])
            explanation = str(data.get("explanation", ""))
        except Exception as exc:  # pragma: no cover - network/model dependent
            logger.warning("propose_change failed for %s (%s); skipping.", ref.constant_name, exc)
            continue

        change = ProposedChange(ref, diagnosis, old_prompt, new_prompt, explanation)
        if change.is_valid():
            proposals.append(change)
        else:
            logger.warning("Proposed change for %s failed validation (placeholder mismatch or no-op); skipping.",
                            ref.constant_name)

    return proposals
