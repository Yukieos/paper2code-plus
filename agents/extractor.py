from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an expert scientific information extraction assistant. Your task is to extract structured UPS-IR information "
    "from the following academic paper text. Output MUST be a valid JSON object strictly following the schema below.\n\n"
    "Reasoning priorities:\n"
    "- Identify the main research question(s) and recursively list every sub-question or hypothesis until concrete actions.\n"
    "- For each question, describe the ordered method steps that address it, including inputs, outputs, hyperparameters, optimization choices, and dependencies.\n"
    "- Construct multiple fine-grained DAGs that cover every major subsection or pipeline in the paper (e.g., training, inference, evaluation, ablations). Each graph must cite its source span and explain what portion of the paper it covers.\n"
    "- When a detail is not stated, explicitly output the string \"unknown\" instead of inventing content.\n\n"
    "Graph coverage requirements:\n"
    "- Produce at least two distinct graphs whenever the paper discusses more than one process. One graph should cover the main training/solution flow; additional graphs should cover evaluation, inference, or ablations as needed.\n"
    "- Nodes must reference existing IDs (questions, methods, method_steps, datasets, etc.), and edges must describe the relationship type (e.g., derives, feeds, evaluates, supervises).\n"
    "- Graph descriptions must cite the section/paragraph they summarize.\n\n"
    "Referenced-field requirements (IDs must already exist; examples shown):\n"
    "1. Use ID prefixes consistently: tasks (t#), questions (q#), methods (m#), method_steps (ms#), datasets (d#), equations (e#), experiments reference m#/d#, graphs (g#), sections (s#), algorithms (a#), figures (f#), parameters (p#), losses (l#), training_pipeline (tp#), optimizations (o#).\n"
    "2. methods[*].uses_datasets must specify dataset IDs (e.g., d1).\n"
    "3. methods[*].uses_equations must specify equation IDs (e.g., e1).\n"
    "4. experiments[*].method and experiments[*].dataset must reference method/dataset IDs and list every reported metric with numeric value when available.\n"
    "5. relations[*].from / relations[*].to may only use IDs from tasks, questions, methods, method_steps, datasets, equations, sections, algorithms, figures, parameters, losses, training_pipeline, or optimizations.\n"
    "6. graphs[*].nodes and graphs[*].edges.from/to must reference those same IDs.\n"
    "7. method_steps[*].method_id must reference a method ID; method_steps[*].question_id must reference a question ID; method_steps[*].depends_on may reference question IDs or other method_step IDs.\n"
    "8. metrics[*].value should be numeric floats when the paper reports numbers; omit the value field if no number is reported.\n"
    "9. Every method_step must populate loss / optimizer / learning_rate / schedule fields; if unspecified write \"unknown\".\n\n"
    "Return JSON using this structure (braces are doubled here to escape templating). Fill the question hierarchy first, then methods + method_steps, then graphs and experiments, then documentary sections so nothing is skipped:\n"
    "{{\n"
    '  "meta": {{\n'
    '    "title": str,\n'
    '    "authors": [str],\n'
    '    "venue": str,\n'
    '    "year": int\n'
    '  }},\n'
    '  "questions": [\n'
    '    {{"id": "q1", "text": str, "role": "primary|sub", "goal": str, "parent": "q1", "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "tasks": [\n'
    '    {{"id": "t1", "name": str, "description": str, "source_reference": str}}, ...\n'
    '  ],\n'
    '  "methods": [\n'
    '    {{"id": "m1", "name": str, "description": str, "architecture": str, "uses_datasets": [str], "uses_equations": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "method_steps": [\n'
    '    {{"id": "ms1", "method_id": "m1", "question_id": "q1", "stage": str, "description": str, "inputs": [str], "outputs": [str], "hyperparameters": [{{"name": str, "value": str, "description": str}}], "loss": str, "optimizer": str, "learning_rate": str, "schedule": str, "objective": str, "target_metric": str, "notes": str, "depends_on": ["ms1"], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "graphs": [\n'
    '    {{"id": "g1", "name": str, "description": str, "focus": str, "nodes": [str], "edges": [{{"from": str, "to": str, "type": str, "description": str}}], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "experiments": [\n'
    '    {{"method": "m1", "dataset": "d1", "setup": str, "metrics": [{{"name": str, "value": float, "unit": str}}], "source_reference": str}}, ...\n'
    '  ],\n'
    '  "relations": [\n'
    '    {{"from": str, "to": str, "type": "answers|requires|refines|supports|evaluates", "rationale": str, "source_reference": str}}, ...\n'
    '  ],\n'
    '  "datasets": [\n'
    '    {{"id": "d1", "name": str, "split": "train", "description": str, "source_reference": str}}, ...\n'
    '  ],\n'
    '  "equations": [\n'
    '    {{"id": "e1", "latex": str, "units": str, "description": str, "source_reference": str}}, ...\n'
    '  ],\n'
    '  "training_pipeline": [\n'
    '    {{"id": "tp1", "name": str, "description": str, "inputs": [str], "outputs": [str], "hyperparameters": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "optimizations": [\n'
    '    {{"id": "o1", "name": str, "objective": str, "optimizer": str, "schedule": str, "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "parameters": [\n'
    '    {{"id": "p1", "name": str, "symbol": str, "value": str, "description": str, "constraints": str, "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "losses": [\n'
    '    {{"id": "l1", "name": str, "formula": str, "description": str, "related_methods": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "sections": [\n'
    '    {{"id": "s1", "title": str, "level": int, "summary": str, "key_points": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "figures": [\n'
    '    {{"id": "f1", "name": str, "description": str, "related_components": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ],\n'
    '  "algorithms": [\n'
    '    {{"id": "a1", "name": str, "inputs": [str], "outputs": [str], "steps": [str], "source_reference": str, "source_text": str}}, ...\n'
    '  ]\n'
    '}}\n\n'
    "Rules:\n"
    "- IDs must remain stable and sequential. Re-use IDs when expanding existing entities; do not relabel previously assigned items.\n"
    "- Every method description must explain architecture pieces, supervision signals, and distinctive tricks (losses, schedules, initialization).\n"
    "- Every method_step must capture the stage goal, inputs/outputs, hyperparameters, and dependencies so that the chain forms a DAG from questions to answers.\n"
    "- Cover every substantive section, figure, algorithm, parameter, loss, dataset, and training/optimization detail described in the paper.\n"
    "- Always provide source_reference (e.g., 'Section 3.2', 'Figure 4', 'Table 2') plus a short source_text quote for traceability.\n"
    "- Prefer exact wording from the paper for titles, variable names, equations, and metrics; do not invent facts.\n"
    "- When the paper omits a field, emit an empty array or omit optional attributes rather than guessing.\n"
    "- Only output the final JSON. No explanation or commentary outside the JSON object.\n\n"
    "Thesis content:\n\"\"\"\n{text}\n\"\"\""
)

VERIFIER_FEEDBACK_RETRY_TEMPLATE = (
    "Your previous UPS-IR extraction from this paper was rejected by downstream schema/integrity "
    "validation with this specific error:\n{verifier_error}\n\n"
    "Re-extract the full UPS-IR from scratch, making sure this specific problem does not recur "
    "(e.g. if a field name or reference was wrong, use the correct one this time). Follow the same "
    "schema and instructions as before.\n\n"
    "Thesis content:\n\"\"\"\n{text}\n\"\"\""
)

RETRY_PROMPT_TEMPLATE = (
    "The initial UPS-IR extraction missed or left empty the following sections: {missing_fields}. "
    "Quality concerns to address:\n"
    "{quality_notes}\n\n"
    "Existing partial content for those sections is shown below; extend or refine it without duplicating IDs:\n"
    "{existing_json}\n\n"
    "Re-read the thesis text and output a JSON object containing ONLY the missing top-level keys listed above. "
    "Honor the quality notes, keep IDs consistent (continue numbering if you must add new entities), and cite precise source references. "
    "Thesis content:\n\"\"\"\n{text}\n\"\"\""
)



@dataclass
class ExtractorConfig:
    """Configuration for the ExtractorAgent."""

    output_path: Path = Path("output/extracted_info.json")
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    min_questions: int = 1
    min_method_steps: int = 1
    min_relations: int = 3
    min_graphs: int = 2
    min_graph_nodes: int = 4
    min_graph_edges: int = 3
    min_question_graph_coverage: float = 0.8
    min_method_steps_per_question: int = 1
    max_quality_retries: int = 2


class ExtractorAgent:
    """
    Call an LLM via LangChain to convert raw text into structured UPS-IR fields.
    """

    def __init__(self, config: Optional[ExtractorConfig] = None, llm: Optional[ChatOpenAI] = None):
        self.config = config or ExtractorConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.llm = llm or ChatOpenAI(model=self.config.model_name, temperature=self.config.temperature)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self._chain = self.prompt | self.llm | StrOutputParser()
        self.retry_prompt = ChatPromptTemplate.from_template(RETRY_PROMPT_TEMPLATE)
        self._retry_chain = self.retry_prompt | self.llm | StrOutputParser()
        self._verifier_feedback_prompt = ChatPromptTemplate.from_template(VERIFIER_FEEDBACK_RETRY_TEMPLATE)
        self._verifier_feedback_chain = self._verifier_feedback_prompt | self.llm | StrOutputParser()

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "text" not in state:
            raise ValueError("ExtractorAgent requires 'text' in the incoming state.")

        text_content = state["text"]

        # ReAct backtrack: the orchestrator sends us back here (instead of
        # just halting) when VerifierAgent rejected the UPS-IR we produced
        # last time. Re-extract with that specific error as context rather
        # than blindly repeating the same (temperature=0, so near-identical)
        # extraction that already failed.
        verifier_feedback = state.pop("verifier_feedback", None)
        if verifier_feedback:
            logger.info("ExtractorAgent re-extracting with verifier feedback: %s", verifier_feedback)
            raw_response = self._verifier_feedback_chain.invoke({
                "verifier_error": verifier_feedback,
                "text": text_content,
            })
        else:
            raw_response = self._chain.invoke({"text": text_content})
        logger.debug("ExtractorAgent raw response: %s", raw_response)

        info = self._try_parse_json(raw_response)
        missing_fields, quality_notes = self._quality_gate(info)
        if missing_fields:
            info = self._run_quality_retries(info, text_content, missing_fields, quality_notes)

        state["info"] = info

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        state.setdefault("artifacts", {})
        state["artifacts"]["extracted_json"] = str(self.config.output_path)

        return state

    @staticmethod
    def _try_parse_json(content: str) -> Dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = ExtractorAgent._strip_fences(cleaned)

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object at the top level.")
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"ExtractorAgent received non-JSON output: {exc}\nContent:\n{cleaned}") from exc

    @staticmethod
    def _strip_fences(block: str) -> str:
        lines = block.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _quality_gate(self, info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        missing: List[str] = []
        notes: List[str] = []

        def mark(field: str, note: str) -> None:
            if field not in missing:
                missing.append(field)
            notes.append(f"- {note}")

        questions = self._ensure_list(info.get("questions"))
        question_ids = {str(q.get("id")) for q in questions if isinstance(q, dict) and q.get("id")}
        if len(question_ids) < self.config.min_questions:
            mark("questions", f"Need at least {self.config.min_questions} questions but found {len(question_ids)}.")

        method_steps = self._ensure_list(info.get("method_steps"))
        if len(method_steps) < self.config.min_method_steps:
            mark("method_steps", f"Need at least {self.config.min_method_steps} method steps but found {len(method_steps)}.")

        graphs = self._ensure_list(info.get("graphs"))
        if len(graphs) < self.config.min_graphs:
            mark("graphs", f"Need at least {self.config.min_graphs} graphs but found {len(graphs)}.")

        relations = self._ensure_list(info.get("relations"))
        if len(relations) < self.config.min_relations:
            mark("relations", f"Need at least {self.config.min_relations} relations but found {len(relations)}.")

        if graphs:
            for graph in graphs:
                graph_id = graph.get("id", "graph")
                node_count = len(self._ensure_list(graph.get("nodes")))
                if node_count < self.config.min_graph_nodes:
                    mark("graphs", f"Graph {graph_id} has {node_count} nodes; need ≥ {self.config.min_graph_nodes}.")
                edge_count = len(self._ensure_list(graph.get("edges")))
                if edge_count < self.config.min_graph_edges:
                    mark("graphs", f"Graph {graph_id} has {edge_count} edges; need ≥ {self.config.min_graph_edges}.")

        step_question_refs = Counter(
            str(step.get("question_id"))
            for step in method_steps
            if isinstance(step, dict) and step.get("question_id")
        )
        underlinked_questions = [
            qid
            for qid in question_ids
            if step_question_refs.get(qid, 0) < self.config.min_method_steps_per_question
        ]
        if underlinked_questions:
            mark(
                "method_steps",
                f"Questions {', '.join(sorted(underlinked_questions))} have fewer than "
                f"{self.config.min_method_steps_per_question} supporting method steps.",
            )

        graph_node_refs = self._collect_graph_node_refs(graphs)
        relation_refs = self._collect_relation_refs(relations)
        referenced_questions = question_ids & (set(step_question_refs.keys()) | graph_node_refs | relation_refs)
        coverage = (len(referenced_questions) / len(question_ids)) if question_ids else 1.0
        if coverage < self.config.min_question_graph_coverage:
            mark(
                "graphs",
                f"Only {coverage:.0%} of questions appear in graphs/method_steps/relations (need "
                f"{self.config.min_question_graph_coverage:.0%}+).",
            )

        required_step_fields = ("loss", "optimizer", "learning_rate", "schedule", "objective", "target_metric", "notes")
        incomplete_steps = []
        for step in method_steps:
            if not isinstance(step, dict):
                continue
            missing_fields = [field for field in required_step_fields if not step.get(field)]
            if missing_fields:
                incomplete_steps.append(f"{step.get('id', 'ms?')} missing {', '.join(missing_fields)}")
        if incomplete_steps:
            mark("method_steps", "Method steps missing required optimization fields: " + "; ".join(incomplete_steps))

        if not missing:
            notes = []

        return missing, notes

    def _run_quality_retries(
        self,
        info: Dict[str, Any],
        text: str,
        missing_fields: List[str],
        quality_notes: List[str],
    ) -> Dict[str, Any]:
        pending = list(dict.fromkeys(missing_fields))
        retries = 0
        merged = info
        notes = list(quality_notes)

        while pending and retries < self.config.max_quality_retries:
            supplemental = self._invoke_retry_chain(text, merged, pending, notes)
            if not supplemental:
                break
            merged = self._merge_info(merged, supplemental)
            pending, notes = self._quality_gate(merged)
            retries += 1

        if pending:
            logger.warning("ExtractorAgent quality gate unresolved for sections: %s", pending)

        return merged

    def _invoke_retry_chain(
        self,
        text: str,
        current_info: Dict[str, Any],
        missing_fields: List[str],
        quality_notes: List[str],
    ) -> Dict[str, Any]:
        context = {field: current_info.get(field, []) for field in missing_fields}
        serialized_context = json.dumps(context, ensure_ascii=False, indent=2)
        formatted_notes = self._format_quality_notes(quality_notes)
        prompt_input = {
            "missing_fields": ", ".join(missing_fields),
            "quality_notes": formatted_notes,
            "existing_json": serialized_context,
            "text": text,
        }

        try:
            raw_retry = self._retry_chain.invoke(prompt_input)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("ExtractorAgent retry invocation failed: %s", exc)
            return {}

        logger.debug("ExtractorAgent retry response: %s", raw_retry)
        try:
            return self._try_parse_json(raw_retry)
        except ValueError as exc:
            logger.warning("ExtractorAgent retry parsing failed: %s", exc)
            return {}

    def _merge_info(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        if not updates:
            return base

        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, list):
                merged[key] = self._merge_lists(merged.get(key), value)
            elif isinstance(value, dict):
                existing = merged.get(key)
                if isinstance(existing, dict):
                    combined = dict(existing)
                    combined.update(value)
                    merged[key] = combined
                else:
                    merged[key] = value
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _merge_lists(existing: Any, additions: Any) -> List[Any]:
        base_list: List[Any] = list(existing) if isinstance(existing, list) else []
        if not isinstance(additions, list):
            return base_list

        id_based = ExtractorAgent._list_has_ids(base_list) or ExtractorAgent._list_has_ids(additions)
        if id_based:
            result = list(base_list)
            id_index: Dict[str, Dict[str, Any]] = {}
            for item in result:
                if isinstance(item, dict) and item.get("id") is not None:
                    identifier = str(item["id"])
                    id_index[identifier] = item

            for item in additions:
                if not isinstance(item, dict) or item.get("id") is None:
                    if item not in result:
                        result.append(item)
                    continue

                identifier = str(item["id"])
                if identifier in id_index:
                    id_index[identifier].update(item)
                else:
                    new_item = dict(item)
                    id_index[identifier] = new_item
                    result.append(new_item)
            return result

        result = list(base_list)
        for item in additions:
            if item not in result:
                result.append(item)
        return result

    @staticmethod
    def _list_has_ids(items: List[Any]) -> bool:
        for item in items:
            if isinstance(item, dict) and item.get("id") is not None:
                return True
        return False

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _collect_graph_node_refs(graphs: List[Any]) -> Set[str]:
        refs: Set[str] = set()
        for graph in graphs:
            nodes = graph.get("nodes") if isinstance(graph, dict) else None
            if isinstance(nodes, list):
                for node in nodes:
                    if node is not None:
                        refs.add(str(node))
        return refs

    @staticmethod
    def _collect_relation_refs(relations: List[Any]) -> Set[str]:
        refs: Set[str] = set()
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            for field in ("from", "to"):
                value = relation.get(field)
                if value is not None:
                    refs.add(str(value))
        return refs

    @staticmethod
    def _format_quality_notes(notes: List[str]) -> str:
        if not notes:
            return "- fill in the missing sections listed above."
        return "\n".join(notes)
