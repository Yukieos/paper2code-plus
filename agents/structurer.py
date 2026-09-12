from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Optional, Iterable, Set


logger = logging.getLogger(__name__)


@dataclass
class StructurerConfig:
    """Configuration for StructurerAgent."""

    output_path: Path = Path("UPS-IR.json")
    version: str = "1.0"


class StructurerAgent:
    """
    Transform the extractor output into a normalized UPS-IR JSON representation.
    """

    def __init__(self, config: Optional[StructurerConfig] = None):
        self.config = config or StructurerConfig()
        self.config.output_path = self.config.output_path.resolve()
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None or "info" not in state:
            raise ValueError("StructurerAgent requires 'info' in the incoming state.")

        info = state["info"]
        ups_ir = self._build_structure(info)

        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(ups_ir, f, ensure_ascii=False, indent=2)

        logger.info("StructurerAgent wrote UPS-IR JSON to %s", self.config.output_path)

        state["ups_ir"] = ups_ir
        state.setdefault("artifacts", {})
        state["artifacts"]["ups_ir_json"] = str(self.config.output_path)

        return state

    def _build_structure(self, info: Dict[str, Any]) -> Dict[str, Any]:
        meta = info.get("meta") or {}
        current_timestamp = datetime.now(timezone.utc).isoformat()

        ups_ir = {
            "version": self.config.version,
            "generated_at": current_timestamp,
            "meta": {
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "venue": meta.get("venue", ""),
                "year": meta.get("year"),
            },
            "questions": self._normalize_questions(info.get("questions", [])),
            "tasks": self._normalize_entities(info.get("tasks", [])),
            "methods": self._normalize_methods(info.get("methods", [])),
            "method_steps": self._normalize_method_steps(info.get("method_steps", [])),
            "datasets": self._normalize_entities(info.get("datasets", [])),
            "equations": self._normalize_entities(info.get("equations", [])),
            "experiments": self._normalize_experiments(info.get("experiments", [])),
            "relations": self._normalize_relations(info.get("relations", [])),
            "graphs": self._normalize_graphs(info.get("graphs", [])),
            "sections": self._normalize_sections(info.get("sections", [])),
            "figures": self._normalize_figures(info.get("figures", [])),
            "algorithms": self._normalize_algorithms(info.get("algorithms", [])),
            "parameters": self._normalize_parameters(info.get("parameters", [])),
            "losses": self._normalize_losses(info.get("losses", [])),
            "training_pipeline": self._normalize_training_pipeline(info.get("training_pipeline", [])),
            "optimizations": self._normalize_optimizations(info.get("optimizations", [])),
        }

        self._ensure_question_definitions(ups_ir)
        self._ensure_method_definitions(ups_ir)
        self._augment_graphs(ups_ir)
        self._sanitize_graphs(ups_ir)

        return ups_ir

    def _normalize_entities(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            normalized.append(item)
        return normalized

    def _normalize_methods(self, items: Any) -> list[Dict[str, Any]]:
        methods = self._normalize_entities(items)
        for method in methods:
            for field in ("uses_datasets", "uses_equations"):
                values = method.get(field)
                if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
                    method[field] = [str(value) for value in values]
                elif values is None:
                    method.pop(field, None)
                else:
                    method[field] = [str(values)]
            self._coerce_fields_to_str(method, ("name", "description", "architecture", "source_reference", "source_text"))
        return methods

    def _normalize_questions(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            if "parent" in item and item["parent"] is not None:
                item["parent"] = str(item["parent"])
            self._coerce_fields_to_str(item, ("text", "role", "goal", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_method_steps(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            for field in ("method_id", "question_id", "stage"):
                if field in item and item[field] is not None:
                    item[field] = str(item[field])
            item["inputs"] = self._ensure_list_of_strings(item.get("inputs"))
            item["outputs"] = self._ensure_list_of_strings(item.get("outputs"))
            item["depends_on"] = self._ensure_list_of_strings(item.get("depends_on"))
            item["hyperparameters"] = self._normalize_hyperparameters(item.get("hyperparameters"))
            for field in ("loss", "optimizer", "learning_rate", "schedule", "objective", "target_metric", "notes"):
                value = item.get(field)
                if value is None or value == "":
                    item[field] = "unknown"
                else:
                    item[field] = str(value)
            self._coerce_fields_to_str(item, ("description", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_hyperparameters(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            self._coerce_fields_to_str(item, ("name", "value", "description"))
            normalized.append(item)
        return normalized

    def _normalize_experiments(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if "method" in item and item["method"] is not None:
                item["method"] = str(item["method"])
            if "dataset" in item and item["dataset"] is not None:
                item["dataset"] = str(item["dataset"])

            metrics = item.get("metrics")
            if isinstance(metrics, dict):
                item["metrics"] = [{"name": str(k), "value": metrics[k]} for k in metrics]
            elif isinstance(metrics, Iterable) and not isinstance(metrics, (str, bytes)):
                normalized_metrics: list[Dict[str, Any]] = []
                for metric in metrics:
                    if isinstance(metric, dict):
                        normalized_metrics.append(metric)
                item["metrics"] = normalized_metrics
            elif metrics is None:
                item.pop("metrics", None)
            else:
                item["metrics"] = [{"value": metrics}]

            normalized.append(item)
        return normalized

    def _normalize_relations(self, items: Any) -> list[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []

        normalized: list[Dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            for field in ("from", "to"):
                if field in item and item[field] is not None:
                    item[field] = str(item[field])
            normalized.append(item)
        return normalized

    def _normalize_graphs(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            item["nodes"] = self._ensure_list_of_strings(item.get("nodes"))
            edges: list[Dict[str, Any]] = []
            for edge in self._iter_dict_entries(item.get("edges", [])):
                edge_item = dict(edge)
                for field in ("from", "to", "type"):
                    if field in edge_item and edge_item[field] is not None:
                        edge_item[field] = str(edge_item[field])
                self._coerce_fields_to_str(edge_item, ("description",))
                edges.append(edge_item)
            item["edges"] = edges
            self._coerce_fields_to_str(item, ("name", "description", "focus", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_sections(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            if "level" in item and item["level"] is not None:
                try:
                    item["level"] = int(item["level"])
                except (TypeError, ValueError):
                    item["level"] = None
            item["key_points"] = self._ensure_list_of_strings(item.get("key_points"))
            self._coerce_fields_to_str(item, ("title", "summary", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_figures(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            item["related_components"] = self._ensure_list_of_strings(item.get("related_components"))
            self._coerce_fields_to_str(item, ("name", "description", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_algorithms(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            for field in ("inputs", "outputs", "steps"):
                item[field] = self._ensure_list_of_strings(item.get(field))
            self._coerce_fields_to_str(item, ("name", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_parameters(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            self._coerce_fields_to_str(
                item,
                ("name", "symbol", "value", "description", "constraints", "source_reference", "source_text"),
            )
            normalized.append(item)
        return normalized

    def _normalize_losses(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            item["related_methods"] = self._ensure_list_of_strings(item.get("related_methods"))
            self._coerce_fields_to_str(item, ("name", "formula", "description", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_training_pipeline(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            for field in ("inputs", "outputs", "hyperparameters"):
                item[field] = self._ensure_list_of_strings(item.get(field))
            self._coerce_fields_to_str(item, ("name", "description", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _normalize_optimizations(self, items: Any) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []
        for entry in self._iter_dict_entries(items):
            item = dict(entry)
            if "id" in item:
                item["id"] = str(item["id"])
            self._coerce_fields_to_str(item, ("name", "objective", "optimizer", "schedule", "source_reference", "source_text"))
            normalized.append(item)
        return normalized

    def _iter_dict_entries(self, items: Any) -> Iterable[Dict[str, Any]]:
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            return []
        return [entry for entry in items if isinstance(entry, dict)]

    def _ensure_list_of_strings(self, values: Any) -> list[str]:
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            return [str(value) for value in values]
        if values is None:
            return []
        return [str(values)]

    def _coerce_fields_to_str(self, item: Dict[str, Any], fields: Iterable[str]) -> None:
        for field in fields:
            if field in item and item[field] is not None:
                item[field] = str(item[field])

    def _augment_graphs(self, ups_ir: Dict[str, Any]) -> None:
        graphs = ups_ir.get("graphs")
        if not isinstance(graphs, list):
            graphs = []
            ups_ir["graphs"] = graphs

        labels = self._build_entity_labels(ups_ir)
        method_flow_graph = self._build_method_flow_graph(ups_ir, labels)
        if method_flow_graph:
            method_flow_graph["id"] = self._next_graph_id(graphs)
            graphs.append(method_flow_graph)

    def _sanitize_graphs(self, ups_ir: Dict[str, Any]) -> None:
        graphs = ups_ir.get("graphs")
        if not isinstance(graphs, list):
            return

        valid_targets = self._collect_valid_targets(ups_ir)
        sanitized: list[Dict[str, Any]] = []

        for graph in graphs:
            if not isinstance(graph, dict):
                continue

            nodes = []
            for node in graph.get("nodes", []):
                node_id = str(node)
                if node_id in valid_targets:
                    nodes.append(node_id)
                else:
                    logger.warning("StructurerAgent dropped unknown graph node '%s' from %s", node_id, graph.get("id"))

            edges: list[Dict[str, Any]] = []
            for edge in self._iter_dict_entries(graph.get("edges", [])):
                source = edge.get("from")
                target = edge.get("to")
                if not source or not target:
                    continue
                source_id = str(source)
                target_id = str(target)
                if source_id not in valid_targets or target_id not in valid_targets:
                    logger.warning(
                        "StructurerAgent dropped graph edge %s -> %s (unknown endpoint) from %s",
                        source_id,
                        target_id,
                        graph.get("id"),
                    )
                    continue
                edge_item = dict(edge)
                edge_item["from"] = source_id
                edge_item["to"] = target_id
                edges.append(edge_item)

            if nodes or edges:
                graph["nodes"] = nodes
                graph["edges"] = edges
                sanitized.append(graph)
            else:
                logger.warning("StructurerAgent dropped empty graph %s after sanitization", graph.get("id"))

        ups_ir["graphs"] = sanitized

    def _ensure_question_definitions(self, ups_ir: Dict[str, Any]) -> None:
        questions = ups_ir.get("questions")
        method_steps = ups_ir.get("method_steps")
        if not isinstance(questions, list):
            questions = []
            ups_ir["questions"] = questions
        if not isinstance(method_steps, list):
            return

        existing_ids = {
            str(question.get("id"))
            for question in questions
            if isinstance(question, dict) and question.get("id") is not None
        }

        for step in method_steps:
            if not isinstance(step, dict):
                continue
            question_id = step.get("question_id")
            if not question_id:
                continue
            question_id = str(question_id)
            step["question_id"] = question_id
            if question_id in existing_ids:
                continue
            stub = self._create_question_stub(question_id, step)
            questions.append(stub)
            existing_ids.add(question_id)
            logger.warning("StructurerAgent synthesized stub question definition for missing id %s", question_id)

    def _ensure_method_definitions(self, ups_ir: Dict[str, Any]) -> None:
        """
        Ensure every method referenced by method_steps has a corresponding entry.
        If the extractor omitted a method definition, synthesize a lightweight stub
        so that downstream validators see a coherent graph.
        """

        methods = ups_ir.get("methods")
        method_steps = ups_ir.get("method_steps")
        if not isinstance(methods, list) or not isinstance(method_steps, list):
            return

        existing_ids = {
            str(method.get("id"))
            for method in methods
            if isinstance(method, dict) and method.get("id") is not None
        }
        question_lookup = self._build_question_lookup(ups_ir.get("questions", []))

        for step in method_steps:
            if not isinstance(step, dict):
                continue
            method_id = step.get("method_id")
            if not method_id:
                continue
            method_id = str(method_id)
            step["method_id"] = method_id
            if method_id in existing_ids:
                continue
            stub = self._create_method_stub(method_id, step, question_lookup)
            methods.append(stub)
            existing_ids.add(method_id)
            logger.warning("StructurerAgent synthesized stub method definition for missing id %s", method_id)

    def _build_question_lookup(self, questions: Any) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for question in self._iter_dict_entries(questions):
            identifier = question.get("id")
            if identifier is not None:
                lookup[str(identifier)] = question
        return lookup

    def _create_method_stub(
        self,
        method_id: str,
        step: Dict[str, Any],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        stage = (step.get("stage") or "").strip()
        description = (step.get("description") or "").strip()
        question = questions.get(step.get("question_id") or "")
        question_text = ""
        if question:
            question_text = str(question.get("text") or "").strip()

        if stage and description:
            name = f"{stage}: {description}"
        elif description:
            name = description
        elif stage:
            name = f"{stage} ({method_id})"
        elif question_text:
            name = question_text
        else:
            name = f"Method {method_id}"

        name = name[:120] if len(name) > 120 else name
        method_description = description or question_text or stage or "unknown"
        source_reference = (
            step.get("source_reference")
            or (question.get("source_reference") if question else None)
            or "unknown"
        )
        source_text = step.get("source_text") or (question.get("source_text") if question else None)

        stub: Dict[str, Any] = {
            "id": method_id,
            "name": str(name) if name else f"Method {method_id}",
            "description": str(method_description),
            "architecture": "unknown",
            "uses_datasets": [],
            "uses_equations": [],
            "source_reference": str(source_reference),
        }
        if source_text:
            stub["source_text"] = str(source_text)
        return stub

    def _create_question_stub(self, question_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        step_description = (step.get("description") or "").strip()
        stage = (step.get("stage") or "").strip()
        text = step_description or stage or f"Auto-generated question {question_id}"
        goal = step.get("objective") or step_description or "unknown"
        source_reference = step.get("source_reference") or "StructurerAgent"
        source_text = step.get("source_text")

        stub: Dict[str, Any] = {
            "id": question_id,
            "text": text,
            "role": "sub",
            "goal": goal,
            "parent": None,
            "source_reference": source_reference,
        }
        if source_text:
            stub["source_text"] = source_text
        return stub

    def _build_entity_labels(self, ups_ir: Dict[str, Any]) -> Dict[str, str]:
        labels: Dict[str, str] = {}

        for question in self._iter_dict_entries(ups_ir.get("questions", [])):
            identifier = question.get("id")
            if identifier is not None:
                labels[str(identifier)] = str(question.get("text") or identifier)

        for method in self._iter_dict_entries(ups_ir.get("methods", [])):
            identifier = method.get("id")
            if identifier is not None:
                labels[str(identifier)] = str(method.get("name") or identifier)

        for step in self._iter_dict_entries(ups_ir.get("method_steps", [])):
            identifier = step.get("id")
            if identifier is None:
                continue
            descriptor = step.get("description") or step.get("stage") or identifier
            labels[str(identifier)] = str(descriptor)

        for task in self._iter_dict_entries(ups_ir.get("tasks", [])):
            identifier = task.get("id")
            if identifier is not None:
                labels[str(identifier)] = str(task.get("name") or identifier)

        return labels

    def _build_method_flow_graph(self, ups_ir: Dict[str, Any], labels: Dict[str, str]) -> Optional[Dict[str, Any]]:
        method_steps = self._iter_dict_entries(ups_ir.get("method_steps", []))
        method_steps = [step for step in method_steps if step.get("id")]
        if not method_steps:
            return None

        questions = self._build_question_lookup(ups_ir.get("questions", []))
        methods = {
            str(method.get("id")): method
            for method in self._iter_dict_entries(ups_ir.get("methods", []))
            if method.get("id")
        }

        nodes: set[str] = set()
        edges: list[Dict[str, Any]] = []
        edge_keys: set[tuple[str, str, str]] = set()
        references: set[str] = set()

        def add_edge(source: str, target: str, edge_type: str) -> None:
            if not source or not target:
                return
            key = (source, target, edge_type)
            if key in edge_keys:
                return
            edge_keys.add(key)
            description = f"{labels.get(source, source)} -> {labels.get(target, target)} ({edge_type})"
            edges.append({"from": source, "to": target, "type": edge_type, "description": description})

        for step in method_steps:
            step_id = str(step["id"])
            nodes.add(step_id)

            method_id = step.get("method_id")
            if method_id:
                method_id = str(method_id)
                nodes.add(method_id)
                add_edge(method_id, step_id, "implements")
                method = methods.get(method_id)
                if method:
                    ref = method.get("source_reference")
                    if ref:
                        references.add(str(ref))

            question_id = step.get("question_id")
            if question_id:
                question_id = str(question_id)
                nodes.add(question_id)
                if method_id:
                    add_edge(question_id, method_id, "addresses")
                else:
                    add_edge(question_id, step_id, "addresses")
                question = questions.get(question_id)
                if question:
                    ref = question.get("source_reference")
                    if ref:
                        references.add(str(ref))

            for dependency in step.get("depends_on", []) or []:
                dep_id = str(dependency)
                nodes.add(dep_id)
                add_edge(dep_id, step_id, "depends_on")

            ref = step.get("source_reference")
            if ref:
                references.add(str(ref))

        if not edges:
            return None

        source_reference = "; ".join(sorted(ref for ref in references if ref)) or "StructurerAgent"

        return {
            "name": "UPS-IR Method Flow",
            "description": "Auto-generated DAG linking questions, methods, and method steps for traceability.",
            "focus": "auto-flow",
            "nodes": sorted(nodes),
            "edges": edges,
            "source_reference": source_reference,
            "source_text": "Derived automatically from structured UPS-IR fields to enrich the pipeline view.",
        }

    def _next_graph_id(self, graphs: list[Dict[str, Any]]) -> str:
        max_index = 0
        for graph in graphs:
            identifier = graph.get("id")
            if identifier is None:
                continue
            match = re.fullmatch(r"g(\d+)", str(identifier))
            if match:
                max_index = max(max_index, int(match.group(1)))
        return f"g{max_index + 1}"

    def _collect_valid_targets(self, ups_ir: Dict[str, Any]) -> Set[str]:
        def collect(key: str) -> Set[str]:
            entries = ups_ir.get(key, [])
            if not isinstance(entries, list):
                return set()
            ids: Set[str] = set()
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id") is not None:
                    ids.add(str(entry["id"]))
            return ids

        valid = set()
        for key in (
            "methods",
            "datasets",
            "equations",
            "tasks",
            "questions",
            "method_steps",
            "sections",
            "algorithms",
            "figures",
            "parameters",
            "losses",
            "training_pipeline",
            "optimizations",
        ):
            valid |= collect(key)
        return valid
