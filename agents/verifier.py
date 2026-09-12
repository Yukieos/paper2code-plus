from __future__ import annotations

import logging
from typing import Any, Dict, Set

from jsonschema import Draft202012Validator, ValidationError


logger = logging.getLogger(__name__)


UPS_IR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "version",
        "generated_at",
        "meta",
        "questions",
        "tasks",
        "methods",
        "method_steps",
        "datasets",
        "equations",
        "experiments",
        "relations",
        "graphs",
        "sections",
        "figures",
        "algorithms",
        "parameters",
        "losses",
        "training_pipeline",
        "optimizations",
    ],
    "properties": {
        "version": {"type": "string"},
        "generated_at": {"type": "string"},
        "meta": {
            "type": "object",
            "required": ["title", "authors", "venue", "year"],
            "properties": {
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "venue": {"type": "string"},
                "year": {"type": ["integer", "string", "null"]},
            },
            #"additionalProperties": True,
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "text", "role", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "goal": {"type": ["string", "null"]},
                    "parent": {"type": ["string", "null"]},
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                #"additionalProperties": True,
            },
        },
        "methods": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "uses_datasets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "uses_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "description": {"type": ["string", "null"]},
                    "architecture": {"type": ["string", "null"]},
                    "source_reference": {"type": ["string", "null"]},
                    "source_text": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "method_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "method_id",
                    "description",
                    "source_reference",
                    "loss",
                    "optimizer",
                    "learning_rate",
                    "schedule",
                    "objective",
                    "target_metric",
                    "notes",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "method_id": {"type": "string"},
                    "question_id": {"type": ["string", "null"]},
                    "stage": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "hyperparameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": ["string", "null"]},
                                "value": {"type": ["string", "null"]},
                                "description": {"type": ["string", "null"]},
                            },
                        },
                        "default": [],
                    },
                    "loss": {"type": "string"},
                    "optimizer": {"type": "string"},
                    "learning_rate": {"type": "string"},
                    "schedule": {"type": "string"},
                    "objective": {"type": "string"},
                    "target_metric": {"type": "string"},
                    "notes": {"type": "string"},
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "split": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "equations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "latex"],
                "properties": {
                    "id": {"type": "string"},
                    "latex": {"type": "string"},
                    "units": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "experiments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["method"],
                "properties": {
                    "method": {"type": "string"},
                    "dataset": {"type": ["string", "null"]},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "object"},
                        "default": [],
                    },
                },
                #"additionalProperties": True,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "type"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "type": {"type": "string"},
                    "rationale": {"type": ["string", "null"]},
                    "source_reference": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "graphs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "nodes", "edges", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": ["string", "null"]},
                    "focus": {"type": ["string", "null"]},
                    "nodes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["from", "to", "type"],
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": ["string", "null"]},
                            },
                        },
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
                #"additionalProperties": True,
            },
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "title", "summary", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "level": {"type": ["integer", "null"]},
                    "summary": {"type": "string"},
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "related_components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "algorithms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "steps", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "symbol": {"type": ["string", "null"]},
                    "value": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "constraints": {"type": ["string", "null"]},
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "losses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "formula": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "related_methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "training_pipeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "description", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "hyperparameters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
        "optimizations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "objective", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "objective": {"type": "string"},
                    "optimizer": {"type": ["string", "null"]},
                    "schedule": {"type": ["string", "null"]},
                    "source_reference": {"type": "string"},
                    "source_text": {"type": ["string", "null"]},
                },
            },
        },
    },
    #"additionalProperties": True,
}


class VerifierAgent:
    """
    Validate UPS-IR output against a JSON schema and perform cross-reference checks.
    """

    def __init__(self) -> None:
        self.validator = Draft202012Validator(UPS_IR_SCHEMA)

    def run(self, state: Dict[str, Any]) -> bool:
        if "ups_ir" not in state:
            raise ValueError("VerifierAgent requires 'ups_ir' in the incoming state.")

        ups_ir = state["ups_ir"]

        try:
            self.validator.validate(ups_ir)
        except ValidationError as exc:
            raise ValueError(f"UPS-IR schema validation failed: {exc.message}") from exc

        self._check_integrity(ups_ir)
        logger.info("VerifierAgent validation successful.")
        return True

    def _check_integrity(self, ups_ir: Dict[str, Any]) -> None:
        method_ids = self._collect_ids(ups_ir.get("methods", []), "methods")
        dataset_ids = self._collect_ids(ups_ir.get("datasets", []), "datasets")
        equation_ids = self._collect_ids(ups_ir.get("equations", []), "equations")
        task_ids = self._collect_ids(ups_ir.get("tasks", []), "tasks")
        question_ids = self._collect_ids(ups_ir.get("questions", []), "questions")
        method_step_ids = self._collect_ids(ups_ir.get("method_steps", []), "method_steps")
        section_ids = self._collect_ids(ups_ir.get("sections", []), "sections")
        algorithm_ids = self._collect_ids(ups_ir.get("algorithms", []), "algorithms")
        figure_ids = self._collect_ids(ups_ir.get("figures", []), "figures")
        parameter_ids = self._collect_ids(ups_ir.get("parameters", []), "parameters")
        loss_ids = self._collect_ids(ups_ir.get("losses", []), "losses")
        training_ids = self._collect_ids(ups_ir.get("training_pipeline", []), "training_pipeline")
        optimization_ids = self._collect_ids(ups_ir.get("optimizations", []), "optimizations")
        graph_ids = self._collect_ids(ups_ir.get("graphs", []), "graphs")

        self._validate_method_links(ups_ir.get("methods", []), dataset_ids, equation_ids)
        self._validate_experiments(ups_ir.get("experiments", []), method_ids, dataset_ids)
        self._validate_method_steps(ups_ir.get("method_steps", []), method_ids, question_ids, method_step_ids)
        self._validate_relations(
            ups_ir.get("relations", []),
            method_ids,
            dataset_ids,
            equation_ids,
            task_ids,
            question_ids,
            method_step_ids,
            section_ids,
            algorithm_ids,
            figure_ids,
            parameter_ids,
            loss_ids,
            training_ids,
            optimization_ids,
        )
        valid_targets = (
            method_ids
            | dataset_ids
            | equation_ids
            | task_ids
            | question_ids
            | method_step_ids
            | section_ids
            | algorithm_ids
            | figure_ids
            | parameter_ids
            | loss_ids
            | training_ids
            | optimization_ids
        )
        self._validate_graphs(ups_ir.get("graphs", []), valid_targets)

    def _collect_ids(self, items: Any, label: str) -> Set[str]:
        ids: Set[str] = set()
        if not isinstance(items, list):
            raise ValueError(f"{label} should be a list.")
        for entry in items:
            if not isinstance(entry, dict):
                raise ValueError(f"Entries in {label} must be objects.")
            identifier = entry.get("id")
            if not identifier:
                raise ValueError(f"Missing 'id' in {label} entry: {entry}")
            if identifier in ids:
                raise ValueError(f"Duplicate id '{identifier}' detected in {label}.")
            ids.add(identifier)
        return ids

    def _validate_method_links(self, methods: Any, dataset_ids: Set[str], equation_ids: Set[str]) -> None:
        for method in methods:
            for dataset in method.get("uses_datasets", []) or []:
                if dataset not in dataset_ids:
                    raise ValueError(f"Method {method.get('id')} references unknown dataset '{dataset}'.")
            for equation in method.get("uses_equations", []) or []:
                if equation not in equation_ids:
                    raise ValueError(f"Method {method.get('id')} references unknown equation '{equation}'.")

    def _validate_experiments(self, experiments: Any, method_ids: Set[str], dataset_ids: Set[str]) -> None:
        for experiment in experiments:
            method = experiment.get("method")
            if method not in method_ids:
                raise ValueError(f"Experiment references unknown method '{method}'.")
            dataset = experiment.get("dataset")
            if dataset and dataset not in dataset_ids:
                raise ValueError(f"Experiment references unknown dataset '{dataset}'.")

    def _validate_method_steps(
        self,
        method_steps: Any,
        method_ids: Set[str],
        question_ids: Set[str],
        method_step_ids: Set[str],
    ) -> None:
        for step in method_steps:
            method_id = step.get("method_id")
            if method_id not in method_ids:
                raise ValueError(f"Method step {step.get('id')} references unknown method '{method_id}'.")

            question_id = step.get("question_id")
            if question_id and question_id not in question_ids:
                raise ValueError(f"Method step {step.get('id')} references unknown question '{question_id}'.")

            for dependency in step.get("depends_on", []) or []:
                if dependency not in method_step_ids and dependency not in question_ids:
                    raise ValueError(
                        f"Method step {step.get('id')} depends on unknown step/question '{dependency}'."
                    )

    def _validate_relations(
        self,
        relations: Any,
        method_ids: Set[str],
        dataset_ids: Set[str],
        equation_ids: Set[str],
        task_ids: Set[str],
        question_ids: Set[str],
        method_step_ids: Set[str],
        section_ids: Set[str],
        algorithm_ids: Set[str],
        figure_ids: Set[str],
        parameter_ids: Set[str],
        loss_ids: Set[str],
        training_ids: Set[str],
        optimization_ids: Set[str],
    ) -> None:
        valid_targets = (
            method_ids
            | dataset_ids
            | equation_ids
            | task_ids
            | question_ids
            | method_step_ids
            | section_ids
            | algorithm_ids
            | figure_ids
            | parameter_ids
            | loss_ids
            | training_ids
            | optimization_ids
        )
        for relation in relations:
            source = relation.get("from")
            target = relation.get("to")
            relation_type = relation.get("type")

            if source not in valid_targets:
                raise ValueError(f"Relation source '{source}' is not a known entity.")
            if target not in valid_targets:
                raise ValueError(f"Relation target '{target}' is not a known entity.")
            if not relation_type:
                raise ValueError("Relation type is required.")

    def _validate_graphs(self, graphs: Any, valid_targets: Set[str]) -> None:
        for graph in graphs:
            for node in graph.get("nodes", []) or []:
                if node not in valid_targets:
                    raise ValueError(f"Graph {graph.get('id')} references unknown node '{node}'.")
            for edge in graph.get("edges", []) or []:
                source = edge.get("from")
                target = edge.get("to")
                if source not in valid_targets:
                    raise ValueError(f"Graph edge source '{source}' is not a known entity.")
                if target not in valid_targets:
                    raise ValueError(f"Graph edge target '{target}' is not a known entity.")
