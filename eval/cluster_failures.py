"""Cluster mined failures into recurring patterns the fixed 18-mode
taxonomy can't distinguish on its own.

The taxonomy answers "which of 18 known categories is this" — a stable,
interpretable baseline. It can't tell you that 12 separate
MALFORMED_JSON_OUTPUT failures across different runs are actually the same
root cause repeated 12 times (or 3 different ones). Clustering embeds each
failure's own description and groups by similarity, which is exactly how
the relation key-drift bug (see git history / README) should have been
surfaced automatically instead of found by manually reading output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from eval.mine_failures import RunResult
from eval.taxonomy import FailureMode


@dataclass
class FailureInstance:
    run_id: str
    mode: FailureMode | None
    text: str  # natural-language description this instance is embedded from


@dataclass
class FailureCluster:
    label: str
    instances: list[FailureInstance] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.instances)

    @property
    def modes(self) -> set[FailureMode]:
        return {i.mode for i in self.instances if i.mode}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "size": self.size,
            "modes": sorted(m.value for m in self.modes),
            "run_ids": [i.run_id for i in self.instances],
            "examples": [i.text for i in self.instances[:3]],
        }


def collect_failure_instances(results: Sequence[RunResult]) -> list[FailureInstance]:
    """One instance per grader signal and per judge verdict, across all
    mined runs — the raw material clustering groups."""
    instances: list[FailureInstance] = []
    for result in results:
        for signal in result.grader_signals.signals:
            text = f"{signal.detail} {signal.evidence}".strip()
            instances.append(FailureInstance(result.run_id, signal.mode, text))
        if result.judge_verdict and result.judge_verdict.primary_failure_mode:
            instances.append(FailureInstance(
                result.run_id, result.judge_verdict.primary_failure_mode, result.judge_verdict.rationale,
            ))
    return instances


def _default_embed_call(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model).embed_documents(texts)


def _default_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0).invoke(prompt).content


def cluster_instances(
    instances: list[FailureInstance],
    embed_call: Callable[[list[str]], list[list[float]]] | None = None,
    eps: float = 0.35,
    min_samples: int = 2,
) -> list[list[FailureInstance]]:
    """DBSCAN over embeddings, by cosine distance.

    DBSCAN (not k-means) deliberately: no need to pick a cluster count up
    front, and a failure too dissimilar from everything else is left as its
    own singleton ("noise", label -1) rather than forced into the nearest
    group — a one-off failure shouldn't be reported as if it were a
    recurring pattern.
    """
    if len(instances) < min_samples:
        return [[i] for i in instances]

    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_distances

    call = embed_call or _default_embed_call
    embeddings = np.array(call([i.text for i in instances]))
    distances = cosine_distances(embeddings)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distances)

    grouped: dict[int, list[FailureInstance]] = {}
    for label, instance in zip(labels, instances):
        grouped.setdefault(int(label), []).append(instance)

    result: list[list[FailureInstance]] = []
    for label, members in grouped.items():
        if label == -1:
            result.extend([[m] for m in members])  # noise points: one-item clusters, not forced together
        else:
            result.append(members)
    return result


def label_cluster(members: list[FailureInstance], llm_call: Callable[[str], str] | None = None,
                   model: str = "gpt-4o-mini") -> str:
    """A short, human-readable name for what a cluster's members have in
    common — a single instance just uses its own (truncated) text.
    """
    if len(members) == 1:
        return members[0].text[:80]

    examples = "\n".join(f"- {m.text[:200]}" for m in members[:8])
    prompt = (
        "These are descriptions of related failures from an ML agent pipeline. "
        "Write ONE short label (under 10 words) naming their common root cause — "
        "not a restatement of the symptom, the underlying reason.\n\n"
        f"{examples}\n\nLabel:"
    )
    call = llm_call or (lambda p: _default_llm_call(p, model=model))
    try:
        return call(prompt).strip().strip('"')
    except Exception:  # pragma: no cover - network/model dependent
        return members[0].text[:80]


def cluster_failures(
    results: Sequence[RunResult],
    embed_call: Callable[[list[str]], list[list[float]]] | None = None,
    llm_call: Callable[[str], str] | None = None,
    eps: float = 0.35,
    min_samples: int = 2,
) -> list[FailureCluster]:
    """Full pipeline: gather instances -> embed+cluster -> label each group.
    Returned largest-first so the most-recurring pattern is easiest to spot.
    """
    instances = collect_failure_instances(results)
    if not instances:
        return []

    groups = cluster_instances(instances, embed_call=embed_call, eps=eps, min_samples=min_samples)
    clusters = [FailureCluster(label=label_cluster(members, llm_call=llm_call), instances=members)
                for members in groups]
    clusters.sort(key=lambda c: -c.size)
    return clusters
