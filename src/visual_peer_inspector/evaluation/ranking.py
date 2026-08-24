"""Evaluator-only ranking measures for frozen label-free peer scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    top1: float
    reciprocal_rank: float
    average_precision: float
    recall_at_2: float
    exact_set: float
    positive_total: int
    retrieved_positive_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "top1": self.top1,
            "reciprocal_rank": self.reciprocal_rank,
            "average_precision": self.average_precision,
            "recall_at_2": self.recall_at_2,
            "exact_set": self.exact_set,
            "positive_total": self.positive_total,
            "retrieved_positive_count": self.retrieved_positive_count,
        }


def ranking_metrics(
    ranked_labels: Sequence[bool],
    *,
    positive_total: int,
    ranked_object_ids: Sequence[str | None] | None = None,
    positive_object_ids: Sequence[str] | None = None,
) -> RankingMetrics:
    """Measure a ranking while retaining missed positives in the denominator."""

    if positive_total <= 0:
        raise ValueError("positive_total must be positive")
    labels = [bool(value) for value in ranked_labels]
    if sum(labels) > positive_total:
        raise ValueError("Retrieved positives cannot exceed the reference total")
    first_positive = next((index for index, value in enumerate(labels) if value), None)
    reciprocal_rank = 0.0 if first_positive is None else 1.0 / (first_positive + 1)
    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(labels, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / positive_total
    recall_at_2 = sum(labels[:2]) / positive_total
    exact_set = 0.0
    if ranked_object_ids is not None or positive_object_ids is not None:
        if ranked_object_ids is None or positive_object_ids is None:
            raise ValueError("Both object ID collections are required for exact-set recovery")
        if len(ranked_object_ids) != len(labels):
            raise ValueError("Ranked object IDs must align with labels")
        selected = {value for value in ranked_object_ids[:positive_total] if value is not None}
        exact_set = float(selected == set(positive_object_ids))
    return RankingMetrics(
        top1=float(bool(labels and labels[0])),
        reciprocal_rank=reciprocal_rank,
        average_precision=average_precision,
        recall_at_2=recall_at_2,
        exact_set=exact_set,
        positive_total=positive_total,
        retrieved_positive_count=sum(labels),
    )


def average_views_within_scene(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Average repeated views before treating scenes as independent units."""

    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["opaque_scene_id"],
            row["opaque_asset_group_id"],
            row["encoder_key"],
            row["route"],
            row["method"],
            row["anomaly_count"],
        )
        groups[key].append(row)
    output = []
    metrics = ("top1", "reciprocal_rank", "average_precision", "recall_at_2", "exact_set")
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        row: dict[str, object] = {
            "opaque_scene_id": key[0],
            "opaque_asset_group_id": key[1],
            "encoder_key": key[2],
            "route": key[3],
            "method": key[4],
            "anomaly_count": key[5],
            "eligible_view_count": len(values),
        }
        for metric in metrics:
            row[metric] = mean(float(value[metric]) for value in values)
        output.append(row)
    return output


def clustered_mean_interval(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    cluster_key: str = "opaque_asset_group_id",
    repetitions: int = 10_000,
    seed: int = 32_040,
) -> dict[str, float | int]:
    """Bootstrap a scene-level mean by resampling complete asset clusters."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError("Bootstrap metric values must be finite")
        clusters[str(row[cluster_key])].append(value)
    if not clusters:
        raise ValueError("At least one cluster is required")
    names = sorted(clusters)
    observed = float(np.mean([value for name in names for value in clusters[name]]))
    generator = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        selected = [generator.choice(names) for _name in names]
        values = [value for name in selected for value in clusters[name]]
        samples.append(float(np.mean(values)))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "cluster_count": len(names),
        "repetitions": repetitions,
        "seed": seed,
    }


def paired_clustered_difference(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    pair_key: str = "opaque_scene_id",
    cluster_key: str = "opaque_asset_group_id",
    repetitions: int = 10_000,
    seed: int = 32_040,
) -> dict[str, float | int]:
    """Bootstrap candidate-minus-baseline differences using paired scenes.

    Complete asset clusters are resampled so repeated scenes from the same source
    asset do not become independent observations.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    baseline = {str(row[pair_key]): row for row in baseline_rows}
    candidate = {str(row[pair_key]): row for row in candidate_rows}
    paired_keys = sorted(set(baseline) & set(candidate))
    if not paired_keys:
        raise ValueError("At least one paired scene is required")

    clusters: dict[str, list[float]] = defaultdict(list)
    for key in paired_keys:
        baseline_row = baseline[key]
        candidate_row = candidate[key]
        baseline_cluster = str(baseline_row[cluster_key])
        candidate_cluster = str(candidate_row[cluster_key])
        if baseline_cluster != candidate_cluster:
            raise ValueError("Paired scenes must retain the same asset cluster")
        difference = float(candidate_row[metric]) - float(baseline_row[metric])
        if not np.isfinite(difference):
            raise ValueError("Paired metric differences must be finite")
        clusters[baseline_cluster].append(difference)

    names = sorted(clusters)
    observed = float(np.mean([value for name in names for value in clusters[name]]))
    generator = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        selected = [generator.choice(names) for _name in names]
        values = [value for name in selected for value in clusters[name]]
        samples.append(float(np.mean(values)))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "paired_scene_count": len(paired_keys),
        "cluster_count": len(names),
        "repetitions": repetitions,
        "seed": seed,
    }


__all__ = [
    "RankingMetrics",
    "average_views_within_scene",
    "clustered_mean_interval",
    "paired_clustered_difference",
    "ranking_metrics",
]
