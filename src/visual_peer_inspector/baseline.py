from __future__ import annotations

from math import sqrt
import random
from statistics import median
from typing import Sequence


def _unit_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    length = sqrt(sum(value * value for value in values))
    if length == 0:
        raise ValueError("Feature vectors cannot be zero")
    return [value / length for value in values]


def peer_mean_scores(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Score each object by its cosine distance from the other objects."""

    if len(vectors) < 2:
        raise ValueError("At least two objects are required")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Feature vectors must have the same non-zero length")

    features = [_unit_vector(vector) for vector in vectors]
    scores = []
    for object_index, feature in enumerate(features):
        peers = [item for index, item in enumerate(features) if index != object_index]
        peer_mean = [
            sum(peer[column] for peer in peers) / len(peers)
            for column in range(len(feature))
        ]
        peer_mean = _unit_vector(peer_mean)
        similarity = sum(
            value * peer_value
            for value, peer_value in zip(feature, peer_mean, strict=True)
        )
        scores.append(1.0 - similarity)
    return scores


def median_peer_scores(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Score each object by its median cosine distance from every peer."""

    if len(vectors) < 2:
        raise ValueError("At least two objects are required")

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Feature vectors must have the same non-zero length")

    features = [_unit_vector(vector) for vector in vectors]
    scores = []
    for object_index, feature in enumerate(features):
        distances = []
        for peer_index, peer in enumerate(features):
            if peer_index == object_index:
                continue
            similarity = sum(
                value * peer_value
                for value, peer_value in zip(feature, peer, strict=True)
            )
            distances.append(1.0 - similarity)
        scores.append(float(median(distances)))
    return scores


def average_precision(labels: Sequence[bool], ranking: Sequence[int]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        raise ValueError("Average precision requires an anomalous object")
    if sorted(ranking) != list(range(len(labels))):
        raise ValueError("Ranking must contain every object index once")

    hits = 0
    precision_total = 0.0
    for rank, object_index in enumerate(ranking, start=1):
        if labels[object_index]:
            hits += 1
            precision_total += hits / rank
    return precision_total / positive_count


def permutation_error(
    vectors: Sequence[Sequence[float]],
    scores: Sequence[float],
    *,
    scorer=peer_mean_scores,
    seed: int = 20260807,
) -> float:
    order = list(range(len(vectors)))
    random.Random(seed).shuffle(order)
    shuffled_scores = scorer([vectors[index] for index in order])
    restored = [0.0] * len(scores)
    for new_index, old_index in enumerate(order):
        restored[old_index] = shuffled_scores[new_index]
    return max(abs(left - right) for left, right in zip(scores, restored, strict=True))
