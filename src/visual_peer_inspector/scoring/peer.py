"""B0, B1 and P1 peer scoring without access to anomaly labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np


@dataclass(frozen=True)
class PeerScoreResult:
    method: str
    status: str
    scores: np.ndarray
    heatmaps: tuple[np.ndarray, ...] = ()


def _unit_rows(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("Feature matrix must have shape objects x dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError("Feature matrix must contain finite values")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(lengths <= 0.0):
        raise ValueError("Feature vectors cannot be zero")
    return array / lengths


def _unscoreable(method: str, object_count: int) -> PeerScoreResult:
    return PeerScoreResult(
        method=method,
        status="insufficient_peers",
        scores=np.full(object_count, np.nan, dtype=np.float64),
    )


def b0_scores(features: Sequence[Sequence[float]] | np.ndarray) -> PeerScoreResult:
    """Score distance from the normalized leave-one-out peer mean."""

    vectors = _unit_rows(features)
    if len(vectors) < 3:
        return _unscoreable("B0", len(vectors))
    total = vectors.sum(axis=0, keepdims=True)
    peer_means = (total - vectors) / (len(vectors) - 1)
    peer_means = _unit_rows(peer_means)
    scores = 1.0 - np.sum(vectors * peer_means, axis=1)
    return PeerScoreResult("B0", "scored", scores)


def b1_scores(features: Sequence[Sequence[float]] | np.ndarray) -> PeerScoreResult:
    """Score the median cosine distance to all other objects in the view."""

    vectors = _unit_rows(features)
    if len(vectors) < 3:
        return _unscoreable("B1", len(vectors))
    distances = 1.0 - vectors @ vectors.T
    np.fill_diagonal(distances, np.nan)
    return PeerScoreResult("B1", "scored", np.nanmedian(distances, axis=1))


def _top_tail_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def p1_scores(
    patch_features: Sequence[np.ndarray],
    valid_patch_masks: Sequence[np.ndarray],
    *,
    top_fraction: float = 0.10,
) -> PeerScoreResult:
    """Score local patch mismatch and return one coarse heatmap per object.

    Each target patch is compared with every valid patch in a peer object. The
    target heatmap is the median nearest-neighbour distance across peers. The
    object score is the median symmetric top-tail distance to its peers.
    """

    if len(patch_features) != len(valid_patch_masks):
        raise ValueError("Patch features and validity masks must have equal length")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if len(patch_features) < 3:
        return _unscoreable("P1", len(patch_features))
    normalized: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    grid_shapes: list[tuple[int, int]] = []
    for features, mask in zip(patch_features, valid_patch_masks, strict=True):
        feature_array = np.asarray(features, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2 or feature_array.ndim != 2:
            raise ValueError("Patch features must be 2D and masks must be patch grids")
        if feature_array.shape[0] != mask_array.size or feature_array.shape[1] == 0:
            raise ValueError("Patch feature count must match the validity grid")
        unit = _unit_rows(feature_array)
        if not np.any(mask_array):
            raise ValueError("Every object must contain at least one valid patch")
        normalized.append(unit)
        valid.append(mask_array.reshape(-1))
        grid_shapes.append(mask_array.shape)

    peer_distances: list[list[float]] = [[] for _ in normalized]
    directed_maps: list[list[np.ndarray]] = [[] for _ in normalized]
    for left in range(len(normalized)):
        left_vectors = normalized[left][valid[left]]
        for right in range(len(normalized)):
            if left == right:
                continue
            right_vectors = normalized[right][valid[right]]
            distances = 1.0 - left_vectors @ right_vectors.T
            left_nearest = distances.min(axis=1)
            right_nearest = distances.min(axis=0)
            symmetric = 0.5 * (
                _top_tail_mean(left_nearest, top_fraction)
                + _top_tail_mean(right_nearest, top_fraction)
            )
            peer_distances[left].append(symmetric)
            full = np.full(valid[left].shape, np.nan, dtype=np.float64)
            full[valid[left]] = left_nearest
            directed_maps[left].append(full)

    scores = np.asarray([np.median(values) for values in peer_distances])
    heatmaps = []
    for maps, shape in zip(directed_maps, grid_shapes, strict=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            heatmaps.append(np.nanmedian(np.stack(maps), axis=0).reshape(shape))
    return PeerScoreResult("P1", "scored", scores, tuple(heatmaps))


def deterministic_ranking(scores: Sequence[float], identifiers: Sequence[str]) -> list[int]:
    """Rank high scores first and break exact ties with stable identifiers."""

    if len(scores) != len(identifiers):
        raise ValueError("Scores and identifiers must have equal length")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Ranking identifiers must be unique")
    pairs = []
    for index, (score, identifier) in enumerate(zip(scores, identifiers, strict=True)):
        numeric = float(score)
        if not np.isfinite(numeric):
            raise ValueError("Ranked scores must be finite")
        pairs.append((-numeric, str(identifier), index))
    return [index for _negative, _identifier, index in sorted(pairs)]


__all__ = [
    "PeerScoreResult",
    "b0_scores",
    "b1_scores",
    "deterministic_ranking",
    "p1_scores",
]
