"""Label-free peer support and mask geometry scores for object comparison."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from visual_peer_inspector.scoring.peer import PeerScoreResult, _unit_rows


def _tail_mean(values: np.ndarray, fraction: float) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not len(values):
        return 0.0
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, len(values) - count)[-count:].mean())


def mutual_missing_scores(
    patch_features: Sequence[np.ndarray],
    valid_patch_masks: Sequence[np.ndarray],
    *,
    tail_fraction: float = 0.05,
) -> PeerScoreResult:
    """Find features shared by peers but absent from one candidate.

    For every peer patch, the method compares its nearest match in the target
    candidate with its nearest match in the remaining peers. Positive evidence
    means that at least two peers support a feature that the target lacks. This
    avoids treating an isolated detail on one peer as evidence of a missing
    part on another object.
    """

    if len(patch_features) != len(valid_patch_masks):
        raise ValueError("Patch features and validity masks must have equal length")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if len(patch_features) < 3:
        return PeerScoreResult(
            "mutual_missing",
            "insufficient_peers",
            np.full(len(patch_features), np.nan, dtype=np.float64),
        )

    vectors: list[np.ndarray] = []
    for features, mask in zip(patch_features, valid_patch_masks, strict=True):
        feature_array = _unit_rows(features)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2 or mask_array.size != len(feature_array):
            raise ValueError("Patch feature count must match the validity grid")
        if not mask_array.any():
            raise ValueError("Every object needs at least one valid patch")
        vectors.append(feature_array[mask_array.ravel()])

    candidate_scores: list[float] = []
    for target_index, target in enumerate(vectors):
        peer_scores: list[float] = []
        for peer_index, peer in enumerate(vectors):
            if peer_index == target_index:
                continue
            remaining = [
                item
                for index, item in enumerate(vectors)
                if index not in {target_index, peer_index}
            ]
            other_peers = np.concatenate(remaining, axis=0)
            distance_to_target = np.min(1.0 - peer @ target.T, axis=1)
            distance_to_support = np.min(1.0 - peer @ other_peers.T, axis=1)
            supported_absence = np.maximum(0.0, distance_to_target - distance_to_support)
            peer_scores.append(_tail_mean(supported_absence, tail_fraction))
        candidate_scores.append(float(np.median(peer_scores)))

    return PeerScoreResult(
        "mutual_missing",
        "scored",
        np.asarray(candidate_scores, dtype=np.float64),
    )


def mask_shape_descriptor(mask: np.ndarray) -> np.ndarray:
    """Return scale-aware silhouette descriptors without using object labels."""

    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2 or not foreground.any():
        raise ValueError("A geometry descriptor requires a non-empty binary mask")
    rows, columns = np.nonzero(foreground)
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)
    area = float(len(rows))
    extent = area / float(height * width)
    centred = np.column_stack((columns - columns.mean(), rows - rows.mean()))
    covariance = centred.T @ centred / max(1, len(centred))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = max(float(eigenvalues[-1]), 1e-12)
    minor = max(float(eigenvalues[0]), 0.0)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - minor / major)))
    direction = eigenvectors[:, -1]
    angle = float(np.arctan2(direction[1], direction[0]))
    padded = np.pad(foreground, 1, constant_values=False)
    neighbours = (
        padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
    )
    perimeter = float(np.count_nonzero(foreground & ~neighbours))
    compactness = float((perimeter * perimeter) / max(area, 1.0))
    return np.asarray(
        [
            np.log(max(area, 1.0)),
            np.log(max(width / max(height, 1), 1e-12)),
            extent,
            eccentricity,
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
            compactness,
        ],
        dtype=np.float64,
    )


def geometry_peer_scores(descriptors: Sequence[np.ndarray]) -> PeerScoreResult:
    """Score robust deviation from peer silhouette geometry."""

    matrix = np.asarray(descriptors, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Geometry descriptors must form a finite matrix")
    if len(matrix) < 3:
        return PeerScoreResult(
            "geometry_peer",
            "insufficient_peers",
            np.full(len(matrix), np.nan, dtype=np.float64),
        )
    centre = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - centre), axis=0)
    fallback = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    scores = []
    for index, descriptor in enumerate(matrix):
        peers = np.delete(matrix, index, axis=0)
        peer_centre = np.median(peers, axis=0)
        scores.append(float(np.sqrt(np.mean(np.square((descriptor - peer_centre) / scale)))))
    return PeerScoreResult("geometry_peer", "scored", np.asarray(scores))


def rank_consensus(*score_arrays: Sequence[float]) -> np.ndarray:
    """Combine independent evidence using within-view mean percentile ranks."""

    if not score_arrays:
        raise ValueError("At least one score array is required")
    arrays = [np.asarray(values, dtype=np.float64) for values in score_arrays]
    if len({len(values) for values in arrays}) != 1 or not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Consensus inputs must be finite arrays of equal length")
    if len(arrays[0]) < 2:
        return np.zeros(len(arrays[0]), dtype=np.float64)
    ranked = []
    for values in arrays:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        for value in np.unique(values):
            tied = np.flatnonzero(np.isclose(values, value, atol=1e-12, rtol=0.0))
            ranks[tied] = ranks[tied].mean()
        ranked.append(ranks / (len(values) - 1))
    return np.mean(np.stack(ranked), axis=0)


__all__ = [
    "geometry_peer_scores",
    "mask_shape_descriptor",
    "mutual_missing_scores",
    "rank_consensus",
]
