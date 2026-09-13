"""A small development-only ranker built on frozen peer-comparison features."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ContextGroup:
    scene: str
    view: str
    route: str
    asset_group: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        tied = np.flatnonzero(np.isclose(values, value, atol=1e-12, rtol=0.0))
        ranks[tied] = ranks[tied].mean()
    return ranks / max(1, len(values) - 1)


def context_features(objects: list[dict[str, Any]], methods: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build relative features without source names, labels or object order."""

    if len(objects) < 3:
        raise ValueError("Context features require at least three candidates")
    score_columns = []
    names = []
    for method in methods:
        values = np.asarray([item["scores"].get(method) for item in objects], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Context inputs require complete finite component scores")
        score_columns.append(_percentile_rank(values))
        names.append(f"rank_{method}")

    geometry = np.asarray([item["geometry"] for item in objects], dtype=np.float64)
    if geometry.shape != (len(objects), 7) or not np.isfinite(geometry).all():
        raise ValueError("Context inputs require seven finite geometry descriptors")
    centre = np.median(geometry, axis=0)
    scale = 1.4826 * np.median(np.abs(geometry - centre), axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    geometry_deviation = np.abs((geometry - centre) / scale)
    names.extend(
        [
            "geometry_log_area",
            "geometry_log_aspect",
            "geometry_extent",
            "geometry_eccentricity",
            "geometry_sin_2_angle",
            "geometry_cos_2_angle",
            "geometry_compactness",
        ]
    )

    embeddings = np.asarray([item["embedding"] for item in objects], dtype=np.float64)
    if embeddings.ndim != 2 or not embeddings.shape[1] or not np.isfinite(embeddings).all():
        raise ValueError("Context inputs require finite object embeddings")
    lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(lengths, 1e-12)
    pairwise = 1.0 - embeddings @ embeddings.T
    np.fill_diagonal(pairwise, np.nan)
    nearest = np.nanmin(pairwise, axis=1)
    median = np.nanmedian(pairwise, axis=1)
    farthest = np.nanmax(pairwise, axis=1)
    peer_centroid = (embeddings.sum(axis=0, keepdims=True) - embeddings) / (len(embeddings) - 1)
    peer_centroid /= np.maximum(np.linalg.norm(peer_centroid, axis=1, keepdims=True), 1e-12)
    centroid_distance = 1.0 - np.sum(embeddings * peer_centroid, axis=1)
    names.extend(["nearest_peer", "median_peer", "farthest_peer", "peer_centroid"])

    score_matrix = np.column_stack(score_columns)
    forward = score_matrix[:, methods.index("forward_02_cov70")]
    missing = score_matrix[:, methods.index("mutual_missing_05")]
    shape = score_matrix[:, methods.index("geometry_peer")]
    interactions = np.column_stack((forward * missing, forward * shape, missing * shape))
    names.extend(["forward_x_missing", "forward_x_geometry", "missing_x_geometry"])
    matrix = np.column_stack(
        (
            score_matrix,
            geometry_deviation,
            nearest,
            median,
            farthest,
            centroid_distance,
            interactions,
        )
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Context feature construction produced a non-finite value")
    return matrix, tuple(names)


def assign_asset_folds(asset_groups: set[str], fold_count: int) -> dict[str, int]:
    """Assign all views of an asset group to one deterministic fold."""

    if fold_count < 2 or len(asset_groups) < fold_count:
        raise ValueError("Cross-validation requires at least one asset group per fold")
    ordered = sorted(asset_groups, key=lambda value: sha256(f"context-fold-v1/{value}".encode()).hexdigest())
    return {asset: index % fold_count for index, asset in enumerate(ordered)}


def _standardizer(groups: list[ContextGroup]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([group.features for group in groups], axis=0)
    centre = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return centre, scale


def _loss_and_gradient(weights: np.ndarray, groups: list[ContextGroup], centre: np.ndarray, scale: np.ndarray, regularization: float) -> tuple[float, np.ndarray]:
    coefficients, intercept = weights[:-1], weights[-1]
    loss = 0.0
    gradient = np.zeros_like(weights)
    used = 0
    for group in groups:
        positives = np.flatnonzero(group.labels > 0)
        if not len(positives):
            continue
        features = (group.features - centre) / scale
        logits = features @ coefficients + intercept
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        target = np.zeros(len(logits), dtype=np.float64)
        target[positives] = 1.0 / len(positives)
        loss -= float(np.sum(target * np.log(np.maximum(probabilities, 1e-15))))
        residual = probabilities - target
        gradient[:-1] += features.T @ residual
        gradient[-1] += residual.sum()
        used += 1
    if not used:
        raise ValueError("The training fold contains no scoreable positive scene")
    loss = loss / used + 0.5 * regularization * float(coefficients @ coefficients)
    gradient = gradient / used
    gradient[:-1] += regularization * coefficients
    return loss, gradient


def fit_context_ranker(groups: list[ContextGroup], *, regularization: float = 1.0) -> dict[str, Any]:
    """Fit a deterministic scene-softmax linear context head."""

    if regularization <= 0:
        raise ValueError("regularization must be positive")
    centre, scale = _standardizer(groups)
    dimension = groups[0].features.shape[1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return _loss_and_gradient(weights, groups, centre, scale, regularization)

    result = minimize(
        objective,
        np.zeros(dimension + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Context ranker optimization failed: {result.message}")
    return {
        "schema": 1,
        "kind": "scene-softmax peer-context ranker",
        "regularization": float(regularization),
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "weights": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "training_groups": len(groups),
        "training_groups_with_positive": sum(bool(np.any(group.labels > 0)) for group in groups),
        "loss": float(result.fun),
        "iterations": int(result.nit),
    }


def score_context_group(group: ContextGroup, model: dict[str, Any]) -> np.ndarray:
    centre = np.asarray(model["centre"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if group.features.shape[1] != len(weights) or len(centre) != len(weights) or len(scale) != len(weights):
        raise ValueError("Context model and feature schema differ")
    return ((group.features - centre) / scale) @ weights + float(model["intercept"])


__all__ = [
    "ContextGroup",
    "assign_asset_folds",
    "context_features",
    "fit_context_ranker",
    "score_context_group",
]
