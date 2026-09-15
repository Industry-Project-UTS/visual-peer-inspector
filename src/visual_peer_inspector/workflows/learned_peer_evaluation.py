"""Grouped development evaluation for learned peer and view ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from math import comb, sqrt
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment, minimize

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.workflows.comparison_io import (
    atomic_json,
    canonical,
    digest_file,
    read_json,
    seal,
    verify_seal,
)
from visual_peer_inspector.workflows.learned_peer_features import METHODS, VIEW_SETS
from visual_peer_inspector.workflows import learned_peer_features as feature_module


STAGE = "development-learned-peer-view-evaluation-v1"
LEARNED_METHODS = ("learned_listwise", "learned_ranknet", "learned_hybrid")
FIXED_METHODS = (
    "B0",
    "P1_current",
    "P2_consensus",
    "forward_02_cov70",
    "robust_forward",
    "robust_borda",
    "soft_support_fusion",
    "view_vote_forward",
    "view_vote_consensus",
    "winner_cluster_forward",
    "winner_cluster_consensus",
)
ALL_METHODS = FIXED_METHODS + LEARNED_METHODS

# The one-standard-error heuristic favours the established forward scorer when
# it remains close to a more complex candidate across grouped folds. This avoids
# selecting a complicated rule from a 100-scene screen.
SIMPLE_METHOD_ORDER = (
    "forward_02_cov70",
    "B0",
    "P1_current",
    "P2_consensus",
    "robust_forward",
    "robust_borda",
    "soft_support_fusion",
    "view_vote_forward",
    "view_vote_consensus",
    "winner_cluster_forward",
    "winner_cluster_consensus",
    "learned_listwise",
    "learned_ranknet",
    "learned_hybrid",
)

METHOD_REGISTRY = {
    "B0": "Mean fused B0 percentile rank",
    "P1_current": "Mean fused current P1 percentile rank",
    "P2_consensus": "Mean fused P2 percentile rank",
    "forward_02_cov70": "Mean fused selected forward-patch percentile rank",
    "robust_forward": "Huber fused forward-patch percentile rank",
    "robust_borda": "Huber fusion of five within-view scorer ranks",
    "soft_support_fusion": "Fixed robust fusion of forward, P1 and cross-view DINO support",
    "view_vote_forward": "Single-view forward rankings accumulated as soft votes across predicted tracks",
    "view_vote_consensus": "Single-view forward, P1 and P2 rankings accumulated as soft votes across predicted tracks",
    "winner_cluster_forward": "Cluster per-view forward winners at fixed DINO distance 0.20; retain unsupported winners as review cases",
    "winner_cluster_consensus": "Cluster per-view consensus winners at fixed DINO distance 0.20; retain unsupported winners as review cases",
    "learned_listwise": "Linear scene-softmax learning-to-rank head",
    "learned_ranknet": "Linear pairwise RankNet-style head",
    "learned_hybrid": "Equal-weight scene-softmax and pairwise loss",
}

METHOD_LABELS = {
    "B0": "B0",
    "P1_current": "P1",
    "P2_consensus": "P2",
    "forward_02_cov70": "Forward patch",
    "robust_forward": "Robust forward",
    "robust_borda": "Borda fusion",
    "soft_support_fusion": "Support fusion",
    "view_vote_forward": "Forward vote",
    "view_vote_consensus": "Consensus vote",
    "winner_cluster_forward": "Winner cluster",
    "winner_cluster_consensus": "Winner consensus",
    "learned_listwise": "Listwise ranker",
    "learned_ranknet": "RankNet",
    "learned_hybrid": "Hybrid ranker",
}


@dataclass(frozen=True)
class RankGroup:
    scene: str
    asset_group: str
    route: str
    view_set_id: str
    view: str
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray


def _balanced_asset_folds(truths: dict[str, dict[str, Any]], fold_count: int = 5) -> dict[str, int]:
    """Balance deterministic folds by asset scene count and dominant category."""

    by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for truth in truths.values():
        by_asset[str(truth["asset_group"])].append(truth)
    if len(by_asset) < fold_count:
        raise ValueError("Grouped evaluation needs at least one asset group per fold")

    fold_sizes = [0] * fold_count
    category_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(fold_count)]
    assignments: dict[str, int] = {}
    ordered = sorted(
        by_asset.items(),
        key=lambda item: (
            -len(item[1]),
            sha256(f"learned-ranker-fold-v1/{item[0]}".encode()).hexdigest(),
        ),
    )
    for asset, rows in ordered:
        categories = [str(row.get("category", "unknown")) for row in rows]
        category = max(sorted(set(categories)), key=categories.count)
        target = min(
            range(fold_count),
            key=lambda fold: (
                fold_sizes[fold],
                category_counts[fold][category],
                fold,
            ),
        )
        assignments[asset] = target
        fold_sizes[target] += len(rows)
        category_counts[target][category] += len(rows)
    if set(assignments) != set(by_asset):
        raise ValueError("Fold assignment is incomplete")
    return assignments


def _truth_label(truth: dict[str, Any], route: str, view: str, candidate: str) -> int:
    reference = truth["correspondence"].get(f"{route}/{view}/{candidate}")
    return int(truth["labels"].get(reference, 0))


def load_rank_groups(
    feature_root: Path, evaluator_root: Path
) -> tuple[list[RankGroup], dict[str, dict[str, Any]], tuple[str, ...], dict[str, Any]]:
    """Verify the label-free lock before reading any evaluator scene."""

    feature_root = Path(feature_root)
    evaluator_root = Path(evaluator_root)
    feature_lock = verify_seal(feature_root, name="feature_lock.json")
    if feature_lock["metadata"].get("evaluator_accessed") is not False:
        raise ValueError("The feature package does not prove a label-free preparation stage")

    evaluator_lock = verify_seal(evaluator_root)
    feature_paths = sorted((feature_root / "scenes").glob("S*.json"))
    feature_paths = [path for path in feature_paths if not path.name.endswith(".receipt.json")]
    if not feature_paths:
        raise ValueError("The feature package has no scene checkpoints")

    truths: dict[str, dict[str, Any]] = {}
    groups: list[RankGroup] = []
    feature_names: tuple[str, ...] | None = None
    for path in feature_paths:
        payload = read_json(path)
        scene = str(payload["scene"])
        truth_path = evaluator_root / "scenes" / f"{scene}.json"
        if not truth_path.is_file():
            raise ValueError(f"Evaluator package is missing scene {scene}")
        truth = read_json(truth_path)
        truths[scene] = truth
        current_names = tuple(payload["feature_names"])
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError("Feature names differ between scene checkpoints")
        for setting in payload["groups"]:
            for view_group in setting["view_groups"]:
                candidates = tuple(view_group["candidate_ids"])
                features = np.asarray(view_group["features"], dtype=np.float64)
                if not candidates and features.size == 0:
                    features = features.reshape(0, len(current_names))
                labels = np.asarray(
                    [
                        _truth_label(truth, setting["route"], view_group["view"], candidate)
                        for candidate in candidates
                    ],
                    dtype=np.int8,
                )
                if features.shape != (len(candidates), len(current_names)):
                    raise ValueError("Feature matrix shape differs from the declared schema")
                if not np.isfinite(features).all():
                    raise ValueError("Feature matrix contains a non-finite value")
                groups.append(
                    RankGroup(
                        scene=scene,
                        asset_group=str(truth["asset_group"]),
                        route=str(setting["route"]),
                        view_set_id=str(setting["view_set_id"]),
                        view=str(view_group["view"]),
                        candidate_ids=candidates,
                        features=features,
                        labels=labels,
                    )
                )
    expected = {path.stem for path in feature_paths}
    if set(truths) != expected:
        raise ValueError("Feature and evaluator scene memberships differ")
    return groups, truths, feature_names or (), evaluator_lock


def _standardizer(groups: list[RankGroup]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([group.features for group in groups], axis=0)
    centre = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    return centre, np.where(scale > 1e-8, scale, 1.0)


def _group_losses(
    weights: np.ndarray,
    groups: list[RankGroup],
    centre: np.ndarray,
    scale: np.ndarray,
    objective: str,
    regularization: float,
) -> tuple[float, np.ndarray, int]:
    loss = 0.0
    gradient = np.zeros_like(weights)
    used = 0
    for group in groups:
        positives = np.flatnonzero(group.labels > 0)
        negatives = np.flatnonzero(group.labels == 0)
        if not len(positives) or not len(negatives):
            continue
        features = (group.features - centre) / scale
        scores = features @ weights
        group_loss = 0.0
        group_gradient = np.zeros_like(weights)

        if objective in {"listwise", "hybrid"}:
            shifted = scores - scores.max()
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            target = np.zeros(len(scores), dtype=np.float64)
            target[positives] = 1.0 / len(positives)
            component_loss = -float(np.sum(target * np.log(np.maximum(probabilities, 1e-15))))
            component_gradient = features.T @ (probabilities - target)
            weight = 1.0 if objective == "listwise" else 0.5
            group_loss += weight * component_loss
            group_gradient += weight * component_gradient

        if objective in {"ranknet", "hybrid"}:
            differences = (
                features[positives][:, None, :] - features[negatives][None, :, :]
            ).reshape(-1, features.shape[1])
            margins = differences @ weights
            component_loss = float(np.logaddexp(0.0, -margins).mean())
            sigmoid_negative = 1.0 / (1.0 + np.exp(np.clip(margins, -60.0, 60.0)))
            component_gradient = -(differences * sigmoid_negative[:, None]).mean(axis=0)
            weight = 1.0 if objective == "ranknet" else 0.5
            group_loss += weight * component_loss
            group_gradient += weight * component_gradient

        loss += group_loss
        gradient += group_gradient
        used += 1
    if not used:
        raise ValueError("A training fold has no groups containing both classes")
    loss = loss / used + 0.5 * regularization * float(weights @ weights)
    gradient = gradient / used + regularization * weights
    return loss, gradient, used


def fit_linear_ranker(
    groups: list[RankGroup], *, objective: str, regularization: float = 1.0
) -> dict[str, Any]:
    """Fit one deterministic linear ranker with scene-balanced loss."""

    if objective not in {"listwise", "ranknet", "hybrid"}:
        raise ValueError("Unknown learning-to-rank objective")
    if regularization <= 0:
        raise ValueError("Regularization must be positive")
    centre, scale = _standardizer(groups)
    dimension = groups[0].features.shape[1]

    def function(weights: np.ndarray) -> tuple[float, np.ndarray]:
        loss, gradient, _ = _group_losses(
            weights, groups, centre, scale, objective, regularization
        )
        return loss, gradient

    result = minimize(
        function,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"{objective} optimization failed: {result.message}")
    _, _, used = _group_losses(result.x, groups, centre, scale, objective, regularization)
    return {
        "schema_version": "linear-peer-ranker/v1",
        "objective": objective,
        "regularization": regularization,
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "weights": result.x.tolist(),
        "loss": float(result.fun),
        "iterations": int(result.nit),
        "training_groups": len(groups),
        "scoreable_training_groups": used,
    }


def score_linear_ranker(group: RankGroup, model: dict[str, Any]) -> np.ndarray:
    centre = np.asarray(model["centre"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if group.features.shape[1] != len(weights):
        raise ValueError("Ranker and feature dimensions differ")
    return ((group.features - centre) / scale) @ weights


def _huber_location(values: list[float], delta: float = 0.25) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Huber fusion requires finite evidence")
    location = float(np.median(array))
    for _ in range(30):
        residual = array - location
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        updated = float(np.sum(weights * array) / np.sum(weights))
        if abs(updated - location) <= 1e-12:
            break
        location = updated
    return location


def _rank_metrics(items: list[dict[str, Any]], labels: dict[str, int]) -> dict[str, Any]:
    positives = {key for key, value in labels.items() if int(value) == 1}
    if not positives:
        raise ValueError("Anomaly ranking requires at least one positive identity")
    ranked = sorted(items, key=lambda item: (-float(item["score"]), str(item["id"])))
    seen: set[str] = set()
    flags: list[bool] = []
    hits = 0
    average_precision = 0.0
    for position, item in enumerate(ranked, 1):
        reference = item.get("reference")
        hit = reference in positives and reference not in seen
        flags.append(hit)
        if hit:
            hits += 1
            average_precision += hits / position
            seen.add(reference)
    reciprocal_rank = next((1.0 / (index + 1) for index, hit in enumerate(flags) if hit), 0.0)
    top_score = float(ranked[0]["score"]) if ranked else 0.0
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else top_score
    raw = np.asarray([float(item["score"]) for item in ranked], dtype=np.float64)
    if len(raw):
        shifted = raw - raw.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
        entropy /= max(np.log(len(raw)), 1.0)
    else:
        entropy = 1.0
    tied = [item for item in ranked if ranked and np.isclose(float(item["score"]), top_score, atol=1e-12, rtol=0.0)]
    tied_flags = [item.get("reference") in positives for item in tied]

    # Candidate IDs make files reproducible, but they must not improve a
    # scientific result. Average each metric over all orders inside score ties.
    expected_ap_sum = 0.0
    expected_reciprocal_rank = 0.0
    expected_hits_at_2 = 0.0
    cumulative_hits = 0
    rank_offset = 0
    group_start = 0
    first_positive_found = False
    while group_start < len(ranked):
        group_end = group_start + 1
        while group_end < len(ranked) and np.isclose(
            float(ranked[group_end]["score"]),
            float(ranked[group_start]["score"]),
            atol=1e-12,
            rtol=0.0,
        ):
            group_end += 1
        group = ranked[group_start:group_end]
        group_size = len(group)
        positive_in_group = sum(item.get("reference") in positives for item in group)
        if positive_in_group:
            for position in range(1, group_size + 1):
                preceding = (
                    0.0
                    if group_size == 1
                    else (position - 1) * (positive_in_group - 1) / (group_size - 1)
                )
                expected_ap_sum += (
                    positive_in_group
                    / group_size
                    * (cumulative_hits + 1.0 + preceding)
                    / (rank_offset + position)
                )
            if not first_positive_found:
                denominator = comb(group_size, positive_in_group)
                expected_reciprocal_rank = sum(
                    comb(group_size - position, positive_in_group - 1)
                    / denominator
                    / (rank_offset + position)
                    for position in range(1, group_size - positive_in_group + 2)
                )
                first_positive_found = True
        available_at_2 = max(0, min(group_size, 2 - rank_offset))
        if available_at_2:
            expected_hits_at_2 += positive_in_group * available_at_2 / group_size
        cumulative_hits += positive_in_group
        rank_offset += group_size
        group_start = group_end
    return {
        "ap": average_precision / len(positives),
        "ap_tie_expected": expected_ap_sum / len(positives),
        "top1": float(bool(flags and flags[0])),
        "top1_tie_expected": float(np.mean(tied_flags)) if tied_flags else 0.0,
        "top1_tie_best": float(any(tied_flags)),
        "top1_tie_worst": float(bool(tied_flags) and all(tied_flags)),
        "top_tie_count": len(tied),
        "recall_at_2": float(sum(flags[:2]) / len(positives)),
        "recall_at_2_tie_expected": float(expected_hits_at_2 / len(positives)),
        "reciprocal_rank": float(reciprocal_rank),
        "reciprocal_rank_tie_expected": float(expected_reciprocal_rank),
        "positive_count": len(positives),
        "ranked_count": len(ranked),
        "winner": ranked[0]["id"] if ranked else None,
        "winner_reference": ranked[0].get("reference") if ranked else None,
        "winner_support": ranked[0].get("support", 0) if ranked else 0,
        "score_margin": top_score - second_score,
        "score_entropy": entropy,
        "confidence": 0.5 * (top_score - second_score) + 0.5 * (1.0 - entropy),
    }


def _track_references(
    tracks: list[dict[str, Any]], truth: dict[str, Any], route: str
) -> dict[str, tuple[str | None, float]]:
    names = sorted(truth["labels"])
    votes = np.zeros((len(tracks), len(names)), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for member in track["members"]:
            reference = truth["correspondence"].get(
                f"{route}/{member['view_id']}/{member['candidate_id']}"
            )
            if reference in names:
                votes[track_index, names.index(reference)] += 1.0
    result = {track["identity_id"]: (None, 0.0) for track in tracks}
    if not len(tracks) or not names:
        return result
    left, right = linear_sum_assignment(-votes)
    for track_index, name_index in zip(left, right, strict=True):
        support = max(1, len(tracks[track_index]["members"]))
        purity = float(votes[track_index, name_index] / support)
        result[tracks[track_index]["identity_id"]] = (
            names[name_index] if purity > 0.5 else None,
            purity,
        )
    return result


def _candidate_score(
    view_group: dict[str, Any], candidate_index: int, method: str, learned: dict[tuple[str, ...], float], key: tuple[str, ...], feature_names: tuple[str, ...]
) -> float | None:
    if method in METHODS:
        value = view_group["method_ranks"][method][candidate_index]
        return None if value is None else float(value)
    if method == "robust_forward":
        value = view_group["method_ranks"]["forward_02_cov70"][candidate_index]
        return None if value is None else float(value)
    if method == "robust_borda":
        components = (
            "B0",
            "P1_current",
            "P2_consensus",
            "forward_02_cov70",
            "mutual_missing_05",
        )
        values = [view_group["method_ranks"][name][candidate_index] for name in components]
        values = [float(value) for value in values if value is not None]
        return None if not values else float(np.mean(values))
    if method == "soft_support_fusion":
        support_index = feature_names.index("cross_anomaly_support_mean")
        support = float(view_group["features"][candidate_index][support_index])
        components = [
            (0.65, view_group["method_ranks"]["forward_02_cov70"][candidate_index]),
            (0.20, view_group["method_ranks"]["P1_current"][candidate_index]),
            (0.15, float(np.clip(support, 0.0, 1.0))),
        ]
        available = [(weight, float(value)) for weight, value in components if value is not None]
        return float(sum(weight * value for weight, value in available) / sum(weight for weight, _ in available))
    if method == "view_vote_forward":
        value = view_group["method_ranks"]["forward_02_cov70"][candidate_index]
        return None if value is None else float(value) ** 4
    if method == "view_vote_consensus":
        values = [
            view_group["method_ranks"][name][candidate_index]
            for name in ("forward_02_cov70", "P1_current", "P2_consensus")
        ]
        values = [float(value) ** 4 for value in values if value is not None]
        return None if not values else float(np.mean(values))
    if method in LEARNED_METHODS:
        return float(learned[key])
    raise ValueError(f"Unknown scorer: {method}")


def _evaluate_scene_setting(
    scene_payload: dict[str, Any],
    setting: dict[str, Any],
    truth: dict[str, Any],
    learned_scores: dict[tuple[str, ...], float],
    feature_names: tuple[str, ...],
    fold: int,
) -> list[dict[str, Any]]:
    by_view = {group["view"]: group for group in setting["view_groups"]}
    rows = []
    for method in ALL_METHODS:
        if method == "winner_cluster_forward":
            method_tracks = setting["winner_tracks"]["forward"]
        elif method == "winner_cluster_consensus":
            method_tracks = setting["winner_tracks"]["consensus"]
        else:
            method_tracks = setting["tracks"]
        references = _track_references(method_tracks, truth, setting["route"])
        items = []
        for track in method_tracks:
            values = []
            for member in track["members"]:
                if method in {"winner_cluster_forward", "winner_cluster_consensus"}:
                    values.append(float(track["cluster_score"]))
                    continue
                group = by_view[member["view_id"]]
                candidate_index = group["candidate_ids"].index(member["candidate_id"])
                key = (
                    scene_payload["scene"],
                    setting["route"],
                    setting["view_set_id"],
                    member["view_id"],
                    member["candidate_id"],
                    method,
                )
                value = _candidate_score(group, candidate_index, method, learned_scores, key, feature_names)
                if value is not None:
                    values.append(value)
            if not values:
                continue
            robust = method in {
                "robust_forward",
                "robust_borda",
                "soft_support_fusion",
                *LEARNED_METHODS,
            }
            reference, purity = references[track["identity_id"]]
            if method in {"winner_cluster_forward", "winner_cluster_consensus"}:
                fused_score = float(track["cluster_score"])
            elif method in {"view_vote_forward", "view_vote_consensus"}:
                fused_score = float(sum(values) / max(1, setting["view_count"]))
            else:
                fused_score = _huber_location(values) if robust else float(np.mean(values))
            items.append(
                {
                    "id": track["identity_id"],
                    "score": fused_score,
                    "reference": reference,
                    "purity": purity,
                    "support": len(values),
                }
            )
        common = {
            "scene": scene_payload["scene"],
            "asset_group": truth["asset_group"],
            "category": truth.get("category", "unknown"),
            "families": "|".join(sorted(str(value) for value in truth.get("families", []))),
            "route": setting["route"],
            "view_set_id": setting["view_set_id"],
            "view_count": int(setting["view_count"]),
            "primary_schedule": bool(setting["view_set_id"].endswith("primary")),
            "fold": fold,
            "method": method,
        }
        rows.append(common | _rank_metrics(items, truth["labels"]))
    return rows


def _bootstrap_paired(
    frame: pd.DataFrame, baseline: str, samples: int = 2000, seed: int = 32040
) -> pd.DataFrame:
    rows = []
    primary = frame[
        (frame["route"] == "automatic")
        & (frame["view_set_id"] == "three-primary")
    ]
    for method in ALL_METHODS:
        if method == baseline:
            continue
        pivot = primary.pivot(
            index=["scene", "asset_group"],
            columns="method",
            values=[
                "top1_tie_expected",
                "ap_tie_expected",
                "recall_at_2_tie_expected",
                "reciprocal_rank_tie_expected",
            ],
        )
        for measure in (
            "top1_tie_expected",
            "ap_tie_expected",
            "recall_at_2_tie_expected",
            "reciprocal_rank_tie_expected",
        ):
            paired = (pivot[(measure, method)] - pivot[(measure, baseline)]).reset_index(name="difference")
            groups = [group["difference"].to_numpy() for _, group in paired.groupby("asset_group")]
            rng = np.random.default_rng(seed)
            draws = np.asarray(
                [
                    np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))]).mean()
                    for _ in range(samples)
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "measure": measure,
                    "difference": float(paired["difference"].mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "asset_groups": len(groups),
                    "scenes": len(paired),
                }
            )
    return pd.DataFrame(rows)


def _risk_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = frame[
        (frame["route"] == "automatic")
        & (frame["view_set_id"] == "three-primary")
    ]
    for method, group in primary.groupby("method"):
        ordered = group.sort_values(["confidence", "scene"], ascending=[False, True])
        for retained in range(5, len(ordered) + 1, 5):
            current = ordered.iloc[:retained]
            rows.append(
                {
                    "method": method,
                    "coverage": retained / len(ordered),
                    "retained_scenes": retained,
                    "retained_top1": float(current["top1_tie_expected"].mean()),
                    "selective_risk": float(1.0 - current["top1_tie_expected"].mean()),
                    "minimum_confidence": float(current["confidence"].min()),
                }
            )
    return pd.DataFrame(rows)


def _save_figures(
    scene_metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    coefficient_frame: pd.DataFrame,
    figure_root: Path,
) -> list[Path]:
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    key_methods = [
        "B0",
        "P1_current",
        "P2_consensus",
        "forward_02_cov70",
        "view_vote_consensus",
        "winner_cluster_forward",
        "winner_cluster_consensus",
        "learned_ranknet",
    ]

    primary = scene_metrics[
        (scene_metrics["route"] == "automatic")
        & (scene_metrics["view_set_id"] == "three-primary")
    ]
    summary = primary[primary["method"].isin(key_methods)].groupby("method")[[
        "top1_tie_expected",
        "ap_tie_expected",
        "recall_at_2_tie_expected",
        "reciprocal_rank_tie_expected",
    ]].mean()
    summary = summary.rename(
        columns={
            "top1_tie_expected": "top1",
            "ap_tie_expected": "ap",
            "recall_at_2_tie_expected": "recall_at_2",
            "reciprocal_rank_tie_expected": "reciprocal_rank",
        }
    ).sort_values("top1", ascending=False)
    summary.index = [METHOD_LABELS[name] for name in summary.index]
    summary = summary.rename(
        columns={
            "top1": "Tie-expected top-1",
            "ap": "Tie-expected AP",
            "recall_at_2": "Tie-expected recall@2",
            "reciprocal_rank": "Tie-expected MRR",
        }
    )
    axis = summary.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), width=0.82)
    axis.set_title("Automatic three-view development comparison")
    axis.set_ylabel("Mean scene metric")
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = figure_root / "primary_method_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    view = scene_metrics[
        (scene_metrics["route"] == "automatic")
        & (scene_metrics["primary_schedule"])
        & (scene_metrics["method"].isin(key_methods))
    ].groupby(["view_count", "method"])["top1_tie_expected"].mean().unstack()
    view = view.rename(columns=METHOD_LABELS)
    view.plot(marker="o", figsize=(11, 6), ylim=(0, 1))
    plt.title("Tie-expected top-1 across fixed view counts")
    plt.ylabel("Tie-expected top-1 accuracy")
    plt.xlabel("Number of views")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = figure_root / "view_count_top1.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    folded = fold_summary[
        (fold_summary["route"] == "automatic")
        & (fold_summary["view_set_id"] == "three-primary")
        & (fold_summary["method"].isin(key_methods))
    ].pivot(index="fold", columns="method", values="top1_tie_expected")
    folded = folded.rename(columns=METHOD_LABELS)
    folded.plot(marker="o", figsize=(11, 6), ylim=(0, 1))
    plt.title("Grouped-fold stability on the primary route")
    plt.ylabel("Tie-expected top-1 accuracy")
    plt.xlabel("Held-out asset fold")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = figure_root / "fold_stability.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    risk_methods = {"forward_02_cov70", "winner_cluster_forward", "winner_cluster_consensus"}
    for method, group in risk_coverage[risk_coverage["method"].isin(risk_methods)].groupby("method"):
        plt.plot(group["coverage"], group["retained_top1"], label=METHOD_LABELS[method])
    plt.title("Development risk-coverage diagnostic")
    plt.xlabel("Coverage")
    plt.ylabel("Retained tie-expected top-1 accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = figure_root / "risk_coverage.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)

    top = coefficient_frame.reindex(coefficient_frame["weight"].abs().sort_values(ascending=False).index).head(20)
    colours = ["#2f6f9f" if value >= 0 else "#b74d4d" for value in top["weight"]]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"][::-1], top["weight"][::-1], color=colours[::-1])
    plt.title("Largest coefficients in the automatic hybrid ranker diagnostic")
    plt.xlabel("Standardised coefficient")
    plt.tight_layout()
    path = figure_root / "hybrid_coefficients.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(path)
    return paths


def _find_rgb(model_input_root: Path, scene: str, view: str, route: str) -> tuple[Path, dict[str, Any]]:
    rows = read_json(Path(model_input_root) / "scenes" / f"{scene}.json")
    row = next(item for item in rows if item["view"] == view and item["route"] == route)
    return Path(model_input_root) / row["rgb"], row


def _gallery_panel(
    image: np.ndarray,
    object_record: dict[str, Any],
    heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    box = [int(value) for value in object_record["box"]]
    mask = decode_binary_mask(object_record["mask"])
    boxed = image.copy()
    x1, y1, x2, y2 = box
    boxed[max(0, y1 - 2) : min(image.shape[0], y1 + 2), x1:x2] = (220, 45, 55)
    boxed[max(0, y2 - 2) : min(image.shape[0], y2 + 2), x1:x2] = (220, 45, 55)
    boxed[y1:y2, max(0, x1 - 2) : min(image.shape[1], x1 + 2)] = (220, 45, 55)
    boxed[y1:y2, max(0, x2 - 2) : min(image.shape[1], x2 + 2)] = (220, 45, 55)
    overlay = image.astype(np.float64)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.asarray([48, 198, 93])
    masked = np.zeros_like(image)
    masked[mask] = image[mask]
    heat = np.asarray(heatmap, dtype=np.float64)
    if not heat.size or not np.isfinite(heat).all():
        heat = np.zeros((4, 4), dtype=np.float64)
    elif heat.ndim == 1:
        side = int(round(np.sqrt(heat.size)))
        if side * side != heat.size:
            raise ValueError("Anomaly heatmap must use a square patch grid")
        heat = heat.reshape(side, side)
    elif heat.ndim != 2:
        raise ValueError("Anomaly heatmap must be one- or two-dimensional")
    heat = np.asarray(
        Image.fromarray(heat.astype(np.float32), mode="F").resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    )
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
    heat_rgb = plt.get_cmap("magma")(heat)[..., :3] * 255.0
    heat_overlay = 0.55 * image + 0.45 * heat_rgb
    return boxed, overlay.astype(np.uint8), masked, heat_overlay.astype(np.uint8)


def _build_private_galleries(
    scene_metrics: pd.DataFrame,
    feature_root: Path,
    scoring_root: Path,
    model_input_root: Path,
    selected_method: str,
    gallery_root: Path,
) -> list[Path]:
    primary = scene_metrics[
        (scene_metrics["route"] == "automatic")
        & (scene_metrics["view_set_id"] == "three-primary")
        & (scene_metrics["method"] == selected_method)
    ].copy()
    correct = primary[primary["top1_tie_worst"] == 1].sort_values(
        ["winner_support", "confidence", "scene"], ascending=[False, False, True]
    )
    uncertain = primary.sort_values(
        ["top_tie_count", "confidence", "scene"], ascending=[False, True, True]
    )
    failures = primary[primary["top1_tie_best"] == 0].sort_values(["confidence", "scene"], ascending=[False, True])
    choices: list[tuple[str, pd.Series]] = []
    used_assets: set[str] = set()
    for label, frame, limit in (("supported_correct", correct, 4), ("uncertain", uncertain, 4), ("failure", failures, 4)):
        chosen = 0
        for _, row in frame.iterrows():
            asset = str(row["asset_group"])
            if asset in used_assets:
                continue
            choices.append((label, row))
            used_assets.add(asset)
            chosen += 1
            if chosen == limit:
                break

    gallery_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (case_type, metric_row) in enumerate(choices, 1):
        scene = str(metric_row["scene"])
        feature = read_json(Path(feature_root) / "scenes" / f"{scene}.json")
        setting = next(
            item for item in feature["groups"]
            if item["route"] == "automatic" and item["view_set_id"] == "three-primary"
        )
        if selected_method == "winner_cluster_forward":
            method_tracks = setting["winner_tracks"]["forward"]
        elif selected_method == "winner_cluster_consensus":
            method_tracks = setting["winner_tracks"]["consensus"]
        else:
            method_tracks = setting["tracks"]
        winner = next(item for item in method_tracks if item["identity_id"] == metric_row["winner"])
        score_rows = read_json(Path(scoring_root) / "scenes" / f"{scene}.json")
        score_lookup = {(item["route"], item["view"]): item for item in score_rows}
        figure, axes = plt.subplots(3, 4, figsize=(15, 10))
        for row_index, view in enumerate(setting["views"]):
            member = next((item for item in winner["members"] if item["view_id"] == view), None)
            rgb_path, model_row = _find_rgb(model_input_root, scene, view, "automatic")
            image = np.asarray(Image.open(rgb_path).convert("RGB"))
            if member is None:
                panels = (image, np.zeros_like(image), np.zeros_like(image), np.zeros_like(image))
            else:
                object_record = next(item for item in model_row["objects"] if item["candidate_id"] == member["candidate_id"])
                score_object = next(
                    item for item in score_lookup[("automatic", view)]["objects"]
                    if item["candidate_id"] == member["candidate_id"]
                )
                heatmap_value = score_object.get("heatmaps", {}).get("forward_02_cov70")
                heatmap = (
                    np.asarray(heatmap_value, dtype=np.float64)
                    if heatmap_value is not None
                    else np.empty(0, dtype=np.float64)
                )
                panels = _gallery_panel(image, object_record, heatmap)
            for column, panel in enumerate(panels):
                axes[row_index, column].imshow(panel)
                axes[row_index, column].axis("off")
                if member is None and column > 0:
                    axes[row_index, column].text(
                        0.5,
                        0.5,
                        "No linked winner\nin this view",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        transform=axes[row_index, column].transAxes,
                    )
                if row_index == 0:
                    axes[row_index, column].set_title(("Selected box", "SAM overlay", "Masked object", "Anomaly evidence")[column])
                if column == 0:
                    axes[row_index, column].set_ylabel(view)
        figure.suptitle(
            f"{case_type.replace('_', ' ').title()} | {scene} | "
            f"tie-expected top-1={metric_row['top1_tie_expected']:.2f} | "
            f"view support={int(metric_row['winner_support'])}/{len(setting['views'])} | "
            f"top tie={int(metric_row['top_tie_count'])} | score gap={metric_row['score_margin']:.4f}"
            + (" | representative tied candidate" if metric_row["top_tie_count"] > 1 else ""),
            fontsize=13,
        )
        plt.tight_layout()
        path = gallery_root / f"example_{index:02d}_{case_type}.png"
        plt.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    atomic_json(
        gallery_root / "gallery_manifest.json",
        {
            "selected_method": selected_method,
            "selection": "deterministic out-of-fold definite-correct cases prioritised by cross-view support, ambiguous cases and definite failures, all with distinct asset groups",
            "files": [path.name for path in outputs],
            "private": True,
        },
    )
    return outputs


def evaluate_metric_candidates(
    feature_root: Path,
    scoring_root: Path,
    evaluator_root: Path,
    model_input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run grouped OOF evaluation, freeze a candidate and generate evidence."""

    feature_root = Path(feature_root)
    scoring_root = Path(scoring_root)
    evaluator_root = Path(evaluator_root)
    model_input_root = Path(model_input_root)
    output_root = Path(output_root)
    groups, truths, feature_names, evaluator_lock = load_rank_groups(feature_root, evaluator_root)
    scoring_lock = verify_seal(scoring_root, name="prediction_lock.json")
    model_lock = verify_seal(model_input_root)
    folds = _balanced_asset_folds(truths)
    protocol = {
        "stage": STAGE,
        "routes": ["automatic", "supplied"],
        "automatic_route_is_deployable": True,
        "supplied_route_is_diagnostic": True,
        "view_sets": {key: list(value) for key, value in VIEW_SETS.items()},
        "methods": list(ALL_METHODS),
        "learning": {"folds": 5, "group": "asset_group", "regularization": 1.0, "primary_training_schedules_only": True},
        "selection": {
            "route": "automatic",
            "view_set_id": "three-primary",
            "rule": "simplest method within one grouped-fold standard error of the best tie-expected top-1, subject to one-standard-error non-inferiority for tie-expected AP and recall at two",
            "tie_handling": "average all scientific metrics over equal-score permutations",
            "complexity_order": list(SIMPLE_METHOD_ORDER),
        },
    }
    scientific_sources = {
        name: inspect.getsource(value)
        for name, value in {
            "group_losses": _group_losses,
            "fit_linear_ranker": fit_linear_ranker,
            "score_linear_ranker": score_linear_ranker,
            "candidate_score": _candidate_score,
            "huber_location": _huber_location,
            "rank_metrics": _rank_metrics,
            "track_references": _track_references,
            "evaluate_scene_setting": _evaluate_scene_setting,
        }.items()
    }
    binding = {
        "protocol": protocol,
        "feature_lock_sha256": digest_file(feature_root / "feature_lock.json"),
        "scoring_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "evaluator_lock_sha256": digest_file(evaluator_root / "artifact_lock.json"),
        "model_input_lock_sha256": digest_file(model_input_root / "artifact_lock.json"),
        "folds": folds,
        "feature_names": list(feature_names),
        "scientific_implementation_sha256": sha256(canonical(scientific_sources)).hexdigest(),
        "feature_implementation_sha256": digest_file(Path(feature_module.__file__)),
    }
    binding_sha256 = sha256(canonical(binding)).hexdigest()
    atomic_json(output_root / "evaluation_binding.json", binding | {"binding_sha256": binding_sha256})
    atomic_json(output_root / "run_status.json", {"status": "running"})

    checkpoint_root = output_root / "fold-models"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    learned_scores: dict[tuple[str, ...], float] = {}
    fold_training_rows = []
    final_models: dict[str, Any] = {}
    for route in ("automatic", "supplied"):
        route_groups = [group for group in groups if group.route == route]
        training_pool = [group for group in route_groups if group.view_set_id.endswith("primary")]
        for method in LEARNED_METHODS:
            objective = method.removeprefix("learned_")
            for fold in range(5):
                held_out = [group for group in route_groups if folds[group.asset_group] == fold]
                train = [group for group in training_pool if folds[group.asset_group] != fold]
                checkpoint = checkpoint_root / f"{route}-{method}-fold-{fold}.json"
                if checkpoint.is_file():
                    record = read_json(checkpoint)
                    if record.get("binding_sha256") != binding_sha256:
                        raise ValueError("A fold checkpoint belongs to another run")
                    model = record["model"]
                    resumed = True
                else:
                    model = fit_linear_ranker(train, objective=objective, regularization=1.0)
                    atomic_json(
                        checkpoint,
                        {"binding_sha256": binding_sha256, "route": route, "method": method, "fold": fold, "model": model},
                    )
                    resumed = False
                for group in held_out:
                    values = score_linear_ranker(group, model)
                    for candidate, value in zip(group.candidate_ids, values, strict=True):
                        learned_scores[(group.scene, route, group.view_set_id, group.view, candidate, method)] = float(value)
                fold_training_rows.append(
                    {"route": route, "method": method, "fold": fold, "train_groups": len(train), "held_out_groups": len(held_out), "loss": model["loss"], "iterations": model["iterations"], "checkpoint_resumed": resumed}
                )
            final_models[f"{route}/{method}"] = fit_linear_ranker(training_pool, objective=objective, regularization=1.0)

    private_root = output_root / "private"
    shareable_root = output_root / "shareable"
    private_root.mkdir(parents=True, exist_ok=True)
    shareable_root.mkdir(parents=True, exist_ok=True)
    atomic_json(private_root / "final_models.json", {"feature_names": list(feature_names), "models": final_models})
    pd.DataFrame(fold_training_rows).to_csv(private_root / "fold_training.csv", index=False)

    scene_rows = []
    for scene in sorted(truths):
        payload = read_json(feature_root / "scenes" / f"{scene}.json")
        truth = truths[scene]
        for setting in payload["groups"]:
            scene_rows.extend(
                _evaluate_scene_setting(payload, setting, truth, learned_scores, feature_names, folds[truth["asset_group"]])
            )
    scene_metrics = pd.DataFrame(scene_rows)
    scene_metrics.to_csv(private_root / "oof_scene_metrics.csv", index=False)

    summary_keys = ["route", "view_set_id", "view_count", "primary_schedule", "method"]
    summary = scene_metrics.groupby(summary_keys).agg(
        scenes=("scene", "size"),
        asset_groups=("asset_group", "nunique"),
        top1=("top1", "mean"),
        ap=("ap", "mean"),
        recall_at_2=("recall_at_2", "mean"),
        reciprocal_rank=("reciprocal_rank", "mean"),
        top1_tie_expected=("top1_tie_expected", "mean"),
        ap_tie_expected=("ap_tie_expected", "mean"),
        recall_at_2_tie_expected=("recall_at_2_tie_expected", "mean"),
        reciprocal_rank_tie_expected=("reciprocal_rank_tie_expected", "mean"),
        top1_tie_best=("top1_tie_best", "mean"),
        top1_tie_worst=("top1_tie_worst", "mean"),
        mean_top_tie_count=("top_tie_count", "mean"),
        top_tie_rate=("top_tie_count", lambda values: float((values > 1).mean())),
        unique_top_rate=("top_tie_count", lambda values: float((values == 1).mean())),
        mean_margin=("score_margin", "mean"),
        mean_entropy=("score_entropy", "mean"),
    ).reset_index()
    summary.to_csv(shareable_root / "method_summary.csv", index=False)

    schedule_level = scene_metrics.groupby(
        ["route", "view_count", "method", "view_set_id"]
    ).agg(
        scenes=("scene", "nunique"),
        top1=("top1", "mean"),
        top1_tie_expected=("top1_tie_expected", "mean"),
        ap=("ap", "mean"),
        ap_tie_expected=("ap_tie_expected", "mean"),
        recall_at_2=("recall_at_2", "mean"),
        recall_at_2_tie_expected=("recall_at_2_tie_expected", "mean"),
        reciprocal_rank=("reciprocal_rank", "mean"),
        reciprocal_rank_tie_expected=("reciprocal_rank_tie_expected", "mean"),
    ).reset_index()
    sensitivity = schedule_level.groupby(["route", "view_count", "method"]).agg(
        schedules=("view_set_id", "nunique"),
        scenes=("scenes", "max"),
        top1=("top1", "mean"),
        top1_tie_expected=("top1_tie_expected", "mean"),
        ap=("ap", "mean"),
        ap_tie_expected=("ap_tie_expected", "mean"),
        recall_at_2=("recall_at_2", "mean"),
        recall_at_2_tie_expected=("recall_at_2_tie_expected", "mean"),
        reciprocal_rank=("reciprocal_rank", "mean"),
        reciprocal_rank_tie_expected=("reciprocal_rank_tie_expected", "mean"),
        schedule_top1_std=("top1_tie_expected", "std"),
    ).reset_index()
    sensitivity["schedule_top1_std"] = sensitivity["schedule_top1_std"].fillna(0.0)
    sensitivity.to_csv(shareable_root / "view_schedule_sensitivity.csv", index=False)

    fold_summary = scene_metrics.groupby(["route", "view_set_id", "view_count", "method", "fold"]).agg(
        scenes=("scene", "size"), top1=("top1", "mean"), top1_tie_expected=("top1_tie_expected", "mean"), ap=("ap", "mean"), ap_tie_expected=("ap_tie_expected", "mean"), recall_at_2=("recall_at_2", "mean"), recall_at_2_tie_expected=("recall_at_2_tie_expected", "mean"), reciprocal_rank=("reciprocal_rank", "mean"), reciprocal_rank_tie_expected=("reciprocal_rank_tie_expected", "mean")
    ).reset_index()
    fold_summary.to_csv(shareable_root / "fold_summary.csv", index=False)

    subgroup = scene_metrics[
        (scene_metrics["route"] == "automatic") & (scene_metrics["primary_schedule"])
    ].groupby(["category", "view_count", "method"]).agg(
        scenes=("scene", "size"), asset_groups=("asset_group", "nunique"), top1=("top1", "mean"), top1_tie_expected=("top1_tie_expected", "mean"), ap=("ap", "mean"), ap_tie_expected=("ap_tie_expected", "mean"), recall_at_2=("recall_at_2", "mean"), recall_at_2_tie_expected=("recall_at_2_tie_expected", "mean")
    ).reset_index()
    subgroup.to_csv(shareable_root / "category_summary.csv", index=False)

    paired = _bootstrap_paired(scene_metrics, "forward_02_cov70")
    paired.to_csv(shareable_root / "paired_asset_intervals.csv", index=False)
    risk = _risk_coverage(scene_metrics)
    risk.to_csv(shareable_root / "risk_coverage.csv", index=False)

    primary = summary[(summary["route"] == "automatic") & (summary["view_set_id"] == "three-primary")].set_index("method")
    baseline = primary.loc["forward_02_cov70"]
    fold_view = fold_summary[
        (fold_summary["route"] == "automatic")
        & (fold_summary["view_set_id"] == "three-primary")
    ]
    fold_matrix = fold_view.pivot(
        index="fold", columns="method", values="top1_tie_expected"
    )
    best_observed = max(
        ALL_METHODS,
        key=lambda method: (
            float(primary.loc[method, "top1_tie_expected"]),
            float(primary.loc[method, "reciprocal_rank_tie_expected"]),
        ),
    )
    best_fold_values = fold_matrix[best_observed].to_numpy(dtype=np.float64)
    best_standard_error = (
        float(np.std(best_fold_values, ddof=1) / sqrt(len(best_fold_values)))
        if len(best_fold_values) > 1
        else 0.0
    )
    one_se_floor = float(primary.loc[best_observed, "top1_tie_expected"] - best_standard_error)
    baseline_ap_folds = fold_view[
        fold_view["method"] == "forward_02_cov70"
    ]["ap_tie_expected"].to_numpy(dtype=np.float64)
    baseline_recall_folds = fold_view[
        fold_view["method"] == "forward_02_cov70"
    ]["recall_at_2_tie_expected"].to_numpy(dtype=np.float64)
    baseline_ap_se = float(np.std(baseline_ap_folds, ddof=1) / sqrt(len(baseline_ap_folds)))
    baseline_recall_se = float(np.std(baseline_recall_folds, ddof=1) / sqrt(len(baseline_recall_folds)))
    candidates = []
    for method in ALL_METHODS:
        if method == "forward_02_cov70":
            continue
        row = primary.loc[method]
        candidate_fold = fold_matrix[method]
        baseline_fold = fold_matrix["forward_02_cov70"]
        folds_improved = int((candidate_fold > baseline_fold).sum())
        within_one_se = bool(row["top1_tie_expected"] >= one_se_floor - 1e-12)
        ap_noninferior = bool(
            row["ap_tie_expected"] >= baseline["ap_tie_expected"] - baseline_ap_se
        )
        recall_noninferior = bool(
            row["recall_at_2_tie_expected"]
            >= baseline["recall_at_2_tie_expected"] - baseline_recall_se
        )
        candidates.append(
            {
                "method": method,
                "deterministic_top1": float(row["top1"]),
                "top1": float(row["top1_tie_expected"]),
                "ap": float(row["ap_tie_expected"]),
                "recall_at_2": float(row["recall_at_2_tie_expected"]),
                "reciprocal_rank": float(row["reciprocal_rank_tie_expected"]),
                "top1_difference": float(row["top1_tie_expected"] - baseline["top1_tie_expected"]),
                "folds_improved": folds_improved,
                "within_one_standard_error": within_one_se,
                "ap_noninferior": ap_noninferior,
                "recall_at_2_noninferior": recall_noninferior,
            }
        )
    candidates = sorted(
        candidates,
        key=lambda item: (-item["top1"], -item["reciprocal_rank"], item["method"]),
    )
    eligible = {
        method
        for method in ALL_METHODS
        if float(primary.loc[method, "top1_tie_expected"]) >= one_se_floor - 1e-12
        and float(primary.loc[method, "ap_tie_expected"])
        >= float(baseline["ap_tie_expected"] - baseline_ap_se)
        and float(primary.loc[method, "recall_at_2_tie_expected"])
        >= float(baseline["recall_at_2_tie_expected"] - baseline_recall_se)
    }
    selected = next(
        (method for method in SIMPLE_METHOD_ORDER if method in eligible),
        "forward_02_cov70",
    )
    decision = {
        "status": "selected_for_frozen_confirmation" if selected != "forward_02_cov70" else "retain_current_forward_scorer",
        "selected_method": selected,
        "best_observed_method": best_observed,
        "primary_route": "automatic",
        "primary_view_set": "three-primary",
        "development_only": True,
        "protected_calibration_used": False,
        "protected_internal_lock_used": False,
        "decision_rule": protocol["selection"],
        "best_grouped_fold_standard_error": best_standard_error,
        "one_standard_error_floor": one_se_floor,
        "baseline_ap_standard_error": baseline_ap_se,
        "baseline_recall_at_2_standard_error": baseline_recall_se,
        "baseline": {
            "deterministic_top1": float(baseline["top1"]),
            "top1": float(baseline["top1_tie_expected"]),
            "ap": float(baseline["ap_tie_expected"]),
            "recall_at_2": float(baseline["recall_at_2_tie_expected"]),
            "reciprocal_rank": float(baseline["reciprocal_rank_tie_expected"]),
        },
        "candidates": candidates,
    }
    atomic_json(shareable_root / "development_decision.json", decision)

    method_registry = pd.DataFrame(
        [{"method": method, "description": METHOD_REGISTRY[method], "learned": method in LEARNED_METHODS, "deployable_route": "automatic"} for method in ALL_METHODS]
    )
    method_registry.to_csv(shareable_root / "method_registry.csv", index=False)
    pd.DataFrame({"feature_index": range(len(feature_names)), "feature": feature_names}).to_csv(shareable_root / "feature_schema.csv", index=False)

    final_model = final_models[f"automatic/{selected}"] if selected in LEARNED_METHODS else final_models["automatic/learned_hybrid"]
    coefficient_frame = pd.DataFrame({"feature": feature_names, "weight": final_model["weights"]})
    coefficient_frame.to_csv(shareable_root / "hybrid_coefficients.csv", index=False)
    figure_paths = _save_figures(scene_metrics, fold_summary, risk, coefficient_frame, shareable_root / "figures")

    gallery_paths = _build_private_galleries(scene_metrics, feature_root, scoring_root, model_input_root, selected, private_root / "reference_gallery")
    selected_primary = scene_metrics[
        (scene_metrics["route"] == "automatic")
        & (scene_metrics["view_set_id"] == "three-primary")
        & (scene_metrics["method"] == selected)
    ]
    rankable = selected_primary["reciprocal_rank_tie_expected"] > 0
    failure_waterfall = pd.DataFrame(
        [
            {"stage": "scenes_evaluated", "count": int(selected_primary["scene"].nunique())},
            {"stage": "anomaly_not_rankable", "count": int((selected_primary["reciprocal_rank_tie_expected"] == 0).sum())},
            {"stage": "unique_correct_top1", "count": int((rankable & (selected_primary["top_tie_count"] == 1) & (selected_primary["top1_tie_worst"] == 1)).sum())},
            {"stage": "unique_wrong_top1", "count": int((rankable & (selected_primary["top_tie_count"] == 1) & (selected_primary["top1_tie_best"] == 0)).sum())},
            {"stage": "tied_top_with_anomaly", "count": int((rankable & (selected_primary["top_tie_count"] > 1) & (selected_primary["top1_tie_best"] == 1)).sum())},
            {"stage": "tied_top_without_anomaly", "count": int((rankable & (selected_primary["top_tie_count"] > 1) & (selected_primary["top1_tie_best"] == 0)).sum())},
        ]
    )
    failure_waterfall.to_csv(shareable_root / "failure_waterfall.csv", index=False)

    atomic_json(
        shareable_root / "evaluation_receipt.json",
        {
            "binding_sha256": binding_sha256,
            "scenes": len(truths),
            "asset_groups": len(set(folds)),
            "scene_metric_rows": len(scene_metrics),
            "selected_method": selected,
            "figure_count": len(figure_paths),
            "private_gallery_count": len(gallery_paths),
            "clearml_status": "pending_post_science",
        },
    )
    atomic_json(output_root / "run_status.json", {"status": "scientific_complete"})
    scientific_paths: list[Path] = [output_root / "evaluation_binding.json", output_root / "run_status.json", private_root / "final_models.json", private_root / "fold_training.csv", private_root / "oof_scene_metrics.csv"]
    scientific_paths.extend(sorted(checkpoint_root.glob("*.json")))
    scientific_paths.extend(sorted(shareable_root.rglob("*")))
    scientific_paths = [path for path in scientific_paths if path.is_file()]
    lock = seal(output_root, scientific_paths, metadata={"stage": STAGE, "selected_method": selected, "development_only": True})
    return {
        "scenes": len(truths),
        "asset_groups": len(set(folds)),
        "methods": len(ALL_METHODS),
        "selected_method": selected,
        "status": decision["status"],
        "artifact_lock_sha256": digest_file(output_root / "artifact_lock.json"),
        "locked_files": len(lock["files"]),
    }


__all__ = [
    "ALL_METHODS",
    "FIXED_METHODS",
    "LEARNED_METHODS",
    "METHOD_REGISTRY",
    "RankGroup",
    "evaluate_metric_candidates",
    "fit_linear_ranker",
    "load_rank_groups",
    "score_linear_ranker",
]
