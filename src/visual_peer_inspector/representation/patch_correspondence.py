"""Foreground patch correspondence for cross-view object identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from visual_peer_inspector.representation.crops import PreparedCrop
from visual_peer_inspector.representation.cross_view_identity import cosine_cost


Node = tuple[str, str]
Solver = Literal["anchor", "consensus"]
PatchMetric = Literal["reciprocal", "sinkhorn"]


@dataclass(frozen=True)
class PatchAssociationConfiguration:
    """One fixed patch correspondence and track-building configuration."""

    name: str
    patch_metric: PatchMetric
    global_weight: float
    maximum_cost: float
    solver: Solver
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 30
    dustbin_similarity: float = 0.35

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("Configuration names must contain letters, numbers or underscores")
        if not 0.0 <= float(self.global_weight) <= 1.0:
            raise ValueError("global_weight must be in [0, 1]")
        if self.patch_metric not in {"reciprocal", "sinkhorn"}:
            raise ValueError("patch_metric must be reciprocal or sinkhorn")
        if not 0.0 < float(self.maximum_cost) <= 2.0:
            raise ValueError("maximum_cost must be in (0, 2]")
        if self.solver not in {"anchor", "consensus"}:
            raise ValueError("solver must be anchor or consensus")
        if float(self.sinkhorn_epsilon) <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if int(self.sinkhorn_iterations) <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if not -1.0 <= float(self.dustbin_similarity) <= 1.0:
            raise ValueError("dustbin_similarity must be in [-1, 1]")


PATCH_CONFIGURATIONS = (
    PatchAssociationConfiguration("reciprocal_hungarian_035", "reciprocal", 0.50, 0.35, "anchor"),
    PatchAssociationConfiguration("sinkhorn_hungarian_035", "sinkhorn", 0.50, 0.35, "anchor"),
    PatchAssociationConfiguration("sinkhorn_consensus_035", "sinkhorn", 0.50, 0.35, "consensus"),
)


def _unit_rows(values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Patch tokens must be a finite two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Patch tokens must have non-zero norm")
    return array / norms


def foreground_patch_evidence(
    patch_features: np.ndarray,
    crop: PreparedCrop,
    *,
    minimum_coverage: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Select normalised DINO tokens and their mask-occupancy weights."""

    patches = _unit_rows(patch_features)
    grid = int(round(np.sqrt(patches.shape[0])))
    if grid * grid != patches.shape[0]:
        raise ValueError("Patch tokens must form a square grid")
    patch_size = crop.foreground.shape[0] // grid
    coverage = crop.foreground.reshape(
        grid,
        patch_size,
        grid,
        patch_size,
    ).mean(axis=(1, 3)).reshape(-1)
    validity = coverage >= minimum_coverage
    fallback = False
    if not validity.any():
        validity[int(np.argmax(coverage))] = True
        fallback = True
    weights = np.asarray(coverage[validity], dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()
    return np.ascontiguousarray(patches[validity]), weights, fallback


def foreground_patch_tokens(
    patch_features: np.ndarray,
    crop: PreparedCrop,
    *,
    minimum_coverage: float = 0.50,
) -> tuple[np.ndarray, bool]:
    """Compatibility wrapper returning foreground tokens and fallback state."""

    tokens, _weights, fallback = foreground_patch_evidence(
        patch_features,
        crop,
        minimum_coverage=minimum_coverage,
    )
    return tokens, fallback


def symmetric_patch_distance(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    retained_fraction: float,
) -> float:
    """Compare two foreground token sets using bidirectional best matches."""

    if not 0.0 < retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be in (0, 1]")
    left_array = _unit_rows(left)
    right_array = _unit_rows(right)
    similarities = np.clip(left_array @ right_array.T, -1.0, 1.0)

    def retained_mean(values: np.ndarray) -> float:
        count = max(1, int(np.ceil(values.size * retained_fraction)))
        return float(np.partition(values, values.size - count)[-count:].mean())

    forward = retained_mean(similarities.max(axis=1))
    backward = retained_mean(similarities.max(axis=0))
    return float(1.0 - np.clip((forward + backward) / 2.0, -1.0, 1.0))


def reciprocal_patch_distance(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> dict[str, float]:
    """Score mutual-nearest foreground patches without many-to-one matches."""

    left_array = _unit_rows(left)
    right_array = _unit_rows(right)
    similarities = np.clip(left_array @ right_array.T, -1.0, 1.0)
    left_best = similarities.argmax(axis=1)
    right_best = similarities.argmax(axis=0)
    pairs = [(index, int(other)) for index, other in enumerate(left_best) if right_best[int(other)] == index]
    if pairs:
        values = np.asarray([similarities[a, b] for a, b in pairs], dtype=np.float64)
        matched_similarity = float(np.median(values))
    else:
        matched_similarity = -1.0
    coverage = float(2 * len(pairs) / (len(left_array) + len(right_array)))
    distance = 0.70 * (1.0 - matched_similarity) + 0.30 * (1.0 - coverage)
    return {
        "patch_distance": float(distance),
        "patch_match_fraction": coverage,
        "patch_entropy": 0.0,
        "patch_pair_count": float(len(pairs)),
    }


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(reduced, axis=axis)


def sinkhorn_patch_distance(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    left_weights: Sequence[float] | None = None,
    right_weights: Sequence[float] | None = None,
    epsilon: float = 0.05,
    iterations: int = 30,
    dustbin_similarity: float = 0.35,
) -> dict[str, float]:
    """Return a dustbin-aware soft assignment distance between patch sets."""

    left_array = _unit_rows(left)
    right_array = _unit_rows(right)
    if epsilon <= 0.0 or iterations <= 0:
        raise ValueError("Sinkhorn epsilon and iterations must be positive")
    similarities = np.clip(left_array @ right_array.T, -1.0, 1.0).astype(np.float64)
    n, m = similarities.shape
    left_mass = np.ones(n, dtype=np.float64) if left_weights is None else np.asarray(left_weights, dtype=np.float64)
    right_mass = np.ones(m, dtype=np.float64) if right_weights is None else np.asarray(right_weights, dtype=np.float64)
    if left_mass.shape != (n,) or right_mass.shape != (m,):
        raise ValueError("Patch weights must match their token counts")
    if np.any(left_mass <= 0) or np.any(right_mass <= 0) or not np.isfinite(left_mass).all() or not np.isfinite(right_mass).all():
        raise ValueError("Patch weights must be finite and positive")
    left_mass /= left_mass.sum()
    right_mass /= right_mass.sum()

    scores = np.full((n + 1, m + 1), float(dustbin_similarity), dtype=np.float64)
    scores[:n, :m] = similarities
    # Real patches share half the total mass. The other half is slack that can
    # flow through the dustbin row or column when the two objects do not align.
    mu = np.concatenate((0.5 * left_mass, np.asarray([0.5])))
    nu = np.concatenate((0.5 * right_mass, np.asarray([0.5])))
    log_kernel = scores / float(epsilon)
    log_mu = np.log(mu)
    log_nu = np.log(nu)
    u = np.zeros_like(log_mu)
    v = np.zeros_like(log_nu)
    for _ in range(int(iterations)):
        u = log_mu - _logsumexp(log_kernel + v[None, :], axis=1)
        v = log_nu - _logsumexp(log_kernel + u[:, None], axis=0)
    transport = np.exp(log_kernel + u[:, None] + v[None, :])
    real = transport[:n, :m]
    matched_mass = float(real.sum())
    matched_fraction = float(np.clip(matched_mass / 0.5, 0.0, 1.0))
    if matched_mass > 1e-12:
        matched_similarity = float((real * similarities).sum() / matched_mass)
        probabilities = real[real > 0] / matched_mass
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        maximum_entropy = math.log(max(2, n * m))
        normalized_entropy = float(np.clip(entropy / maximum_entropy, 0.0, 1.0))
    else:
        matched_similarity = -1.0
        normalized_entropy = 1.0
    distance = 0.70 * (1.0 - matched_similarity) + 0.20 * (1.0 - matched_fraction) + 0.10 * normalized_entropy
    return {
        "patch_distance": float(distance),
        "patch_match_fraction": matched_fraction,
        "patch_entropy": normalized_entropy,
        "patch_pair_count": float(n * m),
    }


def correspondence_cost(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    configuration: PatchAssociationConfiguration,
) -> dict[str, float]:
    """Return global, patch and combined distances for one object pair."""

    global_distance = cosine_cost(left["global_embedding"], right["global_embedding"])
    if configuration.patch_metric == "reciprocal":
        patch = reciprocal_patch_distance(left["patch_tokens"], right["patch_tokens"])
    else:
        patch = sinkhorn_patch_distance(
            left["patch_tokens"],
            right["patch_tokens"],
            left_weights=left.get("patch_weights"),
            right_weights=right.get("patch_weights"),
            epsilon=configuration.sinkhorn_epsilon,
            iterations=configuration.sinkhorn_iterations,
            dustbin_similarity=configuration.dustbin_similarity,
        )
    patch_distance = patch["patch_distance"]
    combined = (
        configuration.global_weight * global_distance
        + (1.0 - configuration.global_weight) * patch_distance
    )
    return patch | {
        "global_distance": float(global_distance),
        "combined_cost": float(combined),
    }


def _normalise_views(views: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    prepared = []
    seen_views: set[str] = set()
    dimensions: set[int] = set()
    for raw_view in views:
        view_id = str(raw_view["view"])
        if not view_id or view_id in seen_views:
            raise ValueError("View identifiers must be unique and non-empty")
        seen_views.add(view_id)
        seen_candidates: set[str] = set()
        objects = []
        for raw_object in raw_view["objects"]:
            candidate_id = str(raw_object["candidate_id"])
            if not candidate_id or candidate_id in seen_candidates:
                raise ValueError("Candidate identifiers must be unique within a view")
            seen_candidates.add(candidate_id)
            global_embedding = np.asarray(raw_object["global_embedding"], dtype=np.float32)
            if global_embedding.ndim != 1 or not global_embedding.size:
                raise ValueError("Global embeddings must be one-dimensional")
            if not np.isfinite(global_embedding).all() or np.linalg.norm(global_embedding) <= 1e-12:
                raise ValueError("Global embeddings must be finite and non-zero")
            dimensions.add(int(global_embedding.size))
            objects.append(
                {
                    "candidate_id": candidate_id,
                    "global_embedding": global_embedding,
                    "patch_tokens": _unit_rows(raw_object["patch_tokens"]),
                    "patch_weights": raw_object.get("patch_weights"),
                }
            )
        prepared.append({"view": view_id, "objects": tuple(sorted(objects, key=lambda item: item["candidate_id"]))})
    if len(prepared) < 2:
        raise ValueError("Patch association requires at least two views")
    if len(dimensions) > 1:
        raise ValueError("All global embeddings must have the same dimension")
    return tuple(sorted(prepared, key=lambda item: item["view"]))


def _pairwise_edges(
    views: Sequence[Mapping[str, Any]],
    configuration: PatchAssociationConfiguration,
) -> dict[frozenset[Node], dict[str, float]]:
    edges: dict[frozenset[Node], dict[str, float]] = {}
    for left, right in combinations(views, 2):
        if not left["objects"] or not right["objects"]:
            continue
        costs = np.empty((len(left["objects"]), len(right["objects"])), dtype=np.float64)
        details: dict[tuple[int, int], dict[str, float]] = {}
        for left_index, left_object in enumerate(left["objects"]):
            for right_index, right_object in enumerate(right["objects"]):
                record = correspondence_cost(left_object, right_object, configuration)
                costs[left_index, right_index] = record["combined_cost"]
                details[(left_index, right_index)] = record
        size = len(left["objects"])
        padded = np.full((size, len(right["objects"]) + size), 10.0, dtype=np.float64)
        padded[:, : len(right["objects"])] = costs
        for index in range(size):
            padded[index, len(right["objects"]) + index] = configuration.maximum_cost
        rows, columns = linear_sum_assignment(padded)
        for left_index, right_index in zip(rows, columns, strict=True):
            if right_index >= len(right["objects"]):
                continue
            record = details[(int(left_index), int(right_index))]
            if record["combined_cost"] > configuration.maximum_cost:
                continue
            left_node = (str(left["view"]), str(left["objects"][left_index]["candidate_id"]))
            right_node = (str(right["view"]), str(right["objects"][right_index]["candidate_id"]))
            alternatives = [configuration.maximum_cost]
            alternatives.extend(float(costs[left_index, value]) for value in range(costs.shape[1]) if value != right_index)
            record = record | {"margin": float(min(alternatives) - record["combined_cost"])}
            edges[frozenset((left_node, right_node))] = record
    return edges


def _cycle_support(edges: Mapping[frozenset[Node], Mapping[str, float]]) -> dict[frozenset[Node], int]:
    neighbours: dict[Node, set[Node]] = {}
    for edge in edges:
        left, right = sorted(edge)
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    return {
        edge: sum(node[0] not in {left[0], right[0]} for node in neighbours.get(left, set()) & neighbours.get(right, set()))
        for edge in edges
        for left, right in [sorted(edge)]
    }


def _tracks_from_edges(
    views: Sequence[Mapping[str, Any]],
    edges: Mapping[frozenset[Node], Mapping[str, float]],
    configuration: PatchAssociationConfiguration,
) -> tuple[list[list[Node]], list[dict[str, Any]], dict[str, float]]:
    nodes = sorted((str(view["view"]), str(item["candidate_id"])) for view in views for item in view["objects"])
    parents = {node: node for node in nodes}
    members = {node: {node} for node in nodes}

    def find(node: Node) -> Node:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def merge(left: Node, right: Node) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if {item[0] for item in members[left_root]} & {item[0] for item in members[right_root]}:
            return False
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        members[left_root] |= members.pop(right_root)
        return True

    support = _cycle_support(edges)
    anchor = min(views, key=lambda view: (-len(view["objects"]), str(view["view"])))
    anchor_view = str(anchor["view"])
    candidates = []
    for edge, record in edges.items():
        left, right = sorted(edge)
        if configuration.solver == "anchor" and anchor_view not in {left[0], right[0]}:
            continue
        priority = (-support.get(edge, 0), -record["margin"], record["combined_cost"], left, right)
        candidates.append((priority, left, right))
    accepted = 0
    accepted_cycle_supported = 0
    accepted_links = []
    for _priority, left, right in sorted(candidates, key=lambda item: item[0]):
        if merge(left, right):
            accepted += 1
            edge = frozenset((left, right))
            cycle_count = support.get(edge, 0)
            accepted_cycle_supported += int(cycle_count > 0)
            record = edges[edge]
            accepted_links.append(
                {
                    "left": {"view_id": left[0], "candidate_id": left[1]},
                    "right": {"view_id": right[0], "candidate_id": right[1]},
                    "combined_cost": float(record["combined_cost"]),
                    "margin": float(record["margin"]),
                    "cycle_support": int(cycle_count),
                    "patch_match_fraction": float(record["patch_match_fraction"]),
                    "patch_entropy": float(record["patch_entropy"]),
                }
            )

    groups = sorted((sorted(group) for group in members.values()), key=lambda group: tuple(group))
    linked_nodes = sum(len(group) for group in groups if len(group) > 1)
    margins = [record["margin"] for record in edges.values()]
    match_fractions = [record["patch_match_fraction"] for record in edges.values()]
    entropies = [record["patch_entropy"] for record in edges.values()]
    return groups, accepted_links, {
        "candidate_edges": float(len(edges)),
        "accepted_edges": float(accepted),
        "cycle_supported_edges": float(accepted_cycle_supported),
        "linked_node_fraction": float(linked_nodes / len(nodes)) if nodes else 0.0,
        "singleton_fraction": float(sum(len(group) == 1 for group in groups) / len(groups)) if groups else 0.0,
        "mean_track_support": float(np.mean([len(group) for group in groups])) if groups else 0.0,
        "mean_edge_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_patch_match_fraction": float(np.mean(match_fractions)) if match_fractions else 0.0,
        "mean_patch_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def associate_patch_views(
    views: Sequence[Mapping[str, Any]],
    configuration: PatchAssociationConfiguration,
) -> dict[str, Any]:
    """Build deterministic tracks from foreground patch correspondences."""

    prepared = _normalise_views(views)
    edges = _pairwise_edges(prepared, configuration)
    groups, links, diagnostics = _tracks_from_edges(prepared, edges, configuration)
    return {
        "configuration": asdict(configuration),
        "views": [str(view["view"]) for view in prepared],
        "tracks": [
            {
                "identity_id": f"I{index:03d}",
                "members": [
                    {"view_id": view_id, "candidate_id": candidate_id}
                    for view_id, candidate_id in group
                ],
            }
            for index, group in enumerate(groups)
        ],
        "accepted_links": links,
        "diagnostics": diagnostics,
    }


__all__ = [
    "PATCH_CONFIGURATIONS",
    "PatchAssociationConfiguration",
    "associate_patch_views",
    "correspondence_cost",
    "foreground_patch_evidence",
    "foreground_patch_tokens",
    "reciprocal_patch_distance",
    "sinkhorn_patch_distance",
    "symmetric_patch_distance",
]
