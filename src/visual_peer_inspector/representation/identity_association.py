"""Deterministic cross-view association candidates for the bounded 08J study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from visual_peer_inspector.representation.cross_view_identity import (
    CandidateObservation,
    IdentityConfiguration,
    ViewObservation,
    match_cross_view_identities,
)


Strategy = Literal["current", "reciprocal", "cycle_pruned", "support_consensus"]
Node = tuple[str, str]


@dataclass(frozen=True)
class AssociationConfiguration:
    """One predeclared association rule used in the development comparison."""

    name: str
    strategy: Strategy
    maximum_distance: float = 0.35

    def __post_init__(self) -> None:
        if self.strategy not in {"current", "reciprocal", "cycle_pruned", "support_consensus"}:
            raise ValueError("Unknown association strategy")
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("Configuration names must contain letters, numbers or underscores")
        if not np.isfinite(self.maximum_distance) or not 0.0 < self.maximum_distance <= 2.0:
            raise ValueError("maximum_distance must be in (0, 2]")


CONFIGURATIONS = (
    AssociationConfiguration("current_hungarian_035", "current", 0.35),
    AssociationConfiguration("reciprocal_020", "reciprocal", 0.20),
    AssociationConfiguration("reciprocal_035", "reciprocal", 0.35),
    AssociationConfiguration("cycle_pruned_035", "cycle_pruned", 0.35),
    AssociationConfiguration("support_consensus_035", "support_consensus", 0.35),
)


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Embeddings must be finite one-dimensional vectors")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Embeddings must have non-zero norm")
    return vector / norm


def _normalise_views(views: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    prepared = []
    dimensions = set()
    for raw_view in views:
        view = str(raw_view["view"])
        if not view or any(item["view"] == view for item in prepared):
            raise ValueError("View identifiers must be unique and non-empty")
        objects = []
        for raw_object in raw_view["objects"]:
            candidate = str(raw_object["candidate_id"])
            if not candidate or any(item["candidate_id"] == candidate for item in objects):
                raise ValueError("Candidate identifiers must be unique within a view")
            embedding = _unit(raw_object["embedding"])
            dimensions.add(int(embedding.size))
            objects.append({"candidate_id": candidate, "embedding": embedding})
        prepared.append({"view": view, "objects": tuple(sorted(objects, key=lambda item: item["candidate_id"]))})
    prepared.sort(key=lambda item: item["view"])
    if len(prepared) < 2:
        raise ValueError("Cross-view association requires at least two views")
    if len(dimensions) > 1:
        raise ValueError("All embeddings must have the same dimension")
    return tuple(prepared)


def _reciprocal_edges(
    views: Sequence[Mapping[str, Any]],
    maximum_distance: float,
) -> dict[frozenset[Node], float]:
    edges: dict[frozenset[Node], float] = {}
    for left, right in combinations(views, 2):
        if not left["objects"] or not right["objects"]:
            continue
        left_matrix = np.stack([item["embedding"] for item in left["objects"]])
        right_matrix = np.stack([item["embedding"] for item in right["objects"]])
        costs = 1.0 - left_matrix @ right_matrix.T
        left_best = np.argmin(costs, axis=1)
        right_best = np.argmin(costs, axis=0)
        for left_index, right_index in enumerate(left_best):
            if int(right_best[right_index]) != left_index:
                continue
            distance = float(costs[left_index, right_index])
            if distance > maximum_distance:
                continue
            left_node = (str(left["view"]), str(left["objects"][left_index]["candidate_id"]))
            right_node = (str(right["view"]), str(right["objects"][right_index]["candidate_id"]))
            edges[frozenset((left_node, right_node))] = distance
    return edges


def _cycle_support(edges: Mapping[frozenset[Node], float]) -> dict[frozenset[Node], int]:
    neighbours: dict[Node, set[Node]] = {}
    for edge in edges:
        left, right = sorted(edge)
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    support = {}
    for edge in edges:
        left, right = sorted(edge)
        common = neighbours.get(left, set()) & neighbours.get(right, set())
        support[edge] = sum(node[0] not in {left[0], right[0]} for node in common)
    return support


def _tracks_from_edges(
    views: Sequence[Mapping[str, Any]],
    edges: Mapping[frozenset[Node], float],
    support: Mapping[frozenset[Node], int],
    strategy: Strategy,
) -> tuple[list[tuple[str, list[Node]]], dict[str, float]]:
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
        if {node[0] for node in members[left_root]} & {node[0] for node in members[right_root]}:
            return False
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        members[left_root] |= members.pop(right_root)
        return True

    view_count = len(views)
    candidates = []
    for edge, distance in edges.items():
        cycle_count = int(support.get(edge, 0))
        if strategy == "cycle_pruned" and view_count >= 3 and cycle_count == 0:
            continue
        left, right = sorted(edge)
        priority = (-cycle_count, distance, left, right) if strategy == "support_consensus" else (distance, left, right)
        candidates.append((priority, left, right, cycle_count))
    candidates.sort(key=lambda item: item[0])
    accepted = 0
    cycle_supported = 0
    for _priority, left, right, cycle_count in candidates:
        if merge(left, right):
            accepted += 1
            cycle_supported += int(cycle_count > 0)
    groups = sorted((sorted(value) for value in members.values()), key=lambda value: tuple(value))
    tracks = [(f"I{index:03d}", group) for index, group in enumerate(groups)]
    linked_nodes = sum(len(group) for group in groups if len(group) > 1)
    return tracks, {
        "candidate_edges": float(len(edges)),
        "accepted_edges": float(accepted),
        "cycle_supported_edges": float(cycle_supported),
        "linked_node_fraction": float(linked_nodes / len(nodes)) if nodes else 0.0,
        "singleton_fraction": float(sum(len(group) == 1 for group in groups) / len(groups)) if groups else 0.0,
        "mean_track_support": float(np.mean([len(group) for group in groups])) if groups else 0.0,
    }


def associate_views(
    views: Sequence[Mapping[str, Any]],
    configuration: AssociationConfiguration,
) -> dict[str, Any]:
    """Associate candidates without reading labels, identities or source metadata."""

    prepared = _normalise_views(views)
    if configuration.strategy == "current":
        observations = tuple(
            ViewObservation(
                view_id=str(view["view"]),
                candidates=tuple(
                    CandidateObservation(
                        candidate_id=str(item["candidate_id"]),
                        embedding=tuple(float(value) for value in item["embedding"]),
                    )
                    for item in view["objects"]
                ),
            )
            for view in prepared
        )
        result = match_cross_view_identities(
            observations,
            configuration=IdentityConfiguration(
                unmatched_cost=configuration.maximum_distance,
                ambiguity_margin=0.05,
                geometry_weight=0.0,
            ),
        )
        tracks = [
            (
                track.identity_id,
                [(member.view_id, member.candidate_id) for member in track.members],
            )
            for track in result.tracks
        ]
        linked_nodes = sum(len(group) for _, group in tracks if len(group) > 1)
        node_count = sum(len(view["objects"]) for view in prepared)
        diagnostics = {
            "candidate_edges": float(sum(item.status == "matched" for pair in result.pairwise_results for item in pair.assignments)),
            "accepted_edges": float(sum(max(0, len(group) - 1) for _, group in tracks)),
            "cycle_supported_edges": float(sum(track.cycle_consistent and len(track.members) >= 3 for track in result.tracks)),
            "linked_node_fraction": float(linked_nodes / node_count) if node_count else 0.0,
            "singleton_fraction": float(sum(len(group) == 1 for _, group in tracks) / len(tracks)) if tracks else 0.0,
            "mean_track_support": float(np.mean([len(group) for _, group in tracks])) if tracks else 0.0,
        }
    else:
        edges = _reciprocal_edges(prepared, configuration.maximum_distance)
        support = _cycle_support(edges)
        tracks, diagnostics = _tracks_from_edges(prepared, edges, support, configuration.strategy)
    return {
        "configuration": asdict(configuration),
        "views": [str(view["view"]) for view in prepared],
        "tracks": [
            {
                "identity_id": identity_id,
                "members": [
                    {"view_id": view_id, "candidate_id": candidate_id}
                    for view_id, candidate_id in group
                ],
            }
            for identity_id, group in tracks
        ],
        "diagnostics": diagnostics,
    }


__all__ = ["AssociationConfiguration", "CONFIGURATIONS", "associate_views"]
