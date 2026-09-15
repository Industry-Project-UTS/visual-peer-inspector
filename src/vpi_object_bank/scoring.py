"""CPU peer scoring over foreground patch banks from explicitly selected views.

The object-bank method changes the reference set of the established forward
scorer: each peer contributes all its available views, then contributes one
distance per target patch. Lower-median and floor-rounded upper-tail reductions
match ``visual_peer_inspector.scoring.forward_patch.forward_patch_scores``.

``view_mean_control`` first averages nearest-patch distances equally across a
peer's usable views, before the same across-peer reduction. It controls the
union bank's extra opportunities for a close match as views are added. It gives
up best-angle matching and is not a selected method or a cardinality-invariance
guarantee: individual view patch counts can still differ.

Object IDs are opaque association hypotheses, not verified truth. Self-exclusion
is guaranteed relative to that supplied map. Unassigned observations cannot be
safely excluded from other views of themselves, so they are retained with an
unavailable bank score and never enter a reference bank. Their within-view
baseline remains scoreable because that baseline uses observation identity only.

Returned mean/top2 values aggregate raw observation scores. They are diagnostics,
not the historical pipeline's within-view-rank-normalized fusion.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
import re
from typing import Any

import numpy as np


SCHEMA_VERSION = "multiview-object-bank/scoring-v1"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_FIELDS = {"view_id", "candidate_id", "object_id", "patches"}
_REQUIRED = {"view_id", "candidate_id", "patches"}


def _token(value: Any, field: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a nonempty opaque token")
    return value


def _normalise_patches(values: Any) -> np.ndarray:
    try:
        patches = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("patches must be a numeric matrix") from error
    if patches.ndim != 2 or patches.shape[1] == 0:
        raise ValueError("patches must have shape (foreground patches, dimensions)")
    if not np.isfinite(patches).all():
        raise ValueError("patches must contain only finite values")
    lengths = np.linalg.norm(patches, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("patch vectors must have finite nonzero norm")
    # An explicitly shaped empty matrix records missing foreground evidence.
    return patches / lengths


def _unavailable(status: str, peer_count: int, reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "patch_scores": [],
        "status": status,
        "peer_count": peer_count,
        "reason": reason,
    }


def _reduce_peer_distances(
    target_patch_count: int, distances: Sequence[np.ndarray], top_fraction: float
) -> dict[str, Any]:
    if target_patch_count == 0:
        return _unavailable(
            "unavailable_patches", len(distances), "No foreground patch features are available."
        )
    if len(distances) < 2:
        return _unavailable(
            "insufficient_peers", len(distances), "At least two usable peer objects are required."
        )
    middle = (len(distances) - 1) // 2
    patch_scores = np.partition(np.stack(distances), middle, axis=0)[middle]
    tail_count = max(1, int(len(patch_scores) * top_fraction))
    score = float(
        np.partition(patch_scores, len(patch_scores) - tail_count)[-tail_count:].mean()
    )
    return {
        "score": score,
        "patch_scores": patch_scores.tolist(),
        "status": "scored",
        "peer_count": len(distances),
        "reason": None,
    }


def _aggregate(observations: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    values = [item[method]["score"] for item in observations if item[method]["score"] is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "top2": float(np.mean(sorted(values, reverse=True)[:2])) if values else None,
        "valid_observations": len(values),
        "unavailable_observations": len(observations) - len(values),
        "status": "scored" if values else "unavailable",
        "aggregation": "raw_observation_scores",
    }


def score_scene(
    observations: Sequence[Mapping[str, Any]],
    *,
    selected_views: Sequence[str],
    top_fraction: float = 0.02,
) -> dict[str, Any]:
    """Score all observations and identities without receiving evaluator labels.

    Each observation supplies ``view_id``, ``candidate_id``, ``patches`` and an
    optional ``object_id``. Patches are the selected foreground rows, with shape
    ``(patch_count, feature_dimension)``; rows are normalized here. A shaped
    empty matrix is retained as unavailable evidence. Unknown fields, duplicate
    candidates, repeated identities within one view, and observations outside
    ``selected_views`` are rejected. Missing selected views are recorded.

    All available views of a known peer identity form one reference bank. Thus
    extra views do not grant that peer extra weight in the across-peer median,
    although a larger bank can offer closer matches. ``view_mean_control``
    averages the nearest distances across each peer's usable views instead of
    taking their minimum. Reference token/view counts expose this distinction.
    The two fusion diagnostics average available values only; top2 uses both
    strongest values when available and the one value when only one exists.
    Neither fusion invents scores for missing observations.
    """
    if isinstance(selected_views, (str, bytes)) or not selected_views:
        raise ValueError("selected_views must be a nonempty sequence of opaque view IDs")
    views = [_token(view, "selected view") for view in selected_views]
    if len(set(views)) != len(views):
        raise ValueError("selected_views must not contain duplicates")
    if isinstance(top_fraction, bool) or not isinstance(top_fraction, Real):
        raise ValueError("top_fraction must be a finite number in (0, 1]")
    top_fraction = float(top_fraction)
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be a finite number in (0, 1]")
    selected = set(views)
    prepared = []
    seen_candidates: set[tuple[str, str]] = set()
    seen_identities: set[tuple[str, str]] = set()
    dimensions: set[int] = set()
    for item in observations:
        if not isinstance(item, Mapping) or not _REQUIRED <= item.keys() or item.keys() - _FIELDS:
            raise ValueError("Observation fields must be view_id, candidate_id, patches and optional object_id")
        view = _token(item["view_id"], "view_id")
        candidate = _token(item["candidate_id"], "candidate_id")
        identity = item.get("object_id")
        if identity is not None:
            identity = _token(identity, "object_id")
        if view not in selected:
            raise ValueError("An observation is outside the selected-view scope")
        if (view, candidate) in seen_candidates:
            raise ValueError("A candidate occurs more than once in the same view")
        if identity is not None and (view, identity) in seen_identities:
            raise ValueError("One object identity cannot contain two candidates from the same view")
        patches = _normalise_patches(item["patches"])
        dimensions.add(patches.shape[1])
        seen_candidates.add((view, candidate))
        if identity is not None:
            seen_identities.add((view, identity))
        prepared.append({"view_id": view, "candidate_id": candidate, "object_id": identity, "patches": patches})
    if len(dimensions) > 1:
        raise ValueError("All patch matrices must have the same feature dimension")
    prepared.sort(key=lambda item: (item["view_id"], item["candidate_id"]))

    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        by_view[item["view_id"]].append(item)
        if item["object_id"] is not None and len(item["patches"]):
            by_identity[item["object_id"]].append(item)
    banks = dict(sorted(by_identity.items()))

    scored = []
    for item in prepared:
        # Reuse each directed observation comparison for both bank policies and
        # the within-view baseline. Keeping patch distances preserves the exact
        # order: nearest within a peer, lower median across peers, then tail.
        pair_distances = {
            (other["view_id"], other["candidate_id"]): np.min(
                1.0 - item["patches"] @ other["patches"].T, axis=1
            )
            for other in prepared
            if (other["view_id"], other["candidate_id"]) != (item["view_id"], item["candidate_id"])
            and len(other["patches"])
        }
        within_distances = [
            pair_distances[(other["view_id"], other["candidate_id"])]
            for other in by_view[item["view_id"]]
            if other["candidate_id"] != item["candidate_id"] and len(other["patches"])
        ]
        identity = item["object_id"]
        peer_tokens: dict[str, int] = {}
        peer_views: dict[str, int] = {}
        if identity is not None:
            union_distances = []
            view_mean_distances = []
            for peer_id, peer_observations in banks.items():
                if peer_id == identity:
                    continue
                distances = np.stack([
                    pair_distances[(other["view_id"], other["candidate_id"])]
                    for other in peer_observations
                ])
                union_distances.append(distances.min(axis=0))
                view_mean_distances.append(distances.mean(axis=0))
                peer_tokens[peer_id] = sum(len(other["patches"]) for other in peer_observations)
                peer_views[peer_id] = len(peer_observations)
            bank_score = _reduce_peer_distances(len(item["patches"]), union_distances, top_fraction)
            control_score = _reduce_peer_distances(len(item["patches"]), view_mean_distances, top_fraction)
        else:
            bank_score = _unavailable(
                "unavailable_identity", 0,
                "No object ID is available, so exclusion of every target view cannot be guaranteed.",
            )
            control_score = dict(bank_score)
        scored.append({
            "view_id": item["view_id"],
            "candidate_id": item["candidate_id"],
            "object_id": identity,
            "identity_status": "assigned" if identity is not None else "unmatched",
            "foreground_patch_count": len(item["patches"]),
            "peer_token_counts": peer_tokens,
            "peer_view_counts": peer_views,
            "within_view": _reduce_peer_distances(len(item["patches"]), within_distances, top_fraction),
            "object_bank": bank_score,
            "view_mean_control": control_score,
        })

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        key = (
            ("assigned", item["object_id"])
            if item["object_id"] is not None
            else ("unmatched", item["view_id"], item["candidate_id"])
        )
        grouped[key].append(item)
    objects = []
    for key, members in sorted(grouped.items()):
        objects.append({
            "object_id": members[0]["object_id"],
            "identity_status": key[0],
            "members": [{"view_id": item["view_id"], "candidate_id": item["candidate_id"]} for item in members],
            "bank_token_count": sum(item["foreground_patch_count"] for item in members) if key[0] == "assigned" else None,
            "within_view": _aggregate(members, "within_view"),
            "object_bank": _aggregate(members, "object_bank"),
            "view_mean_control": _aggregate(members, "view_mean_control"),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_views": sorted(selected),
        "missing_views": sorted(selected - by_view.keys()),
        "top_fraction": top_fraction,
        "feature_dimension": next(iter(dimensions), None),
        "observations": scored,
        "objects": objects,
        "counts": {
            "observations": len(scored),
            "assigned_objects": sum(key[0] == "assigned" for key in grouped),
            "unmatched_observations": sum(item["object_id"] is None for item in scored),
            "usable_reference_banks": len(banks),
            "unavailable_patch_observations": sum(item["foreground_patch_count"] == 0 for item in scored),
            "scored_bank_observations": sum(item["object_bank"]["score"] is not None for item in scored),
        },
    }


__all__ = ["SCHEMA_VERSION", "score_scene"]
