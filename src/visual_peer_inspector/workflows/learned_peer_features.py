"""Prepare label-free peer-ranking features from sealed DINO score records."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from visual_peer_inspector.representation.cross_view_identity import (
    CandidateObservation,
    ViewObservation,
    match_cross_view_identities,
)
from visual_peer_inspector.representation import cross_view_identity as identity_module
from visual_peer_inspector.workflows.comparison_io import (
    atomic_json,
    canonical,
    digest_file,
    read_json,
    seal,
    verify_seal,
)


STAGE = "development-learned-peer-view-features-v1"
METHODS = (
    "B0",
    "B1_control",
    "P1_current",
    "P2_consensus",
    "forward_02_cov70",
    "forward_10_cov0",
    "geometry_peer",
    "mutual_missing_05",
)

# Primary sets reproduce the project protocol. Rotated sets expose view-choice
# sensitivity without adding scenes or changing the split.
VIEW_SETS: dict[str, tuple[str, ...]] = {
    "one-primary": ("V00",),
    "one-rotated-04": ("V04",),
    "one-rotated-08": ("V08",),
    "one-rotated-12": ("V12",),
    "one-rotated-16": ("V16",),
    "two-primary": ("V00", "V10"),
    "two-rotated-04": ("V04", "V14"),
    "two-rotated-08": ("V08", "V18"),
    "two-rotated-02": ("V02", "V12"),
    "two-rotated-06": ("V06", "V16"),
    "three-primary": ("V00", "V07", "V14"),
    "three-rotated-04": ("V04", "V11", "V18"),
    "three-rotated-08": ("V02", "V08", "V15"),
    "three-rotated-12": ("V06", "V12", "V19"),
    "three-rotated-16": ("V03", "V10", "V16"),
    "five-primary": ("V00", "V04", "V08", "V12", "V16"),
    "five-rotated-01": ("V01", "V05", "V09", "V13", "V17"),
    "five-rotated-02": ("V02", "V06", "V10", "V14", "V18"),
    "five-rotated-03": ("V03", "V07", "V11", "V15", "V19"),
}

BASE_FEATURE_NAMES = tuple(
    name
    for method in METHODS
    for name in (f"rank_{method}", f"robust_z_{method}", f"available_{method}")
) + tuple(f"geometry_deviation_{index}" for index in range(7)) + (
    "nearest_peer",
    "median_peer",
    "farthest_peer",
    "peer_centroid",
    "candidate_count",
    "log_candidate_count",
) + tuple(
    f"heatmap_{heatmap}_{suffix}"
    for heatmap in ("P1_current", "forward_02_cov70", "forward_10_cov0")
    for suffix in ("mean", "std", "q90", "tail_mean", "entropy", "available")
) + (
    "forward_x_missing",
    "forward_x_geometry",
    "p1_x_p2",
    "method_rank_mean",
    "method_rank_std",
)


def _percentile(values: np.ndarray) -> np.ndarray:
    """Return order-invariant average percentile ranks in the unit interval."""

    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Percentile inputs must be a finite one-dimensional array")
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def _robust_z(values: np.ndarray) -> np.ndarray:
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if scale <= 1e-8:
        scale = float(np.std(values))
    if scale <= 1e-8:
        scale = 1.0
    return np.clip((values - centre) / scale, -8.0, 8.0)


def _score_columns(values: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float | None]]:
    """Encode incomplete upstream scores without removing their candidates."""

    numeric = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    available = np.isfinite(numeric)
    ranks = np.full(len(numeric), 0.5, dtype=np.float64)
    robust = np.zeros(len(numeric), dtype=np.float64)
    if int(available.sum()) >= 2:
        ranks[available] = _percentile(numeric[available])
        robust[available] = _robust_z(numeric[available])
    elif int(available.sum()) == 1:
        ranks[available] = 0.5
    reported = [float(value) if finite else None for value, finite in zip(ranks, available, strict=True)]
    return ranks, robust, available.astype(np.float64), reported


def _normalise_embeddings(objects: list[dict[str, Any]]) -> np.ndarray:
    embeddings = np.asarray([item["embedding"] for item in objects], dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[1] < 2 or not np.isfinite(embeddings).all():
        raise ValueError("Every candidate must have a finite DINO embedding")
    lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("DINO embeddings must have non-zero norm")
    return embeddings / lengths


def _heatmap_features(values: Any) -> tuple[float, ...]:
    if values is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    shifted = array - array.min()
    total = float(shifted.sum())
    probabilities = (
        np.full(len(array), 1.0 / len(array), dtype=np.float64)
        if total <= 1e-12
        else shifted / total
    )
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
    entropy /= max(np.log(len(array)), 1.0)
    tail_count = max(1, int(np.ceil(0.10 * len(array))))
    return (
        float(np.mean(array)),
        float(np.std(array)),
        float(np.quantile(array, 0.90)),
        float(np.mean(np.partition(array, -tail_count)[-tail_count:])),
        entropy,
        1.0,
    )


def _base_features(objects: list[dict[str, Any]]) -> tuple[np.ndarray, tuple[str, ...]]:
    if not objects:
        return np.empty((0, len(BASE_FEATURE_NAMES)), dtype=np.float64), BASE_FEATURE_NAMES

    columns: list[np.ndarray] = []
    names: list[str] = []
    ranks: dict[str, np.ndarray] = {}
    for method in METHODS:
        rank, robust, available, _ = _score_columns([item["scores"].get(method) for item in objects])
        ranks[method] = rank
        columns.extend((rank, robust, available))
        names.extend((f"rank_{method}", f"robust_z_{method}", f"available_{method}"))

    geometry = np.asarray([item["geometry"] for item in objects], dtype=np.float64)
    if geometry.shape != (len(objects), 7) or not np.isfinite(geometry).all():
        raise ValueError("Geometry input must contain seven finite values per candidate")
    geometry_centre = np.median(geometry, axis=0)
    geometry_scale = 1.4826 * np.median(np.abs(geometry - geometry_centre), axis=0)
    geometry_scale = np.where(geometry_scale > 1e-8, geometry_scale, 1.0)
    geometry_deviation = np.abs((geometry - geometry_centre) / geometry_scale)
    columns.extend(geometry_deviation[:, index] for index in range(7))
    names.extend(f"geometry_deviation_{index}" for index in range(7))

    embeddings = _normalise_embeddings(objects)
    if len(objects) == 1:
        embedding_columns = tuple(np.zeros(1, dtype=np.float64) for _ in range(4))
    else:
        distances = 1.0 - embeddings @ embeddings.T
        np.fill_diagonal(distances, np.nan)
        centroid = (embeddings.sum(axis=0, keepdims=True) - embeddings) / (len(objects) - 1)
        centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
        embedding_columns = (
            np.nanmin(distances, axis=1),
            np.nanmedian(distances, axis=1),
            np.nanmax(distances, axis=1),
            1.0 - np.sum(embeddings * centroid, axis=1),
        )
    columns.extend(embedding_columns)
    names.extend(("nearest_peer", "median_peer", "farthest_peer", "peer_centroid"))
    columns.extend(
        (
            np.full(len(objects), len(objects), dtype=np.float64),
            np.full(len(objects), np.log1p(len(objects)), dtype=np.float64),
        )
    )
    names.extend(("candidate_count", "log_candidate_count"))

    for heatmap_name in ("P1_current", "forward_02_cov70", "forward_10_cov0"):
        matrix = np.asarray(
            [_heatmap_features(item.get("heatmaps", {}).get(heatmap_name)) for item in objects],
            dtype=np.float64,
        )
        columns.extend(matrix[:, index] for index in range(matrix.shape[1]))
        names.extend(
            f"heatmap_{heatmap_name}_{suffix}"
            for suffix in ("mean", "std", "q90", "tail_mean", "entropy", "available")
        )

    columns.extend(
        (
            ranks["forward_02_cov70"] * ranks["mutual_missing_05"],
            ranks["forward_02_cov70"] * ranks["geometry_peer"],
            ranks["P1_current"] * ranks["P2_consensus"],
        )
    )
    names.extend(("forward_x_missing", "forward_x_geometry", "p1_x_p2"))

    rank_matrix = np.column_stack([ranks[method] for method in METHODS])
    columns.extend((rank_matrix.mean(axis=1), rank_matrix.std(axis=1)))
    names.extend(("method_rank_mean", "method_rank_std"))

    result = np.column_stack(columns)
    if result.shape[1] != len(names) or not np.isfinite(result).all():
        raise ValueError("Feature construction produced an invalid matrix")
    if tuple(names) != BASE_FEATURE_NAMES:
        raise ValueError("Feature implementation differs from the declared schema")
    return result, tuple(names)


def _cross_view_support(
    rows: dict[str, dict[str, Any]], views: tuple[str, ...]
) -> dict[tuple[str, str], tuple[float, ...]]:
    """Measure label-free DINO support for every candidate across selected views."""

    dimensions = {
        len(item["embedding"])
        for view in views
        for item in rows[view]["objects"]
    }
    if len(dimensions) > 1:
        raise ValueError("Cross-view embeddings have inconsistent dimensions")
    dimension = next(iter(dimensions), 1)
    normalised = {
        view: (
            _normalise_embeddings(rows[view]["objects"])
            if rows[view]["objects"]
            else np.empty((0, dimension), dtype=np.float64)
        )
        for view in views
    }
    forward_ranks = {
        view: _score_columns(
            [item["scores"].get("forward_02_cov70") for item in rows[view]["objects"]]
        )[0]
        for view in views
    }
    output: dict[tuple[str, str], tuple[float, ...]] = {}
    for view in views:
        objects = rows[view]["objects"]
        for index, item in enumerate(objects):
            best_similarity: list[float] = []
            anomaly_support: list[float] = []
            for other in views:
                if other == view:
                    continue
                if not len(normalised[other]):
                    continue
                similarities = normalised[other] @ normalised[view][index]
                best_similarity.append(float(np.max(similarities)))
                top_count = min(2, len(similarities))
                top_anomaly = np.argsort(forward_ranks[other], kind="mergesort")[-top_count:]
                anomaly_support.append(
                    float(np.max(similarities[top_anomaly] * forward_ranks[other][top_anomaly]))
                )
            if not best_similarity:
                best_similarity = [0.0]
                anomaly_support = [0.0]
            values = np.asarray(best_similarity, dtype=np.float64)
            anomaly_values = np.asarray(anomaly_support, dtype=np.float64)
            output[(view, item["candidate_id"])] = (
                float(values.mean()),
                float(values.min()),
                float(values.max()),
                float(np.mean(values >= 0.70)),
                float(anomaly_values.mean()),
                float(anomaly_values.max()),
                float(len(views)),
                float(np.log1p(len(views))),
            )
    return output


def _identity_tracks(
    rows: dict[str, dict[str, Any]], views: tuple[str, ...]
) -> list[dict[str, Any]]:
    if len(views) == 1:
        view = views[0]
        return [
            {
                "identity_id": f"single-{item['candidate_id']}",
                "members": [{"view_id": view, "candidate_id": item["candidate_id"]}],
                "status": "single_view",
                "minimum_margin": None,
                "cycle_consistent": True,
            }
            for item in sorted(rows[view]["objects"], key=lambda value: value["candidate_id"])
        ]
    observations = tuple(
        ViewObservation(
            view_id=view,
            candidates=tuple(
                CandidateObservation(
                    candidate_id=item["candidate_id"],
                    embedding=tuple(item["embedding"]),
                )
                for item in rows[view]["objects"]
            ),
        )
        for view in views
    )
    result = match_cross_view_identities(observations)
    return [
        {
            "identity_id": track.identity_id,
            "members": [member.to_record() for member in track.members],
            "status": track.status,
            "minimum_margin": track.minimum_margin,
            "cycle_consistent": track.cycle_consistent,
        }
        for track in result.tracks
    ]


def _winner_tracks(
    rows: dict[str, dict[str, Any]],
    views: tuple[str, ...],
    *,
    selector: str,
    distance_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    """Cluster only per-view winners for the bounded vote-and-review route."""

    if selector not in {"forward", "consensus"}:
        raise ValueError("Unknown winner selector")
    nodes: list[dict[str, Any]] = []
    for view in views:
        objects = rows[view]["objects"]
        selector_methods = (
            ("forward_02_cov70",)
            if selector == "forward"
            else ("forward_02_cov70", "P1_current", "P2_consensus")
        )
        method_ranks = [
            _score_columns([item["scores"].get(method) for item in objects])[3]
            for method in selector_methods
        ]
        candidate_scores = []
        for index in range(len(objects)):
            values = [ranks[index] for ranks in method_ranks if ranks[index] is not None]
            candidate_scores.append(None if not values else float(np.mean(values)))
        available = [value for value in candidate_scores if value is not None]
        if not available:
            continue
        best = max(available)
        ordered = sorted(available, reverse=True)
        winner_margin = best - ordered[1] if len(ordered) > 1 else 0.0
        for item, value in zip(objects, candidate_scores, strict=True):
            if value is not None and np.isclose(value, best, atol=1e-12, rtol=0.0):
                embedding = np.asarray(item["embedding"], dtype=np.float64)
                embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
                nodes.append(
                    {
                        "view_id": view,
                        "candidate_id": item["candidate_id"],
                        "embedding": embedding,
                        "winner_margin": float(winner_margin),
                    }
                )

    parent = list(range(len(nodes)))
    component_views = [{node["view_id"]} for node in nodes]
    component_indices = [{index} for index in range(len(nodes))]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if nodes[left]["view_id"] == nodes[right]["view_id"]:
                continue
            distance = 1.0 - float(nodes[left]["embedding"] @ nodes[right]["embedding"])
            edges.append((distance, nodes[left]["view_id"], nodes[left]["candidate_id"], nodes[right]["view_id"], nodes[right]["candidate_id"], left, right))
    for distance, _, _, _, _, left, right in sorted(edges):
        if distance > distance_threshold:
            break
        left_root, right_root = find(left), find(right)
        if left_root == right_root or not component_views[left_root].isdisjoint(component_views[right_root]):
            continue
        combined = component_indices[left_root] | component_indices[right_root]
        maximum_distance = max(
            (
                1.0 - float(nodes[first]["embedding"] @ nodes[second]["embedding"])
                for first in combined
                for second in combined
                if first < second
            ),
            default=0.0,
        )
        if maximum_distance > distance_threshold:
            continue
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep
        component_views[keep] |= component_views[drop]
        component_indices[keep] |= component_indices[drop]

    components: dict[int, list[dict[str, Any]]] = {}
    for index, node in enumerate(nodes):
        components.setdefault(find(index), []).append(node)
    tracks = []
    for track_index, (_, members) in enumerate(sorted(components.items()), 1):
        if len(members) > 1:
            similarities = [
                float(members[left]["embedding"] @ members[right]["embedding"])
                for left in range(len(members))
                for right in range(left + 1, len(members))
            ]
            coherence = float(np.mean(similarities))
        else:
            coherence = 0.0
        vote_fraction = len(members) / len(views)
        mean_winner_margin = float(np.mean([member["winner_margin"] for member in members]))
        # View support is the primary ordering. The secondary term is bounded
        # below one quarter of a single-vote step, so it can break equal-support
        # ties but cannot overpower evidence from an additional view.
        secondary = 0.5 * mean_winner_margin + 0.5 * float(np.clip(coherence, 0.0, 1.0))
        cluster_score = vote_fraction + secondary / (4.0 * len(views))
        tracks.append(
            {
                "identity_id": f"winner-{selector}-{track_index:03d}",
                "members": [
                    {"view_id": member["view_id"], "candidate_id": member["candidate_id"]}
                    for member in sorted(members, key=lambda value: (value["view_id"], value["candidate_id"]))
                ],
                "status": "winner_cluster" if len(members) > 1 else "unconfirmed_winner",
                "minimum_margin": None,
                "cycle_consistent": None,
                "vote_fraction": float(vote_fraction),
                "mean_winner_margin": mean_winner_margin,
                "coherence": coherence,
                "cluster_score": float(cluster_score),
            }
        )
    return tracks


def prepare_scene_features(records: list[dict[str, Any]]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Prepare every declared route and view set without evaluator information."""

    if not records:
        raise ValueError("A scene prediction file cannot be empty")
    scene = str(records[0]["scene"])
    available: dict[str, dict[str, dict[str, Any]]] = {"automatic": {}, "supplied": {}}
    for row in records:
        if row.get("scene") != scene or row.get("route") not in available:
            raise ValueError("Scene predictions contain an unexpected identity")
        view = str(row["view"])
        if view in available[row["route"]]:
            raise ValueError("Scene predictions contain a duplicate route and view")
        available[row["route"]][view] = row

    outputs: list[dict[str, Any]] = []
    feature_names: tuple[str, ...] | None = None
    for route in ("automatic", "supplied"):
        for view_set_id, views in VIEW_SETS.items():
            if any(view not in available[route] for view in views):
                raise ValueError(f"Missing {route} input for {view_set_id}")
            chosen = {view: available[route][view] for view in views}
            support = _cross_view_support(chosen, views)
            view_groups = []
            for view in views:
                objects = chosen[view]["objects"]
                base, current_names = _base_features(objects)
                current_names = current_names + (
                    "cross_similarity_mean",
                    "cross_similarity_min",
                    "cross_similarity_max",
                    "cross_similarity_fraction_ge_070",
                    "cross_anomaly_support_mean",
                    "cross_anomaly_support_max",
                    "view_count",
                    "log_view_count",
                )
                if feature_names is None:
                    feature_names = current_names
                elif feature_names != current_names:
                    raise ValueError("Feature names changed within one preparation run")
                support_matrix = (
                    np.asarray(
                        [support[(view, item["candidate_id"])] for item in objects],
                        dtype=np.float64,
                    )
                    if objects
                    else np.empty((0, 8), dtype=np.float64)
                )
                features = np.column_stack((base, support_matrix))
                view_groups.append(
                    {
                        "view": view,
                        "candidate_ids": [item["candidate_id"] for item in objects],
                        "features": features.tolist(),
                        "method_ranks": {
                            method: _score_columns(
                                [item["scores"].get(method) for item in objects]
                            )[3]
                            for method in METHODS
                        },
                    }
                )
            outputs.append(
                {
                    "route": route,
                    "view_set_id": view_set_id,
                    "views": list(views),
                    "view_count": len(views),
                    "view_groups": view_groups,
                    "tracks": _identity_tracks(chosen, views),
                    "winner_tracks": {
                        "forward": _winner_tracks(chosen, views, selector="forward"),
                        "consensus": _winner_tracks(chosen, views, selector="consensus"),
                    },
                }
            )
    if feature_names is None:
        raise ValueError("No feature groups were prepared")
    return {
        "schema_version": STAGE,
        "scene": scene,
        "groups": outputs,
    }, feature_names


def _verified_checkpoint(path: Path, receipt_path: Path, binding_sha256: str) -> bool:
    if not path.is_file() or not receipt_path.is_file():
        return False
    receipt = read_json(receipt_path)
    return (
        receipt.get("binding_sha256") == binding_sha256
        and receipt.get("sha256") == digest_file(path)
        and receipt.get("bytes") == path.stat().st_size
    )


def prepare_feature_package(scoring_root: Path, output_root: Path) -> dict[str, Any]:
    """Prepare resumable label-free scene features and write a prediction lock."""

    scoring_root = Path(scoring_root)
    output_root = Path(output_root)
    prediction_lock = verify_seal(scoring_root, name="prediction_lock.json")
    scene_paths = sorted((scoring_root / "scenes").glob("S*.json"))
    scene_paths = [path for path in scene_paths if not path.name.endswith(".receipt.json")]
    if not scene_paths:
        raise ValueError("No sealed scoring scenes were found")
    scenes = [path.stem for path in scene_paths]
    binding = {
        "schema_version": STAGE,
        "upstream_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "scene_membership_sha256": sha256(canonical(scenes)).hexdigest(),
        "methods": list(METHODS),
        "view_sets": {key: list(value) for key, value in VIEW_SETS.items()},
        "implementation_sha256": digest_file(Path(__file__)),
        "identity_implementation_sha256": digest_file(Path(identity_module.__file__)),
    }
    binding_sha256 = sha256(canonical(binding)).hexdigest()
    atomic_json(output_root / "run_binding.json", binding | {"binding_sha256": binding_sha256})
    atomic_json(output_root / "run_status.json", {"status": "running"})
    scene_root = output_root / "scenes"
    scene_root.mkdir(parents=True, exist_ok=True)
    reused = 0
    feature_names: tuple[str, ...] | None = None
    for index, source in enumerate(scene_paths, 1):
        target = scene_root / source.name
        receipt = scene_root / f"{source.stem}.receipt.json"
        if _verified_checkpoint(target, receipt, binding_sha256):
            reused += 1
            current = read_json(target)
            names = tuple(current["feature_names"])
        else:
            payload, names = prepare_scene_features(read_json(source))
            payload["feature_names"] = list(names)
            atomic_json(target, payload)
            atomic_json(
                receipt,
                {
                    "binding_sha256": binding_sha256,
                    "sha256": digest_file(target),
                    "bytes": target.stat().st_size,
                },
            )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("A feature checkpoint has a different schema")
        if index % 10 == 0 or index == len(scene_paths):
            atomic_json(
                output_root / "progress.json",
                {"completed_scenes": index, "expected_scenes": len(scene_paths)},
            )
            print(
                {"event": "feature_checkpoint", "completed": index, "expected": len(scene_paths), "reused": reused},
                flush=True,
            )

    scientific = [output_root / "run_binding.json", output_root / "progress.json"]
    shareable_root = output_root / "shareable"
    shareable_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature_index": range(len(feature_names or ())),
            "feature": list(feature_names or ()),
        }
    ).to_csv(shareable_root / "feature_schema.csv", index=False)
    atomic_json(
        shareable_root / "preparation_receipt.json",
        {
            "stage": STAGE,
            "scene_count": len(scenes),
            "feature_count": len(feature_names or ()),
            "route_count": 2,
            "view_set_count": len(VIEW_SETS),
            "evaluator_accessed": False,
        },
    )
    scientific.extend(
        (shareable_root / "feature_schema.csv", shareable_root / "preparation_receipt.json")
    )
    for scene in scenes:
        scientific.extend(
            (scene_root / f"{scene}.json", scene_root / f"{scene}.receipt.json")
        )
    lock = seal(
        output_root,
        scientific,
        name="feature_lock.json",
        metadata={
            "stage": STAGE,
            "scene_count": len(scenes),
            "feature_count": len(feature_names or ()),
            "evaluator_accessed": False,
            "upstream_files": len(prediction_lock["files"]),
        },
    )
    atomic_json(output_root / "run_status.json", {"status": "complete"})
    return {
        "scene_count": len(scenes),
        "feature_count": len(feature_names or ()),
        "reused_scenes": reused,
        "binding_sha256": binding_sha256,
        "feature_lock_sha256": digest_file(output_root / "feature_lock.json"),
        "locked_files": len(lock["files"]),
    }


def feature_package_inventory(feature_root: Path) -> Counter[str]:
    """Return a small verified inventory used by notebook summaries and tests."""

    feature_root = Path(feature_root)
    lock = verify_seal(feature_root, name="feature_lock.json")
    counts: Counter[str] = Counter()
    counts["locked_files"] = len(lock["files"])
    counts["scene_files"] = len(list((feature_root / "scenes").glob("S*.json"))) // 2
    counts["feature_count"] = int(lock["metadata"]["feature_count"])
    return counts


__all__ = [
    "METHODS",
    "STAGE",
    "VIEW_SETS",
    "feature_package_inventory",
    "prepare_feature_package",
    "prepare_scene_features",
]
