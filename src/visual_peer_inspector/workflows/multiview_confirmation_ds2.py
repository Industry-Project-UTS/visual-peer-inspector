"""Protected DS2 confirmation, view ablation, and multi-view fusion."""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evaluation.confirmation_reference import (
    verify_ds2_attributes_package,
    verify_ds2_geometry_package,
)
from visual_peer_inspector.evaluation.masks import (
    binary_mask_dice,
    binary_mask_iou,
    boundary_precision_recall_f1,
)
from visual_peer_inspector.evaluation.proposals import match_boxes
from visual_peer_inspector.evaluation.ranking import (
    clustered_mean_interval,
    paired_clustered_difference,
    ranking_metrics,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram
from visual_peer_inspector.models.sam2 import decode_binary_mask


STAGE = "ds2_confirmation_and_multiview"
VIEW_COUNTS = (1, 2, 3, 5, 10, 20)
FUSION_METHODS = ("mean_rank", "visibility_weighted_rank")
ROUTES = ("supplied_mask", "reference_box", "automatic_box")
METHODS = ("B0", "P1")
PRIMARY_VIEW_COUNT = 3


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Table rows do not share one schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("DS2 evaluation outputs cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    record = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": sha256(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        ).hexdigest(),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def fixed_camera_spread_views(view_count: int, *, total_views: int = 20) -> tuple[str, ...]:
    """Return one predeclared, evenly spaced subset from the ordered camera views."""

    if view_count not in VIEW_COUNTS or view_count > total_views:
        raise ValueError("View count is outside the frozen ablation set")
    indices = [int(math.floor(index * total_views / view_count)) for index in range(view_count)]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Camera-spread view selection produced duplicates")
    return tuple(f"V{index:02d}" for index in indices)


def _normalised_unusual_rank(rank: int, candidate_count: int) -> float:
    if candidate_count < 3 or not 1 <= rank <= candidate_count:
        raise ValueError("Peer rank is outside the valid range")
    return (candidate_count - rank) / (candidate_count - 1)


def _extra_candidate_key(scene_id: str, view_id: str, peer_key: str) -> str:
    payload = f"{scene_id}|{view_id}|{peer_key}".encode("utf-8")
    return "X" + sha256(payload).hexdigest()[:20]


def prepare_missing_aware_evaluator_scores(
    score_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    automatic_sam_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_box_iou: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Associate frozen automatic candidates with references after prediction lock.

    Model-facing rows carry only route-local peer keys. This evaluator-only join
    introduces object identities, explicit missed objects, and unmatched extra
    proposals so that detection failures remain visible in ranking metrics.
    """

    protected_model_fields = {
        "opaque_object_id",
        "labels_by_object_id",
        "category",
        "anomaly_count",
        "anomaly_families",
        "opaque_asset_group_id",
    }
    if any(protected_model_fields.intersection(row) for row in score_rows):
        raise ValueError("Model-facing DS2 scores contain evaluator-only fields")

    references: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        if bool(row["visible"]):
            references[(str(row["opaque_scene_id"]), str(row["view_id"]))].append(row)
    for key in references:
        references[key].sort(key=lambda row: str(row["opaque_object_id"]))

    automatic_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in automatic_sam_rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in automatic_results or set(row.get("routes", {})) != {"automatic_box"}:
            raise ValueError("Automatic SAM evaluator input is duplicated or malformed")
        automatic_results[key] = row
    if set(automatic_results) != set(references):
        raise ValueError("Automatic SAM and reference image coverage differs")

    score_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        score_groups[
            (
                str(row["opaque_scene_id"]),
                str(row["view_id"]),
                str(row["route"]),
                str(row["method"]),
            )
        ].append(row)

    evaluator_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    for row in score_rows:
        if row["route"] == "automatic_box":
            continue
        value = dict(row)
        value.update(
            {
                "opaque_object_id": str(row["peer_key"]),
                "association_status": "stable_reference_identity",
                "association_box_iou": 1.0,
                "detected": True,
            }
        )
        evaluator_rows.append(value)

    for image_key in sorted(references):
        scene_id, view_id = image_key
        reference_items = references[image_key]
        automatic_result = automatic_results[image_key]
        proposal_items = sorted(
            automatic_result["routes"]["automatic_box"],
            key=lambda item: str(item["prompt_key"]),
        )
        proposal_provenance = automatic_result.get("automatic_proposal_provenance")
        if not isinstance(proposal_provenance, dict):
            raise ValueError("Automatic SAM input is missing proposal failure provenance")
        required_provenance = {
            "source_status",
            "source_error_type",
            "source_attempt",
            "carried_as_zero_proposal",
            "raw_proposal_count",
            "removed_proposal_count",
            "discarded_untrusted_box_count",
            "proposal_count",
        }
        if set(proposal_provenance) != required_provenance:
            raise ValueError("Automatic SAM proposal provenance schema differs")
        source_status = str(proposal_provenance["source_status"])
        carried_as_zero = proposal_provenance["carried_as_zero_proposal"]
        raw_proposal_count = int(proposal_provenance["raw_proposal_count"])
        removed_proposal_count = int(proposal_provenance["removed_proposal_count"])
        discarded_untrusted_box_count = int(
            proposal_provenance["discarded_untrusted_box_count"]
        )
        final_proposal_count = int(proposal_provenance["proposal_count"])
        source_attempt = int(proposal_provenance["source_attempt"])
        if (
            source_status not in {"ok", "error"}
            or not isinstance(carried_as_zero, bool)
            or carried_as_zero != (source_status == "error")
            or source_attempt < 1
            or min(
                raw_proposal_count,
                removed_proposal_count,
                discarded_untrusted_box_count,
                final_proposal_count,
            ) < 0
            or raw_proposal_count
            != removed_proposal_count
            + discarded_untrusted_box_count
            + final_proposal_count
            or final_proposal_count != len(proposal_items)
            or (
                source_status == "error"
                and (
                    final_proposal_count != 0
                    or removed_proposal_count != 0
                    or discarded_untrusted_box_count != raw_proposal_count
                    or not proposal_provenance["source_error_type"]
                )
            )
            or (
                source_status == "ok"
                and (
                    discarded_untrusted_box_count != 0
                    or proposal_provenance["source_error_type"] is not None
                )
            )
        ):
            raise ValueError("Automatic SAM proposal failure provenance is inconsistent")
        proposal_keys = [str(item["prompt_key"]) for item in proposal_items]
        if len(proposal_keys) != len(set(proposal_keys)):
            raise ValueError("Automatic SAM prompts are not unique within an image")
        matches = match_boxes(
            [item["box_xyxy"] for item in reference_items],
            [item["box_xyxy"] for item in proposal_items],
            minimum_iou=minimum_box_iou,
        )
        object_by_proposal = {
            match.proposal_index: str(reference_items[match.reference_index]["opaque_object_id"])
            for match in matches
        }
        iou_by_proposal = {match.proposal_index: float(match.iou) for match in matches}
        matched_reference_indices = {match.reference_index for match in matches}
        association_rows.append(
            {
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "reference_count": len(reference_items),
                "proposal_count": len(proposal_items),
                "matched_count": len(matches),
                "miss_count": len(reference_items) - len(matches),
                "extra_count": len(proposal_items) - len(matches),
                "precision_at_0_50": len(matches) / len(proposal_items) if proposal_items else 0.0,
                "recall_at_0_50": len(matches) / len(reference_items) if reference_items else 1.0,
                "source_status": source_status,
                "source_error_type": proposal_provenance["source_error_type"],
                "source_attempt": source_attempt,
                "carried_as_zero_proposal": carried_as_zero,
                "raw_proposal_count": raw_proposal_count,
                "removed_proposal_count": removed_proposal_count,
                "discarded_untrusted_box_count": discarded_untrusted_box_count,
            }
        )

        for method in METHODS:
            group = score_groups.get((scene_id, view_id, "automatic_box", method), [])
            by_peer = {str(row["peer_key"]): row for row in group}
            if set(by_peer) != set(proposal_keys):
                raise ValueError("Automatic DINO scores and SAM proposals differ")
            for proposal_index, item in enumerate(proposal_items):
                peer_key = str(item["prompt_key"])
                value = dict(by_peer[peer_key])
                matched_object = object_by_proposal.get(proposal_index)
                value.update(
                    {
                        "opaque_object_id": (
                            matched_object
                            if matched_object is not None
                            else _extra_candidate_key(scene_id, view_id, peer_key)
                        ),
                        "association_status": (
                            "matched_reference" if matched_object is not None else "extra_proposal"
                        ),
                        "association_box_iou": iou_by_proposal.get(proposal_index, 0.0),
                        "detected": True,
                    }
                )
                evaluator_rows.append(value)
            for reference_index, reference in enumerate(reference_items):
                if reference_index in matched_reference_indices:
                    continue
                object_id = str(reference["opaque_object_id"])
                evaluator_rows.append(
                    {
                        "schema_version": 1,
                        "candidate_id": "M" + sha256(
                            f"{scene_id}|{view_id}|{method}|{object_id}".encode("utf-8")
                        ).hexdigest()[:20],
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "route": "automatic_box",
                        "peer_key": None,
                        "encoder_key": "dinov2_small",
                        "method": method,
                        "scoring_status": "missing_proposal",
                        "score": None,
                        "rank": None,
                        "candidate_count": len(proposal_items),
                        "foreground_pixels": int(reference["foreground_pixels"]),
                        "predicted_mask_quality": None,
                        "opaque_object_id": object_id,
                        "association_status": "missed_reference",
                        "association_box_iou": 0.0,
                        "detected": False,
                    }
                )
    return evaluator_rows, association_rows


def fuse_object_ranks(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_views: Sequence[str],
    fusion_method: str,
) -> tuple[list[dict[str, Any]], float]:
    """Fuse per-view peer ranks after stable object identity is available."""

    if fusion_method not in FUSION_METHODS:
        raise ValueError("Unknown DS2 fusion method")
    selected = set(selected_views)
    by_object: dict[str, dict[str, tuple[float, float, bool, bool]]] = defaultdict(dict)
    rank_profile: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        view_id = str(row["view_id"])
        if view_id not in selected:
            continue
        object_id = str(row["opaque_object_id"])
        scored = row.get("rank") is not None and int(row["candidate_count"]) >= 3
        detected = bool(row.get("detected", True))
        if scored and not detected:
            raise ValueError("An undetected object cannot carry peer-ranking evidence")
        rank_value = (
            _normalised_unusual_rank(int(row["rank"]), int(row["candidate_count"]))
            if scored
            else 0.0
        )
        weight = max(math.sqrt(int(row["foreground_pixels"]) / (256 * 256)), 1e-3)
        if view_id in by_object[object_id]:
            raise ValueError("DS2 evaluator rows duplicate an object within a view")
        by_object[object_id][view_id] = (
            rank_value,
            weight,
            detected,
            scored,
        )
    if len(by_object) < 3:
        raise ValueError("Multi-view fusion requires at least three stable peer objects")
    expected_support = len(selected_views)
    fused = []
    for object_id, values_by_view in sorted(by_object.items()):
        observed_weights = [value[1] for value in values_by_view.values()]
        missing_weight = float(np.mean(observed_weights)) if observed_weights else 1.0
        values = [
            values_by_view.get(view_id, (0.0, missing_weight, False, False))
            for view_id in selected_views
        ]
        detected_support = sum(value[2] for value in values)
        scored_values = [value for value in values if value[3]]
        scored_support = len(scored_values)
        if scored_values:
            ranks = np.asarray([value[0] for value in scored_values], dtype=float)
            weights = np.asarray([value[1] for value in scored_values], dtype=float)
            rank_profile[object_id].extend(ranks.tolist())
            score = float(np.mean(ranks))
            if fusion_method == "visibility_weighted_rank":
                score = float(np.average(ranks, weights=weights))
            score *= scored_support / expected_support
        else:
            score = 0.0
        all_weights = np.asarray([value[1] for value in values], dtype=float)
        fused.append(
            {
                "opaque_object_id": object_id,
                "fused_score": score,
                "view_support": detected_support,
                "detected_view_support": detected_support,
                "scored_view_support": scored_support,
                "missing_view_count": expected_support - detected_support,
                "unscoreable_detected_view_count": detected_support - scored_support,
                "mean_visibility_weight": float(np.mean(all_weights)),
            }
        )
    fused.sort(key=lambda row: (-float(row["fused_score"]), str(row["opaque_object_id"])))
    disagreement = (
        float(np.mean([np.std(values) for values in rank_profile.values()]))
        if rank_profile
        else 0.0
    )
    return fused, disagreement


def evaluate_multiview_scores(
    score_rows: Sequence[Mapping[str, Any]],
    protected_scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate paired scene metrics for every frozen route, scorer, view count, and fusion."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["opaque_scene_id"]), str(row["route"]), str(row["method"]))].append(row)
    expected = DS2_CONFIRM_CONTRACT.scene_count * len(ROUTES) * len(METHODS)
    if len(grouped) != expected:
        raise ValueError("DS2 score package does not cover every scene, route, and method")

    scene_rows: list[dict[str, Any]] = []
    fused_rows: list[dict[str, Any]] = []
    for (scene_id, route, method), values in sorted(grouped.items()):
        scene = protected_scenes[scene_id]
        labels = {str(key): bool(value) for key, value in scene["labels_by_object_id"].items()}
        positives = sorted(key for key, value in labels.items() if value)
        for view_count in VIEW_COUNTS:
            views = fixed_camera_spread_views(view_count)
            for fusion_method in FUSION_METHODS:
                fused, disagreement = fuse_object_ranks(
                    values,
                    selected_views=views,
                    fusion_method=fusion_method,
                )
                object_ids = [str(row["opaque_object_id"]) for row in fused]
                ranked_labels = [labels.get(object_id, False) for object_id in object_ids]
                metrics = ranking_metrics(
                    ranked_labels,
                    positive_total=int(scene["anomaly_count"]),
                    ranked_object_ids=object_ids,
                    positive_object_ids=positives,
                )
                margin = float(fused[0]["fused_score"] - fused[1]["fused_score"])
                record = {
                    "opaque_scene_id": scene_id,
                    "opaque_asset_group_id": str(scene["opaque_asset_group_id"]),
                    "category": str(scene["category"]),
                    "anomaly_families": "+".join(sorted(map(str, scene["anomaly_families"]))),
                    "object_count": int(scene["object_count"]),
                    "anomaly_count": int(scene["anomaly_count"]),
                    "route": route,
                    "method": method,
                    "view_count": view_count,
                    "view_ids": "+".join(views),
                    "fusion_method": fusion_method,
                    "rank_disagreement": disagreement,
                    "confidence_margin": margin,
                    "top_object_id": object_ids[0],
                    **metrics.to_dict(),
                }
                scene_rows.append(record)
                for rank, row in enumerate(fused, start=1):
                    fused_rows.append(
                        {
                            "opaque_scene_id": scene_id,
                            "route": route,
                            "method": method,
                            "view_count": view_count,
                            "fusion_method": fusion_method,
                            "opaque_object_id": row["opaque_object_id"],
                            "fused_score": row["fused_score"],
                            "rank": rank,
                            "view_support": row["view_support"],
                            "detected_view_support": row["detected_view_support"],
                            "scored_view_support": row["scored_view_support"],
                            "unscoreable_detected_view_count": row[
                                "unscoreable_detected_view_count"
                            ],
                        }
                    )
    return scene_rows, fused_rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[column]) for column in columns)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        item: dict[str, Any] = dict(zip(columns, key, strict=True))
        item["scene_count"] = len(values)
        for metric in ("top1", "reciprocal_rank", "average_precision", "recall_at_2", "exact_set", "rank_disagreement", "confidence_margin"):
            item[metric] = float(np.mean([float(row[metric]) for row in values]))
        output.append(item)
    return output


def _unscoreable_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report missing ranking evidence without exposing evaluator identities."""

    output = []
    for route in ROUTES:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["route"] == route and row["method"] == method
            ]
            status_counts: dict[str, int] = defaultdict(int)
            for row in selected:
                status_counts[str(row["scoring_status"])] += 1
            unscoreable = sum(
                count
                for status, count in status_counts.items()
                if status != "scored"
            )
            output.append(
                {
                    "route": route,
                    "method": method,
                    "candidate_view_count": len(selected),
                    "scored_candidate_view_count": status_counts["scored"],
                    "unscoreable_candidate_view_count": unscoreable,
                    "invalid_mask_candidate_view_count": status_counts["invalid_mask"],
                    "insufficient_peer_candidate_view_count": status_counts[
                        "insufficient_peers"
                    ],
                    "missing_proposal_candidate_view_count": status_counts[
                        "missing_proposal"
                    ],
                    "other_unscoreable_candidate_view_count": unscoreable
                    - status_counts["invalid_mask"]
                    - status_counts["insufficient_peers"]
                    - status_counts["missing_proposal"],
                }
            )
    return output


def _mask_metrics(
    geometry_root: Path,
    sam_results_path: Path,
    automatic_sam_results_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: dict[tuple[str, str, str], np.ndarray] = {}
    attributes: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in read_jsonl(geometry_root / "reference_manifest.jsonl"):
        if not bool(row["visible"]):
            continue
        path = geometry_root / "masks" / str(row["mask_path"])
        with Image.open(path) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
        key = (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["opaque_object_id"]))
        references[key] = mask
        attributes[key] = row
    records = []
    seen: set[tuple[str, str, str]] = set()
    for image in read_jsonl(sam_results_path):
        for item in image["routes"]["reference_box"]:
            key = (str(image["opaque_scene_id"]), str(image["view_id"]), str(item["prompt_key"]))
            if key not in references or key in seen:
                raise ValueError("DS2 SAM evaluator identity is missing or duplicated")
            seen.add(key)
            predicted = decode_binary_mask(item["mask_rle"])
            reference = references[key]
            boundary = boundary_precision_recall_f1(reference, predicted, tolerance_pixels=3)
            records.append(
                {
                    "opaque_scene_id": key[0],
                    "view_id": key[1],
                    "opaque_object_id": key[2],
                    "route": "reference_box",
                    "detection_status": "reference_prompt",
                    "mask_iou": binary_mask_iou(reference, predicted),
                    "dice": binary_mask_dice(reference, predicted),
                    "boundary_f1": boundary.f1,
                    "foreground_pixels": int(reference.sum()),
                    "predicted_quality_score": float(item["predicted_quality_score"]),
                }
            )
    if seen != set(references):
        raise ValueError("DS2 SAM evaluation does not cover every supplied object mask")
    reference_records = list(records)
    references_by_image: dict[tuple[str, str], list[tuple[str, Mapping[str, Any], np.ndarray]]] = defaultdict(list)
    for key, mask in references.items():
        references_by_image[key[:2]].append((key[2], attributes[key], mask))
    automatic_matched = 0
    automatic_predictions = 0
    for image in read_jsonl(automatic_sam_results_path):
        image_key = (str(image["opaque_scene_id"]), str(image["view_id"]))
        reference_items = sorted(references_by_image[image_key], key=lambda item: item[0])
        predicted_items = sorted(
            image["routes"]["automatic_box"], key=lambda item: str(item["prompt_key"])
        )
        automatic_predictions += len(predicted_items)
        matches = match_boxes(
            [item[1]["box_xyxy"] for item in reference_items],
            [item["box_xyxy"] for item in predicted_items],
            minimum_iou=0.50,
        )
        prediction_by_reference = {match.reference_index: match for match in matches}
        automatic_matched += len(matches)
        for reference_index, (object_id, _geometry, reference) in enumerate(reference_items):
            match = prediction_by_reference.get(reference_index)
            if match is None:
                records.append(
                    {
                        "opaque_scene_id": image_key[0],
                        "view_id": image_key[1],
                        "opaque_object_id": object_id,
                        "route": "automatic_box",
                        "detection_status": "missed_reference",
                        "mask_iou": 0.0,
                        "dice": 0.0,
                        "boundary_f1": 0.0,
                        "foreground_pixels": int(reference.sum()),
                        "predicted_quality_score": None,
                    }
                )
                continue
            predicted_item = predicted_items[match.proposal_index]
            predicted = decode_binary_mask(predicted_item["mask_rle"])
            boundary = boundary_precision_recall_f1(reference, predicted, tolerance_pixels=3)
            records.append(
                {
                    "opaque_scene_id": image_key[0],
                    "view_id": image_key[1],
                    "opaque_object_id": object_id,
                    "route": "automatic_box",
                    "detection_status": "matched_reference",
                    "mask_iou": binary_mask_iou(reference, predicted),
                    "dice": binary_mask_dice(reference, predicted),
                    "boundary_f1": boundary.f1,
                    "foreground_pixels": int(reference.sum()),
                    "predicted_quality_score": float(predicted_item["predicted_quality_score"]),
                }
            )

    ious = np.asarray([float(row["mask_iou"]) for row in reference_records])
    automatic_records = [row for row in records if row["route"] == "automatic_box"]
    automatic_ious = np.asarray([float(row["mask_iou"]) for row in automatic_records])
    summary = [
        {
            "route": "reference_box",
            "object_view_count": len(reference_records),
            "proposal_count": len(reference_records),
            "matched_count_at_box_iou_0_50": len(reference_records),
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "recall_at_0_75": float(np.mean(ious >= 0.75)),
            "mean_dice": float(np.mean([float(row["dice"]) for row in reference_records])),
            "mean_boundary_f1": float(np.mean([float(row["boundary_f1"]) for row in reference_records])),
        },
        {
            "route": "automatic_box",
            "object_view_count": len(automatic_records),
            "proposal_count": automatic_predictions,
            "matched_count_at_box_iou_0_50": automatic_matched,
            "mean_iou": float(np.mean(automatic_ious)),
            "median_iou": float(np.median(automatic_ious)),
            "recall_at_0_75": float(np.mean(automatic_ious >= 0.75)),
            "mean_dice": float(np.mean([float(row["dice"]) for row in automatic_records])),
            "mean_boundary_f1": float(np.mean([float(row["boundary_f1"]) for row in automatic_records])),
        },
    ]
    return records, summary


def _paired_intervals(scene_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[(str(row["route"]), str(row["method"]), int(row["view_count"]), str(row["fusion_method"]))].append(row)
    output = []
    comparisons = [
        ("p1_vs_b0", ("supplied_mask", "B0", 20, "mean_rank"), ("supplied_mask", "P1", 20, "mean_rank")),
        ("reference_vs_supplied", ("supplied_mask", "P1", 20, "mean_rank"), ("reference_box", "P1", 20, "mean_rank")),
        ("automatic_vs_reference", ("reference_box", "P1", 20, "mean_rank"), ("automatic_box", "P1", 20, "mean_rank")),
        ("three_vs_twenty_views", ("reference_box", "P1", 20, "mean_rank"), ("reference_box", "P1", 3, "mean_rank")),
        ("visibility_vs_mean", ("reference_box", "P1", 3, "mean_rank"), ("reference_box", "P1", 3, "visibility_weighted_rank")),
    ]
    for name, baseline, candidate in comparisons:
        interval = paired_clustered_difference(
            groups[baseline],
            groups[candidate],
            metric="average_precision",
            repetitions=10_000,
            seed=32_040,
        )
        output.append(
            {
                "comparison": name,
                "baseline": "|".join(map(str, baseline)),
                "candidate": "|".join(map(str, candidate)),
                "metric": "average_precision",
                **interval,
            }
        )
    return output


def _subgroups(scene_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row for row in scene_rows
        if row["route"] == "automatic_box"
        and row["method"] == "P1"
        and int(row["view_count"]) == PRIMARY_VIEW_COUNT
    ]
    output = []
    for dimension in ("category", "anomaly_families", "object_count", "anomaly_count"):
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(str(row[dimension]), str(row["fusion_method"]))].append(row)
        for (value, fusion), values in sorted(grouped.items()):
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "fusion_method": fusion,
                    "scene_count": len(values),
                    "average_precision": float(np.mean([float(row["average_precision"]) for row in values])),
                    "top1": float(np.mean([float(row["top1"]) for row in values])),
                    "exact_set": float(np.mean([float(row["exact_set"]) for row in values])),
                }
            )
    return output


def _primary_summary(
    aggregate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the report-facing rows for the fixed three-view application condition."""

    return [
        dict(row)
        for row in aggregate
        if row["route"] == "automatic_box"
        and row["method"] == "P1"
        and int(row["view_count"]) == PRIMARY_VIEW_COUNT
    ]


def _scene_type_summary(
    scene_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Separate one-anomaly and two-anomaly results for the primary route."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        if (
            row["route"] == "automatic_box"
            and row["method"] == "P1"
            and int(row["view_count"]) == PRIMARY_VIEW_COUNT
        ):
            grouped[(int(row["anomaly_count"]), str(row["fusion_method"]))].append(row)
    output = []
    for (anomaly_count, fusion), values in sorted(grouped.items()):
        output.append(
            {
                "anomaly_count": anomaly_count,
                "fusion_method": fusion,
                "scene_count": len(values),
                "top1": float(np.mean([float(row["top1"]) for row in values])),
                "average_precision": float(np.mean([float(row["average_precision"]) for row in values])),
                "recall_at_2": float(np.mean([float(row["recall_at_2"]) for row in values])),
                "exact_set": float(np.mean([float(row["exact_set"]) for row in values])),
            }
        )
    return output


def _proposal_summary(
    association_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarise frozen automatic proposals across all DS2 images."""

    if not association_rows:
        raise ValueError("DS2 proposal evaluation requires image-level associations")
    reference_count = sum(int(row["reference_count"]) for row in association_rows)
    proposal_count = sum(int(row["proposal_count"]) for row in association_rows)
    matched_count = sum(int(row["matched_count"]) for row in association_rows)
    missed_count = reference_count - matched_count
    extra_count = proposal_count - matched_count
    precision = matched_count / proposal_count if proposal_count else 0.0
    recall = matched_count / reference_count if reference_count else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    complete_images = sum(int(row["miss_count"]) == 0 for row in association_rows)
    terminal_failures = sum(row["source_status"] == "error" for row in association_rows)
    genuine_zero_proposals = sum(
        row["source_status"] == "ok" and int(row["proposal_count"]) == 0
        for row in association_rows
    )
    return [
        {
            "route": "automatic_box",
            "image_count": len(association_rows),
            "reference_object_count": reference_count,
            "proposal_count": proposal_count,
            "matched_count_at_iou_0_50": matched_count,
            "missed_object_count": missed_count,
            "extra_proposal_count": extra_count,
            "precision_at_iou_0_50": precision,
            "recall_at_iou_0_50": recall,
            "f1_at_iou_0_50": f1,
            "complete_detection_image_count": complete_images,
            "complete_detection_rate": complete_images / len(association_rows),
            "terminal_failed_image_count": terminal_failures,
            "terminal_failure_zero_proposal_image_count": sum(
                bool(row["carried_as_zero_proposal"]) for row in association_rows
            ),
            "genuine_zero_proposal_image_count": genuine_zero_proposals,
            "discarded_untrusted_box_count": sum(
                int(row["discarded_untrusted_box_count"])
                for row in association_rows
            ),
        }
    ]


def _plot_primary_metrics(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    metrics = ("top1", "average_precision", "recall_at_2", "exact_set")
    positions = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for index, row in enumerate(sorted(rows, key=lambda value: str(value["fusion_method"]))):
        offset = (index - (len(rows) - 1) / 2) * width
        axis.bar(
            positions + offset,
            [float(row[metric]) for metric in metrics],
            width,
            label=str(row["fusion_method"]).replace("_", " "),
        )
    axis.set_xticks(positions, ["Top-1", "Scene AP", "Recall@2", "Exact set"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean score")
    axis.set_title("DS2 three-view application metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_scene_types(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["fusion_method"] == "mean_rank"]
    labels = [
        "1 anomaly" if int(row["anomaly_count"]) == 1 else f"{row['anomaly_count']} anomalies"
        for row in selected
    ]
    positions = np.arange(len(selected))
    width = 0.22
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for index, (metric, label) in enumerate(
        (("top1", "Top-1"), ("average_precision", "Scene AP"), ("recall_at_2", "Recall@2"))
    ):
        axis.bar(positions + (index - 1) * width, [float(row[metric]) for row in selected], width, label=label)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean score")
    axis.set_title("DS2 results by anomaly count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_lowest_subgroups(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = sorted(
        (
            row
            for row in rows
            if row["fusion_method"] == "mean_rank" and int(row["scene_count"]) >= 5
        ),
        key=lambda row: (float(row["average_precision"]), str(row["dimension"]), str(row["value"])),
    )[:10]
    labels = [f"{row['dimension']}={row['value']} (n={row['scene_count']})" for row in selected]
    values = [float(row["average_precision"]) for row in selected]
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    positions = np.arange(len(selected))
    axis.barh(positions, values)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean scene average precision")
    axis.set_title("Lowest supported DS2 subgroups")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _example_panel(
    *,
    rgb_root: Path,
    geometry_root: Path,
    scene_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    protected_scenes: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    selected_rows = [
        row for row in scene_rows
        if row["route"] == "automatic_box"
        and row["method"] == "P1"
        and int(row["view_count"]) == PRIMARY_VIEW_COUNT
        and row["fusion_method"] == "mean_rank"
    ]
    examples = [min(selected_rows, key=lambda row: (float(row["average_precision"]), str(row["opaque_scene_id"]))), max(selected_rows, key=lambda row: (float(row["average_precision"]), str(row["opaque_scene_id"])))]
    score_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in score_rows:
        if row["route"] == "automatic_box" and row["method"] == "P1":
            score_index[(str(row["opaque_scene_id"]), str(row["view_id"]), str(row["opaque_object_id"]))] = row
    geometry: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(geometry_root / "reference_manifest.jsonl"):
        if bool(row["visible"]):
            geometry[(str(row["opaque_scene_id"]), str(row["view_id"]))].append(row)
    views = fixed_camera_spread_views(PRIMARY_VIEW_COUNT)
    panel = Image.new("RGB", (256 * len(views), 300 * len(examples)), "white")
    draw = ImageDraw.Draw(panel)
    for row_number, example in enumerate(examples):
        scene_id = str(example["opaque_scene_id"])
        labels = protected_scenes[scene_id]["labels_by_object_id"]
        for column, view_id in enumerate(views):
            image_path = rgb_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            image = np.asarray(Image.open(image_path).convert("RGB")).copy()
            for item in geometry[(scene_id, view_id)]:
                object_id = str(item["opaque_object_id"])
                score = score_index[(scene_id, view_id, object_id)]
                mask_path = geometry_root / "masks" / str(item["mask_path"])
                mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
                edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.FIND_EDGES)) > 0
                colour = (230, 45, 65) if int(labels[object_id]) else (40, 145, 85)
                image[edge] = np.asarray(colour, dtype=np.uint8)
                x1, y1, _x2, _y2 = [int(value) for value in item["box_xyxy"]]
                # Score labels are added after converting back to a PIL image.
            tile = Image.fromarray(image)
            tile_draw = ImageDraw.Draw(tile)
            for item in geometry[(scene_id, view_id)]:
                object_id = str(item["opaque_object_id"])
                score = score_index[(scene_id, view_id, object_id)]
                x1, y1, _x2, _y2 = [int(value) for value in item["box_xyxy"]]
                if score.get("association_status") == "missed_reference":
                    score_label = "missed"
                elif score.get("scoring_status") == "invalid_mask":
                    score_label = "invalid mask"
                elif score.get("score") is None:
                    score_label = "unscored"
                else:
                    score_label = f"{float(score['score']):.2f}"
                tile_draw.text(
                    (x1, max(0, y1 - 12)),
                    f"{object_id}:{score_label}",
                    fill="black",
                )
            x = column * 256
            y = row_number * 300
            panel.paste(tile, (x, y))
            draw.text((x + 4, y + 260), f"{scene_id} {view_id}", fill="black")
        label = "failure example" if row_number == 0 else "successful example"
        draw.text((4, row_number * 300 + 278), f"{label}, AP {float(example['average_precision']):.3f}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def _plot_view_curve(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["route"] == "automatic_box" and row["method"] == "P1"
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for fusion in FUSION_METHODS:
        values = sorted(
            (row for row in selected if row["fusion_method"] == fusion),
            key=lambda row: int(row["view_count"]),
        )
        axis.plot(
            [int(row["view_count"]) for row in values],
            [float(row["average_precision"]) for row in values],
            marker="o",
            label=fusion.replace("_", " "),
        )
    axis.set_xticks(VIEW_COUNTS)
    axis.set_xlabel("Number of camera-spread views")
    axis.set_ylabel("Mean scene average precision")
    axis.set_title("DS2 multi-view ablation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _select_decision(
    *,
    aggregate: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    mask_summary: Sequence[Mapping[str, Any]],
    subgroup_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del aggregate, intervals, mask_summary, subgroup_rows
    gates = [
        {"gate": "frozen predictions evaluated", "passed": True},
        {"gate": "all three routes reported", "passed": True},
        {"gate": "no DS2 tuning or accuracy cutoff applied", "passed": True},
    ]
    decision = {
        "schema_version": 1,
        "encoder": "dinov2_small",
        "scorer": "P1",
        "scientific_anchor_route": "supplied_mask",
        "reference_diagnostic_route": "reference_box",
        "application_mask_route": "automatic_box",
        "primary_view_count": PRIMARY_VIEW_COUNT,
        "primary_view_ids": list(fixed_camera_spread_views(PRIMARY_VIEW_COUNT)),
        "fusion_method": "mean_rank",
        "status": "confirmation_measured",
        "accuracy_acceptance_gate_applied": False,
        "ds2_used_for_tuning": False,
        "claim_boundary": "Development confirmation only. Calibration and locked evaluation remain unopened.",
    }
    return decision, gates


def run_ds2_protected_evaluation(
    *,
    rgb_root: Path,
    geometry_root: Path,
    attributes_root: Path,
    sam_results_path: Path,
    automatic_sam_results_path: Path,
    representation_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Open protected labels only after predictions are complete and hash locked."""

    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    verify_ds2_geometry_package(geometry_root)
    verify_ds2_attributes_package(attributes_root)
    score_package_path = representation_root / "model-input" / "package_record.json"
    score_record = json.loads(score_package_path.read_text(encoding="utf-8"))
    score_manifest_path = representation_root / "model-input" / "score_manifest.jsonl"
    if (
        score_record.get("cohort_id") != DS2_CONFIRM_CONTRACT.cohort_id
        or score_record.get("evaluator_opened") is not False
        or score_record.get("score_manifest_sha256") != file_sha256(score_manifest_path)
        or score_record.get("all_checks_passed") is not True
    ):
        raise ValueError("DS2 score package is not a frozen label-free input")
    frozen_predictions = {
        "score_package_sha256": file_sha256(score_package_path),
        "score_manifest_sha256": file_sha256(score_manifest_path),
        "sam_results_sha256": file_sha256(sam_results_path),
        "automatic_sam_results_sha256": file_sha256(automatic_sam_results_path),
    }

    protected_rows = read_jsonl(attributes_root / "protected_scene_attributes.jsonl")
    protected_scenes = {str(row["opaque_scene_id"]): row for row in protected_rows}
    if len(protected_scenes) != DS2_CONFIRM_CONTRACT.scene_count:
        raise ValueError("DS2 evaluator attributes have incomplete scene coverage")
    model_score_rows = read_jsonl(score_manifest_path)
    reference_rows = read_jsonl(geometry_root / "reference_manifest.jsonl")
    automatic_sam_rows = read_jsonl(automatic_sam_results_path)
    score_rows, association_rows = prepare_missing_aware_evaluator_scores(
        model_score_rows,
        reference_rows,
        automatic_sam_rows,
    )
    scene_rows, fused_rows = evaluate_multiview_scores(score_rows, protected_scenes)
    unscoreable_summary = _unscoreable_summary(score_rows)
    aggregate = _aggregate(scene_rows, ("route", "method", "view_count", "fusion_method"))
    intervals = _paired_intervals(scene_rows)
    subgroups = _subgroups(scene_rows)
    primary_summary = _primary_summary(aggregate)
    scene_type_summary = _scene_type_summary(scene_rows)
    mask_records, mask_summary = _mask_metrics(
        geometry_root,
        sam_results_path,
        automatic_sam_results_path,
    )
    decision, gates = _select_decision(
        aggregate=aggregate,
        intervals=intervals,
        mask_summary=mask_summary,
        subgroup_rows=subgroups,
    )
    proposal_summary = _proposal_summary(association_rows)

    _write_csv(safe_root / "multiview_summary.csv", aggregate)
    _write_csv(safe_root / "paired_intervals.csv", intervals)
    _write_csv(safe_root / "subgroup_summary.csv", subgroups)
    _write_csv(safe_root / "primary_three_view_summary.csv", primary_summary)
    _write_csv(safe_root / "scene_type_summary.csv", scene_type_summary)
    _write_csv(safe_root / "mask_summary.csv", mask_summary)
    _write_csv(safe_root / "proposal_summary.csv", proposal_summary)
    _write_csv(safe_root / "unscoreable_summary.csv", unscoreable_summary)
    _write_csv(safe_root / "decision_gates.csv", gates)
    _write_json(safe_root / "model_and_route_decision.json", decision)
    _write_json(safe_root / "frozen_prediction_identity.json", frozen_predictions)
    _write_jsonl(private_root / "scene_metrics.jsonl", scene_rows)
    _write_jsonl(private_root / "fused_object_scores.jsonl", fused_rows)
    _write_jsonl(private_root / "mask_metrics.jsonl", mask_records)
    _write_jsonl(private_root / "automatic_proposal_associations.jsonl", association_rows)

    selected_curve = [
        row for row in aggregate
        if row["route"] == "automatic_box" and row["method"] == "P1"
    ]
    _plot_view_curve(selected_curve, safe_root / "multiview_ablation.png")
    route_rows = [
        row for row in aggregate
        if row["method"] == "P1" and int(row["view_count"]) == 20 and row["fusion_method"] == "mean_rank"
    ]
    save_bar_chart([row["route"] for row in route_rows], [row["average_precision"] for row in route_rows], safe_root / "route_average_precision.png", title="DS2 P1 average precision by route", y_label="Mean scene AP", show_values=True)
    save_bar_chart(
        ["precision", "recall", "F1", "complete images"],
        [
            proposal_summary[0]["precision_at_iou_0_50"],
            proposal_summary[0]["recall_at_iou_0_50"],
            proposal_summary[0]["f1_at_iou_0_50"],
            proposal_summary[0]["complete_detection_rate"],
        ],
        safe_root / "locateanything_proposal_quality.png",
        title="DS2 LocateAnything proposal quality at box IoU 0.50",
        y_label="Rate",
        show_values=True,
    )
    save_bar_chart(["IoU", "Dice", "Boundary F1"], [mask_summary[0]["mean_iou"], mask_summary[0]["mean_dice"], mask_summary[0]["mean_boundary_f1"]], safe_root / "sam_mask_quality.png", title="DS2 reference-box SAM quality", y_label="Mean score", show_values=True)
    save_histogram([float(row["rank_disagreement"]) for row in scene_rows if row["route"] == "automatic_box" and row["method"] == "P1" and int(row["view_count"]) == PRIMARY_VIEW_COUNT and row["fusion_method"] == decision["fusion_method"]], safe_root / "rank_disagreement_distribution.png", title="Three-view rank disagreement", x_label="Mean object rank standard deviation", bins=24)
    save_histogram([float(row["mask_iou"]) for row in mask_records], safe_root / "sam_mask_iou_distribution.png", title="DS2 reference-box SAM mask IoU", x_label="Mask IoU", bins=28)
    _plot_primary_metrics(primary_summary, safe_root / "primary_three_view_metrics.png")
    _plot_scene_types(scene_type_summary, safe_root / "scene_type_metrics.png")
    _plot_lowest_subgroups(subgroups, safe_root / "lowest_supported_subgroups.png")
    _example_panel(rgb_root=rgb_root, geometry_root=geometry_root, scene_rows=scene_rows, score_rows=score_rows, protected_scenes=protected_scenes, output_path=private_root / "figures" / "three_view_success_and_failure.png")

    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "view_counts": list(VIEW_COUNTS),
        "fusion_methods": list(FUSION_METHODS),
        "scene_metric_row_count": len(scene_rows),
        "fused_object_row_count": len(fused_rows),
        "unscoreable_candidate_view_count": sum(
            int(row["unscoreable_candidate_view_count"])
            for row in unscoreable_summary
        ),
        "invalid_mask_candidate_view_count": sum(
            int(row["invalid_mask_candidate_view_count"])
            for row in unscoreable_summary
        ),
        "evaluator_opened_after_prediction_lock": True,
        "frozen_predictions": frozen_predictions,
        "decision": decision,
        "all_completion_gates_passed": all(bool(row["passed"]) for row in gates),
    }
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(safe_root)
    _artifact_manifest(private_root)
    return run_record


__all__ = [
    "FUSION_METHODS",
    "VIEW_COUNTS",
    "evaluate_multiview_scores",
    "fixed_camera_spread_views",
    "fuse_object_ranks",
    "prepare_missing_aware_evaluator_scores",
    "run_ds2_protected_evaluation",
]
