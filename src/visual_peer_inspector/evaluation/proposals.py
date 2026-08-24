"""Evaluator-only measurement for frozen object proposals.

This module joins model outputs to supplied-mask geometry only after inference.
It never changes, filters or reruns the proposal model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_MATCH_IOU = 0.50
SECONDARY_MATCH_IOU = 0.75
DIAGNOSTIC_MATCH_IOU = 0.25
DEFAULT_TOPOLOGY_OVERLAP = 0.10
DEFAULT_TOPOLOGY_COVERAGE = 0.50


@dataclass(frozen=True)
class ProposalMatch:
    """One deterministic one-to-one reference and proposal match."""

    reference_index: int
    proposal_index: int
    iou: float


def _normalise_box(value: Sequence[Any]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("Every box must contain four XYXY coordinates")
    box = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in box):
        raise ValueError("Box coordinates must be finite")
    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= 256.0 and 0.0 <= y1 < y2 <= 256.0):
        raise ValueError(f"Box is outside the 256 by 256 image: {box}")
    return box


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection_box(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float] | None:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate IoU for half-open XYXY boxes."""

    intersection = _intersection_box(left, right)
    intersection_area = 0.0 if intersection is None else _box_area(intersection)
    union = _box_area(left) + _box_area(right) - intersection_area
    return 0.0 if union <= 0.0 else intersection_area / union


def _union_area(rectangles: Sequence[Sequence[float]]) -> float:
    """Calculate the exact union area of a small set of rectangles."""

    if not rectangles:
        return 0.0
    xs = sorted({float(rectangle[index]) for rectangle in rectangles for index in (0, 2)})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        if left >= right:
            continue
        intervals = sorted(
            (float(rectangle[1]), float(rectangle[3]))
            for rectangle in rectangles
            if float(rectangle[0]) < right and float(rectangle[2]) > left
        )
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        total += (right - left) * covered_y
    return total


def match_boxes(
    reference_boxes: Sequence[Sequence[float]],
    proposal_boxes: Sequence[Sequence[float]],
    *,
    minimum_iou: float = DEFAULT_MATCH_IOU,
) -> list[ProposalMatch]:
    """Match boxes by eligible cardinality first and total IoU second."""

    if not 0.0 < float(minimum_iou) <= 1.0:
        raise ValueError("minimum_iou must be in (0, 1]")
    references = [_normalise_box(value) for value in reference_boxes]
    proposals = [_normalise_box(value) for value in proposal_boxes]
    if not references or not proposals:
        return []

    row_count = len(references)
    column_count = len(proposals)
    size = max(row_count, column_count)
    scores = np.zeros((size, size), dtype=np.float64)
    ious = np.zeros((row_count, column_count), dtype=np.float64)
    cardinality_bonus = float(size + 1)
    for reference_index, reference_box in enumerate(references):
        for proposal_index, proposal_box in enumerate(proposals):
            value = box_iou(reference_box, proposal_box)
            ious[reference_index, proposal_index] = value
            if value >= minimum_iou:
                tie_break = (reference_index + 1) * (proposal_index + 1) * 1e-12
                scores[reference_index, proposal_index] = (
                    cardinality_bonus + value - tie_break
                )

    selected_rows, selected_columns = linear_sum_assignment(scores, maximize=True)
    matches = [
        ProposalMatch(
            reference_index=int(reference_index),
            proposal_index=int(proposal_index),
            iou=float(ious[reference_index, proposal_index]),
        )
        for reference_index, proposal_index in zip(selected_rows, selected_columns)
        if reference_index < row_count
        and proposal_index < column_count
        and ious[reference_index, proposal_index] >= minimum_iou
    ]
    return sorted(matches, key=lambda item: (item.reference_index, item.proposal_index))


def _topology_counts(
    reference_boxes: Sequence[Sequence[float]],
    proposal_boxes: Sequence[Sequence[float]],
    *,
    matched_reference_ids: set[int],
    matched_proposal_ids: set[int],
    overlap_threshold: float,
    coverage_threshold: float,
) -> tuple[int, int]:
    merge_count = 0
    for proposal_index, proposal in enumerate(proposal_boxes):
        covered_references: list[int] = []
        for reference_index, reference in enumerate(reference_boxes):
            intersection = _intersection_box(reference, proposal)
            covered = 0.0 if intersection is None else _box_area(intersection)
            if covered / _box_area(reference) >= coverage_threshold:
                covered_references.append(reference_index)
        merge_count += int(
            len(covered_references) >= 2
            and (
                proposal_index not in matched_proposal_ids
                or any(index not in matched_reference_ids for index in covered_references)
            )
        )

    split_count = 0
    for reference in reference_boxes:
        clipped: list[tuple[float, float, float, float]] = []
        qualifying = 0
        reference_area = _box_area(reference)
        contributing_proposals: list[int] = []
        for proposal_index, proposal in enumerate(proposal_boxes):
            intersection = _intersection_box(reference, proposal)
            if intersection is None:
                continue
            coverage = _box_area(intersection) / reference_area
            if coverage >= overlap_threshold:
                qualifying += 1
                clipped.append(intersection)
                contributing_proposals.append(proposal_index)
        joint_coverage = _union_area(clipped) / reference_area
        split_count += int(
            qualifying >= 2
            and joint_coverage >= coverage_threshold
            and any(index not in matched_proposal_ids for index in contributing_proposals)
        )
    return merge_count, split_count


def _require_unique_rows(
    rows: Iterable[Mapping[str, Any]], *, name: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in indexed:
            raise ValueError(f"{name} contains a duplicate image key: {key}")
        indexed[key] = row
    return indexed


def evaluate_proposal_dataset(
    *,
    reference_rows: Iterable[Mapping[str, Any]],
    proposal_rows: Iterable[Mapping[str, Any]],
    protected_scene_rows: Iterable[Mapping[str, Any]] | None = None,
    minimum_iou: float = DEFAULT_MATCH_IOU,
    topology_overlap: float = DEFAULT_TOPOLOGY_OVERLAP,
    topology_coverage: float = DEFAULT_TOPOLOGY_COVERAGE,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate every frozen proposal without changing the prediction set."""

    references_by_image: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    reference_keys: set[tuple[str, str, str]] = set()
    for row in reference_rows:
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        if key in reference_keys:
            raise ValueError(f"Reference rows contain a duplicate object view: {key}")
        reference_keys.add(key)
        references_by_image[key[:2]].append(row)

    proposals_by_image = _require_unique_rows(proposal_rows, name="Proposal results")
    if set(references_by_image) != set(proposals_by_image):
        missing = sorted(set(references_by_image) - set(proposals_by_image))
        unexpected = sorted(set(proposals_by_image) - set(references_by_image))
        raise ValueError(
            f"Reference and proposal image coverage differs; missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )

    protected_by_scene: dict[str, Mapping[str, Any]] = {}
    if protected_scene_rows is not None:
        for row in protected_scene_rows:
            scene_id = str(row["opaque_scene_id"])
            if scene_id in protected_by_scene:
                raise ValueError("Protected attributes contain a duplicate scene")
            protected_by_scene[scene_id] = row
        expected_scenes = {key[0] for key in references_by_image}
        if set(protected_by_scene) != expected_scenes:
            raise ValueError("Protected attributes do not match the evaluated scenes")

    image_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    for image_key in sorted(references_by_image):
        scene_id, view_id = image_key
        reference_items = sorted(
            references_by_image[image_key], key=lambda row: str(row["opaque_object_id"])
        )
        visible_items = [row for row in reference_items if bool(row["visible"])]
        reference_boxes = [_normalise_box(row["box_xyxy"]) for row in visible_items]

        proposal_row = proposals_by_image[image_key]
        if str(proposal_row.get("status")) != "ok":
            raise ValueError(f"Frozen proposal row is not successful: {image_key}")
        proposal_boxes = [_normalise_box(box) for box in proposal_row["boxes_xyxy"]]
        if int(proposal_row["proposal_count"]) != len(proposal_boxes):
            raise ValueError(f"Proposal count disagrees with boxes: {image_key}")

        matches = match_boxes(
            reference_boxes, proposal_boxes, minimum_iou=minimum_iou
        )
        diagnostic_matches = match_boxes(
            reference_boxes, proposal_boxes, minimum_iou=DIAGNOSTIC_MATCH_IOU
        )
        secondary_matches = match_boxes(
            reference_boxes, proposal_boxes, minimum_iou=SECONDARY_MATCH_IOU
        )
        matched_reference_ids = {item.reference_index for item in matches}
        matched_proposal_ids = {item.proposal_index for item in matches}
        matched_at_secondary = len(secondary_matches)
        reference_count = len(reference_boxes)
        proposal_count = len(proposal_boxes)
        matched_count = len(matches)
        iou_sum = sum(item.iou for item in matches)
        attributes = protected_by_scene.get(scene_id, {})
        record = {
            "opaque_scene_id": scene_id,
            "view_id": view_id,
            "opaque_asset_group_id": attributes.get("opaque_asset_group_id"),
            "category": attributes.get("category"),
            "object_count": attributes.get("object_count", len(reference_items)),
            "anomaly_count": attributes.get("anomaly_count"),
            "anomaly_families": attributes.get("anomaly_families", []),
            "reference_count": reference_count,
            "proposal_count": proposal_count,
            "count_error": proposal_count - reference_count,
            "exact_count": int(proposal_count == reference_count),
            "matched_at_0_50": matched_count,
            "matched_at_0_25": len(diagnostic_matches),
            "matched_at_0_75": matched_at_secondary,
            "miss_count": reference_count - matched_count,
            "extra_count": proposal_count - matched_count,
            "precision_at_0_50": matched_count / proposal_count if proposal_count else 0.0,
            "recall_at_0_50": matched_count / reference_count if reference_count else 1.0,
            "recall_at_0_75": (
                matched_at_secondary / reference_count if reference_count else 1.0
            ),
            "mean_matched_iou": iou_sum / matched_count if matched_count else 0.0,
            "matched_ious": [item.iou for item in matches],
            "detection_aware_mean_iou": (
                iou_sum / reference_count if reference_count else 1.0
            ),
            "full_image_success": int(
                matched_count == reference_count and proposal_count == reference_count
            ),
            "merge_candidate_count": 0,
            "split_candidate_count": 0,
        }
        image_records.append(record)

        for match in matches:
            match_records.append(
                {
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "opaque_object_id": str(
                        visible_items[match.reference_index]["opaque_object_id"]
                    ),
                    "proposal_index": match.proposal_index,
                    "reference_box_xyxy": list(reference_boxes[match.reference_index]),
                    "proposal_box_xyxy": list(proposal_boxes[match.proposal_index]),
                    "box_iou": match.iou,
                }
            )
        for reference_index, item in enumerate(visible_items):
            if reference_index not in matched_reference_ids:
                best_iou = max(
                    (
                        box_iou(reference_boxes[reference_index], proposal)
                        for proposal in proposal_boxes
                    ),
                    default=0.0,
                )
                merge_candidate = False
                for proposal_index, proposal in enumerate(proposal_boxes):
                    covered_references = []
                    for candidate_index, reference in enumerate(reference_boxes):
                        intersection = _intersection_box(reference, proposal)
                        coverage = (
                            0.0
                            if intersection is None
                            else _box_area(intersection) / _box_area(reference)
                        )
                        if coverage >= topology_coverage:
                            covered_references.append(candidate_index)
                    if (
                        reference_index in covered_references
                        and len(covered_references) >= 2
                        and proposal_index in matched_proposal_ids
                    ):
                        merge_candidate = True
                        break
                if merge_candidate:
                    failure_type = "merge_candidate"
                elif best_iou >= DIAGNOSTIC_MATCH_IOU:
                    failure_type = "weak_localisation_miss"
                else:
                    failure_type = "complete_miss"
                failure_records.append(
                    {
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "failure_type": failure_type,
                        "opaque_object_id": str(item["opaque_object_id"]),
                        "proposal_index": None,
                        "best_iou": best_iou,
                    }
                )
        for proposal_index in range(proposal_count):
            if proposal_index not in matched_proposal_ids:
                best_iou = max(
                    (
                        box_iou(proposal_boxes[proposal_index], reference)
                        for reference in reference_boxes
                    ),
                    default=0.0,
                )
                split_candidate = False
                for reference_index, reference in enumerate(reference_boxes):
                    intersection = _intersection_box(
                        proposal_boxes[proposal_index], reference
                    )
                    proposal_coverage = (
                        0.0
                        if intersection is None
                        else _box_area(intersection)
                        / _box_area(proposal_boxes[proposal_index])
                    )
                    if (
                        proposal_coverage >= topology_coverage
                        and reference_index in matched_reference_ids
                    ):
                        split_candidate = True
                        break
                if split_candidate:
                    failure_type = "split_or_duplicate_candidate"
                elif best_iou >= DIAGNOSTIC_MATCH_IOU:
                    failure_type = "weak_localisation_extra"
                else:
                    failure_type = "background_extra"
                failure_records.append(
                    {
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "failure_type": failure_type,
                        "opaque_object_id": None,
                        "proposal_index": proposal_index,
                        "best_iou": best_iou,
                    }
                )

        image_failure_types = [
            row["failure_type"]
            for row in failure_records
            if row["opaque_scene_id"] == scene_id and row["view_id"] == view_id
        ]
        record["merge_candidate_count"] = image_failure_types.count("merge_candidate")
        record["split_candidate_count"] = image_failure_types.count(
            "split_or_duplicate_candidate"
        )

    return {
        "image_records": image_records,
        "match_records": match_records,
        "failure_records": failure_records,
    }


def summarize_proposal_evaluation(
    image_records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return the declared micro and image-macro proposal metrics."""

    if not image_records:
        raise ValueError("At least one evaluated image is required")
    reference_total = sum(int(row["reference_count"]) for row in image_records)
    proposal_total = sum(int(row["proposal_count"]) for row in image_records)
    matched_total = sum(int(row["matched_at_0_50"]) for row in image_records)
    matched_diagnostic_total = sum(
        int(row["matched_at_0_25"]) for row in image_records
    )
    matched_secondary_total = sum(int(row["matched_at_0_75"]) for row in image_records)
    matched_iou_sum = sum(
        float(row["mean_matched_iou"]) * int(row["matched_at_0_50"])
        for row in image_records
    )
    image_count = len(image_records)
    matched_ious = [
        float(value) for row in image_records for value in row["matched_ious"]
    ]
    count_errors = [int(row["count_error"]) for row in image_records]
    precision = matched_total / proposal_total if proposal_total else 0.0
    recall = matched_total / reference_total if reference_total else 1.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "image_count": image_count,
        "scene_count": len({str(row["opaque_scene_id"]) for row in image_records}),
        "reference_count": reference_total,
        "proposal_count": proposal_total,
        "matched_count_at_0_25": matched_diagnostic_total,
        "matched_count_at_0_50": matched_total,
        "matched_count_at_0_75": matched_secondary_total,
        "exact_count_rate": sum(int(row["exact_count"]) for row in image_records)
        / image_count,
        "count_mean_absolute_error": float(np.mean(np.abs(count_errors))),
        "count_signed_bias": float(np.mean(count_errors)),
        "full_image_success_rate": sum(
            int(row["full_image_success"]) for row in image_records
        )
        / image_count,
        "micro_precision_at_0_50": precision,
        "micro_recall_at_0_50": recall,
        "micro_f1_at_0_50": f1,
        "micro_recall_at_0_75": (
            matched_secondary_total / reference_total if reference_total else 1.0
        ),
        "mean_matched_iou": matched_iou_sum / matched_total if matched_total else 0.0,
        "median_matched_iou": float(np.median(matched_ious)) if matched_ious else 0.0,
        "p10_matched_iou": float(np.quantile(matched_ious, 0.10)) if matched_ious else 0.0,
        "detection_aware_mean_iou": (
            matched_iou_sum / reference_total if reference_total else 1.0
        ),
        "miss_count": reference_total - matched_total,
        "extra_count": proposal_total - matched_total,
        "images_with_misses": sum(int(row["miss_count"]) > 0 for row in image_records),
        "images_with_extras": sum(int(row["extra_count"]) > 0 for row in image_records),
        "merge_candidate_count": sum(
            int(row["merge_candidate_count"]) for row in image_records
        ),
        "split_candidate_count": sum(
            int(row["split_candidate_count"]) for row in image_records
        ),
    }


def clustered_bootstrap_intervals(
    image_records: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = 5_000,
    seed: int = 20260824,
) -> list[dict[str, float | int | str]]:
    """Bootstrap complete asset clusters and keep every view within a scene."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in image_records:
        cluster = row.get("opaque_asset_group_id")
        if cluster is None:
            raise ValueError("Asset-cluster IDs are required for clustered intervals")
        clusters[str(cluster)].append(row)
    if len(clusters) < 2:
        raise ValueError("At least two asset clusters are required")

    cluster_names = sorted(clusters)
    metric_names = [
        "exact_count_rate",
        "micro_precision_at_0_50",
        "micro_recall_at_0_50",
        "micro_recall_at_0_75",
        "mean_matched_iou",
        "detection_aware_mean_iou",
        "full_image_success_rate",
    ]
    point = summarize_proposal_evaluation(image_records)
    draws = {name: np.empty(repetitions, dtype=np.float64) for name in metric_names}
    generator = np.random.default_rng(int(seed))
    for repetition in range(repetitions):
        sampled = generator.choice(cluster_names, size=len(cluster_names), replace=True)
        rows = [row for cluster in sampled for row in clusters[str(cluster)]]
        summary = summarize_proposal_evaluation(rows)
        for name in metric_names:
            draws[name][repetition] = float(summary[name])

    return [
        {
            "metric": name,
            "point_estimate": float(point[name]),
            "ci_lower_95": float(np.quantile(draws[name], 0.025)),
            "ci_upper_95": float(np.quantile(draws[name], 0.975)),
            "bootstrap_repetitions": int(repetitions),
            "asset_cluster_count": len(cluster_names),
            "seed": int(seed),
        }
        for name in metric_names
    ]


__all__ = [
    "DEFAULT_MATCH_IOU",
    "DEFAULT_TOPOLOGY_COVERAGE",
    "DEFAULT_TOPOLOGY_OVERLAP",
    "DIAGNOSTIC_MATCH_IOU",
    "ProposalMatch",
    "SECONDARY_MATCH_IOU",
    "box_iou",
    "clustered_bootstrap_intervals",
    "evaluate_proposal_dataset",
    "match_boxes",
    "summarize_proposal_evaluation",
]
