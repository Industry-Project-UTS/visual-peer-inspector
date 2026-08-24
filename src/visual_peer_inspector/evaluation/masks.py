"""Evaluator-only metrics for frozen whole-object mask routes.

The helpers in this module operate on frozen masks after model inference. They
do not load models, alter predictions, or use labels to select outputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from visual_peer_inspector.tracking import validate_safe_tracking_payload


DEFAULT_BOUNDARY_TOLERANCE_PIXELS = 3
DEFAULT_MASK_IOU_THRESHOLD = 0.75
DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 32_040


@dataclass(frozen=True)
class BoundaryMetrics:
    """Boundary precision, recall and F1 for one mask pair."""

    precision: float
    recall: float
    f1: float
    reference_boundary_pixels: int
    prediction_boundary_pixels: int
    tolerance_pixels: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class MaskSpill:
    """Foreground pixels placed outside a half-open prompt box."""

    foreground_pixels: int
    outside_prompt_pixels: int
    outside_prompt_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _binary_mask(value: Any, *, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    if mask.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.isin(mask, (0, 1, False, True)).all():
        raise ValueError(f"{name} must contain only binary values")
    return mask.astype(bool, copy=False)


def _mask_pair(reference: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    reference_mask = _binary_mask(reference, name="reference mask")
    prediction_mask = _binary_mask(prediction, name="prediction mask")
    if reference_mask.shape != prediction_mask.shape:
        raise ValueError("Reference and prediction masks must have the same shape")
    return reference_mask, prediction_mask


def binary_mask_iou(reference: Any, prediction: Any) -> float:
    """Return binary-mask intersection over union.

    Two empty masks have IoU 1.0. A single empty mask has IoU 0.0.
    """

    reference_mask, prediction_mask = _mask_pair(reference, prediction)
    intersection = int(np.logical_and(reference_mask, prediction_mask).sum())
    union = int(np.logical_or(reference_mask, prediction_mask).sum())
    return 1.0 if union == 0 else intersection / union


def binary_mask_dice(reference: Any, prediction: Any) -> float:
    """Return the Sørensen-Dice score for two binary masks."""

    reference_mask, prediction_mask = _mask_pair(reference, prediction)
    intersection = int(np.logical_and(reference_mask, prediction_mask).sum())
    total = int(reference_mask.sum()) + int(prediction_mask.sum())
    return 1.0 if total == 0 else (2.0 * intersection) / total


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _boundary_tolerance(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boundary tolerance must be a non-negative integer")
    try:
        tolerance = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Boundary tolerance must be a non-negative integer"
        ) from error
    if tolerance != value or tolerance < 0:
        raise ValueError("Boundary tolerance must be a non-negative integer")
    return tolerance


def boundary_precision_recall_f1(
    reference: Any,
    prediction: Any,
    *,
    tolerance_pixels: int = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
) -> BoundaryMetrics:
    """Measure boundary agreement within a fixed Chebyshev pixel tolerance."""

    tolerance = _boundary_tolerance(tolerance_pixels)
    reference_mask, prediction_mask = _mask_pair(reference, prediction)
    reference_boundary = _mask_boundary(reference_mask)
    prediction_boundary = _mask_boundary(prediction_mask)
    reference_count = int(reference_boundary.sum())
    prediction_count = int(prediction_boundary.sum())
    if reference_count == 0 and prediction_count == 0:
        return BoundaryMetrics(1.0, 1.0, 1.0, 0, 0, tolerance)
    if reference_count == 0 or prediction_count == 0:
        return BoundaryMetrics(
            0.0,
            0.0,
            0.0,
            reference_count,
            prediction_count,
            tolerance,
        )

    if tolerance:
        structure = np.ones((3, 3), dtype=bool)
        reference_neighbourhood = binary_dilation(
            reference_boundary, structure=structure, iterations=tolerance
        )
        prediction_neighbourhood = binary_dilation(
            prediction_boundary, structure=structure, iterations=tolerance
        )
    else:
        reference_neighbourhood = reference_boundary
        prediction_neighbourhood = prediction_boundary
    precision = float(
        np.logical_and(prediction_boundary, reference_neighbourhood).sum()
        / prediction_count
    )
    recall = float(
        np.logical_and(reference_boundary, prediction_neighbourhood).sum()
        / reference_count
    )
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (
        precision + recall
    )
    return BoundaryMetrics(
        precision,
        recall,
        f1,
        reference_count,
        prediction_count,
        tolerance,
    )


def mask_spill_outside_prompt_box(
    prediction: Any, prompt_box_xyxy: Sequence[Any]
) -> MaskSpill:
    """Measure predicted foreground outside an integer half-open prompt box."""

    mask = _binary_mask(prediction, name="prediction mask")
    if len(prompt_box_xyxy) != 4:
        raise ValueError("Prompt box must contain four XYXY coordinates")
    coordinates: list[int] = []
    for value in prompt_box_xyxy:
        numeric = float(value)
        if not isfinite(numeric) or numeric != int(numeric):
            raise ValueError("Prompt box coordinates must be finite integers")
        coordinates.append(int(numeric))
    x1, y1, x2, y2 = coordinates
    height, width = mask.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Prompt box is outside the mask dimensions")
    foreground = int(mask.sum())
    inside = int(mask[y1:y2, x1:x2].sum())
    outside = foreground - inside
    fraction = 0.0 if foreground == 0 else outside / foreground
    return MaskSpill(foreground, outside, fraction)


def evaluate_mask_route_image(
    *,
    opaque_scene_id: str,
    view_id: str,
    route: str,
    reference_masks: Mapping[str, Any],
    prediction_rows: Iterable[Mapping[str, Any]],
    mask_iou_threshold: float = DEFAULT_MASK_IOU_THRESHOLD,
    boundary_tolerance_pixels: int = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
) -> list[dict[str, Any]]:
    """Retain matched, missed and unmatched outcomes for one route and image.

    Prediction rows require ``prediction_id``, ``matched_reference_id``,
    ``mask`` and ``prompt_box_xyxy``. Matching is frozen before this function.
    """

    if not str(opaque_scene_id) or not str(view_id) or not str(route):
        raise ValueError("Scene, view and route identifiers cannot be empty")
    threshold = float(mask_iou_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("mask_iou_threshold must be in (0, 1]")
    boundary_tolerance = _boundary_tolerance(boundary_tolerance_pixels)

    references: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for raw_identifier, value in reference_masks.items():
        identifier = str(raw_identifier)
        if not identifier or identifier in references:
            raise ValueError("Reference mask identifiers must be unique and non-empty")
        mask = _binary_mask(value, name=f"reference mask {identifier}")
        if not mask.any():
            raise ValueError("Visible reference masks cannot be empty")
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError("Reference masks must have the same shape")
        references[identifier] = mask
    if not references:
        raise ValueError("At least one visible reference mask is required")

    predictions: list[dict[str, Any]] = []
    prediction_ids: set[str] = set()
    matched_reference_ids: set[str] = set()
    for source in prediction_rows:
        prediction_id = str(source["prediction_id"])
        if not prediction_id or prediction_id in prediction_ids:
            raise ValueError("Prediction identifiers must be unique and non-empty")
        prediction_ids.add(prediction_id)
        mask = _binary_mask(source["mask"], name=f"prediction mask {prediction_id}")
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError("Prediction and reference masks must have the same shape")
        raw_match = source.get("matched_reference_id")
        matched_reference_id = None if raw_match is None else str(raw_match)
        if matched_reference_id is not None:
            if matched_reference_id not in references:
                raise ValueError("A prediction refers to an unknown reference mask")
            if matched_reference_id in matched_reference_ids:
                raise ValueError("More than one prediction is matched to one reference mask")
            matched_reference_ids.add(matched_reference_id)
        spill = mask_spill_outside_prompt_box(mask, source["prompt_box_xyxy"])
        predictions.append(
            {
                "prediction_id": prediction_id,
                "matched_reference_id": matched_reference_id,
                "mask": mask,
                "prompt_box_xyxy": [int(value) for value in source["prompt_box_xyxy"]],
                "spill": spill,
            }
        )

    by_reference = {
        item["matched_reference_id"]: item
        for item in predictions
        if item["matched_reference_id"] is not None
    }
    records: list[dict[str, Any]] = []
    common = {
        "opaque_scene_id": str(opaque_scene_id),
        "view_id": str(view_id),
        "route": str(route),
        "mask_iou_threshold": threshold,
        "boundary_tolerance_pixels": boundary_tolerance,
    }
    for reference_id in sorted(references):
        reference_mask = references[reference_id]
        prediction = by_reference.get(reference_id)
        if prediction is None:
            records.append(
                {
                    **common,
                    "record_kind": "missed_reference",
                    "opaque_object_id": reference_id,
                    "prediction_id": None,
                    "matched": 0,
                    "missed_reference": 1,
                    "unmatched_prediction": 0,
                    "reference_foreground_pixels": int(reference_mask.sum()),
                    "prediction_foreground_pixels": 0,
                    "mask_iou": 0.0,
                    "dice": 0.0,
                    "boundary_precision": 0.0,
                    "boundary_recall": 0.0,
                    "boundary_f1": 0.0,
                    "passes_mask_iou_threshold": 0,
                    "prompt_box_xyxy": None,
                    "outside_prompt_pixels": None,
                    "outside_prompt_fraction": None,
                }
            )
            continue
        prediction_mask = prediction["mask"]
        iou = binary_mask_iou(reference_mask, prediction_mask)
        dice = binary_mask_dice(reference_mask, prediction_mask)
        boundary = boundary_precision_recall_f1(
            reference_mask,
            prediction_mask,
            tolerance_pixels=boundary_tolerance,
        )
        spill = prediction["spill"]
        records.append(
            {
                **common,
                "record_kind": "matched_reference",
                "opaque_object_id": reference_id,
                "prediction_id": prediction["prediction_id"],
                "matched": 1,
                "missed_reference": 0,
                "unmatched_prediction": 0,
                "reference_foreground_pixels": int(reference_mask.sum()),
                "prediction_foreground_pixels": int(prediction_mask.sum()),
                "mask_iou": iou,
                "dice": dice,
                "boundary_precision": boundary.precision,
                "boundary_recall": boundary.recall,
                "boundary_f1": boundary.f1,
                "passes_mask_iou_threshold": int(iou >= threshold),
                "prompt_box_xyxy": prediction["prompt_box_xyxy"],
                "outside_prompt_pixels": spill.outside_prompt_pixels,
                "outside_prompt_fraction": spill.outside_prompt_fraction,
            }
        )

    for prediction in sorted(predictions, key=lambda item: item["prediction_id"]):
        if prediction["matched_reference_id"] is not None:
            continue
        prediction_mask = prediction["mask"]
        spill = prediction["spill"]
        records.append(
            {
                **common,
                "record_kind": "unmatched_prediction",
                "opaque_object_id": None,
                "prediction_id": prediction["prediction_id"],
                "matched": 0,
                "missed_reference": 0,
                "unmatched_prediction": 1,
                "reference_foreground_pixels": None,
                "prediction_foreground_pixels": int(prediction_mask.sum()),
                "mask_iou": None,
                "dice": None,
                "boundary_precision": None,
                "boundary_recall": None,
                "boundary_f1": None,
                "passes_mask_iou_threshold": None,
                "prompt_box_xyxy": prediction["prompt_box_xyxy"],
                "outside_prompt_pixels": spill.outside_prompt_pixels,
                "outside_prompt_fraction": spill.outside_prompt_fraction,
            }
        )
    return records


def summarize_mask_routes(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create aggregate route rows that are safe for reports and ClearML."""

    by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        kind = str(row["record_kind"])
        if kind not in {"matched_reference", "missed_reference", "unmatched_prediction"}:
            raise ValueError(f"Unknown mask-evaluation record kind: {kind}")
        by_route[str(row["route"])].append(row)
    if not by_route:
        raise ValueError("Mask route records cannot be empty")

    summaries: list[dict[str, Any]] = []
    for route in sorted(by_route):
        route_rows = by_route[route]
        threshold_values = {float(row["mask_iou_threshold"]) for row in route_rows}
        tolerance_values = {
            int(row["boundary_tolerance_pixels"]) for row in route_rows
        }
        if len(threshold_values) != 1 or len(tolerance_values) != 1:
            raise ValueError("A route contains mixed mask-evaluation settings")
        reference_rows = [
            row for row in route_rows if row["record_kind"] != "unmatched_prediction"
        ]
        matched_rows = [
            row for row in route_rows if row["record_kind"] == "matched_reference"
        ]
        prediction_rows = [
            row for row in route_rows if row["record_kind"] != "missed_reference"
        ]
        reference_count = len(reference_rows)
        matched_count = len(matched_rows)
        threshold_passes = sum(
            int(row["passes_mask_iou_threshold"]) for row in reference_rows
        )

        def matched_mean(key: str) -> float:
            return (
                float(np.mean([float(row[key]) for row in matched_rows]))
                if matched_rows
                else 0.0
            )

        def detection_aware_mean(key: str) -> float:
            return (
                float(sum(float(row[key]) for row in reference_rows) / reference_count)
                if reference_count
                else 1.0
            )

        spill_fractions = [
            float(row["outside_prompt_fraction"]) for row in prediction_rows
        ]
        summaries.append(
            {
                "route": route,
                "reference_count": reference_count,
                "prediction_count": len(prediction_rows),
                "matched_reference_count": matched_count,
                "missed_reference_count": reference_count - matched_count,
                "unmatched_prediction_count": sum(
                    int(row["unmatched_prediction"]) for row in route_rows
                ),
                "detection_aware_recall_at_0_75": (
                    threshold_passes / reference_count if reference_count else 1.0
                ),
                "mean_matched_iou": matched_mean("mask_iou"),
                "detection_aware_mean_iou": detection_aware_mean("mask_iou"),
                "mean_matched_dice": matched_mean("dice"),
                "detection_aware_mean_dice": detection_aware_mean("dice"),
                "mean_matched_boundary_f1": matched_mean("boundary_f1"),
                "mean_matched_boundary_precision": matched_mean(
                    "boundary_precision"
                ),
                "mean_matched_boundary_recall": matched_mean("boundary_recall"),
                "detection_aware_mean_boundary_f1": detection_aware_mean(
                    "boundary_f1"
                ),
                "detection_aware_mean_boundary_precision": detection_aware_mean(
                    "boundary_precision"
                ),
                "detection_aware_mean_boundary_recall": detection_aware_mean(
                    "boundary_recall"
                ),
                "mean_prediction_spill_fraction": (
                    float(np.mean(spill_fractions)) if spill_fractions else 0.0
                ),
                "prediction_spill_rate": (
                    sum(value > 0.0 for value in spill_fractions)
                    / len(spill_fractions)
                    if spill_fractions
                    else 0.0
                ),
                "mask_iou_threshold": next(iter(threshold_values)),
                "boundary_tolerance_pixels": next(iter(tolerance_values)),
            }
        )
    validate_safe_tracking_payload(summaries, "mask_route_summaries")
    return summaries


def separate_mask_evidence(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return private detail separately from safe aggregate route summaries."""

    private_records = [dict(row) for row in records]
    safe_summaries = summarize_mask_routes(private_records)
    return {
        "private_records": private_records,
        "safe_route_summaries": safe_summaries,
    }


def paired_asset_clustered_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
    scene_key: str = "opaque_scene_id",
    cluster_key: str = "opaque_asset_group_id",
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, int | float | str]:
    """Bootstrap paired scene means while resampling complete asset clusters."""

    if isinstance(repetitions, bool) or int(repetitions) != repetitions:
        raise ValueError("Bootstrap repetitions must be a positive integer")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be a positive integer")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    by_scene: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for row in rows:
        scene_id = str(row[scene_key])
        cluster_id = str(row[cluster_key])
        baseline = float(row[baseline_key])
        candidate = float(row[candidate_key])
        if not scene_id or not cluster_id:
            raise ValueError("Scene and cluster identifiers cannot be empty")
        if not isfinite(baseline) or not isfinite(candidate):
            raise ValueError("Paired bootstrap values must be finite")
        by_scene[scene_id].append((cluster_id, baseline, candidate))
    if not by_scene:
        raise ValueError("Paired bootstrap rows cannot be empty")

    scene_rows: list[tuple[str, str, float, float]] = []
    for scene_id in sorted(by_scene):
        values = by_scene[scene_id]
        clusters = {item[0] for item in values}
        if len(clusters) != 1:
            raise ValueError("One scene cannot belong to multiple asset clusters")
        scene_rows.append(
            (
                scene_id,
                next(iter(clusters)),
                float(np.mean([item[1] for item in values])),
                float(np.mean([item[2] for item in values])),
            )
        )

    cluster_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, cluster_id, baseline, candidate in scene_rows:
        cluster_rows[cluster_id].append((baseline, candidate))
    cluster_ids = sorted(cluster_rows)
    baseline_values = np.asarray([item[2] for item in scene_rows], dtype=np.float64)
    candidate_values = np.asarray([item[3] for item in scene_rows], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrap_differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        selected = [
            pair
            for cluster_index in sampled
            for pair in cluster_rows[cluster_ids[int(cluster_index)]]
        ]
        bootstrap_differences[index] = float(
            np.mean([baseline - candidate for baseline, candidate in selected])
        )
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(
        bootstrap_differences, [alpha, 1.0 - alpha], method="linear"
    )
    result: dict[str, int | float | str] = {
        "direction": "baseline_minus_candidate",
        "scene_count": len(scene_rows),
        "cluster_count": len(cluster_ids),
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_difference": float((baseline_values - candidate_values).mean()),
        "confidence_level": float(confidence_level),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "repetitions": repetitions,
        "seed": int(seed),
    }
    validate_safe_tracking_payload(result, "paired_clustered_bootstrap")
    return result


__all__ = [
    "BoundaryMetrics",
    "DEFAULT_BOOTSTRAP_REPETITIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_BOUNDARY_TOLERANCE_PIXELS",
    "DEFAULT_MASK_IOU_THRESHOLD",
    "MaskSpill",
    "binary_mask_dice",
    "binary_mask_iou",
    "boundary_precision_recall_f1",
    "evaluate_mask_route_image",
    "mask_spill_outside_prompt_box",
    "paired_asset_clustered_bootstrap",
    "separate_mask_evidence",
    "summarize_mask_routes",
]
