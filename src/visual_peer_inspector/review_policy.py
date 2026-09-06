"""Prediction-only review signals shared by calibration and deployed inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from visual_peer_inspector.data.model_safe_package import canonical_json_bytes

PREDICTION_SIGNAL_FIELDS = (
    "margin",
    "agreement",
    "minimum_mask_quality",
    "zero_proposal",
    "incomplete_proposals",
    "duplicate_or_fragmented_proposals",
    "empty_mask",
    "identity_unmatched_or_ambiguous",
    "stage_failure",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class ReviewPolicy:
    """Thresholds and fixed review rules fitted on calibration predictions."""

    minimum_margin: float
    minimum_agreement: float
    minimum_mask_quality: float
    review_on_zero_proposal: bool = True
    review_on_incomplete_proposals: bool = True
    review_on_duplicate_fragment: bool = True
    review_on_empty_mask: bool = True
    review_on_identity_warning: bool = True
    review_on_stage_failure: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_margin", self.minimum_margin),
            ("minimum_agreement", self.minimum_agreement),
            ("minimum_mask_quality", self.minimum_mask_quality),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "review_on_zero_proposal",
            "review_on_incomplete_proposals",
            "review_on_duplicate_fragment",
            "review_on_empty_mask",
            "review_on_identity_warning",
            "review_on_stage_failure",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ReviewPolicy:
        """Parse the exact policy fields and reject a partial deployment policy."""

        expected = set(cls.__dataclass_fields__)
        supplied = set(value)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"Review policy fields differ; missing={missing}, extra={extra}")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(asdict(self))).hexdigest()


def _box_overlap(box_a: Sequence[Any], box_b: Sequence[Any]) -> tuple[float, float]:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0, 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = area_a + area_b - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    containment = 0.0 if min(area_a, area_b) <= 0.0 else intersection / min(area_a, area_b)
    return float(iou), float(containment)


def _has_duplicate_or_fragment_geometry(objects: Sequence[Mapping[str, Any]]) -> bool:
    boxes = [item.get("box_xyxy") for item in objects]
    for left_index, left in enumerate(boxes):
        if not isinstance(left, Sequence) or isinstance(left, (str, bytes)):
            continue
        for right in boxes[left_index + 1 :]:
            if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
                continue
            iou, containment = _box_overlap(left, right)
            if iou >= 0.75 or containment >= 0.90:
                return True
    return False


def build_prediction_review_signals(
    decision: Mapping[str, Any],
    view_rows: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Derive every policy input from model output available in the web route."""

    expected_views = len(tuple(view_rows))
    present_rows = [row for row in view_rows if row is not None]
    stage_failures = sum(
        row is None
        or row.get("terminal_failure") is not None
        or row.get("status") == "terminal_failure"
        for row in view_rows
    )
    objects_by_view = [list(row.get("objects", [])) for row in present_rows]
    proposal_counts = [len(objects) for objects in objects_by_view]
    all_objects = [item for objects in objects_by_view for item in objects]
    qualities = [
        float(item.get("predicted_quality_score", 0.0)) for item in all_objects
    ]
    zero_proposal = any(count == 0 for count in proposal_counts) or not present_rows
    empty_mask = any(int(item.get("foreground_pixels", 0)) <= 0 for item in all_objects)
    duplicate_or_fragmented = any(
        _has_duplicate_or_fragment_geometry(objects) for objects in objects_by_view
    )
    tracks = list(decision.get("tracks", []))
    identity_warning = any(
        str(track.get("identity_status", "")) != "matched"
        or track.get("cycle_consistent") is not True
        for track in tracks
    )
    incomplete = len(present_rows) != expected_views or stage_failures > 0
    fused = decision.get("fused_odd_one_out")
    insufficient = (
        fused is None
        or bool(decision.get("insufficient_evidence"))
        or len(present_rows) < 2
        or stage_failures >= expected_views
        or zero_proposal
    )
    margin = decision.get("margin")
    agreement = decision.get("agreement")
    return {
        "margin": None if margin is None else float(margin),
        "agreement": None if agreement is None else float(agreement),
        "minimum_mask_quality": min(qualities) if qualities else 0.0,
        "zero_proposal": zero_proposal,
        "incomplete_proposals": incomplete,
        "duplicate_or_fragmented_proposals": duplicate_or_fragmented,
        "empty_mask": empty_mask,
        "identity_unmatched_or_ambiguous": identity_warning,
        "stage_failure": stage_failures > 0,
        "insufficient_evidence": insufficient,
    }


def apply_review_policy(
    record: Mapping[str, Any], policy: ReviewPolicy
) -> tuple[str, tuple[str, ...]]:
    """Assign the deployment policy state using prediction-only signals."""

    if bool(record.get("insufficient_evidence")) or record.get("fused_odd_one_out") is None:
        return "insufficient_evidence", ("insufficient_evidence",)
    reasons = []
    for field, enabled in (
        ("zero_proposal", policy.review_on_zero_proposal),
        ("incomplete_proposals", policy.review_on_incomplete_proposals),
        ("duplicate_or_fragmented_proposals", policy.review_on_duplicate_fragment),
        ("empty_mask", policy.review_on_empty_mask),
        ("identity_unmatched_or_ambiguous", policy.review_on_identity_warning),
        ("stage_failure", policy.review_on_stage_failure),
    ):
        if enabled and bool(record.get(field)):
            reasons.append(field)
    if float(record.get("minimum_mask_quality", 0.0)) < policy.minimum_mask_quality:
        reasons.append("low_quality_mask")
    margin = record.get("margin")
    if margin is None or float(margin) < policy.minimum_margin:
        reasons.append("small_p1_margin")
    agreement = record.get("agreement")
    if agreement is None or float(agreement) < policy.minimum_agreement:
        reasons.append("view_disagreement")
    if reasons:
        return "review_required", tuple(sorted(set(reasons)))
    return "result_ready", ()


__all__ = [
    "PREDICTION_SIGNAL_FIELDS",
    "ReviewPolicy",
    "apply_review_policy",
    "build_prediction_review_signals",
]
