"""Label-free object identity matching across uploaded views.

The input contract is deliberately narrow. It accepts detector-local candidate
tokens, DINO global embeddings, and optional geometry normalised to each
uploaded image. Source records, evaluator fields, paths, labels, roles,
categories, and camera metadata are not part of the schema.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

INPUT_SCHEMA_VERSION = "visual-peer-identity-input/v1"
RESULT_SCHEMA_VERSION = "visual-peer-identity-result/v1"
MINIMUM_VIEW_COUNT = 2
MAXIMUM_VIEW_COUNT = 5

_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_FIELD_TERMS = (
    "anomaly",
    "base_asset",
    "camera",
    "category",
    "crosswalk",
    "evaluator",
    "filename",
    "label",
    "object_index",
    "object_order",
    "path",
    "permutation",
    "role",
    "source",
)


def _validate_exact_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    unknown = sorted(keys - allowed, key=str)
    if not unknown:
        return
    forbidden = [
        key
        for key in unknown
        if any(term in str(key).lower() for term in _FORBIDDEN_FIELD_TERMS)
    ]
    if forbidden:
        raise ValueError(f"{context} contains forbidden model-facing fields: {forbidden}")
    raise ValueError(f"{context} contains unknown fields: {unknown}")


def _validate_token(value: Any, *, field_name: str) -> str:
    token = str(value)
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError(
            f"{field_name} must be an opaque token without whitespace or path separators"
        )
    return token


def _finite_tuple(
    values: Sequence[Any],
    *,
    field_name: str,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a numeric sequence")
    try:
        result = tuple(float(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a numeric sequence") from error
    if expected_length is not None and len(result) != expected_length:
        raise ValueError(f"{field_name} must contain {expected_length} values")
    if not result or not all(np.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


@dataclass(frozen=True)
class NormalizedGeometry:
    """Optional deployment geometry expressed in uploaded-image coordinates."""

    box_xyxy_normalized: tuple[float, float, float, float]
    mask_centroid_xy_normalized: tuple[float, float] | None = None
    mask_area_fraction: float | None = None

    def __post_init__(self) -> None:
        box = _finite_tuple(
            self.box_xyxy_normalized,
            field_name="box_xyxy_normalized",
            expected_length=4,
        )
        x1, y1, x2, y2 = box
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("box_xyxy_normalized must lie within the unit square")
        object.__setattr__(self, "box_xyxy_normalized", box)

        if self.mask_centroid_xy_normalized is not None:
            centroid = _finite_tuple(
                self.mask_centroid_xy_normalized,
                field_name="mask_centroid_xy_normalized",
                expected_length=2,
            )
            if any(item < 0.0 or item > 1.0 for item in centroid):
                raise ValueError(
                    "mask_centroid_xy_normalized must lie within the unit square"
                )
            object.__setattr__(self, "mask_centroid_xy_normalized", centroid)

        if self.mask_area_fraction is not None:
            area = float(self.mask_area_fraction)
            if not np.isfinite(area) or not 0.0 < area <= 1.0:
                raise ValueError("mask_area_fraction must be in (0, 1]")
            object.__setattr__(self, "mask_area_fraction", area)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NormalizedGeometry:
        """Parse the strict, deployment-safe geometry mapping."""

        _validate_exact_fields(
            value,
            allowed={
                "box_xyxy_normalized",
                "mask_centroid_xy_normalized",
                "mask_area_fraction",
            },
            required={"box_xyxy_normalized"},
            context="geometry",
        )
        centroid = value.get("mask_centroid_xy_normalized")
        return cls(
            box_xyxy_normalized=tuple(value["box_xyxy_normalized"]),
            mask_centroid_xy_normalized=(
                None if centroid is None else tuple(centroid)
            ),
            mask_area_fraction=value.get("mask_area_fraction"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "box_xyxy_normalized": list(self.box_xyxy_normalized),
            "mask_centroid_xy_normalized": (
                None
                if self.mask_centroid_xy_normalized is None
                else list(self.mask_centroid_xy_normalized)
            ),
            "mask_area_fraction": self.mask_area_fraction,
        }


@dataclass(frozen=True)
class CandidateObservation:
    """One candidate described without dataset or evaluator identity."""

    candidate_id: str
    embedding: tuple[float, ...]
    geometry: NormalizedGeometry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _validate_token(self.candidate_id, field_name="candidate_id"),
        )
        embedding = _finite_tuple(self.embedding, field_name="embedding")
        if len(embedding) < 2:
            raise ValueError("embedding must contain at least two values")
        if float(np.linalg.norm(np.asarray(embedding, dtype=np.float64))) <= 0.0:
            raise ValueError("embedding must have non-zero norm")
        object.__setattr__(self, "embedding", embedding)
        if self.geometry is not None and not isinstance(
            self.geometry, NormalizedGeometry
        ):
            raise ValueError("geometry must be NormalizedGeometry or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CandidateObservation:
        """Parse one candidate while rejecting metadata outside the allowlist."""

        _validate_exact_fields(
            value,
            allowed={"candidate_id", "embedding", "geometry"},
            required={"candidate_id", "embedding"},
            context="candidate",
        )
        geometry_value = value.get("geometry")
        if geometry_value is not None and not isinstance(geometry_value, Mapping):
            raise ValueError("candidate geometry must be a mapping or null")
        return cls(
            candidate_id=value["candidate_id"],
            embedding=tuple(value["embedding"]),
            geometry=(
                None
                if geometry_value is None
                else NormalizedGeometry.from_mapping(geometry_value)
            ),
        )

    def to_record(self, *, include_embedding: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "geometry": None if self.geometry is None else self.geometry.to_record(),
        }
        if include_embedding:
            record["embedding"] = list(self.embedding)
        return record


@dataclass(frozen=True)
class ViewObservation:
    """The candidates detected in one uploaded view."""

    view_id: str
    candidates: tuple[CandidateObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "view_id",
            _validate_token(self.view_id, field_name="view_id"),
        )
        candidates = tuple(self.candidates)
        if any(not isinstance(item, CandidateObservation) for item in candidates):
            raise ValueError("view candidates must be CandidateObservation values")
        candidate_ids = [item.candidate_id for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"view {self.view_id} contains duplicate candidate IDs")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ViewObservation:
        """Parse one view and its strictly allowlisted candidate records."""

        _validate_exact_fields(
            value,
            allowed={"view_id", "candidates"},
            required={"view_id", "candidates"},
            context="view",
        )
        candidates = value["candidates"]
        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            raise TypeError("view candidates must be a sequence")
        parsed = []
        for item in candidates:
            if not isinstance(item, Mapping):
                raise TypeError("each candidate must be a mapping")
            parsed.append(CandidateObservation.from_mapping(item))
        return cls(view_id=value["view_id"], candidates=tuple(parsed))

    def to_record(self, *, include_embeddings: bool = False) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "candidates": [
                item.to_record(include_embedding=include_embeddings)
                for item in self.candidates
            ],
        }


def parse_model_facing_input(payload: Mapping[str, Any]) -> tuple[ViewObservation, ...]:
    """Parse a versioned input payload and reject all non-allowlisted metadata."""

    _validate_exact_fields(
        payload,
        allowed={"schema_version", "views"},
        required={"schema_version", "views"},
        context="identity input",
    )
    if payload["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {INPUT_SCHEMA_VERSION!r}"
        )
    view_values = payload["views"]
    if isinstance(view_values, (str, bytes)) or not isinstance(
        view_values, Sequence
    ):
        raise TypeError("views must be a sequence")
    views = []
    for value in view_values:
        if not isinstance(value, Mapping):
            raise TypeError("each view must be a mapping")
        views.append(ViewObservation.from_mapping(value))
    return validate_views(views)


def validate_views(views: Sequence[ViewObservation]) -> tuple[ViewObservation, ...]:
    """Return canonical view order after validating the 2-to-5-view contract."""

    supplied = tuple(views)
    if any(not isinstance(item, ViewObservation) for item in supplied):
        raise ValueError("views must contain ViewObservation values")
    canonical = tuple(sorted(supplied, key=lambda item: item.view_id))
    if not MINIMUM_VIEW_COUNT <= len(canonical) <= MAXIMUM_VIEW_COUNT:
        raise ValueError(
            f"identity matching requires {MINIMUM_VIEW_COUNT} to "
            f"{MAXIMUM_VIEW_COUNT} uploaded views"
        )
    view_ids = [item.view_id for item in canonical]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("duplicate view IDs are not allowed")
    dimensions = {
        len(candidate.embedding)
        for view in canonical
        for candidate in view.candidates
    }
    if len(dimensions) > 1:
        raise ValueError("all DINO embeddings must have the same dimension")
    return canonical


@dataclass(frozen=True)
class IdentityConfiguration:
    """Development-selected costs used for deterministic matching."""

    unmatched_cost: float = 0.35
    ambiguity_margin: float = 0.05
    geometry_weight: float = 0.0

    def __post_init__(self) -> None:
        unmatched = float(self.unmatched_cost)
        margin = float(self.ambiguity_margin)
        geometry_weight = float(self.geometry_weight)
        if not np.isfinite(unmatched) or not 0.0 < unmatched <= 3.0:
            raise ValueError("unmatched_cost must be in (0, 3]")
        if not np.isfinite(margin) or not 0.0 <= margin <= 3.0:
            raise ValueError("ambiguity_margin must be in [0, 3]")
        if not np.isfinite(geometry_weight) or not 0.0 <= geometry_weight <= 1.0:
            raise ValueError("geometry_weight must be in [0, 1]")
        object.__setattr__(self, "unmatched_cost", unmatched)
        object.__setattr__(self, "ambiguity_margin", margin)
        object.__setattr__(self, "geometry_weight", geometry_weight)

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_IDENTITY_CONFIGURATION = IdentityConfiguration()


PairStatus = Literal["matched", "unmatched_left", "unmatched_right"]


@dataclass(frozen=True)
class PairwiseAssignment:
    """One pairwise assignment or explicit unmatched observation."""

    left_view_id: str
    right_view_id: str
    left_candidate_id: str | None
    right_candidate_id: str | None
    status: PairStatus
    cosine_distance: float | None
    geometry_distance: float | None
    total_cost: float
    alternative_margin: float | None
    ambiguous: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseViewResult:
    """All assignments between two canonically ordered views."""

    left_view_id: str
    right_view_id: str
    assignments: tuple[PairwiseAssignment, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "left_view_id": self.left_view_id,
            "right_view_id": self.right_view_id,
            "assignments": [item.to_record() for item in self.assignments],
        }


@dataclass(frozen=True)
class IdentityMember:
    view_id: str
    candidate_id: str

    def to_record(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CycleCheck:
    """Agreement of all three pairwise links in one three-view cycle."""

    view_ids: tuple[str, str, str]
    consistent: bool

    def to_record(self) -> dict[str, Any]:
        return {"view_ids": list(self.view_ids), "consistent": self.consistent}


TrackStatus = Literal["matched", "ambiguous", "unmatched"]


@dataclass(frozen=True)
class IdentityTrack:
    """A deployment identity with at most one candidate from each view."""

    identity_id: str
    members: tuple[IdentityMember, ...]
    status: TrackStatus
    minimum_margin: float | None
    cycle_consistent: bool
    cycle_checks: tuple[CycleCheck, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "members": [item.to_record() for item in self.members],
            "status": self.status,
            "minimum_margin": self.minimum_margin,
            "cycle_consistent": self.cycle_consistent,
            "cycle_checks": [item.to_record() for item in self.cycle_checks],
        }


@dataclass(frozen=True)
class CrossViewIdentityResult:
    """Versioned result consumed by the Notebook 08F application contract."""

    schema_version: str
    configuration_sha256: str
    input_sha256: str
    view_ids: tuple[str, ...]
    anchor_view_id: str | None
    pairwise_results: tuple[PairwiseViewResult, ...]
    tracks: tuple[IdentityTrack, ...]

    def to_record(self) -> dict[str, Any]:
        statuses = {"matched": 0, "ambiguous": 0, "unmatched": 0}
        for track in self.tracks:
            statuses[track.status] += 1
        return {
            "schema_version": self.schema_version,
            "configuration_sha256": self.configuration_sha256,
            "input_sha256": self.input_sha256,
            "view_ids": list(self.view_ids),
            "anchor_view_id": self.anchor_view_id,
            "pairwise_results": [item.to_record() for item in self.pairwise_results],
            "tracks": [item.to_record() for item in self.tracks],
            "summary": {
                "view_count": len(self.view_ids),
                "track_count": len(self.tracks),
                "matched_track_count": statuses["matched"],
                "ambiguous_track_count": statuses["ambiguous"],
                "unmatched_track_count": statuses["unmatched"],
                "cycle_consistent": all(
                    track.cycle_consistent for track in self.tracks
                ),
            },
        }


def cosine_cost(left: Sequence[float], right: Sequence[float]) -> float:
    """Return DINO global-embedding cosine distance in [0, 2]."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise ValueError("embeddings must be one-dimensional")
    if left_array.shape != right_array.shape:
        raise ValueError("embeddings must have equal dimensions")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("embeddings must contain finite values")
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("embeddings must have non-zero norm")
    similarity = float(np.dot(left_array, right_array) / (left_norm * right_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def normalized_geometry_cost(
    left: NormalizedGeometry | None,
    right: NormalizedGeometry | None,
) -> float | None:
    """Return a bounded 2D geometry difference using upload-derived values only."""

    if left is None or right is None:
        return None

    def box_values(item: NormalizedGeometry) -> tuple[float, ...]:
        x1, y1, x2, y2 = item.box_xyxy_normalized
        width = x2 - x1
        height = y2 - y1
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            width,
            height,
            width * height,
        )

    left_box = box_values(left)
    right_box = box_values(right)
    differences = [
        float(np.hypot(left_box[0] - right_box[0], left_box[1] - right_box[1]))
        / np.sqrt(2.0),
        abs(left_box[2] - right_box[2]),
        abs(left_box[3] - right_box[3]),
        abs(left_box[4] - right_box[4]),
    ]
    if (
        left.mask_centroid_xy_normalized is not None
        and right.mask_centroid_xy_normalized is not None
    ):
        differences.append(
            float(
                np.hypot(
                    left.mask_centroid_xy_normalized[0]
                    - right.mask_centroid_xy_normalized[0],
                    left.mask_centroid_xy_normalized[1]
                    - right.mask_centroid_xy_normalized[1],
                )
            )
            / np.sqrt(2.0)
        )
    if left.mask_area_fraction is not None and right.mask_area_fraction is not None:
        differences.append(abs(left.mask_area_fraction - right.mask_area_fraction))
    return float(np.clip(np.mean(differences), 0.0, 1.0))


def _candidate_cost(
    left: CandidateObservation,
    right: CandidateObservation,
    configuration: IdentityConfiguration,
) -> tuple[float, float | None, float]:
    appearance = cosine_cost(left.embedding, right.embedding)
    geometry = normalized_geometry_cost(left.geometry, right.geometry)
    geometry_penalty = 0.0 if geometry is None else configuration.geometry_weight * geometry
    return appearance, geometry, float(appearance + geometry_penalty)


def match_view_pair(
    left: ViewObservation,
    right: ViewObservation,
    *,
    configuration: IdentityConfiguration = DEFAULT_IDENTITY_CONFIGURATION,
) -> PairwiseViewResult:
    """Match two views with Hungarian assignment and explicit unmatched outputs."""

    if left.view_id == right.view_id:
        raise ValueError("a view cannot be matched to itself")
    if right.view_id < left.view_id:
        left, right = right, left

    left_candidates = left.candidates
    right_candidates = right.candidates
    left_count = len(left_candidates)
    right_count = len(right_candidates)
    assignments: list[PairwiseAssignment] = []

    if left_count == 0:
        for candidate in right_candidates:
            assignments.append(
                PairwiseAssignment(
                    left_view_id=left.view_id,
                    right_view_id=right.view_id,
                    left_candidate_id=None,
                    right_candidate_id=candidate.candidate_id,
                    status="unmatched_right",
                    cosine_distance=None,
                    geometry_distance=None,
                    total_cost=configuration.unmatched_cost,
                    alternative_margin=None,
                    ambiguous=False,
                )
            )
        return PairwiseViewResult(left.view_id, right.view_id, tuple(assignments))

    if right_count == 0:
        for candidate in left_candidates:
            assignments.append(
                PairwiseAssignment(
                    left_view_id=left.view_id,
                    right_view_id=right.view_id,
                    left_candidate_id=candidate.candidate_id,
                    right_candidate_id=None,
                    status="unmatched_left",
                    cosine_distance=None,
                    geometry_distance=None,
                    total_cost=configuration.unmatched_cost,
                    alternative_margin=None,
                    ambiguous=False,
                )
            )
        return PairwiseViewResult(left.view_id, right.view_id, tuple(assignments))

    appearance_costs = np.empty((left_count, right_count), dtype=np.float64)
    geometry_costs = np.full((left_count, right_count), np.nan, dtype=np.float64)
    total_costs = np.empty((left_count, right_count), dtype=np.float64)
    for left_index, left_candidate in enumerate(left_candidates):
        for right_index, right_candidate in enumerate(right_candidates):
            appearance, geometry, total = _candidate_cost(
                left_candidate,
                right_candidate,
                configuration,
            )
            appearance_costs[left_index, right_index] = appearance
            geometry_costs[left_index, right_index] = (
                np.nan if geometry is None else geometry
            )
            total_costs[left_index, right_index] = total

    large_cost = max(10.0, configuration.unmatched_cost + 5.0)
    solver_costs = np.full(
        (left_count, right_count + left_count),
        large_cost,
        dtype=np.float64,
    )
    tie_scale = 1e-10
    for left_index in range(left_count):
        for right_index in range(right_count):
            rank_alignment = (left_index - right_index) ** 2
            solver_costs[left_index, right_index] = (
                total_costs[left_index, right_index]
                + tie_scale * (1.0 + rank_alignment)
            )
        solver_costs[left_index, right_count + left_index] = (
            configuration.unmatched_cost + tie_scale * left_index
        )

    selected_rows, selected_columns = linear_sum_assignment(solver_costs)
    selected_targets: set[int] = set()
    for left_index, column_index in zip(selected_rows, selected_columns, strict=True):
        candidate = left_candidates[int(left_index)]
        if int(column_index) < right_count:
            right_index = int(column_index)
            selected_targets.add(right_index)
            chosen_cost = float(total_costs[left_index, right_index])
            alternatives = [configuration.unmatched_cost]
            alternatives.extend(
                float(total_costs[left_index, index])
                for index in range(right_count)
                if index != right_index
            )
            margin = float(min(alternatives) - chosen_cost)
            geometry_value = float(geometry_costs[left_index, right_index])
            assignments.append(
                PairwiseAssignment(
                    left_view_id=left.view_id,
                    right_view_id=right.view_id,
                    left_candidate_id=candidate.candidate_id,
                    right_candidate_id=right_candidates[right_index].candidate_id,
                    status="matched",
                    cosine_distance=float(appearance_costs[left_index, right_index]),
                    geometry_distance=(
                        None if np.isnan(geometry_value) else geometry_value
                    ),
                    total_cost=chosen_cost,
                    alternative_margin=margin,
                    ambiguous=margin < configuration.ambiguity_margin,
                )
            )
        else:
            alternatives = [
                float(total_costs[left_index, index])
                for index in range(right_count)
            ]
            margin = float(min(alternatives) - configuration.unmatched_cost)
            assignments.append(
                PairwiseAssignment(
                    left_view_id=left.view_id,
                    right_view_id=right.view_id,
                    left_candidate_id=candidate.candidate_id,
                    right_candidate_id=None,
                    status="unmatched_left",
                    cosine_distance=None,
                    geometry_distance=None,
                    total_cost=configuration.unmatched_cost,
                    alternative_margin=margin,
                    ambiguous=margin < configuration.ambiguity_margin,
                )
            )

    for right_index, candidate in enumerate(right_candidates):
        if right_index in selected_targets:
            continue
        assignments.append(
            PairwiseAssignment(
                left_view_id=left.view_id,
                right_view_id=right.view_id,
                left_candidate_id=None,
                right_candidate_id=candidate.candidate_id,
                status="unmatched_right",
                cosine_distance=None,
                geometry_distance=None,
                total_cost=configuration.unmatched_cost,
                alternative_margin=None,
                ambiguous=False,
            )
        )
    assignments.sort(
        key=lambda item: (
            item.left_candidate_id is None,
            item.left_candidate_id or "",
            item.right_candidate_id is None,
            item.right_candidate_id or "",
        )
    )
    return PairwiseViewResult(left.view_id, right.view_id, tuple(assignments))


Node = tuple[str, str]


def _matched_edges(
    pairwise_results: Sequence[PairwiseViewResult],
) -> dict[frozenset[Node], PairwiseAssignment]:
    edges: dict[frozenset[Node], PairwiseAssignment] = {}
    for result in pairwise_results:
        for assignment in result.assignments:
            if assignment.status != "matched":
                continue
            assert assignment.left_candidate_id is not None
            assert assignment.right_candidate_id is not None
            edge = frozenset(
                {
                    (assignment.left_view_id, assignment.left_candidate_id),
                    (assignment.right_view_id, assignment.right_candidate_id),
                }
            )
            edges[edge] = assignment
    return edges


def _partner(
    node: Node,
    *,
    target_view_id: str,
    edges: Mapping[frozenset[Node], PairwiseAssignment],
) -> tuple[Node, PairwiseAssignment] | None:
    candidates = []
    for edge, assignment in edges.items():
        if node not in edge:
            continue
        other = next(item for item in edge if item != node)
        if other[0] == target_view_id:
            candidates.append((other, assignment))
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[1].ambiguous,
            item[1].total_cost,
            item[0][1],
        )
    )
    return candidates[0]


def _build_tracks(
    views: Sequence[ViewObservation],
    pairwise_results: Sequence[PairwiseViewResult],
) -> tuple[IdentityTrack, ...]:
    edges = _matched_edges(pairwise_results)
    candidates_by_view = {
        view.view_id: tuple((view.view_id, item.candidate_id) for item in view.candidates)
        for view in views
    }
    anchor = min(views, key=lambda view: (-len(view.candidates), view.view_id))
    tracks: list[list[Node]] = [
        [(anchor.view_id, candidate.candidate_id)] for candidate in anchor.candidates
    ]
    assigned: set[Node] = {node for track in tracks for node in track}

    for view in views:
        if view.view_id == anchor.view_id:
            continue
        for track in tracks:
            match = _partner(
                track[0],
                target_view_id=view.view_id,
                edges=edges,
            )
            if match is not None and match[0] not in assigned:
                track.append(match[0])
                assigned.add(match[0])

    all_nodes = sorted(node for nodes in candidates_by_view.values() for node in nodes)
    view_ids = [view.view_id for view in views]
    for seed in all_nodes:
        if seed in assigned:
            continue
        track = [seed]
        assigned.add(seed)
        for view_id in view_ids:
            if view_id == seed[0] or any(node[0] == view_id for node in track):
                continue
            options = []
            for member in track:
                match = _partner(member, target_view_id=view_id, edges=edges)
                if match is not None and match[0] not in assigned:
                    options.append(match)
            if options:
                options.sort(
                    key=lambda item: (
                        item[1].ambiguous,
                        item[1].total_cost,
                        item[0][1],
                    )
                )
                track.append(options[0][0])
                assigned.add(options[0][0])
        tracks.append(track)

    prepared = []
    for members in tracks:
        members = sorted(members)
        pair_assignments = []
        pair_consistent = True
        for left_node, right_node in combinations(members, 2):
            assignment = edges.get(frozenset({left_node, right_node}))
            if assignment is None:
                pair_consistent = False
            else:
                pair_assignments.append(assignment)

        cycle_checks = []
        for first, second, third in combinations(members, 3):
            consistent = all(
                frozenset(pair) in edges
                for pair in ((first, second), (second, third), (first, third))
            )
            cycle_checks.append(
                CycleCheck(
                    view_ids=(first[0], second[0], third[0]),
                    consistent=consistent,
                )
            )
        cycle_consistent = pair_consistent and all(
            item.consistent for item in cycle_checks
        )
        margins = [
            item.alternative_margin
            for item in pair_assignments
            if item.alternative_margin is not None
        ]
        minimum_margin = None if not margins else float(min(margins))
        if len(members) == 1:
            status: TrackStatus = "unmatched"
            cycle_consistent = True
        elif not cycle_consistent or any(item.ambiguous for item in pair_assignments):
            status = "ambiguous"
        else:
            status = "matched"
        prepared.append(
            (
                tuple(members),
                status,
                minimum_margin,
                cycle_consistent,
                tuple(cycle_checks),
            )
        )

    prepared.sort(key=lambda item: item[0])
    return tuple(
        IdentityTrack(
            identity_id=f"I{index:03d}",
            members=tuple(
                IdentityMember(view_id=view_id, candidate_id=candidate_id)
                for view_id, candidate_id in members
            ),
            status=status,
            minimum_margin=minimum_margin,
            cycle_consistent=cycle_consistent,
            cycle_checks=cycle_checks,
        )
        for index, (
            members,
            status,
            minimum_margin,
            cycle_consistent,
            cycle_checks,
        ) in enumerate(prepared)
    )


def match_cross_view_identities(
    views: Sequence[ViewObservation],
    *,
    configuration: IdentityConfiguration = DEFAULT_IDENTITY_CONFIGURATION,
) -> CrossViewIdentityResult:
    """Match candidates across two to five arbitrary uploaded views."""

    canonical = validate_views(views)
    pairwise_results = tuple(
        match_view_pair(left, right, configuration=configuration)
        for left, right in combinations(canonical, 2)
    )
    input_record = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "views": [view.to_record(include_embeddings=True) for view in canonical],
    }
    input_payload = json.dumps(
        input_record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    anchor = min(canonical, key=lambda view: (-len(view.candidates), view.view_id))
    return CrossViewIdentityResult(
        schema_version=RESULT_SCHEMA_VERSION,
        configuration_sha256=configuration.sha256,
        input_sha256=sha256(input_payload.encode("utf-8")).hexdigest(),
        view_ids=tuple(view.view_id for view in canonical),
        anchor_view_id=(None if not any(view.candidates for view in canonical) else anchor.view_id),
        pairwise_results=pairwise_results,
        tracks=_build_tracks(canonical, pairwise_results),
    )


__all__ = [
    "INPUT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "CandidateObservation",
    "CrossViewIdentityResult",
    "CycleCheck",
    "IdentityConfiguration",
    "IdentityMember",
    "IdentityTrack",
    "NormalizedGeometry",
    "PairwiseAssignment",
    "PairwiseViewResult",
    "ViewObservation",
    "cosine_cost",
    "match_cross_view_identities",
    "match_view_pair",
    "normalized_geometry_cost",
    "parse_model_facing_input",
    "validate_views",
]
