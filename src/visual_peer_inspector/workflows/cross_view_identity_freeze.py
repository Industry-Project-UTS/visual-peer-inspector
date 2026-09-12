"""Freeze label-free cross-view identity and application output contracts.

Notebook 08F consumes already locked Notebook 08E predictions. It computes and
locks both identity variants before evaluator correspondence is opened. Source
identities, asset clusters, and camera diagnostics are confined to the private
evaluation phase and never enter the web-facing matcher.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
)
from visual_peer_inspector.evaluation.proposal_reference import (
    REFERENCE_GEOMETRY_POLICY,
)
from visual_peer_inspector.evaluation.ranking import (
    clustered_mean_interval,
    paired_clustered_difference,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram
from visual_peer_inspector.frozen_resource import publish_frozen_resource
from visual_peer_inspector.models.sam2 import binary_mask_sha256, decode_binary_mask
from visual_peer_inspector.notebook_provenance import (
    bind_runtime_code_provenance_to_json,
)
from visual_peer_inspector.qualitative_reporting import (
    PredictedObject,
    QualitativeResult,
    QualitativeView,
    select_publication_results,
)
from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    prepare_masked_crop,
    project_crop_heatmap,
)
from visual_peer_inspector.representation.cross_view_identity import (
    INPUT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CandidateObservation,
    IdentityConfiguration,
    NormalizedGeometry,
    ViewObservation,
    match_cross_view_identities,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    save_development_audit_bundle,
    verify_prediction_lock,
)

STAGE = "cross_view_identity_freeze"
WORKFLOW_SCHEMA_VERSION = "visual-peer-08f-workflow/v1"
APPLICATION_CONTRACT_VERSION = "visual-peer-application-output/v1"
IDENTITY_LOCK_VERSION = "visual-peer-08f-prediction-lock/v1"
EVALUATOR_SCHEMA_VERSION = "visual-peer-identity-evaluator/v1"
VARIANTS = ("embedding_only", "normalized_geometry")
VIEW_COUNTS = (2, 3, 5)
DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 32_040
DEFAULT_QUALITATIVE_COUNT = 15


def _json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name} row {line_number} is not an object")
            rows.append(value)
    return rows


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    temporary.replace(path)


def _require_exact_fields(
    row: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = sorted(required.difference(row))
    unknown = sorted(set(row).difference(required | optional))
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {unknown}")


def _finite_optional(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite or null")
    return number


@dataclass(frozen=True)
class ObjectPrediction:
    """One Notebook 08E object without evaluator identity."""

    object_id: str
    box_xyxy: tuple[int, int, int, int]
    mask_rle: Mapping[str, Any]
    mask_sha256: str
    p1_score: float | None
    p1_rank: int | None
    p1_heatmap: Any
    embedding: tuple[float, ...]
    geometry: NormalizedGeometry


@dataclass(frozen=True)
class ViewPrediction:
    """One uploaded view and its automatic object predictions."""

    cohort_id: str
    opaque_scene_id: str
    view_id: str
    image_sha256: str
    status: str
    terminal_failure: str | None
    objects: tuple[ObjectPrediction, ...]


@dataclass(frozen=True)
class VerifiedPredictionSet:
    """Hash-verified Notebook 08E predictions grouped by scene."""

    views_by_scene: Mapping[str, tuple[ViewPrediction, ...]]
    input_lock_sha256s: tuple[str, ...]
    frozen_identity_sha256: str
    cohort_ids: tuple[str, ...]
    embedding_dimension: int


def _normalised_geometry(item: Mapping[str, Any]) -> NormalizedGeometry:
    mask = decode_binary_mask(item["mask_rle"])
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Notebook 08E object mask must be a non-empty 2D array")
    if binary_mask_sha256(mask) != str(item["mask_sha256"]):
        raise ValueError("Notebook 08E object mask differs from its digest")
    height, width = mask.shape
    box = tuple(int(value) for value in item["box_xyxy"])
    if len(box) != 4:
        raise ValueError("Notebook 08E box must have four coordinates")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Notebook 08E box lies outside its mask coordinate space")
    rows, columns = np.nonzero(mask)
    return NormalizedGeometry(
        box_xyxy_normalized=(x1 / width, y1 / height, x2 / width, y2 / height),
        mask_centroid_xy_normalized=(
            (float(columns.mean()) + 0.5) / width,
            (float(rows.mean()) + 0.5) / height,
        ),
        mask_area_fraction=float(mask.mean()),
    )


def _verify_vector_digest(vector: np.ndarray, expected: str) -> None:
    contiguous = np.ascontiguousarray(vector.astype(np.float16, copy=False))
    observed = sha256(contiguous.tobytes(order="C")).hexdigest()
    if observed != expected:
        raise ValueError("A global embedding differs from its index digest")


def _load_one_prediction_root(root: Path) -> tuple[list[ViewPrediction], str, str, int]:
    root = root.expanduser().resolve()
    manifest_path = root / "prediction_manifest.jsonl"
    embeddings_path = root / "global_embeddings.npy"
    index_path = root / "embedding_index.jsonl"
    lock_path = root / "prediction_lock.json"
    for path in (manifest_path, embeddings_path, index_path, lock_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Notebook 08E output is missing {path.name}: {root}"
            )

    lock = verify_prediction_lock(lock_path)
    locked_paths = {
        str(item["path"]): str(item["sha256"]) for item in lock["artifacts"]
    }
    for path in (manifest_path, embeddings_path, index_path):
        relative = path.relative_to(root).as_posix()
        if locked_paths.get(relative) != file_sha256(path):
            raise ValueError(f"Notebook 08E lock does not cover {relative}")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    if (
        embeddings.dtype != np.float16
        or embeddings.ndim != 2
        or embeddings.shape[1] < 2
    ):
        raise ValueError("Notebook 08E global embeddings must be a float16 matrix")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Notebook 08E global embeddings contain non-finite values")

    manifest_rows = _read_jsonl(manifest_path)
    manifest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest_rows:
        _require_exact_fields(
            row,
            required={
                "schema_version",
                "cohort_id",
                "opaque_scene_id",
                "view_id",
                "image_sha256",
                "status",
                "terminal_failure",
                "object_count",
                "objects",
                "per_view_p1_candidate",
                "per_view_p1_score",
            },
            optional=set(),
            context="Notebook 08E prediction row",
        )
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in manifest_by_key:
            raise ValueError("Notebook 08E manifest contains a duplicate scene-view")
        if int(row["object_count"]) != len(row["objects"]):
            raise ValueError("Notebook 08E object count differs from the object list")
        manifest_by_key[key] = row

    index_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    used_rows: set[int] = set()
    for row in _read_jsonl(index_path):
        _require_exact_fields(
            row,
            required={
                "schema_version",
                "cohort_id",
                "opaque_scene_id",
                "view_id",
                "object_id",
                "row_index",
                "feature_dimension",
                "vector_sha256",
            },
            optional=set(),
            context="Notebook 08E embedding index row",
        )
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["object_id"]),
        )
        row_index = int(row["row_index"])
        if key in index_by_key or row_index in used_rows:
            raise ValueError("Notebook 08E embedding index contains a duplicate")
        if not 0 <= row_index < len(embeddings):
            raise ValueError("Notebook 08E embedding row index is outside the matrix")
        if int(row["feature_dimension"]) != embeddings.shape[1]:
            raise ValueError("Notebook 08E embedding dimension differs from the matrix")
        _verify_vector_digest(embeddings[row_index], str(row["vector_sha256"]))
        index_by_key[key] = row
        used_rows.add(row_index)

    predictions: list[ViewPrediction] = []
    indexed_object_keys: set[tuple[str, str, str]] = set()
    for (scene_id, view_id), row in sorted(manifest_by_key.items()):
        objects = []
        object_ids: set[str] = set()
        for item in row["objects"]:
            object_id = str(item["object_id"])
            if object_id in object_ids:
                raise ValueError("Notebook 08E view contains duplicate object IDs")
            object_ids.add(object_id)
            key = (scene_id, view_id, object_id)
            index = index_by_key.get(key)
            if index is None:
                raise ValueError("Notebook 08E object has no global embedding")
            indexed_object_keys.add(key)
            vector = embeddings[int(index["row_index"])].astype(np.float32)
            rank_value = item.get("p1_rank")
            objects.append(
                ObjectPrediction(
                    object_id=object_id,
                    box_xyxy=tuple(int(value) for value in item["box_xyxy"]),
                    mask_rle=dict(item["mask_rle"]),
                    mask_sha256=str(item["mask_sha256"]),
                    p1_score=_finite_optional(item.get("p1_score"), name="P1 score"),
                    p1_rank=None if rank_value is None else int(rank_value),
                    p1_heatmap=item.get("p1_heatmap"),
                    embedding=tuple(float(value) for value in vector),
                    geometry=_normalised_geometry(item),
                )
            )
        predictions.append(
            ViewPrediction(
                cohort_id=str(row["cohort_id"]),
                opaque_scene_id=scene_id,
                view_id=view_id,
                image_sha256=str(row["image_sha256"]),
                status=str(row["status"]),
                terminal_failure=(
                    None
                    if row["terminal_failure"] is None
                    else str(row["terminal_failure"])
                ),
                objects=tuple(sorted(objects, key=lambda value: value.object_id)),
            )
        )
    if indexed_object_keys != set(index_by_key):
        raise ValueError("Notebook 08E embedding index contains unreferenced rows")
    if len(used_rows) != len(embeddings):
        raise ValueError("Notebook 08E embedding matrix contains unindexed rows")
    cohort_ids = {value.cohort_id for value in predictions}
    if len(cohort_ids) != 1:
        raise ValueError("One Notebook 08E root must contain one cohort")
    return (
        predictions,
        file_sha256(lock_path),
        str(lock["identity_sha256"]),
        int(embeddings.shape[1]),
    )


def load_08e_predictions(prediction_roots: Sequence[Path]) -> VerifiedPredictionSet:
    """Load one or more locked Notebook 08E cohort roots."""

    roots = tuple(
        sorted({Path(value).expanduser().resolve() for value in prediction_roots})
    )
    if not roots:
        raise ValueError("At least one Notebook 08E prediction root is required")
    by_scene: dict[str, list[ViewPrediction]] = defaultdict(list)
    lock_hashes: list[str] = []
    identities: set[str] = set()
    dimensions: set[int] = set()
    cohorts: set[str] = set()
    seen_views: set[tuple[str, str]] = set()
    for root in roots:
        views, lock_hash, identity_sha, dimension = _load_one_prediction_root(root)
        lock_hashes.append(lock_hash)
        identities.add(identity_sha)
        dimensions.add(dimension)
        cohorts.update(item.cohort_id for item in views)
        for view in views:
            key = (view.opaque_scene_id, view.view_id)
            if key in seen_views:
                raise ValueError("Notebook 08E roots overlap on a scene-view")
            seen_views.add(key)
            by_scene[view.opaque_scene_id].append(view)
    if len(identities) != 1 or len(dimensions) != 1:
        raise ValueError("Notebook 08E cohorts do not share one frozen model identity")
    canonical = {
        scene_id: tuple(sorted(views, key=lambda value: value.view_id))
        for scene_id, views in sorted(by_scene.items())
    }
    if any(len(views) < 5 for views in canonical.values()):
        raise ValueError(
            "Notebook 08F requires at least five available views per scene"
        )
    return VerifiedPredictionSet(
        views_by_scene=canonical,
        input_lock_sha256s=tuple(sorted(lock_hashes)),
        frozen_identity_sha256=next(iter(identities)),
        cohort_ids=tuple(sorted(cohorts)),
        embedding_dimension=next(iter(dimensions)),
    )


def _selected_views(
    scene_id: str, views: Sequence[ViewPrediction], count: int
) -> tuple[ViewPrediction, ...]:
    if count not in VIEW_COUNTS:
        raise ValueError(f"Unsupported view count: {count}")
    ordered = sorted(
        views,
        key=lambda value: (
            sha256(
                f"08f-view-subset-v1|{scene_id}|{value.view_id}".encode()
            ).hexdigest(),
            value.view_id,
        ),
    )
    return tuple(sorted(ordered[:count], key=lambda value: value.view_id))


def _to_observation(view: ViewPrediction, *, include_geometry: bool) -> ViewObservation:
    return ViewObservation(
        view_id=view.view_id,
        candidates=tuple(
            CandidateObservation(
                candidate_id=item.object_id,
                embedding=item.embedding,
                geometry=item.geometry if include_geometry else None,
            )
            for item in view.objects
        ),
    )


def _case_input_sha256(
    observations: Sequence[ViewObservation],
    *,
    configuration: IdentityConfiguration,
    variant: str,
    case_kind: str,
) -> str:
    return _json_sha256(
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "variant": variant,
            "case_kind": case_kind,
            "configuration_sha256": configuration.sha256,
            "views": [item.to_record(include_embeddings=True) for item in observations],
        }
    )


def _checkpoint_paths(root: Path, variant: str, case_id: str) -> tuple[Path, Path]:
    stage_root = root / "prediction-only" / "checkpoints" / variant
    return (
        stage_root / f"{case_id}.prediction.json",
        stage_root / f"{case_id}.receipt.json",
    )


def _run_checkpointed_case(
    *,
    scene_id: str,
    views: Sequence[ViewPrediction],
    variant: str,
    configuration: IdentityConfiguration,
    case_kind: str,
    output_root: Path,
) -> tuple[dict[str, Any], bool]:
    include_geometry = variant == "normalized_geometry"
    observations = tuple(
        _to_observation(value, include_geometry=include_geometry) for value in views
    )
    input_sha = _case_input_sha256(
        observations,
        configuration=configuration,
        variant=variant,
        case_kind=case_kind,
    )
    case_id = f"{scene_id}-{case_kind}"
    prediction_path, receipt_path = _checkpoint_paths(output_root, variant, case_id)
    if prediction_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") == "complete"
            and receipt.get("input_sha256") == input_sha
            and receipt.get("configuration_sha256") == configuration.sha256
            and receipt.get("output_sha256") == file_sha256(prediction_path)
        ):
            return json.loads(prediction_path.read_text(encoding="utf-8")), True

    result = match_cross_view_identities(observations, configuration=configuration)
    record = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "opaque_scene_id": scene_id,
        "variant": variant,
        "case_kind": case_kind,
        "view_count": len(views),
        "selected_view_ids": [value.view_id for value in views],
        "identity_result": result.to_record(),
    }
    _write_json(prediction_path, record)
    _write_json(
        receipt_path,
        {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "status": "complete",
            "opaque_scene_id": scene_id,
            "variant": variant,
            "case_kind": case_kind,
            "input_sha256": input_sha,
            "configuration_sha256": configuration.sha256,
            "output_sha256": file_sha256(prediction_path),
        },
    )
    return record, False


def _matched_links(result: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    links = set()
    for pair in result["pairwise_results"]:
        for assignment in pair["assignments"]:
            if assignment["status"] == "matched":
                links.add(
                    (
                        str(assignment["left_view_id"]),
                        str(assignment["left_candidate_id"]),
                        str(assignment["right_view_id"]),
                        str(assignment["right_candidate_id"]),
                    )
                )
    return links


def _link_stability(
    full: Mapping[str, Any], missing: Mapping[str, Any], missing_view: str
) -> float:
    full_links = {
        value
        for value in _matched_links(full)
        if value[0] != missing_view and value[2] != missing_view
    }
    missing_links = _matched_links(missing)
    union = full_links | missing_links
    return 1.0 if not union else len(full_links & missing_links) / len(union)


def _prediction_lock(output_root: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    prediction_root = output_root / "prediction-only"
    lock_path = prediction_root / "identity_prediction_lock.json"
    artifacts = []
    for path in sorted(prediction_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Identity predictions cannot contain symbolic links")
        if path.is_file() and path != lock_path:
            artifacts.append(
                {
                    "path": path.relative_to(prediction_root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not artifacts:
        raise ValueError("Identity prediction lock requires completed artifacts")
    record = {
        "schema_version": IDENTITY_LOCK_VERSION,
        "status": "locked_before_evaluator_access",
        **dict(metadata),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _json_sha256(artifacts),
    }
    _write_json(lock_path, record)
    return record


def verify_identity_prediction_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.expanduser().resolve()
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != IDENTITY_LOCK_VERSION
        or record.get("status") != "locked_before_evaluator_access"
    ):
        raise ValueError("Notebook 08F prediction lock is invalid")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != int(
        record.get("artifact_count", -1)
    ):
        raise ValueError("Notebook 08F prediction inventory count differs")
    if record.get("artifact_inventory_sha256") != _json_sha256(artifacts):
        raise ValueError("Notebook 08F prediction inventory hash differs")
    root = lock_path.parent
    for item in artifacts:
        path = (root / str(item["path"])).resolve()
        if root not in path.parents or file_sha256(path) != str(item["sha256"]):
            raise ValueError("Notebook 08F prediction artifact differs from its lock")
    return record


def _load_evaluator_rows(
    evaluator_path: Path,
    *,
    prediction_lock_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    prediction_lock = verify_identity_prediction_lock(prediction_lock_path)
    evaluator_path = evaluator_path.expanduser().resolve(strict=True)
    manifest = json.loads((evaluator_path.parent / "evaluator_manifest.json").read_text())
    if (
        manifest.get("schema_version") != "visual-peer-primary-resource-release/v1"
        or manifest.get("status") != "locked"
        or manifest.get("correspondence_sha256") != file_sha256(evaluator_path)
        or manifest.get("scene_count") != prediction_lock["scene_count"]
        or manifest.get("reference_geometry_policy") != REFERENCE_GEOMETRY_POLICY
    ):
        raise ValueError("Evaluator resource manifest does not match the locked correspondence")
    rows = _read_jsonl(evaluator_path)
    if manifest.get("correspondence_row_count") != len(rows):
        raise ValueError("Evaluator correspondence row count differs from its manifest")
    allowed_optional = {"camera_diagnostic"}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        _require_exact_fields(
            row,
            required={
                "schema_version",
                "opaque_scene_id",
                "opaque_asset_group_id",
                "view_id",
                "object_id",
                "reference_identity_id",
            },
            optional=allowed_optional,
            context="evaluator correspondence row",
        )
        if row["schema_version"] != EVALUATOR_SCHEMA_VERSION:
            raise ValueError("Evaluator correspondence schema version differs")
        key = (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["object_id"]))
        if key in seen:
            raise ValueError("Evaluator correspondence contains a duplicate object")
        seen.add(key)
    scene_ids = sorted({key[0] for key in seen})
    if len(scene_ids) != manifest["scene_count"] or membership_sha256(scene_ids) != manifest.get("membership_sha256"):
        raise ValueError("Evaluator correspondence membership differs from its manifest")
    return rows, file_sha256(evaluator_path)


def _scene_metrics(
    prediction: Mapping[str, Any],
    evaluator_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    asset_by_scene: Mapping[str, str],
) -> dict[str, Any]:
    scene_id = str(prediction["opaque_scene_id"])
    identity = prediction["identity_result"]
    members = [member for track in identity["tracks"] for member in track["members"]]
    references: dict[tuple[str, str], str | None] = {}
    for member in members:
        key = (scene_id, str(member["view_id"]), str(member["candidate_id"]))
        if key not in evaluator_lookup:
            raise ValueError(
                "Evaluator correspondence does not cover a predicted object"
            )
        value = evaluator_lookup[key]["reference_identity_id"]
        references[(key[1], key[2])] = None if value is None else str(value)

    mapped_total = sum(value is not None for value in references.values())
    linked_mapped = 0
    unmatched_observations = 0
    ambiguous_observations = 0
    evaluated_links = 0
    correct_links = 0
    cycles = []
    for track in identity["tracks"]:
        track_members = track["members"]
        if len(track_members) < 2:
            unmatched_observations += len(track_members)
        if track["status"] == "ambiguous":
            ambiguous_observations += len(track_members)
        mapped = []
        for member in track_members:
            reference = references[
                (str(member["view_id"]), str(member["candidate_id"]))
            ]
            if reference is not None:
                mapped.append(reference)
                if len(track_members) >= 2:
                    linked_mapped += 1
        for left_index in range(len(mapped)):
            for right_index in range(left_index + 1, len(mapped)):
                evaluated_links += 1
                correct_links += int(mapped[left_index] == mapped[right_index])
        cycles.extend(bool(item["consistent"]) for item in track["cycle_checks"])

    observation_total = len(members)
    switch_rate = (
        None if evaluated_links == 0 else 1.0 - correct_links / evaluated_links
    )
    return {
        "opaque_scene_id": scene_id,
        "opaque_asset_group_id": asset_by_scene[scene_id],
        "variant": str(prediction["variant"]),
        "case_kind": str(prediction["case_kind"]),
        "view_count": int(prediction["view_count"]),
        "observation_coverage": 0.0
        if mapped_total == 0
        else linked_mapped / mapped_total,
        "unmatched_rate": 0.0
        if observation_total == 0
        else unmatched_observations / observation_total,
        "ambiguity_rate": 0.0
        if observation_total == 0
        else ambiguous_observations / observation_total,
        "pairwise_identity_accuracy": (
            None if evaluated_links == 0 else correct_links / evaluated_links
        ),
        "switch_rate": switch_rate,
        "cycle_consistency_rate": None if not cycles else sum(cycles) / len(cycles),
        "mapped_observation_count": mapped_total,
        "predicted_observation_count": observation_total,
        "evaluated_link_count": evaluated_links,
        "cycle_check_count": len(cycles),
    }


METRICS = (
    "observation_coverage",
    "unmatched_rate",
    "ambiguity_rate",
    "pairwise_identity_accuracy",
    "switch_rate",
    "cycle_consistency_rate",
)


def _aggregate_metrics(scene_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(str(row["variant"]), int(row["view_count"]))].append(row)
    for (variant, view_count), rows in sorted(grouped.items()):
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            output.append(
                {
                    "variant": variant,
                    "view_count": view_count,
                    "metric": metric,
                    "mean": None if not values else float(np.mean(values)),
                    "scene_count": len(rows),
                    "eligible_scene_count": len(values),
                    "status": "not_applicable" if not values else "reported",
                }
            )
    return output


def _clustered_intervals(
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        grouped[(str(row["variant"]), int(row["view_count"]))].append(row)
    for (variant, view_count), rows in sorted(grouped.items()):
        for metric_index, metric in enumerate(METRICS):
            eligible = [row for row in rows if row[metric] is not None]
            clusters = {str(row["opaque_asset_group_id"]) for row in eligible}
            if not eligible or len(clusters) < 2:
                output.append(
                    {
                        "variant": variant,
                        "view_count": view_count,
                        "metric": metric,
                        "mean": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "scene_count": len(eligible),
                        "cluster_count": len(clusters),
                        "repetitions": 0,
                        "seed": seed,
                        "status": "not_applicable"
                        if not eligible
                        else "insufficient_clusters",
                    }
                )
                continue
            interval = clustered_mean_interval(
                eligible,
                metric=metric,
                repetitions=repetitions,
                seed=seed + view_count * 100 + metric_index,
            )
            output.append(
                {
                    "variant": variant,
                    "view_count": view_count,
                    "metric": metric,
                    "mean": interval["mean"],
                    "ci_lower": interval["ci_lower"],
                    "ci_upper": interval["ci_upper"],
                    "scene_count": len(eligible),
                    "cluster_count": interval["cluster_count"],
                    "repetitions": interval["repetitions"],
                    "seed": interval["seed"],
                    "status": "reported",
                }
            )
    return output


def _paired_intervals(
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for view_count in VIEW_COUNTS:
        baseline = [
            row
            for row in scene_rows
            if row["variant"] == "embedding_only" and row["view_count"] == view_count
        ]
        candidate = [
            row
            for row in scene_rows
            if row["variant"] == "normalized_geometry"
            and row["view_count"] == view_count
        ]
        for metric_index, metric in enumerate(METRICS):
            baseline_eligible = [row for row in baseline if row[metric] is not None]
            candidate_eligible = [row for row in candidate if row[metric] is not None]
            baseline_keys = {str(row["opaque_scene_id"]) for row in baseline_eligible}
            candidate_keys = {str(row["opaque_scene_id"]) for row in candidate_eligible}
            paired = baseline_keys & candidate_keys
            clusters = {
                str(row["opaque_asset_group_id"])
                for row in baseline_eligible
                if str(row["opaque_scene_id"]) in paired
            }
            if not paired or len(clusters) < 2:
                output.append(
                    {
                        "view_count": view_count,
                        "metric": metric,
                        "mean_difference_geometry_minus_embedding": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "paired_scene_count": len(paired),
                        "cluster_count": len(clusters),
                        "repetitions": 0,
                        "seed": seed,
                        "status": "not_applicable"
                        if not paired
                        else "insufficient_clusters",
                    }
                )
                continue
            interval = paired_clustered_difference(
                baseline_eligible,
                candidate_eligible,
                metric=metric,
                repetitions=repetitions,
                seed=seed + view_count * 100 + metric_index,
            )
            output.append(
                {
                    "view_count": view_count,
                    "metric": metric,
                    "mean_difference_geometry_minus_embedding": interval[
                        "mean_difference"
                    ],
                    "ci_lower": interval["ci_lower"],
                    "ci_upper": interval["ci_upper"],
                    "paired_scene_count": interval["paired_scene_count"],
                    "cluster_count": interval["cluster_count"],
                    "repetitions": interval["repetitions"],
                    "seed": interval["seed"],
                    "status": "reported",
                }
            )
    return output


def _freeze_decision(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_configuration: IdentityConfiguration,
    geometry_configuration: IdentityConfiguration,
    stability_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    five_view = {
        str(row["metric"]): row
        for row in paired_rows
        if int(row["view_count"]) == 5 and row["status"] == "reported"
    }
    higher_is_better = {
        "observation_coverage",
        "pairwise_identity_accuracy",
        "cycle_consistency_rate",
    }
    lower_is_better = {"unmatched_rate", "ambiguity_rate", "switch_rate"}
    favorable = []
    harmful = []
    for metric, row in five_view.items():
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        if metric in higher_is_better:
            if lower > 0.0:
                favorable.append(metric)
            elif upper < 0.0:
                harmful.append(metric)
        elif metric in lower_is_better:
            if upper < 0.0:
                favorable.append(metric)
            elif lower > 0.0:
                harmful.append(metric)
    stability_passed = bool(stability_rows) and all(
        bool(row["order_invariant"])
        and bool(row["duplicate_view_rejected"])
        and bool(row["missing_view_completed"])
        for row in stability_rows
    )
    geometry_is_justified = bool(favorable) and not harmful and stability_passed
    chosen = "normalized_geometry" if geometry_is_justified else "embedding_only"
    chosen_config = (
        geometry_configuration if geometry_is_justified else baseline_configuration
    )
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "status": "frozen",
        "chosen_variant": chosen,
        "chosen_configuration": asdict(chosen_config),
        "chosen_configuration_sha256": chosen_config.sha256,
        "decision_rule": (
            "Use normalized geometry only when the five-view asset-clustered paired "
            "interval shows at least one favorable change, shows no adverse change, "
            "and all structural stability checks pass. Otherwise retain the simpler "
            "embedding-only method."
        ),
        "favorable_metrics": sorted(favorable),
        "harmful_metrics": sorted(harmful),
        "structural_stability_passed": stability_passed,
        "absolute_performance_threshold_used": False,
        "camera_metadata_used_by_chosen_method": False,
        "human_correction_position": "optional_review_after_automatic_result",
    }


def _fused_odd_one_out(
    prediction: Mapping[str, Any],
    view_lookup: Mapping[str, ViewPrediction],
) -> tuple[str | None, float | None, float | None, str, dict[str, str]]:
    scores: list[tuple[float, str, int, int]] = []
    candidate_to_track: dict[str, str] = {}
    for track in prediction["identity_result"]["tracks"]:
        track_scores = []
        top_votes = 0
        for member in track["members"]:
            object_id = str(member["candidate_id"])
            candidate_to_track[f"{member['view_id']}|{object_id}"] = str(
                track["identity_id"]
            )
            item = next(
                value
                for value in view_lookup[str(member["view_id"])].objects
                if value.object_id == object_id
            )
            if item.p1_score is not None:
                track_scores.append(item.p1_score)
                top_votes += int(item.p1_rank == 1)
        if track_scores:
            scores.append(
                (
                    float(np.mean(track_scores)),
                    str(track["identity_id"]),
                    top_votes,
                    len(track_scores),
                )
            )
    scores.sort(key=lambda value: (-value[0], value[1]))
    if not scores:
        return None, None, None, "insufficient_evidence", candidate_to_track
    top = scores[0]
    agreement = top[2] / top[3]
    margin = None if len(scores) < 2 else max(0.0, top[0] - scores[1][0])
    track = next(
        value
        for value in prediction["identity_result"]["tracks"]
        if value["identity_id"] == top[1]
    )
    tied = len(scores) > 1 and math.isclose(
        top[0], scores[1][0], rel_tol=0.0, abs_tol=1e-12
    )
    state = (
        "review_recommended"
        if track["status"] != "matched" or not track["cycle_consistent"] or tied
        else "result_ready"
    )
    return top[1], agreement, margin, state, candidate_to_track


def _find_rgb(media_roots: Sequence[Path], view: ViewPrediction) -> Path:
    relative_options = (
        Path("media") / view.opaque_scene_id / "rgb" / f"{view.view_id}.png",
        Path(view.opaque_scene_id) / "rgb" / f"{view.view_id}.png",
    )
    for root in media_roots:
        root = root.expanduser().resolve()
        for relative in relative_options:
            path = (root / relative).resolve()
            if (
                root in path.parents
                and path.is_file()
                and file_sha256(path) == view.image_sha256
            ):
                return path
    raise FileNotFoundError(
        f"RGB for {view.opaque_scene_id}/{view.view_id} was not found by digest"
    )


def _combined_heatmap(
    view: ViewPrediction, rgb: Image.Image, object_ids: set[str]
) -> np.ndarray | None:
    """Project the selected track's patch evidence through each object crop."""
    arrays = []
    for item in view.objects:
        if item.p1_heatmap is None or item.object_id not in object_ids:
            continue
        array = np.asarray(item.p1_heatmap, dtype=np.float32)
        if array.ndim != 2 or not array.size:
            raise ValueError("Notebook 08E P1 heatmap must be a non-empty 2D array")
        mask = decode_binary_mask(item.mask_rle)
        crop = prepare_masked_crop(rgb, mask, configuration=CropConfiguration())
        arrays.append(project_crop_heatmap(array, crop, mask))
    if not arrays:
        return None
    combined = np.fmax.reduce(arrays)
    return combined if np.isfinite(combined).any() else None


def _qualitative_result(
    prediction: Mapping[str, Any],
    scene_views: Sequence[ViewPrediction],
    *,
    media_roots: Sequence[Path],
    method_sha256: str,
) -> QualitativeResult:
    view_lookup = {value.view_id: value for value in scene_views}
    winner, agreement, margin, state, candidate_to_track = _fused_odd_one_out(
        prediction,
        view_lookup,
    )
    views = []
    for view_id in prediction["selected_view_ids"]:
        view = view_lookup[str(view_id)]
        objects = []
        for item in view.objects:
            track_id = candidate_to_track.get(f"{view.view_id}|{item.object_id}")
            if track_id is None:
                continue
            objects.append(
                PredictedObject(
                    object_id=track_id,
                    box_xyxy=item.geometry.box_xyxy_normalized,
                    sam_mask=decode_binary_mask(item.mask_rle),
                    p1_score=item.p1_score,
                    p1_rank=item.p1_rank,
                )
            )
        rgb_path = _find_rgb(media_roots, view)
        selected_ids = {
            item.object_id for item in view.objects
            if winner is not None
            and candidate_to_track.get(f"{view.view_id}|{item.object_id}") == winner
        }
        with Image.open(rgb_path) as opened:
            heatmap = _combined_heatmap(view, opened.convert("RGB"), selected_ids)
        views.append(
            QualitativeView(
                view_id=view.view_id,
                rgb=rgb_path,
                objects=tuple(objects),
                heatmap=heatmap,
                coordinate_space="normalized",
            )
        )
    return QualitativeResult(
        result_id=str(prediction["opaque_scene_id"]),
        views=tuple(views),
        fused_odd_one_out=winner,
        agreement=agreement,
        margin=margin,
        review_state=state,
        publication_metadata={
            "identity_method": str(prediction["variant"]),
            "identity_configuration_sha256": method_sha256,
            "identity_result_schema": RESULT_SCHEMA_VERSION,
        },
    )


def _application_contract(
    *,
    freeze: Mapping[str, Any],
    identity_lock_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": APPLICATION_CONTRACT_VERSION,
        "status": "frozen",
        "supported_view_count": {"minimum": 2, "maximum": 5},
        "identity_method": freeze["chosen_variant"],
        "identity_configuration": freeze["chosen_configuration"],
        "identity_configuration_sha256": freeze["chosen_configuration_sha256"],
        "identity_result_schema": RESULT_SCHEMA_VERSION,
        "identity_prediction_lock_sha256": identity_lock_sha256,
        "frozen_model_identity_sha256": freeze[
            "frozen_model_identity_sha256"
        ],
        "required_application_outputs": {
            "view_predictions": "boxes, masks, P1 scores, and stable identity tokens",
            "fused_prediction": "ranked odd-one-out identity and relative score",
            "evidence": "per-view highlights, masks, and coarse P1 heatmaps",
            "review_state": "result_ready, review_recommended, or insufficient_evidence",
        },
        "model_facing_inputs": [
            "uploaded RGB pixels",
            "automatic boxes",
            "automatic masks",
            "DINO global embeddings",
            "normalized upload geometry",
            "P1 scores",
        ],
        "forbidden_model_inputs": [
            "source identifiers",
            "evaluator correspondence",
            "labels",
            "roles",
            "categories",
            "file paths",
            "object order",
            "camera metadata",
        ],
        "camera_metadata_scope": "evaluator_diagnostic_only",
        "human_correction_scope": "optional_review_after_automatic_result",
    }


def _application_prediction_record(
    prediction: Mapping[str, Any],
    scene_views: Sequence[ViewPrediction],
    *,
    method_sha256: str,
) -> dict[str, Any]:
    """Return one concrete result using deployment fields only."""

    view_lookup = {value.view_id: value for value in scene_views}
    winner, agreement, margin, state, candidate_to_track = _fused_odd_one_out(
        prediction,
        view_lookup,
    )
    views = []
    for view_id in prediction["selected_view_ids"]:
        view = view_lookup[str(view_id)]
        objects = []
        for item in view.objects:
            objects.append(
                {
                    "object_id": item.object_id,
                    "identity_id": candidate_to_track.get(
                        f"{view.view_id}|{item.object_id}"
                    ),
                    "box_xyxy_normalized": list(item.geometry.box_xyxy_normalized),
                    "mask_rle": dict(item.mask_rle),
                    "mask_sha256": item.mask_sha256,
                    "p1_score": item.p1_score,
                    "p1_rank": item.p1_rank,
                }
            )
        views.append(
            {
                "view_id": view.view_id,
                "status": view.status,
                "terminal_failure": view.terminal_failure,
                "objects": objects,
            }
        )
    record = {
        "schema_version": APPLICATION_CONTRACT_VERSION,
        "result_id": str(prediction["opaque_scene_id"]),
        "identity_method": str(prediction["variant"]),
        "identity_configuration_sha256": method_sha256,
        "view_count": int(prediction["view_count"]),
        "views": views,
        "identity_tracks": prediction["identity_result"]["tracks"],
        "fused_odd_one_out": winner,
        "agreement": agreement,
        "margin": margin,
        "review_state": state,
    }
    record["result_sha256"] = _json_sha256(record)
    return record


def _safe_manifest(root: Path) -> dict[str, Any]:
    run_record_path = root / "run_record.json"
    if run_record_path.is_file():
        bind_runtime_code_provenance_to_json(run_record_path)
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Safe output cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    record = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _json_sha256(artifacts),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def run_cross_view_identity_freeze(
    *,
    prediction_roots: Sequence[Path],
    evaluator_correspondence_path: Path,
    media_roots: Sequence[Path],
    output_root: Path,
    unmatched_cost: float = 0.35,
    ambiguity_margin: float = 0.05,
    geometry_weight: float = 0.15,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    qualitative_count: int = DEFAULT_QUALITATIVE_COUNT,
) -> dict[str, Any]:
    """Run the two-method development comparison and freeze one web contract."""

    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if not 12 <= qualitative_count <= 18:
        raise ValueError("qualitative_count must be between 12 and 18")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "private" / "failure_receipts.jsonl"
    failures: list[dict[str, Any]] = []

    verified = load_08e_predictions(prediction_roots)
    print(json.dumps({"event": "identity_inputs_verified", "scene_count": len(verified.views_by_scene)}), flush=True)
    configurations = {
        "embedding_only": IdentityConfiguration(
            unmatched_cost=unmatched_cost,
            ambiguity_margin=ambiguity_margin,
            geometry_weight=0.0,
        ),
        "normalized_geometry": IdentityConfiguration(
            unmatched_cost=unmatched_cost,
            ambiguity_margin=ambiguity_margin,
            geometry_weight=geometry_weight,
        ),
    }
    prediction_rows = []
    stability_rows = []
    resumed_count = 0
    processed_count = 0

    for scene_index, (scene_id, scene_views) in enumerate(verified.views_by_scene.items(), start=1):
        for variant in VARIANTS:
            configuration = configurations[variant]
            five_view_record = None
            five_views = _selected_views(scene_id, scene_views, 5)
            for count in VIEW_COUNTS:
                selected = _selected_views(scene_id, scene_views, count)
                try:
                    record, resumed = _run_checkpointed_case(
                        scene_id=scene_id,
                        views=selected,
                        variant=variant,
                        configuration=configuration,
                        case_kind=f"views-{count}",
                        output_root=output_root,
                    )
                    resumed_count += int(resumed)
                    processed_count += int(not resumed)
                    prediction_rows.append(record)
                    if count == 5:
                        five_view_record = record
                except Exception as error:  # noqa: BLE001 - preserve each failed case as evidence
                    failures.append(
                        {
                            "schema_version": WORKFLOW_SCHEMA_VERSION,
                            "stage": STAGE,
                            "opaque_scene_id": scene_id,
                            "variant": variant,
                            "case_kind": f"views-{count}",
                            "error_type": type(error).__name__,
                            "status": "failed",
                        }
                    )
            if five_view_record is None:
                continue
            observations = tuple(
                _to_observation(view, include_geometry=variant == "normalized_geometry")
                for view in five_views
            )
            order_result = match_cross_view_identities(
                tuple(reversed(observations)), configuration=configuration
            ).to_record()
            order_invariant = order_result == five_view_record["identity_result"]
            missing_index = int(
                sha256(f"08f-missing-v1|{scene_id}".encode()).hexdigest(), 16
            ) % len(five_views)
            missing_view_id = five_views[missing_index].view_id
            missing_views = tuple(
                ViewPrediction(
                    cohort_id=view.cohort_id,
                    opaque_scene_id=view.opaque_scene_id,
                    view_id=view.view_id,
                    image_sha256=view.image_sha256,
                    status=view.status,
                    terminal_failure=view.terminal_failure,
                    objects=() if index == missing_index else view.objects,
                )
                for index, view in enumerate(five_views)
            )
            missing_record, resumed = _run_checkpointed_case(
                scene_id=scene_id,
                views=missing_views,
                variant=variant,
                configuration=configuration,
                case_kind="views-5-missing",
                output_root=output_root,
            )
            resumed_count += int(resumed)
            processed_count += int(not resumed)
            duplicate_rejected = False
            try:
                match_cross_view_identities(
                    (observations[0], observations[1], observations[0]),
                    configuration=configuration,
                )
            except ValueError:
                duplicate_rejected = True
            stability_rows.append(
                {
                    "opaque_scene_id": scene_id,
                    "variant": variant,
                    "order_invariant": order_invariant,
                    "duplicate_view_rejected": duplicate_rejected,
                    "missing_view_completed": True,
                    "missing_view_id": missing_view_id,
                    "retained_link_stability": _link_stability(
                        five_view_record["identity_result"],
                        missing_record["identity_result"],
                        missing_view_id,
                    ),
                }
            )
        if scene_index % 25 == 0 or scene_index == len(verified.views_by_scene):
            print(json.dumps({
                "event": "identity_predictions_checkpointed",
                "completed_scenes": scene_index,
                "total_scenes": len(verified.views_by_scene),
                "processed_cases": processed_count,
                "resumed_cases": resumed_count,
                "failed_cases": len(failures),
            }), flush=True)

    _write_jsonl(
        output_root / "prediction-only" / "identity_predictions.jsonl", prediction_rows
    )
    _write_jsonl(
        output_root / "prediction-only" / "stability_checks.jsonl", stability_rows
    )
    _write_json(
        output_root / "prediction-only" / "method_configurations.json",
        {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "variants": {name: asdict(value) for name, value in configurations.items()},
            "configuration_sha256s": {
                name: value.sha256 for name, value in configurations.items()
            },
            "camera_metadata_used": False,
        },
    )
    _prediction_lock(
        output_root,
        {
            "input_08e_lock_sha256s": list(verified.input_lock_sha256s),
            "frozen_model_identity_sha256": verified.frozen_identity_sha256,
            "scene_count": len(verified.views_by_scene),
            "prediction_count": len(prediction_rows),
            "failure_count": len(failures),
        },
    )
    identity_lock_path = (
        output_root / "prediction-only" / "identity_prediction_lock.json"
    )
    evaluator_rows, evaluator_sha = _load_evaluator_rows(
        evaluator_correspondence_path,
        prediction_lock_path=identity_lock_path,
    )
    print("Identity predictions locked; evaluating correspondence", flush=True)

    evaluator_lookup = {
        (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["object_id"])): row
        for row in evaluator_rows
    }
    asset_by_scene: dict[str, str] = {}
    for row in evaluator_rows:
        scene_id = str(row["opaque_scene_id"])
        asset = str(row["opaque_asset_group_id"])
        if scene_id in asset_by_scene and asset_by_scene[scene_id] != asset:
            raise ValueError(
                "One scene cannot belong to multiple evaluator asset clusters"
            )
        asset_by_scene[scene_id] = asset
    if set(verified.views_by_scene).difference(asset_by_scene):
        raise ValueError(
            "Evaluator correspondence does not cover every prediction scene"
        )

    scene_rows = [
        _scene_metrics(row, evaluator_lookup, asset_by_scene) for row in prediction_rows
    ]
    aggregate_rows = _aggregate_metrics(scene_rows)
    interval_rows = _clustered_intervals(
        scene_rows, repetitions=bootstrap_repetitions, seed=bootstrap_seed
    )
    print("Clustered intervals complete; calculating paired comparisons", flush=True)
    paired_rows = _paired_intervals(
        scene_rows, repetitions=bootstrap_repetitions, seed=bootstrap_seed
    )
    freeze = _freeze_decision(
        paired_rows,
        baseline_configuration=configurations["embedding_only"],
        geometry_configuration=configurations["normalized_geometry"],
        stability_rows=stability_rows,
    )
    freeze["frozen_model_identity_sha256"] = verified.frozen_identity_sha256

    safe_root = output_root / "safe"
    private_root = output_root / "private"
    _write_csv(
        safe_root / "method_summary.csv",
        (
            "variant",
            "view_count",
            "metric",
            "mean",
            "scene_count",
            "eligible_scene_count",
            "status",
        ),
        aggregate_rows,
    )
    _write_csv(
        safe_root / "clustered_intervals.csv",
        (
            "variant",
            "view_count",
            "metric",
            "mean",
            "ci_lower",
            "ci_upper",
            "scene_count",
            "cluster_count",
            "repetitions",
            "seed",
            "status",
        ),
        interval_rows,
    )
    _write_csv(
        safe_root / "paired_method_intervals.csv",
        (
            "view_count",
            "metric",
            "mean_difference_geometry_minus_embedding",
            "ci_lower",
            "ci_upper",
            "paired_scene_count",
            "cluster_count",
            "repetitions",
            "seed",
            "status",
        ),
        paired_rows,
    )
    _write_csv(
        safe_root / "stability_summary.csv",
        (
            "variant",
            "scene_count",
            "order_invariance_rate",
            "duplicate_rejection_rate",
            "missing_view_completion_rate",
            "mean_retained_link_stability",
        ),
        [
            {
                "variant": variant,
                "scene_count": len(rows),
                "order_invariance_rate": float(
                    np.mean([row["order_invariant"] for row in rows])
                ),
                "duplicate_rejection_rate": float(
                    np.mean([row["duplicate_view_rejected"] for row in rows])
                ),
                "missing_view_completion_rate": float(
                    np.mean([row["missing_view_completed"] for row in rows])
                ),
                "mean_retained_link_stability": float(
                    np.mean([row["retained_link_stability"] for row in rows])
                ),
            }
            for variant in VARIANTS
            for rows in [[row for row in stability_rows if row["variant"] == variant]]
        ],
    )
    _write_csv(
        private_root / "evaluator_scene_metrics.csv",
        tuple(scene_rows[0]) if scene_rows else ("opaque_scene_id",),
        scene_rows,
    )
    _write_json(safe_root / "method_freeze.json", freeze)
    _write_json(
        private_root / "evaluator_access_receipt.json",
        {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "status": "opened_after_identity_prediction_lock",
            "identity_prediction_lock_sha256": file_sha256(identity_lock_path),
            "evaluator_correspondence_sha256": evaluator_sha,
            "camera_diagnostic_row_count": sum(
                "camera_diagnostic" in row for row in evaluator_rows
            ),
            "camera_metadata_used_by_model": False,
        },
    )

    # These figures are computed from the reported rows and contain no evaluator IDs.
    five_view_lookup = {
        (row["variant"], row["metric"]): row["mean"]
        for row in aggregate_rows
        if row["view_count"] == 5 and row["mean"] is not None
    }
    figure_root = safe_root / "figures"
    review_figure_root = output_root / "review-only" / "figures"
    if all(
        (variant, "pairwise_identity_accuracy") in five_view_lookup
        for variant in VARIANTS
    ):
        save_bar_chart(
            ["Embedding only", "Embedding + geometry"],
            [
                five_view_lookup[(variant, "pairwise_identity_accuracy")]
                for variant in VARIANTS
            ],
            figure_root / "five_view_identity_accuracy.png",
            title="Five-view identity link accuracy",
            y_label="Mean rate",
            show_values=True,
        )
    stability_values = [row["retained_link_stability"] for row in stability_rows]
    if stability_values:
        save_histogram(
            stability_values,
            figure_root / "missing_view_link_stability.png",
            title="Identity links retained after one missing view",
            x_label="Retained-link Jaccard score",
            bins=12,
        )

    chosen_predictions = [
        row
        for row in prediction_rows
        if row["variant"] == freeze["chosen_variant"] and row["view_count"] == 5
    ]
    qualitative_values = []
    for row in chosen_predictions:
        try:
            qualitative_values.append(
                _qualitative_result(
                    row,
                    verified.views_by_scene[str(row["opaque_scene_id"])],
                    media_roots=media_roots,
                    method_sha256=str(freeze["chosen_configuration_sha256"]),
                )
            )
        except Exception as error:  # noqa: BLE001 - preserve failed qualitative cases as evidence
            failures.append(
                {
                    "schema_version": WORKFLOW_SCHEMA_VERSION,
                    "stage": "qualitative_render_input",
                    "opaque_scene_id": str(row["opaque_scene_id"]),
                    "variant": str(row["variant"]),
                    "case_kind": "views-5",
                    "error_type": type(error).__name__,
                    "status": "failed",
                }
            )
    selected_results = (
        select_publication_results(
            qualitative_values,
            count=min(qualitative_count, len(qualitative_values)),
            namespace="toysad8k-08f-deployment-panels-v1",
        )
        if qualitative_values
        else ()
    )
    qualitative_rows = []
    for result in selected_results:
        bundle = save_development_audit_bundle(
            result, review_figure_root / "odd-one-out"
        )
        qualitative_rows.append(
            {
                "result_id": result.result_id,
                "fused_odd_one_out": result.fused_odd_one_out,
                "agreement": result.agreement,
                "margin": result.margin,
                "review_state": result.review_state,
                **bundle,
            }
        )
    _write_csv(
        output_root / "review-only" / "qualitative_manifest.csv",
        (
            "result_id",
            "fused_odd_one_out",
            "agreement",
            "margin",
            "review_state",
            "prediction_panel",
            "prediction_sha256",
            "sam_evidence_panel",
            "sam_evidence_sha256",
        ),
        qualitative_rows,
    )

    _write_jsonl(failure_path, failures)
    failure_counts = Counter(
        (str(row["error_type"]), str(row["case_kind"]), str(row["variant"]))
        for row in failures
    )
    _write_csv(
        safe_root / "failure_summary.csv",
        ("error_type", "case_kind", "variant", "count"),
        [
            {
                "error_type": error_type,
                "case_kind": case_kind,
                "variant": variant,
                "count": count,
            }
            for (error_type, case_kind, variant), count in sorted(
                failure_counts.items()
            )
        ],
    )
    application_contract = _application_contract(
        freeze=freeze,
        identity_lock_sha256=file_sha256(identity_lock_path),
    )
    _write_json(safe_root / "application_output_contract.json", application_contract)
    application_predictions = [
        _application_prediction_record(
            row,
            verified.views_by_scene[str(row["opaque_scene_id"])],
            method_sha256=str(freeze["chosen_configuration_sha256"]),
        )
        for row in prediction_rows
        if row["variant"] == freeze["chosen_variant"]
    ]
    _write_jsonl(
        output_root / "downstream" / "application_predictions.jsonl",
        application_predictions,
    )
    method_resource_root = output_root / "publish" / "method-freeze"
    gates = [
        {
            "gate": "08e_prediction_locks_verified",
            "passed": bool(verified.input_lock_sha256s),
        },
        {"gate": "identity_predictions_locked_before_evaluator", "passed": True},
        {
            "gate": "two_three_five_view_cases_complete",
            "passed": len(prediction_rows)
            == len(verified.views_by_scene) * len(VARIANTS) * len(VIEW_COUNTS),
        },
        {
            "gate": "input_order_invariance",
            "passed": bool(stability_rows)
            and all(row["order_invariant"] for row in stability_rows),
        },
        {
            "gate": "duplicate_view_rejection",
            "passed": bool(stability_rows)
            and all(row["duplicate_view_rejected"] for row in stability_rows),
        },
        {
            "gate": "missing_view_execution",
            "passed": bool(stability_rows)
            and all(row["missing_view_completed"] for row in stability_rows),
        },
        {"gate": "no_model_or_render_failures", "passed": not failures},
        {
            "gate": "method_and_application_contract_frozen",
            "passed": freeze["status"] == "frozen",
        },
        {
            "gate": "versioned_application_results_written",
            "passed": len(application_predictions)
            == len(verified.views_by_scene) * len(VIEW_COUNTS),
        },
        {
            "gate": "deployment_valid_odd_one_out_evidence",
            "passed": len(qualitative_rows) == qualitative_count,
        },
        {
            "gate": "camera_metadata_excluded_from_model",
            "passed": not freeze["camera_metadata_used_by_chosen_method"],
        },
        {
            "gate": "no_absolute_accuracy_gate",
            "passed": not freeze["absolute_performance_threshold_used"],
        },
    ]
    _write_csv(safe_root / "acceptance_gates.csv", ("gate", "passed"), gates)
    all_gates_passed = all(row["passed"] for row in gates)
    if method_resource_root.exists():
        shutil.rmtree(method_resource_root)
    method_resource_manifest_sha256 = None
    if all_gates_passed:
        publish_frozen_resource(
            resource_root=method_resource_root,
            stage="cross_view_identity_method_release",
            files={
                "application_output_contract.json": safe_root
                / "application_output_contract.json",
                "method_freeze.json": safe_root / "method_freeze.json",
            },
            bindings={
                "method_freeze_sha256": file_sha256(safe_root / "method_freeze.json"),
                "application_output_contract_sha256": file_sha256(
                    safe_root / "application_output_contract.json"
                ),
                "identity_prediction_lock_sha256": file_sha256(identity_lock_path),
            },
        )
        method_resource_manifest_sha256 = file_sha256(
            method_resource_root / "artifact_manifest.json"
        )
    run_record = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "stage": STAGE,
        "status": "complete"
        if all_gates_passed
        else "completed_with_failed_gates",
        "input_08e_lock_sha256s": list(verified.input_lock_sha256s),
        "frozen_model_identity_sha256": verified.frozen_identity_sha256,
        "identity_prediction_lock_sha256": file_sha256(identity_lock_path),
        "evaluator_correspondence_sha256": evaluator_sha,
        "cohort_ids": list(verified.cohort_ids),
        "scene_count": len(verified.views_by_scene),
        "embedding_dimension": verified.embedding_dimension,
        "prediction_count": len(prediction_rows),
        "processed_this_run": processed_count,
        "resumed_from_verified_receipts": resumed_count,
        "failure_count": len(failures),
        "qualitative_panel_count": len(qualitative_rows),
        "chosen_variant": freeze["chosen_variant"],
        "all_acceptance_gates_passed": all_gates_passed,
        "absolute_performance_threshold_used": False,
        "safe_artifact_manifest_path": "artifact_manifest.json",
        "method_resource_manifest_sha256": method_resource_manifest_sha256,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _safe_manifest(safe_root)
    return run_record


__all__ = [
    "APPLICATION_CONTRACT_VERSION",
    "DEFAULT_BOOTSTRAP_REPETITIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_QUALITATIVE_COUNT",
    "EVALUATOR_SCHEMA_VERSION",
    "IDENTITY_LOCK_VERSION",
    "STAGE",
    "VARIANTS",
    "VIEW_COUNTS",
    "WORKFLOW_SCHEMA_VERSION",
    "VerifiedPredictionSet",
    "load_08e_predictions",
    "run_cross_view_identity_freeze",
    "verify_identity_prediction_lock",
]
