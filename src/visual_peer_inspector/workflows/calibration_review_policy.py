"""Calibrate and freeze the three-state review policy after protected inference.

The module verifies the immutable Notebook 09A prediction bundle before it opens
calibration-only correspondence. Neural, proposal, segmentation, representation,
identity, view, and fusion settings are carried forward without modification.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from visual_peer_inspector.data.cohort_contract import CALIBRATION_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.evaluation.proposals import match_boxes
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.qualitative_reporting import (
    EvaluatorObject,
    EvaluatorOverlay,
    EvaluatorView,
    PredictedObject,
    QualitativeResult,
    QualitativeView,
    render_evaluator_overlay_panel,
    save_panel_png,
)
from visual_peer_inspector.review_policy import (
    PREDICTION_SIGNAL_FIELDS,
    ReviewPolicy,
    apply_review_policy,
    build_prediction_review_signals,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    verify_prediction_lock,
)

SCHEMA_VERSION = "visual-peer-09b-review-policy/v1"
EVALUATOR_SCHEMA = "visual-peer-calibration-correspondence/v1"
EXPECTED_VIEW_COUNTS = (2, 3, 5)
POLICY_STATES = ("result_ready", "review_required", "insufficient_evidence")
ALWAYS_REVIEW_REASONS = (
    "zero_proposal",
    "incomplete_proposals",
    "duplicate_or_fragmented_proposals",
    "empty_mask",
    "identity_unmatched_or_ambiguous",
    "stage_failure",
)


def _digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                raise TypeError(f"JSONL row {line_number} is not an object: {path.name}")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic CSV table, including a header for an empty table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot infer CSV columns from an empty table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"CSV rows have inconsistent columns: {path.name}")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class VerifiedCalibrationBoundary:
    """Proof that the protected predictions were locked before evaluator access."""

    lock_path: Path
    lock_sha256: str
    lock_record: Mapping[str, Any]
    run_record_path: Path
    run_record_sha256: str
    method_record_path: Path
    method_record_sha256: str
    chosen_configuration_sha256: str
    prediction_manifest_path: Path
    multiview_manifest_path: Path
    embedding_path: Path
    embedding_index_path: Path


def verify_calibration_boundary(
    *,
    prediction_lock_path: Path,
    run_record_path: Path,
    method_record_path: Path,
) -> VerifiedCalibrationBoundary:
    """Verify every Notebook 09A prediction artifact without opening labels."""

    lock_path = Path(prediction_lock_path).expanduser().resolve(strict=True)
    lock = verify_prediction_lock(lock_path)
    if lock.get("cohort_id") != CALIBRATION_CONTRACT.cohort_id:
        raise ValueError("Prediction lock does not identify the calibration cohort")
    if lock.get("cohort_membership_sha256") != CALIBRATION_CONTRACT.membership_sha256:
        raise ValueError("Calibration prediction membership differs from the fixed split")
    if int(lock.get("scene_count", -1)) != CALIBRATION_CONTRACT.scene_count:
        raise ValueError("Calibration prediction lock does not contain 600 scenes")
    if int(lock.get("image_count", -1)) != CALIBRATION_CONTRACT.image_count:
        raise ValueError("Calibration prediction lock does not contain 12,000 images")

    root = lock_path.parent
    required = {
        "prediction_manifest.jsonl",
        "global_embeddings.npy",
        "embedding_index.jsonl",
        "multiview_prediction_manifest.jsonl",
    }
    locked_names = {str(item["path"]) for item in lock["artifacts"]}
    if not required.issubset(locked_names):
        raise ValueError("Prediction lock is missing a required 09A artifact")
    manifest_path = root / "prediction_manifest.jsonl"
    multiview_path = root / "multiview_prediction_manifest.jsonl"
    embedding_path = root / "global_embeddings.npy"
    embedding_index_path = root / "embedding_index.jsonl"

    prediction_rows = _read_jsonl(manifest_path)
    if len(prediction_rows) != CALIBRATION_CONTRACT.image_count:
        raise ValueError("Prediction manifest does not contain 12,000 scene-view rows")
    scene_view_keys = {
        (str(row.get("opaque_scene_id", "")), str(row.get("view_id", "")))
        for row in prediction_rows
    }
    if len(scene_view_keys) != CALIBRATION_CONTRACT.image_count or any(not all(key) for key in scene_view_keys):
        raise ValueError("Prediction manifest scene-view keys are incomplete or duplicated")

    multiview_rows = _read_jsonl(multiview_path)
    expected_multiview = CALIBRATION_CONTRACT.scene_count * len(EXPECTED_VIEW_COUNTS)
    if len(multiview_rows) != expected_multiview:
        raise ValueError("Multi-view manifest does not cover every calibration view contract")
    per_scene: dict[str, set[int]] = {}
    for row in multiview_rows:
        scene_id = str(row.get("opaque_scene_id", ""))
        per_scene.setdefault(scene_id, set()).add(int(row.get("view_count", -1)))
        if row.get("prediction_only") is not True:
            raise ValueError("09A multi-view rows must be prediction-only")
    if len(per_scene) != CALIBRATION_CONTRACT.scene_count or any(
        counts != set(EXPECTED_VIEW_COUNTS) for counts in per_scene.values()
    ):
        raise ValueError("Multi-view rows do not provide exact 2, 3 and 5 view results")

    embeddings = np.load(embedding_path, allow_pickle=False, mmap_mode="r")
    if embeddings.dtype != np.float16 or embeddings.ndim != 2:
        raise ValueError("Global embeddings must be one float16 matrix")
    embedding_rows = _read_jsonl(embedding_index_path)
    if len(embedding_rows) != len(embeddings):
        raise ValueError("Embedding index and matrix row counts differ")
    if [int(row.get("row_index", -1)) for row in embedding_rows] != list(range(len(embedding_rows))):
        raise ValueError("Embedding index row numbers are not contiguous")

    run_path = Path(run_record_path).expanduser().resolve(strict=True)
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run_record.get("status") != "completed"
        or run_record.get("cohort_id") != CALIBRATION_CONTRACT.cohort_id
        or run_record.get("prediction_only") is not True
        or run_record.get("evaluator_opened") is not False
        or run_record.get("accuracy_metrics_reported") is not False
        or run_record.get("policy_record_sha256") is not None
    ):
        raise ValueError("09A run record does not preserve the prediction-only boundary")

    method_path = Path(method_record_path).expanduser().resolve(strict=True)
    method_sha256 = file_sha256(method_path)
    if run_record.get("method_record_sha256") != method_sha256:
        raise ValueError("09A and the supplied method freeze have different hashes")
    method = json.loads(method_path.read_text(encoding="utf-8"))
    if method.get("status") != "frozen" or method.get("schema_version") != "visual-peer-08f-workflow/v1":
        raise ValueError("Method record is not the frozen Notebook 08F workflow")
    chosen = method.get("chosen_configuration")
    if not isinstance(chosen, Mapping):
        raise TypeError("Method record has no chosen configuration")
    chosen_hash = _digest(chosen)
    if method.get("chosen_configuration_sha256") != chosen_hash:
        raise ValueError("Chosen method configuration differs from its recorded hash")
    if str(chosen.get("score_method", chosen.get("scoring_method", ""))).upper() != "P1":
        raise ValueError("Calibration policy requires the frozen P1 method")
    if str(chosen.get("fusion_method", "")) != "mean_normalized_rank":
        raise ValueError("Calibration policy requires the frozen fusion method")

    return VerifiedCalibrationBoundary(
        lock_path=lock_path,
        lock_sha256=file_sha256(lock_path),
        lock_record=lock,
        run_record_path=run_path,
        run_record_sha256=file_sha256(run_path),
        method_record_path=method_path,
        method_record_sha256=method_sha256,
        chosen_configuration_sha256=chosen_hash,
        prediction_manifest_path=manifest_path,
        multiview_manifest_path=multiview_path,
        embedding_path=embedding_path,
        embedding_index_path=embedding_index_path,
    )


def open_calibration_evaluator(
    boundary: VerifiedCalibrationBoundary,
    evaluator_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open calibration correspondence only after a verified prediction boundary."""

    if not isinstance(boundary, VerifiedCalibrationBoundary):
        raise TypeError("A verified calibration prediction boundary is required")
    path = Path(evaluator_path).expanduser().resolve(strict=True)
    if path.is_dir() or path.name == "evaluator_manifest.json":
        root = path if path.is_dir() else path.parent
        return _open_raw_calibration_evaluator(boundary, root)
    rows = _read_jsonl(path)
    expected_count = CALIBRATION_CONTRACT.scene_count * len(EXPECTED_VIEW_COUNTS)
    if len(rows) != expected_count:
        raise ValueError("Calibration evaluator must cover 600 scenes at 2, 3 and 5 views")
    keys: set[tuple[str, int]] = set()
    groups: set[str] = set()
    for row in rows:
        if row.get("schema_version") != EVALUATOR_SCHEMA:
            raise ValueError("Calibration evaluator correspondence schema is invalid")
        scene_id = str(row.get("opaque_scene_id", ""))
        view_count = int(row.get("view_count", -1))
        group_id = str(row.get("opaque_asset_group_id", ""))
        acceptable = row.get("acceptable_anomalous_identity_ids")
        if not scene_id or view_count not in EXPECTED_VIEW_COUNTS or not group_id:
            raise ValueError("Calibration evaluator contains an invalid key")
        if not isinstance(acceptable, list) or not acceptable or not all(str(value) for value in acceptable):
            raise ValueError("Calibration evaluator requires acceptable anomalous identities")
        if int(row.get("reference_object_count", 0)) < 3:
            raise ValueError("Calibration evaluator reference object count is invalid")
        key = (scene_id, view_count)
        if key in keys:
            raise ValueError("Calibration evaluator contains duplicate scene-view contracts")
        keys.add(key)
        groups.add(group_id)
    if len(groups) != CALIBRATION_CONTRACT.asset_group_count:
        raise ValueError("Calibration evaluator asset-group count differs from the fixed split")
    receipt = {
        "schema_version": 1,
        "status": "opened_after_prediction_lock",
        "prediction_lock_sha256": boundary.lock_sha256,
        "evaluator_sha256": file_sha256(path),
        "evaluator_row_count": len(rows),
        "asset_group_count": len(groups),
    }
    return rows, receipt


def _open_raw_calibration_evaluator(
    boundary: VerifiedCalibrationBoundary,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map raw protected references to locked calibration identity tracks."""

    manifest_path = root / "evaluator_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "visual-peer-official-evaluator-manifest/v1"
        or manifest.get("cohort_id") != CALIBRATION_CONTRACT.cohort_id
        or manifest.get("source_membership_sha256")
        != CALIBRATION_CONTRACT.membership_sha256
        or int(manifest.get("scene_count", -1)) != CALIBRATION_CONTRACT.scene_count
        or int(manifest.get("image_count", -1)) != CALIBRATION_CONTRACT.image_count
    ):
        raise ValueError("Raw calibration evaluator manifest differs from the fixed cohort")
    expected_files = {
        "crosswalk_private.jsonl",
        "labels_private.jsonl",
        "geometry_private.jsonl",
    }
    artifacts = manifest.get("files")
    if not isinstance(artifacts, list) or {
        str(item.get("path")) for item in artifacts
    } != expected_files:
        raise ValueError("Raw calibration evaluator file allowlist differs")
    for item in artifacts:
        artifact = (root / str(item["path"])).resolve()
        if root not in artifact.parents or artifact.is_symlink() or not artifact.is_file():
            raise ValueError("Raw calibration evaluator contains an unsafe path")
        if file_sha256(artifact) != str(item.get("sha256")):
            raise ValueError("Raw calibration evaluator artifact differs from its hash")

    raw_crosswalk = _read_jsonl(root / "crosswalk_private.jsonl")
    raw_labels = _read_jsonl(root / "labels_private.jsonl")
    raw_geometry = _read_jsonl(root / "geometry_private.jsonl")
    crosswalk_by_scene = {str(row["opaque_scene_id"]): row for row in raw_crosswalk}
    labels_by_scene = {str(row["opaque_scene_id"]): row for row in raw_labels}
    geometry_by_scene = {str(row["opaque_scene_id"]): row for row in raw_geometry}
    scene_ids = set(crosswalk_by_scene)
    if (
        len(raw_crosswalk) != len(crosswalk_by_scene)
        or len(raw_labels) != len(labels_by_scene)
        or len(raw_geometry) != len(geometry_by_scene)
        or len(scene_ids) != CALIBRATION_CONTRACT.scene_count
        or set(labels_by_scene) != scene_ids
        or set(geometry_by_scene) != scene_ids
    ):
        raise ValueError("Raw calibration evaluator files have different coverage")
    groups = {
        str(row.get("base_asset_id", "")) for row in raw_crosswalk
    }
    if "" in groups or len(groups) != CALIBRATION_CONTRACT.asset_group_count:
        raise ValueError("Raw calibration evaluator asset groups differ from the fixed split")
    for scene_id in scene_ids:
        records = (
            crosswalk_by_scene[scene_id],
            labels_by_scene[scene_id],
            geometry_by_scene[scene_id],
        )
        if any(record.get("cohort_id") != CALIBRATION_CONTRACT.cohort_id for record in records):
            raise ValueError("Raw calibration evaluator row identifies another cohort")

    prediction_rows, multiview_rows = _prediction_rows(boundary)
    predictions = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in prediction_rows
    }
    candidate_reference: dict[tuple[str, str, str], str] = {}
    for scene_id in sorted(scene_ids):
        bridge = crosswalk_by_scene[scene_id]
        object_ids = [
            key
            for key, _index in sorted(
                bridge["object_id_map"].items(), key=lambda item: int(item[1])
            )
        ]
        for view in geometry_by_scene[scene_id]["views"]:
            view_id = str(view["view_id"])
            boxes = list(view["boxes_xyxy"])
            if len(boxes) != len(object_ids):
                raise ValueError("Raw calibration geometry and labels do not align")
            predicted = list(predictions[(scene_id, view_id)].get("objects", []))
            for match in match_boxes(
                boxes,
                [item["box_xyxy"] for item in predicted],
                minimum_iou=0.50,
            ):
                candidate_reference[
                    (scene_id, view_id, str(predicted[match.proposal_index]["object_id"]))
                ] = object_ids[match.reference_index]

    rows = []
    for decision in multiview_rows:
        scene_id = str(decision["opaque_scene_id"])
        raw_label = labels_by_scene[scene_id]
        anomalous = {
            str(item["object_id"])
            for item in raw_label["object_labels"]
            if bool(item["is_anomaly"])
        }
        acceptable = []
        for track in decision.get("tracks", []):
            references = {
                candidate_reference.get(
                    (scene_id, str(member["view_id"]), str(member["candidate_id"]))
                )
                for member in track.get("members", [])
            }
            if references.intersection(anomalous):
                acceptable.append(str(track["identity_id"]))
        if not acceptable:
            acceptable = [f"UNMATCHED:{value}" for value in sorted(anomalous)]
        reference_count = len(raw_label["object_labels"])
        rows.append(
            {
                "schema_version": EVALUATOR_SCHEMA,
                "opaque_scene_id": scene_id,
                "view_count": int(decision["view_count"]),
                "opaque_asset_group_id": str(
                    crosswalk_by_scene[scene_id]["base_asset_id"]
                ),
                "acceptable_anomalous_identity_ids": sorted(set(acceptable)),
                "reference_object_count": reference_count,
                "object_count_band": (
                    "3" if reference_count == 3 else "4" if reference_count == 4 else "5_plus"
                ),
                "category": str(crosswalk_by_scene[scene_id].get("category", "not_recorded")),
                "anomaly_type": str(raw_label.get("anomaly_type", "not_recorded")),
            }
        )
    rows.sort(key=lambda row: (str(row["opaque_scene_id"]), int(row["view_count"])))
    expected_count = CALIBRATION_CONTRACT.scene_count * len(EXPECTED_VIEW_COUNTS)
    if len(rows) != expected_count:
        raise ValueError("Raw calibration conversion has incomplete view contracts")
    receipt = {
        "schema_version": 1,
        "status": "opened_after_prediction_lock",
        "prediction_lock_sha256": boundary.lock_sha256,
        "evaluator_sha256": file_sha256(manifest_path),
        "evaluator_row_count": len(rows),
        "asset_group_count": len(
            {str(row["opaque_asset_group_id"]) for row in rows}
        ),
        "raw_reference_mapping_created_after_lock": True,
    }
    return rows, receipt


def _prediction_rows(boundary: VerifiedCalibrationBoundary) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _read_jsonl(boundary.prediction_manifest_path), _read_jsonl(boundary.multiview_manifest_path)


def build_calibration_signal_record(
    decision: Mapping[str, Any],
    view_rows: Sequence[Mapping[str, Any] | None],
    evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    """Join evaluation targets without allowing them to alter policy inputs."""

    scene_id = str(decision["opaque_scene_id"])
    view_count = int(decision["view_count"])
    reference_count = int(evaluator["reference_object_count"])
    signals = build_prediction_review_signals(decision, view_rows)
    acceptable = {
        str(value) for value in evaluator["acceptable_anomalous_identity_ids"]
    }
    fused = decision.get("fused_odd_one_out")
    result_available = fused is not None and not bool(
        decision.get("insufficient_evidence")
    )
    correct = None if fused is None else str(fused) in acceptable
    object_count_band = str(
        evaluator.get(
            "object_count_band",
            "3" if reference_count == 3 else "4" if reference_count == 4 else "5_plus",
        )
    )
    return {
        "opaque_scene_id": scene_id,
        "view_count": view_count,
        "opaque_asset_group_id": str(evaluator["opaque_asset_group_id"]),
        "fused_odd_one_out": fused,
        "result_available": result_available,
        "is_correct": correct,
        **signals,
        "category": str(evaluator.get("category", "not_recorded")),
        "anomaly_type": str(evaluator.get("anomaly_type", "not_recorded")),
        "object_count_band": object_count_band,
        "acceptable_anomalous_identity_ids": sorted(acceptable),
    }


def build_signal_records(
    boundary: VerifiedCalibrationBoundary,
    evaluator_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join locked predictions to evaluator correspondence and derive review signals."""

    image_rows, multiview_rows = _prediction_rows(boundary)
    images_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): dict(row)
        for row in image_rows
    }
    evaluator_by_key = {
        (str(row["opaque_scene_id"]), int(row["view_count"])): dict(row)
        for row in evaluator_rows
    }
    prediction_keys = {
        (str(row["opaque_scene_id"]), int(row["view_count"])) for row in multiview_rows
    }
    if prediction_keys != set(evaluator_by_key):
        raise ValueError("Prediction and calibration evaluator coverage differs")

    records = []
    for decision in sorted(multiview_rows, key=lambda row: (str(row["opaque_scene_id"]), int(row["view_count"]))):
        scene_id = str(decision["opaque_scene_id"])
        view_count = int(decision["view_count"])
        evaluator = evaluator_by_key[(scene_id, view_count)]
        view_rows = [images_by_key.get((scene_id, str(view_id))) for view_id in decision["view_ids"]]
        records.append(build_calibration_signal_record(decision, view_rows, evaluator))
    return records


def policy_grid(records: Sequence[Mapping[str, Any]]) -> tuple[ReviewPolicy, ...]:
    """Create a bounded threshold grid from fixed values and observed calibration quantiles."""

    def thresholds(field: str, fixed: Sequence[float]) -> list[float]:
        values = np.asarray([float(row[field]) for row in records if row.get(field) is not None], dtype=np.float64)
        observed = [] if not len(values) else [float(value) for value in np.quantile(values, (0.1, 0.25, 0.5, 0.75))]
        return sorted({round(min(1.0, max(0.0, value)), 6) for value in (*fixed, *observed)})

    margins = thresholds("margin", (0.0, 0.05, 0.10, 0.20))
    agreements = thresholds("agreement", (0.0, 0.50, 0.75, 1.0))
    qualities = thresholds("minimum_mask_quality", (0.0, 0.25, 0.50, 0.75))
    return tuple(
        ReviewPolicy(margin, agreement, quality)
        for margin in margins
        for agreement in agreements
        for quality in qualities
    )


def assign_group_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    seed: str = "visual-peer-09b-grouped-cv-v1",
) -> dict[str, int]:
    """Assign complete opaque asset groups to deterministic balanced folds."""

    groups: dict[str, int] = {}
    for row in records:
        group = str(row["opaque_asset_group_id"])
        groups[group] = groups.get(group, 0) + 1
    if len(groups) < 2:
        raise ValueError("Grouped cross-validation requires at least two asset groups")
    count = min(max(2, fold_count), len(groups))
    loads = [0] * count
    assignments: dict[str, int] = {}
    ordered = sorted(groups, key=lambda group: (-groups[group], sha256(f"{seed}|{group}".encode()).hexdigest(), group))
    for group in ordered:
        fold = min(range(count), key=lambda index: (loads[index], index))
        assignments[group] = fold
        loads[fold] += groups[group]
    return assignments


def _policy_summary(records: Sequence[Mapping[str, Any]], policy: ReviewPolicy) -> dict[str, float | int | None]:
    states = [apply_review_policy(row, policy)[0] for row in records]
    ready = [row for row, state in zip(records, states, strict=True) if state == "result_ready"]
    predicted = [row for row in records if row.get("is_correct") is not None]
    wrong_ready = [row for row in ready if row.get("is_correct") is False]
    wrong = [row for row in predicted if row.get("is_correct") is False]
    return {
        "scene_contract_count": len(records),
        "result_ready_count": len(ready),
        "review_required_count": states.count("review_required"),
        "insufficient_evidence_count": states.count("insufficient_evidence"),
        "automatic_coverage": 0.0 if not records else len(ready) / len(records),
        "review_burden": 0.0 if not records else states.count("review_required") / len(records),
        "insufficient_evidence_rate": 0.0 if not records else states.count("insufficient_evidence") / len(records),
        "wrong_result_rate": None if not predicted else len(wrong) / len(predicted),
        "false_ready_rate": None if not ready else len(wrong_ready) / len(ready),
        "wrong_result_ready_count": len(wrong_ready),
    }


def _selection_loss(summary: Mapping[str, Any], *, review_cost: float, insufficient_cost: float) -> float:
    count = max(1, int(summary["scene_contract_count"]))
    return (
        int(summary["wrong_result_ready_count"]) / count
        + review_cost * int(summary["review_required_count"]) / count
        + insufficient_cost * int(summary["insufficient_evidence_count"]) / count
    )


def choose_policy(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[ReviewPolicy],
    *,
    review_cost: float = 0.15,
    insufficient_cost: float = 0.25,
) -> tuple[ReviewPolicy, list[dict[str, Any]]]:
    """Choose the lowest documented risk-review loss, with conservative ties."""

    if not 0.0 <= review_cost <= 1.0 or not 0.0 <= insufficient_cost <= 1.0:
        raise ValueError("Policy costs must be between 0 and 1")
    rows = []
    for candidate in candidates:
        summary = _policy_summary(records, candidate)
        rows.append({
            "policy_sha256": candidate.sha256,
            "minimum_margin": candidate.minimum_margin,
            "minimum_agreement": candidate.minimum_agreement,
            "minimum_mask_quality": candidate.minimum_mask_quality,
            **summary,
            "selection_loss": _selection_loss(summary, review_cost=review_cost, insufficient_cost=insufficient_cost),
        })
    if not rows:
        raise ValueError("At least one review policy candidate is required")
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["selection_loss"]),
            1.0 if row["false_ready_rate"] is None else float(row["false_ready_rate"]),
            -float(row["automatic_coverage"]),
            -float(row["minimum_margin"]),
            -float(row["minimum_agreement"]),
            -float(row["minimum_mask_quality"]),
            str(row["policy_sha256"]),
        ),
    )
    selected_hash = str(ranked[0]["policy_sha256"])
    selected = next(candidate for candidate in candidates if candidate.sha256 == selected_hash)
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and float(other["automatic_coverage"]) >= float(row["automatic_coverage"])
            and (1.0 if other["false_ready_rate"] is None else float(other["false_ready_rate"]))
            <= (1.0 if row["false_ready_rate"] is None else float(row["false_ready_rate"]))
            and (
                float(other["automatic_coverage"]) > float(row["automatic_coverage"])
                or (1.0 if other["false_ready_rate"] is None else float(other["false_ready_rate"]))
                < (1.0 if row["false_ready_rate"] is None else float(row["false_ready_rate"]))
            )
            for other in rows
        )
        frontier.append({**row, "pareto_efficient": not dominated, "selected": row["policy_sha256"] == selected_hash})
    return selected, sorted(frontier, key=lambda row: (float(row["automatic_coverage"]), str(row["policy_sha256"])))


def grouped_cross_validation(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    review_cost: float = 0.15,
    insufficient_cost: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit review thresholds on grouped training folds and score held-out assets."""

    assignments = assign_group_folds(records, fold_count=fold_count)
    fold_rows = []
    evaluated = []
    for fold in sorted(set(assignments.values())):
        training = [row for row in records if assignments[str(row["opaque_asset_group_id"])] != fold]
        validation = [row for row in records if assignments[str(row["opaque_asset_group_id"])] == fold]
        selected, _ = choose_policy(
            training,
            policy_grid(training),
            review_cost=review_cost,
            insufficient_cost=insufficient_cost,
        )
        summary = _policy_summary(validation, selected)
        fold_rows.append({
            "fold": fold,
            "training_asset_groups": len({str(row["opaque_asset_group_id"]) for row in training}),
            "validation_asset_groups": len({str(row["opaque_asset_group_id"]) for row in validation}),
            "validation_scene_contracts": len(validation),
            "minimum_margin": selected.minimum_margin,
            "minimum_agreement": selected.minimum_agreement,
            "minimum_mask_quality": selected.minimum_mask_quality,
            **summary,
        })
        for row in validation:
            state, reasons = apply_review_policy(row, selected)
            evaluated.append({**dict(row), "cv_fold": fold, "policy_state": state, "review_reasons": list(reasons)})
    keys = [(str(row["opaque_scene_id"]), int(row["view_count"])) for row in evaluated]
    if len(keys) != len(set(keys)) or len(evaluated) != len(records):
        raise ValueError("Grouped cross-validation did not score each scene contract once")
    return sorted(fold_rows, key=lambda row: int(row["fold"])), sorted(
        evaluated, key=lambda row: (str(row["opaque_scene_id"]), int(row["view_count"]))
    )


def _risk_score(row: Mapping[str, Any]) -> float:
    """Return a label-free review risk score, not a probability."""

    margin = 0.0 if row.get("margin") is None else float(row["margin"])
    agreement = 0.0 if row.get("agreement") is None else float(row["agreement"])
    quality = float(row.get("minimum_mask_quality", 0.0))
    components = (
        float(bool(row.get("zero_proposal"))),
        float(bool(row.get("incomplete_proposals"))),
        float(bool(row.get("duplicate_or_fragmented_proposals"))),
        float(bool(row.get("empty_mask"))),
        float(bool(row.get("identity_unmatched_or_ambiguous"))),
        float(bool(row.get("stage_failure"))),
        1.0 - min(1.0, max(0.0, margin)),
        1.0 - min(1.0, max(0.0, agreement)),
        1.0 - min(1.0, max(0.0, quality)),
    )
    return float(np.mean(components))


def cross_validated_calibration_curve(
    cv_rows: Sequence[Mapping[str, Any]],
    *,
    bin_count: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calibrate error risk on training folds and report held-out reliability."""

    if bin_count < 3:
        raise ValueError("At least three calibration bins are required")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    scored = []
    for fold in sorted({int(row["cv_fold"]) for row in cv_rows}):
        training = [row for row in cv_rows if int(row["cv_fold"]) != fold and row.get("is_correct") is not None]
        validation = [row for row in cv_rows if int(row["cv_fold"]) == fold and row.get("is_correct") is not None]
        totals = np.zeros(bin_count, dtype=np.int64)
        wrong = np.zeros(bin_count, dtype=np.int64)
        for row in training:
            index = min(bin_count - 1, int(np.searchsorted(edges, _risk_score(row), side="right") - 1))
            index = max(0, index)
            totals[index] += 1
            wrong[index] += int(row["is_correct"] is False)
        global_rate = (int(wrong.sum()) + 1.0) / (int(totals.sum()) + 2.0)
        for row in validation:
            score = _risk_score(row)
            index = min(bin_count - 1, int(np.searchsorted(edges, score, side="right") - 1))
            index = max(0, index)
            probability = global_rate if totals[index] == 0 else (wrong[index] + 1.0) / (totals[index] + 2.0)
            scored.append({
                **dict(row),
                "review_risk_score": score,
                "calibrated_error_probability": float(probability),
            })
    reliability = []
    for index in range(bin_count):
        lower, upper = float(edges[index]), float(edges[index + 1])
        members = [
            row for row in scored
            if min(bin_count - 1, int(np.searchsorted(edges, float(row["calibrated_error_probability"]), side="right") - 1)) == index
        ]
        if not members:
            continue
        wrong_count = sum(row["is_correct"] is False for row in members)
        low, high = wilson_interval(wrong_count, len(members))
        reliability.append({
            "probability_bin": index,
            "lower_bound": lower,
            "upper_bound": upper,
            "scene_contract_count": len(members),
            "mean_calibrated_error_probability": float(np.mean([row["calibrated_error_probability"] for row in members])),
            "observed_wrong_result_rate": wrong_count / len(members),
            "observed_rate_ci_low": low,
            "observed_rate_ci_high": high,
        })
    return scored, reliability


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Return a Wilson confidence interval for one binomial rate."""

    if total == 0:
        return None, None
    if not 0 <= successes <= total:
        raise ValueError("Binomial counts are invalid")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate_with_ci(numerator: int, denominator: int) -> tuple[float | None, float | None, float | None]:
    if denominator == 0:
        return None, None, None
    low, high = wilson_interval(numerator, denominator)
    return numerator / denominator, low, high


def clustered_rate_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    numerator: Callable[[Mapping[str, Any]], bool],
    denominator: Callable[[Mapping[str, Any]], bool],
    resamples: int = 500,
    seed: str = "visual-peer-09b-clustered-interval-v1",
) -> tuple[float | None, float | None, float | None]:
    """Bootstrap a rate by resampling complete base asset groups."""

    if resamples < 100:
        raise ValueError("At least 100 grouped bootstrap resamples are required")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["opaque_asset_group_id"]), []).append(row)
    group_ids = sorted(grouped)
    observed_denominator = sum(denominator(row) for row in rows)
    if not group_ids or observed_denominator == 0:
        return None, None, None
    observed = sum(numerator(row) for row in rows) / observed_denominator
    random_seed = int(sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    generator = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(resamples):
        sampled = generator.choice(group_ids, size=len(group_ids), replace=True)
        sampled_rows = [row for group_id in sampled for row in grouped[str(group_id)]]
        sampled_denominator = sum(denominator(row) for row in sampled_rows)
        if sampled_denominator:
            estimates.append(sum(numerator(row) for row in sampled_rows) / sampled_denominator)
    if not estimates:
        return observed, observed, observed
    low, high = np.quantile(np.asarray(estimates, dtype=np.float64), (0.025, 0.975))
    return observed, float(low), float(high)


def primary_metric_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return defined primary policy metrics with asset-grouped intervals."""

    total = len(rows)
    ready = [row for row in rows if row["policy_state"] == "result_ready"]
    review = [row for row in rows if row["policy_state"] == "review_required"]
    predicted = [row for row in rows if row.get("is_correct") is not None]
    wrong = [row for row in predicted if row["is_correct"] is False]
    wrong_ready = [row for row in ready if row.get("is_correct") is False]
    definitions = (
        (
            "automatic_coverage",
            len(ready),
            total,
            lambda row: row["policy_state"] == "result_ready",
            lambda _row: True,
            "result_ready divided by all scene-view contracts",
        ),
        (
            "wrong_result_rate",
            len(wrong),
            len(predicted),
            lambda row: row.get("is_correct") is False,
            lambda row: row.get("is_correct") is not None,
            "wrong fused result divided by contracts with a fused result",
        ),
        (
            "false_ready_rate",
            len(wrong_ready),
            len(ready),
            lambda row: row["policy_state"] == "result_ready" and row.get("is_correct") is False,
            lambda row: row["policy_state"] == "result_ready",
            "wrong result_ready divided by all result_ready contracts",
        ),
        (
            "review_burden",
            len(review),
            total,
            lambda row: row["policy_state"] == "review_required",
            lambda _row: True,
            "review_required divided by all scene-view contracts",
        ),
    )
    output = []
    for metric, numerator_count, denominator_count, numerator, denominator, definition in definitions:
        value, low, high = clustered_rate_interval(
            rows,
            numerator=numerator,
            denominator=denominator,
            seed=f"visual-peer-09b-primary-{metric}",
        )
        output.append({
            "metric": metric,
            "numerator": numerator_count,
            "denominator": denominator_count,
            "value": value,
            "ci_low": low,
            "ci_high": high,
            "ci_method": "base_asset_grouped_bootstrap_500",
            "definition": definition,
        })
    return output


def reason_burden_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report how often each reason sends a contract to review."""

    total = len(rows)
    reasons = sorted({reason for row in rows for reason in row.get("review_reasons", [])})
    output = []
    for reason in reasons:
        count = sum(reason in row.get("review_reasons", []) for row in rows)
        value, low, high = _rate_with_ci(count, total)
        output.append({
            "reason": reason,
            "scene_contract_count": count,
            "share_of_all_contracts": value,
            "ci_low": low,
            "ci_high": high,
        })
    return output or [{"reason": "none", "scene_contract_count": 0, "share_of_all_contracts": 0.0, "ci_low": 0.0, "ci_high": 0.0}]


def subgroup_metric_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report coverage and false-ready intervals for fixed and available subgroups."""

    output = []
    for field in ("view_count", "category", "anomaly_type", "object_count_band"):
        for value in sorted({str(row[field]) for row in rows}):
            members = [row for row in rows if str(row[field]) == value]
            ready = [row for row in members if row["policy_state"] == "result_ready"]
            wrong_ready = sum(row.get("is_correct") is False for row in ready)
            coverage, coverage_low, coverage_high = clustered_rate_interval(
                members,
                numerator=lambda row: row["policy_state"] == "result_ready",
                denominator=lambda _row: True,
                seed=f"visual-peer-09b-subgroup-coverage-{field}-{value}",
            )
            false_ready, false_low, false_high = clustered_rate_interval(
                members,
                numerator=lambda row: row["policy_state"] == "result_ready" and row.get("is_correct") is False,
                denominator=lambda row: row["policy_state"] == "result_ready",
                seed=f"visual-peer-09b-subgroup-false-ready-{field}-{value}",
            )
            output.append({
                "subgroup_field": field,
                "subgroup_value": value,
                "scene_contract_count": len(members),
                "result_ready_count": len(ready),
                "automatic_coverage": coverage,
                "coverage_ci_low": coverage_low,
                "coverage_ci_high": coverage_high,
                "wrong_result_ready_count": wrong_ready,
                "false_ready_rate": false_ready,
                "false_ready_ci_low": false_low,
                "false_ready_ci_high": false_high,
                "ci_method": "base_asset_grouped_bootstrap_500",
            })
    return output


def state_summary_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Count each policy state for each fixed view contract."""

    output = []
    for view_count in EXPECTED_VIEW_COUNTS:
        subset = [row for row in rows if int(row["view_count"]) == view_count]
        for state in POLICY_STATES:
            count = sum(row["policy_state"] == state for row in subset)
            output.append({
                "view_count": view_count,
                "policy_state": state,
                "scene_contract_count": count,
                "share": 0.0 if not subset else count / len(subset),
            })
    return output


def freeze_review_policy(
    *,
    boundary: VerifiedCalibrationBoundary,
    policy: ReviewPolicy,
    cv_rows: Sequence[Mapping[str, Any]],
    evaluator_access_receipt: Mapping[str, Any],
    output_path: Path,
    review_cost: float,
    insufficient_cost: float,
) -> dict[str, Any]:
    """Freeze only the calibration-derived review policy and its provenance."""

    method = json.loads(boundary.method_record_path.read_text(encoding="utf-8"))
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "method_record_sha256": boundary.method_record_sha256,
        "chosen_configuration_sha256": boundary.chosen_configuration_sha256,
        "calibration_prediction_lock_sha256": boundary.lock_sha256,
        "calibration_evaluator_sha256": evaluator_access_receipt["evaluator_sha256"],
        "calibration_cohort_id": CALIBRATION_CONTRACT.cohort_id,
        "calibration_membership_sha256": CALIBRATION_CONTRACT.membership_sha256,
        "calibration_scene_count": CALIBRATION_CONTRACT.scene_count,
        "calibration_asset_group_count": CALIBRATION_CONTRACT.asset_group_count,
        "minimum_margin": policy.minimum_margin,
        "minimum_agreement": policy.minimum_agreement,
        "minimum_mask_quality": policy.minimum_mask_quality,
        "policy_configuration": asdict(policy),
        "policy_configuration_sha256": policy.sha256,
        "immutable_pipeline_identity": dict(boundary.lock_record["identity"]),
        "immutable_pipeline_identity_sha256": boundary.lock_record["identity_sha256"],
        "immutable_method_configuration": method["chosen_configuration"],
        "selection": {
            "cross_validation": "base_asset_grouped",
            "fold_count": len({int(row["cv_fold"]) for row in cv_rows}),
            "review_cost": review_cost,
            "insufficient_evidence_cost": insufficient_cost,
            "raw_p1_is_probability": False,
            "universal_accuracy_gate": None,
        },
        "states": {
            "result_ready": "A relative unusualness result is available without a triggered review condition.",
            "review_required": "A relative unusualness result is available, but one or more evidence checks require review.",
            "insufficient_evidence": "The pipeline cannot support a relative unusualness result for this input.",
        },
        "application_claim": "relative unusualness among the uploaded peer objects",
        "all_normal_scene_claim_supported": False,
    }
    record["policy_record_sha256"] = _digest(record)
    _write_json(Path(output_path), record)
    return record


def _natural_key(value: str) -> tuple[Any, ...]:
    import re

    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _evaluator_example_strata(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Describe private audit strata without changing the complete row table."""

    strata = [f"policy_state:{row['policy_state']}"]
    if row["policy_state"] == "result_ready" and row.get("is_correct") is False:
        strata.append("wrong_result_ready")
    acceptable_ids = tuple(
        str(value) for value in row.get("acceptable_anomalous_identity_ids", [])
    )
    if len(acceptable_ids) > 1:
        strata.append("multiple_acceptable_anomalies")
    for reason in (*ALWAYS_REVIEW_REASONS, "insufficient_evidence"):
        if bool(row.get(reason)):
            strata.append(f"terminal_signal:{reason}")
    strata.extend(
        f"review_reason:{reason}"
        for reason in sorted({str(value) for value in row.get("review_reasons", [])})
    )
    return tuple(strata)


def select_evaluator_examples(
    rows: Sequence[Mapping[str, Any]], *, count: int = 16
) -> list[dict[str, Any]]:
    """Select a bounded deterministic private audit across available strata."""

    if count < 1:
        raise ValueError("Example count must be positive")

    row_keys = [
        (str(row["opaque_scene_id"]), int(row["view_count"])) for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Evaluator example rows must have unique scene and view keys")

    def stable_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        scene_id = str(row["opaque_scene_id"])
        view_count = int(row["view_count"])
        payload = f"visual-peer-09b-private-example-v2|{scene_id}|{view_count}"
        return sha256(payload.encode()).hexdigest(), scene_id, view_count

    ordered = sorted(rows, key=stable_key)
    available_review_reasons = sorted(
        {
            str(reason)
            for row in ordered
            for reason in row.get("review_reasons", [])
        }
    )
    priority_strata = (
        "wrong_result_ready",
        *(f"policy_state:{state}" for state in POLICY_STATES),
        "multiple_acceptable_anomalies",
        *(
            f"terminal_signal:{reason}"
            for reason in (*ALWAYS_REVIEW_REASONS, "insufficient_evidence")
        ),
        *(f"review_reason:{reason}" for reason in available_review_reasons),
    )
    strata_by_key = {
        (str(row["opaque_scene_id"]), int(row["view_count"])): set(
            _evaluator_example_strata(row)
        )
        for row in ordered
    }
    selected: list[Mapping[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    covered: set[str] = set()
    for stratum in priority_strata:
        if len(selected) >= count:
            break
        if stratum in covered:
            continue
        candidates = [
            row
            for row in ordered
            if (str(row["opaque_scene_id"]), int(row["view_count"]))
            not in selected_keys
            and stratum
            in strata_by_key[
                (str(row["opaque_scene_id"]), int(row["view_count"]))
            ]
        ]
        if not candidates:
            continue
        chosen = candidates[0]
        chosen_key = (str(chosen["opaque_scene_id"]), int(chosen["view_count"]))
        selected.append(chosen)
        selected_keys.add(chosen_key)
        covered.update(strata_by_key[chosen_key])

    for row in ordered:
        if len(selected) >= count:
            break
        row_key = (str(row["opaque_scene_id"]), int(row["view_count"]))
        if row_key not in selected_keys:
            selected.append(row)
            selected_keys.add(row_key)

    return [
        {
            **dict(row),
            "example_selection_strata": list(_evaluator_example_strata(row)),
        }
        for row in selected
    ]


def render_evaluator_examples(
    *,
    rows: Sequence[Mapping[str, Any]],
    boundary: VerifiedCalibrationBoundary,
    model_input_root: Path,
    output_root: Path,
    count: int = 16,
) -> list[dict[str, Any]]:
    """Render deterministic post-evaluation panels inside evaluator-only storage."""

    model_root = Path(model_input_root).expanduser().resolve(strict=True)
    image_rows, multiview_rows = _prediction_rows(boundary)
    images_by_key = {(str(row["opaque_scene_id"]), str(row["view_id"])): row for row in image_rows}
    decisions = {(str(row["opaque_scene_id"]), int(row["view_count"])): row for row in multiview_rows}
    inventory = []
    for index, evaluated in enumerate(select_evaluator_examples(rows, count=count), 1):
        scene_id = str(evaluated["opaque_scene_id"])
        view_count = int(evaluated["view_count"])
        decision = decisions[(scene_id, view_count)]
        member_identity = {
            (str(member["view_id"]), str(member["candidate_id"])): str(track["identity_id"])
            for track in decision.get("tracks", [])
            for member in track.get("members", [])
        }
        views = []
        for view_id in decision["view_ids"]:
            item = images_by_key.get((scene_id, str(view_id)))
            image_path = model_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            if item is None or item.get("terminal_failure") is not None or not image_path.is_file():
                views.append(QualitativeView(view_id=str(view_id), rgb=None, missing_reason="prediction unavailable"))
                continue
            objects = []
            heatmap = None
            for predicted in item.get("objects", []):
                identity_id = member_identity.get((str(view_id), str(predicted["object_id"])), str(predicted["object_id"]))
                objects.append(PredictedObject(
                    object_id=identity_id,
                    box_xyxy=tuple(float(value) for value in predicted["box_xyxy"]),
                    sam_mask=decode_binary_mask(predicted["mask_rle"]),
                    p1_score=predicted.get("p1_score"),
                    p1_rank=predicted.get("p1_rank"),
                ))
                if identity_id == decision.get("fused_odd_one_out"):
                    heatmap = predicted.get("p1_heatmap")
            views.append(QualitativeView(
                view_id=str(view_id),
                rgb=image_path,
                objects=tuple(objects),
                heatmap=None if heatmap is None else np.asarray(heatmap, dtype=np.float32),
            ))
        qualitative = QualitativeResult(
            result_id=scene_id,
            views=tuple(views),
            fused_odd_one_out=decision.get("fused_odd_one_out"),
            agreement=decision.get("agreement"),
            margin=decision.get("margin"),
            review_state=str(evaluated["policy_state"]),
            publication_metadata={"route": "frozen_automatic", "view_count": view_count},
        )
        evaluator_views = []
        for view in evaluated.get("reference_views", []):
            evaluator_views.append(EvaluatorView(
                view_id=str(view["view_id"]),
                objects=tuple(EvaluatorObject(
                    object_id=str(obj["object_id"]),
                    box_xyxy=None if obj.get("box_xyxy") is None else tuple(float(value) for value in obj["box_xyxy"]),
                    mask=obj.get("mask"),
                ) for obj in view.get("objects", [])),
            ))
        acceptable_ids = tuple(
            sorted(
                {
                    str(value)
                    for value in evaluated["acceptable_anomalous_identity_ids"]
                },
                key=_natural_key,
            )
        )
        overlay = EvaluatorOverlay(
            views=tuple(evaluator_views),
            acceptable_reference_ids=acceptable_ids,
        )
        prefix = "wrong-result-ready" if evaluated["policy_state"] == "result_ready" and evaluated.get("is_correct") is False else "calibration-example"
        output_path = Path(output_root) / f"{prefix}-{index:02d}.png"
        save_panel_png(render_evaluator_overlay_panel(qualitative, overlay, panel_size=220), output_path)
        inventory.append({
            "example_number": index,
            "path": output_path.as_posix(),
            "policy_state": evaluated["policy_state"],
            "wrong_result_ready": prefix == "wrong-result-ready",
            "review_reasons": list(evaluated.get("review_reasons", [])),
            "selection_strata": list(evaluated["example_selection_strata"]),
            "acceptable_anomalous_identity_ids": list(acceptable_ids),
        })
    return inventory


def artifact_manifest(root: Path, *, excluded_names: set[str] | None = None) -> dict[str, Any]:
    """Hash the files below one output root."""

    excluded = set() if excluded_names is None else set(excluded_names)
    files = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Evaluation outputs cannot contain symbolic links")
        if path.is_file() and path.name not in excluded:
            files.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            })
    return {
        "schema_version": 1,
        "artifact_count": len(files),
        "artifacts": files,
        "artifact_inventory_sha256": _digest(files),
    }


def exact_resume_receipt_valid(receipt_path: Path, input_hashes: Mapping[str, str]) -> bool:
    """Return true only when exact inputs and every recorded output still match."""

    path = Path(receipt_path)
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt.get("input_hashes") != dict(input_hashes):
            return False
        root = path.parent
        return all(file_sha256(root / item["path"]) == item["sha256"] for item in receipt.get("outputs", []))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def write_exact_resume_receipt(
    receipt_path: Path,
    *,
    input_hashes: Mapping[str, str],
    output_paths: Sequence[Path],
) -> dict[str, Any]:
    """Seal completed evaluation outputs for an exact restart."""

    path = Path(receipt_path)
    root = path.parent.resolve()
    rows = []
    for output in sorted(Path(item).resolve() for item in output_paths):
        if root not in output.parents:
            raise ValueError("Resume receipt outputs must remain below its output root")
        rows.append({"path": output.relative_to(root).as_posix(), "sha256": file_sha256(output)})
    record = {
        "schema_version": 1,
        "status": "complete",
        "input_hashes": dict(input_hashes),
        "outputs": rows,
        "outputs_sha256": _digest(rows),
        "exact_restart_available": True,
    }
    _write_json(path, record)
    return record


__all__ = [
    "ALWAYS_REVIEW_REASONS",
    "EVALUATOR_SCHEMA",
    "EXPECTED_VIEW_COUNTS",
    "POLICY_STATES",
    "PREDICTION_SIGNAL_FIELDS",
    "SCHEMA_VERSION",
    "ReviewPolicy",
    "VerifiedCalibrationBoundary",
    "apply_review_policy",
    "artifact_manifest",
    "assign_group_folds",
    "build_calibration_signal_record",
    "build_signal_records",
    "choose_policy",
    "clustered_rate_interval",
    "cross_validated_calibration_curve",
    "exact_resume_receipt_valid",
    "freeze_review_policy",
    "grouped_cross_validation",
    "open_calibration_evaluator",
    "policy_grid",
    "primary_metric_table",
    "reason_burden_table",
    "render_evaluator_examples",
    "select_evaluator_examples",
    "state_summary_table",
    "subgroup_metric_table",
    "verify_calibration_boundary",
    "wilson_interval",
    "write_csv",
    "write_exact_resume_receipt",
]
