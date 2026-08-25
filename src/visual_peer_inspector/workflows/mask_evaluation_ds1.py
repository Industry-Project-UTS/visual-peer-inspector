"""Evaluator-only DS1 proposal and SAM mask measurement workflow."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.ds1_mask_route_package import build_ds1_mask_route_package
from visual_peer_inspector.data.model_safe_package import canonical_json_bytes, file_sha256, read_jsonl
from visual_peer_inspector.evaluation.ds1_evaluator_package import verify_ds1_evaluator_package
from visual_peer_inspector.evaluation.masks import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    DEFAULT_MASK_IOU_THRESHOLD,
    evaluate_mask_route_image,
    paired_asset_clustered_bootstrap,
    summarize_mask_routes,
)
from visual_peer_inspector.evaluation.proposals import (
    clustered_bootstrap_intervals,
    evaluate_proposal_dataset,
    summarize_proposal_evaluation,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram, save_scatter_plot, write_csv_rows
from visual_peer_inspector.models.sam2 import binary_mask_sha256, decode_binary_mask, encode_binary_mask
from visual_peer_inspector.models.locateanything import (
    MODEL_PRODUCER_KIND as LOCATEANYTHING_MODEL_PRODUCER_KIND,
    validate_proposal_result,
)
from visual_peer_inspector.models.sam2 import MODEL_PRODUCER_KIND as SAM_MODEL_PRODUCER_KIND


STAGE = "ds1_proposal_and_mask_evaluation"
ROUTES = ("reference_box", "automatic_box")
PROPOSAL_RECALL_MINIMUM = 0.95
PROPOSAL_PRECISION_MINIMUM = 0.90
REFERENCE_MASK_RECALL_MINIMUM = 0.90
REFERENCE_MASK_IOU_MINIMUM = 0.90
AUTOMATIC_MASK_RECALL_MINIMUM = 0.90
AUTOMATIC_MASK_IOU_MINIMUM = 0.90
SUBGROUP_MAXIMUM_ALLOWED_DROP = 0.10
SUBGROUP_MINIMUM_CLUSTERS = 5
SUBGROUP_MINIMUM_SCENES = 10
SUBGROUP_MINIMUM_SUPPORT = 50


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _result_path(root: Path, filename: str) -> Path:
    choices = (root / filename, root / "private" / filename)
    existing = [path for path in choices if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Expected exactly one {filename} under {root}")
    return existing[0]


def _reference_mask(root: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = root / "masks" / str(row["mask_path"])
    if file_sha256(path) != str(row["mask_sha256"]):
        raise ValueError("Supplied mask differs from its evaluator digest")
    with Image.open(path) as opened:
        mask = np.asarray(opened.convert("L")) >= 128
    if mask.shape != (256, 256) or not mask.any():
        raise ValueError("Visible supplied mask has invalid geometry")
    return mask


def _aggregate_mask_group(rows: Sequence[Mapping[str, Any]], dimension: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values = row.get(dimension)
        if dimension == "anomaly_family":
            values = values or ["none"]
        elif not isinstance(values, list):
            values = [values]
        for value in values:
            grouped[(str(value), str(row["route"]))].append(row)
    output = []
    for (value, route), items in sorted(grouped.items()):
        summary = summarize_mask_routes(items)[0]
        output.append({"dimension": dimension, "value": value, **summary})
    return output


def _proposal_subgroups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions = ("view_id", "category", "object_count", "anomaly_families")
    output: list[dict[str, Any]] = []
    for dimension in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            values = row.get(dimension)
            if dimension == "anomaly_families":
                values = values or ["none"]
            elif not isinstance(values, list):
                values = [values]
            for value in values:
                groups[str(value)].append(row)
        for value, items in sorted(groups.items()):
            summary = summarize_proposal_evaluation(items)
            output.append({
                "dimension": "anomaly_family" if dimension == "anomaly_families" else dimension,
                "value": value,
                "image_count": summary["image_count"],
                "proposal_count": summary["proposal_count"],
                "reference_count": summary["reference_count"],
                "precision_at_0_50": summary["micro_precision_at_0_50"],
                "recall_at_0_50": summary["micro_recall_at_0_50"],
                "mean_matched_iou": summary["mean_matched_iou"],
                "miss_count": summary["miss_count"],
                "extra_count": summary["extra_count"],
                "merge_candidate_count": summary["merge_candidate_count"],
                "split_candidate_count": summary["split_candidate_count"],
            })
    return output


def _paired_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    paired: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        if row["record_kind"] == "unmatched_prediction" or row["route"] not in ROUTES:
            continue
        key = (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["opaque_object_id"]))
        paired[key][str(row["route"])] = row
    output = []
    for key, routes in sorted(paired.items()):
        if set(routes) != set(ROUTES):
            continue
        left, right = routes["reference_box"], routes["automatic_box"]
        output.append({
            "opaque_scene_id": key[0],
            "opaque_asset_group_id": left["opaque_asset_group_id"],
            "reference_iou": float(left["mask_iou"]),
            "automatic_iou": float(right["mask_iou"]),
            "reference_dice": float(left["dice"]),
            "automatic_dice": float(right["dice"]),
            "reference_boundary_f1": float(left["boundary_f1"]),
            "automatic_boundary_f1": float(right["boundary_f1"]),
        })
    return output


def _artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    _write_json(root / "artifact_manifest.json", {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": sha256(b"".join(canonical_json_bytes(row) + b"\n" for row in rows)).hexdigest(),
    })


def _verify_artifact_manifest(root: Path, required_path: str) -> dict[str, Any]:
    """Verify a complete upstream artifact root and one required payload."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("Upstream artifact root is missing or unsafe")
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ValueError("Upstream artifact manifest count is invalid")
    declared: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        path = root / relative
        if (
            not relative
            or relative in declared
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != str(row.get("sha256"))
            or path.stat().st_size != int(row.get("size_bytes", -1))
        ):
            raise ValueError("Upstream artifact manifest contains invalid evidence")
        declared.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != declared or required_path not in declared:
        raise ValueError("Upstream artifact manifest coverage is incomplete")
    expected_content = sha256(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    ).hexdigest()
    if manifest.get("manifest_content_sha256") != expected_content:
        raise ValueError("Upstream artifact manifest content hash differs")
    return manifest


def _expected_cohort() -> dict[str, Any]:
    return {
        key: getattr(DS1_SCREEN_CONTRACT, key)
        for key in (
            "cohort_id",
            "release_id",
            "role",
            "pilot_stage",
            "membership_sha256",
            "scene_count",
            "views_per_scene",
            "image_count",
            "asset_group_count",
        )
    }


def _verify_upstream_identity(
    evaluator_root: Path,
    proposal_root: Path,
    sam_root: Path,
    proposal_path: Path,
    sam_path: Path,
    proposal_rows: Sequence[Mapping[str, Any]],
    sam_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Bind the evaluator to exact accepted model runs and their prompt chain."""

    evaluator_record = json.loads(
        (evaluator_root / "package_record.json").read_text(encoding="utf-8")
    )
    crosswalk_record = json.loads(
        (evaluator_root / "crosswalk_package_record.json").read_text(encoding="utf-8")
    )
    if evaluator_record.get("proposal_results_sha256") != file_sha256(proposal_path):
        raise ValueError("Evaluator crosswalk identifies another proposal result set")
    if crosswalk_record.get("proposal_results_sha256") != file_sha256(proposal_path):
        raise ValueError("Evaluator crosswalk package identifies another proposal result set")

    proposal_safe = proposal_root / "safe"
    proposal_private = proposal_root / "private"
    _verify_artifact_manifest(proposal_safe, "run_record.json")
    _verify_artifact_manifest(proposal_private, "proposal_results.jsonl")
    proposal_run_path = proposal_safe / "run_record.json"
    proposal_run = json.loads(proposal_run_path.read_text(encoding="utf-8"))
    if (
        proposal_run.get("stage") != "ds1_locateanything_screen"
        or proposal_run.get("status") != "completed"
        or proposal_run.get("run_mode") != "model_inference"
        or proposal_run.get("inference_stage") != "full"
        or proposal_run.get("producer_kind") != LOCATEANYTHING_MODEL_PRODUCER_KIND
        or proposal_run.get("completion_gate_passed") is not True
        or any(
            proposal_run.get("cohort", {}).get(key) != value
            for key, value in _expected_cohort().items()
        )
    ):
        raise ValueError("Notebook 04.1 evidence is not a completed DS1 model run")
    for row in proposal_rows:
        validate_proposal_result(row)
    proposal_config = str(proposal_run.get("configuration_sha256", ""))
    proposal_inputs = {str(row["input_manifest_sha256"]) for row in proposal_rows}
    proposal_output_identity = proposal_run.get("output_identity", {})
    if (
        len(proposal_inputs) != 1
        or {str(row["configuration_sha256"]) for row in proposal_rows}
        != {proposal_config}
        or {str(row["producer_kind"]) for row in proposal_rows}
        != {LOCATEANYTHING_MODEL_PRODUCER_KIND}
        or any(row.get("status") != "ok" for row in proposal_rows)
        or proposal_output_identity.get("result_filename") != "proposal_results.jsonl"
        or proposal_output_identity.get("result_sha256") != file_sha256(proposal_path)
        or proposal_output_identity.get("proposal_results_sha256")
        != file_sha256(proposal_path)
        or int(proposal_output_identity.get("result_count", -1)) != len(proposal_rows)
    ):
        raise ValueError("Notebook 04.1 proposal rows differ from their accepted run")
    model_input_manifest_sha256 = next(iter(proposal_inputs))
    if (
        crosswalk_record.get("model_input_manifest_sha256")
        != model_input_manifest_sha256
    ):
        raise ValueError("Evaluator crosswalk identifies another RGB manifest")

    sam_safe = sam_root / "safe"
    sam_private = sam_root / "private"
    _verify_artifact_manifest(sam_safe, "run_record.json")
    _verify_artifact_manifest(sam_private, "mask_results.jsonl")
    sam_run_path = sam_safe / "run_record.json"
    sam_run = json.loads(sam_run_path.read_text(encoding="utf-8"))
    prompt_record_sha256 = str(crosswalk_record.get("prompt_package_record_sha256", ""))
    if (
        sam_run.get("stage") != "ds1_sam21_prompted_masks"
        or sam_run.get("status") != "completed"
        or sam_run.get("run_mode") != "model_inference"
        or sam_run.get("inference_stage") != "full"
        or sam_run.get("producer_kind") != SAM_MODEL_PRODUCER_KIND
        or sam_run.get("completion_gate_passed") is not True
        or any(
            sam_run.get("cohort", {}).get(key) != value
            for key, value in _expected_cohort().items()
        )
        or sam_run.get("prompt_package_sha256") != prompt_record_sha256
    ):
        raise ValueError("Notebook 05.1 evidence is not the linked completed DS1 model run")
    sam_configuration = str(sam_run.get("configuration_sha256", ""))
    sam_output_identity = sam_run.get("output_identity", {})
    sam_input_identity = sam_run.get("input_identity", {})
    if not sam_configuration or not prompt_record_sha256:
        raise ValueError("Notebook 05.1 evidence is missing frozen configuration hashes")
    if (
        sam_input_identity.get("prompt_package_record_sha256")
        != prompt_record_sha256
        or sam_input_identity.get("prompt_manifest_sha256")
        != crosswalk_record.get("prompt_manifest_sha256")
        or sam_output_identity.get("mask_results_sha256")
        != file_sha256(sam_path)
        or int(sam_output_identity.get("mask_result_count", -1)) != len(sam_rows)
    ):
        raise ValueError("Notebook 05.1 run record does not commit its mask results")
    for row in sam_rows:
        if (
            row.get("producer_kind") != SAM_MODEL_PRODUCER_KIND
            or row.get("cohort_id") != DS1_SCREEN_CONTRACT.cohort_id
            or row.get("status") != "ok"
            or row.get("configuration_sha256") != sam_configuration
            or row.get("prompt_package_sha256") != prompt_record_sha256
            or set(row.get("routes", {})) != set(ROUTES)
        ):
            raise ValueError("Notebook 05.1 mask rows differ from their accepted run")
    return {
        "proposal_run_record_sha256": file_sha256(proposal_run_path),
        "proposal_artifact_manifest_sha256": file_sha256(
            proposal_safe / "artifact_manifest.json"
        ),
        "sam_run_record_sha256": file_sha256(sam_run_path),
        "sam_artifact_manifest_sha256": file_sha256(
            sam_safe / "artifact_manifest.json"
        ),
        "prompt_package_record_sha256": prompt_record_sha256,
        "prompt_manifest_sha256": str(crosswalk_record["prompt_manifest_sha256"]),
        "model_input_manifest_sha256": model_input_manifest_sha256,
        "proposal_configuration_sha256": proposal_config,
        "sam_configuration_sha256": sam_configuration,
    }


def _dimension_values(row: Mapping[str, Any], dimension: str) -> list[str]:
    value = row.get(dimension)
    if dimension in {"anomaly_family", "anomaly_families"}:
        values = value or ["none"]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(item) for item in values]


def _cluster_ratio_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    by_cluster: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        cluster = str(row["opaque_asset_group_id"])
        by_cluster[cluster][0] += float(row[numerator_key])
        by_cluster[cluster][1] += float(row[denominator_key])
    values = np.asarray([by_cluster[key] for key in sorted(by_cluster)], dtype=np.float64)
    if not len(values) or values[:, 1].sum() <= 0:
        raise ValueError("Subgroup bootstrap denominator is empty")
    point = float(values[:, 0].sum() / values[:, 1].sum())
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    sampled = values[draws].sum(axis=1)
    ratios = np.divide(
        sampled[:, 0],
        sampled[:, 1],
        out=np.zeros(len(sampled), dtype=np.float64),
        where=sampled[:, 1] > 0,
    )
    return point, float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _subgroup_bootstrap_evidence(
    proposal_records: Sequence[Mapping[str, Any]],
    mask_records: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return declared subgroup intervals, worst supported rows and gates."""

    specifications: list[tuple[str, str, str, float, list[dict[str, Any]]]] = []
    proposal_dimensions = ("view_id", "category", "object_count", "anomaly_families")
    for metric, numerator, denominator, threshold in (
        ("precision_at_0_50", "matched_at_0_50", "proposal_count", PROPOSAL_PRECISION_MINIMUM),
        ("recall_at_0_50", "matched_at_0_50", "reference_count", PROPOSAL_RECALL_MINIMUM),
    ):
        for dimension in proposal_dimensions:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source in proposal_records:
                for value in _dimension_values(source, dimension):
                    grouped[value].append({
                        **source,
                        "_numerator": float(source[numerator]),
                        "_denominator": float(source[denominator]),
                    })
            for value, rows in grouped.items():
                specifications.append((
                    "proposal",
                    "anomaly_family" if dimension == "anomaly_families" else dimension,
                    value,
                    threshold,
                    [{**row, "_metric": metric} for row in rows],
                ))

    mask_dimensions = ("view_id", "category", "object_count", "anomaly_family")
    for route in ROUTES:
        reference_rows = [
            row
            for row in mask_records
            if row["route"] == route and row["record_kind"] != "unmatched_prediction"
        ]
        for metric, numerator, threshold in (
            ("detection_aware_recall_at_0_75", "passes_mask_iou_threshold", REFERENCE_MASK_RECALL_MINIMUM if route == "reference_box" else AUTOMATIC_MASK_RECALL_MINIMUM),
            ("detection_aware_mean_iou", "mask_iou", REFERENCE_MASK_IOU_MINIMUM if route == "reference_box" else AUTOMATIC_MASK_IOU_MINIMUM),
        ):
            for dimension in mask_dimensions:
                grouped = defaultdict(list)
                for source in reference_rows:
                    for value in _dimension_values(source, dimension):
                        grouped[value].append({
                            **source,
                            "_numerator": float(source[numerator]),
                            "_denominator": 1.0,
                        })
                for value, rows in grouped.items():
                    specifications.append((
                        route,
                        dimension,
                        value,
                        threshold,
                        [{**row, "_metric": metric} for row in rows],
                    ))

    interval_rows: list[dict[str, Any]] = []
    for component, dimension, value, threshold, rows in specifications:
        clusters = {str(row["opaque_asset_group_id"]) for row in rows}
        scenes = {str(row["opaque_scene_id"]) for row in rows}
        support_count = int(sum(float(row["_denominator"]) for row in rows))
        support_unit = (
            "proposals"
            if rows[0]["_metric"] == "precision_at_0_50"
            else "references"
        )
        eligible = (
            len(clusters) >= SUBGROUP_MINIMUM_CLUSTERS
            and len(scenes) >= SUBGROUP_MINIMUM_SCENES
            and support_count >= SUBGROUP_MINIMUM_SUPPORT
        )
        stable_seed = seed + int(
            sha256(
                f"{component}|{dimension}|{value}|{rows[0]['_metric']}".encode()
            ).hexdigest()[:8],
            16,
        )
        numerator_total = sum(float(row["_numerator"]) for row in rows)
        denominator_total = sum(float(row["_denominator"]) for row in rows)
        point = 0.0 if denominator_total <= 0 else float(
            numerator_total / denominator_total
        )
        lower: float | str = ""
        upper: float | str = ""
        if eligible:
            point, lower, upper = _cluster_ratio_interval(
                rows,
                numerator_key="_numerator",
                denominator_key="_denominator",
                repetitions=repetitions,
                seed=stable_seed,
            )
        interval_rows.append({
            "component": component,
            "metric": rows[0]["_metric"],
            "dimension": dimension,
            "value": value,
            "point_estimate": point,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "bootstrap_repetitions": repetitions if eligible else 0,
            "cluster_count": len(clusters),
            "scene_count": len(scenes),
            "support_count": support_count,
            "support_unit": support_unit,
            "minimum_clusters": SUBGROUP_MINIMUM_CLUSTERS,
            "minimum_scenes": SUBGROUP_MINIMUM_SCENES,
            "minimum_support": SUBGROUP_MINIMUM_SUPPORT,
            "eligible": "yes" if eligible else "no",
            "declared_overall_threshold": threshold,
            "worst_subgroup_threshold": threshold - SUBGROUP_MAXIMUM_ALLOWED_DROP,
        })

    worst_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    keys = sorted({(row["component"], row["metric"]) for row in interval_rows})
    for component, metric in keys:
        eligible_rows = [
            row
            for row in interval_rows
            if row["component"] == component
            and row["metric"] == metric
            and row["eligible"] == "yes"
        ]
        if not eligible_rows:
            raise ValueError(f"No supported subgroup exists for {component} {metric}")
        worst = min(
            eligible_rows,
            key=lambda row: (
                float(row["point_estimate"]),
                str(row["dimension"]),
                str(row["value"]),
            ),
        )
        gate_observed = float(worst["ci_lower_95"])
        status = "pass" if gate_observed >= float(
            worst["worst_subgroup_threshold"]
        ) else "fail"
        worst_rows.append({
            "component": component,
            "metric": metric,
            "dimension": worst["dimension"],
            "value": worst["value"],
            "point_estimate": worst["point_estimate"],
            "ci_lower_95": worst["ci_lower_95"],
            "cluster_count": worst["cluster_count"],
            "scene_count": worst["scene_count"],
            "support_count": worst["support_count"],
            "support_unit": worst["support_unit"],
            "gate_statistic": "ci_lower_95",
            "gate_observed": gate_observed,
            "threshold": worst["worst_subgroup_threshold"],
            "status": status,
        })
        gate_rows.append({
            "gate": f"{component} worst supported subgroup {metric} lower 95% CI",
            "observed": gate_observed,
            "threshold": float(worst["worst_subgroup_threshold"]),
            "status": status,
        })
    return interval_rows, worst_rows, gate_rows


def _image_route_scores(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float | int]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["opaque_scene_id"]), str(row["view_id"]), str(row["route"]))].append(row)
    output: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for key, rows in grouped.items():
        references = [row for row in rows if row["record_kind"] != "unmatched_prediction"]
        predictions = [row for row in rows if row["record_kind"] != "missed_reference"]
        output[key] = {
            "reference_count": len(references),
            "prediction_count": len(predictions),
            "miss_count": sum(row["record_kind"] == "missed_reference" for row in rows),
            "extra_count": sum(row["record_kind"] == "unmatched_prediction" for row in rows),
            "detection_aware_mean_iou": float(
                np.mean([float(row["mask_iou"]) for row in references])
            ),
        }
    return output


def _diagnostic_selections(
    scores: Mapping[tuple[str, str, str], Mapping[str, float | int]],
) -> list[dict[str, Any]]:
    automatic = {
        (scene, view): value
        for (scene, view, route), value in scores.items()
        if route == "automatic_box"
    }
    reference = {
        (scene, view): value
        for (scene, view, route), value in scores.items()
        if route == "reference_box"
    }
    if set(automatic) != set(reference) or not automatic:
        raise ValueError("Diagnostic route coverage differs")
    median = float(
        np.median([float(row["detection_aware_mean_iou"]) for row in automatic.values()])
    )
    representative_key = min(
        automatic,
        key=lambda key: (
            abs(float(automatic[key]["detection_aware_mean_iou"]) - median),
            key,
        ),
    )
    worst_key = min(
        automatic,
        key=lambda key: (float(automatic[key]["detection_aware_mean_iou"]), key),
    )
    outlier_key = max(
        automatic,
        key=lambda key: (
            float(reference[key]["detection_aware_mean_iou"])
            - float(automatic[key]["detection_aware_mean_iou"]),
            tuple(-ord(character) for character in "|".join(key)),
        ),
    )
    return [
        {"diagnostic_kind": kind, "opaque_scene_id": key[0], "view_id": key[1]}
        for kind, key in (
            ("representative", representative_key),
            ("worst", worst_key),
            ("outlier", outlier_key),
        )
    ]


def _mask_overlay(reference: np.ndarray, prediction: np.ndarray) -> Image.Image:
    rgb = np.zeros((*reference.shape, 3), dtype=np.uint8)
    only_reference = np.logical_and(reference, np.logical_not(prediction))
    only_prediction = np.logical_and(prediction, np.logical_not(reference))
    overlap = np.logical_and(reference, prediction)
    rgb[only_reference] = (42, 190, 90)
    rgb[only_prediction] = (220, 60, 140)
    rgb[overlap] = (245, 245, 245)
    return Image.fromarray(rgb, mode="RGB")


def _write_private_diagnostic_panel(
    path: Path,
    *,
    scene_id: str,
    view_id: str,
    evaluator_root: Path,
    geometry_by_image: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    sam_row: Mapping[str, Any],
) -> None:
    reference_masks = [
        _reference_mask(evaluator_root, row)
        for _, row in sorted(geometry_by_image[(scene_id, view_id)].items())
    ]
    reference_union = np.logical_or.reduce(reference_masks)
    route_unions: dict[str, np.ndarray] = {}
    for route in ROUTES:
        masks = [decode_binary_mask(item["mask_rle"]) for item in sam_row["routes"][route]]
        route_unions[route] = (
            np.logical_or.reduce(masks) if masks else np.zeros_like(reference_union)
        )
    labels = ("supplied reference", "reference-box SAM", "automatic-box SAM")
    predictions = (reference_union, route_unions["reference_box"], route_unions["automatic_box"])
    panel = Image.new("RGB", (256 * 3, 284), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
        panel.paste(_mask_overlay(reference_union, prediction), (index * 256, 28))
        draw.text((index * 256 + 8, 8), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def run_ds1_mask_evaluation(
    *,
    evaluator_root: Path,
    proposal_output_root: Path,
    sam_output_root: Path,
    output_root: Path,
    evaluator_provenance: Mapping[str, Any],
    mask_iou_threshold: float = DEFAULT_MASK_IOU_THRESHOLD,
    boundary_tolerance_pixels: int = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 32_040,
) -> dict[str, Any]:
    """Measure frozen DS1 proposals and masks, then publish safe evidence."""

    started = time.perf_counter()
    roots = [path.expanduser().resolve() for path in (evaluator_root, proposal_output_root, sam_output_root)]
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"DS1 evaluation output already exists: {output_root}")
    if len(set(roots)) != len(roots) or any(path.is_symlink() for path in roots):
        raise ValueError("DS1 evaluator inputs must be distinct, non-symbolic roots")
    evaluator_root, proposal_output_root, sam_output_root = roots
    evaluator_record = verify_ds1_evaluator_package(evaluator_root)
    references = read_jsonl(evaluator_root / "reference_manifest.jsonl")
    attributes = read_jsonl(evaluator_root / "protected_scene_attributes.jsonl")
    crosswalk = read_jsonl(evaluator_root / "evaluator_crosswalk.jsonl")
    proposal_path = _result_path(proposal_output_root, "proposal_results.jsonl")
    sam_path = _result_path(sam_output_root, "mask_results.jsonl")
    proposal_rows = read_jsonl(proposal_path)
    sam_rows = read_jsonl(sam_path)
    if len(proposal_rows) != DS1_SCREEN_CONTRACT.image_count or len(sam_rows) != DS1_SCREEN_CONTRACT.image_count:
        raise ValueError("DS1 proposal and SAM results must cover all 2,000 images")
    upstream_identity = _verify_upstream_identity(
        evaluator_root,
        proposal_output_root,
        sam_output_root,
        proposal_path,
        sam_path,
        proposal_rows,
        sam_rows,
    )

    proposal_eval = evaluate_proposal_dataset(
        reference_rows=references,
        proposal_rows=proposal_rows,
        protected_scene_rows=attributes,
    )
    proposal_summary = summarize_proposal_evaluation(proposal_eval["image_records"])
    proposal_intervals = clustered_bootstrap_intervals(
        proposal_eval["image_records"], repetitions=bootstrap_repetitions, seed=bootstrap_seed
    )

    attributes_by_scene = {str(row["opaque_scene_id"]): row for row in attributes}
    geometry_by_image: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in references:
        if bool(row["visible"]):
            geometry_by_image[(str(row["opaque_scene_id"]), str(row["view_id"]))][str(row["opaque_object_id"])] = row
    crosswalk_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"]), str(row["route"]), str(row["prompt_key"])): row
        for row in crosswalk
    }
    reference_prompt_by_object: dict[tuple[str, str, str], str] = {}
    for row in crosswalk:
        if row["route"] != "reference_box" or row.get("opaque_object_id") is None:
            continue
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["opaque_object_id"]),
        )
        if key in reference_prompt_by_object:
            raise ValueError("DS1 evaluator crosswalk repeats a reference object")
        reference_prompt_by_object[key] = str(row["prompt_key"])
    sam_by_image = {(str(row["opaque_scene_id"]), str(row["view_id"])): row for row in sam_rows}
    proposal_by_image = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in proposal_rows
    }
    if set(sam_by_image) != set(geometry_by_image):
        raise ValueError("DS1 SAM and evaluator image coverage differs")
    if set(proposal_by_image) != set(sam_by_image):
        raise ValueError("DS1 proposal and SAM image coverage differs")

    records: list[dict[str, Any]] = []
    route_package_rows: list[dict[str, Any]] = []
    decoded_masks = 0
    consumed_crosswalk_keys: set[tuple[str, str, str, str]] = set()
    for scene_id, view_id in sorted(geometry_by_image):
        reference_masks = {
            object_id: _reference_mask(evaluator_root, row)
            for object_id, row in geometry_by_image[(scene_id, view_id)].items()
        }
        sam_row = sam_by_image[(scene_id, view_id)]
        proposal_row = proposal_by_image[(scene_id, view_id)]
        if sam_row["image_sha256"] != proposal_row["image_sha256"]:
            raise ValueError("DS1 proposal and SAM rows identify different RGB inputs")
        attributes_row = attributes_by_scene[scene_id]
        package_routes: dict[str, list[dict[str, Any]]] = {route: [] for route in ("supplied_mask", *ROUTES)}
        for route in ROUTES:
            prediction_rows = []
            for item in sam_row["routes"][route]:
                prompt_key = str(item["prompt_key"])
                link = crosswalk_by_key.get((scene_id, view_id, route, prompt_key))
                if link is None:
                    raise ValueError("DS1 SAM prompt has no frozen evaluator crosswalk")
                if str(link.get("image_sha256")) != str(sam_row["image_sha256"]):
                    raise ValueError("DS1 evaluator prompt identifies another RGB input")
                if [int(value) for value in link["prompt_box_xyxy"]] != [
                    int(value) for value in item["box_xyxy"]
                ]:
                    raise ValueError("DS1 SAM prompt box differs from its evaluator crosswalk")
                consumed_crosswalk_keys.add((scene_id, view_id, route, prompt_key))
                mask = decode_binary_mask(item["mask_rle"])
                if binary_mask_sha256(mask) != str(item["mask_sha256"]):
                    raise ValueError("DS1 SAM mask differs from its digest")
                decoded_masks += 1
                prediction_rows.append({
                    "prediction_id": prompt_key,
                    "matched_reference_id": link.get("opaque_object_id"),
                    "mask": mask,
                    "prompt_box_xyxy": item["box_xyxy"],
                })
                package_routes[route].append({
                    "prompt_key": prompt_key,
                    "peer_key": (
                        reference_prompt_by_object.get(
                            (scene_id, view_id, str(link["opaque_object_id"]))
                        )
                        if link.get("opaque_object_id") is not None
                        else None
                    ),
                    "box_xyxy": item["box_xyxy"],
                    "foreground_pixels": int(item["foreground_pixels"]),
                    "mask_status": item["mask_status"],
                    "mask_sha256": item["mask_sha256"],
                    "mask_rle": item["mask_rle"],
                    "predicted_quality_score": float(item["predicted_quality_score"]),
                })
            route_records = evaluate_mask_route_image(
                opaque_scene_id=scene_id,
                view_id=view_id,
                route=route,
                reference_masks=reference_masks,
                prediction_rows=prediction_rows,
                mask_iou_threshold=mask_iou_threshold,
                boundary_tolerance_pixels=boundary_tolerance_pixels,
            )
            for record in route_records:
                records.append({
                    **record,
                    "opaque_asset_group_id": attributes_row["opaque_asset_group_id"],
                    "category": attributes_row["category"],
                    "object_count": attributes_row["object_count"],
                    "anomaly_family": attributes_row.get("anomaly_families", []),
                })
        for object_id, mask in sorted(reference_masks.items()):
            geometry = geometry_by_image[(scene_id, view_id)][object_id]
            prompt_key = reference_prompt_by_object.get((scene_id, view_id, object_id))
            if prompt_key is None:
                raise ValueError("DS1 supplied mask has no frozen reference prompt identity")
            package_routes["supplied_mask"].append({
                "prompt_key": prompt_key,
                "peer_key": prompt_key,
                "box_xyxy": geometry["box_xyxy"],
                "foreground_pixels": int(mask.sum()),
                "mask_status": "non_empty",
                "mask_sha256": binary_mask_sha256(mask),
                "mask_rle": encode_binary_mask(mask),
            })
        route_package_rows.append({
            "schema_version": 1,
            "opaque_scene_id": scene_id,
            "view_id": view_id,
            "image_sha256": str(sam_row["image_sha256"]),
            "routes": package_routes,
        })
    if consumed_crosswalk_keys != set(crosswalk_by_key):
        raise ValueError("DS1 evaluator crosswalk contains unused or missing prompts")

    route_summary = summarize_mask_routes(records)
    mask_subgroups = []
    for dimension in ("view_id", "category", "object_count", "anomaly_family"):
        mask_subgroups.extend(_aggregate_mask_group(records, dimension))
    paired = _paired_rows(records)
    bootstrap_rows = []
    for metric in ("iou", "dice", "boundary_f1"):
        result = paired_asset_clustered_bootstrap(
            paired,
            baseline_key=f"reference_{metric}",
            candidate_key=f"automatic_{metric}",
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        bootstrap_rows.append({"metric": f"reference_minus_automatic_{metric}", **result})
    subgroup_intervals, worst_subgroups, subgroup_gate_rows = _subgroup_bootstrap_evidence(
        proposal_eval["image_records"],
        records,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    diagnostic_scores = _image_route_scores(records)
    diagnostic_selections = _diagnostic_selections(diagnostic_scores)

    staging = output_root.with_name(output_root.name + ".incomplete")
    staging.mkdir(parents=True)
    safe = staging / "safe"
    private = staging / "private"
    safe.mkdir()
    private.mkdir()
    try:
        _write_jsonl(private / "proposal_image_records.jsonl", proposal_eval["image_records"])
        _write_jsonl(private / "proposal_match_records.jsonl", proposal_eval["match_records"])
        _write_jsonl(private / "proposal_failure_records.jsonl", proposal_eval["failure_records"])
        _write_jsonl(private / "mask_evaluation_records.jsonl", records)
        write_csv_rows(safe / "proposal_summary.csv", ["metric", "value"], [{"metric": key, "value": value} for key, value in proposal_summary.items()])
        proposal_subgroups = _proposal_subgroups(proposal_eval["image_records"])
        write_csv_rows(safe / "proposal_subgroups.csv", list(proposal_subgroups[0]), proposal_subgroups)
        write_csv_rows(safe / "proposal_bootstrap_intervals.csv", list(proposal_intervals[0]), proposal_intervals)
        write_csv_rows(safe / "mask_route_summary.csv", list(route_summary[0]), route_summary)
        write_csv_rows(safe / "mask_subgroups.csv", list(mask_subgroups[0]), mask_subgroups)
        write_csv_rows(safe / "mask_bootstrap_intervals.csv", list(bootstrap_rows[0]), bootstrap_rows)
        write_csv_rows(
            safe / "subgroup_bootstrap_intervals.csv",
            list(subgroup_intervals[0]),
            subgroup_intervals,
        )
        write_csv_rows(
            safe / "worst_subgroup_summary.csv",
            list(worst_subgroups[0]),
            worst_subgroups,
        )
        diagnostic_summary_rows = []
        private_diagnostic_rows = []
        for selection in diagnostic_selections:
            key = (selection["opaque_scene_id"], selection["view_id"])
            for route in ROUTES:
                score = diagnostic_scores[(*key, route)]
                diagnostic_summary_rows.append({
                    "diagnostic_kind": selection["diagnostic_kind"],
                    "route": route,
                    **score,
                })
                private_diagnostic_rows.append({
                    **selection,
                    "route": route,
                    **score,
                })
            _write_private_diagnostic_panel(
                private / f"{selection['diagnostic_kind']}_mask_panel.png",
                scene_id=key[0],
                view_id=key[1],
                evaluator_root=evaluator_root,
                geometry_by_image=geometry_by_image,
                sam_row=sam_by_image[key],
            )
        write_csv_rows(
            safe / "diagnostic_summary.csv",
            list(diagnostic_summary_rows[0]),
            diagnostic_summary_rows,
        )
        _write_jsonl(private / "diagnostic_selection.jsonl", private_diagnostic_rows)
        failure_counts: dict[str, int] = defaultdict(int)
        for row in proposal_eval["failure_records"]:
            failure_counts[str(row["failure_type"])] += 1
        for row in sam_rows:
            if row.get("status") != "ok":
                failure_counts[f"sam_{row.get('status', 'unknown')}"] += 1
        failure_rows = [{"failure_type": key, "count": value} for key, value in sorted(failure_counts.items())]
        if not failure_rows:
            failure_rows = [{"failure_type": "none", "count": 0}]
        write_csv_rows(safe / "failure_summary.csv", ["failure_type", "count"], failure_rows)
        proposal_seconds = [float(row.get("runtime_seconds", 0.0)) for row in proposal_rows]
        sam_seconds = [float(row.get("total_runtime_seconds", 0.0)) for row in sam_rows]
        runtime = {
            "proposal_total_seconds": float(sum(proposal_seconds)),
            "proposal_median_seconds_per_image": float(np.median(proposal_seconds)),
            "proposal_p95_seconds_per_image": float(np.quantile(proposal_seconds, 0.95)),
            "sam_total_seconds": float(sum(sam_seconds)),
            "sam_median_seconds_per_image": float(np.median(sam_seconds)),
            "sam_p95_seconds_per_image": float(np.quantile(sam_seconds, 0.95)),
            "evaluator_seconds": float(time.perf_counter() - started),
        }
        _write_json(safe / "runtime_summary.json", runtime)
        route_by_name = {row["route"]: row for row in route_summary}
        gates = [
            ("proposal precision at 0.50", proposal_summary["micro_precision_at_0_50"], PROPOSAL_PRECISION_MINIMUM),
            ("proposal recall at 0.50", proposal_summary["micro_recall_at_0_50"], PROPOSAL_RECALL_MINIMUM),
            ("reference mask recall at 0.75", route_by_name["reference_box"]["detection_aware_recall_at_0_75"], REFERENCE_MASK_RECALL_MINIMUM),
            ("reference detection-aware mean mask IoU", route_by_name["reference_box"]["detection_aware_mean_iou"], REFERENCE_MASK_IOU_MINIMUM),
            ("automatic mask recall at 0.75", route_by_name["automatic_box"]["detection_aware_recall_at_0_75"], AUTOMATIC_MASK_RECALL_MINIMUM),
            ("automatic detection-aware mean mask IoU", route_by_name["automatic_box"]["detection_aware_mean_iou"], AUTOMATIC_MASK_IOU_MINIMUM),
        ]
        decision_gates = [{"gate": name, "observed": float(value), "threshold": threshold, "status": "pass" if float(value) >= threshold else "fail"} for name, value, threshold in gates]
        decision_gates.extend(subgroup_gate_rows)
        write_csv_rows(safe / "decision_gates.csv", ["gate", "observed", "threshold", "status"], decision_gates)
        proposal_passed = all(row["status"] == "pass" for row in decision_gates[:2]) and all(
            row["status"] == "pass" for row in subgroup_gate_rows if row["gate"].startswith("proposal ")
        )
        reference_passed = all(row["status"] == "pass" for row in decision_gates[2:4]) and all(
            row["status"] == "pass" for row in subgroup_gate_rows if row["gate"].startswith("reference_box ")
        )
        automatic_passed = all(row["status"] == "pass" for row in decision_gates[4:6]) and all(
            row["status"] == "pass" for row in subgroup_gate_rows if row["gate"].startswith("automatic_box ")
        )
        if proposal_passed and reference_passed and automatic_passed:
            component_decision = "proceed_with_locateanything_and_sam21"
        elif not proposal_passed:
            component_decision = "refine_or_replace_locateanything"
        elif not reference_passed:
            component_decision = "refine_or_replace_sam21"
        else:
            component_decision = "retain_sam21_and_refine_automatic_proposals"
        save_histogram([float(row["box_iou"]) for row in proposal_eval["match_records"]], safe / "proposal_matched_iou_distribution.png", title="DS1 matched proposal IoU", x_label="Box IoU", bins=20)
        for route in ROUTES:
            values = [float(row["mask_iou"]) for row in records if row["route"] == route and row["record_kind"] != "unmatched_prediction"]
            save_histogram(values, safe / f"{route}_mask_iou_distribution.png", title=f"DS1 {route.replace('_', ' ')} mask IoU", x_label="Mask IoU", bins=20)
        save_bar_chart(["precision", "recall"], [proposal_summary["micro_precision_at_0_50"], proposal_summary["micro_recall_at_0_50"]], safe / "proposal_precision_recall.png", title="DS1 proposal quality at IoU 0.50", y_label="Rate", show_values=True)
        save_bar_chart(list(ROUTES), [route_by_name[route]["detection_aware_recall_at_0_75"] for route in ROUTES], safe / "mask_recall_by_route.png", title="DS1 detection-aware mask recall", y_label="Recall", show_values=True)
        automatic_records = [row for row in records if row["route"] == "automatic_box" and row["record_kind"] == "matched_reference"]
        save_scatter_plot([float(row.get("outside_prompt_fraction") or 0.0) for row in automatic_records], [float(row["mask_iou"]) for row in automatic_records], safe / "automatic_spill_vs_mask_iou.png", title="DS1 automatic mask spill and IoU", x_label="Outside-prompt fraction", y_label="Mask IoU")
        build_ds1_mask_route_package(
            rows=route_package_rows,
            output_root=staging / "model_input",
            provenance={
                "evaluator_package_sha256": file_sha256(evaluator_root / "package_record.json"),
                "proposal_results_sha256": file_sha256(proposal_path),
                "sam_results_sha256": file_sha256(sam_path),
                "model_input_manifest_sha256": upstream_identity["model_input_manifest_sha256"],
                "prompt_package_record_sha256": upstream_identity["prompt_package_record_sha256"],
                "prompt_manifest_sha256": upstream_identity["prompt_manifest_sha256"],
                "route_link_record_sha256": file_sha256(
                    evaluator_root / "crosswalk_package_record.json"
                ),
            },
        )
        run_record = {
            "schema_version": 1,
            "stage": STAGE,
            "status": "completed",
            "tracking_status": "pending",
            "cohort": {key: getattr(DS1_SCREEN_CONTRACT, key) for key in ("cohort_id", "release_id", "role", "pilot_stage", "membership_sha256", "scene_count", "views_per_scene", "image_count", "asset_group_count")},
            "coverage": {"reference_rows": len(references), "decoded_sam_masks": decoded_masks, "proposal_images": len(proposal_rows), "sam_images": len(sam_rows)},
            "settings": {"mask_iou_threshold": mask_iou_threshold, "boundary_tolerance_pixels": boundary_tolerance_pixels, "bootstrap_repetitions": bootstrap_repetitions, "bootstrap_seed": bootstrap_seed},
            "proposal_summary": proposal_summary,
            "mask_route_summary": route_summary,
            "component_decision": {"status": component_decision, "all_gates_passed": all(row["status"] == "pass" for row in decision_gates), "final_acceptance": False, "reason": "DS1 is a development screen. The component result must be confirmed on the locked internal and official evaluation roles."},
            "inputs": {
                "evaluator_record_sha256": file_sha256(evaluator_root / "package_record.json"),
                "crosswalk_package_record_sha256": file_sha256(evaluator_root / "crosswalk_package_record.json"),
                "proposal_results_sha256": file_sha256(proposal_path),
                "sam_results_sha256": file_sha256(sam_path),
                "mask_route_record_sha256": file_sha256(staging / "model_input" / "package_record.json"),
                **upstream_identity,
            },
            "evaluator_provenance": dict(evaluator_provenance),
            "runtime": runtime,
        }
        _write_json(safe / "run_record.json", run_record)
        _artifact_manifest(private)
        _artifact_manifest(safe)
        os.replace(staging, output_root)
        return run_record
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["run_ds1_mask_evaluation"]
