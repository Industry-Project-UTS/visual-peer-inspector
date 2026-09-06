"""Protected, prediction-only inference for calibration and internal-lock cohorts.

The workflow opens a model-facing RGB package only after its prerequisite freeze
records verify. It never accepts evaluator resources. Every neural stage is
restartable from content-addressed shard receipts, and the final prediction
bundle is locked before a later evaluator notebook can consume it.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visual_peer_inspector.data.cohort_contract import CohortContract
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.frozen_resource import verify_frozen_resource
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.qualitative_reporting import (
    PredictedObject,
    QualitativeResult,
    QualitativeView,
)
from visual_peer_inspector.representation.cross_view_identity import (
    CandidateObservation,
    IdentityConfiguration,
    NormalizedGeometry,
    ViewObservation,
    match_cross_view_identities,
)
from visual_peer_inspector.review_policy import (
    ReviewPolicy,
    apply_review_policy,
    build_prediction_review_signals,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    TERMINAL_FAILURES,
    EmptyMaskFailure,
    FeatureFailure,
    FrozenPipelineIdentity,
    StageOutcome,
    ZeroProposalFailure,
    lock_predictions,
    save_development_audit_bundle,
)

SCHEMA_VERSION = 1
STAGES = ("locateanything", "sam21", "dinov2_small", "peer_scoring")
VIEW_CONTRACTS = {
    2: ("V00", "V10"),
    3: ("V00", "V06", "V13"),
    5: ("V00", "V04", "V08", "V12", "V16"),
}
FORBIDDEN_RESOURCE_TERMS = (
    "crosswalk",
    "evaluator",
    "ground-truth",
    "ground_truth",
    "label",
    "reference",
    "restricted",
)


def _digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    key = (str(record.get("opaque_scene_id", "")), str(record.get("view_id", "")))
    if not all(key):
        raise ValueError("Every inference record requires opaque_scene_id and view_id")
    return key


def _require_exact_view_contracts(value: Mapping[str, Any]) -> None:
    observed = {int(key): tuple(items) for key, items in value.items()}
    if observed != VIEW_CONTRACTS:
        raise ValueError("Frozen method record does not contain the exact 2, 3 and 5 view contracts")


def load_method_freeze(
    path: Path,
    *,
    application_contract_path: Path | None = None,
    require_resource: bool = False,
) -> dict[str, Any]:
    """Verify the Notebook 08F method lock and return its deployment settings."""

    source = Path(path).expanduser().resolve(strict=True)
    if require_resource:
        if application_contract_path is None:
            raise ValueError("The frozen method resource requires its application contract")
        contract_source = Path(application_contract_path).expanduser().resolve(strict=True)
        if contract_source.parent != source.parent:
            raise ValueError("Method and application contract must share one frozen resource")
        verify_frozen_resource(
            source.parent,
            expected_stage="cross_view_identity_method_release",
            expected_files={
                "application_output_contract.json",
                "method_freeze.json",
            },
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != "visual-peer-08f-workflow/v1":
        raise ValueError("Method freeze schema is not the Notebook 08F workflow schema")
    if value.get("status") != "frozen":
        raise ValueError("Method freeze status must be frozen")
    chosen = value.get("chosen_configuration")
    if not isinstance(chosen, Mapping):
        raise TypeError("Method freeze is missing chosen_configuration")
    if value.get("chosen_configuration_sha256") != _digest(chosen):
        raise ValueError("Chosen method configuration differs from its recorded hash")
    frozen_model_identity = value.get("frozen_model_identity_sha256")
    if not _is_sha256(frozen_model_identity):
        raise ValueError("Method freeze is missing the frozen model identity")
    application_value: Mapping[str, Any] = {}
    if application_contract_path is not None:
        contract_source = Path(application_contract_path).expanduser().resolve(strict=True)
        application_value = json.loads(contract_source.read_text(encoding="utf-8"))
        if application_value.get("schema_version") != "visual-peer-application-output/v1":
            raise ValueError("Application output contract schema is invalid")
        if application_value.get("frozen_model_identity_sha256") != frozen_model_identity:
            raise ValueError("Application contract identifies another frozen model route")
    configuration = chosen.get("identity_configuration")
    if not isinstance(configuration, Mapping):
        configuration = application_value.get("identity_configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("Method freeze is missing the cross-view identity configuration")
    identity_configuration = IdentityConfiguration(
        unmatched_cost=float(configuration["unmatched_cost"]),
        ambiguity_margin=float(configuration["ambiguity_margin"]),
        geometry_weight=float(configuration.get("geometry_weight", 0.0)),
    )
    recorded_identity_hash = application_value.get("identity_configuration_sha256")
    if recorded_identity_hash is not None and recorded_identity_hash != identity_configuration.sha256:
        raise ValueError("Application identity configuration differs from its recorded hash")
    contracts = chosen.get("view_contracts", VIEW_CONTRACTS)
    _require_exact_view_contracts(contracts)
    if str(chosen.get("score_method", chosen.get("scoring_method", "P1"))).upper() != "P1":
        raise ValueError("Protected inference requires the frozen P1 score method")
    fusion = str(chosen.get("fusion_method", "mean_normalized_rank"))
    if fusion != "mean_normalized_rank":
        raise ValueError("Protected inference requires mean_normalized_rank fusion")
    return {
        "path": source,
        "sha256": file_sha256(source),
        "identity_configuration": identity_configuration,
        "identity_configuration_sha256": identity_configuration.sha256,
        "view_contracts": VIEW_CONTRACTS,
        "fusion_method": fusion,
        "score_method": "P1",
        "frozen_model_identity_sha256": str(frozen_model_identity),
    }


def verify_frozen_base_route(
    base_identity: FrozenPipelineIdentity,
    method: Mapping[str, Any],
) -> None:
    """Stop protected inference when the mounted base route differs from 08E."""

    if base_identity.sha256 != method.get("frozen_model_identity_sha256"):
        raise ValueError(
            "Mounted prompt or model configuration differs from the frozen 08E route"
        )


def load_policy_freeze(
    path: Path, *, method_sha256: str, require_resource: bool = False
) -> dict[str, Any]:
    """Verify the Notebook 09B policy lock against the exact 08F method record."""

    source = Path(path).expanduser().resolve(strict=True)
    if require_resource:
        verify_frozen_resource(
            source.parent,
            expected_stage="calibration_review_policy_release",
            expected_files={"review_policy_freeze.json"},
        )
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != "visual-peer-09b-review-policy/v1":
        raise ValueError("Review policy schema is not the Notebook 09B policy schema")
    if value.get("status") not in {"frozen", "locked"}:
        raise ValueError("Review policy status must be frozen or locked")
    linked = value.get("method_record_sha256", value.get("method_freeze_sha256"))
    if linked != method_sha256:
        raise ValueError("Review policy was not frozen against the supplied method record")
    configuration = value.get("policy_configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("Review policy does not contain its complete policy configuration")
    policy = ReviewPolicy.from_mapping(configuration)
    if value.get("policy_configuration_sha256") != policy.sha256:
        raise ValueError("Review policy configuration differs from its recorded hash")
    for field in ("minimum_margin", "minimum_agreement", "minimum_mask_quality"):
        if float(value.get(field, -1.0)) != float(getattr(policy, field)):
            raise ValueError(f"Review policy top-level {field} differs from its configuration")
    return {
        "path": source,
        "sha256": file_sha256(source),
        "minimum_margin": policy.minimum_margin,
        "minimum_agreement": policy.minimum_agreement,
        "minimum_mask_quality": policy.minimum_mask_quality,
        "policy_configuration": asdict(policy),
        "policy_configuration_sha256": policy.sha256,
    }


def assert_evaluator_absent(
    mounted_roots: Sequence[Path], *, permitted_roots: Sequence[Path]
) -> None:
    """Reject evaluator-like paths while pruning only exact permitted subtrees."""

    permitted = tuple(Path(root).resolve() for root in permitted_roots)

    def is_permitted_subtree(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == allowed or allowed in resolved.parents for allowed in permitted)

    def reject_name(path: Path) -> None:
        if any(term in path.name.lower() for term in FORBIDDEN_RESOURCE_TERMS):
            raise ValueError(
                "Evaluator-like resource is mounted during protected inference: "
                f"{path.name}"
            )

    for root in mounted_roots:
        candidate = Path(root).resolve()
        if is_permitted_subtree(candidate):
            continue
        reject_name(candidate)
        if not candidate.is_dir():
            continue
        for current_text, directory_names, file_names in os.walk(candidate):
            current = Path(current_text).resolve()
            if is_permitted_subtree(current):
                directory_names[:] = []
                continue
            kept_directories = []
            for directory_name in directory_names:
                child = current / directory_name
                if is_permitted_subtree(child):
                    continue
                reject_name(child)
                kept_directories.append(directory_name)
            directory_names[:] = kept_directories
            for file_name in file_names:
                child = current / file_name
                if not is_permitted_subtree(child):
                    reject_name(child)


def verify_protected_input(model_input_root: Path, contract: CohortContract) -> dict[str, Any]:
    """Verify one role-specific, model-facing RGB package without opening evaluator data."""

    root = Path(model_input_root).expanduser().resolve(strict=True)
    record = verify_model_input_package(root, contract=contract)
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and any(term in path.name.lower() for term in FORBIDDEN_RESOURCE_TERMS)
    ]
    if forbidden:
        raise ValueError("Protected RGB package contains an evaluator-like file")
    return record.to_dict()


def build_protected_identity(
    base_identity: FrozenPipelineIdentity,
    *,
    contract: CohortContract,
    method_sha256: str,
    policy_sha256: str | None,
) -> FrozenPipelineIdentity:
    """Bind a frozen neural route to one protected role and its release gates."""

    if not _is_sha256(method_sha256) or (policy_sha256 is not None and not _is_sha256(policy_sha256)):
        raise ValueError("Freeze record hashes must be lowercase SHA-256 digests")
    return FrozenPipelineIdentity(
        split_sha256=(
            contract.model_membership_sha256 or contract.membership_sha256
        ),
        configuration_sha256=_digest(
            {
                "base_configuration_sha256": base_identity.configuration_sha256,
                "method_record_sha256": method_sha256,
                "policy_record_sha256": policy_sha256,
                "cohort_id": contract.cohort_id,
            }
        ),
        locateanything_sha256=base_identity.locateanything_sha256,
        sam21_sha256=base_identity.sam21_sha256,
        dinov2_sha256=base_identity.dinov2_sha256,
        preprocessing_sha256=base_identity.preprocessing_sha256,
        schema_sha256=_digest(
            {
                "name": "visual-peer-protected-prediction",
                "version": SCHEMA_VERSION,
                "stages": STAGES,
                "view_contracts": VIEW_CONTRACTS,
            }
        ),
    )


def _shard_paths(root: Path, index: int) -> tuple[Path, Path, Path, Path]:
    stem = f"shard-{index:05d}"
    return (
        root / f"{stem}.jsonl",
        root / f"{stem}.embeddings.npy",
        root / f"{stem}.embedding-index.jsonl",
        root / f"{stem}.receipt.json",
    )


def _receipt_valid(
    paths: tuple[Path, Path, Path, Path],
    *,
    stage: str,
    identity_sha256: str,
    input_sha256: str,
) -> bool:
    output_path, embeddings_path, index_path, receipt_path = paths
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "complete"
            or receipt.get("stage") != stage
            or receipt.get("identity_sha256") != identity_sha256
            or receipt.get("input_sha256") != input_sha256
            or receipt.get("output_sha256") != file_sha256(output_path)
        ):
            return False
        embeddings_hash = receipt.get("embeddings_sha256")
        index_hash = receipt.get("embedding_index_sha256")
        if embeddings_hash is None:
            return index_hash is None and not embeddings_path.exists() and not index_path.exists()
        return (
            embeddings_path.is_file()
            and index_path.is_file()
            and file_sha256(embeddings_path) == embeddings_hash
            and file_sha256(index_path) == index_hash
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_exact_sharded_stage(
    *,
    stage: str,
    input_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    identity: FrozenPipelineIdentity,
    worker: Callable[[Mapping[str, Any]], StageOutcome | Mapping[str, Any]],
    shard_size: int,
    shard_limit: int | None = None,
    maximum_new_shards: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run deterministic shards and resume only an exact identity and input hash."""

    if stage not in STAGES or shard_size < 1:
        raise ValueError("Stage or shard size is invalid")
    ordered = sorted((dict(row) for row in input_records), key=_record_key)
    keys = [_record_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Stage input contains duplicate scene-view keys")
    stage_root = Path(output_root) / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(ordered), shard_size))
    if shard_limit is not None:
        if shard_limit < 1:
            raise ValueError("shard_limit must be positive")
        starts = starts[:shard_limit]
    if maximum_new_shards is not None and maximum_new_shards < 1:
        raise ValueError("maximum_new_shards must be positive")
    processed = resumed = 0
    new_shards = 0
    failures = {name: 0 for name in TERMINAL_FAILURES}
    started = time.monotonic()
    for shard_index, start in enumerate(starts):
        shard = ordered[start : start + shard_size]
        input_hash = _digest(shard)
        paths = _shard_paths(stage_root, shard_index)
        output_path, embeddings_path, index_path, receipt_path = paths
        if _receipt_valid(
            paths,
            stage=stage,
            identity_sha256=identity.sha256,
            input_sha256=input_hash,
        ):
            resumed += len(shard)
            if progress_callback:
                progress_callback("shard_resumed", {"stage": stage, "shard": shard_index})
            continue
        if maximum_new_shards is not None and new_shards >= maximum_new_shards:
            break
        output_rows: list[dict[str, Any]] = []
        embedding_blocks: list[np.ndarray] = []
        index_rows: list[dict[str, Any]] = []
        offset = 0
        for source in shard:
            scene_id, view_id = _record_key(source)
            try:
                raw = worker(source)
                outcome = raw if isinstance(raw, StageOutcome) else StageOutcome(record=raw)
                result = dict(outcome.record)
                result.update({
                    "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "terminal_failure": result.get("terminal_failure"),
                })
                failure = result.get("terminal_failure")
                if failure is not None:
                    if failure not in failures:
                        raise ValueError("Worker returned an unsupported terminal failure")
                    failures[str(failure)] += 1
                if outcome.global_embeddings is not None:
                    block = np.asarray(outcome.global_embeddings, dtype=np.float16)
                    if block.ndim == 1:
                        block = block.reshape(1, -1)
                    if block.ndim != 2 or block.shape[1] < 1 or not np.all(np.isfinite(block)):
                        raise ValueError("Embedding block must be a finite matrix")
                    block = np.ascontiguousarray(block)
                    result.update({
                        "global_embedding_start": offset,
                        "global_embedding_count": len(block),
                        "global_embedding_dimension": int(block.shape[1]),
                        "global_embedding_sha256": sha256(block.tobytes(order="C")).hexdigest(),
                    })
                    index_rows.append({
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "start": offset,
                        "count": len(block),
                        "dimension": int(block.shape[1]),
                        "sha256": result["global_embedding_sha256"],
                    })
                    embedding_blocks.append(block)
                    offset += len(block)
            except Exception as error:  # noqa: BLE001 - retain every terminal stage failure
                if isinstance(error, ZeroProposalFailure):
                    name = "zero_proposal"
                elif isinstance(error, EmptyMaskFailure):
                    name = "empty_mask"
                elif isinstance(error, FeatureFailure):
                    name = "feature_failure"
                else:
                    name = "stage_failure"
                failures[name] += 1
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "cohort_id": source.get("cohort_id"),
                    "image_path": source.get("image_path"),
                    "image_sha256": source.get("image_sha256"),
                    "status": "terminal_failure",
                    "terminal_failure": name,
                    "error_type": type(error).__name__,
                }
            output_rows.append(result)
            processed += 1
        _write_jsonl(output_path, output_rows)
        embeddings_hash = index_hash = None
        if embedding_blocks:
            combined = np.concatenate(embedding_blocks, axis=0).astype(np.float16)
            temporary = embeddings_path.with_suffix(".npy.incomplete")
            with temporary.open("wb") as handle:
                np.save(handle, combined, allow_pickle=False)
            temporary.replace(embeddings_path)
            _write_jsonl(index_path, index_rows)
            embeddings_hash = file_sha256(embeddings_path)
            index_hash = file_sha256(index_path)
        else:
            embeddings_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
        _write_json(receipt_path, {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "stage": stage,
            "shard_index": shard_index,
            "input_count": len(shard),
            "input_sha256": input_hash,
            "output_count": len(output_rows),
            "output_sha256": file_sha256(output_path),
            "embeddings_sha256": embeddings_hash,
            "embedding_index_sha256": index_hash,
            "identity_sha256": identity.sha256,
        })
        if progress_callback:
            progress_callback("shard_complete", {"stage": stage, "shard": shard_index})
        new_shards += 1
    covered_rows = read_stage(stage_root) if list(stage_root.glob("shard-*.receipt.json")) else []
    failures = {name: 0 for name in TERMINAL_FAILURES}
    for row in covered_rows:
        failure_name = row.get("terminal_failure")
        if failure_name is None:
            continue
        if failure_name not in failures:
            raise ValueError("Verified stage contains an unsupported terminal failure")
        failures[str(failure_name)] += 1
    return {
        "stage": stage,
        "available_input_count": len(ordered),
        "covered_input_count": len(covered_rows),
        "processed_this_run": processed,
        "resumed": resumed,
        "terminal_failures": int(sum(failures.values())),
        "failure_counts": failures,
        "completed_shards": len(list(stage_root.glob("shard-*.receipt.json"))),
        "duration_seconds": round(time.monotonic() - started, 6),
        "identity_sha256": identity.sha256,
    }


def read_stage(stage_root: Path) -> list[dict[str, Any]]:
    """Read completed shards after verifying their declared output digests."""

    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(Path(stage_root).glob("shard-*.receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        output_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".jsonl"))
        if receipt.get("status") != "complete" or file_sha256(output_path) != receipt.get("output_sha256"):
            raise ValueError(f"Invalid shard receipt: {receipt_path.name}")
        rows.extend(_read_jsonl(output_path))
    keys = [_record_key(row) for row in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Completed stage is empty or contains duplicate scene-view keys")
    return sorted(rows, key=_record_key)


def read_stage_embeddings(stage_root: Path) -> tuple[np.ndarray, dict[tuple[str, str], dict[str, Any]]]:
    """Load verified embedding shards and a scene-view block index."""

    blocks: list[np.ndarray] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}
    offset = 0
    for receipt_path in sorted(Path(stage_root).glob("shard-*.receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("embeddings_sha256") is None:
            continue
        embeddings_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".embeddings.npy"))
        index_path = receipt_path.with_name(receipt_path.name.replace(".receipt.json", ".embedding-index.jsonl"))
        if file_sha256(embeddings_path) != receipt["embeddings_sha256"] or file_sha256(index_path) != receipt["embedding_index_sha256"]:
            raise ValueError("Embedding shard differs from its receipt")
        block = np.load(embeddings_path, allow_pickle=False)
        if block.dtype != np.float16 or block.ndim != 2:
            raise ValueError("Embedding shard is not a float16 matrix")
        for row in _read_jsonl(index_path):
            key = (str(row["opaque_scene_id"]), str(row["view_id"]))
            if key in index:
                raise ValueError("Embedding index contains duplicate scene-view keys")
            index[key] = {**row, "start": offset + int(row["start"])}
        blocks.append(block)
        offset += len(block)
    if not blocks:
        return np.empty((0, 0), dtype=np.float16), index
    return np.concatenate(blocks, axis=0), index


def _normalised_rank(rank: Any, count: int) -> float | None:
    if rank is None or count < 3:
        return None
    value = int(rank)
    if not 1 <= value <= count:
        raise ValueError("P1 rank is outside its candidate count")
    return float((count - value) / (count - 1))


def _geometry(item: Mapping[str, Any], image_size: tuple[int, int]) -> NormalizedGeometry:
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in item["box_xyxy"])
    x1, y1 = max(0.0, x1 / width), max(0.0, y1 / height)
    x2, y2 = min(1.0, x2 / width), min(1.0, y2 / height)
    mask = decode_binary_mask(item["mask_rle"])
    coordinates = np.argwhere(mask)
    centroid = None
    if len(coordinates):
        centroid = (float(coordinates[:, 1].mean() / width), float(coordinates[:, 0].mean() / height))
    return NormalizedGeometry(
        box_xyxy_normalized=(x1, y1, x2, y2),
        mask_centroid_xy_normalized=centroid,
        mask_area_fraction=float(mask.mean()),
    )


def build_multiview_predictions(
    score_rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    embedding_index: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    identity_configuration: IdentityConfiguration,
    policy: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Associate objects and fuse P1 ranks for the exact 2, 3 and 5 view contracts."""

    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    for source in score_rows:
        row = dict(source)
        scene_id, view_id = _record_key(row)
        by_scene.setdefault(scene_id, {})[view_id] = row
    deployed_policy = None
    if policy is not None:
        policy_configuration = policy.get("policy_configuration")
        if not isinstance(policy_configuration, Mapping):
            raise TypeError("Frozen review policy is incomplete")
        deployed_policy = ReviewPolicy.from_mapping(policy_configuration)
    results: list[dict[str, Any]] = []
    for scene_id in sorted(by_scene):
        scene_rows = by_scene[scene_id]
        for view_count, view_ids in VIEW_CONTRACTS.items():
            observations = []
            local: dict[tuple[str, str], dict[str, Any]] = {}
            view_failures = []
            selected_view_rows = [scene_rows.get(view_id) for view_id in view_ids]
            for view_id in view_ids:
                row = scene_rows.get(view_id)
                candidates = []
                if row is None or row.get("terminal_failure") is not None:
                    view_failures.append({
                        "view_id": view_id,
                        "terminal_failure": "missing_prediction" if row is None else row.get("terminal_failure"),
                    })
                else:
                    objects = list(row.get("objects", []))
                    block = embedding_index.get((scene_id, view_id))
                    if objects and (block is None or int(block["count"]) != len(objects)):
                        raise ValueError("Object count differs from the embedding block index")
                    vectors = np.empty((0, 0), dtype=np.float16) if not objects else embeddings[int(block["start"]):int(block["start"]) + len(objects)]
                    with Image.open(str(row["image_path"])) as opened:
                        image_size = opened.size
                    for item, vector in zip(objects, vectors, strict=True):
                        object_id = str(item["object_id"])
                        candidates.append(CandidateObservation(
                            candidate_id=object_id,
                            embedding=tuple(float(value) for value in vector),
                            geometry=_geometry(item, image_size),
                        ))
                        local[(view_id, object_id)] = dict(item)
                observations.append(ViewObservation(view_id=view_id, candidates=tuple(candidates)))
            identity_result = match_cross_view_identities(observations, configuration=identity_configuration)
            fused_tracks = []
            for track in identity_result.tracks:
                values = []
                rank_one = 0
                for member in track.members:
                    item = local[(member.view_id, member.candidate_id)]
                    rank = _normalised_rank(item.get("p1_rank"), len(scene_rows[member.view_id].get("objects", [])))
                    if rank is not None:
                        values.append(rank)
                        rank_one += int(item.get("p1_rank") == 1)
                fused_tracks.append({
                    "identity_id": track.identity_id,
                    "members": [member.to_record() for member in track.members],
                    "identity_status": track.status,
                    "identity_minimum_margin": track.minimum_margin,
                    "cycle_consistent": track.cycle_consistent,
                    "scored_view_count": len(values),
                    "fused_unusual_score": None if not values else float(np.mean(values)),
                    "rank_one_agreement": None if not values else float(rank_one / len(values)),
                })
            scored = sorted(
                (item for item in fused_tracks if item["fused_unusual_score"] is not None),
                key=lambda item: (-float(item["fused_unusual_score"]), str(item["identity_id"])),
            )
            winner = None if not scored else scored[0]
            margin = None if len(scored) < 2 else float(scored[0]["fused_unusual_score"] - scored[1]["fused_unusual_score"])
            agreement = None if winner is None else winner["rank_one_agreement"]
            insufficient = winner is None or int(winner["scored_view_count"]) < 2 or bool(view_failures)
            result = {
                "schema_version": SCHEMA_VERSION,
                "opaque_scene_id": scene_id,
                "view_count": view_count,
                "view_ids": list(view_ids),
                "identity_result": identity_result.to_record(),
                "tracks": fused_tracks,
                "fused_odd_one_out": None if winner is None else winner["identity_id"],
                "fused_unusual_score": None if winner is None else winner["fused_unusual_score"],
                "margin": margin,
                "agreement": agreement,
                "insufficient_evidence": insufficient,
                "view_failures": view_failures,
                "prediction_only": True,
            }
            signals = build_prediction_review_signals(result, selected_view_rows)
            result.update(signals)
            if deployed_policy is None:
                policy_state = (
                    "insufficient_evidence"
                    if signals["insufficient_evidence"]
                    else "policy_pending"
                )
                review_reasons = (
                    ["insufficient_evidence"]
                    if signals["insufficient_evidence"]
                    else []
                )
                review_signal = policy_state
            else:
                policy_state, reasons = apply_review_policy(
                    result,
                    deployed_policy,
                )
                review_reasons = list(reasons)
                review_signal = {
                    "result_ready": "automatic",
                    "review_required": "review",
                    "insufficient_evidence": "insufficient_evidence",
                }[policy_state]
            result.update({
                "policy_state": policy_state,
                "review_reasons": review_reasons,
                "review_signal": review_signal,
            })
            results.append(result)
    return results


def publish_protected_bundle(
    *,
    cohort_root: Path,
    identity: FrozenPipelineIdentity,
    contract: CohortContract,
    method: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write and lock prediction-only manifests, embeddings and multi-view decisions."""

    root = Path(cohort_root).resolve()
    score_rows = read_stage(root / "peer_scoring")
    representation_rows = read_stage(root / "dinov2_small")
    embeddings, block_index = read_stage_embeddings(root / "dinov2_small")
    representation_by_key = {_record_key(row): row for row in representation_rows}
    if {_record_key(row) for row in score_rows} != set(representation_by_key):
        raise ValueError("Scoring and representation coverage differs")
    manifest_rows = []
    embedding_rows = []
    published_blocks = []
    output_offset = 0
    for row in score_rows:
        key = _record_key(row)
        objects = list(row.get("objects", []))
        block = block_index.get(key)
        vectors = np.empty((0, 0), dtype=np.float16)
        if objects:
            if block is None or int(block["count"]) != len(objects):
                raise ValueError("Object count differs from the embedding index")
            vectors = embeddings[int(block["start"]):int(block["start"]) + len(objects)]
            published_blocks.append(vectors)
        public_objects = []
        for offset, (item, vector) in enumerate(zip(objects, vectors, strict=True)):
            public_objects.append({
                "object_id": str(item["object_id"]),
                "box_xyxy": [int(value) for value in item["box_xyxy"]],
                "mask_rle": item["mask_rle"],
                "mask_sha256": str(item["mask_sha256"]),
                "foreground_pixels": int(item["foreground_pixels"]),
                "predicted_quality_score": float(item["predicted_quality_score"]),
                "p1_status": str(item["p1_status"]),
                "p1_score": item.get("p1_score"),
                "p1_rank": item.get("p1_rank"),
                "p1_heatmap": item.get("p1_heatmap"),
            })
            embedding_rows.append({
                "opaque_scene_id": key[0],
                "view_id": key[1],
                "object_id": str(item["object_id"]),
                "row_index": output_offset + offset,
                "feature_dimension": len(vector),
                "vector_sha256": sha256(np.ascontiguousarray(vector).tobytes(order="C")).hexdigest(),
            })
        output_offset += len(objects)
        manifest_rows.append({
            "schema_version": SCHEMA_VERSION,
            "cohort_id": contract.cohort_id,
            "opaque_scene_id": key[0],
            "view_id": key[1],
            "image_sha256": str(row.get("image_sha256", "")),
            "status": str(row.get("status", "terminal_failure")),
            "terminal_failure": row.get("terminal_failure"),
            "object_count": len(public_objects),
            "objects": public_objects,
            "per_view_p1_candidate": row.get("per_view_p1_candidate"),
            "per_view_p1_score": row.get("per_view_p1_score"),
        })
    if len(manifest_rows) != contract.image_count:
        raise ValueError("Full protected prediction coverage is required before locking")
    multiview = build_multiview_predictions(
        score_rows,
        embeddings,
        block_index,
        identity_configuration=method["identity_configuration"],
        policy=policy,
    )
    model_safe = root / "model-safe"
    model_safe.mkdir(parents=True, exist_ok=True)
    manifest_path = model_safe / "prediction_manifest.jsonl"
    embedding_path = model_safe / "global_embeddings.npy"
    embedding_index_path = model_safe / "embedding_index.jsonl"
    multiview_path = model_safe / "multiview_prediction_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(embedding_index_path, embedding_rows)
    _write_jsonl(multiview_path, multiview)
    combined = np.concatenate(published_blocks, axis=0).astype(np.float16) if published_blocks else np.empty((0, 0), dtype=np.float16)
    temporary = embedding_path.with_suffix(".npy.incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, combined, allow_pickle=False)
    temporary.replace(embedding_path)
    lock = lock_predictions(
        prediction_root=model_safe,
        identity=identity,
        cohort_id=contract.cohort_id,
        cohort_membership_sha256=(
            contract.model_membership_sha256 or contract.membership_sha256
        ),
        scene_count=contract.scene_count,
        image_count=contract.image_count,
    )
    return {
        "prediction_manifest_sha256": file_sha256(manifest_path),
        "embedding_sha256": file_sha256(embedding_path),
        "embedding_index_sha256": file_sha256(embedding_index_path),
        "multiview_prediction_manifest_sha256": file_sha256(multiview_path),
        "prediction_lock_sha256": file_sha256(model_safe / "prediction_lock.json"),
        "prediction_count": len(manifest_rows),
        "multiview_result_count": len(multiview),
        "embedding_count": len(combined),
        "lock": lock,
    }


def render_prediction_audit(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    multiview_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    count: int = 12,
) -> list[dict[str, Any]]:
    """Render deterministic five-view panels containing predictions only."""

    score_by_key = {_record_key(row): row for row in score_rows}
    selected = select_prediction_audit_rows(
        score_rows=score_rows,
        multiview_rows=multiview_rows,
        count=count,
    )
    evidence = []
    for decision in selected:
        scene_id = str(decision["opaque_scene_id"])
        member_identity = {
            (str(member["view_id"]), str(member["candidate_id"])): str(track["identity_id"])
            for track in decision["tracks"]
            for member in track["members"]
        }
        views = []
        for view_id in decision["view_ids"]:
            row = score_by_key.get((scene_id, str(view_id)))
            if row is None or row.get("terminal_failure") is not None:
                views.append(QualitativeView(
                    view_id=str(view_id),
                    rgb=None,
                    missing_reason="prediction unavailable",
                ))
                continue
            predicted = []
            heatmap = None
            for item in row.get("objects", []):
                identity_id = member_identity.get((str(view_id), str(item["object_id"])), str(item["object_id"]))
                predicted.append(PredictedObject(
                    object_id=identity_id,
                    box_xyxy=tuple(float(value) for value in item["box_xyxy"]),
                    sam_mask=decode_binary_mask(item["mask_rle"]),
                    p1_score=item.get("p1_score"),
                    p1_rank=item.get("p1_rank"),
                ))
                if identity_id == decision.get("fused_odd_one_out"):
                    heatmap = item.get("p1_heatmap")
            views.append(QualitativeView(
                view_id=str(view_id),
                rgb=str(row["image_path"]),
                objects=tuple(predicted),
                heatmap=None if heatmap is None else np.asarray(heatmap, dtype=np.float32),
            ))
        result = QualitativeResult(
            result_id=f"panel-{len(evidence) + 1:03d}",
            views=tuple(views),
            fused_odd_one_out=decision.get("fused_odd_one_out"),
            agreement=decision.get("agreement"),
            margin=decision.get("margin"),
            review_state=str(decision["review_signal"]),
            publication_metadata={"route": "frozen_automatic", "view_count": 5},
        )
        files = save_development_audit_bundle(result, Path(output_root))
        cohort_root = Path(output_root).resolve().parent.parent
        evidence.append(
            {
                "result_number": len(evidence) + 1,
                "prediction_panel": Path(files["prediction_panel"])
                .resolve()
                .relative_to(cohort_root)
                .as_posix(),
                "prediction_sha256": files["prediction_sha256"],
                "sam_evidence_panel": Path(files["sam_evidence_panel"])
                .resolve()
                .relative_to(cohort_root)
                .as_posix(),
                "sam_evidence_sha256": files["sam_evidence_sha256"],
                "selection_reason": decision["audit_selection_reason"],
                "selection_strata": decision["audit_selection_strata"],
            }
        )
    return evidence


def _prediction_audit_strata(
    decision: Mapping[str, Any],
    score_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return prediction-only strata for one qualitative candidate."""

    strata = []
    policy_state = str(decision.get("policy_state", "")).strip()
    review_signal = str(decision.get("review_signal", "")).strip()
    if policy_state:
        strata.append(f"policy_state:{policy_state}")
    if review_signal:
        strata.append(f"review_signal:{review_signal}")
    scene_id = str(decision["opaque_scene_id"])
    failures = []
    for view_id in decision["view_ids"]:
        row = score_by_key.get((scene_id, str(view_id)))
        failure = "missing_prediction" if row is None else row.get("terminal_failure")
        if failure is not None:
            failures.append(str(failure))
    if failures and len(failures) == len(decision["view_ids"]):
        strata.append("failure_scope:full_scene")
    strata.extend(f"terminal_failure:{name}" for name in sorted(set(failures)))
    return tuple(strata)


def select_prediction_audit_rows(
    *,
    score_rows: Sequence[Mapping[str, Any]],
    multiview_rows: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Select label-free qualitative rows by state and failure before hash fill."""

    if count < 1:
        return []
    score_by_key = {_record_key(row): row for row in score_rows}
    candidates = [
        dict(row) for row in multiview_rows if int(row["view_count"]) == 5
    ]
    candidates.sort(
        key=lambda row: (
            _digest(
                {
                    "namespace": "protected-audit-v2",
                    "scene": row["opaque_scene_id"],
                }
            ),
            str(row["opaque_scene_id"]),
        )
    )
    strata_by_scene = {
        str(row["opaque_scene_id"]): _prediction_audit_strata(row, score_by_key)
        for row in candidates
    }
    available = {stratum for values in strata_by_scene.values() for stratum in values}
    preferred = [
        "policy_state:result_ready",
        "policy_state:review_required",
        "policy_state:insufficient_evidence",
        "policy_state:policy_pending",
        "review_signal:automatic",
        "review_signal:review",
        "review_signal:identity_review",
        "review_signal:insufficient_evidence",
        "review_signal:policy_pending",
        "failure_scope:full_scene",
        *(f"terminal_failure:{name}" for name in TERMINAL_FAILURES),
        "terminal_failure:missing_prediction",
    ]
    ordered_strata = [value for value in preferred if value in available]
    ordered_strata.extend(sorted(available.difference(ordered_strata)))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()
    for stratum in ordered_strata:
        if stratum in covered or len(selected) >= count:
            continue
        candidate = next(
            (
                row
                for row in candidates
                if str(row["opaque_scene_id"]) not in selected_ids
                and stratum in strata_by_scene[str(row["opaque_scene_id"])]
            ),
            None,
        )
        if candidate is None:
            continue
        scene_id = str(candidate["opaque_scene_id"])
        candidate["audit_selection_reason"] = stratum
        candidate["audit_selection_strata"] = list(strata_by_scene[scene_id])
        selected.append(candidate)
        selected_ids.add(scene_id)
        covered.update(strata_by_scene[scene_id])
    for candidate in candidates:
        if len(selected) >= count:
            break
        scene_id = str(candidate["opaque_scene_id"])
        if scene_id in selected_ids:
            continue
        candidate["audit_selection_reason"] = "stable_hash_fill"
        candidate["audit_selection_strata"] = list(strata_by_scene[scene_id])
        selected.append(candidate)
        selected_ids.add(scene_id)
    return selected


def safe_aggregate_tables(multiview_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return identifier-free view and review summaries for reporting and ClearML."""

    contract_rows = []
    review_rows = []
    for view_count in sorted(VIEW_CONTRACTS):
        rows = [row for row in multiview_rows if int(row["view_count"]) == view_count]
        insufficient_count = sum(bool(row["insufficient_evidence"]) for row in rows)
        result_available_count = len(rows) - insufficient_count
        result_ready_count = sum(
            row.get("policy_state") == "result_ready" for row in rows
        )
        review_required_count = sum(
            row.get("policy_state") == "review_required" for row in rows
        )
        policy_pending_count = sum(
            row.get("policy_state") == "policy_pending" for row in rows
        )
        if result_available_count != (
            result_ready_count + review_required_count + policy_pending_count
        ):
            raise ValueError("Available-result policy states do not sum to the scene count")
        contract_rows.append({
            "view_count": view_count,
            "scene_count": len(rows),
            "result_available_count": result_available_count,
            "result_ready_count": result_ready_count,
            "review_required_count": review_required_count,
            "policy_pending_count": policy_pending_count,
            "insufficient_evidence_count": insufficient_count,
            "mean_margin": None if not any(row.get("margin") is not None for row in rows) else float(np.mean([row["margin"] for row in rows if row.get("margin") is not None])),
            "mean_agreement": None if not any(row.get("agreement") is not None for row in rows) else float(np.mean([row["agreement"] for row in rows if row.get("agreement") is not None])),
        })
        for signal in ("automatic", "review", "identity_review", "insufficient_evidence", "policy_pending"):
            review_rows.append({
                "view_count": view_count,
                "review_signal": signal,
                "scene_count": sum(row["review_signal"] == signal for row in rows),
            })
    return contract_rows, review_rows


__all__ = [
    "STAGES",
    "VIEW_CONTRACTS",
    "assert_evaluator_absent",
    "build_multiview_predictions",
    "build_protected_identity",
    "load_method_freeze",
    "load_policy_freeze",
    "publish_protected_bundle",
    "read_stage",
    "read_stage_embeddings",
    "render_prediction_audit",
    "run_exact_sharded_stage",
    "safe_aggregate_tables",
    "select_prediction_audit_rows",
    "verify_frozen_base_route",
    "verify_protected_input",
]
