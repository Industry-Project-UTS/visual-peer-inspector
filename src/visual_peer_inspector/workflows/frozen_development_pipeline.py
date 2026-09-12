"""Frozen, resumable development inference and evidence contracts.

The module keeps model-facing predictions separate from development references.
Every shard is content-addressed, and evaluator access is permitted only after a
prediction bundle has been locked.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
)
from visual_peer_inspector.qualitative_reporting import (
    QualitativeResult,
    render_prediction_panel,
    save_panel_png,
)
from visual_peer_inspector.workflows.runtime_failures import (
    raise_if_infrastructure_failure,
)

SCHEMA_VERSION = 1
STAGE_ORDER = ("locateanything", "sam21", "dinov2_small", "peer_scoring")
TERMINAL_FAILURES = (
    "zero_proposal",
    "empty_mask",
    "feature_failure",
    "stage_failure",
)
DEFAULT_AUDIT_COUNT = 15


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


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
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


@dataclass(frozen=True)
class FrozenPipelineIdentity:
    """Complete identity of one frozen prediction route."""

    split_sha256: str
    configuration_sha256: str
    locateanything_sha256: str
    sam21_sha256: str
    dinov2_sha256: str
    preprocessing_sha256: str
    schema_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def sha256(self) -> str:
        return _json_sha256(asdict(self))


@dataclass(frozen=True)
class StageOutcome:
    """One serialisable stage result and optional global embeddings."""

    record: Mapping[str, Any]
    global_embeddings: np.ndarray | None = None


class ZeroProposalFailure(RuntimeError):
    """LocateAnything returned no usable object proposal."""


class EmptyMaskFailure(RuntimeError):
    """SAM returned no usable foreground for an object proposal."""


class FeatureFailure(RuntimeError):
    """The frozen visual encoder did not return valid features."""


_FAILURE_BY_EXCEPTION = {
    ZeroProposalFailure: "zero_proposal",
    EmptyMaskFailure: "empty_mask",
    FeatureFailure: "feature_failure",
}


def _failure_name(error: Exception) -> str:
    for exception_type, name in _FAILURE_BY_EXCEPTION.items():
        if isinstance(error, exception_type):
            return name
    return "stage_failure"


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    scene_id = str(record.get("opaque_scene_id", "")).strip()
    view_id = str(record.get("view_id", "")).strip()
    if not scene_id or not view_id:
        raise ValueError("Every stage record requires opaque_scene_id and view_id")
    return scene_id, view_id


def _receipt_paths(stage_root: Path, shard_index: int) -> tuple[Path, Path, Path, Path]:
    name = f"shard-{shard_index:05d}"
    return (
        stage_root / f"{name}.jsonl",
        stage_root / f"{name}.embeddings.npy",
        stage_root / f"{name}.embedding-index.jsonl",
        stage_root / f"{name}.receipt.json",
    )


def _valid_receipt(
    receipt_path: Path,
    output_path: Path,
    embeddings_path: Path,
    embedding_index_path: Path,
    *,
    stage: str,
    identity: FrozenPipelineIdentity,
    input_sha256: str,
) -> bool:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "complete"
            or receipt.get("stage") != stage
            or receipt.get("identity_sha256") != identity.sha256
            or receipt.get("input_sha256") != input_sha256
            or receipt.get("output_sha256") != file_sha256(output_path)
        ):
            return False
        expected_embeddings = receipt.get("embeddings_sha256")
        expected_index = receipt.get("embedding_index_sha256")
        if expected_embeddings is None and expected_index is None:
            return not embeddings_path.exists() and not embedding_index_path.exists()
        return (
            embeddings_path.is_file()
            and embedding_index_path.is_file()
            and file_sha256(embeddings_path) == expected_embeddings
            and file_sha256(embedding_index_path) == expected_index
            and np.load(embeddings_path, allow_pickle=False).dtype == np.float16
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_sharded_stage(
    *,
    stage: str,
    input_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    identity: FrozenPipelineIdentity,
    worker: Callable[[Mapping[str, Any]], StageOutcome | Mapping[str, Any]],
    shard_size: int,
    maximum_new_shards: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one deterministic stage with content-addressed shard receipts."""

    if stage not in STAGE_ORDER:
        raise ValueError(f"Unsupported frozen stage: {stage}")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if maximum_new_shards is not None and maximum_new_shards < 1:
        raise ValueError("maximum_new_shards must be positive")
    ordered = sorted((dict(row) for row in input_records), key=_record_key)
    keys = [_record_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("Stage input contains duplicate scene-view keys")
    stage_root = Path(output_root) / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    processed = resumed = failed = new_shards = 0
    failure_counts = {name: 0 for name in TERMINAL_FAILURES}
    shard_count = (len(ordered) + shard_size - 1) // shard_size

    for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
        shard = ordered[start : start + shard_size]
        input_digest = _json_sha256(shard)
        output_path, embeddings_path, index_path, receipt_path = _receipt_paths(
            stage_root, shard_index
        )
        if _valid_receipt(
            receipt_path,
            output_path,
            embeddings_path,
            index_path,
            stage=stage,
            identity=identity,
            input_sha256=input_digest,
        ):
            resumed += len(shard)
            if progress_callback is not None:
                progress_callback(
                    "shard_resumed", {"stage": stage, "shard": shard_index}
                )
            continue
        if maximum_new_shards is not None and new_shards >= maximum_new_shards:
            break

        output_rows: list[dict[str, Any]] = []
        embedding_blocks: list[np.ndarray] = []
        embedding_index: list[dict[str, Any]] = []
        embedding_offset = 0
        for input_row in shard:
            scene_id, view_id = _record_key(input_row)
            try:
                raw_outcome = worker(input_row)
                outcome = (
                    raw_outcome
                    if isinstance(raw_outcome, StageOutcome)
                    else StageOutcome(record=raw_outcome)
                )
                result = dict(outcome.record)
                result.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stage": stage,
                        "opaque_scene_id": scene_id,
                        "view_id": view_id,
                        "terminal_failure": result.get("terminal_failure"),
                    }
                )
                failure_name = result.get("terminal_failure")
                if failure_name is not None:
                    if failure_name not in TERMINAL_FAILURES:
                        raise ValueError(
                            "Stage returned an unsupported terminal failure"
                        )
                    failure_counts[str(failure_name)] += 1
                    failed += 1
                if outcome.global_embeddings is not None:
                    block = np.asarray(outcome.global_embeddings)
                    if block.ndim == 1:
                        block = block.reshape(1, -1)
                    if (
                        block.ndim != 2
                        or block.shape[1] < 1
                        or not np.all(np.isfinite(block))
                    ):
                        raise FeatureFailure(
                            "Global embeddings are not a finite matrix"
                        )
                    block16 = np.ascontiguousarray(block.astype(np.float16))
                    block_digest = sha256(block16.tobytes(order="C")).hexdigest()
                    result["global_embedding_start"] = embedding_offset
                    result["global_embedding_count"] = int(block16.shape[0])
                    result["global_embedding_dimension"] = int(block16.shape[1])
                    result["global_embedding_sha256"] = block_digest
                    embedding_index.append(
                        {
                            "opaque_scene_id": scene_id,
                            "view_id": view_id,
                            "start": embedding_offset,
                            "count": int(block16.shape[0]),
                            "dimension": int(block16.shape[1]),
                            "sha256": block_digest,
                        }
                    )
                    embedding_blocks.append(block16)
                    embedding_offset += int(block16.shape[0])
            except Exception as error:  # noqa: BLE001 - retain every terminal stage failure
                raise_if_infrastructure_failure(error)
                failure_name = _failure_name(error)
                failure_counts[failure_name] += 1
                failed += 1
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "cohort_id": input_row.get("cohort_id"),
                    "image_path": input_row.get("image_path"),
                    "image_sha256": input_row.get("image_sha256"),
                    "status": "terminal_failure",
                    "terminal_failure": failure_name,
                    "error_type": type(error).__name__,
                }
            output_rows.append(result)
            processed += 1

        _write_jsonl(output_path, output_rows)
        embeddings_digest = index_digest = None
        if embedding_blocks:
            embeddings = np.concatenate(embedding_blocks, axis=0).astype(np.float16)
            temporary = embeddings_path.with_suffix(".npy.incomplete")
            with temporary.open("wb") as handle:
                np.save(handle, embeddings, allow_pickle=False)
            temporary.replace(embeddings_path)
            _write_jsonl(index_path, embedding_index)
            embeddings_digest = file_sha256(embeddings_path)
            index_digest = file_sha256(index_path)
        else:
            embeddings_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
        _write_json(
            receipt_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "stage": stage,
                "shard_index": shard_index,
                "input_count": len(shard),
                "input_sha256": input_digest,
                "output_count": len(output_rows),
                "output_sha256": file_sha256(output_path),
                "embeddings_sha256": embeddings_digest,
                "embedding_index_sha256": index_digest,
                "identity_sha256": identity.sha256,
            },
        )
        if progress_callback is not None:
            progress_callback(
                "shard_complete",
                {"stage": stage, "shard": shard_index, "shards": shard_count},
            )
        new_shards += 1
    completed_shards = 0
    covered_input_count = 0
    verified_rows: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
        shard = ordered[start : start + shard_size]
        paths = _receipt_paths(stage_root, shard_index)
        if _valid_receipt(
            paths[3],
            paths[0],
            paths[1],
            paths[2],
            stage=stage,
            identity=identity,
            input_sha256=_json_sha256(shard),
        ):
            completed_shards += 1
            covered_input_count += len(shard)
            verified_rows.extend(_read_jsonl(paths[0]))
    failure_counts = {name: 0 for name in TERMINAL_FAILURES}
    for row in verified_rows:
        failure_name = row.get("terminal_failure")
        if failure_name is None:
            continue
        if failure_name not in failure_counts:
            raise ValueError("Verified stage contains an unsupported terminal failure")
        failure_counts[str(failure_name)] += 1
    failed = sum(failure_counts.values())
    return {
        "stage": stage,
        "input_count": len(ordered),
        "processed_this_run": processed,
        "resumed": resumed,
        "terminal_failures": failed,
        "failure_counts": failure_counts,
        "shard_count": shard_count,
        "completed_shards": completed_shards,
        "covered_input_count": covered_input_count,
        "complete": completed_shards == shard_count,
        "identity_sha256": identity.sha256,
    }


def read_completed_stage(stage_root: Path) -> list[dict[str, Any]]:
    """Read only shards whose receipts and content hashes still verify."""

    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(Path(stage_root).glob("shard-*.receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        output_path = receipt_path.with_name(
            receipt_path.name.replace(".receipt.json", ".jsonl")
        )
        if (
            receipt.get("status") != "complete"
            or not output_path.is_file()
            or file_sha256(output_path) != receipt.get("output_sha256")
        ):
            raise ValueError(f"Invalid completed shard: {receipt_path.name}")
        rows.extend(_read_jsonl(output_path))
    if not rows:
        raise ValueError("No completed shards were found")
    return sorted(rows, key=_record_key)


def validate_recovery_stage_chain(
    cohort_root: Path,
    input_records: Sequence[Mapping[str, Any]],
    identity: FrozenPipelineIdentity,
    shard_size: int,
) -> int:
    """Reject incompatible saved inputs before loading any model weights."""

    ordered = sorted((dict(row) for row in input_records), key=_record_key)
    checked = 0
    for stage in STAGE_ORDER:
        stage_root = Path(cohort_root) / stage
        receipts = sorted(stage_root.glob("shard-*.receipt.json"))
        if not receipts:
            ordered = []
            continue
        for path in receipts:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            index = int(receipt["shard_index"])
            shard = ordered[index * shard_size : (index + 1) * shard_size]
            paths = _receipt_paths(stage_root, index)
            if (
                not shard
                or path != paths[3]
                or not _valid_receipt(
                    path,
                    paths[0],
                    paths[1],
                    paths[2],
                    stage=stage,
                    identity=identity,
                    input_sha256=_json_sha256(shard),
                )
            ):
                raise ValueError(
                    f"Saved {stage} shard {index} does not match current inputs"
                )
            checked += 1
        ordered = read_completed_stage(stage_root)
    return checked


def stage_global_embeddings(
    stage_root: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load verified float16 embedding shards in deterministic order."""

    blocks: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    output_offset = 0
    for receipt_path in sorted(Path(stage_root).glob("shard-*.receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("embeddings_sha256") is None:
            continue
        embeddings_path = receipt_path.with_name(
            receipt_path.name.replace(".receipt.json", ".embeddings.npy")
        )
        index_path = receipt_path.with_name(
            receipt_path.name.replace(".receipt.json", ".embedding-index.jsonl")
        )
        if (
            file_sha256(embeddings_path) != receipt["embeddings_sha256"]
            or file_sha256(index_path) != receipt["embedding_index_sha256"]
        ):
            raise ValueError("Embedding shard differs from its receipt")
        block = np.load(embeddings_path, allow_pickle=False)
        if block.dtype != np.float16 or block.ndim != 2:
            raise ValueError("Global embeddings must be a float16 matrix")
        for row in _read_jsonl(index_path):
            adjusted = dict(row)
            adjusted["start"] = output_offset + int(row["start"])
            index_rows.append(adjusted)
        blocks.append(block)
        output_offset += len(block)
    if not blocks:
        raise ValueError("No global embedding shards were found")
    return np.concatenate(blocks, axis=0), index_rows


def publish_prediction_bundle(
    *,
    cohort_root: Path,
    identity: FrozenPipelineIdentity,
    cohort_id: str,
    cohort_membership_sha256: str,
    scene_count: int,
    image_count: int,
) -> dict[str, Any]:
    """Publish the strict model-safe boundary consumed by Notebook 08F."""

    root = Path(cohort_root).resolve()
    score_rows = read_completed_stage(root / "peer_scoring")
    representation_rows = read_completed_stage(root / "dinov2_small")
    embeddings, block_index = stage_global_embeddings(root / "dinov2_small")
    score_by_key = {_record_key(row): row for row in score_rows}
    representation_by_key = {_record_key(row): row for row in representation_rows}
    block_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row for row in block_index
    }
    if set(score_by_key) != set(representation_by_key):
        raise ValueError("Scoring and representation image coverage differs")

    manifest_rows: list[dict[str, Any]] = []
    embedding_index_rows: list[dict[str, Any]] = []
    ordered_blocks: list[np.ndarray] = []
    next_row = 0
    for key in sorted(score_by_key):
        score_row = score_by_key[key]
        objects = list(score_row.get("objects", []))
        block = block_by_key.get(key)
        if objects:
            if block is None or int(block["count"]) != len(objects):
                raise ValueError("Object count differs from the embedding index")
            start = int(block["start"])
            vectors = np.asarray(
                embeddings[start : start + len(objects)], dtype=np.float16
            )
            if len(vectors) != len(objects):
                raise ValueError("Global embedding slice is incomplete")
            ordered_blocks.append(vectors)
            for offset, (item, vector) in enumerate(zip(objects, vectors, strict=True)):
                embedding_index_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "cohort_id": cohort_id,
                        "opaque_scene_id": key[0],
                        "view_id": key[1],
                        "object_id": str(item["object_id"]),
                        "row_index": next_row + offset,
                        "feature_dimension": int(vectors.shape[1]),
                        "vector_sha256": sha256(
                            np.ascontiguousarray(vector).tobytes(order="C")
                        ).hexdigest(),
                    }
                )
            next_row += len(objects)
        public_objects = []
        for item in objects:
            public_objects.append(
                {
                    "object_id": str(item["object_id"]),
                    "box_xyxy": [int(value) for value in item["box_xyxy"]],
                    "mask_rle": item["mask_rle"],
                    "mask_sha256": str(item["mask_sha256"]),
                    "foreground_pixels": int(item["foreground_pixels"]),
                    "predicted_quality_score": float(item["predicted_quality_score"]),
                    "b0_status": str(item["b0_status"]),
                    "b0_score": item.get("b0_score"),
                    "p1_status": str(item["p1_status"]),
                    "p1_score": item.get("p1_score"),
                    "p1_rank": item.get("p1_rank"),
                    "p1_heatmap": item.get("p1_heatmap"),
                }
            )
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cohort_id": cohort_id,
                "opaque_scene_id": key[0],
                "view_id": key[1],
                "image_sha256": str(score_row.get("image_sha256", "")),
                "status": str(score_row.get("status", "terminal_failure")),
                "terminal_failure": score_row.get("terminal_failure"),
                "object_count": len(public_objects),
                "objects": public_objects,
                "per_view_p1_candidate": score_row.get("per_view_p1_candidate"),
                "per_view_p1_score": score_row.get("per_view_p1_score"),
            }
        )

    published_embeddings = (
        np.concatenate(ordered_blocks, axis=0).astype(np.float16)
        if ordered_blocks
        else np.empty((0, 0), dtype=np.float16)
    )
    output_root = root / "model-safe"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "prediction_manifest.jsonl"
    embeddings_path = output_root / "global_embeddings.npy"
    index_path = output_root / "embedding_index.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(index_path, embedding_index_rows)
    temporary = embeddings_path.with_suffix(".npy.incomplete")
    with temporary.open("wb") as handle:
        np.save(handle, published_embeddings, allow_pickle=False)
    temporary.replace(embeddings_path)
    lock = lock_predictions(
        prediction_root=output_root,
        identity=identity,
        cohort_id=cohort_id,
        cohort_membership_sha256=cohort_membership_sha256,
        scene_count=scene_count,
        image_count=image_count,
    )
    return {
        "prediction_manifest": manifest_path.as_posix(),
        "prediction_manifest_sha256": file_sha256(manifest_path),
        "global_embeddings": embeddings_path.as_posix(),
        "global_embeddings_sha256": file_sha256(embeddings_path),
        "embedding_index": index_path.as_posix(),
        "embedding_index_sha256": file_sha256(index_path),
        "prediction_lock": (output_root / "prediction_lock.json").as_posix(),
        "prediction_lock_sha256": file_sha256(output_root / "prediction_lock.json"),
        "prediction_count": len(manifest_rows),
        "embedding_count": len(published_embeddings),
        "lock": lock,
    }


def _artifact_rows(root: Path, excluded_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Prediction artifacts cannot contain symbolic links")
        if path.is_file() and path.name not in excluded_names:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def lock_predictions(
    *,
    prediction_root: Path,
    identity: FrozenPipelineIdentity,
    cohort_id: str,
    cohort_membership_sha256: str,
    scene_count: int,
    image_count: int,
) -> dict[str, Any]:
    """Seal model outputs before any development reference can be opened."""

    if not _is_sha256(cohort_membership_sha256):
        raise ValueError("Cohort membership must be a SHA-256 digest")
    if scene_count < 1 or image_count < scene_count:
        raise ValueError("Prediction lock counts are invalid")
    root = Path(prediction_root).resolve()
    lock_path = root / "prediction_lock.json"
    artifacts = _artifact_rows(root, {"prediction_lock.json"})
    if not artifacts:
        raise ValueError("A prediction lock requires completed artifacts")
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked",
        "cohort_id": cohort_id,
        "cohort_membership_sha256": cohort_membership_sha256,
        "scene_count": scene_count,
        "image_count": image_count,
        "identity": asdict(identity),
        "identity_sha256": identity.sha256,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _json_sha256(artifacts),
    }
    _write_json(lock_path, record)
    return record


def verify_prediction_lock(lock_path: Path) -> dict[str, Any]:
    """Verify the complete content-addressed prediction boundary."""

    path = Path(lock_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    identity = FrozenPipelineIdentity(**record["identity"])
    if (
        record.get("status") != "locked"
        or record.get("identity_sha256") != identity.sha256
    ):
        raise ValueError("Prediction lock identity is invalid")
    root = path.parent
    artifacts = record.get("artifacts", [])
    if int(record.get("artifact_count", -1)) != len(artifacts):
        raise ValueError("Prediction lock artifact count differs")
    if record.get("artifact_inventory_sha256") != _json_sha256(artifacts):
        raise ValueError("Prediction artifact inventory digest differs")
    for item in artifacts:
        artifact_path = (root / str(item["path"])).resolve()
        if (
            root not in artifact_path.parents
            or file_sha256(artifact_path) != item["sha256"]
        ):
            raise ValueError("Prediction artifact differs from its lock")
    return record


def combine_development_locks(
    *,
    ds1_lock: Path,
    remainder_lock: Path,
    ds2_lock: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Combine 100 + 600 + verified 300 scenes only under one identity."""

    records = [
        verify_prediction_lock(path) for path in (ds1_lock, remainder_lock, ds2_lock)
    ]
    expected = {
        "toysad8k-development-screen-100-v1": 100,
        "toysad8k-development-remainder-600-v1": 600,
        "toysad8k-development-confirm-300-v1": 300,
    }
    observed = {
        str(record["cohort_id"]): int(record["scene_count"]) for record in records
    }
    if observed != expected:
        raise ValueError("Development locks do not match the 100 + 600 + 300 protocol")
    identity_digests = {str(record["identity_sha256"]) for record in records}
    if len(identity_digests) != 1:
        raise ValueError("Development predictions use different frozen identities")
    split_digests = {str(record["identity"]["split_sha256"]) for record in records}
    if len(split_digests) != 1:
        raise ValueError("Development predictions use different split identities")
    combined = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "cohort_id": "toysad8k-development-full-1000-v1",
        "scene_count": 1_000,
        "image_count": sum(int(record["image_count"]) for record in records),
        "identity": records[0]["identity"],
        "identity_sha256": records[0]["identity_sha256"],
        "component_locks": [
            {
                "cohort_id": record["cohort_id"],
                "lock_sha256": file_sha256(path),
            }
            for record, path in zip(
                records, (ds1_lock, remainder_lock, ds2_lock), strict=True
            )
        ],
        "ds2_reused_only_after_full_identity_check": True,
    }
    _write_json(Path(output_path), combined)
    return combined


def load_development_evaluator(evaluator_path: Path, prediction_lock_path: Path) -> Any:
    """Open a development reference only after verifying the prediction lock."""

    verify_prediction_lock(prediction_lock_path)
    path = Path(evaluator_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Development evaluator resource is unavailable")
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reference_only_fusion(
    prediction_rows: Sequence[Mapping[str, Any]],
    evaluator_correspondence: Mapping[str, Any],
) -> dict[str, Any]:
    """Return evaluator-aligned analysis with an explicit non-deployment scope."""

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "reference_only",
        "deployment_valid": False,
        "prediction_count": len(prediction_rows),
        "correspondence_sha256": _json_sha256(evaluator_correspondence),
        "interpretation": "Evaluator correspondence is used only for development analysis.",
    }


def validate_ds2_backfill_inputs(
    *,
    locate_run_record_path: Path,
    automatic_sam_record_path: Path,
    mask_route_record_path: Path,
    selected_prompt_path: Path,
    expected_locate_model_id: str,
    expected_locate_revision: str,
    expected_locate_configuration_sha256: str,
    expected_prompt: str,
    expected_proposal_policy: str,
    expected_sam_configuration_sha256: str,
) -> dict[str, Any]:
    """Fail closed before reusing DS2 LocateAnything and SAM predictions."""

    locate_path = Path(locate_run_record_path).resolve()
    sam_path = Path(automatic_sam_record_path).resolve()
    routes_path = Path(mask_route_record_path).resolve()
    selected_prompt_path = Path(selected_prompt_path).resolve()
    locate = json.loads(locate_path.read_text(encoding="utf-8"))
    sam = json.loads(sam_path.read_text(encoding="utf-8"))
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    selected_prompt = json.loads(selected_prompt_path.read_text(encoding="utf-8"))
    required_locate = {
        "schema_version",
        "stage",
        "status",
        "cohort_id",
        "image_count",
        "ds1_freeze_sha256",
        "ds1_generation_configuration_sha256",
        "effective_generation_configuration_sha256",
        "proposal_package_sha256",
    }
    required_prompt = {
        "schema_version",
        "stage",
        "development_only",
        "requires_unchanged_ds2_confirmation",
        "model_id",
        "model_revision",
        "generation_configuration_sha256",
        "selected",
    }
    required_sam = {
        "schema_version",
        "stage",
        "status",
        "route",
        "cohort_id",
        "image_count",
        "configuration_sha256",
        "prompt_package_sha256",
        "sam_results_sha256",
        "all_completion_checks_passed",
    }
    required_routes = {
        "schema_version",
        "package_kind",
        "cohort_id",
        "image_count",
        "mask_routes_sha256",
        "automatic_sam_results_sha256",
        "all_checks_passed",
    }
    if required_locate.difference(locate):
        raise ValueError("DS2 LocateAnything run record lacks required identity fields")
    if required_prompt.difference(selected_prompt):
        raise ValueError("Frozen prompt record lacks required identity fields")
    if required_sam.difference(sam):
        raise ValueError(
            "DS2 automatic SAM record lacks required frozen identity fields"
        )
    if required_routes.difference(routes):
        raise ValueError("DS2 mask-route record lacks required schema fields")
    selected = selected_prompt.get("selected", {})
    if (
        locate["schema_version"] != 1
        or locate["stage"] != "ds2_frozen_locateanything"
        or locate["status"] != "completed"
        or locate["cohort_id"] != "toysad8k-development-confirm-300-v1"
        or int(locate["image_count"]) != 6_000
        or locate["ds1_freeze_sha256"] != file_sha256(selected_prompt_path)
        or locate["ds1_generation_configuration_sha256"]
        != selected_prompt["generation_configuration_sha256"]
        or locate["effective_generation_configuration_sha256"]
        != expected_locate_configuration_sha256
    ):
        raise ValueError("DS2 LocateAnything identity differs from the frozen route")
    if (
        selected_prompt["schema_version"] != 1
        or selected_prompt["stage"] != "ds1_locateanything_prompt_freeze"
        or selected_prompt["development_only"] is not True
        or selected_prompt["requires_unchanged_ds2_confirmation"] is not True
        or selected_prompt["model_id"] != expected_locate_model_id
        or selected_prompt["model_revision"] != expected_locate_revision
        or selected.get("prompt") != expected_prompt
        or selected.get("postprocessing_policy") != expected_proposal_policy
    ):
        raise ValueError("Frozen prompt identity differs from the expected route")
    if (
        sam["schema_version"] != 1
        or sam["status"] != "completed"
        or sam["route"] != "automatic_box"
        or sam["cohort_id"] != "toysad8k-development-confirm-300-v1"
        or int(sam["image_count"]) != 6_000
        or sam["all_completion_checks_passed"] is not True
        or sam["configuration_sha256"] != expected_sam_configuration_sha256
        or sam["prompt_package_sha256"] != locate["proposal_package_sha256"]
    ):
        raise ValueError("DS2 automatic SAM identity differs from the frozen route")
    if (
        routes["schema_version"] != 1
        or routes["package_kind"] != "ds2-confirmation-mask-routes"
        or routes["cohort_id"] != "toysad8k-development-confirm-300-v1"
        or int(routes["image_count"]) != 6_000
        or routes["all_checks_passed"] is not True
        or routes["automatic_sam_results_sha256"] != sam["sam_results_sha256"]
    ):
        raise ValueError("DS2 mask-route schema differs from the frozen route")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "cohort_id": locate["cohort_id"],
        "locate_record_sha256": file_sha256(locate_path),
        "selected_prompt_sha256": file_sha256(selected_prompt_path),
        "automatic_sam_record_sha256": file_sha256(sam_path),
        "mask_route_record_sha256": file_sha256(routes_path),
        "locate_configuration_sha256": expected_locate_configuration_sha256,
        "sam_configuration_sha256": expected_sam_configuration_sha256,
        "proposal_policy": expected_proposal_policy,
        "schema_identity_sha256": _json_sha256(
            {
                "locate_schema_version": locate["schema_version"],
                "prompt_schema_version": selected_prompt["schema_version"],
                "sam_schema_version": sam["schema_version"],
                "mask_route_schema_version": routes["schema_version"],
                "mask_routes_sha256": routes["mask_routes_sha256"],
            }
        ),
        "reuse_scope": "locateanything_and_sam_only",
        "requires_dinov2_and_scoring_rebuild": True,
    }


def select_development_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    base_count: int = DEFAULT_AUDIT_COUNT,
    namespace: str = "toysad8k-development-audit-v1",
) -> list[dict[str, Any]]:
    """Choose prediction-only examples plus one exemplar per terminal failure."""

    if not 12 <= base_count <= 18:
        raise ValueError("The base audit sample must contain 12 to 18 results")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for source in records:
        row = dict(source)
        key = _record_key(row)
        if key in indexed:
            raise ValueError("Audit candidates contain duplicate scene-view keys")
        indexed[key] = row
    ranked = sorted(
        indexed.values(),
        key=lambda row: sha256(
            f"{namespace}|{_record_key(row)[0]}|{_record_key(row)[1]}".encode()
        ).hexdigest(),
    )
    selected = ranked[: min(base_count, len(ranked))]
    selected_keys = {_record_key(row) for row in selected}
    for failure in sorted(
        {
            str(row.get("terminal_failure"))
            for row in ranked
            if row.get("terminal_failure")
        }
    ):
        if failure not in TERMINAL_FAILURES:
            raise ValueError("Audit candidate contains an unsupported terminal failure")
        exemplar = next(row for row in ranked if row.get("terminal_failure") == failure)
        if _record_key(exemplar) not in selected_keys:
            selected.append(exemplar)
            selected_keys.add(_record_key(exemplar))
    return selected


def bounded_tracking_call(
    operation: Callable[[], Any],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Run optional tracking without allowing it to block inference."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    outcome: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put_nowait(("complete", operation()))
        except Exception as error:  # noqa: BLE001 - optional tracking returns a failure receipt
            outcome.put_nowait(("failed", type(error).__name__))

    # A daemon thread cannot hold notebook completion if an optional tracker stalls.
    thread = threading.Thread(target=invoke, name="clearml-safe", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        return {"status": "timed_out", "value": None, "error_type": "TimeoutError"}
    status, value = outcome.get_nowait()
    if status == "failed":
        return {"status": "failed", "value": None, "error_type": value}
    return {"status": "complete", "value": value, "error_type": None}


def _mask_array(value: Any, size: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            array = np.asarray(opened.convert("L")) > 0
    else:
        array = np.asarray(value)
    if array.shape != (size[1], size[0]):
        array = (
            np.asarray(
                Image.fromarray(array.astype(np.uint8) * 255).resize(
                    size, Image.Resampling.NEAREST
                )
            )
            > 0
        )
    return array.astype(bool)


def render_mask_evidence_panel(
    result: QualitativeResult, *, panel_size: int = 256
) -> Image.Image:
    """Render SAM outlines, binary masks, and masked-on-black evidence."""

    views = sorted(
        (view for view in result.views if view.rgb is not None),
        key=lambda view: view.view_id,
    )
    if not views:
        raise ValueError("Mask evidence requires at least one present view")
    margin, gap, label_height = 16, 12, 30
    width = margin * 2 + 3 * panel_size + 2 * gap
    height = 70 + len(views) * (panel_size + label_height + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (margin, 16), f"SAM evidence: {result.result_id}", fill=(30, 39, 49), font=font
    )
    headings = ("Mask outline", "Binary mask", "Masked on black")
    for column, heading in enumerate(headings):
        draw.text(
            (margin + column * (panel_size + gap), 42),
            heading,
            fill=(30, 39, 49),
            font=font,
        )
    for row_index, view in enumerate(views):
        source = (
            view.rgb
            if isinstance(view.rgb, Image.Image)
            else Image.open(view.rgb)
            if isinstance(view.rgb, (str, Path))
            else Image.fromarray(np.asarray(view.rgb, dtype=np.uint8))
        )
        rgb = source.convert("RGB").resize(
            (panel_size, panel_size), Image.Resampling.BILINEAR
        )
        combined = np.zeros((panel_size, panel_size), dtype=bool)
        for item in view.objects:
            combined |= _mask_array(item.sam_mask, (panel_size, panel_size))
        binary = Image.fromarray(combined.astype(np.uint8) * 255, mode="L")
        edge = binary.filter(ImageFilter.FIND_EDGES).point(
            lambda value: 255 if value else 0
        )
        outline = rgb.copy()
        outline.paste(Image.new("RGB", outline.size, (230, 45, 70)), mask=edge)
        rgb_array = np.asarray(rgb)
        masked = Image.fromarray(
            np.where(combined[:, :, None], rgb_array, 0).astype(np.uint8)
        )
        y = 70 + row_index * (panel_size + label_height + gap)
        for column, panel in enumerate((outline, binary.convert("RGB"), masked)):
            x = margin + column * (panel_size + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1), outline=(170, 178, 188)
            )
        draw.text(
            (margin, y + panel_size + 6), view.view_id, fill=(30, 39, 49), font=font
        )
    return canvas


def save_development_audit_bundle(
    result: QualitativeResult,
    output_root: Path,
) -> dict[str, str]:
    """Save publication-safe prediction, heatmap, and mask evidence panels."""

    root = Path(output_root)
    prediction_path = save_panel_png(
        render_prediction_panel(result), root / f"{result.result_id}-prediction.png"
    )
    mask_path = save_panel_png(
        render_mask_evidence_panel(result),
        root / f"{result.result_id}-sam-evidence.png",
    )
    return {
        "prediction_panel": prediction_path.as_posix(),
        "prediction_sha256": file_sha256(prediction_path),
        "sam_evidence_panel": mask_path.as_posix(),
        "sam_evidence_sha256": file_sha256(mask_path),
    }


__all__ = [
    "DEFAULT_AUDIT_COUNT",
    "TERMINAL_FAILURES",
    "EmptyMaskFailure",
    "FeatureFailure",
    "FrozenPipelineIdentity",
    "StageOutcome",
    "ZeroProposalFailure",
    "bounded_tracking_call",
    "combine_development_locks",
    "load_development_evaluator",
    "lock_predictions",
    "publish_prediction_bundle",
    "read_completed_stage",
    "reference_only_fusion",
    "render_mask_evidence_panel",
    "run_sharded_stage",
    "save_development_audit_bundle",
    "select_development_audit",
    "stage_global_embeddings",
    "validate_ds2_backfill_inputs",
    "verify_prediction_lock",
]
