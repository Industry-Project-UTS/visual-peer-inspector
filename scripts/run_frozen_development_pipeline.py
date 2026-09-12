#!/usr/bin/env python3
"""Run the frozen LocateAnything, SAM 2.1, DINOv2-small and peer route."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import (
    CALIBRATION_CONTRACT,
    DEVELOPMENT_FULL_CONTRACT,
    DEVELOPMENT_REMAINDER_CONTRACT,
    DS1_SCREEN_CONTRACT,
    DS2_CONFIRM_CONTRACT,
    INTERNAL_LOCK_CONTRACT,
    OFFICIAL_SEEN_TEST_CONTRACT,
    OFFICIAL_UNSEEN_TEST_CONTRACT,
)
from visual_peer_inspector.data.confirmation_mask_routes import (
    verify_ds2_mask_route_package,
)
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.models.locateanything import (
    LocateAnythingWorker,
    ProposalJob,
    build_proposal_jobs,
)
from visual_peer_inspector.models.sam2 import (
    Sam2Configuration,
    Sam2ImageWorker,
    SamImageJob,
    SamPrompt,
    decode_binary_mask,
)
from visual_peer_inspector.models.visual_encoder import (
    FrozenVisualEncoder,
    specification_by_key,
)
from visual_peer_inspector.qualitative_reporting import (
    PredictedObject,
    QualitativeResult,
    QualitativeView,
)
from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    prepare_masked_crop,
    project_crop_heatmap,
)
from visual_peer_inspector.scoring.peer import (
    b0_scores,
    deterministic_ranking,
    p1_scores,
)
from visual_peer_inspector.workflows.frozen_development_pipeline import (
    EmptyMaskFailure,
    FeatureFailure,
    FrozenPipelineIdentity,
    StageOutcome,
    ZeroProposalFailure,
    combine_development_locks,
    load_development_evaluator,
    publish_prediction_bundle,
    read_completed_stage,
    reference_only_fusion,
    run_sharded_stage,
    save_development_audit_bundle,
    select_development_audit,
    validate_ds2_backfill_inputs,
    validate_recovery_stage_chain,
)
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    prune_verified_state,
    restore_recovery_bundle,
    write_recovery_tracking_root,
    write_safe_artifact_manifest,
)
from visual_peer_inspector.workflows.locateanything_confirmation_ds2 import (
    load_frozen_ds1_configuration,
    resolve_ds2_operational_policy,
)
from visual_peer_inspector.workflows.locateanything_prompt_freeze_ds1 import (
    PROMPT_POSTPROCESSING_POLICY_BY_KEY,
)
from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
    apply_proposal_policy,
)
from visual_peer_inspector.workflows.representation_confirmation_ds2 import (
    load_ds2_candidates,
)

DINO_MODEL_KEY = "dinov2_small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
FROZEN_PROPOSAL_POLICY = "area_nms_075_cap8"
COHORTS = {
    "ds1": DS1_SCREEN_CONTRACT,
    "remainder": DEVELOPMENT_REMAINDER_CONTRACT,
}
PREFLIGHT_COHORTS = {
    **COHORTS,
    "calibration": CALIBRATION_CONTRACT,
    "internal-lock": INTERNAL_LOCK_CONTRACT,
    "official-seen": OFFICIAL_SEEN_TEST_CONTRACT,
    "official-unseen": OFFICIAL_UNSEEN_TEST_CONTRACT,
}
DEVELOPMENT_RECOVERY_TOP_LEVELS = (
    DS1_SCREEN_CONTRACT.cohort_id,
    DEVELOPMENT_REMAINDER_CONTRACT.cohort_id,
)


def _digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, **dict(details)}, sort_keys=True), flush=True)


def _clear_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - GPU cleanup is best effort after a completed stage
        return


def _development_recovery_binding(
    args: argparse.Namespace, identity: FrozenPipelineIdentity
) -> dict[str, Any]:
    """Bind a checkpoint to the exact pipeline and both model-facing inputs."""

    return {
        "stage": "08e-frozen-development-pipeline",
        "pipeline_identity_sha256": identity.sha256,
        "cohorts": {
            DS1_SCREEN_CONTRACT.cohort_id: {
                "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
                "package_record_sha256": file_sha256(
                    args.ds1_model_input_root / "package_record.json"
                ),
                "manifest_sha256": file_sha256(
                    args.ds1_model_input_root / "neutral_corpus_manifest.jsonl"
                ),
            },
            DEVELOPMENT_REMAINDER_CONTRACT.cohort_id: {
                "membership_sha256": DEVELOPMENT_REMAINDER_CONTRACT.membership_sha256,
                "package_record_sha256": file_sha256(
                    args.remainder_model_input_root / "package_record.json"
                ),
                "manifest_sha256": file_sha256(
                    args.remainder_model_input_root / "neutral_corpus_manifest.jsonl"
                ),
            },
        },
    }


def _ensure_cohort_identity(
    root: Path, *, identity: FrozenPipelineIdentity, cohort_id: str
) -> None:
    record = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "identity_sha256": identity.sha256,
    }
    path = root / "pipeline_identity.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(
                "Existing output belongs to another frozen pipeline identity"
            )
    else:
        _write_json(path, record)


def _ds2_recovery_binding(
    args: argparse.Namespace, identity: FrozenPipelineIdentity
) -> dict[str, Any]:
    """Bind a DS2 representation checkpoint to verified proposal and mask inputs."""

    return {
        "stage": "08e-ds2-backfill",
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "pipeline_identity_sha256": identity.sha256,
        "input_membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "rgb_package_record_sha256": file_sha256(args.rgb_root / "package_record.json"),
        "rgb_manifest_sha256": file_sha256(
            args.rgb_root / "neutral_corpus_manifest.jsonl"
        ),
        "locate_run_record_sha256": file_sha256(args.locate_run_record),
        "automatic_sam_record_sha256": file_sha256(args.automatic_sam_record),
        "mask_route_record_sha256": file_sha256(
            args.mask_route_root / "package_record.json"
        ),
    }


def build_identity(
    selected_prompt: Path,
) -> tuple[FrozenPipelineIdentity, Any, Mapping[str, Any]]:
    """Resolve and hash every part of the frozen automatic route."""

    locate_configuration, policy_key, _freeze_sha256 = load_frozen_ds1_configuration(
        selected_prompt
    )
    if policy_key != FROZEN_PROPOSAL_POLICY:
        raise ValueError(
            f"Notebook 08E requires the frozen {FROZEN_PROPOSAL_POLICY} policy"
        )
    effective_locate, execution_policy = resolve_ds2_operational_policy(
        locate_configuration
    )
    sam_configuration = Sam2Configuration(device="cuda", precision="float16")
    crop_configuration = CropConfiguration()
    route_configuration = {
        "proposal_policy": policy_key,
        "locate_configuration_sha256": effective_locate.sha256,
        "locate_execution_policy": execution_policy,
        "sam_configuration_sha256": sam_configuration.sha256,
        "dino_model_key": DINO_MODEL_KEY,
        "dino_revision": DINO_REVISION,
        "crop_configuration_sha256": crop_configuration.sha256,
        "methods": ["B0", "P1"],
        "p1_top_fraction": 0.10,
        "minimum_peer_count": 3,
    }
    identity = FrozenPipelineIdentity(
        split_sha256=DEVELOPMENT_FULL_CONTRACT.membership_sha256,
        configuration_sha256=_digest(route_configuration),
        locateanything_sha256=_digest(
            {
                "model_id": effective_locate.model_id,
                "revision": effective_locate.model_revision,
            }
        ),
        sam21_sha256=_digest(sam_configuration.to_dict()),
        dinov2_sha256=_digest({"model_key": DINO_MODEL_KEY, "revision": DINO_REVISION}),
        preprocessing_sha256=crop_configuration.sha256,
        schema_sha256=_digest(
            {
                "name": "visual-peer-frozen-development-prediction",
                "version": 1,
                "stages": ["locateanything", "sam21", "dinov2_small", "peer_scoring"],
            }
        ),
    )
    return identity, effective_locate, execution_policy


def _input_records(model_input_root: Path, contract: Any) -> list[dict[str, Any]]:
    verify_model_input_package(model_input_root, contract=contract)
    jobs = build_proposal_jobs(model_input_root, contract.cohort_id)
    if len(jobs) != contract.image_count:
        raise ValueError("Model-input image count differs from the cohort contract")
    return [
        {
            "opaque_scene_id": job.opaque_scene_id,
            "view_id": job.view_id,
            "cohort_id": job.cohort_id,
            "input_manifest_sha256": job.input_manifest_sha256,
            "image_path": job.image_path.as_posix(),
            "image_sha256": job.image_sha256,
        }
        for job in jobs
    ]


def _locate_worker(model: LocateAnythingWorker, policy_key: str):
    policy = PROMPT_POSTPROCESSING_POLICY_BY_KEY[policy_key]

    def run(record: Mapping[str, Any]) -> Mapping[str, Any]:
        job = ProposalJob(
            cohort_id=str(record["cohort_id"]),
            input_manifest_sha256=str(record["input_manifest_sha256"]),
            opaque_scene_id=str(record["opaque_scene_id"]),
            view_id=str(record["view_id"]),
            image_path=Path(str(record["image_path"])),
            image_sha256=str(record["image_sha256"]),
        )
        result = model.run_job(job, attempt=1)
        if result.get("status") != "ok":
            raise ZeroProposalFailure("LocateAnything did not complete this image")
        boxes, _indices, removed = apply_proposal_policy(result["boxes_xyxy"], policy)
        if not boxes:
            raise ZeroProposalFailure("LocateAnything returned no usable proposal")
        return {
            "status": "ok",
            "cohort_id": record["cohort_id"],
            "image_path": record["image_path"],
            "image_sha256": record["image_sha256"],
            "proposal_policy": policy_key,
            "raw_proposal_count": int(result["proposal_count"]),
            "removed_proposal_count": int(sum(removed.values())),
            "proposal_count": len(boxes),
            "boxes_xyxy": boxes,
            "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        }

    return run


def _sam_worker(model: Sam2ImageWorker, identity: FrozenPipelineIdentity):
    def run(record: Mapping[str, Any]) -> Mapping[str, Any]:
        if record.get("terminal_failure") == "zero_proposal":
            raise ZeroProposalFailure("No proposal was available for SAM")
        boxes = [
            tuple(int(value) for value in box) for box in record.get("boxes_xyxy", [])
        ]
        if not boxes:
            raise ZeroProposalFailure("No proposal was available for SAM")
        prompts = tuple(
            SamPrompt(route="automatic_box", prompt_key=f"P{index:02d}", box_xyxy=box)
            for index, box in enumerate(boxes, 1)
        )
        job = SamImageJob(
            opaque_scene_id=str(record["opaque_scene_id"]),
            view_id=str(record["view_id"]),
            image_path=Path(str(record["image_path"])),
            image_sha256=str(record["image_sha256"]),
            prompts=prompts,
            prompt_package_sha256=identity.sha256,
            cohort_id=str(record["cohort_id"]),
        )
        result = model.process(job, 1)
        objects = []
        for item in result["routes"]["automatic_box"]:
            if int(item["foreground_pixels"]) <= 0:
                continue
            objects.append(
                {
                    "object_id": f"{record['view_id']}-{item['prompt_key']}",
                    "box_xyxy": item["box_xyxy"],
                    "predicted_quality_score": item["predicted_quality_score"],
                    "foreground_pixels": item["foreground_pixels"],
                    "mask_sha256": item["mask_sha256"],
                    "mask_rle": item["mask_rle"],
                }
            )
        if not objects:
            raise EmptyMaskFailure("SAM returned no non-empty object mask")
        return {
            "status": "ok",
            "cohort_id": record["cohort_id"],
            "image_path": record["image_path"],
            "image_sha256": record["image_sha256"],
            "objects": objects,
            "object_count": len(objects),
            "embedding_runtime_seconds": result["embedding_runtime_seconds"],
            "prompt_runtime_seconds": result["prompt_runtime_seconds"],
        }

    return run


def _valid_patch_mask(crop: Any, grid_size: int) -> np.ndarray:
    valid = crop.patch_validity(grid_size, minimum_fraction=0.50)
    if valid.any():
        return valid
    patch_size = crop.foreground.shape[0] // grid_size
    coverage = crop.foreground.reshape(
        grid_size, patch_size, grid_size, patch_size
    ).mean(axis=(1, 3))
    valid.flat[int(np.argmax(coverage))] = True
    return valid


def _dino_worker(encoder: FrozenVisualEncoder, crop_configuration: CropConfiguration):
    def run(record: Mapping[str, Any]) -> StageOutcome:
        terminal = record.get("terminal_failure")
        if terminal == "zero_proposal":
            raise ZeroProposalFailure("No proposal was available for representation")
        if terminal == "empty_mask":
            raise EmptyMaskFailure("No non-empty mask was available for representation")
        objects = list(record.get("objects", []))
        if not objects:
            raise EmptyMaskFailure("No non-empty mask was available for representation")
        with Image.open(str(record["image_path"])) as opened:
            rgb = opened.convert("RGB")
            crops = [
                prepare_masked_crop(
                    rgb, item["mask_rle"], configuration=crop_configuration
                )
                for item in objects
            ]
        try:
            encoded = encoder.encode([crop.rgb for crop in crops])
        except Exception as error:
            raise FeatureFailure("DINOv2-small feature extraction failed") from error
        globals_ = np.asarray(encoded.global_features, dtype=np.float32)
        if globals_.shape[0] != len(objects) or not np.all(np.isfinite(globals_)):
            raise FeatureFailure("DINOv2-small returned invalid global features")
        if not encoded.patch_features or encoded.patch_grid_size is None:
            raise FeatureFailure("DINOv2-small returned no patch features")
        valid_masks = [
            _valid_patch_mask(crop, int(encoded.patch_grid_size)) for crop in crops
        ]
        b0 = b0_scores(globals_)
        p1 = p1_scores(encoded.patch_features, valid_masks, top_fraction=0.10)
        ranks: dict[int, int] = {}
        if p1.status == "scored":
            order = deterministic_ranking(
                p1.scores, [str(item["object_id"]) for item in objects]
            )
            ranks = {index: rank for rank, index in enumerate(order, 1)}
        result_objects = []
        for index, item in enumerate(objects):
            result_objects.append(
                {
                    **item,
                    "b0_status": b0.status,
                    "b0_score": None
                    if b0.status != "scored"
                    else float(b0.scores[index]),
                    "p1_status": p1.status,
                    "p1_score": None
                    if p1.status != "scored"
                    else float(p1.scores[index]),
                    "p1_rank": None if p1.status != "scored" else int(ranks[index]),
                    "p1_heatmap": None
                    if p1.status != "scored"
                    else np.asarray(p1.heatmaps[index]).tolist(),
                }
            )
        return StageOutcome(
            record={
                "status": "ok",
                "cohort_id": record["cohort_id"],
                "image_path": record["image_path"],
                "image_sha256": record["image_sha256"],
                "scoring_status": p1.status,
                "objects": result_objects,
                "object_count": len(result_objects),
                "patch_grid_size": int(encoded.patch_grid_size),
            },
            global_embeddings=globals_,
        )

    return run


def _score_worker(record: Mapping[str, Any]) -> Mapping[str, Any]:
    terminal = record.get("terminal_failure")
    if terminal == "zero_proposal":
        raise ZeroProposalFailure("LocateAnything produced no usable proposal")
    if terminal == "empty_mask":
        raise EmptyMaskFailure("SAM produced no usable mask")
    if terminal == "feature_failure":
        raise FeatureFailure("DINOv2-small produced no usable feature")
    objects = list(record.get("objects", []))
    scored = [item for item in objects if item.get("p1_score") is not None]
    strongest = (
        max(scored, key=lambda item: (float(item["p1_score"]), str(item["object_id"])))
        if scored
        else None
    )
    return {
        "status": "ok" if scored else "insufficient_peers",
        "cohort_id": record["cohort_id"],
        "image_path": record["image_path"],
        "image_sha256": record["image_sha256"],
        "object_count": len(objects),
        "objects": objects,
        "per_view_p1_candidate": None if strongest is None else strongest["object_id"],
        "per_view_p1_score": None if strongest is None else strongest["p1_score"],
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Load every pinned model and process one usable image before a long run."""

    if args.maximum_image_attempts < 1:
        raise ValueError("maximum_image_attempts must be positive")
    contract = PREFLIGHT_COHORTS[args.cohort]
    identity, locate_configuration, execution_policy = build_identity(
        args.selected_prompt
    )
    input_rows = _input_records(args.model_input_root, contract)
    attempts: list[dict[str, Any]] = []
    locate_result: Mapping[str, Any] | None = None
    selected_record: Mapping[str, Any] | None = None

    locate = LocateAnythingWorker(
        locate_configuration,
        device=args.device,
        model_source=args.locate_model_source,
        execution_policy=execution_policy,
        progress_callback=lambda event, detail: _progress(
            f"preflight_locate_{event}", detail
        ),
    )
    locate_worker = _locate_worker(locate, FROZEN_PROPOSAL_POLICY)
    for record in input_rows[: args.maximum_image_attempts]:
        try:
            locate_result = locate_worker(record)
            selected_record = record
            attempts.append({"status": "usable_proposals"})
            break
        except ZeroProposalFailure:
            attempts.append({"status": "no_usable_proposal"})
    del locate
    _clear_gpu()
    if locate_result is None or selected_record is None:
        raise RuntimeError(
            "LocateAnything loaded but produced no usable proposal in the bounded preflight set"
        )

    sam = Sam2ImageWorker(
        Sam2Configuration(device=args.device, precision="float16"),
        args.sam_checkpoint,
    )
    sam_result = _sam_worker(sam, identity)({**selected_record, **dict(locate_result)})
    del sam
    _clear_gpu()

    encoder = FrozenVisualEncoder(
        specification_by_key(DINO_MODEL_KEY),
        revision=DINO_REVISION,
        device=args.device,
    )
    representation = _dino_worker(encoder, CropConfiguration())(
        {**selected_record, **dict(sam_result)}
    )
    del encoder
    _clear_gpu()
    score = _score_worker(representation.record)
    result = {
        "schema_version": 1,
        "status": "passed",
        "stage": "frozen_pipeline_model_preflight",
        "cohort_id": contract.cohort_id,
        "pipeline_identity_sha256": identity.sha256,
        "attempted_image_count": len(attempts),
        "locateanything_proposal_count": int(locate_result["proposal_count"]),
        "sam_object_count": int(sam_result["object_count"]),
        "dino_object_count": int(representation.record["object_count"]),
        "peer_scoring_status": str(score["status"]),
        "models_loaded_sequentially": True,
    }
    _write_json(args.output_path, result)
    _progress("model_preflight_completed", result)
    return result


def _scene_audit_panels(
    score_rows: list[dict[str, Any]], output_root: Path
) -> list[dict[str, Any]]:
    audit_rows = select_development_audit(score_rows, base_count=15)
    scene_ids = []
    for row in audit_rows:
        scene_id = str(row["opaque_scene_id"])
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        by_scene.setdefault(str(row["opaque_scene_id"]), []).append(row)
    evidence = []
    for scene_id in scene_ids:
        rows = sorted(by_scene[scene_id], key=lambda row: str(row["view_id"]))
        step = max(1, len(rows) // 5)
        chosen = rows[::step][:5]
        if len(chosen) < 2:
            continue
        views = []
        for row in chosen:
            predicted = []
            heatmap = None
            with Image.open(str(row["image_path"])) as opened:
                rgb = opened.convert("RGB")
            for item in row.get("objects", []):
                predicted.append(
                    PredictedObject(
                        object_id=str(item["object_id"]),
                        box_xyxy=tuple(float(value) for value in item["box_xyxy"]),
                        sam_mask=decode_binary_mask(item["mask_rle"]),
                        p1_score=item.get("p1_score"),
                        p1_rank=item.get("p1_rank"),
                    )
                )
                if item.get("p1_rank") == 1:
                    patch_heatmap = item.get("p1_heatmap")
                    if patch_heatmap is not None:
                        mask = decode_binary_mask(item["mask_rle"])
                        crop = prepare_masked_crop(
                            rgb, mask, configuration=CropConfiguration()
                        )
                        projected = project_crop_heatmap(patch_heatmap, crop, mask)
                        heatmap = projected if np.isfinite(projected).any() else None
            views.append(
                QualitativeView(
                    view_id=str(row["view_id"]),
                    rgb=str(row["image_path"]),
                    objects=tuple(predicted),
                    heatmap=None
                    if heatmap is None
                    else np.asarray(heatmap, dtype=np.float32),
                )
            )
        panel_id = f"panel-{len(evidence) + 1:03d}"
        result = QualitativeResult(
            result_id=panel_id,
            views=tuple(views),
            fused_odd_one_out=None,
            agreement=None,
            margin=None,
            review_state="prediction_only",
            publication_metadata={
                "route": "frozen_automatic",
                "view_count": len(views),
            },
        )
        files = save_development_audit_bundle(result, output_root)
        evidence.append(
            {
                "panel_id": panel_id,
                "prediction_panel": Path(files["prediction_panel"])
                .resolve()
                .relative_to(output_root.resolve().parent.parent)
                .as_posix(),
                "prediction_sha256": files["prediction_sha256"],
                "sam_evidence_panel": Path(files["sam_evidence_panel"])
                .resolve()
                .relative_to(output_root.resolve().parent.parent)
                .as_posix(),
                "sam_evidence_sha256": files["sam_evidence_sha256"],
            }
        )
    return evidence


def run_cohort(args: argparse.Namespace) -> dict[str, Any]:
    contract = COHORTS[args.cohort]
    identity, locate_configuration, execution_policy = build_identity(
        args.selected_prompt
    )
    input_rows = _input_records(args.model_input_root, contract)
    cohort_root = args.output_root.expanduser().resolve() / contract.cohort_id

    locate = LocateAnythingWorker(
        locate_configuration,
        device=args.device,
        model_source=args.locate_model_source,
        execution_policy=execution_policy,
        progress_callback=lambda event, detail: _progress(f"locate_{event}", detail),
    )
    locate_record = run_sharded_stage(
        stage="locateanything",
        input_records=input_rows,
        output_root=cohort_root,
        identity=identity,
        worker=_locate_worker(locate, FROZEN_PROPOSAL_POLICY),
        shard_size=args.shard_size,
        progress_callback=_progress,
    )
    del locate
    _clear_gpu()

    sam_configuration = Sam2Configuration(device=args.device, precision="float16")
    sam = Sam2ImageWorker(sam_configuration, args.sam_checkpoint)
    proposal_rows = read_completed_stage(cohort_root / "locateanything")
    sam_record = run_sharded_stage(
        stage="sam21",
        input_records=proposal_rows,
        output_root=cohort_root,
        identity=identity,
        worker=_sam_worker(sam, identity),
        shard_size=args.shard_size,
        progress_callback=_progress,
    )
    del sam
    _clear_gpu()

    crop_configuration = CropConfiguration()
    encoder = FrozenVisualEncoder(
        specification_by_key(DINO_MODEL_KEY),
        revision=DINO_REVISION,
        device=args.device,
    )
    mask_rows = read_completed_stage(cohort_root / "sam21")
    representation_record = run_sharded_stage(
        stage="dinov2_small",
        input_records=mask_rows,
        output_root=cohort_root,
        identity=identity,
        worker=_dino_worker(encoder, crop_configuration),
        shard_size=args.shard_size,
        progress_callback=_progress,
    )
    del encoder
    _clear_gpu()

    representation_rows = read_completed_stage(cohort_root / "dinov2_small")
    scoring_record = run_sharded_stage(
        stage="peer_scoring",
        input_records=representation_rows,
        output_root=cohort_root,
        identity=identity,
        worker=_score_worker,
        shard_size=args.shard_size,
        progress_callback=_progress,
    )
    score_rows = read_completed_stage(cohort_root / "peer_scoring")
    audit_evidence = _scene_audit_panels(
        score_rows, cohort_root / "review-only" / "figures"
    )
    _write_json(
        cohort_root / "review-only" / "audit_evidence.json",
        {"items": audit_evidence},
    )
    run_record = {
        "schema_version": 1,
        "status": "complete",
        "stage": "frozen_development_pipeline",
        "cohort_id": contract.cohort_id,
        "scene_count": contract.scene_count,
        "image_count": contract.image_count,
        "identity_sha256": identity.sha256,
        "stages": {
            "locateanything": locate_record,
            "sam21": sam_record,
            "dinov2_small": representation_record,
            "peer_scoring": scoring_record,
        },
        "audit_panel_count": len(audit_evidence),
        "prediction_only": True,
        "evaluator_opened": False,
    }
    _write_json(cohort_root / "safe" / "run_record.json", run_record)
    bundle_record = publish_prediction_bundle(
        cohort_root=cohort_root,
        identity=identity,
        cohort_id=contract.cohort_id,
        cohort_membership_sha256=contract.membership_sha256,
        scene_count=contract.scene_count,
        image_count=contract.image_count,
    )
    run_record["prediction_bundle"] = {
        key: value for key, value in bundle_record.items() if key != "lock"
    }
    _write_json(cohort_root / "safe" / "run_record.json", run_record)
    write_safe_artifact_manifest(cohort_root / "safe")
    return run_record


def _preserve_failed_run(
    args: argparse.Namespace,
    binding: Mapping[str, Any],
    top_levels: tuple[str, ...],
    stem: str,
    error: Exception,
) -> None:
    """Keep completed shards recoverable when a later stage or report raises."""

    if args.recovery_output_directory is None:
        return
    if not any(args.output_root.glob("*/**/shard-*.receipt.json")):
        return
    directory = args.recovery_output_directory.expanduser().resolve()
    try:
        for cohort_id in top_levels:
            record_path = args.output_root / cohort_id / "safe/run_record.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record.update(status="interrupted", error_type=type(error).__name__)
                _write_json(record_path, record)
        recovery = create_recovery_bundle(
            source_root=args.output_root,
            bundle_path=directory / f"{stem}-recovery.zip",
            manifest_path=directory / f"{stem}-recovery-manifest.json",
            binding=binding,
            allowed_top_levels=top_levels,
        )
        _progress(
            "failure_recovery_saved",
            {
                "artifact_count": recovery["artifact_count"],
                "bundle_sha256": recovery["bundle_sha256"],
                "error_type": type(error).__name__,
            },
        )
    except Exception as recovery_error:
        # Preserve the original exception; loose atomic shards remain on disk.
        _progress(
            "failure_recovery_error", {"error_type": type(recovery_error).__name__}
        )


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    identity, _configuration, _policy = build_identity(args.selected_prompt)
    binding = _development_recovery_binding(args, identity)
    try:
        return _run_development(args)
    except Exception as error:
        _preserve_failed_run(
            args, binding, DEVELOPMENT_RECOVERY_TOP_LEVELS, "notebook08e", error
        )
        raise


def _run_development(args: argparse.Namespace) -> dict[str, Any]:
    """Process the 100-scene screen and 600-scene remainder with one model load."""

    identity, locate_configuration, execution_policy = build_identity(
        args.selected_prompt
    )
    sources = (
        (DS1_SCREEN_CONTRACT, args.ds1_model_input_root),
        (DEVELOPMENT_REMAINDER_CONTRACT, args.remainder_model_input_root),
    )
    cohort_rows = {
        contract.cohort_id: _input_records(model_root, contract)
        for contract, model_root in sources
    }
    cohort_roots = {
        contract.cohort_id: args.output_root.expanduser().resolve() / contract.cohort_id
        for contract, _model_root in sources
    }
    recovery_binding = _development_recovery_binding(args, identity)
    recovery_inputs = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery_inputs) and not all(recovery_inputs):
        raise ValueError("Recovery requires both the ZIP and sidecar manifest")
    if all(recovery_inputs):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=args.output_root,
            expected_binding=recovery_binding,
            allowed_top_levels=DEVELOPMENT_RECOVERY_TOP_LEVELS,
        )
    for contract, _model_root in sources:
        _ensure_cohort_identity(
            cohort_roots[contract.cohort_id],
            identity=identity,
            cohort_id=contract.cohort_id,
        )
        verified_shards = validate_recovery_stage_chain(
            cohort_roots[contract.cohort_id],
            cohort_rows[contract.cohort_id],
            identity,
            args.shard_size,
        )
        _progress(
            "recovery_inputs_verified",
            {
                "cohort_id": contract.cohort_id,
                "verified_shards": verified_shards,
            },
        )
    stage_records: dict[str, dict[str, Any]] = {
        contract.cohort_id: {} for contract, _ in sources
    }
    maximum_new_shards = (
        args.maximum_new_shards if args.run_mode == "incremental" else None
    )

    locate = LocateAnythingWorker(
        locate_configuration,
        device=args.device,
        model_source=args.locate_model_source,
        execution_policy=execution_policy,
        progress_callback=lambda event, detail: _progress(f"locate_{event}", detail),
    )
    for contract, _model_root in sources:
        stage_records[contract.cohort_id]["locateanything"] = run_sharded_stage(
            stage="locateanything",
            input_records=cohort_rows[contract.cohort_id],
            output_root=cohort_roots[contract.cohort_id],
            identity=identity,
            worker=_locate_worker(locate, FROZEN_PROPOSAL_POLICY),
            shard_size=args.shard_size,
            maximum_new_shards=maximum_new_shards,
            progress_callback=_progress,
        )
    del locate
    _clear_gpu()

    sam = Sam2ImageWorker(
        Sam2Configuration(device=args.device, precision="float16"),
        args.sam_checkpoint,
    )
    for contract, _model_root in sources:
        root = cohort_roots[contract.cohort_id]
        stage_records[contract.cohort_id]["sam21"] = run_sharded_stage(
            stage="sam21",
            input_records=read_completed_stage(root / "locateanything"),
            output_root=root,
            identity=identity,
            worker=_sam_worker(sam, identity),
            shard_size=args.shard_size,
            maximum_new_shards=maximum_new_shards,
            progress_callback=_progress,
        )
    del sam
    _clear_gpu()

    crop_configuration = CropConfiguration()
    encoder = FrozenVisualEncoder(
        specification_by_key(DINO_MODEL_KEY),
        revision=DINO_REVISION,
        device=args.device,
    )
    for contract, _model_root in sources:
        root = cohort_roots[contract.cohort_id]
        stage_records[contract.cohort_id]["dinov2_small"] = run_sharded_stage(
            stage="dinov2_small",
            input_records=read_completed_stage(root / "sam21"),
            output_root=root,
            identity=identity,
            worker=_dino_worker(encoder, crop_configuration),
            shard_size=args.shard_size,
            maximum_new_shards=maximum_new_shards,
            progress_callback=_progress,
        )
    del encoder
    _clear_gpu()

    records = {}
    for contract, _model_root in sources:
        root = cohort_roots[contract.cohort_id]
        stage_records[contract.cohort_id]["peer_scoring"] = run_sharded_stage(
            stage="peer_scoring",
            input_records=read_completed_stage(root / "dinov2_small"),
            output_root=root,
            identity=identity,
            worker=_score_worker,
            shard_size=args.shard_size,
            maximum_new_shards=maximum_new_shards,
            progress_callback=_progress,
        )
        complete = (
            all(
                stage.get("complete") is True
                for stage in stage_records[contract.cohort_id].values()
            )
            and stage_records[contract.cohort_id]["peer_scoring"]["covered_input_count"]
            == contract.image_count
        )
        record = {
            "schema_version": 1,
            "status": "reporting" if complete else "checkpointed",
            "stage": "frozen_development_pipeline",
            "cohort_id": contract.cohort_id,
            "scene_count": contract.scene_count,
            "image_count": contract.image_count,
            "identity_sha256": identity.sha256,
            "stages": stage_records[contract.cohort_id],
            "audit_panel_count": 0,
            "prediction_only": True,
            "evaluator_opened": False,
        }
        _write_json(root / "safe" / "run_record.json", record)
        if complete:
            score_rows = read_completed_stage(root / "peer_scoring")
            evidence = _scene_audit_panels(score_rows, root / "review-only" / "figures")
            _write_json(
                root / "review-only" / "audit_evidence.json",
                {"items": evidence},
            )
            record["audit_panel_count"] = len(evidence)
            bundle = publish_prediction_bundle(
                cohort_root=root,
                identity=identity,
                cohort_id=contract.cohort_id,
                cohort_membership_sha256=contract.membership_sha256,
                scene_count=contract.scene_count,
                image_count=contract.image_count,
            )
            record["prediction_bundle"] = {
                key: value for key, value in bundle.items() if key != "lock"
            }
            record["status"] = "complete"
        origin_path = root / "recovery_origin.json"
        if origin_path.is_file():
            record["recovery_origin"] = json.loads(
                origin_path.read_text(encoding="utf-8")
            )
        _write_json(root / "safe" / "run_record.json", record)
        write_safe_artifact_manifest(root / "safe")
        records[contract.cohort_id] = record
    result = {
        "schema_version": 1,
        "status": (
            "complete"
            if all(record["status"] == "complete" for record in records.values())
            else "checkpointed"
        ),
        "identity_sha256": identity.sha256,
        "cohorts": records,
        "recovery_imported": bool(all(recovery_inputs)),
    }
    _write_json(args.output_root / "development_progress.json", result)
    if args.recovery_output_directory is not None:
        recovery_directory = args.recovery_output_directory.expanduser().resolve()
        recovery_directory.mkdir(parents=True, exist_ok=True)
        bundle_path = recovery_directory / "notebook08e-recovery.zip"
        manifest_path = recovery_directory / "notebook08e-recovery-manifest.json"
        recovery = create_recovery_bundle(
            source_root=args.output_root,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            binding=recovery_binding,
            allowed_top_levels=DEVELOPMENT_RECOVERY_TOP_LEVELS,
        )
        result["recovery"] = {
            "bundle": bundle_path.as_posix(),
            "manifest": manifest_path.as_posix(),
            "bundle_sha256": recovery["bundle_sha256"],
            "artifact_count": recovery["artifact_count"],
        }
        write_recovery_tracking_root(
            safe_root=args.output_root / "tracking-safe",
            stage="frozen_development_pipeline",
            pipeline_status=result["status"],
            metrics={
                "cohort_count": len(records),
                "complete_cohort_count": sum(
                    record["status"] == "complete" for record in records.values()
                ),
                "covered_image_count": sum(
                    int(record["stages"]["peer_scoring"]["covered_input_count"])
                    for record in records.values()
                ),
                "target_image_count": sum(
                    int(record["image_count"]) for record in records.values()
                ),
                "recovery_artifact_count": recovery["artifact_count"],
            },
        )
        if args.prune_after_recovery:
            result["pruned_file_count"] = prune_verified_state(
                source_root=args.output_root,
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                expected_binding=recovery_binding,
                allowed_top_levels=DEVELOPMENT_RECOVERY_TOP_LEVELS,
            )
    _write_json(args.output_root / "development_progress.json", result)
    return result


def backfill_ds2(args: argparse.Namespace) -> dict[str, Any]:
    identity, _configuration, _policy = build_identity(args.selected_prompt)
    binding = _ds2_recovery_binding(args, identity)
    try:
        return _backfill_ds2(args)
    except Exception as error:
        _preserve_failed_run(
            args,
            binding,
            (DS2_CONFIRM_CONTRACT.cohort_id,),
            "notebook08e-ds2-backfill",
            error,
        )
        raise


def _backfill_ds2(args: argparse.Namespace) -> dict[str, Any]:
    """Repackage verified DS2 predictions under the 08E identity and schema."""

    identity, locate_configuration, _execution_policy = build_identity(
        args.selected_prompt
    )
    sam_configuration = Sam2Configuration(device="cuda", precision="float16")
    attestation = validate_ds2_backfill_inputs(
        locate_run_record_path=args.locate_run_record,
        automatic_sam_record_path=args.automatic_sam_record,
        mask_route_record_path=args.mask_route_root / "package_record.json",
        selected_prompt_path=args.selected_prompt,
        expected_locate_model_id=locate_configuration.model_id,
        expected_locate_revision=locate_configuration.model_revision,
        expected_locate_configuration_sha256=locate_configuration.sha256,
        expected_prompt=locate_configuration.prompt,
        expected_proposal_policy=FROZEN_PROPOSAL_POLICY,
        expected_sam_configuration_sha256=sam_configuration.sha256,
    )
    verify_model_input_package(args.rgb_root, contract=DS2_CONFIRM_CONTRACT)
    verify_ds2_mask_route_package(args.mask_route_root)
    recovery_binding = _ds2_recovery_binding(args, identity)
    recovery_inputs = (args.recovery_bundle, args.recovery_manifest)
    if any(recovery_inputs) and not all(recovery_inputs):
        raise ValueError("DS2 recovery requires both the ZIP and sidecar manifest")
    if all(recovery_inputs):
        restore_recovery_bundle(
            bundle_path=args.recovery_bundle,
            manifest_path=args.recovery_manifest,
            destination_root=args.output_root,
            expected_binding=recovery_binding,
            allowed_top_levels=(DS2_CONFIRM_CONTRACT.cohort_id,),
        )
    grouped = load_ds2_candidates(args.rgb_root, args.mask_route_root)
    image_jobs = {
        (job.opaque_scene_id, job.view_id): job
        for job in build_proposal_jobs(args.rgb_root, DS2_CONFIRM_CONTRACT.cohort_id)
    }
    mask_rows = []
    for scene_id, view_id in sorted(image_jobs):
        candidates = grouped[(scene_id, view_id, "automatic_box")]
        job = image_jobs[(scene_id, view_id)]
        if not candidates:
            mask_rows.append(
                {
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
                    "image_path": job.image_path.as_posix(),
                    "image_sha256": job.image_sha256,
                    "status": "terminal_failure",
                    "terminal_failure": "zero_proposal",
                    "objects": [],
                }
            )
            continue
        objects = [
            {
                "object_id": candidate.candidate_id,
                "box_xyxy": list(candidate.box_xyxy),
                "predicted_quality_score": candidate.predicted_quality_score,
                "foreground_pixels": candidate.foreground_pixels,
                "mask_sha256": candidate.mask_sha256,
                "mask_rle": dict(candidate.mask_rle),
            }
            for candidate in candidates
            if candidate.foreground_pixels > 0
        ]
        mask_rows.append(
            {
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
                "image_path": job.image_path.as_posix(),
                "image_sha256": job.image_sha256,
                "status": "ok" if objects else "terminal_failure",
                "terminal_failure": None if objects else "empty_mask",
                "objects": objects,
            }
        )

    root = args.output_root.expanduser().resolve() / DS2_CONFIRM_CONTRACT.cohort_id
    _ensure_cohort_identity(
        root,
        identity=identity,
        cohort_id=DS2_CONFIRM_CONTRACT.cohort_id,
    )
    _write_json(root / "safe" / "backfill_attestation.json", attestation)
    maximum_new_shards = (
        args.maximum_new_shards if args.run_mode == "incremental" else None
    )
    encoder = FrozenVisualEncoder(
        specification_by_key(DINO_MODEL_KEY),
        revision=DINO_REVISION,
        device=args.device,
    )
    representation_record = run_sharded_stage(
        stage="dinov2_small",
        input_records=mask_rows,
        output_root=root,
        identity=identity,
        worker=_dino_worker(encoder, CropConfiguration()),
        shard_size=args.shard_size,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    del encoder
    _clear_gpu()
    scoring_record = run_sharded_stage(
        stage="peer_scoring",
        input_records=read_completed_stage(root / "dinov2_small"),
        output_root=root,
        identity=identity,
        worker=_score_worker,
        shard_size=args.shard_size,
        maximum_new_shards=maximum_new_shards,
        progress_callback=_progress,
    )
    complete = (
        representation_record.get("complete") is True
        and scoring_record.get("complete") is True
        and scoring_record["input_count"] == DS2_CONFIRM_CONTRACT.image_count
    )
    evidence: list[dict[str, Any]] = []
    if complete:
        score_rows = read_completed_stage(root / "peer_scoring")
        evidence = _scene_audit_panels(score_rows, root / "review-only" / "figures")
        _write_json(
            root / "review-only" / "audit_evidence.json",
            {"items": evidence},
        )
    record = {
        "schema_version": 1,
        "status": "complete" if complete else "checkpointed",
        "stage": "frozen_development_pipeline",
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "identity_sha256": identity.sha256,
        "reused_stages": ["locateanything", "sam21"],
        "rebuilt_stages": ["dinov2_small", "peer_scoring"],
        "backfill_attestation_sha256": _digest(attestation),
        "stages": {
            "dinov2_small": representation_record,
            "peer_scoring": scoring_record,
        },
        "audit_panel_count": len(evidence),
        "prediction_only": True,
        "evaluator_opened": False,
    }
    _write_json(root / "safe" / "run_record.json", record)
    if complete:
        bundle = publish_prediction_bundle(
            cohort_root=root,
            identity=identity,
            cohort_id=DS2_CONFIRM_CONTRACT.cohort_id,
            cohort_membership_sha256=DS2_CONFIRM_CONTRACT.membership_sha256,
            scene_count=DS2_CONFIRM_CONTRACT.scene_count,
            image_count=DS2_CONFIRM_CONTRACT.image_count,
        )
        record["prediction_bundle"] = {
            key: value for key, value in bundle.items() if key != "lock"
        }
    if args.recovery_output_directory is not None and not complete:
        recovery_directory = args.recovery_output_directory.expanduser().resolve()
        recovery_directory.mkdir(parents=True, exist_ok=True)
        bundle_path = recovery_directory / "notebook08e-ds2-backfill-recovery.zip"
        manifest_path = (
            recovery_directory / "notebook08e-ds2-backfill-recovery-manifest.json"
        )
        recovery = create_recovery_bundle(
            source_root=args.output_root,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            binding=recovery_binding,
            allowed_top_levels=(DS2_CONFIRM_CONTRACT.cohort_id,),
        )
        record["recovery"] = {
            "bundle": bundle_path.as_posix(),
            "manifest": manifest_path.as_posix(),
            "bundle_sha256": recovery["bundle_sha256"],
            "artifact_count": recovery["artifact_count"],
        }
        write_recovery_tracking_root(
            safe_root=args.output_root / "tracking-safe",
            stage="frozen_development_pipeline",
            pipeline_status=record["status"],
            metrics={
                "cohort_count": 1,
                "complete_cohort_count": int(complete),
                "covered_image_count": int(scoring_record["covered_input_count"]),
                "target_image_count": DS2_CONFIRM_CONTRACT.image_count,
                "recovery_artifact_count": recovery["artifact_count"],
            },
        )
        if args.prune_after_recovery:
            record["pruned_file_count"] = prune_verified_state(
                source_root=args.output_root,
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                expected_binding=recovery_binding,
                allowed_top_levels=(DS2_CONFIRM_CONTRACT.cohort_id,),
            )
    _write_json(root / "safe" / "run_record.json", record)
    write_safe_artifact_manifest(root / "safe")
    return record


def combine(args: argparse.Namespace) -> dict[str, Any]:
    return combine_development_locks(
        ds1_lock=args.ds1_lock,
        remainder_lock=args.remainder_lock,
        ds2_lock=args.ds2_lock,
        output_path=args.output_path,
    )


def evaluate_reference(args: argparse.Namespace) -> dict[str, Any]:
    predictions = read_completed_stage(
        args.prediction_lock.parent.parent / "peer_scoring"
    )
    evaluator = load_development_evaluator(args.evaluator_path, args.prediction_lock)
    return reference_only_fusion(predictions, {"rows": evaluator})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--cohort", choices=tuple(PREFLIGHT_COHORTS), required=True
    )
    preflight_parser.add_argument("--model-input-root", type=Path, required=True)
    preflight_parser.add_argument("--selected-prompt", type=Path, required=True)
    preflight_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--output-path", type=Path, required=True)
    preflight_parser.add_argument("--locate-model-source", type=Path)
    preflight_parser.add_argument("--device", default="cuda")
    preflight_parser.add_argument("--maximum-image-attempts", type=int, default=5)
    run_parser = subparsers.add_parser("run-cohort")
    run_parser.add_argument("--cohort", choices=tuple(COHORTS), required=True)
    run_parser.add_argument("--model-input-root", type=Path, required=True)
    run_parser.add_argument("--selected-prompt", type=Path, required=True)
    run_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--locate-model-source", type=Path)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--shard-size", type=int, default=100)

    development_parser = subparsers.add_parser("run-development")
    development_parser.add_argument("--ds1-model-input-root", type=Path, required=True)
    development_parser.add_argument(
        "--remainder-model-input-root", type=Path, required=True
    )
    development_parser.add_argument("--selected-prompt", type=Path, required=True)
    development_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    development_parser.add_argument("--output-root", type=Path, required=True)
    development_parser.add_argument("--locate-model-source", type=Path)
    development_parser.add_argument("--device", default="cuda")
    development_parser.add_argument("--shard-size", type=int, default=100)
    development_parser.add_argument(
        "--run-mode", choices=("incremental", "full"), default="full"
    )
    development_parser.add_argument("--maximum-new-shards", type=int, default=10)
    development_parser.add_argument("--recovery-bundle", type=Path)
    development_parser.add_argument("--recovery-manifest", type=Path)
    development_parser.add_argument("--recovery-output-directory", type=Path)
    development_parser.add_argument("--prune-after-recovery", action="store_true")

    backfill_parser = subparsers.add_parser("backfill-ds2")
    backfill_parser.add_argument("--rgb-root", type=Path, required=True)
    backfill_parser.add_argument("--mask-route-root", type=Path, required=True)
    backfill_parser.add_argument("--locate-run-record", type=Path, required=True)
    backfill_parser.add_argument("--automatic-sam-record", type=Path, required=True)
    backfill_parser.add_argument("--selected-prompt", type=Path, required=True)
    backfill_parser.add_argument("--output-root", type=Path, required=True)
    backfill_parser.add_argument("--device", default="cuda")
    backfill_parser.add_argument("--shard-size", type=int, default=100)
    backfill_parser.add_argument(
        "--run-mode", choices=("incremental", "full"), default="full"
    )
    backfill_parser.add_argument("--maximum-new-shards", type=int, default=10)
    backfill_parser.add_argument("--recovery-bundle", type=Path)
    backfill_parser.add_argument("--recovery-manifest", type=Path)
    backfill_parser.add_argument("--recovery-output-directory", type=Path)
    backfill_parser.add_argument("--prune-after-recovery", action="store_true")

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--ds1-lock", type=Path, required=True)
    combine_parser.add_argument("--remainder-lock", type=Path, required=True)
    combine_parser.add_argument("--ds2-lock", type=Path, required=True)
    combine_parser.add_argument("--output-path", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate-reference")
    evaluate_parser.add_argument("--prediction-lock", type=Path, required=True)
    evaluate_parser.add_argument("--evaluator-path", type=Path, required=True)
    evaluate_parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.command == "preflight":
        record = run_preflight(args)
    elif args.command == "run-cohort":
        record = run_cohort(args)
    elif args.command == "run-development":
        record = run_development(args)
    elif args.command == "backfill-ds2":
        record = backfill_ds2(args)
    elif args.command == "combine":
        record = combine(args)
    else:
        record = evaluate_reference(args)
        _write_json(args.output_path, record)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
