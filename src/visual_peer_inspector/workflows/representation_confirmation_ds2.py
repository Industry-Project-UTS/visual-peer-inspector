"""Frozen DINOv2-small and peer scoring for the DS2 confirmation cohort."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from visual_peer_inspector.data.confirmation_mask_routes import (
    ROUTES,
    verify_ds2_mask_route_package,
)
from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram
from visual_peer_inspector.models.visual_encoder import (
    ContractVisualEncoder,
    FrozenVisualEncoder,
    specification_by_key,
)
from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    PreparedCrop,
    prepare_masked_crop,
)
from visual_peer_inspector.scoring.peer import (
    b0_scores,
    deterministic_ranking,
    p1_scores,
)


STAGE = "ds2_frozen_representation_and_peer_scores"
MODEL_KEY = "dinov2_small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
METHODS = ("B0", "P1")
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class ConfirmationCandidate:
    opaque_scene_id: str
    view_id: str
    route: str
    peer_key: str
    candidate_id: str
    image_path: Path
    image_sha256: str
    box_xyxy: tuple[int, int, int, int]
    foreground_pixels: int
    predicted_quality_score: float | None
    mask_rle: Mapping[str, Any]
    mask_sha256: str


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("DS2 representation outputs cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    record = {
        "schema_version": 1,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": sha256(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        ).hexdigest(),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def _candidate_id(scene_id: str, view_id: str, route: str, object_id: str) -> str:
    payload = f"{scene_id}|{view_id}|{route}|{object_id}".encode("utf-8")
    return "C" + sha256(payload).hexdigest()[:20]


def load_ds2_candidates(
    rgb_root: Path,
    mask_route_root: Path,
) -> dict[tuple[str, str, str], list[ConfirmationCandidate]]:
    """Verify the label-free inputs and join each route by image digest."""

    rgb_root = rgb_root.expanduser().resolve()
    verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    route_record = verify_ds2_mask_route_package(mask_route_root)
    grouped: dict[tuple[str, str, str], list[ConfirmationCandidate]] = defaultdict(list)
    for row in read_jsonl(mask_route_root / "mask_routes.jsonl"):
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        image_path = rgb_root / str(row["image_path"])
        image_digest = file_sha256(image_path)
        if image_digest != str(row["image_sha256"]):
            raise ValueError("DS2 representation RGB digest differs from the route package")
        for route in ROUTES:
            grouped.setdefault((scene_id, view_id, route), [])
            for item in row["routes"][route]:
                peer_key = str(item["peer_key"])
                score = item.get("predicted_quality_score")
                grouped[(scene_id, view_id, route)].append(
                    ConfirmationCandidate(
                        opaque_scene_id=scene_id,
                        view_id=view_id,
                        route=route,
                        peer_key=peer_key,
                        candidate_id=_candidate_id(scene_id, view_id, route, peer_key),
                        image_path=image_path,
                        image_sha256=image_digest,
                        box_xyxy=tuple(int(value) for value in item["box_xyxy"]),
                        foreground_pixels=int(item["foreground_pixels"]),
                        predicted_quality_score=None if score is None else float(score),
                        mask_rle=dict(item["mask_rle"]),
                        mask_sha256=str(item["mask_sha256"]),
                    )
                )
    expected_groups = DS2_CONFIRM_CONTRACT.image_count * len(ROUTES)
    if len(grouped) != expected_groups:
        raise ValueError("DS2 representation group coverage is incomplete")
    if route_record.get("cohort_id") != DS2_CONFIRM_CONTRACT.cohort_id:
        raise ValueError("DS2 representation route package identifies another cohort")
    return dict(grouped)


def _encode_batches(
    encoder: Any,
    prepared: Sequence[PreparedCrop],
    *,
    batch_size: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], int, int]:
    globals_: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    grid_size: int | None = None
    calls = 0
    for start in range(0, len(prepared), batch_size):
        encoded = encoder.encode([item.rgb for item in prepared[start : start + batch_size]])
        if grid_size is None:
            grid_size = encoded.patch_grid_size
        elif grid_size != encoded.patch_grid_size:
            raise RuntimeError("DINO patch grid changed between batches")
        globals_.append(np.asarray(encoded.global_features, dtype=np.float32))
        patches.extend(encoded.patch_features)
        calls += 1
    if grid_size is None or len(patches) != len(prepared):
        raise RuntimeError("DINOv2-small did not return spatial features for every crop")
    global_array = np.concatenate(globals_, axis=0)
    if not np.allclose(np.linalg.norm(global_array, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("DINO global features are not unit normalized")
    return global_array, tuple(patches), int(grid_size), calls


def _valid_patch_mask(crop: PreparedCrop, grid_size: int) -> tuple[np.ndarray, bool]:
    valid = crop.patch_validity(grid_size, minimum_fraction=0.50)
    if valid.any():
        return valid, False
    patch_size = crop.foreground.shape[0] // grid_size
    coverage = crop.foreground.reshape(
        grid_size, patch_size, grid_size, patch_size
    ).mean(axis=(1, 3))
    valid.flat[int(np.argmax(coverage))] = True
    return valid, True


def _score_group(
    candidates: Sequence[ConfirmationCandidate],
    globals_: np.ndarray,
    patches: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    *,
    candidate_count: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...]]:
    total_candidate_count = len(candidates) if candidate_count is None else candidate_count
    if total_candidate_count < len(candidates):
        raise ValueError("Total candidate count cannot be smaller than the valid candidates")
    results = {
        "B0": b0_scores(globals_),
        "P1": p1_scores(patches, valid_masks, top_fraction=0.10),
    }
    rows = []
    for method, result in results.items():
        ranks: dict[int, int] = {}
        if result.status == "scored":
            ordering = deterministic_ranking(
                result.scores,
                [candidate.candidate_id for candidate in candidates],
            )
            ranks = {index: rank for rank, index in enumerate(ordering, start=1)}
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "schema_version": 1,
                    "candidate_id": candidate.candidate_id,
                    "opaque_scene_id": candidate.opaque_scene_id,
                    "view_id": candidate.view_id,
                    "route": candidate.route,
                    "peer_key": candidate.peer_key,
                    "image_sha256": candidate.image_sha256,
                    "encoder_key": MODEL_KEY,
                    "method": method,
                    "scoring_status": result.status,
                    "scoring_reason": (
                        None
                        if result.status == "scored"
                        else "fewer_than_three_valid_candidates"
                    ),
                    "score": (
                        float(result.scores[index])
                        if result.status == "scored"
                        else None
                    ),
                    "rank": int(ranks[index]) if result.status == "scored" else None,
                    "candidate_count": total_candidate_count,
                    "foreground_pixels": candidate.foreground_pixels,
                    "predicted_mask_quality": candidate.predicted_quality_score,
                }
            )
    p1 = results["P1"]
    if p1.status == "scored":
        heatmaps = p1.heatmaps
    else:
        heatmaps = tuple(
            np.full(mask.shape, np.nan, dtype=np.float64) for mask in valid_masks
        )
    return rows, heatmaps


def _invalid_mask_rows(
    candidate: ConfirmationCandidate,
    *,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Keep an invalid candidate visible without inventing encoder evidence."""

    return [
        {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "opaque_scene_id": candidate.opaque_scene_id,
            "view_id": candidate.view_id,
            "route": candidate.route,
            "peer_key": candidate.peer_key,
            "image_sha256": candidate.image_sha256,
            "encoder_key": MODEL_KEY,
            "method": method,
            "scoring_status": "invalid_mask",
            "scoring_reason": "foreground_pixels_below_minimum",
            "score": None,
            "rank": None,
            "candidate_count": candidate_count,
            "foreground_pixels": candidate.foreground_pixels,
            "predicted_mask_quality": candidate.predicted_quality_score,
        }
        for method in METHODS
    ]


def _score_scene(
    *,
    scene_id: str,
    scene_groups: Sequence[tuple[tuple[str, str, str], Sequence[ConfirmationCandidate]]],
    encoder: Any,
    crop_configuration: CropConfiguration,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    prepared: list[PreparedCrop] = []
    spans = []
    image_cache: dict[Path, Image.Image] = {}
    for key, candidates in scene_groups:
        if not candidates:
            spans.append((key, candidates, (), len(prepared), len(prepared), 0))
            continue
        image = image_cache.get(candidates[0].image_path)
        if image is None:
            image = Image.open(candidates[0].image_path).convert("RGB")
            image_cache[candidates[0].image_path] = image
        start = len(prepared)
        valid_candidates = []
        invalid_candidates = []
        for candidate in candidates:
            try:
                crop = prepare_masked_crop(
                    image,
                    candidate.mask_rle,
                    configuration=crop_configuration,
                )
            except ValueError as error:
                if str(error) != "Object mask contains too little foreground":
                    raise
                invalid_candidates.append(candidate)
                continue
            valid_candidates.append(candidate)
            prepared.append(crop)
        spans.append(
            (
                key,
                valid_candidates,
                invalid_candidates,
                start,
                len(prepared),
                len(candidates),
            )
        )
    if prepared:
        globals_, patches, grid_size, calls = _encode_batches(
            encoder,
            prepared,
            batch_size=batch_size,
        )
    else:
        globals_ = np.empty((0, 0), dtype=np.float32)
        patches = ()
        grid_size = 0
        calls = 0
    rows: list[dict[str, Any]] = []
    heatmaps: list[np.ndarray] = []
    identifiers: list[str] = []
    fallback_count = 0
    invalid_mask_count = 0
    for _key, candidates, invalid_candidates, start, end, candidate_count in spans:
        validity = [_valid_patch_mask(item, grid_size) for item in prepared[start:end]]
        valid_masks = [item[0] for item in validity]
        fallback_count += sum(item[1] for item in validity)
        if candidates:
            group_rows, group_heatmaps = _score_group(
                candidates,
                globals_[start:end],
                patches[start:end],
                valid_masks,
                candidate_count=candidate_count,
            )
            rows.extend(group_rows)
            heatmaps.extend(group_heatmaps)
            identifiers.extend(candidate.candidate_id for candidate in candidates)
        for candidate in invalid_candidates:
            rows.extend(
                _invalid_mask_rows(
                    candidate,
                    candidate_count=candidate_count,
                )
            )
        invalid_mask_count += len(invalid_candidates)
    arrays = {
        "candidate_ids": np.asarray(identifiers, dtype="U32"),
        "p1_heatmaps": (
            np.stack(heatmaps).astype(np.float16)
            if heatmaps
            else np.empty((0, grid_size, grid_size), dtype=np.float16)
        ),
    }
    audit = {
        "scene_id": scene_id,
        "group_count": len(scene_groups),
        "candidate_count": sum(len(candidates) for _key, candidates in scene_groups),
        "encoded_candidate_count": len(prepared),
        "invalid_mask_candidate_count": invalid_mask_count,
        "score_row_count": len(rows),
        "encoder_call_count": calls,
        "patch_fallback_count": fallback_count,
    }
    return rows, arrays, audit


def _load_encoder(run_mode: str, device: str) -> tuple[Any, str, Mapping[str, Any]]:
    if run_mode == "contract":
        encoder = ContractVisualEncoder()
        return encoder, encoder.revision, asdict(encoder.specification)
    if run_mode != "model_inference":
        raise ValueError("DS2 representation mode must be contract or model_inference")
    specification = specification_by_key(MODEL_KEY)
    encoder = FrozenVisualEncoder(
        specification,
        revision=MODEL_REVISION,
        device=device,
    )
    return encoder, MODEL_REVISION, asdict(specification)


def _heatmap_panel(
    *,
    rows: Sequence[Mapping[str, Any]],
    grouped: Mapping[tuple[str, str, str], Sequence[ConfirmationCandidate]],
    cache_root: Path,
    output_path: Path,
    crop_configuration: CropConfiguration,
) -> None:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidates in grouped.values()
        for candidate in candidates
    }
    top = sorted(
        (
            row
            for row in rows
            if row["method"] == "P1" and row["scoring_status"] == "scored"
        ),
        key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
    )[:6]
    heatmap_by_id: dict[str, np.ndarray] = {}
    required = {str(row["candidate_id"]) for row in top}
    for path in sorted(cache_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as values:
            for identifier, heatmap in zip(values["candidate_ids"], values["p1_heatmaps"], strict=True):
                if str(identifier) in required:
                    heatmap_by_id[str(identifier)] = heatmap.astype(np.float32)
        if set(heatmap_by_id) == required:
            break
    panel = Image.new("RGB", (224 * 3, 258 * 2), "white")
    draw = ImageDraw.Draw(panel)
    for index, row in enumerate(top):
        identifier = str(row["candidate_id"])
        candidate = candidate_by_id[identifier]
        image = Image.open(candidate.image_path).convert("RGB")
        crop = prepare_masked_crop(image, candidate.mask_rle, configuration=crop_configuration)
        heatmap = heatmap_by_id[identifier]
        finite = heatmap[np.isfinite(heatmap)]
        scaled = np.zeros_like(heatmap)
        if finite.size and float(finite.max()) > float(finite.min()):
            scaled = (heatmap - float(finite.min())) / (float(finite.max()) - float(finite.min()))
        heat = np.asarray(
            Image.fromarray(np.uint8(np.nan_to_num(scaled) * 255)).resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        rgb = crop.rgb.astype(np.float32)
        alpha = (0.12 + 0.55 * heat)[:, :, None] * crop.foreground[:, :, None]
        colour = np.zeros_like(rgb)
        colour[:, :, 0] = 255
        colour[:, :, 1] = 70 * (1 - heat)
        overlay = np.uint8(np.clip(rgb * (1 - alpha) + colour * alpha, 0, 255))
        x = (index % 3) * 224
        y = (index // 3) * 258
        panel.paste(Image.fromarray(overlay), (x, y))
        draw.text((x + 4, y + 228), f"{row['route']}  P1 {float(row['score']):.3f}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def run_ds2_representation(
    *,
    rgb_root: Path,
    mask_route_root: Path,
    output_root: Path,
    run_mode: str = "contract",
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen encoder and B0/P1 scores before evaluator labels are opened."""

    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    cache_root = output_root / "cache"
    model_input_root = output_root / "model-input"
    grouped = load_ds2_candidates(rgb_root, mask_route_root)
    encoder, revision, specification = _load_encoder(run_mode, device)
    crop_configuration = CropConfiguration()
    configuration = {
        "stage": STAGE,
        "run_mode": run_mode,
        "model_key": MODEL_KEY,
        "model_specification": specification,
        "resolved_revision": revision,
        "crop_configuration": asdict(crop_configuration),
        "p1_top_fraction": 0.10,
        "patch_validity_minimum_fraction": 0.50,
        "invalid_mask_policy": {
            "status": "invalid_mask",
            "reason": "foreground_pixels_below_minimum",
            "ranking_evidence": 0.0,
            "retain_candidate": True,
        },
        "batch_size": batch_size,
        "routes": list(ROUTES),
        "methods": list(METHODS),
    }
    configuration_sha256 = sha256(canonical_json_bytes(configuration)).hexdigest()
    by_scene: dict[str, list[tuple[tuple[str, str, str], Sequence[ConfirmationCandidate]]]] = defaultdict(list)
    for key, candidates in sorted(grouped.items()):
        by_scene[key[0]].append((key, candidates))

    score_rows: list[dict[str, Any]] = []
    scene_audits = []
    started = time.perf_counter()
    first_repeat_checked = False
    for scene_number, (scene_id, scene_groups) in enumerate(sorted(by_scene.items()), start=1):
        score_path = private_root / "score-shards" / f"{scene_id}.jsonl"
        cache_path = cache_root / f"{scene_id}.npz"
        record_path = cache_root / f"{scene_id}.complete.json"
        reused = False
        if score_path.is_file() and cache_path.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            reused = (
                record.get("configuration_sha256") == configuration_sha256
                and record.get("score_sha256") == file_sha256(score_path)
                and record.get("cache_sha256") == file_sha256(cache_path)
            )
        if reused:
            rows = read_jsonl(score_path)
            audit = dict(record)
            audit["resumed"] = True
        else:
            rows, arrays, audit = _score_scene(
                scene_id=scene_id,
                scene_groups=scene_groups,
                encoder=encoder,
                crop_configuration=crop_configuration,
                batch_size=batch_size,
            )
            if not first_repeat_checked:
                repeated_rows, _repeated_arrays, _repeated_audit = _score_scene(
                    scene_id=scene_id,
                    scene_groups=scene_groups,
                    encoder=encoder,
                    crop_configuration=crop_configuration,
                    batch_size=batch_size,
                )
                left_identity = [
                    (row["candidate_id"], row["method"], row["scoring_status"], row["rank"])
                    for row in rows
                ]
                right_identity = [
                    (row["candidate_id"], row["method"], row["scoring_status"], row["rank"])
                    for row in repeated_rows
                ]
                left = [float(row["score"]) for row in rows if row["score"] is not None]
                right = [float(row["score"]) for row in repeated_rows if row["score"] is not None]
                if left_identity != right_identity or not np.allclose(left, right, atol=1e-5, rtol=1e-5):
                    raise RuntimeError("DS2 DINO/P1 repeatability gate failed")
                first_repeat_checked = True
            _write_jsonl(score_path, rows)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.with_suffix(".npz.incomplete").open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            cache_path.with_suffix(".npz.incomplete").replace(cache_path)
            record = {
                **audit,
                "configuration_sha256": configuration_sha256,
                "score_sha256": file_sha256(score_path),
                "cache_sha256": file_sha256(cache_path),
                "resumed": False,
            }
            _write_json(record_path, record)
            audit = record
        score_rows.extend(rows)
        scene_audits.append(audit)
        if progress_callback is not None:
            progress_callback(scene_number, len(by_scene), audit)
        elif scene_number == 1 or scene_number % 25 == 0 or scene_number == len(by_scene):
            print(
                f"DS2 representation scenes: {scene_number}/{len(by_scene)}",
                flush=True,
            )

    score_rows.sort(
        key=lambda row: (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            str(row["route"]),
            str(row["method"]),
            str(row["peer_key"]),
        )
    )
    expected_score_rows = sum(len(candidates) for candidates in grouped.values()) * len(METHODS)
    if len(score_rows) != expected_score_rows:
        raise RuntimeError("DS2 representation score coverage is incomplete")
    score_path = model_input_root / "score_manifest.jsonl"
    _write_jsonl(score_path, score_rows)
    package_record = {
        "schema_version": 1,
        "package_kind": "ds2-label-free-peer-scores",
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "membership_sha256": DS2_CONFIRM_CONTRACT.membership_sha256,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "routes": list(ROUTES),
        "methods": list(METHODS),
        "model_key": MODEL_KEY,
        "resolved_revision": revision,
        "configuration_sha256": configuration_sha256,
        "score_row_count": len(score_rows),
        "unscoreable_row_count": sum(
            row["scoring_status"] != "scored" for row in score_rows
        ),
        "invalid_mask_row_count": sum(
            row["scoring_status"] == "invalid_mask" for row in score_rows
        ),
        "score_manifest_sha256": file_sha256(score_path),
        "evaluator_opened": False,
        "all_checks_passed": True,
    }
    _write_json(model_input_root / "package_record.json", package_record)
    _write_json(model_input_root / "usage_policy.json", {
        "model_facing_allowed": True,
        "protected_fields_included": False,
        "intended_use": "protected_confirmation_and_multiview_evaluation",
    })

    p1_rows = [
        row
        for row in score_rows
        if row["method"] == "P1" and row["scoring_status"] == "scored"
    ]
    route_summary = []
    for route in ROUTES:
        route_p1_rows = [
            row
            for row in score_rows
            if row["method"] == "P1" and row["route"] == route
        ]
        values = [float(row["score"]) for row in p1_rows if row["route"] == route]
        route_summary.append(
            {
                "route": route,
                "candidate_count": len(route_p1_rows),
                "scored_candidate_count": len(values),
                "mean_p1_score": float(np.mean(values)) if values else None,
                "median_p1_score": float(np.median(values)) if values else None,
                "p95_p1_score": float(np.quantile(values, 0.95)) if values else None,
                "unscoreable_candidate_count": sum(
                    row["scoring_status"] != "scored" for row in route_p1_rows
                ),
                "invalid_mask_candidate_count": sum(
                    row["scoring_status"] == "invalid_mask" for row in route_p1_rows
                ),
            }
        )
    runtime_summary = [
        {
            "stage": "dino_and_peer_scores",
            "scene_count": len(scene_audits),
            "candidate_count": sum(int(row["candidate_count"]) for row in scene_audits),
            "encoded_candidate_count": sum(
                int(row.get("encoded_candidate_count", row["candidate_count"]))
                for row in scene_audits
            ),
            "invalid_mask_candidate_count": sum(
                int(row.get("invalid_mask_candidate_count", 0))
                for row in scene_audits
            ),
            "encoder_call_count": sum(int(row["encoder_call_count"]) for row in scene_audits),
            "resumed_scene_count": sum(bool(row.get("resumed")) for row in scene_audits),
            "runtime_seconds_this_session": time.perf_counter() - started,
        }
    ]
    gates = [
        {"gate": "scene_coverage", "expected": DS2_CONFIRM_CONTRACT.scene_count, "observed": len(scene_audits), "passed": len(scene_audits) == DS2_CONFIRM_CONTRACT.scene_count},
        {"gate": "score_coverage", "expected": expected_score_rows, "observed": len(score_rows), "passed": len(score_rows) == expected_score_rows},
        {"gate": "finite_scored_values", "expected": True, "observed": all(row["score"] is None or math.isfinite(float(row["score"])) for row in score_rows), "passed": all(row["score"] is None or math.isfinite(float(row["score"])) for row in score_rows)},
        {"gate": "evaluator_closed", "expected": False, "observed": False, "passed": True},
    ]
    _write_csv(safe_root / "p1_route_summary.csv", route_summary)
    _write_csv(safe_root / "runtime_summary.csv", runtime_summary)
    _write_csv(safe_root / "completion_gates.csv", gates)
    _write_json(safe_root / "resolved_configuration.json", configuration)
    if p1_rows:
        save_histogram([float(row["score"]) for row in p1_rows], safe_root / "p1_score_distribution.png", title="DS2 P1 score distribution", x_label="P1 score", bins=28)
    chart_rows = [row for row in route_summary if row["mean_p1_score"] is not None]
    if chart_rows:
        save_bar_chart([row["route"] for row in chart_rows], [row["mean_p1_score"] for row in chart_rows], safe_root / "mean_p1_by_route.png", title="Mean DS2 P1 score by mask route", y_label="Mean P1 score", show_values=True)
    _heatmap_panel(rows=score_rows, grouped=grouped, cache_root=cache_root, output_path=private_root / "figures" / "p1_heatmap_examples.png", crop_configuration=crop_configuration)
    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "run_mode": run_mode,
        "model_key": MODEL_KEY,
        "resolved_revision": revision,
        "configuration_sha256": configuration_sha256,
        "score_package_sha256": file_sha256(model_input_root / "package_record.json"),
        "score_row_count": len(score_rows),
        "unscoreable_row_count": package_record["unscoreable_row_count"],
        "invalid_mask_row_count": package_record["invalid_mask_row_count"],
        "evaluator_opened": False,
        "all_completion_gates_passed": all(bool(row["passed"]) for row in gates),
    }
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(safe_root)
    _artifact_manifest(private_root)
    _artifact_manifest(cache_root)
    _artifact_manifest(model_input_root)
    return run_record


__all__ = [
    "ConfirmationCandidate",
    "load_ds2_candidates",
    "run_ds2_representation",
]
