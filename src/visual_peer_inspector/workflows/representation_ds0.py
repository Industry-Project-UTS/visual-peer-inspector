"""DS0 frozen-encoder extraction, peer scoring and evaluator-only analysis.

The workflow has a strict boundary. RGB images, masks, encoders and scorers run
before evaluator labels or crosswalks are opened. The label-free score package
is hash-locked first. Evaluator-only results remain private, while safe outputs
contain only aggregate evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.representation_package import (
    RepresentationCandidate,
    load_representation_candidates,
)
from visual_peer_inspector.evaluation.ranking import (
    average_views_within_scene,
    clustered_mean_interval,
    paired_clustered_difference,
    ranking_metrics,
)
from visual_peer_inspector.models.visual_encoder import (
    ContractVisualEncoder,
    EncoderSpecification,
    FrozenVisualEncoder,
    encoder_registry_rows,
    encoder_registry_sha256,
    specification_by_key,
)
from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    PreparedCrop,
    prepare_masked_crop,
)
from visual_peer_inspector.representation.geometry import geometry_feature_matrices
from visual_peer_inspector.scoring.peer import (
    PeerScoreResult,
    b0_scores,
    b1_scores,
    deterministic_ranking,
    p1_scores,
)


SCHEMA_VERSION = 1
STAGE = "ds0_representation_and_peer_scores"
EXPECTED_SCENES = 20
EXPECTED_IMAGES = 400
ROUTES = ("supplied_mask", "reference_box", "automatic_box")
METHODS = ("B0", "B1", "P1", "shape_B1", "layout_B1")
DEFAULT_ENCODERS = (
    "dinov2_small",
    "dinov2_small_registers",
    "dinov3_small",
    "dinov3_small_plus",
    "dinov2_base",
    "resnet50_control",
)


def _route_scope(encoder_key: str) -> tuple[str, ...]:
    """Keep encoder screening fair while avoiding redundant route inference."""

    if encoder_key in {"dinov2_small", "contract_encoder"}:
        return ROUTES
    return ("supplied_mask",)


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
        raise ValueError(f"Cannot write an empty CSV table: {path.name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"CSV rows do not share one schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    return sha256(payload).hexdigest()


def _content_manifest(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact roots cannot contain symbolic links")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _write_artifact_manifest(root: Path) -> dict[str, Any]:
    rows = _content_manifest(root, exclude={"artifact_manifest.json"})
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": _sha256_rows(rows),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    record = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    expected = _content_manifest(root, exclude={"artifact_manifest.json"})
    if record.get("artifacts") != expected:
        raise ValueError("Artifact manifest differs from the files on disk")
    if record.get("manifest_content_sha256") != _sha256_rows(expected):
        raise ValueError("Artifact manifest content hash differs")
    return record


def _safe_configuration(
    *,
    encoder_keys: Sequence[str],
    crop_configuration: CropConfiguration,
    run_mode: str,
) -> dict[str, Any]:
    specifications = []
    for key in encoder_keys:
        if key == "contract_encoder":
            specifications.append(asdict(ContractVisualEncoder.specification))
        else:
            specifications.append(asdict(specification_by_key(key)))
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "run_mode": run_mode,
        "encoder_keys": list(encoder_keys),
        "encoders": specifications,
        "route_scope_by_encoder": {
            key: list(_route_scope(key)) for key in encoder_keys
        },
        "encoder_registry_sha256": encoder_registry_sha256(),
        "crop_configuration": asdict(crop_configuration),
        "crop_configuration_sha256": crop_configuration.sha256,
        "global_input_size": 224,
        "patch_validity_minimum_fraction": 0.50,
        "p1_top_fraction": 0.10,
        "minimum_candidate_count": 3,
        "random_seed": 32_040,
    }


def _prepare_candidates(
    candidates: Sequence[RepresentationCandidate],
    image: Image.Image,
    configuration: CropConfiguration,
) -> list[PreparedCrop]:
    return [
        prepare_masked_crop(image, candidate.mask_rle, configuration=configuration)
        for candidate in candidates
    ]


def _score_rows(
    *,
    candidates: Sequence[RepresentationCandidate],
    encoder_key: str,
    global_features: np.ndarray,
    patch_features: Sequence[np.ndarray],
    valid_patch_masks: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...]]:
    b0 = b0_scores(global_features)
    b1 = b1_scores(global_features)
    if patch_features:
        p1 = p1_scores(patch_features, valid_patch_masks, top_fraction=0.10)
    else:
        p1 = PeerScoreResult(
            "P1", "not_supported", np.full(len(candidates), np.nan, dtype=np.float64)
        )
    shape_features, layout_features = geometry_feature_matrices(
        [candidate.box_xyxy for candidate in candidates],
        [candidate.foreground_pixels for candidate in candidates],
    )
    shape = b1_scores(shape_features)
    layout = b1_scores(layout_features)
    method_results = {
        "B0": b0,
        "B1": b1,
        "P1": p1,
        "shape_B1": shape,
        "layout_B1": layout,
    }
    rows = []
    for method, result in method_results.items():
        if method in {"shape_B1", "layout_B1"} and encoder_key not in {
            "dinov2_small",
            "contract_encoder",
        }:
            continue
        ranks: dict[int, int] = {}
        if result.status == "scored":
            ordering = deterministic_ranking(
                result.scores, [candidate.candidate_id for candidate in candidates]
            )
            ranks = {index: rank for rank, index in enumerate(ordering, start=1)}
        for index, candidate in enumerate(candidates):
            value = float(result.scores[index])
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": candidate.candidate_id,
                    "opaque_scene_id": candidate.opaque_scene_id,
                    "view_id": candidate.view_id,
                    "route": candidate.route,
                    "prompt_key": candidate.prompt_key,
                    "encoder_key": encoder_key if method in {"B0", "B1", "P1"} else "geometry",
                    "method": method,
                    "status": result.status,
                    "score": value if math.isfinite(value) else None,
                    "rank": ranks.get(index),
                    "candidate_count": len(candidates),
                }
            )
    return rows, p1.heatmaps


def _load_encoder(
    key: str,
    *,
    device: str,
    hf_token: str | None,
) -> tuple[Any, dict[str, Any]]:
    if key == "contract_encoder":
        encoder = ContractVisualEncoder()
        return encoder, {
            "encoder_key": key,
            "status": "available",
            "model_id": encoder.specification.model_id,
            "resolved_revision": encoder.revision,
            "family": encoder.specification.family,
            "screen_stage": "contract",
            "gated": False,
            "reason": "",
        }
    specification = specification_by_key(key)
    token = hf_token if specification.gated else None
    try:
        revision = FrozenVisualEncoder.resolve_revision(
            specification.model_id, token=token
        )
        encoder = FrozenVisualEncoder(
            specification,
            revision=revision,
            device=device,
            token=token,
        )
    except Exception as error:
        if specification.gated:
            return None, {
                "encoder_key": key,
                "status": "unavailable",
                "model_id": specification.model_id,
                "resolved_revision": None,
                "family": specification.family,
                "screen_stage": specification.screen_stage,
                "gated": True,
                "reason": f"{type(error).__name__}: gated model access or load failed",
            }
        raise
    return encoder, {
        "encoder_key": key,
        "status": "available",
        "model_id": specification.model_id,
        "resolved_revision": revision,
        "family": specification.family,
        "screen_stage": specification.screen_stage,
        "gated": specification.gated,
        "reason": "",
    }


def _scene_shard_paths(cache_root: Path, private_root: Path, encoder_key: str, scene_id: str) -> tuple[Path, Path, Path]:
    return (
        cache_root / encoder_key / f"{scene_id}.npz",
        private_root / "score_shards" / encoder_key / f"{scene_id}.jsonl",
        cache_root / encoder_key / f"{scene_id}.complete.json",
    )


def _valid_completed_shard(
    feature_path: Path,
    score_path: Path,
    record_path: Path,
    *,
    configuration_sha256: str,
) -> bool:
    if not all(path.is_file() for path in (feature_path, score_path, record_path)):
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return (
            record.get("configuration_sha256") == configuration_sha256
            and record.get("feature_sha256") == file_sha256(feature_path)
            and record.get("score_sha256") == file_sha256(score_path)
            and record.get("status") == "complete"
        )
    except Exception:
        return False


def _encoder_scene_shard(
    *,
    scene_id: str,
    scene_groups: Sequence[tuple[tuple[str, str, str], Sequence[RepresentationCandidate]]],
    encoder: Any,
    encoder_key: str,
    crop_configuration: CropConfiguration,
    configuration_sha256: str,
    cache_root: Path,
    private_root: Path,
) -> dict[str, Any]:
    feature_path, score_path, record_path = _scene_shard_paths(
        cache_root, private_root, encoder_key, scene_id
    )
    if _valid_completed_shard(
        feature_path,
        score_path,
        record_path,
        configuration_sha256=configuration_sha256,
    ):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["resumed"] = True
        return record
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    score_rows: list[dict[str, Any]] = []
    identifiers: list[str] = []
    global_features: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    heatmap_shapes: list[tuple[int, int]] = []
    crop_hashes: list[str] = []
    group_count = 0
    for (_scene, _view, _route), candidates in scene_groups:
        image = Image.open(candidates[0].image_path).convert("RGB")
        prepared = _prepare_candidates(candidates, image, crop_configuration)
        encoded = encoder.encode([crop.rgb for crop in prepared])
        if len(encoded.global_features) != len(candidates):
            raise RuntimeError("Encoder returned the wrong number of global features")
        if not np.all(np.isfinite(encoded.global_features)):
            raise RuntimeError("Encoder returned non-finite global features")
        norms = np.linalg.norm(encoded.global_features, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError("Global encoder features are not unit normalized")
        valid_masks: list[np.ndarray] = []
        if encoded.patch_features:
            if encoded.patch_grid_size is None:
                raise RuntimeError("Patch features require a declared grid size")
            valid_masks = [
                crop.patch_validity(encoded.patch_grid_size, minimum_fraction=0.50)
                for crop in prepared
            ]
        rows, group_heatmaps = _score_rows(
            candidates=candidates,
            encoder_key=encoder_key,
            global_features=encoded.global_features,
            patch_features=encoded.patch_features,
            valid_patch_masks=valid_masks,
        )
        score_rows.extend(rows)
        identifiers.extend(candidate.candidate_id for candidate in candidates)
        global_features.extend(encoded.global_features)
        crop_hashes.extend(crop.sha256 for crop in prepared)
        if group_heatmaps:
            heatmaps.extend(group_heatmaps)
            heatmap_shapes.extend(value.shape for value in group_heatmaps)
        else:
            heatmaps.extend(np.empty((0, 0), dtype=np.float32) for _ in candidates)
            heatmap_shapes.extend((0, 0) for _ in candidates)
        group_count += 1
    arrays: dict[str, Any] = {
        "candidate_ids": np.asarray(identifiers, dtype="U32"),
        "global_features": np.asarray(global_features, dtype=np.float16),
        "crop_sha256": np.asarray(crop_hashes, dtype="U64"),
        "heatmap_shapes": np.asarray(heatmap_shapes, dtype=np.int16),
    }
    non_empty_heatmaps = [value for value in heatmaps if value.size]
    if non_empty_heatmaps:
        shape = non_empty_heatmaps[0].shape
        if any(value.shape != shape for value in non_empty_heatmaps):
            raise RuntimeError("One encoder produced inconsistent patch grids")
        arrays["p1_heatmaps"] = np.stack(
            [
                value if value.size else np.full(shape, np.nan, dtype=np.float64)
                for value in heatmaps
            ]
        ).astype(np.float16)
    temporary_feature = feature_path.with_suffix(".npz.incomplete")
    with temporary_feature.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_feature.replace(feature_path)
    temporary_score = score_path.with_suffix(".jsonl.incomplete")
    _write_jsonl(temporary_score, score_rows)
    temporary_score.replace(score_path)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "encoder_key": encoder_key,
        "scene_id": scene_id,
        "configuration_sha256": configuration_sha256,
        "candidate_count": len(identifiers),
        "score_row_count": len(score_rows),
        "group_count": group_count,
        "feature_sha256": file_sha256(feature_path),
        "score_sha256": file_sha256(score_path),
        "runtime_seconds": time.perf_counter() - started,
        "resumed": False,
    }
    _write_json(record_path, record)
    return record


def _make_crop_panel(
    grouped: Mapping[tuple[str, str, str], Sequence[RepresentationCandidate]],
    output_path: Path,
    configuration: CropConfiguration,
) -> None:
    first_scene, first_view = sorted({(scene, view) for scene, view, _route in grouped})[0]
    tiles = []
    labels = []
    for route in ROUTES:
        candidates = grouped[(first_scene, first_view, route)]
        image = Image.open(candidates[0].image_path).convert("RGB")
        for candidate, crop in zip(
            candidates[:3], _prepare_candidates(candidates[:3], image, configuration), strict=True
        ):
            tiles.append(Image.fromarray(crop.rgb, mode="RGB"))
            labels.append(f"{route}\n{candidate.prompt_key}")
    width = 224 * 3
    height = 254 * math.ceil(len(tiles) / 3)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        x = (index % 3) * 224
        y = (index // 3) * 254
        panel.paste(tile, (x, y))
        draw.text((x + 5, y + 227), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def _make_p1_panel(
    *,
    encoder_key: str,
    cache_root: Path,
    grouped: Mapping[tuple[str, str, str], Sequence[RepresentationCandidate]],
    output_path: Path,
    configuration: CropConfiguration,
) -> None:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidates in grouped.values()
        for candidate in candidates
    }
    candidates_for_panel: list[tuple[float, str, np.ndarray]] = []
    for shard_path in sorted((cache_root / encoder_key).glob("*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            if "p1_heatmaps" not in shard:
                continue
            identifiers = shard["candidate_ids"].tolist()
            maps = shard["p1_heatmaps"].astype(np.float32)
            for identifier, heatmap in zip(identifiers, maps, strict=True):
                finite = heatmap[np.isfinite(heatmap)]
                if finite.size:
                    candidates_for_panel.append(
                        (float(np.mean(finite)), str(identifier), heatmap)
                    )
        if len(candidates_for_panel) >= 30:
            break
    if not candidates_for_panel:
        return
    selected = sorted(candidates_for_panel, key=lambda item: (-item[0], item[1]))[:6]
    header_height = 42
    tile_height = 258
    panel = Image.new("RGB", (224 * 3, header_height + tile_height * 2), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "P1 coarse mismatch examples", fill="black")
    draw.text((8, 24), "Red indicates higher feature mismatch, not an exact defect boundary", fill="black")
    for index, (_mean, identifier, heatmap) in enumerate(selected):
        candidate = candidate_by_id[identifier]
        image = Image.open(candidate.image_path).convert("RGB")
        crop = prepare_masked_crop(
            image, candidate.mask_rle, configuration=configuration
        )
        finite = heatmap[np.isfinite(heatmap)]
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        scaled = np.zeros_like(heatmap, dtype=np.float32)
        if upper > lower:
            scaled = (heatmap - lower) / (upper - lower)
        scaled[~np.isfinite(scaled)] = 0.0
        heat = np.asarray(
            Image.fromarray(np.uint8(scaled * 255), mode="L").resize(
                (224, 224), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        rgb = crop.rgb.astype(np.float32)
        alpha = (0.10 + 0.55 * heat)[:, :, None] * crop.foreground[:, :, None]
        colour = np.zeros_like(rgb)
        colour[:, :, 0] = 255
        colour[:, :, 1] = 80 * (1.0 - heat)
        overlay = np.uint8(np.clip(rgb * (1.0 - alpha) + colour * alpha, 0, 255))
        x = (index % 3) * 224
        y = header_height + (index // 3) * tile_height
        panel.paste(Image.fromarray(overlay, mode="RGB"), (x, y))
        draw.text(
            (x + 4, y + 228),
            f"{candidate.route} {candidate.prompt_key}",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def _make_private_pca(
    *,
    encoder_key: str,
    cache_root: Path,
    grouped: Mapping[tuple[str, str, str], Sequence[RepresentationCandidate]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    route_by_candidate = {
        candidate.candidate_id: candidate.route
        for candidates in grouped.values()
        for candidate in candidates
    }
    identifiers = []
    features = []
    for shard_path in sorted((cache_root / encoder_key).glob("*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            identifiers.extend(map(str, shard["candidate_ids"].tolist()))
            features.append(shard["global_features"].astype(np.float32))
    if not features:
        return
    matrix = np.concatenate(features)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    points = left[:, :2] * singular[:2]
    colours = {
        "supplied_mask": "#2f6f9f",
        "reference_box": "#4c9f70",
        "automatic_box": "#d47b32",
    }
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for route in ROUTES:
        indexes = [
            index for index, identifier in enumerate(identifiers)
            if route_by_candidate[identifier] == route
        ]
        axis.scatter(
            points[indexes, 0], points[indexes, 1], s=8, alpha=0.35,
            label=route, color=colours[route],
        )
    axis.set_title(f"Global feature PCA: {encoder_key}")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_label_free_ds0(
    *,
    rgb_root: Path,
    mask_route_root: Path,
    output_root: Path,
    encoder_keys: Sequence[str] = DEFAULT_ENCODERS,
    device: str = "cuda",
    hf_token: str | None = None,
    run_mode: str = "model_evidence",
    crop_configuration: CropConfiguration = CropConfiguration(),
) -> dict[str, Any]:
    """Create a frozen score package before any evaluator input is opened."""

    output_root = output_root.expanduser().resolve()
    model_input_root = output_root / "model_input"
    private_root = output_root / "private"
    cache_root = output_root / "cache"
    safe_root = output_root / "safe"
    if (model_input_root / "package_record.json").exists():
        raise FileExistsError("A completed immutable score package already exists")
    for root in (model_input_root, private_root, cache_root, safe_root):
        if root.is_symlink():
            raise ValueError("Representation output roots cannot be symbolic links")
        root.mkdir(parents=True, exist_ok=True)
    grouped, input_audit = load_representation_candidates(rgb_root, mask_route_root)
    configuration = _safe_configuration(
        encoder_keys=encoder_keys,
        crop_configuration=crop_configuration,
        run_mode=run_mode,
    )
    configuration_sha256 = sha256(canonical_json_bytes(configuration)).hexdigest()
    _write_json(safe_root / "resolved_configuration.json", configuration)
    _write_json(safe_root / "input_audit.json", input_audit.to_dict())
    _write_json(safe_root / "model_registry.json", {"models": encoder_registry_rows()})
    _make_crop_panel(grouped, private_root / "figures" / "three_route_crop_panel.png", crop_configuration)

    by_scene: dict[str, list[tuple[tuple[str, str, str], Sequence[RepresentationCandidate]]]] = defaultdict(list)
    for key, candidates in sorted(grouped.items()):
        by_scene[key[0]].append((key, candidates))
    if len(by_scene) != EXPECTED_SCENES:
        raise ValueError("Representation input must contain exactly 20 scenes")

    availability_rows = []
    shard_records = []
    for encoder_key in encoder_keys:
        encoder_routes = _route_scope(encoder_key)
        encoder, availability = _load_encoder(
            encoder_key, device=device, hf_token=hf_token
        )
        if encoder is None:
            availability.update(
                {
                    "feature_dimension": None,
                    "patch_grid_size": None,
                    "repeat_minimum_cosine": None,
                    "repeat_maximum_absolute_delta": None,
                    "repeatability_passed": False,
                    "peak_gpu_memory_bytes": 0,
                }
            )
            availability_rows.append(availability)
            continue
        first_key = next(
            key for key in sorted(grouped) if key[2] in encoder_routes
        )
        first_candidates = grouped[first_key]
        first_image = Image.open(first_candidates[0].image_path).convert("RGB")
        first_prepared = _prepare_candidates(
            first_candidates, first_image, crop_configuration
        )
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        first_batch = encoder.encode([crop.rgb for crop in first_prepared])
        repeated_batch = encoder.encode([crop.rgb for crop in first_prepared])
        repeated_cosines = np.sum(
            first_batch.global_features * repeated_batch.global_features, axis=1
        )
        repeated_delta = float(
            np.max(
                np.abs(
                    first_batch.global_features - repeated_batch.global_features
                )
            )
        )
        availability.update(
            {
                "route_scope": "+".join(encoder_routes),
                "feature_dimension": first_batch.feature_dimension,
                "patch_grid_size": first_batch.patch_grid_size,
                "repeat_minimum_cosine": float(np.min(repeated_cosines)),
                "repeat_maximum_absolute_delta": repeated_delta,
                "repeatability_passed": bool(
                    np.min(repeated_cosines) >= 0.9999 and repeated_delta <= 1e-3
                ),
                "peak_gpu_memory_bytes": 0,
            }
        )
        encoder_configuration_sha256 = sha256(
            canonical_json_bytes(
                {
                    "pipeline_configuration_sha256": configuration_sha256,
                    "encoder_key": encoder_key,
                    "model_id": availability["model_id"],
                    "resolved_revision": availability["resolved_revision"],
                }
            )
        ).hexdigest()
        for scene_id, scene_groups in sorted(by_scene.items()):
            selected_scene_groups = [
                value for value in scene_groups if value[0][2] in encoder_routes
            ]
            shard_records.append(
                _encoder_scene_shard(
                    scene_id=scene_id,
                    scene_groups=selected_scene_groups,
                    encoder=encoder,
                    encoder_key=encoder_key,
                    crop_configuration=crop_configuration,
                    configuration_sha256=encoder_configuration_sha256,
                    cache_root=cache_root,
                    private_root=private_root,
                )
            )
        try:
            if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
                availability["peak_gpu_memory_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
        except (AttributeError, RuntimeError):
            availability["peak_gpu_memory_bytes"] = 0
        availability_rows.append(availability)
        del encoder
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    availability_columns = (
        "encoder_key", "status", "model_id", "resolved_revision", "family",
        "screen_stage", "gated", "reason", "feature_dimension", "patch_grid_size",
        "repeat_minimum_cosine", "repeat_maximum_absolute_delta",
        "repeatability_passed", "peak_gpu_memory_bytes", "route_scope",
    )
    availability_rows = [
        {column: row.get(column) for column in availability_columns}
        for row in availability_rows
    ]
    available_keys = [
        row["encoder_key"] for row in availability_rows if row["status"] == "available"
    ]
    if not available_keys:
        raise RuntimeError("No encoder completed the DS0 representation run")
    score_rows = []
    for encoder_key in available_keys:
        for scene_id in sorted(by_scene):
            _feature, score_path, _record = _scene_shard_paths(
                cache_root, private_root, encoder_key, scene_id
            )
            score_rows.extend(read_jsonl(score_path))
    score_rows.sort(
        key=lambda row: (
            row["encoder_key"], row["opaque_scene_id"], row["view_id"],
            row["route"], row["method"], row["candidate_id"],
        )
    )
    _write_jsonl(model_input_root / "score_manifest.jsonl", score_rows)
    _write_json(
        safe_root / "resolved_model_revisions.json",
        {"encoders": availability_rows},
    )
    _write_json(
        model_input_root / "usage_policy.json",
        {
            "schema_version": SCHEMA_VERSION,
            "model_facing_allowed": True,
            "protected_fields_included": False,
            "intended_use": "view_fusion_and_calibration",
            "warning": "Prompt keys are route-local and are not cross-view identities.",
        },
    )
    package_record = {
        "schema_version": SCHEMA_VERSION,
        "package_kind": "ds0-label-free-peer-scores",
        "stage": STAGE,
        "run_mode": run_mode,
        "scene_count": EXPECTED_SCENES,
        "image_count": EXPECTED_IMAGES,
        "score_row_count": len(score_rows),
        "available_encoders": available_keys,
        "membership_sha256": input_audit.membership_sha256,
        "input_join_sha256": input_audit.join_sha256,
        "configuration_sha256": configuration_sha256,
        "score_manifest_sha256": file_sha256(model_input_root / "score_manifest.jsonl"),
        "all_checks_passed": True,
    }
    _write_json(model_input_root / "package_record.json", package_record)
    _write_artifact_manifest(model_input_root)

    _write_csv(safe_root / "encoder_availability.csv", availability_rows)
    availability_by_key = {row["encoder_key"]: row for row in availability_rows}
    runtime_rows = []
    for encoder_key in available_keys:
        records = [row for row in shard_records if row["encoder_key"] == encoder_key]
        candidate_count = sum(int(row["candidate_count"]) for row in records)
        runtime_seconds = sum(float(row["runtime_seconds"]) for row in records)
        runtime_rows.append(
            {
                "encoder_key": encoder_key,
                "scene_count": len(records),
                "candidate_count": candidate_count,
                "runtime_seconds": runtime_seconds,
                "milliseconds_per_candidate": 1000.0 * runtime_seconds / candidate_count,
                "resumed_scene_count": sum(bool(row["resumed"]) for row in records),
                "cache_bytes": sum(
                    _scene_shard_paths(cache_root, private_root, encoder_key, row["scene_id"])[0].stat().st_size
                    for row in records
                ),
                "peak_gpu_memory_bytes": int(
                    availability_by_key[encoder_key]["peak_gpu_memory_bytes"] or 0
                ),
            }
        )
    _write_csv(safe_root / "runtime_summary.csv", runtime_rows)
    _write_json(
        safe_root / "cache_summary.json",
        {
            "configuration_sha256": configuration_sha256,
            "scene_shard_count": len(shard_records),
            "resumed_scene_shard_count": sum(bool(row["resumed"]) for row in shard_records),
            "cache_is_private": True,
            "global_features_retained": True,
            "patch_features_retained": False,
            "p1_heatmaps_retained": True,
        },
    )
    for encoder_key in available_keys:
        encoder_record = availability_by_key[encoder_key]
        if encoder_record["patch_grid_size"] is not None:
            _make_p1_panel(
                encoder_key=encoder_key,
                cache_root=cache_root,
                grouped=grouped,
                output_path=(
                    private_root / "figures" / f"p1_heatmap_examples_{encoder_key}.png"
                ),
                configuration=crop_configuration,
            )
        _make_private_pca(
            encoder_key=encoder_key,
            cache_root=cache_root,
            grouped=grouped,
            output_path=(
                private_root / "figures" / f"global_feature_pca_{encoder_key}.png"
            ),
        )
    _write_artifact_manifest(private_root)
    _write_artifact_manifest(cache_root)
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "label_free_scores_complete",
        "run_mode": run_mode,
        "available_encoder_count": len(available_keys),
        "requested_encoder_count": len(encoder_keys),
        "score_package_sha256": file_sha256(model_input_root / "package_record.json"),
        "configuration_sha256": configuration_sha256,
        "evaluator_opened": False,
    }
    _write_json(safe_root / "label_free_run_record.json", run_record)
    return run_record


def _crosswalk_maps(evaluator_root: Path) -> tuple[dict[tuple[str, str, str, str], str | None], dict[str, dict[str, Any]]]:
    scene_rows = read_jsonl(
        evaluator_root / "scene_attributes" / "protected_scene_attributes.jsonl"
    )
    scenes = {str(row["opaque_scene_id"]): row for row in scene_rows}
    if len(scenes) != EXPECTED_SCENES:
        raise ValueError("Evaluator scene attributes do not contain DS0")
    crosswalk: dict[tuple[str, str, str, str], str | None] = {}
    for row in read_jsonl(
        evaluator_root / "sam_crosswalk" / "sam_evaluator_crosswalk.jsonl"
    ):
        route = str(row["route"])
        key = (
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
            route,
            str(row["prompt_key"]),
        )
        value = row.get("matched_opaque_object_id")
        crosswalk[key] = None if value is None else str(value)
    return crosswalk, scenes


def _evaluate_scores(
    score_rows: Sequence[Mapping[str, Any]],
    crosswalk: Mapping[tuple[str, str, str, str], str | None],
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    coverage_rows = []
    for row in score_rows:
        if row["method"] in {"shape_B1", "layout_B1"} and row["encoder_key"] != "geometry":
            raise ValueError("Geometry score rows use an invalid encoder key")
        key = (
            str(row["encoder_key"]), str(row["route"]), str(row["method"]),
            str(row["opaque_scene_id"]), str(row["view_id"]),
        )
        groups[key].append(row)
    evaluation_rows = []
    for (encoder_key, route, method, scene_id, view_id), values in sorted(groups.items()):
        statuses = {str(row["status"]) for row in values}
        if statuses != {"scored"}:
            coverage_rows.append(
                {
                    "encoder_key": encoder_key,
                    "route": route,
                    "method": method,
                    "opaque_scene_id": scene_id,
                    "view_id": view_id,
                    "status": sorted(statuses)[0],
                    "candidate_count": len(values),
                }
            )
            continue
        ordering = sorted(
            values,
            key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
        )
        scene = scenes[scene_id]
        labels = {str(key): bool(value) for key, value in scene["labels_by_object_id"].items()}
        mapped_ids: list[str | None] = []
        ranked_labels = []
        for row in ordering:
            lookup_route = "reference_box" if route == "supplied_mask" else route
            crosswalk_key = (scene_id, view_id, lookup_route, str(row["prompt_key"]))
            if crosswalk_key not in crosswalk:
                raise ValueError("A scored candidate is missing from the evaluator crosswalk")
            object_id = crosswalk[crosswalk_key]
            mapped_ids.append(object_id)
            ranked_labels.append(False if object_id is None else labels[object_id])
        positive_ids = [key for key, value in labels.items() if value]
        metrics = ranking_metrics(
            ranked_labels,
            positive_total=int(scene["anomaly_count"]),
            ranked_object_ids=mapped_ids,
            positive_object_ids=positive_ids,
        )
        evaluation_rows.append(
            {
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "opaque_asset_group_id": str(scene["opaque_asset_group_id"]),
                "category": str(scene["category"]),
                "anomaly_families": "+".join(sorted(map(str, scene["anomaly_families"]))),
                "anomaly_count": int(scene["anomaly_count"]),
                "encoder_key": encoder_key,
                "route": route,
                "method": method,
                "candidate_count": len(ordering),
                "top_candidate_id": str(ordering[0]["candidate_id"]),
                "top_candidate_object_id": mapped_ids[0],
                "unmatched_candidate_count": sum(value is None for value in mapped_ids),
                "missed_anomaly_count": int(scene["anomaly_count"]) - sum(ranked_labels),
                **metrics.to_dict(),
            }
        )
        coverage_rows.append(
            {
                "encoder_key": encoder_key,
                "route": route,
                "method": method,
                "opaque_scene_id": scene_id,
                "view_id": view_id,
                "status": "scored",
                "candidate_count": len(values),
            }
        )
    return evaluation_rows, coverage_rows


def _aggregate_rows(
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[tuple(str(row[column]) for column in group_columns)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        row: dict[str, Any] = {column: value for column, value in zip(group_columns, key, strict=True)}
        row["scene_count"] = len(values)
        for metric in ("top1", "reciprocal_rank", "average_precision", "recall_at_2", "exact_set"):
            row[metric] = float(np.mean([float(value[metric]) for value in values]))
        output.append(row)
    return output


def _bootstrap_rows(scene_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[(str(row["encoder_key"]), str(row["route"]), str(row["method"]))].append(row)
    output = []
    for (encoder_key, route, method), values in sorted(groups.items()):
        interval = clustered_mean_interval(
            values,
            metric="average_precision",
            repetitions=10_000,
            seed=32_040,
        )
        output.append(
            {
                "encoder_key": encoder_key,
                "route": route,
                "method": method,
                "metric": "average_precision",
                **interval,
            }
        )
    return output


def _paired_difference_rows(
    scene_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare bounded DS0 candidates against like-for-like anchor results."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        groups[(str(row["encoder_key"]), str(row["route"]), str(row["method"]))].append(row)

    comparisons = []
    preferred_anchor = ("dinov2_small", "supplied_mask", "B0")
    anchor_key = preferred_anchor
    if anchor_key not in groups:
        fallbacks = sorted(
            key
            for key in groups
            if key[1:] == ("supplied_mask", "B0") and key[0] != "geometry"
        )
        if not fallbacks:
            return comparisons
        anchor_key = fallbacks[0]
    if anchor_key not in groups:
        return comparisons
    baseline = groups[anchor_key]
    candidates: list[tuple[str, tuple[str, str, str]]] = []
    candidates.extend(
        ("encoder_vs_anchor", key)
        for key in sorted(groups)
        if key[1:] == ("supplied_mask", "B0") and key != anchor_key and key[0] != "geometry"
    )
    candidates.extend(
        ("scorer_vs_anchor", key)
        for key in sorted(groups)
        if key[0] == anchor_key[0] and key[1] == "supplied_mask" and key[2] in {"B1", "P1"}
    )
    candidates.extend(
        ("route_vs_anchor", key)
        for key in sorted(groups)
        if key[0] == anchor_key[0] and key[2] == "B0" and key[1] in {"reference_box", "automatic_box"}
    )
    for family, key in candidates:
        interval = paired_clustered_difference(
            baseline,
            groups[key],
            metric="average_precision",
            repetitions=10_000,
            seed=32_040,
        )
        comparisons.append(
            {
                "comparison_family": family,
                "candidate_encoder": key[0],
                "candidate_route": key[1],
                "candidate_method": key[2],
                "baseline_encoder": anchor_key[0],
                "baseline_route": anchor_key[1],
                "baseline_method": anchor_key[2],
                "metric": "average_precision",
                **interval,
            }
        )
    return comparisons


def _plot_bar(rows: Sequence[Mapping[str, Any]], x: str, y: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row[x]) for row in rows]
    values = [float(row[y]) for row in rows]
    figure, axis = plt.subplots(figsize=(max(7, len(rows) * 0.8), 4.8))
    axis.bar(range(len(values)), values, color="#2f6f9f")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_ylabel(y.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _model_agreement_rows(
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    predictions: dict[str, dict[tuple[str, str], str | None]] = defaultdict(dict)
    for row in evaluation_rows:
        if row["route"] == "supplied_mask" and row["method"] == "B0" and row["encoder_key"] != "geometry":
            predictions[str(row["encoder_key"])][
                (str(row["opaque_scene_id"]), str(row["view_id"]))
            ] = None if row["top_candidate_object_id"] is None else str(row["top_candidate_object_id"])
    rows = []
    for left in sorted(predictions):
        for right in sorted(predictions):
            keys = sorted(set(predictions[left]) & set(predictions[right]))
            if not keys:
                continue
            rows.append(
                {
                    "encoder_a": left,
                    "encoder_b": right,
                    "comparable_view_count": len(keys),
                    "top_candidate_agreement": float(
                        np.mean(
                            [predictions[left][key] == predictions[right][key] for key in keys]
                        )
                    ),
                }
            )
    return rows


def _plot_agreement(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    encoders = sorted({str(row["encoder_a"]) for row in rows})
    matrix = np.full((len(encoders), len(encoders)), np.nan)
    positions = {value: index for index, value in enumerate(encoders)}
    for row in rows:
        matrix[positions[str(row["encoder_a"])], positions[str(row["encoder_b"])]] = float(
            row["top_candidate_agreement"]
        )
    figure, axis = plt.subplots(figsize=(max(5, len(encoders) * 1.1), 4.8))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    axis.set_xticks(range(len(encoders)), encoders, rotation=35, ha="right")
    axis.set_yticks(range(len(encoders)), encoders)
    axis.set_title("Supplied-mask B0 top-candidate agreement")
    for row in range(len(encoders)):
        for column in range(len(encoders)):
            if np.isfinite(matrix[row, column]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Agreement")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_efficiency(
    encoder_summary: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    runtime_by_encoder = {str(row["encoder_key"]): row for row in runtime_rows}
    points = [row for row in encoder_summary if str(row["encoder_key"]) in runtime_by_encoder]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in points:
        key = str(row["encoder_key"])
        runtime = runtime_by_encoder[key]
        x = float(runtime["milliseconds_per_candidate"])
        y = float(row["average_precision"])
        memory = max(20.0, float(runtime["peak_gpu_memory_bytes"]) / 50_000_000)
        axis.scatter(x, y, s=memory, alpha=0.65, color="#2f6f9f")
        axis.annotate(key, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Feature extraction and scoring time per object crop (milliseconds)")
    axis.set_ylabel("Mean scene AP")
    axis.set_title("DS0 supplied-mask B0 accuracy and normalized runtime")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_coverage(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row["encoder_key"] == "dinov2_small" and row["method"] == "B0"
    ]
    if not selected:
        selected = [row for row in rows if row["method"] == "B0"]
    labels = [f"{row['route']}\n{row['status']}" for row in selected]
    values = [int(row["view_count"]) for row in selected]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(range(len(values)), values, color="#4c9f70")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("View count")
    axis.set_title("Rankable coverage by mask route")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_private_ds0_evaluation(
    *,
    representation_root: Path,
    evaluator_root: Path,
) -> dict[str, Any]:
    """Open private evaluator data only after the score package is immutable."""

    representation_root = representation_root.expanduser().resolve()
    safe_root = representation_root / "safe"
    private_root = representation_root / "private"
    model_input_root = representation_root / "model_input"
    package_record = json.loads(
        (model_input_root / "package_record.json").read_text(encoding="utf-8")
    )
    verify_artifact_manifest(model_input_root)
    score_package_sha256 = file_sha256(model_input_root / "package_record.json")
    if not package_record.get("all_checks_passed"):
        raise ValueError("Label-free score package did not pass its gates")
    crosswalk, scenes = _crosswalk_maps(evaluator_root.expanduser().resolve())
    score_rows = read_jsonl(model_input_root / "score_manifest.jsonl")
    evaluation_rows, coverage_rows = _evaluate_scores(score_rows, crosswalk, scenes)
    if not evaluation_rows:
        raise RuntimeError("No rankable evaluator records were produced")
    _write_jsonl(private_root / "evaluator" / "per_view_metrics.jsonl", evaluation_rows)
    _write_jsonl(private_root / "evaluator" / "coverage_records.jsonl", coverage_rows)
    scene_rows = average_views_within_scene(evaluation_rows)
    _write_jsonl(private_root / "evaluator" / "scene_metrics.jsonl", scene_rows)
    _write_csv(
        private_root / "evaluator" / "subgroup_summary.csv",
        _aggregate_rows(
            scene_rows,
            group_columns=("encoder_key", "route", "method", "anomaly_count"),
        ),
    )

    encoder_summary = _aggregate_rows(
        [
            row for row in scene_rows
            if row["route"] == "supplied_mask" and row["method"] == "B0"
        ],
        group_columns=("encoder_key",),
    )
    scorer_summary = _aggregate_rows(
        [row for row in scene_rows if row["route"] == "supplied_mask"],
        group_columns=("encoder_key", "method"),
    )
    route_summary = _aggregate_rows(
        scene_rows,
        group_columns=("encoder_key", "route", "method"),
    )
    scene_type_summary = _aggregate_rows(
        scene_rows,
        group_columns=("encoder_key", "route", "method", "anomaly_count"),
    )
    geometry_summary = _aggregate_rows(
        [row for row in scene_rows if row["encoder_key"] == "geometry"],
        group_columns=("route", "method"),
    )
    view_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        view_groups[(str(row["encoder_key"]), str(row["route"]), str(row["method"]), str(row["view_id"]))].append(row)
    view_summary = []
    for key, values in sorted(view_groups.items()):
        view_summary.append(
            {
                "encoder_key": key[0],
                "route": key[1],
                "method": key[2],
                "view_id": key[3],
                "scene_count": len(values),
                "average_precision": float(np.mean([float(row["average_precision"]) for row in values])),
                "top1": float(np.mean([float(row["top1"]) for row in values])),
            }
        )
    clustered_intervals = _bootstrap_rows(scene_rows)
    paired_differences = _paired_difference_rows(scene_rows)
    for name, rows in (
        ("encoder_summary.csv", encoder_summary),
        ("scorer_summary.csv", scorer_summary),
        ("route_summary.csv", route_summary),
        ("scene_type_summary.csv", scene_type_summary),
        ("geometry_summary.csv", geometry_summary),
        ("view_summary.csv", view_summary),
        ("clustered_intervals.csv", clustered_intervals),
        ("paired_differences.csv", paired_differences),
    ):
        _write_csv(safe_root / name, rows)

    coverage_counts: dict[tuple[str, str, str, str], int] = Counter(
        (str(row["encoder_key"]), str(row["route"]), str(row["method"]), str(row["status"]))
        for row in coverage_rows
    )
    coverage_summary = [
        {
            "encoder_key": key[0],
            "route": key[1],
            "method": key[2],
            "status": key[3],
            "view_count": count,
        }
        for key, count in sorted(coverage_counts.items())
    ]
    _write_csv(safe_root / "coverage_summary.csv", coverage_summary)
    agreement_rows = _model_agreement_rows(evaluation_rows)
    _write_csv(safe_root / "model_agreement.csv", agreement_rows)

    _plot_bar(
        encoder_summary,
        "encoder_key",
        "average_precision",
        safe_root / "figures" / "encoder_b0_scene_ap.png",
        "DS0 supplied-mask B0 by encoder",
    )
    non_geometry_encoders = [
        str(row["encoder_key"]) for row in encoder_summary if row["encoder_key"] != "geometry"
    ]
    anchor_encoder_key = (
        "dinov2_small" if "dinov2_small" in non_geometry_encoders else non_geometry_encoders[0]
    )
    anchor_scorers = [
        row for row in scorer_summary if row["encoder_key"] == anchor_encoder_key
    ]
    _plot_bar(
        anchor_scorers,
        "method",
        "average_precision",
        safe_root / "figures" / "anchor_scorer_comparison.png",
        "DS0 supplied-mask peer scorer comparison",
    )
    anchor_routes = [
        row for row in route_summary
        if row["encoder_key"] == anchor_encoder_key and row["method"] == "B0"
    ]
    _plot_bar(
        anchor_routes,
        "route",
        "average_precision",
        safe_root / "figures" / "anchor_route_comparison.png",
        "DS0 B0 route comparison",
    )
    _plot_bar(
        [row for row in scene_type_summary if row["encoder_key"] == anchor_encoder_key and row["route"] == "supplied_mask" and row["method"] == "B0"],
        "anomaly_count",
        "average_precision",
        safe_root / "figures" / "one_vs_two_anomaly_scene_ap.png",
        f"{anchor_encoder_key} supplied-mask B0: one vs two anomalies",
    )
    _plot_agreement(
        agreement_rows,
        safe_root / "figures" / "model_agreement.png",
    )
    with (safe_root / "runtime_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        runtime_rows = list(csv.DictReader(handle))
    _plot_efficiency(
        encoder_summary,
        runtime_rows,
        safe_root / "figures" / "encoder_efficiency.png",
    )
    _plot_coverage(
        coverage_summary,
        safe_root / "figures" / "rankable_coverage.png",
    )
    geometry_plot_rows = []
    anchor_appearance = next(
        (
            row for row in route_summary
            if row["encoder_key"] == "dinov2_small"
            and row["route"] == "supplied_mask"
            and row["method"] == "B0"
        ),
        None,
    )
    if anchor_appearance is None:
        anchor_appearance = next(
            (
                row for row in route_summary
                if row["encoder_key"] != "geometry"
                and row["route"] == "supplied_mask"
                and row["method"] == "B0"
            ),
            None,
        )
    if anchor_appearance is not None:
        geometry_plot_rows.append(
            {"branch": "appearance_B0", "average_precision": anchor_appearance["average_precision"]}
        )
    geometry_plot_rows.extend(
        {
            "branch": str(row["method"]),
            "average_precision": row["average_precision"],
        }
        for row in geometry_summary
        if row["route"] == "supplied_mask"
    )
    _plot_bar(
        geometry_plot_rows,
        "branch",
        "average_precision",
        safe_root / "figures" / "geometry_vs_appearance.png",
        "Appearance and separate geometry branches",
    )

    metric_definitions = {
        "B0": "Cosine distance to the normalized leave-one-out peer mean.",
        "B1": "Median cosine distance to all peer objects in the same view.",
        "P1": "Median symmetric top-tail patch mismatch to peer objects.",
        "shape_B1": "Median peer distance using shape features only.",
        "layout_B1": "Median peer distance using centroid and peer-spacing features only.",
        "average_precision": "Ranking AP with missed visible anomalies retained in the denominator.",
        "claim_boundary": "DS0 validates mechanics and route effects. It does not select the final encoder.",
    }
    _write_json(safe_root / "metric_definitions.json", metric_definitions)
    decision = {
        "decision": "retain_dinov2_small_anchor_pending_ds1",
        "final_encoder_selected": False,
        "reason": "DS0 is a 20-scene development smoke cohort. Encoder selection requires the 100-scene DS1 supplied-mask screen.",
        "available_encoders": package_record["available_encoders"],
        "dino3_ds0_comparison_completed": {
            "dinov3_small",
            "dinov3_small_plus",
        }.issubset(set(package_record["available_encoders"])),
        "ds1_promotion_rule": {
            "minimum_mean_scene_ap_gain": 0.01,
            "paired_ci_lower_must_exceed": 0.0,
            "maximum_declared_subgroup_loss": 0.03,
            "maximum_runtime_ratio": 2.0,
        },
    }
    _write_json(safe_root / "promotion_decision.json", decision)
    requested_encoders = json.loads(
        (safe_root / "resolved_configuration.json").read_text(encoding="utf-8")
    )["encoder_keys"]
    available_encoders = set(package_record["available_encoders"])
    required_ungated = {
        key
        for key in requested_encoders
        if key == "contract_encoder" or not specification_by_key(key).gated
    }
    gates = [
        {"gate": "score_package_locked_before_evaluator", "passed": True},
        {"gate": "labels_absent_from_feature_and_score_generation", "passed": True},
        {"gate": "required_ungated_encoder_screen_succeeded", "passed": required_ungated.issubset(available_encoders)},
        {"gate": "gated_encoder_access_outcome_recorded", "passed": all(row["status"] in {"available", "unavailable"} for row in json.loads((safe_root / "resolved_model_revisions.json").read_text(encoding="utf-8"))["encoders"])},
        {"gate": "automatic_insufficient_peer_views_reported", "passed": any(row["route"] == "automatic_box" and row["status"] == "insufficient_peers" for row in coverage_summary)},
        {"gate": "one_and_two_anomaly_metrics_separated", "passed": {row["anomaly_count"] for row in scene_type_summary} == {"1", "2"} or {str(row["anomaly_count"]) for row in scene_type_summary} == {"1", "2"}},
        {"gate": "ds0_not_used_as_final_encoder_selection", "passed": not decision["final_encoder_selected"]},
    ]
    _write_csv(safe_root / "completion_gates.csv", gates)
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "completed",
        "run_mode": package_record["run_mode"],
        "score_package_sha256": score_package_sha256,
        "evaluator_opened_after_score_lock": True,
        "evaluation_row_count": len(evaluation_rows),
        "scene_metric_row_count": len(scene_rows),
        "all_completion_gates_passed": all(bool(row["passed"]) for row in gates),
        "decision": decision["decision"],
    }
    _write_json(safe_root / "run_record.json", run_record)
    _write_artifact_manifest(private_root)
    _write_artifact_manifest(safe_root)
    return run_record


def run_ds0_representation_pipeline(
    *,
    rgb_root: Path,
    mask_route_root: Path,
    evaluator_root: Path,
    output_root: Path,
    encoder_keys: Sequence[str] = DEFAULT_ENCODERS,
    device: str = "cuda",
    hf_token: str | None = None,
    run_mode: str = "model_evidence",
) -> dict[str, Any]:
    """Run label-free inference first, then the evaluator-only stage."""

    label_free = run_label_free_ds0(
        rgb_root=rgb_root,
        mask_route_root=mask_route_root,
        output_root=output_root,
        encoder_keys=encoder_keys,
        device=device,
        hf_token=hf_token,
        run_mode=run_mode,
    )
    evaluated = run_private_ds0_evaluation(
        representation_root=output_root,
        evaluator_root=evaluator_root,
    )
    return {"label_free": label_free, "evaluated": evaluated}


__all__ = [
    "DEFAULT_ENCODERS",
    "run_ds0_representation_pipeline",
    "run_label_free_ds0",
    "run_private_ds0_evaluation",
    "verify_artifact_manifest",
]
