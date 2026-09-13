"""Prediction-only scoring comparison with reusable features and checked resumes."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.representation.crops import CropConfiguration, prepare_masked_crop
from visual_peer_inspector.scoring.peer import b0_scores, b1_scores, p1_scores
from visual_peer_inspector.scoring.forward_patch import forward_patch_scores, patch_policy
from visual_peer_inspector.scoring.peer_support import (
    geometry_peer_scores,
    mask_shape_descriptor,
    mutual_missing_scores,
    rank_consensus,
)
from visual_peer_inspector.workflows.comparison_io import atomic_json, canonical, digest_file, implementation_digest, read_json, safe_path, seal, token, verify_seal
from visual_peer_inspector.workflows.kaggle_recovery import create_recovery_bundle, restore_recovery_bundle

REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
METHODS = (
    "B0",
    "B1_control",
    "P1_current",
    "forward_10_cov0",
    "forward_02_cov70",
    "mutual_missing_05",
    "geometry_peer",
    "P2_consensus",
)
CROP = CropConfiguration()
PROTOCOL = {
    "study": "development-scoring-comparison-v1",
    "encoder": "facebook/dinov2-small", "revision": REVISION,
    "crop": asdict(CROP), "methods": list(METHODS),
    "current_p1": {"coverage": .50, "tail": .10, "rounding": "ceil", "fallback": "largest", "reduction": "symmetric_pair_tail_then_median"},
    "forward": {"reduction": "lower_median_patch_then_tail", "rounding": "floor", "fallback": "all"},
    "feature_cache": "float32_lossless_npz", "feature_batch_size": 4,
    "mutual_missing": {"tail": .05, "evidence": "positive target-minus-shared-peer distance"},
    "geometry": {"descriptors": ["log_area", "log_aspect", "extent", "eccentricity", "sin_2_angle", "cos_2_angle", "compactness"]},
    "p2_consensus": {"reduction": "mean within-view percentile rank", "components": ["forward_02_cov70", "mutual_missing_05", "geometry_peer"]},
    "context_ranker": {"folds": 5, "group": "asset_group", "training": "scene-softmax", "regularization": 1.0, "status": "development candidate"},
    "primary_views": ["V00", "V07", "V14"],
    "secondary_views": {"2": ["V00", "V10"], "5": ["V00", "V04", "V08", "V12", "V16"]},
    "twenty_view_scope": "reference-identity capability diagnostic only",
    "promotion": "manual development decision; not protected confirmation",
}

RECOVERY_TOP_LEVELS = (
    "features",
    "preflight.json",
    "progress.json",
    "run_binding.json",
    "run_status.json",
    "scenes",
)


def preflight(root):
    root = Path(root)
    verify_seal(root)
    index = read_json(root / "index.json")
    if set(index) != {"schema", "study", "scenes", "smoke_scenes", "views", "routes"}:
        raise ValueError("Unexpected model index fields")
    if index["study"] != PROTOCOL["study"] or index["routes"] != ["supplied", "automatic"]:
        raise ValueError("Wrong study package")
    if len(index["scenes"]) != len(set(index["scenes"])) or not set(index["smoke_scenes"]) <= set(index["scenes"]):
        raise ValueError("Invalid scene membership")
    counts = Counter()
    allowed_files = {"index.json", "artifact_lock.json"}
    for scene in index["scenes"]:
        token(scene)
        allowed_files.add(f"scenes/{scene}.json")
        rows = read_json(root / "scenes" / f"{scene}.json")
        expected = {(v, r) for v in index["views"] for r in index["routes"]}
        if len(rows) != len(expected) or {(r["view"], r["route"]) for r in rows} != expected:
            raise ValueError("Missing or duplicated route/view records")
        for row in rows:
            if set(row) != {"scene", "view", "route", "rgb", "rgb_sha256", "objects"} or row["scene"] != scene:
                raise ValueError("Unexpected model record fields")
            rgb_path = safe_path(root, row["rgb"])
            allowed_files.add(row["rgb"])
            if digest_file(rgb_path) != row["rgb_sha256"]:
                raise ValueError("RGB checksum mismatch")
            with Image.open(rgb_path) as image:
                shape = (image.height, image.width)
                image.verify()
            seen = set()
            for obj in row["objects"]:
                if set(obj) != {"candidate_id", "box", "mask", "mask_sha256"}:
                    raise ValueError("Unexpected object fields")
                candidate = token(obj["candidate_id"])
                if candidate in seen:
                    raise ValueError("Duplicate candidate")
                seen.add(candidate)
                mask = decode_binary_mask(obj["mask"])
                if mask.shape != shape or not mask.any() or sha256(mask.astype(np.uint8).tobytes()).hexdigest() != obj["mask_sha256"]:
                    raise ValueError("Invalid mask geometry or checksum")
                ys, xs = np.where(mask)
                if obj["box"] != [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]:
                    raise ValueError("Box does not enclose its mask")
            counts[row["route"] + "_views"] += 1
            counts[row["route"] + "_objects"] += len(row["objects"])
            counts[row["route"] + "_insufficient_views"] += len(row["objects"]) < 3
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if present != allowed_files:
        raise ValueError("The model-only package contains missing or unexpected files")
    return {"scenes": len(index["scenes"]), "counts": dict(counts), "input_lock_sha256": digest_file(root / "artifact_lock.json")}


def cache_features(path, crops, encoder, binding):
    """Reuse features only if their input binding and checksum both match."""
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        record = read_json(receipt)
        if record["binding"] != binding:
            raise ValueError("Feature cache belongs to different inputs or preprocessing")
        if digest_file(path) != record["sha256"]:
            raise ValueError("Feature cache is corrupt; quarantine it before retrying")
        with np.load(path, allow_pickle=False) as data:
            batch = EncodedBatch(data["global"], tuple(data["patches"]), int(data["grid"]), data["global"].shape[1])
        if not np.isfinite(batch.global_features).all() or not all(np.isfinite(p).all() for p in batch.patch_features):
            raise ValueError("Feature cache contains non-finite values")
        return batch, True
    chunks = [encoder.encode([x.rgb for x in crops[i:i+4]]) for i in range(0, len(crops), 4)]
    batch = EncodedBatch(np.concatenate([x.global_features for x in chunks]), tuple(p for x in chunks for p in x.patch_features), chunks[0].patch_grid_size, chunks[0].feature_dimension)
    if batch.patch_grid_size != 16:
        raise ValueError("DINOv2-small must return a 16 by 16 patch grid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{"global": batch.global_features.astype(np.float32), "patches": np.stack(batch.patch_features).astype(np.float32), "grid": batch.patch_grid_size})
    temporary.replace(path)
    atomic_json(receipt, {"binding": binding, "sha256": digest_file(path)})
    return batch, False


def score_view(row, root, output, encoder, producer):
    start = time.monotonic()
    rgb = np.asarray(Image.open(safe_path(root, row["rgb"])).convert("RGB"))
    objects = row["objects"]
    crops = [prepare_masked_crop(rgb, decode_binary_mask(obj["mask"]), configuration=CROP) for obj in objects]
    results, fallback = {}, {}
    cache_hit = False
    embeddings = []
    if objects:
        from visual_peer_inspector.models import visual_encoder
        from visual_peer_inspector.representation import crops as crop_module
        binding = sha256(canonical({"producer": producer, "revision": REVISION, "crop": asdict(CROP), "encoder_source": digest_file(visual_encoder.__file__), "crop_source": digest_file(crop_module.__file__), "image": row["rgb_sha256"], "objects": [(o["candidate_id"], c.sha256) for o, c in zip(objects, crops, strict=True)]})).hexdigest()
        cached = output / "features" / row["scene"] / f"{row['view']}-{row['route']}.npz"
        batch, cache_hit = cache_features(cached, crops, encoder, binding)
        embeddings = batch.global_features.tolist()
        if len(objects) >= 3:
            masks50 = [patch_policy(c, 16, .50, fallback="largest") for c in crops]
            masks70 = [patch_policy(c, 16, .70, fallback="all") for c in crops]
            shape_descriptors = np.stack([mask_shape_descriptor(c.foreground) for c in crops])
            results = {
                "B0": b0_scores(batch.global_features),
                "B1_control": b1_scores(batch.global_features),
                "P1_current": p1_scores(batch.patch_features, [m for m, _ in masks50]),
                "forward_10_cov0": forward_patch_scores(batch.patch_features, [np.ones((16,16), bool) for _ in crops], top_fraction=.10),
                "forward_02_cov70": forward_patch_scores(batch.patch_features, [m for m, _ in masks70], top_fraction=.02),
                "mutual_missing_05": mutual_missing_scores(batch.patch_features, [m for m, _ in masks50], tail_fraction=.05),
                "geometry_peer": geometry_peer_scores(shape_descriptors),
            }
            consensus = rank_consensus(
                results["forward_02_cov70"].scores,
                results["mutual_missing_05"].scores,
                results["geometry_peer"].scores,
            )
            results["P2_consensus"] = type(results["B0"])("P2_consensus", "scored", consensus)
            fallback = {"P1_current": sum(f for _, f in masks50), "forward_02_cov70": sum(f for _, f in masks70)}
    scored = []
    for i, obj in enumerate(objects):
        scores, maps = {}, {}
        for method in METHODS:
            result = results.get(method)
            scores[method] = None if result is None else float(result.scores[i])
            if result is not None and result.heatmaps:
                maps[method] = [[float(v) if np.isfinite(v) else None for v in line] for line in result.heatmaps[i]]
        descriptor = mask_shape_descriptor(crops[i].foreground).tolist() if objects else []
        scored.append({"candidate_id": obj["candidate_id"], "scores": scores, "embedding": embeddings[i], "geometry": descriptor, "heatmaps": maps})
    return {"scene": row["scene"], "view": row["view"], "route": row["route"], "status": "scored" if results else "insufficient_peers", "objects": scored, "patch_fallbacks": fallback, "seconds": time.monotonic()-start, "feature_cache_hit": cache_hit}


def _write_recovery(output, binding):
    bundle = output.parent / f"{output.name}-recovery.zip"
    manifest = output.parent / f"{output.name}-recovery-manifest.json"
    create_recovery_bundle(
        source_root=output,
        bundle_path=bundle,
        manifest_path=manifest,
        binding=binding,
        allowed_top_levels=RECOVERY_TOP_LEVELS,
    )
    return bundle, manifest


def predict(
    root,
    output,
    source_root,
    encoder_factory,
    *,
    cohort="smoke20",
    producer="dinov2-small",
    recovery_bundle=None,
    recovery_manifest=None,
):
    root, output = Path(root), Path(output)
    audit = preflight(root)
    index = read_json(root / "index.json")
    scenes = index["smoke_scenes"] if cohort == "smoke20" else index["scenes"]
    versions = {}
    for package in ("torch", "transformers", "numpy", "pillow", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    binding = {"protocol": PROTOCOL, "producer": producer, "input_lock_sha256": audit["input_lock_sha256"], "implementation_sha256": implementation_digest(source_root), "runtime_versions": versions}
    output.parent.mkdir(parents=True, exist_ok=True)
    if recovery_bundle is not None or recovery_manifest is not None:
        if recovery_bundle is None or recovery_manifest is None:
            raise ValueError("Recovery requires both the ZIP and its sidecar manifest")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Restore recovery only into a new output directory")
        restore_recovery_bundle(
            bundle_path=Path(recovery_bundle),
            manifest_path=Path(recovery_manifest),
            destination_root=output,
            expected_binding=binding,
            allowed_top_levels=RECOVERY_TOP_LEVELS,
        )
        print(json.dumps({"event": "recovery_restored", "bundle": Path(recovery_bundle).name}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_binding.json").exists() and read_json(output / "run_binding.json") != json.loads(canonical(binding)):
        raise ValueError("Existing output has a different source, runtime or input binding")
    atomic_json(output / "run_binding.json", binding)
    atomic_json(output / "preflight.json", audit)
    atomic_json(output / "run_status.json", {"status": "running", "cohort": cohort})
    encoder = None
    checkpoints = []
    try:
        for number, scene in enumerate(scenes, 1):
            checkpoint = output / "scenes" / f"{scene}.json"
            receipt = checkpoint.with_suffix(".receipt.json")
            if checkpoint.exists() and receipt.exists():
                if read_json(receipt)["sha256"] != digest_file(checkpoint):
                    raise ValueError("Scene checkpoint is corrupt; do not silently accept it")
                if read_json(receipt)["binding_sha256"] != sha256(canonical(binding)).hexdigest():
                    raise ValueError("Scene checkpoint binding mismatch")
                print(json.dumps({"event": "scene_resumed", "scene_number": number, "total": len(scenes)}), flush=True)
            else:
                if encoder is None:
                    print(json.dumps({"event": "model_load_start", "revision": REVISION}), flush=True)
                    encoder = encoder_factory()
                    print(json.dumps({"event": "model_ready"}), flush=True)
                rows = read_json(root / "scenes" / f"{scene}.json")
                predictions = []
                for row in rows:
                    predictions.append(score_view(row, root, output, encoder, producer))
                    print(json.dumps({"event": "view_complete", "scene_number": number, "view": row["view"], "route": row["route"]}), flush=True)
                atomic_json(checkpoint, predictions)
                atomic_json(receipt, {"sha256": digest_file(checkpoint), "binding_sha256": sha256(canonical(binding)).hexdigest()})
                # Raw patch tensors are useful only until their scene result is sealed.
                # The scene checkpoint retains compact embeddings, scores and evidence.
                transient = output / "features" / scene
                if transient.exists():
                    shutil.rmtree(transient)
            checkpoints.extend([checkpoint, receipt])
            atomic_json(output / "progress.json", {"complete_scenes": number, "expected_scenes": len(scenes), "cohort": cohort})
            if number % 10 == 0 or number == len(scenes):
                _write_recovery(output, binding)
        seal(
            output,
            checkpoints,
            name="feature_lock.json",
            metadata={"scenes": scenes, "cohort": cohort, "producer": producer, "format": "compact object embeddings, component scores and heatmaps"},
        )
        lock_paths = checkpoints + [output / "run_binding.json", output / "preflight.json", output / "feature_lock.json"]
        seal(output, lock_paths, name="prediction_lock.json", metadata={"scenes": scenes, "cohort": cohort, "producer": producer, "status": "complete", "feature_lock_sha256": digest_file(output / "feature_lock.json")})
        # Reporting is downstream of locked predictions. A rendering error cannot erase them.
        from visual_peer_inspector.workflows.comparison_prediction_figures import prediction_previews
        preview_paths = prediction_previews(root, output, scenes)
        seal(output, preview_paths, name="preview_lock.json", metadata={"prediction_lock_sha256": digest_file(output / "prediction_lock.json")})
        atomic_json(output / "run_status.json", {"status": "complete", "cohort": cohort, "scene_count": len(scenes)})
        _write_recovery(output, binding)
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "cohort": cohort})
        try:
            _write_recovery(output, binding)
        except Exception as recovery_error:
            print(json.dumps({"event": "recovery_export_failed", "error_type": type(recovery_error).__name__}), flush=True)
        raise
    return verify_seal(output, name="prediction_lock.json")
