"""Label-free FP32 CUDA compatibility check against the sealed CPU DINO cache.

This module is deliberately separate from production cache extraction.  A
passing receipt is evidence that one reviewed CUDA runtime reproduced the
existing CPU feature and decision contract on a fixed development sample; it
does not authorize mixing CPU- and GPU-bound feature roots.
"""
from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.scoring.forward_patch import patch_policy
from visual_peer_inspector.scoring.peer_support import mask_shape_descriptor
from visual_peer_inspector.web.decision import component_scores, decide_cached
from visual_peer_inspector.web.deployment import pipeline_config
from vpi_object_bank.protocol import REVISION
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal

from .native_cache import _feature_collection, _gates, native_regions


SCHEMA = "vpi-dino-cpu-gpu-fp32-compatibility/v1"
GPU_PINNED = {
    "torch": "2.13.0+cu126",
    "transformers": "5.14.1",
    "numpy": "2.5.3",
    "pillow": "12.3.0",
}
SAMPLE_SCENES = 10
SAMPLE_VIEWS = ("V00", "V07", "V14")
SAMPLE_SEED = "vpi-dino-fp32-cuda-compatibility-v1"
THRESHOLDS = {
    "patch_max_abs": 1.0e-3,
    "global_max_abs": 5.0e-4,
    "minimum_cosine": 0.99999,
    "component_max_abs": 5.0e-4,
    "automatic_decision_exact_fraction": 1.0,
}


def compatible_gpu_runtime(*, versions=None, cuda_available=None):
    """Verify exact userspace pins and actual CUDA availability before imports."""
    if versions is None:
        try:
            versions = {name: importlib.metadata.version(name) for name in GPU_PINNED}
        except importlib.metadata.PackageNotFoundError:
            return False
    if versions != GPU_PINNED:
        return False
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    return bool(cuda_available)


class CudaFp32Encoder:
    """Fixed DINOv2-small inference with CUDA autocast explicitly disabled."""

    def __init__(self, model_source):
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.manual_seed(32040)
        torch.cuda.manual_seed_all(32040)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.functional = functional
        self.model = AutoModel.from_pretrained(
            Path(model_source).resolve(), local_files_only=True
        ).eval().to(device="cuda:0", dtype=torch.float32)

    def encode(self, crops: Sequence[np.ndarray]):
        if not crops:
            raise ValueError("At least one crop is required")
        array = np.stack([np.asarray(crop, dtype=np.float32) for crop in crops])
        if array.shape[1:] != (224, 224, 3):
            raise ValueError("Every compatibility crop must be 224 x 224 RGB")
        tensor = self.torch.from_numpy(array / 255.0).permute(0, 3, 1, 2)
        mean = self.torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        inputs = ((tensor - mean) / std).to("cuda:0", dtype=self.torch.float32)
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", enabled=False
        ):
            output = self.model(pixel_values=inputs)
        tokens = output.last_hidden_state
        register_count = int(getattr(self.model.config, "num_register_tokens", 0))
        patch_size = getattr(self.model.config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        grid = 224 // int(patch_size)
        patches = tokens[:, 1 + register_count :, :]
        if grid != 16 or patches.shape[1:] != (256, 384):
            raise ValueError("CUDA encoder changed the fixed DINO patch schema")
        patches = self.functional.normalize(patches.float(), dim=2)
        globals_ = self.functional.normalize(tokens[:, 0, :].float(), dim=1)
        return (
            globals_.detach().cpu().numpy().astype(np.float32),
            tuple(value.detach().cpu().numpy().astype(np.float32) for value in patches),
        )


def _selected_scenes(scenes):
    values = sorted(set(scenes))
    if len(values) != 100:
        raise ValueError("Compatibility input must be the exact sealed 100-scene CPU cache")
    ranked = sorted(values, key=lambda scene: sha256(f"{SAMPLE_SEED}:{scene}".encode()).hexdigest())
    return sorted(ranked[:SAMPLE_SCENES])


def _cosine(left, right):
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0.0 and np.array_equal(a, b) else float(np.dot(a, b) / denominator)


def _decision_signature(result):
    def members(row):
        return tuple(sorted((m["view_id"], m["candidate_id"]) for m in row["members"]))
    return {
        "ranking": tuple(members(row) for row in result["ranking"]),
        "tracks": tuple(sorted(members(row) for row in result["tracks"])),
    }


def _score_differences(cpu_rows, gpu_rows):
    maximum = 0.0
    for view in SAMPLE_VIEWS:
        cpu = {o["candidate_id"]: o["scores"] for o in cpu_rows[view]["objects"]}
        gpu = {o["candidate_id"]: o["scores"] for o in gpu_rows[view]["objects"]}
        if set(cpu) != set(gpu):
            raise ValueError("CPU/GPU score candidates differ")
        for candidate in cpu:
            if set(cpu[candidate]) != set(gpu[candidate]):
                raise ValueError("CPU/GPU component inventories differ")
            for name in cpu[candidate]:
                left, right = cpu[candidate][name], gpu[candidate][name]
                if left is None or right is None:
                    if left is not right:
                        raise ValueError("CPU/GPU scoreability differs")
                else:
                    maximum = max(maximum, abs(float(left) - float(right)))
    return maximum


def run_gpu_compatibility(cfg, source_root, *, encoder_factory=None):
    """Compare a fixed label-free sample and write a sealed compatibility receipt."""
    contract, freeze_hash = _gates(cfg, source_root)
    if cfg.get("role") != "development":
        raise ValueError("GPU compatibility must use only the development cohort")
    regions, region_hash = native_regions(cfg, contract)
    cpu_root = Path(cfg["reference_feature_root"])
    cpu_binding, _cpu_signature, cpu_units, cpu_locks = _feature_collection(
        {"feature_roots": [cpu_root]}, region_hash
    )
    verify_seal(cpu_root, name=("feature_lock.json" if (cpu_root / "feature_lock.json").is_file() else "feature_shard_lock.json"))
    scenes = _selected_scenes(scene for scene, _view in cpu_units)
    required = {(scene, view) for scene in scenes for view in SAMPLE_VIEWS}
    if not required <= set(cpu_units) or not required <= set(regions):
        raise ValueError("CPU reference or native regions omit a fixed compatibility unit")

    model = Path(cfg["model_source"])
    expected = pipeline_config()["encoder"]
    if REVISION != expected["revision"] or any(
        digest_file(model / filename) != expected[key]
        for filename, key in (("model.safetensors", "sha256"), ("config.json", "config_sha256"))
    ):
        raise ValueError("Compatibility job requires the immutable application DINO snapshot")
    if encoder_factory is None:
        if not compatible_gpu_runtime():
            raise ValueError("Compatibility job requires the exact DINO CUDA FP32 runtime")
        encoder = CudaFp32Encoder(model)
    else:
        encoder = encoder_factory()

    details = []
    maxima = {"patch": 0.0, "global": 0.0, "component": 0.0}
    minimum_cosine = 1.0
    exact_decisions = 0
    compared_candidates = 0
    for scene in scenes:
        cpu_records, gpu_records = [], []
        for view in SAMPLE_VIEWS:
            row = regions[scene, view]
            rgb = np.asarray(Image.open(row["_rgb_path"]).convert("RGB"))
            native = {o["candidate_id"]: o for o in row["objects"]}
            unit = cpu_units[scene, view]
            meta = read_json(unit / "observations.json")
            if (meta["scene"], meta["view"], meta["route"], meta["rgb_sha256"]) != (
                scene, view, "automatic", row["rgb_sha256"]
            ):
                raise ValueError("CPU compatibility unit provenance differs")
            observations = sorted(meta["objects"], key=lambda item: item["candidate_id"])
            if {o["candidate_id"] for o in observations} != set(native):
                raise ValueError("CPU compatibility unit candidate inventory differs")
            crops = [prepare_masked_crop(rgb, decode_binary_mask(native[o["candidate_id"]]["mask"])) for o in observations]
            gpu_globals, gpu_patches = [], []
            for offset in range(0, len(crops), 4):
                globals_batch, patches_batch = encoder.encode([c.rgb for c in crops[offset : offset + 4]])
                if globals_batch.shape != (len(patches_batch), 384):
                    raise ValueError("CUDA encoder global feature schema differs")
                gpu_globals.extend(globals_batch)
                gpu_patches.extend(patches_batch)
            with np.load(unit / "features.npz", allow_pickle=False) as arrays:
                for index, (item, crop, gpu_global, gpu_patch) in enumerate(
                    zip(observations, crops, gpu_globals, gpu_patches, strict=True)
                ):
                    obj = native[item["candidate_id"]]
                    array_index = item["array_index"]
                    cpu_patch, cpu_global = arrays[f"p{array_index}"], arrays[f"g{array_index}"]
                    valid70, _ = patch_policy(crop, 16, 0.70, fallback="all")
                    valid50, _ = patch_policy(crop, 16, 0.50, fallback="largest")
                    if (
                        item["crop_sha256"] != crop.sha256
                        or item["mask_sha256"] != obj["mask_sha256"]
                        or not np.array_equal(arrays[f"v{array_index}"], valid70)
                    ):
                        raise ValueError("CPU reference crop/mask policy differs")
                    for name, value, shape in (
                        ("cpu_patch", cpu_patch, (256, 384)),
                        ("gpu_patch", gpu_patch, (256, 384)),
                        ("cpu_global", cpu_global, (384,)),
                        ("gpu_global", gpu_global, (384,)),
                    ):
                        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                            raise ValueError(f"{name} is not finite FP32 with the fixed shape")
                    patch_error = float(np.max(np.abs(cpu_patch - gpu_patch)))
                    global_error = float(np.max(np.abs(cpu_global - gpu_global)))
                    cosine = min(_cosine(cpu_patch, gpu_patch), _cosine(cpu_global, gpu_global))
                    maxima["patch"] = max(maxima["patch"], patch_error)
                    maxima["global"] = max(maxima["global"], global_error)
                    minimum_cosine = min(minimum_cosine, cosine)
                    compared_candidates += 1
                    details.append({"scene": scene, "view": view, "candidate_id": item["candidate_id"],
                        "patch_max_abs": patch_error, "global_max_abs": global_error, "minimum_cosine": cosine})
                    common = {"view_id": view, "candidate_id": item["candidate_id"],
                        "valid50": valid50, "valid70": valid70,
                        "geometry": mask_shape_descriptor(crop.foreground)}
                    cpu_records.append({**common, "embedding": cpu_global.tolist(), "patches": cpu_patch})
                    gpu_records.append({**common, "embedding": gpu_global.tolist(), "patches": gpu_patch})
        cpu_scores, _ = component_scores(cpu_records)
        gpu_scores, _ = component_scores(gpu_records)
        for view in SAMPLE_VIEWS:
            cpu_scores.setdefault(view, {"objects": []})
            gpu_scores.setdefault(view, {"objects": []})
        component_error = _score_differences(cpu_scores, gpu_scores)
        maxima["component"] = max(maxima["component"], component_error)
        cpu_decision = decide_cached(cpu_scores, [], views=SAMPLE_VIEWS, mode="automatic")
        gpu_decision = decide_cached(gpu_scores, [], views=SAMPLE_VIEWS, mode="automatic")
        decision_exact = _decision_signature(cpu_decision) == _decision_signature(gpu_decision)
        exact_decisions += int(decision_exact)
        details.append({"scene": scene, "component_max_abs": component_error,
            "automatic_decision_exact": decision_exact})

    exact_fraction = exact_decisions / len(scenes)
    passed = (
        compared_candidates > 0
        and maxima["patch"] <= THRESHOLDS["patch_max_abs"]
        and maxima["global"] <= THRESHOLDS["global_max_abs"]
        and minimum_cosine >= THRESHOLDS["minimum_cosine"]
        and maxima["component"] <= THRESHOLDS["component_max_abs"]
        and exact_fraction >= THRESHOLDS["automatic_decision_exact_fraction"]
    )
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "comparison_details.json", {"schema": SCHEMA, "rows": details})
    receipt = {
        "schema": SCHEMA,
        "status": "compatible" if passed else "incompatible",
        "production_gpu_release_allowed": passed,
        "cpu_gpu_feature_root_mixing_allowed": False,
        "evaluator_or_anomaly_labels_opened": False,
        "sample_selection": {"seed": SAMPLE_SEED, "scenes": scenes, "views": list(SAMPLE_VIEWS)},
        "compared_units": len(required),
        "compared_candidates": compared_candidates,
        "thresholds": THRESHOLDS,
        "observed": {"patch_max_abs": maxima["patch"], "global_max_abs": maxima["global"],
            "minimum_cosine": minimum_cosine, "component_max_abs": maxima["component"],
            "automatic_decision_exact_fraction": exact_fraction},
        "cpu_reference": {"binding": cpu_binding, "feature_locks": cpu_locks},
        "gpu_runtime": GPU_PINNED,
        "model_files": {"model.safetensors": expected["sha256"], "config.json": expected["config_sha256"]},
        "region_lock_sha256": region_hash,
        "method_freeze_sha256": freeze_hash,
        "source_sha256": digest_file(__file__),
    }
    atomic_json(output / "gpu_compatibility_receipt.json", receipt)
    seal(output, [output / "comparison_details.json", output / "gpu_compatibility_receipt.json"])
    return receipt


def require_compatible_gpu(receipt_root, *, region_hash, model_files):
    """Gate production GPU extraction on one exact, sealed compatibility receipt."""
    root = Path(receipt_root)
    verify_seal(root)
    receipt = read_json(root / "gpu_compatibility_receipt.json")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "compatible"
        or receipt.get("production_gpu_release_allowed") is not True
        or receipt.get("cpu_gpu_feature_root_mixing_allowed") is not False
        or receipt.get("region_lock_sha256") != region_hash
        or receipt.get("model_files") != model_files
        or receipt.get("thresholds") != THRESHOLDS
    ):
        raise ValueError("GPU production cache is not covered by a passing compatibility receipt")
    return receipt
