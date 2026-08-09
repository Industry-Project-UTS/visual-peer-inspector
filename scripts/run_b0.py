#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector import (
    average_precision,
    load_scene_view,
    peer_mean_scores,
    permutation_error,
)
from visual_peer_inspector.chunks import load_json


DEFAULT_CONFIG = PROJECT_ROOT / "data" / "manifests" / "development_40.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the B0 peer-mean baseline")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def choose_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(requested)


def crop_object(
    image: Image.Image,
    mask_path: Path,
    *,
    threshold: int,
    padding_fraction: float,
    background: tuple[int, int, int],
) -> Image.Image:
    with Image.open(mask_path) as source:
        mask = source.convert("L").point(lambda value: 255 if value > threshold else 0)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError(f"Object mask is empty: {mask_path}")

    left, top, right, bottom = bounds
    padding = max(1, round(max(right - left, bottom - top) * padding_fraction))
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    image_crop = image.crop(crop_box)
    mask_crop = mask.crop(crop_box)
    background_image = Image.new("RGB", image_crop.size, background)
    return Image.composite(image_crop, background_image, mask_crop)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    config = load_json(args.config)
    device = choose_device(args.device)
    model_config = config["model"]
    crop_config = config["crop"]
    background = tuple(int(value) for value in crop_config["background_rgb"])
    if len(background) != 3:
        raise ValueError("background_rgb must contain three values")

    load_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(
        str(model_config["repo_id"]),
        revision=str(model_config["revision"]),
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        str(model_config["repo_id"]),
        revision=str(model_config["revision"]),
        local_files_only=args.local_files_only,
    )
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - load_started

    scene_results = []
    run_started = time.perf_counter()
    for scene_spec in config["scenes"]:
        scene_id = (
            str(scene_spec["scene_id"])
            if isinstance(scene_spec, dict)
            else str(scene_spec)
        )
        scene_started = time.perf_counter()
        scene = load_scene_view(
            args.data_root,
            scene_id,
            split=str(config["split"]),
            view_id=str(config["view_id"]),
        )
        with Image.open(scene.image_path) as source:
            image = source.convert("RGB")
        crops = [
            crop_object(
                image,
                item.mask_path,
                threshold=int(crop_config["mask_threshold"]),
                padding_fraction=float(crop_config["padding_fraction"]),
                background=background,
            )
            for item in scene.objects
        ]

        inputs = processor(images=crops, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            output_features = model(pixel_values=pixel_values)
        vectors = output_features.last_hidden_state[:, 0, :].detach().cpu().tolist()
        scores = peer_mean_scores(vectors)
        ranking = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        labels = [item.is_anomaly for item in scene.objects]
        ranks = {object_index: rank for rank, object_index in enumerate(ranking, 1)}

        scene_results.append(
            {
                "scene_id": scene.scene_id,
                "object_count": len(scene.objects),
                "anomaly_count": scene.anomaly_count,
                "top1_correct": labels[ranking[0]],
                "average_precision": average_precision(labels, ranking),
                "permutation_error": permutation_error(vectors, scores),
                "seconds": time.perf_counter() - scene_started,
                "objects": [
                    {
                        "index": item.index,
                        "name": item.name,
                        "is_anomaly": item.is_anomaly,
                        "variant": item.anomaly_variant,
                        "score": score,
                        "rank": ranks[item.index],
                    }
                    for item, score in zip(scene.objects, scores, strict=True)
                ],
            }
        )
        print(f"{scene.scene_id}: top1={labels[ranking[0]]}", flush=True)

    run_seconds = time.perf_counter() - run_started
    result = {
        "schema_version": 1,
        "run_id": str(config["id"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": str(config["dataset_revision"]),
        "split": str(config["split"]),
        "view_id": str(config["view_id"]),
        "device": str(device),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "pillow")
        },
        "config_sha256": file_hash(args.config),
        "model_load_seconds": model_load_seconds,
        "run_seconds": run_seconds,
        "scene_count": len(scene_results),
        "top1_correct_count": sum(item["top1_correct"] for item in scene_results),
        "mean_average_precision": sum(
            float(item["average_precision"]) for item in scene_results
        )
        / len(scene_results),
        "permutation_checks_pass": all(
            float(item["permutation_error"]) < 1e-9 for item in scene_results
        ),
        "scenes": scene_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Top-1: {result['top1_correct_count']}/{result['scene_count']}")
    print(f"Mean AP: {result['mean_average_precision']:.4f}")
    print(f"Output: {args.output.expanduser().resolve()}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
