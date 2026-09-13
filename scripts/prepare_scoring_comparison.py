"""Build separate model and evaluator packages from existing development evidence."""

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from visual_peer_inspector.models.sam2 import binary_mask_sha256, decode_binary_mask, encode_binary_mask
from visual_peer_inspector.workflows.comparison_io import atomic_json, canonical, digest_file, read_json, read_lines, seal


def supplied_mask(path, reference):
    if digest_file(path) != reference["mask_sha256"]:
        raise ValueError("Reference mask file checksum mismatch")
    mask = np.asarray(Image.open(path).convert("L")) >= 128
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if mask.any() else None
    if int(mask.sum()) != reference["foreground_pixels"] or box != reference["box_xyxy"]:
        raise ValueError("Reference mask decoding differs from the verified geometry")
    return mask


def prepare(workspace, destination):
    workspace, destination = Path(workspace), Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new package directory; do not overwrite accepted inputs")
    local = workspace / "code/.local"
    evidence = workspace / "industry-project/active/implementation/evidence"
    media = local / "ds1-model-safe-v1/model-input"
    geometry = local / "ds1-oracle-geometry-v1"
    automatic_root = evidence / "08e-frozen-development/v5-complete/toysad8k_frozen_development_pipeline/toysad8k-development-screen-100-v1/model-safe"
    correspondence_path = evidence / "08e1-development-identity/v4-complete/08e1_toysad8k_development_identity_evaluator_release/development-identity-evaluator/evaluator_correspondence.jsonl"
    attributes_path = local / "ds1-evaluator-attributes-v1/protected_scene_attributes.jsonl"
    old_evaluator = read_json(correspondence_path.parent / "evaluator_manifest.json")
    if digest_file(correspondence_path) != old_evaluator["correspondence_sha256"]:
        raise ValueError("Existing evaluator correspondence has changed")
    roles_path = local / "toysad/selection/toysad8k-train-2000-v2/experiment_roles.jsonl"
    membership = sorted(r["opaque_scene_id"] for r in read_lines(roles_path) if r.get("pilot_stage") == "screen_100" and r.get("role") == "development")
    if len(membership) != 100:
        raise ValueError("The role ledger must identify exactly the existing screen100 scenes")
    catalogue = {r["opaque_scene_id"]: r for r in read_lines(media / "neutral_corpus_manifest.jsonl")}
    if set(catalogue) != set(membership):
        raise ValueError("RGB membership differs from the existing development screen")
    attributes = {r["opaque_scene_id"]: r for r in read_lines(attributes_path)}
    predictions = {(r["opaque_scene_id"], r["view_id"]): r for r in read_lines(automatic_root / "prediction_manifest.jsonl")}
    references = defaultdict(list)
    for row in read_lines(geometry / "reference_manifest.jsonl"):
        references[row["opaque_scene_id"], row["view_id"]].append(row)
    correspondence = {(r["opaque_scene_id"], r["view_id"], r["object_id"]): r["reference_identity_id"] for r in read_lines(correspondence_path) if r["opaque_scene_id"] in membership}
    smoke = sorted(membership, key=lambda sid: sha256(("comparison-smoke-v1/" + sid).encode()).hexdigest())[:20]
    index = {"schema": 1, "study": "development-scoring-comparison-v1", "scenes": membership, "smoke_scenes": sorted(smoke), "views": [f"V{i:02d}" for i in range(20)], "routes": ["supplied", "automatic"]}
    model = destination / "model-input"
    evaluator = destination / "evaluator-only"
    model.mkdir(parents=True)
    evaluator.mkdir()
    counts = defaultdict(int)
    for scene in membership:
        truth = attributes[scene]
        evaluation = {"scene": scene, "asset_group": truth["opaque_asset_group_id"], "category": truth["category"], "families": truth["anomaly_families"], "labels": truth["labels_by_object_id"], "correspondence": {}}
        records = []
        for view in index["views"]:
            source_image = media / catalogue[scene]["media_root"] / "rgb" / f"{view}.png"
            relative = f"media/{scene}/{view}.png"
            (model / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, model / relative)
            image_hash = digest_file(source_image)
            old = predictions[scene, view]
            if old["image_sha256"] != image_hash:
                raise ValueError("Automatic masks belong to different RGB input")
            for route in index["routes"]:
                items = references[scene, view] if route == "supplied" else old["objects"]
                objects = []
                for item in items:
                    original = item["opaque_object_id"] if route == "supplied" else item["object_id"]
                    candidate = "C" + sha256(f"{scene}/{view}/{route}/{original}".encode()).hexdigest()[:16]
                    if route == "supplied":
                        mask_path = geometry / "masks" / item["mask_path"]
                        # Rendered background values can be 1, not exactly zero.
                        mask = supplied_mask(mask_path, item)
                        identity = original
                    else:
                        mask = decode_binary_mask(item["mask_rle"])
                        if binary_mask_sha256(mask) != item["mask_sha256"]:
                            raise ValueError("Automatic mask does not match its original digest")
                        identity = correspondence.get((scene, view, original))
                    if not mask.any():
                        counts[route + "_empty_masks"] += 1
                        continue
                    y, x = np.where(mask)
                    objects.append({"candidate_id": candidate, "box": [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)], "mask": encode_binary_mask(mask), "mask_sha256": sha256(mask.astype(np.uint8).tobytes()).hexdigest()})
                    evaluation["correspondence"][f"{route}/{view}/{candidate}"] = identity
                objects.sort(key=lambda x: x["candidate_id"])
                counts[route + "_objects"] += len(objects)
                records.append({"scene": scene, "view": view, "route": route, "rgb": relative, "rgb_sha256": image_hash, "objects": objects})
        atomic_json(model / "scenes" / f"{scene}.json", records)
        atomic_json(evaluator / "scenes" / f"{scene}.json", evaluation)
    atomic_json(model / "index.json", index)
    seal(model, [p for p in model.rglob("*") if p.is_file()], metadata={"kind": "model-input", "study": index["study"]})
    atomic_json(evaluator / "index.json", {"scenes": membership, "model_lock_sha256": digest_file(model / "artifact_lock.json"), "correspondence_source_sha256": digest_file(correspondence_path), "attributes_source_sha256": digest_file(attributes_path), "role_ledger_sha256": digest_file(roles_path), "automatic_prediction_sha256": digest_file(automatic_root / "prediction_manifest.jsonl")})
    seal(evaluator, [p for p in evaluator.rglob("*") if p.is_file()], metadata={"kind": "evaluator-only"})
    atomic_json(destination / "preparation_receipt.json", {"scene_count": 100, "view_count": 2000, "route_view_count": 4000, "counts": dict(counts), "scope": "existing development screen only", "model_lock_sha256": digest_file(model / "artifact_lock.json")})
    print("Prepared 100 existing development scenes, 20 views each, two mask routes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.workspace, args.output)
