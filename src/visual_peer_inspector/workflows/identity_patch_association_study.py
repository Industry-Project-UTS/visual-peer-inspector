"""Compare foreground patch correspondences for cross-view object identity."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import shutil
import time
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import CropConfiguration, prepare_masked_crop
from visual_peer_inspector.representation.identity_association import (
    AssociationConfiguration,
    associate_views,
)
from visual_peer_inspector.representation.patch_correspondence import (
    PATCH_CONFIGURATIONS,
    associate_patch_views,
    foreground_patch_evidence,
)
from visual_peer_inspector.workflows.comparison_evaluation import (
    fused_ranking,
    metrics,
    normalized_scores,
    track_references,
)
from visual_peer_inspector.workflows.comparison_io import (
    atomic_json,
    canonical,
    digest_file,
    read_json,
    safe_path,
    seal,
    token,
    verify_seal,
)
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle,
    restore_recovery_bundle,
)
from visual_peer_inspector.workflows.scoring_comparison import PROTOCOL as SCORING_PROTOCOL


STAGE = "development_identity_patch_association"
SCHEMA_VERSION = "visual-peer-identity-patch-association/v1"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
CROP = CropConfiguration()
VIEW_SETS = {
    2: ("V00", "V10"),
    3: ("V00", "V07", "V14"),
    5: ("V00", "V04", "V08", "V12", "V16"),
}
REQUIRED_VIEWS = tuple(sorted({view for values in VIEW_SETS.values() for view in values}))
RANKING_METHODS = ("P1_current", "forward_02_cov70", "P2_consensus")
PRIMARY_METHOD = "P1_current"
BASELINE_CONFIGURATION_ID = "current_hungarian_035"
BASELINE_CONFIGURATION = AssociationConfiguration(
    BASELINE_CONFIGURATION_ID,
    "current",
    0.35,
)
RECOVERY_TOP_LEVELS = (
    "preflight.json",
    "progress.json",
    "run_binding.json",
    "run_status.json",
    "scenes",
)
PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "scope": "existing nested 100-scene development screen",
    "split_change": False,
    "route": "automatic",
    "encoder": "facebook/dinov2-small",
    "encoder_revision": MODEL_REVISION,
    "crop": {
        "input_size": CROP.input_size,
        "padding_fraction": CROP.padding_fraction,
        "background_rgb": list(CROP.background_rgb),
        "minimum_foreground_pixels": CROP.minimum_foreground_pixels,
    },
    "view_sets": {str(key): list(value) for key, value in VIEW_SETS.items()},
    "baseline": BASELINE_CONFIGURATION.__dict__,
    "patch_configurations": [value.__dict__ for value in PATCH_CONFIGURATIONS],
    "foreground_patch_coverage": 0.50,
    "ranking_methods": list(RANKING_METHODS),
    "selection": (
        "Promote only when identity F1 and scene AP have positive asset-grouped "
        "paired intervals, top-1 and recall at two do not regress, linked-node "
        "coverage does not fall, and singleton rate does not rise."
    ),
    "clearml": "post-compute aggregate evidence only",
}


def _runtime_versions() -> dict[str, str]:
    output = {}
    for package in ("torch", "transformers", "numpy", "pandas", "pillow", "scipy"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "not-installed"
    return output


def _implementation_digest(source_root: Path) -> str:
    paths = sorted((source_root / "src").rglob("*.py"))
    paths += sorted((source_root / "scripts").glob("*identity_patch_association*.py"))
    paths += sorted((source_root / "requirements").glob("identity-patch-association*.txt"))
    if not paths:
        raise ValueError("Identity patch-association source inventory is empty")
    inventory = {str(path.relative_to(source_root)): digest_file(path) for path in paths}
    return sha256(canonical(inventory)).hexdigest()


def _prediction_binding(
    model_root: Path,
    scoring_root: Path,
    source_root: Path,
    *,
    producer: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "producer": producer,
        "model_input_lock_sha256": digest_file(model_root / "artifact_lock.json"),
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "scoring_feature_lock_sha256": digest_file(scoring_root / "feature_lock.json"),
        "implementation_sha256": _implementation_digest(source_root),
        "runtime_versions": _runtime_versions(),
    }


def preflight(model_root: Path, scoring_root: Path) -> dict[str, Any]:
    """Verify the locked 08G source and exact model-only package."""

    model_root = Path(model_root).resolve()
    scoring_root = Path(scoring_root).resolve()
    verify_seal(model_root)
    scoring_lock = verify_seal(scoring_root, name="prediction_lock.json")
    verify_seal(scoring_root, name="feature_lock.json")
    binding = read_json(scoring_root / "run_binding.json")
    model_lock = digest_file(model_root / "artifact_lock.json")
    if binding.get("input_lock_sha256") != model_lock:
        raise ValueError("The 08G predictions and model-only package do not match")
    if binding.get("protocol") != json.loads(json.dumps(SCORING_PROTOCOL)):
        raise ValueError("The 08G scoring protocol differs from the locked implementation")
    scenes = tuple(scoring_lock.get("metadata", {}).get("scenes", ()))
    if not scenes:
        raise ValueError("The 08G prediction lock contains no scenes")
    model_index = read_json(model_root / "index.json")
    if not set(scenes) <= set(model_index.get("scenes", ())):
        raise ValueError("The model-only package does not contain all locked 08G scenes")
    if not set(REQUIRED_VIEWS) <= set(model_index.get("views", ())):
        raise ValueError("The model-only package is missing required identity views")
    if model_index.get("routes") != ["supplied", "automatic"]:
        raise ValueError("The model-only route contract changed")
    for scene in scenes:
        token(scene)
        source_rows = read_json(model_root / "scenes" / f"{scene}.json")
        score_rows = read_json(scoring_root / "scenes" / f"{scene}.json")
        source = {(row["route"], row["view"]): row for row in source_rows}
        scores = {(row["route"], row["view"]): row for row in score_rows}
        for view in REQUIRED_VIEWS:
            key = ("automatic", view)
            if key not in source or key not in scores:
                raise ValueError(f"Scene {scene} is missing automatic view {view}")
            source_ids = {item["candidate_id"] for item in source[key]["objects"]}
            score_ids = {item["candidate_id"] for item in scores[key]["objects"]}
            if source_ids != score_ids:
                raise ValueError("The 08G candidate identities differ from the model package")
            if digest_file(safe_path(model_root, source[key]["rgb"])) != source[key]["rgb_sha256"]:
                raise ValueError("An identity patch-association RGB checksum does not match")
    return {
        "scene_count": len(scenes),
        "scenes": list(scenes),
        "view_count": len(REQUIRED_VIEWS),
        "configuration_count_per_scene": (1 + len(PATCH_CONFIGURATIONS))
        * len(VIEW_SETS),
        "model_input_lock_sha256": model_lock,
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
    }


def _encode_scene(
    model_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    score_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    model_root: Path,
    encoder: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prepared_views = []
    fallback_total: dict[str, int] = defaultdict(int)
    for view in REQUIRED_VIEWS:
        model_row = model_rows[("automatic", view)]
        score_row = score_rows[("automatic", view)]
        image = np.asarray(Image.open(safe_path(model_root, model_row["rgb"])).convert("RGB"))
        score_by_id = {item["candidate_id"]: item for item in score_row["objects"]}
        crops = [
            prepare_masked_crop(image, decode_binary_mask(item["mask"]), configuration=CROP)
            for item in model_row["objects"]
        ]
        object_tokens: list[np.ndarray] = []
        object_weights: list[np.ndarray] = []
        view_fallbacks = {"foreground_patch_selection": 0}
        if crops:
            batches = []
            for start in range(0, len(crops), 4):
                batches.append(encoder.encode([crop.rgb for crop in crops[start : start + 4]]))
            from visual_peer_inspector.models.visual_encoder import EncodedBatch

            batch = EncodedBatch(
                global_features=np.concatenate([item.global_features for item in batches]),
                patch_features=tuple(value for item in batches for value in item.patch_features),
                patch_grid_size=batches[0].patch_grid_size,
                feature_dimension=batches[0].feature_dimension,
            )
            if batch.patch_grid_size is None:
                raise ValueError("Patch association requires spatial DINO tokens")
            for patch_features, crop in zip(batch.patch_features, crops, strict=True):
                selected, weights, fallback = foreground_patch_evidence(
                    patch_features,
                    crop,
                    minimum_coverage=0.50,
                )
                object_tokens.append(selected)
                object_weights.append(weights)
                view_fallbacks["foreground_patch_selection"] += int(fallback)
        for key, value in view_fallbacks.items():
            fallback_total[key] += value
        objects = []
        for index, model_object in enumerate(model_row["objects"]):
            candidate = str(model_object["candidate_id"])
            score_object = score_by_id[candidate]
            objects.append(
                {
                    "candidate_id": candidate,
                    "global_embedding": [float(value) for value in score_object["embedding"]],
                    "patch_tokens": object_tokens[index].tolist(),
                    "patch_weights": object_weights[index].tolist(),
                }
            )
        prepared_views.append({"view": view, "objects": objects})
    return prepared_views, dict(sorted(fallback_total.items()))


def _scene_prediction(
    scene: str,
    model_root: Path,
    scoring_root: Path,
    encoder: Any,
) -> dict[str, Any]:
    model_rows = {
        (row["route"], row["view"]): row
        for row in read_json(model_root / "scenes" / f"{scene}.json")
    }
    score_rows = {
        (row["route"], row["view"]): row
        for row in read_json(scoring_root / "scenes" / f"{scene}.json")
    }
    prepared_views, fallback_counts = _encode_scene(
        model_rows,
        score_rows,
        model_root,
        encoder,
    )
    by_view = {row["view"]: row for row in prepared_views}
    configurations = []
    for view_count, view_ids in VIEW_SETS.items():
        selected = [by_view[view] for view in view_ids]
        baseline_views = [
            {
                "view": row["view"],
                "objects": [
                    {
                        "candidate_id": item["candidate_id"],
                        "embedding": item["global_embedding"],
                    }
                    for item in row["objects"]
                ],
            }
            for row in selected
        ]
        baseline_association = associate_views(baseline_views, BASELINE_CONFIGURATION)
        baseline_association["accepted_links"] = []
        baseline_association["diagnostics"]["mean_edge_margin"] = None
        baseline_association["diagnostics"]["mean_patch_match_fraction"] = None
        baseline_association["diagnostics"]["mean_patch_entropy"] = None
        configurations.append(
            {
                "configuration_id": BASELINE_CONFIGURATION_ID,
                "family": "global_baseline",
                "view_count": view_count,
                "views": list(view_ids),
                "association": baseline_association,
            }
        )
        for configuration in PATCH_CONFIGURATIONS:
            configurations.append(
                {
                    "configuration_id": configuration.name,
                    "family": "foreground_patch_correspondence",
                    "view_count": view_count,
                    "views": list(view_ids),
                    "association": associate_patch_views(selected, configuration),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "route": "automatic",
        "fallback_counts": fallback_counts,
        "configurations": configurations,
    }


def _write_recovery(output: Path, binding: Mapping[str, Any]) -> tuple[Path, Path]:
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
    model_root: Path,
    scoring_root: Path,
    output: Path,
    source_root: Path,
    encoder_factory: Any,
    *,
    producer: str = "dinov2-small-patch-pooling",
    recovery_bundle: Path | None = None,
    recovery_manifest: Path | None = None,
) -> dict[str, Any]:
    """Generate all label-free identity configurations with scene checkpoints."""

    model_root, scoring_root, output, source_root = map(
        Path,
        (model_root, scoring_root, output, source_root),
    )
    audit = preflight(model_root, scoring_root)
    binding = _prediction_binding(
        model_root,
        scoring_root,
        source_root,
        producer=producer,
    )
    if recovery_bundle is not None or recovery_manifest is not None:
        if recovery_bundle is None or recovery_manifest is None:
            raise ValueError("Recovery requires both the ZIP and its manifest")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Restore recovery only into a new output directory")
        restore_recovery_bundle(
            bundle_path=Path(recovery_bundle),
            manifest_path=Path(recovery_manifest),
            destination_root=output,
            expected_binding=binding,
            allowed_top_levels=RECOVERY_TOP_LEVELS,
        )
        print(json.dumps({"event": "recovery_restored"}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_binding.json").is_file():
        if read_json(output / "run_binding.json") != json.loads(canonical(binding)):
            raise ValueError("Existing identity output belongs to different inputs or code")
    atomic_json(output / "run_binding.json", binding)
    atomic_json(output / "preflight.json", audit)
    atomic_json(output / "run_status.json", {"status": "running", "scene_count": audit["scene_count"]})
    binding_sha256 = sha256(canonical(binding)).hexdigest()
    encoder = None
    checkpoint_paths = []
    started = time.monotonic()
    try:
        for number, scene in enumerate(audit["scenes"], 1):
            checkpoint = output / "scenes" / f"{scene}.json"
            receipt = checkpoint.with_suffix(".receipt.json")
            if checkpoint.is_file() and receipt.is_file():
                saved = read_json(receipt)
                if saved.get("sha256") != digest_file(checkpoint):
                    raise ValueError("An identity scene checkpoint is corrupt")
                if saved.get("binding_sha256") != binding_sha256:
                    raise ValueError("An identity scene checkpoint has a different binding")
                print(json.dumps({"event": "scene_resumed", "scene_number": number, "total": audit["scene_count"]}), flush=True)
            else:
                if checkpoint.exists() or receipt.exists():
                    raise ValueError("An identity scene has an incomplete checkpoint pair")
                if encoder is None:
                    print(json.dumps({"event": "model_load_start", "revision": MODEL_REVISION}), flush=True)
                    encoder = encoder_factory()
                    print(json.dumps({"event": "model_ready"}), flush=True)
                record = _scene_prediction(scene, model_root, scoring_root, encoder)
                atomic_json(checkpoint, record)
                atomic_json(receipt, {"sha256": digest_file(checkpoint), "binding_sha256": binding_sha256})
                print(json.dumps({"event": "scene_complete", "scene_number": number, "total": audit["scene_count"]}), flush=True)
            checkpoint_paths.extend((checkpoint, receipt))
            atomic_json(output / "progress.json", {"complete_scenes": number, "expected_scenes": audit["scene_count"]})
            if number % 10 == 0 or number == audit["scene_count"]:
                _write_recovery(output, binding)
        seal(
            output,
            checkpoint_paths + [output / "run_binding.json", output / "preflight.json"],
            name="prediction_lock.json",
            metadata={
                "status": "complete",
                "scenes": audit["scenes"],
                "configuration_count_per_scene": audit["configuration_count_per_scene"],
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        atomic_json(output / "run_status.json", {"status": "complete", "scene_count": audit["scene_count"]})
        _write_recovery(output, binding)
        return verify_seal(output, name="prediction_lock.json")
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "scene_count": audit["scene_count"]})
        try:
            _write_recovery(output, binding)
        except Exception as recovery_error:
            print(json.dumps({"event": "recovery_export_failed", "error_type": type(recovery_error).__name__}), flush=True)
        raise


def _groups_from_configuration(configuration: Mapping[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    groups = []
    for track in configuration["association"]["tracks"]:
        groups.append(
            (
                str(track["identity_id"]),
                [(str(item["view_id"]), str(item["candidate_id"])) for item in track["members"]],
            )
        )
    return groups


def _identity_metrics(
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    truth: Mapping[str, Any],
    views: Sequence[str],
) -> dict[str, float]:
    correspondence = truth["correspondence"]
    linked_correct = 0
    linked_total = 0
    for _group_id, members in groups:
        for left, right in combinations(members, 2):
            left_reference = correspondence.get(f"automatic/{left[0]}/{left[1]}")
            right_reference = correspondence.get(f"automatic/{right[0]}/{right[1]}")
            if left_reference is None or right_reference is None:
                continue
            linked_total += 1
            linked_correct += int(left_reference == right_reference)
    by_reference: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, reference in correspondence.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "automatic" and parts[1] in views and reference is not None:
            by_reference[str(reference)].append((parts[1], parts[2]))
    possible = sum(len(items) * (len(items) - 1) // 2 for items in by_reference.values())
    precision = linked_correct / linked_total if linked_total else 0.0
    recall = linked_correct / possible if possible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    references = track_references(groups, correspondence, "automatic", truth["labels"])
    purities = [purity for reference, purity in references.values() if reference is not None]
    multi_member_tracks = 0
    switched_tracks = 0
    completeness = []
    for group_id, members in groups:
        mapped = {
            correspondence.get(f"automatic/{view}/{candidate}")
            for view, candidate in members
        }
        mapped.discard(None)
        if len(members) > 1:
            multi_member_tracks += 1
            switched_tracks += int(len(mapped) > 1)
        reference, _purity = references[group_id]
        if reference is not None:
            completeness.append(len({view for view, _candidate in members}) / len(views))
    return {
        "identity_link_precision": precision,
        "identity_link_recall": recall,
        "identity_link_f1": f1,
        "identity_track_purity": float(np.mean(purities)) if purities else 0.0,
        "identity_switch_rate": (
            switched_tracks / multi_member_tracks if multi_member_tracks else 0.0
        ),
        "track_completeness": float(np.mean(completeness)) if completeness else 0.0,
    }


def _bootstrap_differences(
    frame: pd.DataFrame,
    *,
    samples: int = 2_000,
    seed: int = 32_040,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (view_count, method), group in frame.groupby(["view_count", "method"]):
        baseline = BASELINE_CONFIGURATION_ID
        for measure in ("ap", "top1", "identity_link_f1"):
            pivot = group.pivot(index=["scene", "asset_group"], columns="configuration_id", values=measure)
            if baseline not in pivot.columns:
                continue
            for candidate in sorted(column for column in pivot.columns if column != baseline):
                paired = pivot[[baseline, candidate]].dropna()
                differences = {
                    asset: values[candidate].to_numpy() - values[baseline].to_numpy()
                    for asset, values in paired.reset_index().groupby("asset_group")
                }
                if not differences:
                    continue
                asset_names = sorted(differences)
                draws = []
                for _ in range(samples):
                    selected = rng.choice(asset_names, size=len(asset_names), replace=True)
                    draws.append(float(np.concatenate([differences[name] for name in selected]).mean()))
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "view_count": view_count,
                        "method": method,
                        "measure": measure,
                        "candidate": candidate,
                        "baseline": baseline,
                        "difference": float(paired[candidate].mean() - paired[baseline].mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "asset_groups": len(asset_names),
                        "scenes": len(paired),
                    }
                )
    return pd.DataFrame(rows)


def _render_summary_figures(
    summary: pd.DataFrame,
    link_coverage: pd.DataFrame,
    output: Path,
    *,
    scientific_result: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_root = output / "shareable" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = []
    primary = summary[(summary["view_count"] == 3) & (summary["method"] == PRIMARY_METHOD)].copy()
    primary = primary.sort_values(["top1", "ap", "identity_link_f1"], ascending=False).head(12)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(primary["configuration_id"], primary["top1"], color="#315f86")
    axis.invert_yaxis()
    title_prefix = "" if scientific_result else "Software validation only: "
    axis.set(xlabel="Top-1 accuracy", title=f"{title_prefix}three-view automatic-route comparison", xlim=(0, 1))
    figure.tight_layout()
    path = figure_root / "primary_top1_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for count, group in summary[summary["method"] == PRIMARY_METHOD].groupby("view_count"):
        axis.scatter(group["identity_link_f1"], group["top1"], label=f"{count} views", alpha=0.75)
    axis.set(xlabel="Identity link F1", ylabel="Top-1 accuracy", title=f"{title_prefix}identity quality and odd-one-out accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    path = figure_root / "identity_vs_top1.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    if not link_coverage.empty:
        figure, axis = plt.subplots(figsize=(8, 5.5))
        for configuration, group in link_coverage[link_coverage["view_count"] == 3].groupby("configuration_id"):
            axis.plot(group["retained_fraction"], group["wrong_link_rate"], marker="o", label=configuration)
        axis.set(
            xlabel="Retained automatic links",
            ylabel="Wrong-link rate",
            title=f"{title_prefix}three-view identity coverage and error",
            xlim=(0.45, 0.95),
            ylim=(0, 1),
        )
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = figure_root / "identity_coverage_vs_error.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def _winning_members(
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    truth: Mapping[str, Any],
    views: Sequence[str],
    method: str,
) -> tuple[set[tuple[str, str]], float]:
    score_by_view = {view: normalized_scores(rows[("automatic", view)]["objects"], method) for view in views}
    references = track_references(groups, truth["correspondence"], "automatic", truth["labels"])
    items = []
    member_map = {}
    for group_id, members in groups:
        values = [score_by_view[view][candidate] for view, candidate in members if candidate in score_by_view[view]]
        if not values:
            continue
        reference, purity = references[group_id]
        items.append({"id": group_id, "score": float(np.mean(values)), "reference": reference, "support": len(values), "purity": purity})
        member_map[group_id] = set(members)
    scored = metrics(items, truth["labels"])
    return member_map.get(scored["winner"], set()), float(scored["top1"])


def _render_gallery(
    model_root: Path,
    scoring_root: Path,
    evaluator_root: Path,
    prediction_root: Path,
    output: Path,
    scene_frame: pd.DataFrame,
    chosen_configuration: str,
) -> list[Path]:
    baseline = BASELINE_CONFIGURATION_ID
    method = PRIMARY_METHOD
    comparison = scene_frame[(scene_frame["view_count"] == 3) & (scene_frame["method"] == method)]
    pivot = comparison.pivot(index="scene", columns="configuration_id", values="top1")
    if baseline not in pivot or chosen_configuration not in pivot:
        return []
    pivot["change"] = pivot[chosen_configuration] - pivot[baseline]
    improves = list(pivot.sort_values("change", ascending=False).head(4).index)
    regressions = list(pivot.sort_values("change", ascending=True).head(4).index)
    fill = sorted(pivot.index, key=lambda value: sha256(f"identity-gallery/{value}".encode()).hexdigest())
    scenes = []
    for scene in improves + regressions + fill:
        if scene not in scenes:
            scenes.append(scene)
        if len(scenes) == 12:
            break
    gallery_root = output / "private" / "figures"
    gallery_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    saved = []
    for number, scene in enumerate(scenes, 1):
        model_rows = {(row["route"], row["view"]): row for row in read_json(model_root / "scenes" / f"{scene}.json")}
        score_rows = {(row["route"], row["view"]): row for row in read_json(scoring_root / "scenes" / f"{scene}.json")}
        truth = read_json(evaluator_root / "scenes" / f"{scene}.json")
        prediction = read_json(prediction_root / "scenes" / f"{scene}.json")
        config = next(
            item
            for item in prediction["configurations"]
            if item["configuration_id"] == chosen_configuration
            and item["view_count"] == 3
        )
        groups = _groups_from_configuration(config)
        winners, correct = _winning_members(groups, score_rows, truth, config["views"], method)
        panels = []
        for view in config["views"]:
            row = model_rows[("automatic", view)]
            image = Image.open(safe_path(model_root, row["rgb"])).convert("RGB")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_array = np.asarray(overlay).copy()
            draw = ImageDraw.Draw(image)
            for item in row["objects"]:
                candidate = str(item["candidate_id"])
                box = tuple(int(value) for value in item["box"])
                reference = truth["correspondence"].get(f"automatic/{view}/{candidate}")
                if truth["labels"].get(reference, 0) == 1:
                    draw.rectangle(box, outline=(36, 145, 73), width=5)
                else:
                    draw.rectangle(box, outline=(44, 105, 170), width=2)
                if (view, candidate) in winners:
                    mask = decode_binary_mask(item["mask"])
                    overlay_array[mask, :3] = (220, 45, 55)
                    overlay_array[mask, 3] = 85
                    draw.rectangle(box, outline=(220, 45, 55), width=5)
            blended = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_array, "RGBA")).convert("RGB")
            panels.append(blended)
        width = sum(image.width for image in panels)
        height = max(image.height for image in panels)
        combined = Image.new("RGB", (width, height), "white")
        left = 0
        for panel in panels:
            combined.paste(panel, (left, 0))
            left += panel.width
        file_name = f"identity_example_{number:02d}.png"
        path = gallery_root / file_name
        combined.save(path)
        saved.append(path)
        manifest.append(
            {
                "file": file_name,
                "selection_reason": "improvement" if scene in improves else "regression" if scene in regressions else "fixed_hash_sample",
                "top1_correct": correct,
            }
        )
    atomic_json(output / "private" / "gallery_manifest.json", {"examples": manifest, "legend": {"red": "selected identity", "green": "reference unusual object", "blue": "other automatic candidates"}})
    return saved


def evaluate(
    model_root: Path,
    scoring_root: Path,
    evaluator_root: Path,
    prediction_root: Path,
    output: Path,
) -> pd.DataFrame:
    """Open development labels only after predictions are locked and compare variants."""

    model_root, scoring_root, evaluator_root, prediction_root, output = map(
        Path,
        (model_root, scoring_root, evaluator_root, prediction_root, output),
    )
    prediction_lock = verify_seal(prediction_root, name="prediction_lock.json")
    verify_seal(model_root)
    verify_seal(scoring_root, name="prediction_lock.json")
    verify_seal(evaluator_root)
    evaluator_index = read_json(evaluator_root / "index.json")
    binding = read_json(prediction_root / "run_binding.json")
    producer = str(binding.get("producer", "unknown"))
    scientific_result = producer != "contract-encoder"
    if evaluator_index.get("model_lock_sha256") != binding.get("model_input_lock_sha256"):
        raise ValueError("The identity predictions and evaluator package do not match")
    scenes = tuple(prediction_lock.get("metadata", {}).get("scenes", ()))
    if not set(scenes) <= set(evaluator_index.get("scenes", ())):
        raise ValueError("The evaluator package is missing identity patch-association scenes")
    evaluation_binding = {
        "prediction_lock_sha256": digest_file(prediction_root / "prediction_lock.json"),
        "evaluator_lock_sha256": digest_file(evaluator_root / "artifact_lock.json"),
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "evaluation_source_sha256": digest_file(Path(__file__)),
        "protocol": PROTOCOL,
        "prediction_producer": producer,
        "scientific_result": scientific_result,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_binding.json").is_file() and read_json(output / "evaluation_binding.json") != json.loads(canonical(evaluation_binding)):
        raise ValueError("Existing evaluation output belongs to different inputs or code")
    atomic_json(output / "evaluation_binding.json", evaluation_binding)
    atomic_json(output / "run_status.json", {"status": "running", "scene_count": len(scenes)})
    scene_rows = []
    link_rows = []
    try:
        for number, scene in enumerate(scenes, 1):
            predictions = read_json(prediction_root / "scenes" / f"{scene}.json")
            truth = read_json(evaluator_root / "scenes" / f"{scene}.json")
            rows = {(row["route"], row["view"]): row for row in read_json(scoring_root / "scenes" / f"{scene}.json")}
            common = {
                "scene": scene,
                "asset_group": truth["asset_group"],
                "category": truth["category"],
                "families": "+".join(truth["families"]),
            }
            for configuration in predictions["configurations"]:
                groups = _groups_from_configuration(configuration)
                identity_values = _identity_metrics(groups, truth, configuration["views"])
                diagnostics = configuration["association"]["diagnostics"]
                for link in configuration["association"].get("accepted_links", []):
                    left = link["left"]
                    right = link["right"]
                    left_reference = truth["correspondence"].get(
                        f"automatic/{left['view_id']}/{left['candidate_id']}"
                    )
                    right_reference = truth["correspondence"].get(
                        f"automatic/{right['view_id']}/{right['candidate_id']}"
                    )
                    confidence = (
                        float(link["margin"])
                        + 0.02 * float(link["cycle_support"])
                        + 0.05 * float(link["patch_match_fraction"])
                        - 0.05 * float(link["patch_entropy"])
                    )
                    link_rows.append(
                        {
                            "scene": scene,
                            "asset_group": truth["asset_group"],
                            "configuration_id": configuration["configuration_id"],
                            "view_count": configuration["view_count"],
                            "confidence": confidence,
                            "correct": float(
                                left_reference is not None
                                and right_reference is not None
                                and left_reference == right_reference
                            ),
                        }
                    )
                for method in RANKING_METHODS:
                    items = fused_ranking(
                        rows,
                        truth,
                        "automatic",
                        configuration["views"],
                        method,
                        identity="predicted",
                        precomputed=groups,
                    )
                    scene_rows.append(
                        common
                        | {
                            "configuration_id": configuration["configuration_id"],
                            "family": configuration["family"],
                            "view_count": configuration["view_count"],
                            "method": method,
                        }
                        | diagnostics
                        | identity_values
                        | metrics(items, truth["labels"])
                    )
            atomic_json(output / "progress.json", {"evaluated_scenes": number, "expected_scenes": len(scenes)})
            print(json.dumps({"event": "scene_evaluated", "scene_number": number, "total": len(scenes)}), flush=True)
        frame = pd.DataFrame(scene_rows)
        link_frame = pd.DataFrame(link_rows)
        private = output / "private"
        shareable = output / "shareable"
        private.mkdir(parents=True, exist_ok=True)
        shareable.mkdir(parents=True, exist_ok=True)
        frame.to_csv(private / "scene_metrics.csv", index=False)
        link_frame.to_csv(private / "accepted_link_metrics.csv", index=False)
        group_keys = ["configuration_id", "family", "view_count", "method"]
        summary = frame.groupby(group_keys).agg(
            scenes=("scene", "size"),
            asset_groups=("asset_group", "nunique"),
            ap=("ap", "mean"),
            top1=("top1", "mean"),
            recall_at_2=("recall_at_2", "mean"),
            auroc=("auroc", "mean"),
            identity_link_precision=("identity_link_precision", "mean"),
            identity_link_recall=("identity_link_recall", "mean"),
            identity_link_f1=("identity_link_f1", "mean"),
            identity_track_purity=("identity_track_purity", "mean"),
            identity_switch_rate=("identity_switch_rate", "mean"),
            track_completeness=("track_completeness", "mean"),
            linked_node_fraction=("linked_node_fraction", "mean"),
            singleton_fraction=("singleton_fraction", "mean"),
            mean_track_support=("mean_track_support", "mean"),
            mean_edge_margin=("mean_edge_margin", "mean"),
            mean_patch_match_fraction=("mean_patch_match_fraction", "mean"),
            mean_patch_entropy=("mean_patch_entropy", "mean"),
            winner_support=("winner_support", "mean"),
            winner_purity=("winner_purity", "mean"),
        ).reset_index()
        summary.to_csv(shareable / "patch_association_summary.csv", index=False)
        frame.groupby(group_keys + ["category"])[["ap", "top1", "identity_link_f1"]].mean().reset_index().to_csv(shareable / "category_summary.csv", index=False)
        frame.groupby(group_keys + ["families"])[["ap", "top1", "identity_link_f1"]].mean().reset_index().to_csv(shareable / "family_summary.csv", index=False)
        fallback_rows = []
        for scene in scenes:
            record = read_json(prediction_root / "scenes" / f"{scene}.json")
            fallback_rows.append({"scene": scene, **record["fallback_counts"]})
        fallback_frame = pd.DataFrame(fallback_rows)
        fallback_frame.drop(columns=["scene"]).sum().rename("fallback_count").reset_index(name="fallback_count").rename(columns={"index": "coverage"}).to_csv(shareable / "patch_fallback_summary.csv", index=False)

        primary = summary[(summary["view_count"] == 3) & (summary["method"] == PRIMARY_METHOD)].copy()
        primary = primary.sort_values(["identity_link_f1", "ap", "top1", "configuration_id"], ascending=[False, False, False, True])
        primary.to_csv(shareable / "primary_candidate_ranking.csv", index=False)
        baseline = primary[primary["configuration_id"] == BASELINE_CONFIGURATION_ID].iloc[0]
        intervals = _bootstrap_differences(frame)
        intervals.to_csv(shareable / "paired_asset_intervals.csv", index=False)
        coverage_rows = []
        for (configuration_id, view_count), group in link_frame.groupby(
            ["configuration_id", "view_count"]
        ):
            ordered = group.sort_values(
                ["confidence", "scene"], ascending=[False, True]
            )
            for retained_fraction in (0.50, 0.70, 0.90, 1.00):
                retained = ordered.head(max(1, math.ceil(len(ordered) * retained_fraction)))
                coverage_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "view_count": view_count,
                        "retained_fraction": retained_fraction,
                        "retained_links": len(retained),
                        "available_links": len(ordered),
                        "wrong_link_rate": float(1.0 - retained["correct"].mean()),
                    }
                )
        link_coverage = pd.DataFrame(coverage_rows)
        link_coverage.to_csv(shareable / "identity_coverage_vs_error.csv", index=False)
        interval_index = intervals[
            (intervals["view_count"] == 3) & (intervals["method"] == PRIMARY_METHOD)
        ].set_index(["candidate", "measure"])
        eligible = []
        for row in primary.itertuples():
            if row.configuration_id == BASELINE_CONFIGURATION_ID:
                continue
            ap_interval = interval_index.loc[(row.configuration_id, "ap")]
            identity_interval = interval_index.loc[(row.configuration_id, "identity_link_f1")]
            if (
                float(ap_interval["ci_low"]) > 0.0
                and float(identity_interval["ci_low"]) > 0.0
                and row.top1 >= baseline.top1
                and row.recall_at_2 >= baseline.recall_at_2
                and row.linked_node_fraction >= baseline.linked_node_fraction
                and row.singleton_fraction <= baseline.singleton_fraction
            ):
                eligible.append(row)
        selected = (
            max(eligible, key=lambda row: (row.identity_link_f1, row.ap, row.top1))
            if eligible
            else None
        )
        review_rows = primary[primary["configuration_id"] != BASELINE_CONFIGURATION_ID]
        review_candidate = str(review_rows.sort_values(["identity_link_f1", "ap", "top1"], ascending=False).iloc[0]["configuration_id"])
        chosen = selected.configuration_id if selected else BASELINE_CONFIGURATION_ID
        atomic_json(
            shareable / "development_decision.json",
            {
                "status": "promote" if selected else "retain_current",
                "selected_configuration": chosen,
                "review_candidate": review_candidate,
                "baseline": BASELINE_CONFIGURATION_ID,
                "primary_route": "automatic masks, predicted identity, three views",
                "primary_scorer": PRIMARY_METHOD,
                "promotion": (
                    "Candidate passed the joint identity, ranking and anti-fragmentation rule."
                    if selected
                    else "No candidate passed the joint identity, ranking and anti-fragmentation rule."
                ),
                "protected_data_used": False,
                "split_changed": False,
                "prediction_producer": producer,
                "scientific_result": scientific_result,
            },
        )
        if not scientific_result:
            decision = read_json(shareable / "development_decision.json")
            decision["status"] = "software_validation_only"
            decision["promotion"] = "Contract-encoder outputs cannot select or promote a scientific method."
            atomic_json(shareable / "development_decision.json", decision)
        figure_paths = _render_summary_figures(
            summary,
            link_coverage,
            output,
            scientific_result=scientific_result,
        )
        gallery_paths = _render_gallery(model_root, scoring_root, evaluator_root, prediction_root, output, frame, review_candidate)
        atomic_json(
            shareable / "evaluation_receipt.json",
            {
                "schema_version": SCHEMA_VERSION,
                "scene_count": len(scenes),
                "asset_group_count": int(frame["asset_group"].nunique()),
                "prediction_lock_sha256": evaluation_binding["prediction_lock_sha256"],
                "evaluator_lock_sha256": evaluation_binding["evaluator_lock_sha256"],
                "scope": "development only",
                "prediction_producer": producer,
                "scientific_result": scientific_result,
                "candidate_count": len(primary),
                "figure_count": len(figure_paths),
                "private_gallery_count": len(gallery_paths),
            },
        )
        atomic_json(output / "run_status.json", {"status": "complete", "scene_count": len(scenes)})
        seal(output, [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_lock.json"])
        return summary
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "scene_count": len(scenes)})
        raise


__all__ = [
    "BASELINE_CONFIGURATION_ID",
    "MODEL_REVISION",
    "PROTOCOL",
    "RANKING_METHODS",
    "REQUIRED_VIEWS",
    "VIEW_SETS",
    "evaluate",
    "predict",
    "preflight",
]
