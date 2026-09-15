"""Run the bounded development-only cross-view association study."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from visual_peer_inspector.representation.identity_association import CONFIGURATIONS, associate_views
from visual_peer_inspector.workflows.comparison_evaluation import fused_ranking, metrics, track_references
from visual_peer_inspector.workflows.comparison_io import (
    atomic_json,
    canonical,
    digest_file,
    read_json,
    safe_path,
    seal,
    verify_seal,
)


SCHEMA_VERSION = "visual-peer-identity-association/v1"
STAGE = "development_identity_association"
VIEW_SETS = {
    2: ("V00", "V10"),
    3: ("V00", "V07", "V14"),
    5: ("V00", "V04", "V08", "V12", "V16"),
}
RANKING_METHODS = ("P1_current", "forward_02_cov70", "P2_consensus")
BASELINE_CONFIGURATION = "current_hungarian_035"
PRIMARY_METHOD = "forward_02_cov70"
PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "scope": "existing nested 100-scene development screen",
    "route": "automatic",
    "split_change": False,
    "new_model_inference": False,
    "descriptor": "sealed DINOv2-small global embedding from Notebook 08G",
    "view_sets": {str(count): list(views) for count, views in VIEW_SETS.items()},
    "configurations": [configuration.__dict__ for configuration in CONFIGURATIONS],
    "ranking_methods": list(RANKING_METHODS),
    "selection_rule": (
        "Promote only when asset-grouped paired intervals support improvements in "
        "identity F1 and scene AP together, with no mean regression in top-1 or recall at two."
    ),
    "protected_roles_used": False,
}


def _implementation_digest(source_root: Path) -> str:
    paths = sorted((source_root / "src").rglob("*.py"))
    paths += sorted((source_root / "scripts").glob("*identity_association*.py"))
    if not paths:
        raise ValueError("Association source inventory is empty")
    inventory = {str(path.relative_to(source_root)): digest_file(path) for path in paths}
    return sha256(canonical(inventory)).hexdigest()


def preflight(scoring_root: Path) -> dict[str, Any]:
    scoring_root = Path(scoring_root)
    lock = verify_seal(scoring_root, name="prediction_lock.json")
    scenes = tuple(lock.get("metadata", {}).get("scenes", ()))
    if len(scenes) != 100 or len(set(scenes)) != len(scenes):
        raise ValueError("The association study requires the sealed 100-scene 08G screen")
    required_views = set(view for views in VIEW_SETS.values() for view in views)
    for scene in scenes:
        rows = read_json(scoring_root / "scenes" / f"{scene}.json")
        automatic = {str(row["view"]): row for row in rows if row["route"] == "automatic"}
        if not required_views <= set(automatic):
            raise ValueError(f"{scene} is missing required automatic-route views")
        for view in required_views:
            for item in automatic[view]["objects"]:
                embedding = np.asarray(item["embedding"], dtype=float)
                if embedding.ndim != 1 or embedding.size == 0 or not np.all(np.isfinite(embedding)):
                    raise ValueError(f"{scene}/{view} contains an invalid embedding")
                if any(method not in item["scores"] for method in RANKING_METHODS):
                    raise ValueError(f"{scene}/{view} is missing a required score field")
    return {
        "status": "passed",
        "scene_count": len(scenes),
        "scenes": list(scenes),
        "required_views": sorted(required_views),
        "configuration_count": len(CONFIGURATIONS) * len(VIEW_SETS),
        "new_model_inference": False,
    }


def _binding(scoring_root: Path, source_root: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "implementation_sha256": _implementation_digest(source_root),
        "protocol": PROTOCOL,
    }


def _scene_prediction(scene: str, scoring_root: Path) -> dict[str, Any]:
    rows = {
        (str(row["route"]), str(row["view"])): row
        for row in read_json(scoring_root / "scenes" / f"{scene}.json")
    }
    configurations = []
    for view_count, view_ids in VIEW_SETS.items():
        prepared = [
            {
                "view": view,
                "objects": [
                    {"candidate_id": item["candidate_id"], "embedding": item["embedding"]}
                    for item in rows[("automatic", view)]["objects"]
                ],
            }
            for view in view_ids
        ]
        for configuration in CONFIGURATIONS:
            result = associate_views(prepared, configuration)
            configurations.append(
                {
                    "configuration_id": configuration.name,
                    "strategy": configuration.strategy,
                    "maximum_distance": configuration.maximum_distance,
                    "view_count": view_count,
                    "views": list(view_ids),
                    "association": result,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "route": "automatic",
        "configurations": configurations,
    }


def predict(scoring_root: Path, output: Path, source_root: Path) -> dict[str, Any]:
    """Lock label-free association candidates with scene-level resume receipts."""

    scoring_root, output, source_root = map(Path, (scoring_root, output, source_root))
    audit = preflight(scoring_root)
    binding = _binding(scoring_root, source_root)
    binding_digest = sha256(canonical(binding)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_binding.json").is_file() and read_json(output / "run_binding.json") != json.loads(canonical(binding)):
        raise ValueError("Existing association output belongs to different inputs or code")
    atomic_json(output / "run_binding.json", binding)
    atomic_json(output / "preflight.json", audit)
    atomic_json(output / "run_status.json", {"status": "running", "scene_count": audit["scene_count"]})
    checkpoint_paths = []
    started = time.monotonic()
    try:
        for number, scene in enumerate(audit["scenes"], 1):
            checkpoint = output / "scenes" / f"{scene}.json"
            receipt = checkpoint.with_suffix(".receipt.json")
            if checkpoint.is_file() and receipt.is_file():
                saved = read_json(receipt)
                if saved.get("sha256") != digest_file(checkpoint) or saved.get("binding_sha256") != binding_digest:
                    raise ValueError(f"{scene} has a corrupt or mismatched checkpoint")
                event = "scene_resumed"
            else:
                if checkpoint.exists() or receipt.exists():
                    raise ValueError(f"{scene} has an incomplete checkpoint pair")
                atomic_json(checkpoint, _scene_prediction(scene, scoring_root))
                atomic_json(receipt, {"sha256": digest_file(checkpoint), "binding_sha256": binding_digest})
                event = "scene_complete"
            checkpoint_paths.extend((checkpoint, receipt))
            atomic_json(output / "progress.json", {"complete_scenes": number, "expected_scenes": audit["scene_count"]})
            print(json.dumps({"event": event, "scene_number": number, "total": audit["scene_count"]}), flush=True)
        seal(
            output,
            checkpoint_paths + [output / "run_binding.json", output / "preflight.json"],
            name="prediction_lock.json",
            metadata={
                "status": "complete",
                "scenes": audit["scenes"],
                "elapsed_seconds": time.monotonic() - started,
                "configuration_count_per_scene": audit["configuration_count"],
            },
        )
        atomic_json(output / "run_status.json", {"status": "complete", "scene_count": audit["scene_count"]})
        return verify_seal(output, name="prediction_lock.json")
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


def _groups(configuration: Mapping[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            str(track["identity_id"]),
            [(str(item["view_id"]), str(item["candidate_id"])) for item in track["members"]],
        )
        for track in configuration["association"]["tracks"]
    ]


def _identity_metrics(
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    truth: Mapping[str, Any],
    views: Sequence[str],
) -> dict[str, float]:
    linked_correct = 0
    linked_total = 0
    for _group_id, members in groups:
        for left, right in combinations(members, 2):
            left_reference = truth["correspondence"].get(f"automatic/{left[0]}/{left[1]}")
            right_reference = truth["correspondence"].get(f"automatic/{right[0]}/{right[1]}")
            if left_reference is None or right_reference is None:
                continue
            linked_total += 1
            linked_correct += int(left_reference == right_reference)
    by_reference: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, reference in truth["correspondence"].items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "automatic" and parts[1] in views and reference is not None:
            by_reference[str(reference)].append((parts[1], parts[2]))
    possible = sum(len(items) * (len(items) - 1) // 2 for items in by_reference.values())
    precision = linked_correct / linked_total if linked_total else 0.0
    recall = linked_correct / possible if possible else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    references = track_references(groups, truth["correspondence"], "automatic", truth["labels"])
    purities = [purity for reference, purity in references.values() if reference is not None]
    return {
        "identity_link_precision": precision,
        "identity_link_recall": recall,
        "identity_link_f1": f1,
        "identity_track_purity": float(np.mean(purities)) if purities else 0.0,
    }


def _paired_intervals(frame: pd.DataFrame, samples: int = 2_000, seed: int = 32_040) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for (view_count, method), group in frame.groupby(["view_count", "method"]):
        for measure in ("ap", "top1", "recall_at_2", "identity_link_f1"):
            pivot = group.pivot(index=["scene", "asset_group"], columns="configuration_id", values=measure)
            if BASELINE_CONFIGURATION not in pivot:
                continue
            for candidate in sorted(column for column in pivot if column != BASELINE_CONFIGURATION):
                paired = pivot[[BASELINE_CONFIGURATION, candidate]].dropna().reset_index()
                differences = {
                    asset: part[candidate].to_numpy() - part[BASELINE_CONFIGURATION].to_numpy()
                    for asset, part in paired.groupby("asset_group")
                }
                if not differences:
                    continue
                assets = sorted(differences)
                draws = []
                for _ in range(samples):
                    chosen = rng.choice(assets, len(assets), replace=True)
                    draws.append(float(np.concatenate([differences[asset] for asset in chosen]).mean()))
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    {
                        "view_count": view_count,
                        "method": method,
                        "measure": measure,
                        "candidate": candidate,
                        "baseline": BASELINE_CONFIGURATION,
                        "difference": float(paired[candidate].mean() - paired[BASELINE_CONFIGURATION].mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "asset_groups": len(assets),
                        "scenes": len(paired),
                    }
                )
    return pd.DataFrame(rows)


def _render_figures(summary: pd.DataFrame, intervals: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    root = output / "shareable" / "figures"
    root.mkdir(parents=True, exist_ok=True)
    primary = summary[(summary["view_count"] == 3) & (summary["method"] == PRIMARY_METHOD)].copy()
    paths = []

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    label_offsets = {
        "current_hungarian_035": (-42, -12),
        "reciprocal_020": (5, 6),
        "reciprocal_035": (5, -12),
        "cycle_pruned_035": (5, 6),
        "support_consensus_035": (5, 8),
    }
    short_labels = {
        "current_hungarian_035": "current",
        "reciprocal_020": "reciprocal 0.20",
        "reciprocal_035": "reciprocal 0.35",
        "cycle_pruned_035": "cycle-pruned",
        "support_consensus_035": "support consensus",
    }
    for _, row in primary.iterrows():
        axis.scatter(row["identity_link_f1"], row["ap"], s=75)
        axis.annotate(
            short_labels[row["configuration_id"]],
            (row["identity_link_f1"], row["ap"]),
            fontsize=8,
            xytext=label_offsets[row["configuration_id"]],
            textcoords="offset points",
        )
    axis.set(xlabel="Identity link F1", ylabel="Scene average precision", title="Identity quality and odd-one-out ranking")
    figure.tight_layout()
    paths.append(root / "identity_f1_vs_scene_ap.png")
    figure.savefig(paths[-1], dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = primary.sort_values("identity_link_f1", ascending=False)
    positions = np.arange(len(order))
    axes[0].barh(positions - 0.18, order["identity_link_precision"], height=0.34, label="Precision", color="#315f86")
    axes[0].barh(positions + 0.18, order["identity_link_recall"], height=0.34, label="Recall", color="#8bb8d9")
    axes[0].set(yticks=positions, yticklabels=order["configuration_id"], title="Identity link precision and recall", xlabel="Value", xlim=(0, 1))
    axes[0].legend()
    axes[1].barh(order["configuration_id"], order["top1"], color="#477a5b")
    axes[1].set(title="Three-view top-1 accuracy", xlabel="Accuracy", xlim=(0, 1))
    figure.tight_layout()
    paths.append(root / "primary_association_comparison.png")
    figure.savefig(paths[-1], dpi=180)
    plt.close(figure)

    interval_rows = intervals[(intervals["view_count"] == 3) & (intervals["method"] == PRIMARY_METHOD) & (intervals["measure"].isin(["ap", "identity_link_f1"]))]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    labels = [f"{row.candidate}: {row.measure}" for row in interval_rows.itertuples()]
    values = interval_rows["difference"].to_numpy()
    errors = np.vstack((values - interval_rows["ci_low"].to_numpy(), interval_rows["ci_high"].to_numpy() - values))
    positions = np.arange(len(labels))
    axis.errorbar(values, positions, xerr=errors, fmt="o", color="#315f86", capsize=3)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(yticks=positions, yticklabels=labels, xlabel="Paired difference from current method", title="Asset-grouped 95% intervals")
    figure.tight_layout()
    paths.append(root / "paired_primary_intervals.png")
    figure.savefig(paths[-1], dpi=180)
    plt.close(figure)
    return paths


def _winner_members(
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    truth: Mapping[str, Any],
    views: Sequence[str],
) -> tuple[set[tuple[str, str]], float]:
    items = fused_ranking(rows, truth, "automatic", views, PRIMARY_METHOD, identity="predicted", precomputed=groups)
    scored = metrics(items, truth["labels"])
    member_map = {identity_id: set(members) for identity_id, members in groups}
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
    comparison = scene_frame[(scene_frame["view_count"] == 3) & (scene_frame["method"] == PRIMARY_METHOD)]
    pivot = comparison.pivot(index="scene", columns="configuration_id", values="top1")
    if BASELINE_CONFIGURATION not in pivot or chosen_configuration not in pivot:
        return []
    pivot["change"] = pivot[chosen_configuration] - pivot[BASELINE_CONFIGURATION]
    ordered = list(pivot.sort_values("change", ascending=False).head(4).index)
    ordered += list(pivot.sort_values("change", ascending=True).head(4).index)
    ordered += sorted(pivot.index, key=lambda scene: sha256(f"08j-gallery/{scene}".encode()).hexdigest())
    scenes = list(dict.fromkeys(ordered))[:12]
    root = output / "private" / "figures"
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    saved = []
    for number, scene in enumerate(scenes, 1):
        model_rows = {(row["route"], row["view"]): row for row in read_json(model_root / "scenes" / f"{scene}.json")}
        score_rows = {(row["route"], row["view"]): row for row in read_json(scoring_root / "scenes" / f"{scene}.json")}
        truth = read_json(evaluator_root / "scenes" / f"{scene}.json")
        prediction = read_json(prediction_root / "scenes" / f"{scene}.json")
        configuration = next(item for item in prediction["configurations"] if item["configuration_id"] == chosen_configuration and item["view_count"] == 3)
        groups = _groups(configuration)
        winners, correct = _winner_members(groups, score_rows, truth, configuration["views"])
        panels = []
        for view in configuration["views"]:
            row = model_rows[("automatic", view)]
            image = Image.open(safe_path(model_root, row["rgb"])).convert("RGB")
            draw = ImageDraw.Draw(image)
            for item in row["objects"]:
                candidate = str(item["candidate_id"])
                box = tuple(int(value) for value in item["box"])
                reference = truth["correspondence"].get(f"automatic/{view}/{candidate}")
                colour, width = ((41, 145, 74), 5) if truth["labels"].get(reference, 0) == 1 else ((55, 105, 170), 2)
                if (view, candidate) in winners:
                    colour, width = (220, 45, 55), 5
                draw.rectangle(box, outline=colour, width=width)
            panels.append(image)
        canvas = Image.new("RGB", (sum(image.width for image in panels), max(image.height for image in panels)), "white")
        left = 0
        for panel in panels:
            canvas.paste(panel, (left, 0))
            left += panel.width
        path = root / f"association_example_{number:02d}.png"
        canvas.save(path)
        saved.append(path)
        manifest.append({"file": path.name, "scene": scene, "top1_correct": correct, "change": float(pivot.loc[scene, "change"])})
    atomic_json(output / "private" / "gallery_manifest.json", {"legend": {"red": "selected track", "green": "reference unusual object", "blue": "other proposal"}, "examples": manifest})
    return saved


def evaluate(
    model_root: Path,
    scoring_root: Path,
    evaluator_root: Path,
    prediction_root: Path,
    output: Path,
) -> pd.DataFrame:
    """Open labels only after prediction locking and compare all bounded candidates."""

    model_root, scoring_root, evaluator_root, prediction_root, output = map(Path, (model_root, scoring_root, evaluator_root, prediction_root, output))
    verify_seal(model_root)
    verify_seal(scoring_root, name="prediction_lock.json")
    verify_seal(evaluator_root)
    prediction_lock = verify_seal(prediction_root, name="prediction_lock.json")
    scenes = tuple(prediction_lock.get("metadata", {}).get("scenes", ()))
    evaluator_index = read_json(evaluator_root / "index.json")
    if not set(scenes) <= set(evaluator_index.get("scenes", ())):
        raise ValueError("The evaluator resource is missing association-study scenes")
    binding = {
        "schema_version": SCHEMA_VERSION,
        "prediction_lock_sha256": digest_file(prediction_root / "prediction_lock.json"),
        "scoring_prediction_lock_sha256": digest_file(scoring_root / "prediction_lock.json"),
        "evaluator_lock_sha256": digest_file(evaluator_root / "artifact_lock.json"),
        "protocol": PROTOCOL,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_binding.json").is_file() and read_json(output / "evaluation_binding.json") != json.loads(canonical(binding)):
        raise ValueError("Existing evaluation output belongs to different inputs")
    atomic_json(output / "evaluation_binding.json", binding)
    atomic_json(output / "run_status.json", {"status": "running", "scene_count": len(scenes)})
    scene_rows = []
    try:
        for number, scene in enumerate(scenes, 1):
            prediction = read_json(prediction_root / "scenes" / f"{scene}.json")
            truth = read_json(evaluator_root / "scenes" / f"{scene}.json")
            rows = {(row["route"], row["view"]): row for row in read_json(scoring_root / "scenes" / f"{scene}.json")}
            for configuration in prediction["configurations"]:
                groups = _groups(configuration)
                identity_values = _identity_metrics(groups, truth, configuration["views"])
                diagnostics = configuration["association"]["diagnostics"]
                for method in RANKING_METHODS:
                    ranked = fused_ranking(rows, truth, "automatic", configuration["views"], method, identity="predicted", precomputed=groups)
                    scene_rows.append(
                        {
                            "scene": scene,
                            "asset_group": truth["asset_group"],
                            "category": truth["category"],
                            "families": "+".join(truth["families"]),
                            "configuration_id": configuration["configuration_id"],
                            "strategy": configuration["strategy"],
                            "maximum_distance": configuration["maximum_distance"],
                            "view_count": configuration["view_count"],
                            "method": method,
                        }
                        | diagnostics
                        | identity_values
                        | metrics(ranked, truth["labels"])
                    )
            atomic_json(output / "progress.json", {"evaluated_scenes": number, "expected_scenes": len(scenes)})
            print(json.dumps({"event": "scene_evaluated", "scene_number": number, "total": len(scenes)}), flush=True)
        frame = pd.DataFrame(scene_rows)
        private, shareable = output / "private", output / "shareable"
        private.mkdir(parents=True, exist_ok=True)
        shareable.mkdir(parents=True, exist_ok=True)
        frame.to_csv(private / "scene_metrics.csv", index=False)
        keys = ["configuration_id", "strategy", "maximum_distance", "view_count", "method"]
        measures = [
            "ap", "top1", "recall_at_2", "auroc", "identity_link_precision", "identity_link_recall",
            "identity_link_f1", "identity_track_purity", "linked_node_fraction", "singleton_fraction",
            "mean_track_support", "winner_support", "winner_purity",
        ]
        summary = frame.groupby(keys).agg(
            scenes=("scene", "size"),
            asset_groups=("asset_group", "nunique"),
            **{measure: (measure, "mean") for measure in measures},
        ).reset_index()
        summary.to_csv(shareable / "association_summary.csv", index=False)
        intervals = _paired_intervals(frame)
        intervals.to_csv(shareable / "paired_asset_intervals.csv", index=False)
        frame.groupby(keys + ["category"])[["ap", "top1", "identity_link_f1"]].mean().reset_index().to_csv(shareable / "category_summary.csv", index=False)
        frame.groupby(keys + ["families"])[["ap", "top1", "identity_link_f1"]].mean().reset_index().to_csv(shareable / "family_summary.csv", index=False)
        primary = summary[(summary["view_count"] == 3) & (summary["method"] == PRIMARY_METHOD)].copy()
        primary = primary.sort_values(["ap", "identity_link_f1", "top1", "configuration_id"], ascending=[False, False, False, True])
        primary.to_csv(shareable / "primary_candidate_ranking.csv", index=False)
        baseline = primary[primary["configuration_id"] == BASELINE_CONFIGURATION].iloc[0]
        interval_index = intervals[(intervals["view_count"] == 3) & (intervals["method"] == PRIMARY_METHOD)].set_index(["candidate", "measure"])
        eligible = []
        for row in primary.itertuples():
            if row.configuration_id == BASELINE_CONFIGURATION:
                continue
            ap_interval = interval_index.loc[(row.configuration_id, "ap")]
            identity_interval = interval_index.loc[(row.configuration_id, "identity_link_f1")]
            if (
                float(ap_interval["ci_low"]) > 0.0
                and float(identity_interval["ci_low"]) > 0.0
                and row.top1 >= baseline.top1
                and row.recall_at_2 >= baseline.recall_at_2
            ):
                eligible.append(row)
        selected = max(eligible, key=lambda row: (row.ap, row.identity_link_f1, row.top1)) if eligible else None
        review_candidate = str(primary.iloc[0]["configuration_id"])
        decision = {
            "status": "promote" if selected else "retain_current",
            "selected_configuration": selected.configuration_id if selected else BASELINE_CONFIGURATION,
            "review_candidate": review_candidate,
            "baseline_configuration": BASELINE_CONFIGURATION,
            "primary_method": PRIMARY_METHOD,
            "primary_view_count": 3,
            "reason": (
                "A candidate improved identity F1 and scene AP with positive asset-grouped intervals and no mean top-1 or recall-at-two regression."
                if selected
                else "No candidate satisfied the joint identity and ranking rule. Retain the current association method and stop this refinement line."
            ),
            "protected_roles_used": False,
            "split_changed": False,
        }
        atomic_json(shareable / "development_decision.json", decision)
        figure_paths = _render_figures(summary, intervals, output)
        gallery_paths = _render_gallery(model_root, scoring_root, evaluator_root, prediction_root, output, frame, review_candidate)
        atomic_json(
            shareable / "evaluation_receipt.json",
            {
                "schema_version": SCHEMA_VERSION,
                "scene_count": len(scenes),
                "asset_group_count": int(frame["asset_group"].nunique()),
                "configuration_count": len(CONFIGURATIONS),
                "view_set_count": len(VIEW_SETS),
                "figure_count": len(figure_paths),
                "private_gallery_count": len(gallery_paths),
                "prediction_lock_sha256": binding["prediction_lock_sha256"],
                "evaluator_lock_sha256": binding["evaluator_lock_sha256"],
                "scope": "development only",
            },
        )
        atomic_json(output / "run_status.json", {"status": "complete", "scene_count": len(scenes)})
        seal(output, [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_lock.json"])
        return summary
    except Exception as error:
        atomic_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


__all__ = [
    "BASELINE_CONFIGURATION",
    "PRIMARY_METHOD",
    "PROTOCOL",
    "RANKING_METHODS",
    "VIEW_SETS",
    "evaluate",
    "predict",
    "preflight",
]
