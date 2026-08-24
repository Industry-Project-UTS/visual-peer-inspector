#!/usr/bin/env python3
"""Evaluate frozen DS0 proposals against protected supplied-mask geometry."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.evaluation.proposal_reference import (
    verify_ds0_attributes_package,
    verify_ds0_geometry_package,
)
from visual_peer_inspector.evaluation.proposals import (
    clustered_bootstrap_intervals,
    evaluate_proposal_dataset,
    summarize_proposal_evaluation,
)


SCIENTIFIC_EXACT_COUNT_MINIMUM = 0.90
SCIENTIFIC_RECALL_AT_050_MINIMUM = 0.95
DIAGNOSTIC_PRECISION_AT_050_MINIMUM = 0.95
DIAGNOSTIC_RECALL_AT_075_MINIMUM = 0.85
DIAGNOSTIC_FULL_IMAGE_SUCCESS_MINIMUM = 0.85
DIAGNOSTIC_DETECTION_AWARE_IOU_MINIMUM = 0.85
DIAGNOSTIC_SUBGROUP_RECALL_FLOOR = 0.90
WEB_EXACT_COUNT_MINIMUM = 0.95
REPEAT_AGREEMENT_MINIMUM = 0.98
MINIMUM_SUBGROUP_SCENES = 5
MINIMUM_SUBGROUP_ASSETS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--attributes-root", type=Path, required=True)
    parser.add_argument("--proposal-results", type=Path, required=True)
    parser.add_argument("--repeat-agreement", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    return parser.parse_args()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_value(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=10
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def aggregate_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_proposal_evaluation(rows)
    return {
        "image_count": summary["image_count"],
        "scene_count": summary["scene_count"],
        "asset_cluster_count": len(
            {str(row["opaque_asset_group_id"]) for row in rows}
        ),
        "exact_count_rate": summary["exact_count_rate"],
        "precision_at_0_50": summary["micro_precision_at_0_50"],
        "recall_at_0_50": summary["micro_recall_at_0_50"],
        "recall_at_0_75": summary["micro_recall_at_0_75"],
        "mean_matched_iou": summary["mean_matched_iou"],
        "detection_aware_mean_iou": summary["detection_aware_mean_iou"],
        "miss_count": summary["miss_count"],
        "extra_count": summary["extra_count"],
    }


def build_subgroups(
    image_records: list[Mapping[str, Any]], *, include_protected: bool = False
) -> list[dict[str, Any]]:
    definitions: list[tuple[str, str, list[Mapping[str, Any]]]] = [
        ("overall", "all", image_records)
    ]
    fields = ["view_id", "object_count"]
    if include_protected:
        fields.append("category")
    for field in fields:
        values = sorted({str(row[field]) for row in image_records})
        for value in values:
            definitions.append(
                (
                    field,
                    value,
                    [row for row in image_records if str(row[field]) == value],
                )
            )
    if include_protected:
        families = sorted(
            {str(value) for row in image_records for value in row["anomaly_families"]}
        )
        for family in families:
            definitions.append(
                (
                    "anomaly_family",
                    family,
                    [row for row in image_records if family in row["anomaly_families"]],
                )
            )

    output: list[dict[str, Any]] = []
    for group_type, group_value, rows in definitions:
        scenes = {str(row["opaque_scene_id"]) for row in rows}
        assets = {str(row["opaque_asset_group_id"]) for row in rows}
        suppressed = len(scenes) < MINIMUM_SUBGROUP_SCENES or len(assets) < MINIMUM_SUBGROUP_ASSETS
        counts = {
            "group_type": group_type,
            "group_value": group_value,
            "scene_count": len(scenes),
            "asset_cluster_count": len(assets),
            "image_count": len(rows),
            "suppressed": suppressed,
        }
        output.append(counts if suppressed else {**counts, **aggregate_group(rows)})
    return output


def build_metric_rows(
    summary: Mapping[str, Any], intervals: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    interval_lookup = {str(row["metric"]): row for row in intervals}
    rows: list[dict[str, Any]] = []
    for name, value in summary.items():
        interval = interval_lookup.get(name)
        rows.append(
            {
                "metric": name,
                "value": value,
                "ci_lower_95": None if interval is None else interval["ci_lower_95"],
                "ci_upper_95": None if interval is None else interval["ci_upper_95"],
                "scope": "DS0 provisional integration screen",
            }
        )
    return rows


def build_gate_rows(
    summary: Mapping[str, Any],
    repeat_rate: float,
    subgroup_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    floor_rows = [
        row
        for row in subgroup_rows
        if not bool(row["suppressed"])
        and row["group_type"] in {"view_id", "object_count"}
    ]
    subgroup_recall_floor = min(float(row["recall_at_0_50"]) for row in floor_rows)
    gates = [
        (
            "scientific_exact_count",
            float(summary["exact_count_rate"]),
            SCIENTIFIC_EXACT_COUNT_MINIMUM,
            "greater_or_equal",
            "DS0 readiness",
            True,
        ),
        (
            "scientific_box_recall_at_0_50",
            float(summary["micro_recall_at_0_50"]),
            SCIENTIFIC_RECALL_AT_050_MINIMUM,
            "greater_or_equal",
            "DS0 readiness",
            True,
        ),
        (
            "diagnostic_box_precision_at_0_50",
            float(summary["micro_precision_at_0_50"]),
            DIAGNOSTIC_PRECISION_AT_050_MINIMUM,
            "greater_or_equal",
            "DS0 diagnostic health",
            True,
        ),
        (
            "diagnostic_box_recall_at_0_75",
            float(summary["micro_recall_at_0_75"]),
            DIAGNOSTIC_RECALL_AT_075_MINIMUM,
            "greater_or_equal",
            "DS0 diagnostic health",
            True,
        ),
        (
            "diagnostic_full_image_success",
            float(summary["full_image_success_rate"]),
            DIAGNOSTIC_FULL_IMAGE_SUCCESS_MINIMUM,
            "greater_or_equal",
            "DS0 diagnostic health",
            True,
        ),
        (
            "diagnostic_detection_aware_iou",
            float(summary["detection_aware_mean_iou"]),
            DIAGNOSTIC_DETECTION_AWARE_IOU_MINIMUM,
            "greater_or_equal",
            "DS0 diagnostic health",
            True,
        ),
        (
            "diagnostic_view_and_count_recall_floor",
            subgroup_recall_floor,
            DIAGNOSTIC_SUBGROUP_RECALL_FLOOR,
            "greater_or_equal",
            "DS0 diagnostic health",
            True,
        ),
        (
            "repeat_box_agreement",
            float(repeat_rate),
            REPEAT_AGREEMENT_MINIMUM,
            "greater_or_equal",
            "DS0 readiness and web default",
            True,
        ),
        (
            "web_exact_count",
            float(summary["exact_count_rate"]),
            WEB_EXACT_COUNT_MINIMUM,
            "greater_or_equal",
            "preliminary web default",
            False,
        ),
    ]
    rows = [
        {
            "gate": name,
            "observed": observed,
            "threshold": threshold,
            "comparison": comparison,
            "scope": scope,
            "passed": observed >= threshold,
            "blocking_for_ds0_sam": blocking,
        }
        for name, observed, threshold, comparison, scope, blocking in gates
    ]
    scientific = all(
        bool(row["passed"])
        for row in rows
        if bool(row["blocking_for_ds0_sam"])
    )
    rows.append(
        {
            "gate": "proceed_to_ds0_sam",
            "observed": int(scientific),
            "threshold": 1,
            "comparison": "equal",
            "scope": "DS0 decision",
            "passed": scientific,
            "blocking_for_ds0_sam": False,
        }
    )
    return rows


def save_plots(
    safe_root: Path, image_records: list[Mapping[str, Any]], summary: Mapping[str, Any]
) -> None:
    plt.figure(figsize=(7, 4.5))
    errors = [int(row["count_error"]) for row in image_records]
    values, counts = np.unique(errors, return_counts=True)
    plt.bar(values, counts, color="#4c78a8")
    plt.axvline(0, color="#222222", linewidth=1)
    plt.xlabel("Proposal count minus visible reference count")
    plt.ylabel("Images")
    plt.title("DS0 proposal count error")
    plt.tight_layout()
    plt.savefig(safe_root / "count_error_distribution.png", dpi=180)
    plt.close()

    matched_ious = [
        float(row["mean_matched_iou"])
        for row in image_records
        if int(row["matched_at_0_50"]) > 0
    ]
    plt.figure(figsize=(7, 4.5))
    plt.hist(matched_ious, bins=np.linspace(0.5, 1.0, 21), color="#59a14f", edgecolor="white")
    plt.axvline(float(summary["mean_matched_iou"]), color="#222222", linewidth=1.5)
    plt.xlabel("Mean matched box IoU per image")
    plt.ylabel("Images")
    plt.title("DS0 matched proposal quality")
    plt.tight_layout()
    plt.savefig(safe_root / "matched_iou_distribution.png", dpi=180)
    plt.close()

    failure_names = ["Misses", "Extras", "Merge candidates", "Split candidates"]
    failure_values = [
        int(summary["miss_count"]),
        int(summary["extra_count"]),
        int(summary["merge_candidate_count"]),
        int(summary["split_candidate_count"]),
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(failure_names, failure_values, color=["#e15759", "#f28e2b", "#b07aa1", "#76b7b2"])
    plt.ylabel("Count")
    plt.title("DS0 proposal failure indicators")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(safe_root / "failure_indicators.png", dpi=180)
    plt.close()


def save_failure_gallery(
    private_root: Path,
    media_root: Path,
    image_records: list[Mapping[str, Any]],
    reference_rows: list[Mapping[str, Any]],
    proposal_rows: list[Mapping[str, Any]],
) -> None:
    refs: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in reference_rows:
        if bool(row["visible"]):
            refs.setdefault(
                (str(row["opaque_scene_id"]), str(row["view_id"])), []
            ).append(row)
    proposals = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in proposal_rows
    }
    ranked = sorted(
        image_records,
        key=lambda row: (
            -int(row["miss_count"]) - int(row["extra_count"]),
            float(row["detection_aware_mean_iou"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
        ),
    )[:20]
    canvas = Image.new("RGB", (5 * 300, 4 * 330), "white")
    for index, row in enumerate(ranked):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        image_path = media_root / key[0] / "rgb" / f"{key[1]}.png"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").resize((280, 280))
        draw = ImageDraw.Draw(image)
        scale = 280 / 256
        for item in refs.get(key, []):
            box = [int(round(value * scale)) for value in item["box_xyxy"]]
            draw.rectangle(box, outline="#00a65a", width=3)
        for box_value in proposals[key]["boxes_xyxy"]:
            box = [int(round(value * scale)) for value in box_value]
            draw.rectangle(box, outline="#e63946", width=3)
        x = (index % 5) * 300 + 10
        y = (index // 5) * 330 + 10
        canvas.paste(image, (x, y))
        label = (
            f"{key[0]} / {key[1]}  miss {row['miss_count']}  "
            f"extra {row['extra_count']}"
        )
        ImageDraw.Draw(canvas).text((x, y + 287), label, fill="#111111")
    private_root.mkdir(parents=True, exist_ok=True)
    canvas.save(private_root / "proposal_failure_gallery.png")


def build_artifact_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_root}")
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    safe_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    geometry_record = verify_ds0_geometry_package(args.geometry_root)
    attributes_record = verify_ds0_attributes_package(args.attributes_root)
    reference_rows = read_jsonl(args.geometry_root / "reference_manifest.jsonl")
    protected_rows = read_jsonl(
        args.attributes_root / "protected_scene_attributes.jsonl"
    )
    proposal_rows = read_jsonl(args.proposal_results)
    repeat_record = json.loads(args.repeat_agreement.read_text(encoding="utf-8"))
    repeat_rate = float(repeat_record["box_agreement_rate"])

    evaluated = evaluate_proposal_dataset(
        reference_rows=reference_rows,
        proposal_rows=proposal_rows,
        protected_scene_rows=protected_rows,
    )
    image_records = evaluated["image_records"]
    summary = summarize_proposal_evaluation(image_records)
    intervals = clustered_bootstrap_intervals(
        image_records,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    subgroup_rows = build_subgroups(image_records)
    protected_subgroup_rows = build_subgroups(image_records, include_protected=True)
    gate_rows = build_gate_rows(summary, repeat_rate, subgroup_rows)

    write_jsonl(private_root / "image_metrics.jsonl", image_records)
    write_jsonl(private_root / "proposal_matches.jsonl", evaluated["match_records"])
    write_jsonl(private_root / "proposal_failures.jsonl", evaluated["failure_records"])
    write_csv(private_root / "protected_subgroup_summary.csv", protected_subgroup_rows)
    write_csv(safe_root / "proposal_metrics.csv", build_metric_rows(summary, intervals))
    write_csv(safe_root / "confidence_intervals.csv", intervals)
    write_csv(safe_root / "subgroup_summary.csv", subgroup_rows)
    write_csv(safe_root / "decision_gates.csv", gate_rows)
    save_plots(safe_root, image_records, summary)
    if args.media_root is not None:
        save_failure_gallery(
            private_root,
            args.media_root.expanduser().resolve(),
            image_records,
            reference_rows,
            proposal_rows,
        )

    source_files = [
        Path(__file__),
        Path(__file__).parents[1] / "src/visual_peer_inspector/evaluation/proposals.py",
        Path(__file__).parents[1]
        / "src/visual_peer_inspector/evaluation/proposal_reference.py",
    ]
    run_record = {
        "schema_version": 1,
        "evaluation_scope": "DS0 provisional integration screen",
        "final_400_scene_acceptance_assessed": False,
        "geometry_package_record_sha256": file_sha256(
            args.geometry_root / "package_record.json"
        ),
        "attributes_package_record_sha256": file_sha256(
            args.attributes_root / "package_record.json"
        ),
        "reference_manifest_sha256": geometry_record["reference_manifest_sha256"],
        "protected_attributes_sha256": attributes_record[
            "protected_attributes_sha256"
        ],
        "proposal_results_sha256": file_sha256(args.proposal_results),
        "repeat_agreement_sha256": file_sha256(args.repeat_agreement),
        "proposal_configuration_sha256": sorted(
            {str(row["configuration_sha256"]) for row in proposal_rows}
        ),
        "proposal_model_ids": sorted({str(row["model_id"]) for row in proposal_rows}),
        "proposal_model_revisions": sorted(
            {str(row["model_revision"]) for row in proposal_rows}
        ),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "evaluator_source_sha256": {
            path.name: file_sha256(path) for path in source_files
        },
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
        "summary": summary,
        "repeat_box_agreement": repeat_rate,
        "proceed_to_ds0_sam": next(
            bool(row["passed"])
            for row in gate_rows
            if row["gate"] == "proceed_to_ds0_sam"
        ),
    }
    write_json(safe_root / "run_record.json", run_record)
    manifest = build_artifact_manifest(safe_root)
    write_json(
        safe_root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_count": len(manifest),
            "artifacts": manifest,
            "manifest_content_sha256": sha256(
                b"".join(canonical_json_bytes(row) + b"\n" for row in manifest)
            ).hexdigest(),
        },
    )
    private_manifest = build_artifact_manifest(private_root)
    write_json(
        private_root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_count": len(private_manifest),
            "artifacts": private_manifest,
            "manifest_content_sha256": sha256(
                b"".join(canonical_json_bytes(row) + b"\n" for row in private_manifest)
            ).hexdigest(),
        },
    )

    print(
        "DS0 proposal evaluation complete: "
        f"exact count {summary['exact_count_rate']:.3f}, "
        f"recall@0.50 {summary['micro_recall_at_0_50']:.3f}, "
        f"matched IoU {summary['mean_matched_iou']:.3f}, "
        f"proceed to SAM {run_record['proceed_to_ds0_sam']}"
    )


if __name__ == "__main__":
    main()
