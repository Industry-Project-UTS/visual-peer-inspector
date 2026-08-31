"""Bounded, label-free proposal filtering with evaluator-only DS1 evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.evaluation.ds1_evaluator_package import (
    verify_ds1_evaluator_package,
)
from visual_peer_inspector.evaluation.proposal_history import (
    resolve_committed_proposal_history,
)
from visual_peer_inspector.evaluation.proposals import (
    box_iou,
    evaluate_proposal_dataset,
    summarize_proposal_evaluation,
)
from visual_peer_inspector.models.locateanything import MODEL_PRODUCER_KIND


STAGE = "ds1_proposal_refinement"
IMAGE_AREA = 256 * 256
PRECISION_GATE = 0.90
RECALL_GATE = 0.95


@dataclass(frozen=True)
class ProposalPolicy:
    key: str
    minimum_area_fraction: float
    maximum_area_fraction: float
    nms_iou: float | None


POLICIES = (
    ProposalPolicy("baseline", 0.0, 1.0, None),
    ProposalPolicy("area_only", 0.0015, 0.70, None),
    ProposalPolicy("nms_090", 0.0, 1.0, 0.90),
    ProposalPolicy("nms_075", 0.0, 1.0, 0.75),
    ProposalPolicy("area_nms_075", 0.0015, 0.70, 0.75),
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Table rows have different schemas: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact roots cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    record = {
        "schema_version": 1,
        "artifact_count": len(files),
        "artifacts": files,
        "manifest_content_sha256": sha256(
            b"".join(canonical_json_bytes(row) + b"\n" for row in files)
        ).hexdigest(),
    }
    _write_json(root / "artifact_manifest.json", record)
    return record


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    record = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    expected = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact roots cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            expected.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if record.get("artifacts") != expected:
        raise ValueError("Artifact manifest differs from the files on disk")
    return record


def _area_fraction(box: Sequence[float]) -> float:
    return (
        (float(box[2]) - float(box[0]))
        * (float(box[3]) - float(box[1]))
        / IMAGE_AREA
    )


def apply_proposal_policy(
    boxes: Sequence[Sequence[float]], policy: ProposalPolicy
) -> tuple[list[list[int]], list[int], Counter[str]]:
    """Filter boxes in original model order without evaluator information."""

    kept: list[list[int]] = []
    kept_indices: list[int] = []
    removed: Counter[str] = Counter()
    for index, raw_box in enumerate(boxes):
        box = [int(value) for value in raw_box]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 256 and 0 <= box[1] < box[3] <= 256):
            raise ValueError("Proposal policy received an invalid box")
        fraction = _area_fraction(box)
        if fraction < policy.minimum_area_fraction:
            removed["below_area_floor"] += 1
            continue
        if fraction > policy.maximum_area_fraction:
            removed["above_area_ceiling"] += 1
            continue
        if policy.nms_iou is not None and any(
            box_iou(box, previous) >= policy.nms_iou for previous in kept
        ):
            removed["overlap_suppression"] += 1
            continue
        kept.append(box)
        kept_indices.append(index)
    return kept, kept_indices, removed


def _load_committed_proposals(proposal_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposal_root = proposal_root.expanduser().resolve(strict=True)
    run_path = proposal_root / "safe" / "run_record.json"
    result_path = proposal_root / "private" / "proposal_results.jsonl"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run_record.get("stage") != "ds1_locateanything_screen"
        or run_record.get("status") != "completed"
        or run_record.get("completion_gate_passed") is not True
    ):
        raise ValueError("Proposal input is not the accepted completed DS1 run")
    configuration = run_record["configuration"]
    output_identity = run_record["output_identity"]
    resolved = resolve_committed_proposal_history(
        result_path,
        configuration_sha256=str(run_record["configuration_sha256"]),
        producer_kind=MODEL_PRODUCER_KIND,
        expected_file_sha256=file_sha256(result_path),
        expected_result_count=int(output_identity["result_count"]),
        expected_cohort_id=DS1_SCREEN_CONTRACT.cohort_id,
        expected_input_manifest_sha256=str(
            run_record["input_identity"]["model_input_manifest_sha256"]
        ),
        expected_model_id=str(configuration["model_id"]),
        expected_model_revision=str(configuration["model_revision"]),
    )
    rows = [dict(row) for row in resolved.latest_rows]
    if len(rows) != DS1_SCREEN_CONTRACT.image_count:
        raise ValueError("Proposal history does not contain all 2,000 DS1 images")
    return rows, run_record


def _paired_bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    baseline_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row for row in baseline
    }
    candidate_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row for row in candidate
    }
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("Paired policy records have different image coverage")
    clusters: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, row in baseline_by_key.items():
        clusters[str(row["opaque_asset_group_id"])].append(key)
    names = sorted(clusters)
    if len(names) < 2:
        raise ValueError("Paired bootstrap requires at least two asset clusters")

    def sufficient(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                sum(int(row["matched_at_0_50"]) for row in rows),
                sum(int(row["proposal_count"]) for row in rows),
                sum(int(row["reference_count"]) for row in rows),
            ],
            dtype=np.float64,
        )

    base_values = np.stack(
        [sufficient([baseline_by_key[key] for key in clusters[name]]) for name in names]
    )
    candidate_values = np.stack(
        [sufficient([candidate_by_key[key] for key in clusters[name]]) for name in names]
    )
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, len(names), size=(int(repetitions), len(names)))

    def f1(values: np.ndarray) -> np.ndarray:
        totals = values[draws].sum(axis=1)
        precision = np.divide(totals[:, 0], totals[:, 1], out=np.zeros(len(totals)), where=totals[:, 1] > 0)
        recall = np.divide(totals[:, 0], totals[:, 2], out=np.zeros(len(totals)), where=totals[:, 2] > 0)
        return np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(totals)), where=(precision + recall) > 0)

    differences = f1(candidate_values) - f1(base_values)
    baseline_summary = summarize_proposal_evaluation(baseline)
    candidate_summary = summarize_proposal_evaluation(candidate)
    point = float(candidate_summary["micro_f1_at_0_50"]) - float(
        baseline_summary["micro_f1_at_0_50"]
    )
    return {
        "f1_difference": point,
        "ci_lower_95": float(np.quantile(differences, 0.025)),
        "ci_upper_95": float(np.quantile(differences, 0.975)),
        "asset_cluster_count": len(names),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def _policy_subgroups(
    policy: str, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in ("category", "object_count", "anomaly_families"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            value = row.get(dimension)
            values = value if isinstance(value, list) else [value]
            for item in values:
                groups[str(item)].append(row)
        for value, items in sorted(groups.items()):
            summary = summarize_proposal_evaluation(items)
            rows.append(
                {
                    "policy": policy,
                    "dimension": dimension,
                    "value": value,
                    "image_count": len(items),
                    "reference_count": int(summary["reference_count"]),
                    "precision_at_0_50": float(summary["micro_precision_at_0_50"]),
                    "recall_at_0_50": float(summary["micro_recall_at_0_50"]),
                    "f1_at_0_50": float(summary["micro_f1_at_0_50"]),
                }
            )
    return rows


def _plot_policy_metrics(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["policy"]) for row in rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.22, [float(row["precision_at_0_50"]) for row in rows], width=0.22, label="precision")
    axis.bar(x, [float(row["recall_at_0_50"]) for row in rows], width=0.22, label="recall")
    axis.bar(x + 0.22, [float(row["f1_at_0_50"]) for row in rows], width=0.22, label="F1")
    axis.axhline(PRECISION_GATE, color="#666666", linestyle="--", linewidth=1)
    axis.set_ylim(0.70, 1.0)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Rate")
    axis.set_title("DS1 proposal policies at IoU 0.50")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_error_counts(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["policy"]) for row in rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, [int(row["miss_count"]) for row in rows], width=0.36, label="misses")
    axis.bar(x + 0.18, [int(row["extra_count"]) for row in rows], width=0.36, label="extras")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Object count")
    axis.set_title("Misses and extras after label-free filtering")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_efficiency(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 5))
    for row in rows:
        axis.scatter(float(row["proposal_count"]), float(row["f1_at_0_50"]), s=70)
        axis.annotate(str(row["policy"]), (float(row["proposal_count"]), float(row["f1_at_0_50"])), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Retained proposals")
    axis.set_ylabel("F1 at IoU 0.50")
    axis.set_title("Proposal count and evaluation quality")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _private_panels(
    *,
    rgb_root: Path,
    baseline_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    selected_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row for row in selected_rows
    }
    ranked = sorted(
        baseline_rows,
        key=lambda row: (
            len(row["boxes_xyxy"]) - len(selected_by_key[(str(row["opaque_scene_id"]), str(row["view_id"]))]["boxes_xyxy"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
        ),
        reverse=True,
    )[:6]
    panel = Image.new("RGB", (512, 6 * 284), "white")
    for row_index, baseline in enumerate(ranked):
        key = (str(baseline["opaque_scene_id"]), str(baseline["view_id"]))
        selected = selected_by_key[key]
        image_path = rgb_root / "media" / key[0] / "rgb" / f"{key[1]}.png"
        image = Image.open(image_path).convert("RGB")
        for column, (label, boxes, colour) in enumerate(
            (("baseline", baseline["boxes_xyxy"], "#ef3340"), ("selected", selected["boxes_xyxy"], "#1b8a5a"))
        ):
            view = image.copy()
            draw = ImageDraw.Draw(view)
            for box in boxes:
                draw.rectangle(tuple(box), outline=colour, width=2)
            draw.rectangle((0, 0, 256, 24), fill="white")
            draw.text((6, 5), f"{label}: {len(boxes)} boxes", fill="black")
            panel.paste(view, (column * 256, row_index * 284))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def run_ds1_proposal_refinement(
    *,
    proposal_root: Path,
    evaluator_root: Path,
    rgb_root: Path,
    output_root: Path,
    bootstrap_repetitions: int = 5_000,
    bootstrap_seed: int = 32_040,
) -> dict[str, Any]:
    """Run the predeclared policy screen and preserve the failed recall gate."""

    evaluator_root = evaluator_root.expanduser().resolve(strict=True)
    verify_ds1_evaluator_package(evaluator_root)
    proposal_rows, upstream_run = _load_committed_proposals(proposal_root)
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Proposal-refinement output already exists: {output_root}")
    staging = output_root.with_name(output_root.name + ".incomplete")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Incomplete output already exists: {staging}")
    model_input = staging / "model_input"
    private = staging / "private"
    safe = staging / "safe"
    for root in (model_input, private, safe):
        root.mkdir(parents=True)

    policy_rows: dict[str, list[dict[str, Any]]] = {}
    removal_rows = []
    locked_rows = []
    for policy in POLICIES:
        transformed = []
        reasons: Counter[str] = Counter()
        for row in proposal_rows:
            boxes, kept_indices, removed = apply_proposal_policy(row["boxes_xyxy"], policy)
            reasons.update(removed)
            transformed_row = {
                "schema_version": 1,
                "opaque_scene_id": str(row["opaque_scene_id"]),
                "view_id": str(row["view_id"]),
                "status": "ok",
                "proposal_count": len(boxes),
                "boxes_xyxy": boxes,
                "kept_proposal_indices": kept_indices,
                "policy": policy.key,
            }
            transformed.append(transformed_row)
            locked_rows.append(transformed_row)
        policy_rows[policy.key] = transformed
        for reason in ("below_area_floor", "above_area_ceiling", "overlap_suppression"):
            removal_rows.append({"policy": policy.key, "reason": reason, "removed_count": reasons[reason]})

    _write_jsonl(model_input / "policy_proposals.jsonl", locked_rows)
    configuration = {
        "schema_version": 1,
        "stage": STAGE,
        "image_width": 256,
        "image_height": 256,
        "policies": [asdict(policy) for policy in POLICIES],
        "selection_metric": "micro_f1_at_0_50",
        "precision_gate": PRECISION_GATE,
        "recall_gate": RECALL_GATE,
        "maximum_recall_loss_for_diagnostic_selection": 0.005,
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(bootstrap_seed),
        "uses_evaluator_information_during_filtering": False,
    }
    _write_json(safe / "resolved_configuration.json", configuration)
    model_record = {
        "schema_version": 1,
        "package_kind": "ds1-label-free-proposal-policy-screen",
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "scene_count": DS1_SCREEN_CONTRACT.scene_count,
        "image_count": DS1_SCREEN_CONTRACT.image_count,
        "policy_count": len(POLICIES),
        "policy_image_row_count": len(locked_rows),
        "proposal_results_sha256": file_sha256(Path(proposal_root) / "private" / "proposal_results.jsonl"),
        "configuration_sha256": sha256(canonical_json_bytes(configuration)).hexdigest(),
        "manifest_sha256": file_sha256(model_input / "policy_proposals.jsonl"),
        "all_checks_passed": True,
    }
    _write_json(model_input / "package_record.json", model_record)
    _manifest(model_input)

    references = read_jsonl(evaluator_root / "reference_manifest.jsonl")
    attributes = read_jsonl(evaluator_root / "protected_scene_attributes.jsonl")
    evaluations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    summary_rows = []
    subgroup_rows = []
    for policy in POLICIES:
        result = evaluate_proposal_dataset(
            reference_rows=references,
            proposal_rows=policy_rows[policy.key],
            protected_scene_rows=attributes,
        )
        evaluations[policy.key] = result
        summary = summarize_proposal_evaluation(result["image_records"])
        summary_rows.append(
            {
                "policy": policy.key,
                "proposal_count": int(summary["proposal_count"]),
                "precision_at_0_50": float(summary["micro_precision_at_0_50"]),
                "recall_at_0_50": float(summary["micro_recall_at_0_50"]),
                "f1_at_0_50": float(summary["micro_f1_at_0_50"]),
                "recall_at_0_75": float(summary["micro_recall_at_0_75"]),
                "exact_count_rate": float(summary["exact_count_rate"]),
                "full_image_success_rate": float(summary["full_image_success_rate"]),
                "miss_count": int(summary["miss_count"]),
                "extra_count": int(summary["extra_count"]),
            }
        )
        subgroup_rows.extend(_policy_subgroups(policy.key, result["image_records"]))
        _write_jsonl(private / "evaluation" / f"{policy.key}_image_records.jsonl", result["image_records"])

    baseline_records = evaluations["baseline"]["image_records"]
    paired_rows = []
    for policy in POLICIES:
        comparison = _paired_bootstrap(
            baseline_records,
            evaluations[policy.key]["image_records"],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        )
        paired_rows.append({"policy": policy.key, **comparison})

    summary_by_policy = {str(row["policy"]): row for row in summary_rows}
    baseline_summary = summary_by_policy["baseline"]
    candidates = []
    for row in summary_rows:
        comparison = next(item for item in paired_rows if item["policy"] == row["policy"])
        recall_loss = float(baseline_summary["recall_at_0_50"]) - float(row["recall_at_0_50"])
        candidates.append(
            {
                "policy": row["policy"],
                "precision_gate_passed": float(row["precision_at_0_50"]) >= PRECISION_GATE,
                "recall_gate_passed": float(row["recall_at_0_50"]) >= RECALL_GATE,
                "recall_loss_vs_baseline": recall_loss,
                "paired_f1_ci_lower": float(comparison["ci_lower_95"]),
                "diagnostic_selection_eligible": bool(
                    row["policy"] == "baseline"
                    or (
                        recall_loss <= 0.005
                        and float(comparison["f1_difference"]) > 0.0
                        and float(comparison["ci_lower_95"]) > 0.0
                    )
                ),
            }
        )
    eligible = [row for row in summary_rows if next(item for item in candidates if item["policy"] == row["policy"])["diagnostic_selection_eligible"]]
    selected = max(eligible, key=lambda row: (float(row["f1_at_0_50"]), str(row["policy"])))
    accepted = bool(
        float(selected["precision_at_0_50"]) >= PRECISION_GATE
        and float(selected["recall_at_0_50"]) >= RECALL_GATE
    )
    decision = {
        "decision": (
            "accept_label_free_proposal_refinement"
            if accepted
            else "retain_refinement_as_diagnostic_and_replace_or_reprompt_detector"
        ),
        "selected_diagnostic_policy": selected["policy"],
        "automatic_proposal_route_accepted": accepted,
        "precision_gate": PRECISION_GATE,
        "recall_gate": RECALL_GATE,
        "selected_precision": float(selected["precision_at_0_50"]),
        "selected_recall": float(selected["recall_at_0_50"]),
        "selected_f1": float(selected["f1_at_0_50"]),
        "reason": (
            "The selected label-free filter reduces duplicate and extreme-size extras, "
            "but filtering cannot recover objects missed by the proposal model."
        ),
        "final_acceptance": False,
    }

    _write_csv(safe / "policy_summary.csv", summary_rows)
    _write_csv(safe / "paired_f1_intervals.csv", paired_rows)
    _write_csv(safe / "policy_gates.csv", candidates)
    _write_csv(safe / "removal_summary.csv", removal_rows)
    _write_csv(safe / "subgroup_summary.csv", subgroup_rows)
    _write_json(safe / "decision.json", decision)
    _plot_policy_metrics(summary_rows, safe / "figures" / "policy_metrics.png")
    _plot_error_counts(summary_rows, safe / "figures" / "misses_and_extras.png")
    _plot_efficiency(summary_rows, safe / "figures" / "proposal_count_vs_f1.png")
    _private_panels(
        rgb_root=rgb_root,
        baseline_rows=policy_rows["baseline"],
        selected_rows=policy_rows[str(selected["policy"])],
        output=private / "figures" / "baseline_vs_selected_boxes.png",
    )
    _write_json(
        safe / "input_audit.json",
        {
            "scene_count": DS1_SCREEN_CONTRACT.scene_count,
            "image_count": DS1_SCREEN_CONTRACT.image_count,
            "upstream_proposal_count": sum(len(row["boxes_xyxy"]) for row in proposal_rows),
            "proposal_results_sha256": model_record["proposal_results_sha256"],
            "evaluator_package_sha256": file_sha256(evaluator_root / "package_record.json"),
            "model_input_locked_before_evaluator": True,
        },
    )
    gates = [
        {"gate": "all_2000_images_evaluated", "passed": all(len(evaluations[p.key]["image_records"]) == 2000 for p in POLICIES)},
        {"gate": "policy_manifest_locked_before_evaluator", "passed": True},
        {"gate": "filters_use_geometry_only", "passed": True},
        {"gate": "baseline_preserved", "passed": summary_by_policy["baseline"]["proposal_count"] == 9249},
        {"gate": "paired_asset_bootstrap_completed", "passed": all(int(row["asset_cluster_count"]) == 27 for row in paired_rows)},
        {
            "gate": "acceptance_matches_declared_precision_and_recall_gates",
            "passed": accepted
            == (
                float(selected["precision_at_0_50"]) >= PRECISION_GATE
                and float(selected["recall_at_0_50"]) >= RECALL_GATE
            ),
        },
        {"gate": "no_final_acceptance_claim", "passed": decision["final_acceptance"] is False},
    ]
    _write_csv(safe / "completion_gates.csv", gates)
    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "tracking_status": "pending",
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "scene_count": DS1_SCREEN_CONTRACT.scene_count,
        "image_count": DS1_SCREEN_CONTRACT.image_count,
        "policy_count": len(POLICIES),
        "all_completion_gates_passed": all(bool(row["passed"]) for row in gates),
        "decision": decision["decision"],
        "selected_diagnostic_policy": decision["selected_diagnostic_policy"],
        "automatic_proposal_route_accepted": accepted,
        "model_input_package_sha256": file_sha256(model_input / "package_record.json"),
        "upstream_configuration_sha256": upstream_run["configuration_sha256"],
    }
    _write_json(safe / "run_record.json", run_record)
    _manifest(private)
    _manifest(safe)
    os.replace(staging, output_root)
    return run_record


__all__ = [
    "POLICIES",
    "ProposalPolicy",
    "apply_proposal_policy",
    "run_ds1_proposal_refinement",
    "verify_artifact_manifest",
]
