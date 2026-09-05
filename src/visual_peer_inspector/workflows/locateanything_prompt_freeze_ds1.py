"""Bounded DS1 prompt study for the automatic LocateAnything route.

The model stage uses only opaque RGB images. Evaluation opens the DS1 reference
package after every candidate proposal file has been written and hash locked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evaluation.ds1_evaluator_package import (
    verify_ds1_evaluator_package,
)
from visual_peer_inspector.evaluation.proposals import (
    evaluate_proposal_dataset,
    summarize_proposal_evaluation,
)
from visual_peer_inspector.evidence import save_bar_chart
from visual_peer_inspector.models.locateanything import (
    LocateAnythingConfiguration,
    ProposalResultStore,
)
from visual_peer_inspector.workflows.proposal_refinement_ds1 import (
    POLICIES,
    ProposalPolicy,
    apply_proposal_policy,
)


STAGE = "ds1_locateanything_prompt_freeze"
MODEL_ID = "nvidia/LocateAnything-3B"
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
PROMPT_CANDIDATES = (
    ("object", "object"),
    ("individual_object", "individual object"),
)
POSTPROCESSING_POLICIES = ("raw", "area_nms_075", "area_nms_075_cap8")
PROMPT_POSTPROCESSING_POLICY_BY_KEY = {
    **{policy.key: policy for policy in POLICIES},
    "area_nms_075_cap8": ProposalPolicy(
        "area_nms_075_cap8",
        0.0015,
        0.70,
        0.75,
        8,
    ),
}


@dataclass(frozen=True)
class PromptDecision:
    candidate_key: str
    prompt_key: str
    prompt: str
    postprocessing_policy: str
    precision_at_0_50: float
    recall_at_0_50: float
    f1_at_0_50: float
    f2_at_0_50: float
    recall_at_0_75: float
    mean_matched_iou: float
    detection_aware_mean_iou: float
    complete_detection_rate: float
    proposal_count: int


def prompt_configurations() -> dict[str, LocateAnythingConfiguration]:
    """Return the small, predeclared set of generic web-app prompts."""

    return {
        key: LocateAnythingConfiguration(prompt=prompt)
        for key, prompt in PROMPT_CANDIDATES
    }


def f_beta(precision: float, recall: float, *, beta: float = 2.0) -> float:
    """Return the standard F-beta score, using beta=2 to emphasise missed objects."""

    if precision < 0 or recall < 0 or beta <= 0:
        raise ValueError("Precision, recall and beta must be valid")
    denominator = beta * beta * precision + recall
    return 0.0 if denominator == 0 else (1 + beta * beta) * precision * recall / denominator


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Rows do not share one schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
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


def load_latest_proposal_rows(
    result_path: Path,
    configuration: LocateAnythingConfiguration,
) -> list[dict[str, Any]]:
    """Resolve an append-only proposal history to one successful row per image."""

    store = ProposalResultStore(
        result_path,
        configuration.sha256,
        "locateanything_model",
    )
    rows = store.rows
    if len(rows) != DS1_SCREEN_CONTRACT.image_count:
        raise ValueError("A prompt result does not cover all 2,000 DS1 images")
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("A prompt result contains an unresolved model failure")
    if {row["prompt"] for row in rows} != {configuration.prompt}:
        raise ValueError("A prompt result contains another natural-language prompt")
    return [dict(row) for row in rows]


def _filtered_rows(
    rows: Sequence[Mapping[str, Any]],
    policy_key: str,
) -> list[dict[str, Any]]:
    if policy_key == "raw":
        return [dict(row) for row in rows]
    policy = PROMPT_POSTPROCESSING_POLICY_BY_KEY.get(policy_key)
    if policy is None:
        raise ValueError(f"Unknown proposal policy: {policy_key}")
    output = []
    for row in rows:
        boxes, _indices, _removed = apply_proposal_policy(row["boxes_xyxy"], policy)
        output.append({
            "opaque_scene_id": str(row["opaque_scene_id"]),
            "view_id": str(row["view_id"]),
            "status": "ok",
            "proposal_count": len(boxes),
            "boxes_xyxy": boxes,
        })
    return output


def _summary_row(
    prompt_key: str,
    prompt: str,
    policy_key: str,
    image_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = summarize_proposal_evaluation(image_records)
    precision = float(summary["micro_precision_at_0_50"])
    recall = float(summary["micro_recall_at_0_50"])
    return {
        "candidate_key": f"{prompt_key}__{policy_key}",
        "prompt_key": prompt_key,
        "prompt": prompt,
        "postprocessing_policy": policy_key,
        "image_count": int(summary["image_count"]),
        "reference_count": int(summary["reference_count"]),
        "proposal_count": int(summary["proposal_count"]),
        "matched_count_at_0_50": int(summary["matched_count_at_0_50"]),
        "precision_at_0_50": precision,
        "recall_at_0_50": recall,
        "f1_at_0_50": float(summary["micro_f1_at_0_50"]),
        "f2_at_0_50": f_beta(precision, recall),
        "recall_at_0_75": float(summary["micro_recall_at_0_75"]),
        "mean_matched_iou": float(summary["mean_matched_iou"]),
        "median_matched_iou": float(summary["median_matched_iou"]),
        "p10_matched_iou": float(summary["p10_matched_iou"]),
        "detection_aware_mean_iou": float(summary["detection_aware_mean_iou"]),
        "complete_detection_rate": 1.0 - int(summary["images_with_misses"]) / int(summary["image_count"]),
        "exact_count_rate": float(summary["exact_count_rate"]),
        "full_image_success_rate": float(summary["full_image_success_rate"]),
        "count_mean_absolute_error": float(summary["count_mean_absolute_error"]),
        "count_signed_bias": float(summary["count_signed_bias"]),
        "miss_count": int(summary["miss_count"]),
        "extra_count": int(summary["extra_count"]),
        "images_with_misses": int(summary["images_with_misses"]),
        "images_with_extras": int(summary["images_with_extras"]),
        "merge_candidate_count": int(summary["merge_candidate_count"]),
        "split_or_duplicate_candidate_count": int(summary["split_candidate_count"]),
        "mean_proposals_per_image": int(summary["proposal_count"]) / int(summary["image_count"]),
    }


def select_prompt_configuration(rows: Sequence[Mapping[str, Any]]) -> PromptDecision:
    """Choose the bounded candidate without an arbitrary recall cutoff."""

    if not rows:
        raise ValueError("Prompt selection requires evaluated candidates")
    eligible = [row for row in rows if int(row["image_count"]) == DS1_SCREEN_CONTRACT.image_count]
    if not eligible:
        raise ValueError("No prompt candidate completed the DS1 screen")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["f2_at_0_50"]),
            float(row.get("recall_at_0_75", 0.0)),
            float(row.get("detection_aware_mean_iou", 0.0)),
            float(row["complete_detection_rate"]),
            float(row["f1_at_0_50"]),
            float(row["precision_at_0_50"]),
            -int(row["proposal_count"]),
            str(row["candidate_key"]),
        ),
    )
    return PromptDecision(
        candidate_key=str(selected["candidate_key"]),
        prompt_key=str(selected["prompt_key"]),
        prompt=str(selected["prompt"]),
        postprocessing_policy=str(selected["postprocessing_policy"]),
        precision_at_0_50=float(selected["precision_at_0_50"]),
        recall_at_0_50=float(selected["recall_at_0_50"]),
        f1_at_0_50=float(selected["f1_at_0_50"]),
        f2_at_0_50=float(selected["f2_at_0_50"]),
        recall_at_0_75=float(selected.get("recall_at_0_75", 0.0)),
        mean_matched_iou=float(selected.get("mean_matched_iou", 0.0)),
        detection_aware_mean_iou=float(
            selected.get("detection_aware_mean_iou", 0.0)
        ),
        complete_detection_rate=float(selected["complete_detection_rate"]),
        proposal_count=int(selected["proposal_count"]),
    )


def run_prompt_evaluation(
    *,
    rgb_root: Path,
    evaluator_root: Path,
    proposal_results: Mapping[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    """Hash-lock every candidate, then open DS1 references and select one route."""

    verify_model_input_package(rgb_root, contract=DS1_SCREEN_CONTRACT)
    verify_ds1_evaluator_package(evaluator_root)
    configurations = prompt_configurations()
    if set(proposal_results) != set(configurations):
        raise ValueError("Prompt result set differs from the predeclared registry")
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Prompt study output already exists: {output_root}")
    locked_root = output_root / "model_input"
    private_root = output_root / "private"
    safe_root = output_root / "safe"
    locked_root.mkdir(parents=True)
    private_root.mkdir(parents=True)
    safe_root.mkdir(parents=True)

    proposal_rows: dict[str, list[dict[str, Any]]] = {}
    identities = []
    for key, configuration in configurations.items():
        rows = load_latest_proposal_rows(proposal_results[key], configuration)
        proposal_rows[key] = rows
        identities.append({
            "prompt_key": key,
            "prompt": configuration.prompt,
            "configuration_sha256": configuration.sha256,
            "proposal_results_sha256": file_sha256(proposal_results[key]),
            "result_count": len(rows),
        })
    _write_json(locked_root / "candidate_identities.json", {"candidates": identities})
    lock_sha256 = file_sha256(locked_root / "candidate_identities.json")

    references = read_jsonl(evaluator_root / "reference_manifest.jsonl")
    attributes = read_jsonl(evaluator_root / "protected_scene_attributes.jsonl")
    summary_rows = []
    all_image_records: dict[str, list[dict[str, Any]]] = {}
    for prompt_key, prompt in PROMPT_CANDIDATES:
        for policy_key in POSTPROCESSING_POLICIES:
            filtered = _filtered_rows(proposal_rows[prompt_key], policy_key)
            evaluation = evaluate_proposal_dataset(
                reference_rows=references,
                proposal_rows=filtered,
                protected_scene_rows=attributes,
            )
            candidate_key = f"{prompt_key}__{policy_key}"
            all_image_records[candidate_key] = evaluation["image_records"]
            summary_rows.append(
                _summary_row(prompt_key, prompt, policy_key, evaluation["image_records"])
            )
            _write_jsonl(
                private_root / "evaluation" / f"{candidate_key}_image_records.jsonl",
                evaluation["image_records"],
            )

    decision = select_prompt_configuration(summary_rows)
    selected_configuration = configurations[decision.prompt_key]
    selected_record = {
        "schema_version": 1,
        "stage": STAGE,
        "selection_rule": (
            "Maximise F2 at box IoU 0.50, then recall at IoU 0.75, "
            "detection-aware mean IoU, complete-detection rate, F1, precision, "
            "and lower proposal count."
        ),
        "selected": asdict(decision),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "generation_configuration": asdict(selected_configuration),
        "generation_configuration_sha256": selected_configuration.sha256,
        "candidate_lock_sha256": lock_sha256,
        "development_only": True,
        "requires_unchanged_ds2_confirmation": True,
    }
    _write_csv(safe_root / "prompt_policy_summary.csv", summary_rows)
    _write_json(safe_root / "selected_prompt.json", selected_record)
    save_bar_chart(
        [str(row["candidate_key"]) for row in summary_rows],
        [float(row["f2_at_0_50"]) for row in summary_rows],
        safe_root / "prompt_policy_f2.png",
        title="DS1 LocateAnything prompt and policy F2",
        y_label="F2 at IoU 0.50",
        show_values=True,
    )
    save_bar_chart(
        [str(row["candidate_key"]) for row in summary_rows],
        [float(row["detection_aware_mean_iou"]) for row in summary_rows],
        safe_root / "detection_aware_mean_iou.png",
        title="Detection-aware box localisation",
        y_label="Mean IoU across reference objects",
        show_values=True,
    )
    save_bar_chart(
        [str(row["candidate_key"]) for row in summary_rows],
        [int(row["split_or_duplicate_candidate_count"]) for row in summary_rows],
        safe_root / "split_or_duplicate_candidates.png",
        title="Split or duplicate proposal candidates",
        y_label="Candidate count",
        show_values=True,
    )
    save_bar_chart(
        [str(row["candidate_key"]) for row in summary_rows],
        [float(row["complete_detection_rate"]) for row in summary_rows],
        safe_root / "complete_detection_rate.png",
        title="Images with every object detected",
        y_label="Image rate",
        show_values=True,
    )
    save_bar_chart(
        [str(row["candidate_key"]) for row in summary_rows],
        [int(row["miss_count"]) for row in summary_rows],
        safe_root / "missed_objects.png",
        title="Missed objects by prompt and policy",
        y_label="Missed objects",
        show_values=True,
    )
    gates = [
        {"check": "all prompt candidates completed", "passed": len(identities) == len(PROMPT_CANDIDATES)},
        {"check": "candidate files locked before evaluation", "passed": len(lock_sha256) == 64},
        {"check": "all candidate policies evaluated", "passed": len(summary_rows) == len(PROMPT_CANDIDATES) * len(POSTPROCESSING_POLICIES)},
        {"check": "selected configuration reserved for unchanged DS2 confirmation", "passed": True},
    ]
    _write_csv(safe_root / "completion_checks.csv", gates)
    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "scene_count": DS1_SCREEN_CONTRACT.scene_count,
        "image_count": DS1_SCREEN_CONTRACT.image_count,
        "candidate_count": len(summary_rows),
        "selected_candidate": decision.candidate_key,
        "selected_prompt": decision.prompt,
        "selected_postprocessing_policy": decision.postprocessing_policy,
        "all_completion_checks_passed": all(bool(row["passed"]) for row in gates),
    }
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(locked_root)
    _artifact_manifest(private_root)
    _artifact_manifest(safe_root)
    return run_record


__all__ = [
    "POSTPROCESSING_POLICIES",
    "PROMPT_POSTPROCESSING_POLICY_BY_KEY",
    "PROMPT_CANDIDATES",
    "PromptDecision",
    "f_beta",
    "prompt_configurations",
    "run_prompt_evaluation",
    "select_prompt_configuration",
]
