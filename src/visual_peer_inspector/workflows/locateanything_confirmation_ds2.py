"""Frozen LocateAnything inference for the untouched DS2 confirmation cohort."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram, save_image_grid
from visual_peer_inspector.models.locateanything import (
    CONTRACT_PRODUCER_KIND,
    MODEL_PRODUCER_KIND,
    LocateAnythingConfiguration,
    ProposalResultStore,
    bounded_locateanything_execution_policy,
    build_proposal_jobs,
    execution_policy_sha256,
    repeat_agreement,
    run_proposal_jobs,
    summarize_proposal_rows,
)
from visual_peer_inspector.workflows.locateanything_prompt_freeze_ds1 import (
    PROMPT_POSTPROCESSING_POLICY_BY_KEY,
)
from visual_peer_inspector.workflows.proposal_refinement_ds1 import apply_proposal_policy


STAGE = "ds2_frozen_locateanything"
PILOT_IMAGE_COUNT = 20
ALLOWED_POLICIES = frozenset({"raw", "area_nms_075", "area_nms_075_cap8"})
DS2_MAX_NEW_TOKENS = 4_096
DS2_MAXIMUM_ATTEMPTS_PER_JOB = 2
DS2_ATTEMPT_TIMEOUT_SECONDS = 75.0
DS2_TERMINAL_FAILURE_DISPOSITION = "carry_as_zero_proposal"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    """Write a checksum inventory for one DS2 output boundary."""

    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("DS2 LocateAnything outputs cannot contain symbolic links")
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
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


def _proposal_panel(
    jobs: Sequence[Any],
    proposals: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Save a fixed 20-image private panel with the frozen proposal boxes."""

    proposal_index = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in proposals
    }
    items = []
    for job in _pilot_jobs(jobs):
        row = proposal_index[job.key]
        items.append(
            {
                "path": job.image_path,
                "boxes": row["boxes_xyxy"],
                "label": f"{job.opaque_scene_id} / {job.view_id} / {row['proposal_count']} boxes",
            }
        )
    save_image_grid(
        items,
        output_path,
        title="DS2 LocateAnything proposal examples",
        columns=5,
    )


def load_frozen_ds1_configuration(
    selected_prompt_path: Path,
) -> tuple[LocateAnythingConfiguration, str, str]:
    """Load the exact DS1 choice without consulting DS2 geometry or labels."""

    path = selected_prompt_path.expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("stage") != "ds1_locateanything_prompt_freeze"
        or record.get("development_only") is not True
        or record.get("requires_unchanged_ds2_confirmation") is not True
    ):
        raise ValueError("LocateAnything configuration is not a frozen DS1 decision")
    selected = record.get("selected", {})
    policy = str(selected.get("postprocessing_policy", ""))
    if policy not in ALLOWED_POLICIES:
        raise ValueError("Frozen LocateAnything post-processing policy is unsupported")
    configuration_values = record.get("generation_configuration")
    if not isinstance(configuration_values, dict):
        raise ValueError("Frozen LocateAnything generation configuration is missing")
    expected_fields = set(LocateAnythingConfiguration.__dataclass_fields__)
    if set(configuration_values) != expected_fields:
        raise ValueError("Frozen LocateAnything configuration fields differ")
    configuration = LocateAnythingConfiguration(**configuration_values)
    if (
        configuration.sha256 != record.get("generation_configuration_sha256")
        or configuration.prompt != selected.get("prompt")
    ):
        raise ValueError("Frozen LocateAnything configuration digest or prompt differs")
    return configuration, policy, file_sha256(path)


def resolve_ds2_operational_policy(
    frozen_configuration: LocateAnythingConfiguration,
) -> tuple[LocateAnythingConfiguration, dict[str, Any]]:
    """Keep the DS1 model decision while bounding DS2 execution cost."""

    effective_configuration = replace(
        frozen_configuration,
        max_new_tokens=DS2_MAX_NEW_TOKENS,
    )
    execution_policy = bounded_locateanything_execution_policy(
        effective_configuration,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        terminal_failure_disposition=DS2_TERMINAL_FAILURE_DISPOSITION,
        attempt_timeout_seconds=DS2_ATTEMPT_TIMEOUT_SECONDS,
    )
    return effective_configuration, execution_policy


def _apply_frozen_policy(
    row: Mapping[str, Any],
    policy_key: str,
) -> tuple[list[list[int]], int]:
    boxes = [[int(value) for value in box] for box in row["boxes_xyxy"]]
    if policy_key == "raw":
        return boxes, 0
    policy = PROMPT_POSTPROCESSING_POLICY_BY_KEY.get(policy_key)
    if policy is None:
        raise ValueError(f"Unknown frozen proposal policy: {policy_key}")
    filtered, _indices, removed = apply_proposal_policy(boxes, policy)
    return filtered, int(sum(removed.values()))


def build_frozen_automatic_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_key: str,
    freeze_sha256: str,
) -> list[dict[str, Any]]:
    """Create the label-free box package consumed by automatic SAM inference."""

    output = []
    for row in sorted(rows, key=lambda item: (str(item["opaque_scene_id"]), str(item["view_id"]))):
        source_status = str(row.get("status"))
        if source_status not in {"ok", "error"}:
            raise ValueError("DS2 automatic proposal row has an invalid terminal status")
        raw_proposal_count = int(row["proposal_count"])
        if source_status == "error":
            if raw_proposal_count != len(row.get("boxes_xyxy", [])):
                raise ValueError("Failed DS2 proposal count differs from its parsed boxes")
            boxes, removed = [], 0
            discarded_untrusted = raw_proposal_count
        else:
            boxes, removed = _apply_frozen_policy(row, policy_key)
            discarded_untrusted = 0
        if raw_proposal_count != removed + discarded_untrusted + len(boxes):
            raise ValueError("DS2 proposal accounting does not balance")
        output.append(
            {
                "schema_version": 2,
                "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
                "opaque_scene_id": str(row["opaque_scene_id"]),
                "view_id": str(row["view_id"]),
                "image_sha256": str(row["image_sha256"]),
                "configuration_sha256": str(row["configuration_sha256"]),
                "ds1_freeze_sha256": freeze_sha256,
                "postprocessing_policy": policy_key,
                "raw_proposal_count": raw_proposal_count,
                "removed_proposal_count": removed,
                "discarded_untrusted_box_count": discarded_untrusted,
                "proposal_count": len(boxes),
                "boxes_xyxy": boxes,
                "source_status": source_status,
                "source_error_type": (
                    str(row["error_type"]) if row.get("error_type") else None
                ),
                "source_attempt": int(row["attempt"]),
                "carried_as_zero_proposal": source_status == "error",
            }
        )
    return output


def _pilot_jobs(jobs: Sequence[Any]) -> list[Any]:
    """Select a deterministic operational pilot without opening evaluator data."""

    if len(jobs) < PILOT_IMAGE_COUNT:
        raise ValueError("DS2 proposal pilot requires at least 20 images")
    step = max(1, len(jobs) // PILOT_IMAGE_COUNT)
    selected = [jobs[index] for index in range(0, len(jobs), step)][:PILOT_IMAGE_COUNT]
    if len(selected) != PILOT_IMAGE_COUNT:
        raise RuntimeError("DS2 proposal pilot selection is incomplete")
    return selected


def _require_proposal_readiness(
    one_rows: Sequence[Mapping[str, Any]],
    reloaded_one_rows: Sequence[Mapping[str, Any]],
    pilot_rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    pilot_proposals: Sequence[Mapping[str, Any]],
    *,
    pilot_history_count_before_resume: int,
    pilot_history_count_after_resume: int,
) -> None:
    """Enforce operational integrity without imposing a quality percentage."""

    def has_valid_structure(row: Mapping[str, Any]) -> bool:
        boxes = row.get("boxes_xyxy")
        return (
            row.get("status") == "ok"
            and isinstance(boxes, list)
            and int(row.get("proposal_count", -1)) == len(boxes)
            and all(
                len(box) == 4
                and all(isinstance(value, int) for value in box)
                and 0 <= box[0] < box[2] <= 256
                and 0 <= box[1] < box[3] <= 256
                for box in boxes
            )
        )

    def has_trusted_box(row: Mapping[str, Any]) -> bool:
        return has_valid_structure(row) and bool(row["boxes_xyxy"])

    if (
        len(one_rows) != 1
        or one_rows[0]["status"] != "ok"
        or one_rows != list(reloaded_one_rows)
    ):
        raise RuntimeError("DS2 LocateAnything one-image readiness gate failed")
    if not has_valid_structure(one_rows[0]):
        raise RuntimeError("DS2 LocateAnything one-image output gate failed")

    def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
        values = {
            (str(row["opaque_scene_id"]), str(row["view_id"])): row
            for row in rows
        }
        if len(rows) != PILOT_IMAGE_COUNT or len(values) != PILOT_IMAGE_COUNT:
            raise RuntimeError("DS2 LocateAnything pilot coverage gate failed")
        return values

    pilot_by_key = indexed(pilot_rows)
    repeat_by_key = indexed(repeat_rows)
    if any(row.get("status") not in {"ok", "error"} for row in pilot_rows):
        raise RuntimeError("DS2 LocateAnything pilot terminal-status gate failed")
    if any(row.get("status") not in {"ok", "error"} for row in repeat_rows):
        raise RuntimeError("DS2 LocateAnything repeat terminal-status gate failed")
    if set(pilot_by_key) != set(repeat_by_key):
        raise RuntimeError("DS2 LocateAnything pilot coverage gate failed")
    if any(
        pilot_by_key[key]["image_sha256"] != repeat_by_key[key]["image_sha256"]
        or pilot_by_key[key]["configuration_sha256"]
        != repeat_by_key[key]["configuration_sha256"]
        for key in pilot_by_key
    ):
        raise RuntimeError("DS2 LocateAnything pilot identity gate failed")
    if not any(
        pilot_by_key[key]["status"] == "ok" and repeat_by_key[key]["status"] == "ok"
        for key in pilot_by_key
    ):
        raise RuntimeError("DS2 LocateAnything pilot common-success gate failed")
    if not any(has_trusted_box(row) for row in pilot_rows):
        raise RuntimeError("DS2 LocateAnything pilot produced no trusted boxes")
    disposition_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in pilot_proposals
    }
    if set(disposition_by_key) != set(pilot_by_key) or any(
        row["status"] == "error"
        and (
            int(disposition_by_key[key]["proposal_count"]) != 0
            or disposition_by_key[key]["boxes_xyxy"] != []
            or disposition_by_key[key]["carried_as_zero_proposal"] is not True
        )
        for key, row in pilot_by_key.items()
    ):
        raise RuntimeError("DS2 LocateAnything pilot failure-disposition gate failed")
    if pilot_history_count_before_resume != pilot_history_count_after_resume:
        raise RuntimeError("DS2 LocateAnything pilot resume gate failed")


def run_ds2_locateanything(
    *,
    rgb_root: Path,
    selected_prompt_path: Path,
    output_root: Path,
    worker: Any,
    run_mode: str,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay one frozen DS1 proposal configuration over every DS2 RGB image."""

    if run_mode not in {"contract", "model_inference"}:
        raise ValueError("DS2 LocateAnything mode must be contract or model_inference")
    expected_producer = CONTRACT_PRODUCER_KIND if run_mode == "contract" else MODEL_PRODUCER_KIND
    if str(worker.producer_kind) != expected_producer:
        raise ValueError("LocateAnything worker does not match the requested run mode")
    rgb_root = rgb_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    frozen_configuration, policy_key, freeze_sha256 = load_frozen_ds1_configuration(
        selected_prompt_path
    )
    configuration, execution_policy = resolve_ds2_operational_policy(
        frozen_configuration
    )
    policy_sha256 = execution_policy_sha256(execution_policy)
    if worker.configuration != configuration:
        raise ValueError("LocateAnything worker differs from the effective DS2 configuration")
    if getattr(worker, "execution_policy", None) != execution_policy:
        raise ValueError("LocateAnything worker differs from the effective DS2 execution policy")
    jobs = build_proposal_jobs(rgb_root, DS2_CONFIRM_CONTRACT.cohort_id)
    if len(jobs) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 LocateAnything jobs must cover all confirmation images")

    private_root = output_root / "private"
    safe_root = output_root / "safe"
    model_input_root = output_root / "model-input"
    main_store = ProposalResultStore(
        private_root / "raw_proposal_history.jsonl",
        configuration.sha256,
        expected_producer,
        execution_policy,
    )
    main_store.reconcile_jobs(jobs)

    one_store = ProposalResultStore(
        private_root / "one_image_history.jsonl",
        configuration.sha256,
        expected_producer,
        execution_policy,
    )
    run_proposal_jobs(
        worker=worker,
        jobs=jobs[:1],
        result_store=one_store,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        maximum_consecutive_failed_jobs=None,
    )
    reloaded_one_store = ProposalResultStore(
        private_root / "one_image_history.jsonl",
        configuration.sha256,
        expected_producer,
        execution_policy,
    )
    reloaded_one_store.reconcile_jobs(jobs[:1])

    pilot = _pilot_jobs(jobs)
    pilot_store = ProposalResultStore(
        private_root / "pilot_history.jsonl",
        configuration.sha256,
        expected_producer,
        execution_policy,
    )
    repeat_store = ProposalResultStore(
        private_root / "pilot_repeat_history.jsonl",
        configuration.sha256,
        expected_producer,
        execution_policy,
    )
    run_proposal_jobs(
        worker=worker,
        jobs=pilot,
        result_store=pilot_store,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        maximum_consecutive_failed_jobs=None,
    )
    run_proposal_jobs(
        worker=worker,
        jobs=pilot,
        result_store=repeat_store,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        maximum_consecutive_failed_jobs=None,
    )
    repeat = repeat_agreement(pilot_store.rows, repeat_store.rows)
    pilot_proposals = build_frozen_automatic_proposals(
        pilot_store.rows,
        policy_key=policy_key,
        freeze_sha256=freeze_sha256,
    )
    pilot_history_count_before_resume = len(pilot_store.history_rows)
    run_proposal_jobs(
        worker=worker,
        jobs=pilot,
        result_store=pilot_store,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        maximum_consecutive_failed_jobs=None,
    )
    _require_proposal_readiness(
        one_store.rows,
        reloaded_one_store.rows,
        pilot_store.rows,
        repeat_store.rows,
        pilot_proposals,
        pilot_history_count_before_resume=pilot_history_count_before_resume,
        pilot_history_count_after_resume=len(pilot_store.history_rows),
    )

    started = time.perf_counter()
    session = run_proposal_jobs(
        worker=worker,
        jobs=jobs,
        result_store=main_store,
        maximum_attempts_per_job=DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        maximum_consecutive_failed_jobs=None,
        progress_callback=progress_callback,
    )
    rows = main_store.rows
    if len(rows) != DS2_CONFIRM_CONTRACT.image_count:
        raise RuntimeError("DS2 LocateAnything did not produce one terminal row per image")
    proposals = build_frozen_automatic_proposals(
        rows,
        policy_key=policy_key,
        freeze_sha256=freeze_sha256,
    )
    proposal_path = model_input_root / "automatic_proposals.jsonl"
    _write_jsonl(proposal_path, proposals)
    package_record = {
        "schema_version": 1,
        "package_kind": "ds2-frozen-automatic-proposals",
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "scene_count": DS2_CONFIRM_CONTRACT.scene_count,
        "image_count": DS2_CONFIRM_CONTRACT.image_count,
        "model_id": configuration.model_id,
        "model_revision": configuration.model_revision,
        "prompt": configuration.prompt,
        "postprocessing_policy": policy_key,
        "configuration_sha256": configuration.sha256,
        "ds1_generation_configuration_sha256": frozen_configuration.sha256,
        "effective_generation_configuration_sha256": configuration.sha256,
        "effective_operational_policy_sha256": policy_sha256,
        "effective_max_new_tokens": DS2_MAX_NEW_TOKENS,
        "maximum_attempts_per_job": DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        "attempt_timeout_seconds": DS2_ATTEMPT_TIMEOUT_SECONDS,
        "terminal_failure_disposition": DS2_TERMINAL_FAILURE_DISPOSITION,
        "ds1_freeze_sha256": freeze_sha256,
        "proposal_manifest_sha256": file_sha256(proposal_path),
        "proposal_count": sum(int(row["proposal_count"]) for row in proposals),
        "discarded_untrusted_box_count": sum(
            int(row["discarded_untrusted_box_count"]) for row in proposals
        ),
        "terminal_failed_image_count": sum(row["status"] == "error" for row in rows),
        "carried_zero_proposal_count": sum(
            bool(row["carried_as_zero_proposal"]) for row in proposals
        ),
        "protected_fields_included": False,
        "evaluator_opened": False,
    }
    _write_json(model_input_root / "package_record.json", package_record)
    failure_counts = dict(
        sorted(
            Counter(
                str(row["error_type"])
                for row in rows
                if row["status"] == "error"
            ).items()
        )
    )
    operational = summarize_proposal_rows(rows, execution_policy)
    operational.update(
        {
            "terminal_row_count": len(rows),
            "terminal_zero_proposal_image_count": sum(
                int(row["proposal_count"]) == 0 for row in rows
            ),
            "failure_type_counts": failure_counts,
            "discarded_untrusted_box_count": package_record[
                "discarded_untrusted_box_count"
            ],
            "effective_operational_policy_sha256": policy_sha256,
            "effective_max_new_tokens": DS2_MAX_NEW_TOKENS,
            "maximum_attempts_per_job": DS2_MAXIMUM_ATTEMPTS_PER_JOB,
            "attempt_timeout_seconds": DS2_ATTEMPT_TIMEOUT_SECONDS,
            "terminal_failure_disposition": DS2_TERMINAL_FAILURE_DISPOSITION,
        }
    )
    _write_json(safe_root / "operational_summary.json", operational)
    _write_json(safe_root / "repeat_agreement.json", repeat)
    _write_csv(
        safe_root / "completion_checks.csv",
        [
            {"check": "all DS2 images completed", "passed": len(rows) == DS2_CONFIRM_CONTRACT.image_count},
            {"check": "all terminal failures carried as zero proposals", "passed": package_record["terminal_failed_image_count"] == package_record["carried_zero_proposal_count"]},
            {"check": "DS1 prompt, model and post-processing kept", "passed": True},
            {"check": "bounded DS2 operational policy recorded", "passed": len(policy_sha256) == 64},
            {"check": "evaluator remained closed", "passed": True},
        ],
    )
    proposal_counts = [int(row["proposal_count"]) for row in proposals]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    save_histogram(
        proposal_counts,
        safe_root / "proposal_count_distribution.png",
        title="DS2 LocateAnything proposals per image",
        x_label="Post-processed proposals per image",
        bins=16,
    )
    save_histogram(
        runtimes,
        safe_root / "runtime_distribution.png",
        title="DS2 LocateAnything runtime",
        x_label="Seconds per image",
        bins=24,
    )
    save_bar_chart(
        ["terminal rows", "model errors", "zero proposals"],
        [
            len(rows),
            package_record["terminal_failed_image_count"],
            sum(count == 0 for count in proposal_counts),
        ],
        safe_root / "completion_summary.png",
        title="DS2 LocateAnything completion",
        y_label="Images",
        integer_ticks=True,
        show_values=True,
    )
    _proposal_panel(
        jobs,
        proposals,
        private_root / "figures" / "proposal_examples.png",
    )
    run_record = {
        "schema_version": 1,
        "stage": STAGE,
        "status": "completed",
        "run_mode": run_mode,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "configuration_sha256": configuration.sha256,
        "ds1_generation_configuration_sha256": frozen_configuration.sha256,
        "effective_generation_configuration_sha256": configuration.sha256,
        "effective_operational_policy_sha256": policy_sha256,
        "effective_max_new_tokens": DS2_MAX_NEW_TOKENS,
        "maximum_attempts_per_job": DS2_MAXIMUM_ATTEMPTS_PER_JOB,
        "attempt_timeout_seconds": DS2_ATTEMPT_TIMEOUT_SECONDS,
        "terminal_failure_disposition": DS2_TERMINAL_FAILURE_DISPOSITION,
        "ds1_freeze_sha256": freeze_sha256,
        "proposal_package_sha256": file_sha256(model_input_root / "package_record.json"),
        "image_count": len(rows),
        "proposal_count": package_record["proposal_count"],
        "discarded_untrusted_box_count": package_record[
            "discarded_untrusted_box_count"
        ],
        "zero_proposal_image_count": sum(int(row["proposal_count"]) == 0 for row in proposals),
        "terminal_failed_image_count": package_record["terminal_failed_image_count"],
        "carried_zero_proposal_count": package_record["carried_zero_proposal_count"],
        "failure_type_counts": failure_counts,
        "pilot_box_agreement_rate": repeat["box_agreement_rate"],
        "one_image_gate_passed": len(one_store.rows) == 1 and one_store.rows[0]["status"] == "ok",
        "one_image_status": one_store.rows[0]["status"],
        "runtime_seconds_this_session": time.perf_counter() - started,
        "session": session,
        "evaluator_opened": False,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(safe_root)
    _artifact_manifest(private_root)
    _artifact_manifest(model_input_root)
    return run_record


__all__ = [
    "ALLOWED_POLICIES",
    "build_frozen_automatic_proposals",
    "load_frozen_ds1_configuration",
    "resolve_ds2_operational_policy",
    "run_ds2_locateanything",
]
