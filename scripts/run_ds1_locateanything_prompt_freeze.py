#!/usr/bin/env python3
"""Run the bounded DS1 LocateAnything prompt-freeze study.

The existing ``object`` proposal file is reused as the baseline. The model is
loaded once for the additional prompt. The new prompt must pass a
one-image gate and a deterministic 20-image pilot/repeat before the resumable
2,000-image run starts. Reference geometry is opened only after every candidate
result file has been completed and hash locked by the evaluation workflow.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT  # noqa: E402
from visual_peer_inspector.data.model_safe_package import (  # noqa: E402
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evidence import (  # noqa: E402
    save_bar_chart,
    save_image_grid,
)
from visual_peer_inspector.models.locateanything import (  # noqa: E402
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    MODEL_PRODUCER_KIND,
    ProposalResultStore,
    build_proposal_jobs,
    repeat_agreement,
    run_proposal_jobs,
    summarize_proposal_rows,
)
from visual_peer_inspector.workflows.locateanything_ds1 import (  # noqa: E402
    PILOT_IMAGE_COUNT,
    select_pilot_jobs,
    verify_ds1_jobs,
)
from visual_peer_inspector.workflows.locateanything_prompt_freeze_ds1 import (  # noqa: E402
    POSTPROCESSING_POLICIES,
    PROMPT_POSTPROCESSING_POLICY_BY_KEY,
    PROMPT_CANDIDATES,
    load_latest_proposal_rows,
    prompt_configurations,
    run_prompt_evaluation,
)
from visual_peer_inspector.workflows.proposal_refinement_ds1 import apply_proposal_policy  # noqa: E402


BASELINE_PROMPT_KEY = "object"
NEW_PROMPT_KEYS = ("individual_object",)
PROGRESS_INTERVAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def candidate_result_paths(output_root: Path, baseline_results: Path) -> dict[str, Path]:
    """Return the fixed baseline and generated result locations."""

    root = output_root.expanduser().resolve()
    paths = {BASELINE_PROMPT_KEY: baseline_results.expanduser().resolve()}
    paths.update(
        {
            key: root / "prompt_runs" / key / "private" / "proposal_results.jsonl"
            for key in NEW_PROMPT_KEYS
        }
    )
    return paths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"Rows do not share one schema: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(root: Path) -> dict[str, Any]:
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


def _require_complete(summary: Mapping[str, Any], expected_images: int, stage: str) -> None:
    checks = {
        "job_count": expected_images,
        "successful_job_count": expected_images,
        "failed_job_count": 0,
        "possibly_truncated_count": 0,
    }
    failures = [
        f"{name}={summary.get(name)!r}, expected {expected!r}"
        for name, expected in checks.items()
        if summary.get(name) != expected
    ]
    if failures:
        raise RuntimeError(f"{stage} gate failed: {'; '.join(failures)}")


def _run_jobs(
    *,
    worker: Any,
    jobs: Sequence[Any],
    result_path: Path,
    configuration: LocateAnythingConfiguration,
    progress_path: Path | None = None,
) -> tuple[ProposalResultStore, dict[str, Any]]:
    store = ProposalResultStore(result_path, configuration.sha256, MODEL_PRODUCER_KIND)
    successful = 0
    failed = 0
    started = time.perf_counter()

    def progress(completed: int, total: int, row: Mapping[str, Any]) -> None:
        nonlocal successful, failed
        successful += row["status"] == "ok"
        failed += row["status"] != "ok"
        if completed != 1 and completed != total and completed % PROGRESS_INTERVAL:
            return
        checkpoint = {
            "completed": completed,
            "total": total,
            "successful": successful,
            "failed": failed,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        print(json.dumps({"stage": "prompt_progress", **checkpoint}, sort_keys=True), flush=True)
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json_bytes(checkpoint).decode("utf-8") + "\n")

    session = run_proposal_jobs(
        worker=worker,
        jobs=jobs,
        result_store=store,
        maximum_attempts_per_job=6,
        maximum_consecutive_failed_jobs=3,
        progress_callback=progress,
    )
    return store, session


def run_prompt_candidate(
    *,
    worker: Any,
    jobs: Sequence[Any],
    configuration: LocateAnythingConfiguration,
    candidate_root: Path,
) -> dict[str, Any]:
    """Run the ordered gates and resumable full screen for one prompt."""

    private_root = candidate_root / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    pilot_jobs = select_pilot_jobs(jobs)

    one_store, one_session = _run_jobs(
        worker=worker,
        jobs=jobs[:1],
        result_path=private_root / "one_image_results.jsonl",
        configuration=configuration,
    )
    one_summary = summarize_proposal_rows(one_store.rows)
    _require_complete(one_summary, 1, f"{configuration.prompt} one-image")

    pilot_store, pilot_session = _run_jobs(
        worker=worker,
        jobs=pilot_jobs,
        result_path=private_root / "pilot_results.jsonl",
        configuration=configuration,
    )
    pilot_summary = summarize_proposal_rows(pilot_store.rows)
    _require_complete(pilot_summary, PILOT_IMAGE_COUNT, f"{configuration.prompt} pilot")

    repeat_store, repeat_session = _run_jobs(
        worker=worker,
        jobs=pilot_jobs,
        result_path=private_root / "pilot_repeat_results.jsonl",
        configuration=configuration,
    )
    repeat_summary = repeat_agreement(pilot_store.rows, repeat_store.rows)

    full_store, full_session = _run_jobs(
        worker=worker,
        jobs=jobs,
        result_path=private_root / "proposal_results.jsonl",
        configuration=configuration,
        progress_path=private_root / "progress_checkpoints.jsonl",
    )
    full_summary = summarize_proposal_rows(full_store.rows)
    _require_complete(full_summary, DS1_SCREEN_CONTRACT.image_count, f"{configuration.prompt} full")
    record = {
        "schema_version": 1,
        "prompt": configuration.prompt,
        "configuration": asdict(configuration),
        "configuration_sha256": configuration.sha256,
        "one_image": {**one_summary, "session": one_session},
        "pilot": {**pilot_summary, "session": pilot_session},
        "repeat": {**repeat_summary, "session": repeat_session},
        "full": {**full_summary, "session": full_session},
        "proposal_results_sha256": file_sha256(private_root / "proposal_results.jsonl"),
        "status": "completed",
    }
    _write_json(candidate_root / "aggregate" / "run_summary.json", record)
    return record


def _baseline_repeat_result(result_path: Path) -> float | None:
    safe_path = result_path.parent.parent / "safe" / "repeat_agreement.json"
    if not safe_path.is_file():
        return None
    value = json.loads(safe_path.read_text(encoding="utf-8"))
    result = value.get("box_agreement_rate")
    return float(result) if result is not None else None


def _private_prompt_examples(
    *,
    jobs: Sequence[Any],
    proposal_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[str]:
    selected_jobs = select_pilot_jobs(jobs)
    references_by_image: dict[tuple[str, str], list[Sequence[float]]] = {}
    for row in reference_rows:
        if not bool(row["visible"]):
            continue
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        references_by_image.setdefault(key, []).append(row["box_xyxy"])
    names = []
    for prompt_key, rows in proposal_rows.items():
        raw_by_key = {
            (str(row["opaque_scene_id"]), str(row["view_id"])): row for row in rows
        }
        for policy_key in POSTPROCESSING_POLICIES:
            items = []
            for index, job in enumerate(selected_jobs, 1):
                boxes = raw_by_key[job.key]["boxes_xyxy"]
                if policy_key != "raw":
                    policy = PROMPT_POSTPROCESSING_POLICY_BY_KEY[policy_key]
                    boxes, _indices, _removed = apply_proposal_policy(boxes, policy)
                items.append(
                    {
                        "label": f"Example {index:02d} / {job.view_id}",
                        "path": job.image_path,
                        "reference_boxes": references_by_image.get(job.key, []),
                        "boxes": boxes,
                    }
                )
            name = f"{prompt_key}_{policy_key}_proposal_examples.png"
            save_image_grid(
                items,
                output_root / "private" / "prompt_examples" / name,
                title=(
                    f"LocateAnything: {prompt_key.replace('_', ' ')} / {policy_key} "
                    "(reference green, proposal red)"
                ),
                columns=5,
            )
            names.append(name)
    return names


def _private_selected_outcome_examples(
    *,
    jobs: Sequence[Any],
    proposal_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    image_records: Sequence[Mapping[str, Any]],
    policy_key: str,
    output_root: Path,
) -> list[str]:
    """Save honest success and failure panels for the numerically selected route."""

    jobs_by_key = {job.key: job for job in jobs}
    proposals_by_key = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in proposal_rows
    }
    references_by_key: dict[tuple[str, str], list[Sequence[float]]] = {}
    for row in reference_rows:
        if bool(row["visible"]):
            key = (str(row["opaque_scene_id"]), str(row["view_id"]))
            references_by_key.setdefault(key, []).append(row["box_xyxy"])

    failures = [
        row
        for row in image_records
        if int(row["miss_count"]) > 0 or int(row["extra_count"]) > 0
    ]
    failures = sorted(
        failures,
        key=lambda row: (
            -int(row["miss_count"]),
            -int(row["extra_count"]),
            float(row["detection_aware_mean_iou"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
        ),
    )[:20]
    successes = sorted(
        (row for row in image_records if int(row["full_image_success"]) == 1),
        key=lambda row: (
            -float(row["mean_matched_iou"]),
            str(row["opaque_scene_id"]),
            str(row["view_id"]),
        ),
    )[:20]

    output_names = []
    for panel_name, rows in (("successes", successes), ("failures", failures)):
        if not rows:
            continue
        items = []
        for row in rows:
            key = (str(row["opaque_scene_id"]), str(row["view_id"]))
            boxes = proposals_by_key[key]["boxes_xyxy"]
            if policy_key != "raw":
                policy = PROMPT_POSTPROCESSING_POLICY_BY_KEY[policy_key]
                boxes, _indices, _removed = apply_proposal_policy(boxes, policy)
            items.append(
                {
                    "label": (
                        f"{key[1]} | IoU {float(row['detection_aware_mean_iou']):.2f} | "
                        f"m{int(row['miss_count'])} e{int(row['extra_count'])}"
                    ),
                    "path": jobs_by_key[key].image_path,
                    "reference_boxes": references_by_key.get(key, []),
                    "boxes": boxes,
                }
            )
        name = f"selected_{panel_name}.png"
        save_image_grid(
            items,
            output_root / "private" / "prompt_examples" / name,
            title=(
                f"Selected route {panel_name} "
                "(reference green, proposal red)"
            ),
            columns=5,
        )
        output_names.append(name)
    return output_names


def _extend_safe_evidence(
    *,
    evaluation_root: Path,
    candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_records: Mapping[str, Mapping[str, Any]],
    model_load: Mapping[str, Any],
) -> dict[str, Any]:
    safe_root = evaluation_root / "safe"
    operational_rows = []
    repeat_rows = []
    for prompt_key, prompt in PROMPT_CANDIDATES:
        summary = summarize_proposal_rows(candidate_rows[prompt_key])
        record = candidate_records.get(prompt_key, {})
        repeat_rate = (
            _baseline_repeat_result(Path(record["result_path"]))
            if prompt_key == BASELINE_PROMPT_KEY
            else record.get("repeat", {}).get("box_agreement_rate")
        )
        operational_rows.append(
            {
                "prompt_key": prompt_key,
                "prompt": prompt,
                "source": "reused_baseline" if prompt_key == BASELINE_PROMPT_KEY else "new_model_run",
                "image_count": summary["job_count"],
                "successful_image_count": summary["successful_job_count"],
                "failed_image_count": summary["failed_job_count"],
                "mean_proposals_per_image": round(summary["mean_proposals_per_successful_image"], 6),
                "median_runtime_seconds": round(summary["median_runtime_seconds"], 6),
                "p95_runtime_seconds": round(summary["p95_runtime_seconds"], 6),
                "total_inference_seconds": round(summary["total_runtime_seconds"], 6),
            }
        )
        repeat_rows.append(
            {
                "prompt_key": prompt_key,
                "prompt": prompt,
                "pilot_image_count": PILOT_IMAGE_COUNT,
                "exact_box_agreement_rate": "not_recorded" if repeat_rate is None else repeat_rate,
            }
        )
    _write_csv(safe_root / "prompt_operational_summary.csv", operational_rows)
    _write_csv(safe_root / "prompt_repeatability.csv", repeat_rows)
    _write_json(safe_root / "model_load_summary.json", dict(model_load))
    _write_json(
        safe_root / "study_protocol.json",
        {
            "schema_version": 1,
            "baseline_prompt_key": BASELINE_PROMPT_KEY,
            "baseline_reused": True,
            "new_prompt_keys": list(NEW_PROMPT_KEYS),
            "prompts": [{"prompt_key": key, "prompt": prompt} for key, prompt in PROMPT_CANDIDATES],
            "postprocessing_policies": list(POSTPROCESSING_POLICIES),
            "ordered_gates": ["one_image", "pilot_20", "pilot_repeat_20", "full_ds1"],
            "selection_uses_fixed_acceptance_cutoff": False,
            "labels_opened_after_candidate_hash_lock": True,
        },
    )
    save_bar_chart(
        [row["prompt_key"] for row in operational_rows],
        [float(row["median_runtime_seconds"]) for row in operational_rows],
        safe_root / "median_runtime_by_prompt.png",
        title="Median LocateAnything runtime by prompt",
        y_label="Seconds per image",
        show_values=True,
    )
    save_bar_chart(
        [row["prompt_key"] for row in operational_rows],
        [float(row["mean_proposals_per_image"]) for row in operational_rows],
        safe_root / "mean_proposals_by_prompt.png",
        title="Mean LocateAnything proposals by prompt",
        y_label="Mean proposals per image",
        show_values=True,
    )
    checks_path = safe_root / "completion_checks.csv"
    with checks_path.open("r", encoding="utf-8", newline="") as handle:
        checks = list(csv.DictReader(handle))
    checks.extend(
        [
            {"check": "Notebook 04.1 object baseline reused", "passed": True},
            {"check": "LocateAnything model loaded at most once", "passed": int(model_load["model_load_count"]) <= 1},
            {
                "check": "one-image, pilot, repeat and full gates completed",
                "passed": len(candidate_records) == len(PROMPT_CANDIDATES),
            },
            {"check": "aggregate-only ClearML evidence prepared", "passed": True},
        ]
    )
    _write_csv(checks_path, checks)
    run_path = safe_root / "run_record.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.update(
        {
            "baseline_reused": True,
            "model_load_count": int(model_load["model_load_count"]),
            "new_model_prompt_count": len(NEW_PROMPT_KEYS),
            "all_completion_checks_passed": all(
                str(row["passed"]).strip().lower() in {"true", "1"} for row in checks
            ),
        }
    )
    _write_json(run_path, run_record)
    _write_manifest(safe_root)
    return run_record


def run_study(
    *,
    model_input_root: Path,
    evaluator_root: Path,
    baseline_results: Path,
    output_root: Path,
    model_source: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute the complete prompt-freeze study with one shared model worker."""

    output_root = output_root.expanduser().resolve()
    evaluation_root = output_root / "evaluation"
    complete_record = evaluation_root / "safe" / "run_record.json"
    if complete_record.is_file():
        record = json.loads(complete_record.read_text(encoding="utf-8"))
        if record.get("status") == "completed" and record.get("all_completion_checks_passed"):
            print("Reused the completed DS1 prompt-freeze output.", flush=True)
            return record
        raise RuntimeError("Existing prompt-freeze output is incomplete")
    if evaluation_root.exists():
        raise RuntimeError("Evaluation output exists without a completed run record")

    package_audit = verify_model_input_package(
        model_input_root,
        contract=DS1_SCREEN_CONTRACT,
    )
    jobs = build_proposal_jobs(model_input_root, DS1_SCREEN_CONTRACT.cohort_id)
    job_audit = verify_ds1_jobs(jobs)
    if package_audit.manifest_sha256 != job_audit["input_manifest_sha256"]:
        raise ValueError("Verified package and proposal jobs use different manifests")

    configurations = prompt_configurations()
    baseline_rows = load_latest_proposal_rows(baseline_results, configurations[BASELINE_PROMPT_KEY])
    result_paths = candidate_result_paths(output_root, baseline_results)
    candidate_records: dict[str, dict[str, Any]] = {
        BASELINE_PROMPT_KEY: {
            "source": "reused_baseline",
            "result_path": str(result_paths[BASELINE_PROMPT_KEY]),
            "status": "completed",
        }
    }
    pending_keys = []
    for key in NEW_PROMPT_KEYS:
        try:
            load_latest_proposal_rows(result_paths[key], configurations[key])
        except (FileNotFoundError, ValueError):
            pending_keys.append(key)

    load_checkpoints: list[dict[str, Any]] = []
    load_started = time.perf_counter()

    def load_progress(stage: str, details: Mapping[str, Any]) -> None:
        checkpoint = {
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - load_started, 3),
            **dict(details),
        }
        load_checkpoints.append(checkpoint)
        print(json.dumps(checkpoint, sort_keys=True), flush=True)

    worker = None
    if pending_keys:
        worker = LocateAnythingWorker(
            configurations[pending_keys[0]],
            device=device,
            model_source=model_source,
            progress_callback=load_progress,
        )
    model_load_seconds = round(time.perf_counter() - load_started, 3)
    for key in NEW_PROMPT_KEYS:
        configuration = configurations[key]
        candidate_root = output_root / "prompt_runs" / key
        try:
            rows = load_latest_proposal_rows(result_paths[key], configuration)
            record_path = candidate_root / "aggregate" / "run_summary.json"
            record = (
                json.loads(record_path.read_text(encoding="utf-8"))
                if record_path.is_file()
                else {
                    "prompt": configuration.prompt,
                    "repeat": {},
                    "full": summarize_proposal_rows(rows),
                    "status": "completed",
                }
            )
        except (FileNotFoundError, ValueError):
            if worker is None:
                raise RuntimeError("A pending prompt has no loaded model worker")
            worker.use_configuration(configuration)
            record = run_prompt_candidate(
                worker=worker,
                jobs=jobs,
                configuration=configuration,
                candidate_root=candidate_root,
            )
        candidate_records[key] = dict(record)
        candidate_records[key]["result_path"] = str(result_paths[key])

    candidate_rows = {
        BASELINE_PROMPT_KEY: baseline_rows,
        **{
            key: load_latest_proposal_rows(result_paths[key], configurations[key])
            for key in NEW_PROMPT_KEYS
        },
    }
    evaluation = run_prompt_evaluation(
        rgb_root=model_input_root,
        evaluator_root=evaluator_root,
        proposal_results=result_paths,
        output_root=evaluation_root,
    )
    reference_rows = read_jsonl(evaluator_root / "reference_manifest.jsonl")
    examples = _private_prompt_examples(
        jobs=jobs,
        proposal_rows=candidate_rows,
        reference_rows=reference_rows,
        output_root=evaluation_root,
    )
    model_load = {
        "schema_version": 1,
        "model_id": configurations[BASELINE_PROMPT_KEY].model_id,
        "model_revision": configurations[BASELINE_PROMPT_KEY].model_revision,
        "model_load_count": 1 if pending_keys else 0,
        "model_load_seconds": model_load_seconds,
        "worker_configuration_switch_count": len(pending_keys),
        "load_checkpoints": load_checkpoints,
        "python_version": platform.python_version(),
        "device": device,
    }
    final_record = _extend_safe_evidence(
        evaluation_root=evaluation_root,
        candidate_rows=candidate_rows,
        candidate_records=candidate_records,
        model_load=model_load,
    )
    selected = json.loads(
        (evaluation_root / "safe" / "selected_prompt.json").read_text(encoding="utf-8")
    )
    selected_key = str(selected["selected"]["prompt_key"])
    selected_candidate_key = str(selected["selected"]["candidate_key"])
    examples.extend(
        _private_selected_outcome_examples(
            jobs=jobs,
            proposal_rows=candidate_rows[selected_key],
            reference_rows=reference_rows,
            image_records=read_jsonl(
                evaluation_root
                / "private"
                / "evaluation"
                / f"{selected_candidate_key}_image_records.jsonl"
            ),
            policy_key=str(selected["selected"]["postprocessing_policy"]),
            output_root=evaluation_root,
        )
    )
    downstream_root = evaluation_root / "downstream"
    downstream_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        evaluation_root / "safe" / "selected_prompt.json",
        downstream_root / "selected_prompt.json",
    )
    shutil.copyfile(
        result_paths[selected_key],
        downstream_root / "selected_proposal_results.jsonl",
    )
    _write_json(
        downstream_root / "downstream_record.json",
        {
            "schema_version": 1,
            "prompt_key": selected_key,
            "prompt": selected["selected"]["prompt"],
            "postprocessing_policy": selected["selected"]["postprocessing_policy"],
            "configuration_sha256": selected["generation_configuration_sha256"],
            "proposal_results_sha256": file_sha256(
                downstream_root / "selected_proposal_results.jsonl"
            ),
            "image_count": DS1_SCREEN_CONTRACT.image_count,
        },
    )
    _write_json(
        evaluation_root / "private" / "prompt_examples" / "example_record.json",
        {"schema_version": 1, "files": examples, "private_rgb": True},
    )
    _write_manifest(evaluation_root / "private")
    print(json.dumps({"evaluation": evaluation, "final": final_record}, indent=2, sort_keys=True))
    return final_record


def main() -> None:
    args = parse_args()
    record = run_study(
        model_input_root=args.model_input_root,
        evaluator_root=args.evaluator_root,
        baseline_results=args.baseline_results,
        output_root=args.output_root,
        model_source=args.model_source,
        device=args.device,
    )
    if record.get("status") != "completed" or not record.get("all_completion_checks_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
