"""SAM 2.1 orchestration for the 100-scene DS1 screen.

This module keeps the DS0 workflow unchanged. It consumes a model-safe DS1
prompt package, runs the same pinned SAM implementation, and writes private
mask records separately from aggregate evidence.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import time
from typing import Any, Mapping

from PIL import Image

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
)
from visual_peer_inspector.data.sam_prompt_package_ds1 import (
    verify_ds1_sam_prompt_package,
)
from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_histogram,
    save_scatter_plot,
    write_csv_rows,
)
from visual_peer_inspector.models.sam2 import (
    CONTRACT_PRODUCER_KIND,
    MODEL_PRODUCER_KIND,
    ContractSamWorker,
    Sam2Configuration,
    Sam2ImageWorker,
    SamImageJob,
    SamPrompt,
    SamResultStore,
    collect_sam_evidence,
    compare_repeat_results,
    run_sam_jobs,
)
from visual_peer_inspector.workflows.sam_ds0 import (
    _save_private_mask_grid,
    _save_private_route_comparison_grid,
)


PILOT_IMAGE_COUNT = 20
ROUTES = ("reference_box", "automatic_box")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "manifest_content_sha256": sha256(
                b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
            ).hexdigest(),
        },
    )


def _safe_media_path(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("DS1 SAM image path is unsafe")
    resolved = root.joinpath(*relative.parts).resolve()
    if root not in resolved.parents:
        raise ValueError("DS1 SAM image path escapes the media root")
    return resolved


def _normalise_box(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("DS1 SAM prompts require four integer XYXY coordinates")
    values = tuple(int(item) for item in value)
    if list(values) != value:
        raise ValueError("DS1 SAM prompt coordinates must be integers")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise ValueError("DS1 SAM prompt box is outside the 256 by 256 image")
    return values


def import_ds1_sam_resume(resume_root: Path, output_root: Path) -> dict[str, Any]:
    """Import only atomic DS1 result checkpoints from a prior private run."""

    resume_root = resume_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if (
        resume_root == output_root
        or resume_root in output_root.parents
        or output_root in resume_root.parents
    ):
        raise ValueError("DS1 SAM resume source and output root must be separate")
    source_private = resume_root / "private"
    if source_private.is_symlink() or not source_private.is_dir():
        raise ValueError("DS1 SAM resume source has no private result root")
    copied = 0
    reused = 0
    for directory in ("image_results", "repeat_image_results"):
        source_directory = source_private / directory
        if not source_directory.exists():
            continue
        if source_directory.is_symlink() or not source_directory.is_dir():
            raise ValueError("DS1 SAM resume result directory is unsafe")
        for source in sorted(source_directory.glob("S??????/V??.json")):
            if source.is_symlink() or not source.is_file():
                raise ValueError("DS1 SAM resume result file is unsafe")
            relative = source.relative_to(source_private)
            destination = output_root / "private" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_symlink() or destination.read_bytes() != source.read_bytes():
                    raise ValueError("A local DS1 SAM checkpoint differs from the resume source")
                reused += 1
            else:
                temporary = destination.with_suffix(".json.incomplete")
                shutil.copyfile(source, temporary)
                os.replace(temporary, destination)
                copied += 1
    return {
        "used": True,
        "copied_result_files": copied,
        "reused_result_files": reused,
        "attempt_history_copied": False,
        "attempt_history_policy": "not_imported_without_bound_run_identity",
    }


def build_ds1_sam_jobs(prompt_root: Path, media_root: Path) -> list[SamImageJob]:
    """Validate the DS1 prompt package and build all 2,000 image jobs."""

    prompt_root = prompt_root.expanduser().resolve(strict=True)
    media_root = media_root.expanduser().resolve(strict=True)
    record_path = prompt_root / "package_record.json"
    manifest_path = prompt_root / "sam_prompt_manifest.jsonl"
    record = verify_ds1_sam_prompt_package(prompt_root)

    rows = read_jsonl(manifest_path)
    if len(rows) != DS1_SCREEN_CONTRACT.image_count:
        raise ValueError("DS1 SAM prompt package must contain 2,000 images")
    package_sha256 = file_sha256(record_path)
    jobs: list[SamImageJob] = []
    keys: set[tuple[str, str]] = set()
    scene_ids: set[str] = set()
    views_by_scene: dict[str, set[str]] = {}
    route_counts = Counter()
    for row in rows:
        if row.get("cohort_id") != DS1_SCREEN_CONTRACT.cohort_id:
            raise ValueError("DS1 SAM prompt row identifies another cohort")
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        key = (scene_id, view_id)
        if key in keys:
            raise ValueError("DS1 SAM prompt package contains a duplicate image")
        keys.add(key)
        scene_ids.add(scene_id)
        views_by_scene.setdefault(scene_id, set()).add(view_id)
        route_mapping = row.get("routes")
        if not isinstance(route_mapping, dict) or set(route_mapping) != set(ROUTES):
            raise ValueError("DS1 SAM prompt package must contain both frozen routes")
        prompts: list[SamPrompt] = []
        for route, prefix in (("reference_box", "R"), ("automatic_box", "P")):
            route_prompts = route_mapping[route]
            expected_keys = [f"{prefix}{index:03d}" for index in range(len(route_prompts))]
            observed_keys = [str(prompt["prompt_key"]) for prompt in route_prompts]
            boxes = [_normalise_box(prompt["box_xyxy"]) for prompt in route_prompts]
            if observed_keys != expected_keys or boxes != sorted(boxes):
                raise ValueError("DS1 SAM prompts are not in canonical geometric order")
            route_counts[route] += len(route_prompts)
            prompts.extend(
                SamPrompt(route=route, prompt_key=prompt_key, box_xyxy=box)
                for prompt_key, box in zip(observed_keys, boxes, strict=True)
            )
        image_path = _safe_media_path(media_root, str(row["image_path"]))
        image_sha256 = file_sha256(image_path)
        if image_sha256 != str(row["image_sha256"]):
            raise ValueError("DS1 SAM image hash differs from the prompt manifest")
        jobs.append(
            SamImageJob(
                opaque_scene_id=scene_id,
                view_id=view_id,
                image_path=image_path,
                image_sha256=image_sha256,
                prompts=tuple(prompts),
                prompt_package_sha256=package_sha256,
                cohort_id=DS1_SCREEN_CONTRACT.cohort_id,
            )
        )
    if len(scene_ids) != DS1_SCREEN_CONTRACT.scene_count:
        raise ValueError("DS1 SAM prompt package must contain 100 scenes")
    if any(
        len(view_ids) != DS1_SCREEN_CONTRACT.views_per_scene
        for view_ids in views_by_scene.values()
    ):
        raise ValueError("Every DS1 SAM scene must contain all 20 views")
    for route, key in (
        ("reference_box", "reference_prompt_count"),
        ("automatic_box", "automatic_prompt_count"),
    ):
        if int(record.get(key, -1)) != route_counts[route]:
            raise ValueError(f"DS1 {route} prompt count differs from the package record")
    return sorted(jobs, key=lambda job: job.key)


def _pilot_jobs(jobs: list[SamImageJob], stress_job: SamImageJob) -> list[SamImageJob]:
    by_key = {job.key: job for job in jobs}
    scenes = sorted({job.opaque_scene_id for job in jobs})
    views = sorted({job.view_id for job in jobs})
    selected = [stress_job]
    unused_scenes = [value for value in scenes if value != stress_job.opaque_scene_id]
    unused_views = [value for value in views if value != stress_job.view_id]
    selected.extend(
        by_key[(scene_id, view_id)]
        for scene_id, view_id in zip(unused_scenes[:19], unused_views[:19], strict=True)
    )
    if (
        len(selected) != PILOT_IMAGE_COUNT
        or len({job.opaque_scene_id for job in selected}) != PILOT_IMAGE_COUNT
        or len({job.view_id for job in selected}) != PILOT_IMAGE_COUNT
    ):
        raise ValueError("The DS1 SAM pilot must span 20 scenes and 20 view positions")
    return sorted(selected, key=lambda job: job.key)


def _attempt_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["status"]), int(row["attempt"])) for row in rows)
    summary = [
        {"status": status, "attempt": attempt, "count": count}
        for (status, attempt), count in sorted(counts.items())
    ]
    return summary or [{"status": "not_run_this_session", "attempt": 0, "count": 0}]


def _failure_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        str(row["error_type"]) for row in rows if str(row["status"]) == "error"
    )
    return (
        [{"error_type": key, "count": value} for key, value in sorted(counts.items())]
        if counts
        else [{"error_type": "none", "count": 0}]
    )


def run_ds1_sam_workflow(
    *,
    prompt_root: Path,
    media_root: Path,
    output_root: Path,
    configuration: Sam2Configuration,
    run_mode: str,
    inference_stage: str,
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None,
    provenance: Mapping[str, Any],
    platform_versions: Mapping[str, Any],
    gpu_evidence: Mapping[str, Any],
    model_load_rows: list[Mapping[str, Any]],
    progress_callback=None,
) -> dict[str, Any]:
    """Run DS1 preflight, repeatability, resumable inference and evidence."""

    if run_mode not in {"contract_smoke", "model_inference"}:
        raise ValueError("run_mode must be contract_smoke or model_inference")
    if inference_stage not in {"one_image", "pilot", "full"}:
        raise ValueError("inference_stage must be one_image, pilot or full")
    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    safe_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    jobs = build_ds1_sam_jobs(prompt_root, media_root)
    job_by_key = {job.key: job for job in jobs}
    stress_job = max(jobs, key=lambda job: (len(job.prompts), job.key))
    pilot_jobs = _pilot_jobs(jobs, stress_job)
    producer_kind = MODEL_PRODUCER_KIND if run_mode == "model_inference" else CONTRACT_PRODUCER_KIND
    result_store = SamResultStore(
        private_root / "image_results",
        configuration.sha256,
        jobs[0].prompt_package_sha256,
        producer_kind,
        DS1_SCREEN_CONTRACT.cohort_id,
    )
    result_store.reconcile_jobs(jobs)
    repeat_store = SamResultStore(
        private_root / "repeat_image_results",
        configuration.sha256,
        jobs[0].prompt_package_sha256,
        producer_kind,
        DS1_SCREEN_CONTRACT.cohort_id,
    )
    repeat_store.reconcile_jobs(pilot_jobs)
    required_main = [stress_job] if inference_stage == "one_image" else pilot_jobs if inference_stage == "pilot" else jobs
    required_repeat = pilot_jobs if inference_stage in {"pilot", "full"} else []
    needs_worker = any(job.key not in result_store.completed_keys for job in required_main) or any(
        job.key not in repeat_store.completed_keys for job in required_repeat
    )
    worker = None
    if needs_worker and run_mode == "model_inference":
        if checkpoint_path is None or checkpoint_sha256 != configuration.checkpoint_sha256:
            raise ValueError("Pending DS1 inference requires the pinned SAM checkpoint")
        worker = Sam2ImageWorker(configuration, checkpoint_path, progress_callback=progress_callback)
    elif needs_worker:
        worker = ContractSamWorker(configuration)
    elif run_mode == "model_inference" and checkpoint_sha256 != configuration.checkpoint_sha256:
        raise ValueError("Resumed DS1 SAM checkpoint identity differs")

    attempts_path = private_root / "attempts.jsonl"
    attempts = read_jsonl(attempts_path) if attempts_path.is_file() else []
    valid_keys = {job.key for job in jobs}
    if any((str(row.get("opaque_scene_id")), str(row.get("view_id"))) not in valid_keys for row in attempts):
        raise ValueError("Stored DS1 SAM attempt history contains another image")
    started = time.perf_counter()

    def record_attempt(row: Mapping[str, Any]) -> None:
        value = dict(row)
        attempts.append(value)
        with attempts_path.open("ab") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
            handle.flush()

    def report(completed: int, total: int, row: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(
                "inference_progress",
                {"completed": completed, "total": total, "latest_status": row["status"]},
            )

    if worker is not None:
        run_sam_jobs(worker=worker, jobs=[stress_job], result_store=result_store, maximum_consecutive_failed_jobs=1, progress_callback=report, attempt_callback=record_attempt)
    if stress_job.key not in result_store.completed_keys:
        raise RuntimeError("The one-image DS1 SAM preflight did not complete")
    if inference_stage in {"pilot", "full"} and worker is not None:
        run_sam_jobs(worker=worker, jobs=pilot_jobs, result_store=result_store, maximum_consecutive_failed_jobs=1, progress_callback=report, attempt_callback=record_attempt)
        run_sam_jobs(worker=worker, jobs=pilot_jobs, result_store=repeat_store, maximum_consecutive_failed_jobs=1, progress_callback=report, attempt_callback=record_attempt)

    pilot_keys = {job.key for job in pilot_jobs}
    pilot_rows = [row for row in result_store.rows if (str(row["opaque_scene_id"]), str(row["view_id"])) in pilot_keys]
    repeat = (
        compare_repeat_results(pilot_rows, repeat_store.rows)
        if inference_stage in {"pilot", "full"}
        else {"prompt_count": 0, "mask_digest_agreement_rate": 0.0, "score_agreement_rate": 0.0}
    )
    preflight = {
        "schema_version": 1,
        "one_image_completed": stress_job.key in result_store.completed_keys,
        "pilot_required": inference_stage in {"pilot", "full"},
        "pilot_image_count": len(pilot_rows),
        "pilot_scene_count": len({str(row["opaque_scene_id"]) for row in pilot_rows}),
        "pilot_view_count": len({str(row["view_id"]) for row in pilot_rows}),
        "repeat_prompt_count": repeat["prompt_count"],
        "repeat_mask_digest_agreement_rate": repeat["mask_digest_agreement_rate"],
        "repeat_score_agreement_rate": repeat["score_agreement_rate"],
    }
    preflight["passed"] = preflight["one_image_completed"] and (
        inference_stage == "one_image"
        or (
            preflight["pilot_image_count"] == PILOT_IMAGE_COUNT
            and preflight["pilot_scene_count"] == PILOT_IMAGE_COUNT
            and preflight["pilot_view_count"] == PILOT_IMAGE_COUNT
            and repeat["prompt_count"] == sum(len(job.prompts) for job in pilot_jobs)
            and repeat["mask_digest_agreement_rate"] == 1.0
            and repeat["score_agreement_rate"] == 1.0
        )
    )
    _write_json(safe_root / "preflight_gates.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("DS1 SAM preflight gates failed")

    run_error = None
    if inference_stage == "full" and worker is not None:
        try:
            run_sam_jobs(worker=worker, jobs=jobs, result_store=result_store, maximum_consecutive_failed_jobs=2, progress_callback=report, attempt_callback=record_attempt)
        except Exception as error:
            run_error = error
    result_store.write_index(private_root / "mask_results.jsonl")
    if repeat_store.rows:
        repeat_store.write_index(private_root / "repeat_mask_results.jsonl")
    rows = result_store.rows
    evidence = collect_sam_evidence(rows)
    route_counts = {route: sum(len(row["routes"][route]) for row in rows) for route in ROUTES}
    expected_jobs = [stress_job] if inference_stage == "one_image" else pilot_jobs if inference_stage == "pilot" else jobs
    expected_route_counts = {
        route: sum(sum(prompt.route == route for prompt in job.prompts) for job in expected_jobs)
        for route in ROUTES
    }
    expected_images = 1 if inference_stage == "one_image" else PILOT_IMAGE_COUNT if inference_stage == "pilot" else DS1_SCREEN_CONTRACT.image_count
    unresolved = {job.key for job in expected_jobs} - result_store.completed_keys
    gates = [
        {"gate": "image coverage", "observed": evidence["image_count"], "expected": expected_images, "status": "pass" if evidence["image_count"] == expected_images else "fail"},
        {"gate": "reference-route masks", "observed": route_counts["reference_box"], "expected": expected_route_counts["reference_box"], "status": "pass" if route_counts["reference_box"] == expected_route_counts["reference_box"] else "fail"},
        {"gate": "automatic-route masks", "observed": route_counts["automatic_box"], "expected": expected_route_counts["automatic_box"], "status": "pass" if route_counts["automatic_box"] == expected_route_counts["automatic_box"] else "fail"},
        {"gate": "unresolved failures", "observed": len(unresolved), "expected": 0, "status": "pass" if not unresolved else "fail"},
        {"gate": "repeat mask digest agreement", "observed": repeat["mask_digest_agreement_rate"], "expected": 1.0 if inference_stage != "one_image" else "not assessed", "status": "pass" if inference_stage == "one_image" or repeat["mask_digest_agreement_rate"] == 1.0 else "fail"},
        {"gate": "repeat score agreement", "observed": repeat["score_agreement_rate"], "expected": 1.0 if inference_stage != "one_image" else "not assessed", "status": "pass" if inference_stage == "one_image" or repeat["score_agreement_rate"] == 1.0 else "fail"},
    ]
    gate_passed = all(row["status"] == "pass" for row in gates)
    write_csv_rows(safe_root / "completion_gates.csv", ["gate", "observed", "expected", "status"], gates)
    write_csv_rows(safe_root / "route_summary.csv", ["route", "mask_count", "empty_mask_count", "empty_mask_rate", "mean_area_fraction", "median_area_fraction", "p95_area_fraction", "mean_predicted_quality_score", "p10_predicted_quality_score"], evidence["route_rows"])
    write_csv_rows(safe_root / "view_summary.csv", ["view_id", "route", "mask_count", "empty_mask_count", "mean_area_fraction", "mean_predicted_quality_score"], evidence["view_rows"])
    write_csv_rows(safe_root / "attempt_summary.csv", ["status", "attempt", "count"], _attempt_summary(attempts))
    write_csv_rows(safe_root / "failure_summary.csv", ["error_type", "count"], _failure_summary(attempts))
    _write_json(safe_root / "repeat_agreement.json", repeat)
    _write_json(safe_root / "runtime_summary.json", {"workflow_seconds": round(time.perf_counter() - started, 4), "median_image_runtime_seconds": evidence["median_image_runtime_seconds"], "p95_image_runtime_seconds": evidence["p95_image_runtime_seconds"], "median_prompts_per_image": evidence["median_prompts_per_image"]})
    _write_json(safe_root / "mask_geometry_summary.json", {"routes": evidence["route_rows"]})
    save_histogram(evidence["image_runtimes"], safe_root / "runtime_distribution.png", title="DS1 SAM 2.1 runtime distribution", x_label="Seconds per image", bins=20)
    save_histogram(evidence["prompt_counts"], safe_root / "mask_count_distribution.png", title="DS1 SAM 2.1 mask count per image", x_label="Masks per image", bins=12)
    save_scatter_plot(evidence["prompt_counts"], evidence["image_runtimes"], safe_root / "runtime_by_prompt_count.png", title="DS1 SAM 2.1 runtime by prompt count", x_label="Prompts per image", y_label="Seconds per image")
    for route, values in evidence["area_values_by_route"].items():
        save_histogram(values, safe_root / f"{route}_mask_area_distribution.png", title=f"{route.replace('_', ' ').title()} mask area distribution", x_label="Mask area as a fraction of image area", bins=20)
    for route, values in evidence["score_values_by_route"].items():
        save_histogram(values, safe_root / f"{route}_predicted_score_distribution.png", title=f"{route.replace('_', ' ').title()} predicted score distribution", x_label="SAM predicted quality score", bins=20)
    save_bar_chart([row["route"] for row in evidence["route_rows"]], [row["mask_count"] for row in evidence["route_rows"]], safe_root / "mask_count_by_route.png", title="Completed DS1 SAM 2.1 masks by route", y_label="Masks")
    if pilot_rows:
        _save_private_mask_grid(private_root / "reference_mask_overlays.png", pilot_rows, job_by_key, "reference_box")
        _save_private_mask_grid(private_root / "automatic_mask_overlays.png", pilot_rows, job_by_key, "automatic_box")
        _save_private_route_comparison_grid(private_root / "mask_route_examples.png", private_root / "qualitative_example_manifest.csv", pilot_rows, job_by_key)

    recorded_gpu = dict(gpu_evidence)
    if isinstance(worker, Sam2ImageWorker):
        recorded_gpu.update({"peak_allocated_gib": round(worker.torch.cuda.max_memory_allocated() / (1024**3), 4), "peak_reserved_gib": round(worker.torch.cuda.max_memory_reserved() / (1024**3), 4)})
    output_identity = {
        "mask_results_filename": "mask_results.jsonl",
        "mask_results_sha256": file_sha256(private_root / "mask_results.jsonl"),
        "mask_result_count": len(rows),
    }
    repeat_result_path = private_root / "repeat_mask_results.jsonl"
    if repeat_result_path.is_file():
        output_identity.update(
            {
                "repeat_mask_results_filename": repeat_result_path.name,
                "repeat_mask_results_sha256": file_sha256(repeat_result_path),
                "repeat_mask_result_count": len(repeat_store.rows),
            }
        )
    run_record = {
        "schema_version": 1,
        "stage": "ds1_sam21_prompted_masks",
        "status": "completed" if gate_passed else "failed",
        "tracking_status": "pending",
        "cohort": {
            key: getattr(DS1_SCREEN_CONTRACT, key)
            for key in ("cohort_id", "release_id", "role", "pilot_stage", "membership_sha256", "scene_count", "views_per_scene", "image_count", "asset_group_count")
        },
        "run_mode": run_mode,
        "inference_stage": inference_stage,
        "producer_kind": producer_kind,
        "python_version": platform.python_version(),
        "configuration": configuration.to_dict(),
        "configuration_sha256": configuration.sha256,
        "prompt_package_sha256": jobs[0].prompt_package_sha256,
        "input_identity": {
            "prompt_package_record_sha256": jobs[0].prompt_package_sha256,
            "prompt_manifest_sha256": file_sha256(
                prompt_root / "sam_prompt_manifest.jsonl"
            ),
        },
        "output_identity": output_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": dict(provenance),
        "platform_versions": dict(platform_versions),
        "gpu": recorded_gpu,
        "model_load_checkpoints": list(model_load_rows),
        "summary": {key: evidence[key] for key in ("image_count", "scene_count", "mask_count", "empty_mask_count", "median_image_runtime_seconds", "p95_image_runtime_seconds")},
        "reference_prompt_count": route_counts["reference_box"],
        "automatic_prompt_count": route_counts["automatic_box"],
        "repeat_agreement": repeat,
        "completion_gate_passed": gate_passed,
        "error_type": type(run_error).__name__ if run_error is not None else None,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _write_artifact_manifest(private_root)
    _write_artifact_manifest(safe_root)
    return run_record


__all__ = [
    "build_ds1_sam_jobs",
    "import_ds1_sam_resume",
    "run_ds1_sam_workflow",
]
