"""Complete DS0 SAM 2.1 operational workflow for Notebook 05."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
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
    SamResultStore,
    build_sam_jobs,
    collect_sam_evidence,
    compare_repeat_results,
    decode_binary_mask,
    run_sam_jobs,
)


EXPECTED_IMAGE_COUNT = 400
EXPECTED_REFERENCE_PROMPTS = 1640
EXPECTED_AUTOMATIC_PROMPTS = 1616
PILOT_IMAGE_COUNT = 20


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
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
    return rows


def _write_artifact_manifest(root: Path) -> None:
    rows = _artifact_rows(root)
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


def _pilot_jobs(
    jobs: list[SamImageJob], stress_job: SamImageJob
) -> list[SamImageJob]:
    scenes = sorted({job.opaque_scene_id for job in jobs})
    by_key = {job.key: job for job in jobs}
    remaining_scenes = [scene for scene in scenes if scene != stress_job.opaque_scene_id]
    remaining_views = [f"V{index:02d}" for index in range(20) if f"V{index:02d}" != stress_job.view_id]
    selected = [stress_job]
    selected.extend(
        by_key[(scene_id, view_id)]
        for scene_id, view_id in zip(remaining_scenes, remaining_views)
    )
    if len(selected) != PILOT_IMAGE_COUNT:
        raise ValueError("The SAM pilot must represent all 20 scenes and view positions")
    return sorted(selected, key=lambda job: job.key)


def _route_count(rows: list[Mapping[str, Any]], route: str) -> int:
    return sum(len(row["routes"][route]) for row in rows)


def _attempt_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["status"]), int(row["attempt"])) for row in rows)
    return [
        {"status": status, "attempt": attempt, "count": count}
        for (status, attempt), count in sorted(counts.items())
    ]


def _failure_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        str(row["error_type"])
        for row in rows
        if str(row["status"]) == "error"
    )
    if not counts:
        return [{"error_type": "none", "count": 0}]
    return [
        {"error_type": error_type, "count": count}
        for error_type, count in sorted(counts.items())
    ]


def _save_private_mask_grid(
    path: Path,
    rows: list[Mapping[str, Any]],
    job_by_key: Mapping[tuple[str, str], SamImageJob],
    route: str,
) -> None:
    selected = sorted(rows, key=lambda row: (row["opaque_scene_id"], row["view_id"]))[:20]
    cell = 290
    label_height = 34
    columns = 5
    canvas = Image.new("RGB", (columns * cell, 4 * (cell + label_height)), "white")
    palette = [
        (230, 57, 70, 85),
        (29, 114, 184, 85),
        (42, 157, 143, 85),
        (244, 162, 97, 85),
        (155, 93, 229, 85),
        (0, 160, 90, 85),
    ]
    for index, row in enumerate(selected):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        with Image.open(job_by_key[key].image_path) as opened:
            base = opened.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        for mask_index, item in enumerate(row["routes"][route]):
            mask = Image.fromarray(
                (decode_binary_mask(item["mask_rle"]).astype("uint8") * 255), mode="L"
            )
            colour = Image.new("RGBA", base.size, palette[mask_index % len(palette)])
            overlay.alpha_composite(Image.composite(colour, Image.new("RGBA", base.size), mask))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(item["box_xyxy"], outline=palette[mask_index % len(palette)][:3], width=2)
        composed = Image.alpha_composite(base, overlay).convert("RGB").resize((270, 270))
        x = (index % columns) * cell + 10
        y = (index // columns) * (cell + label_height) + 5
        canvas.paste(composed, (x, y))
        ImageDraw.Draw(canvas).text(
            (x, y + 274), f"{key[0]} / {key[1]}", fill="#111111"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _mask_overlay(
    base: Image.Image,
    route_items: list[Mapping[str, Any]],
) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    palette = [
        (230, 57, 70, 92),
        (29, 114, 184, 92),
        (42, 157, 143, 92),
        (244, 162, 97, 92),
        (155, 93, 229, 92),
        (0, 160, 90, 92),
    ]
    draw = ImageDraw.Draw(overlay)
    for mask_index, item in enumerate(route_items):
        colour = palette[mask_index % len(palette)]
        mask = Image.fromarray(
            decode_binary_mask(item["mask_rle"]).astype("uint8") * 255,
            mode="L",
        )
        fill = Image.new("RGBA", base.size, colour)
        overlay.alpha_composite(
            Image.composite(fill, Image.new("RGBA", base.size), mask)
        )
        draw.rectangle(
            item["box_xyxy"],
            outline=colour[:3],
            width=2,
        )
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _evenly_spaced_rows(
    rows: list[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (str(row["opaque_scene_id"]), str(row["view_id"])),
    )
    if count <= 0 or count > len(ordered):
        raise ValueError("Qualitative example count is outside the available rows")
    if count == 1:
        return [ordered[0]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("Qualitative example selection produced duplicate rows")
    return [ordered[index] for index in indices]


def _save_private_route_comparison_grid(
    path: Path,
    manifest_path: Path,
    rows: list[Mapping[str, Any]],
    job_by_key: Mapping[tuple[str, str], SamImageJob],
) -> None:
    selected = _evenly_spaced_rows(rows, 6)
    panel_size = 200
    panel_gap = 8
    group_gap = 20
    groups_per_row = 2
    group_width = 3 * panel_size + 2 * panel_gap
    row_height = panel_size + 72
    header_height = 64
    canvas_width = groups_per_row * group_width + (groups_per_row - 1) * group_gap
    group_rows = (len(selected) + groups_per_row - 1) // groups_per_row
    canvas_height = header_height + group_rows * row_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "SAM 2.1 qualitative route examples",
        fill="#1f2933",
        font=_font(22, bold=True),
    )
    draw.text(
        (8, 36),
        "The same RGB view is shown with reference-box and LocateAnything-box prompts.",
        fill="#52606d",
        font=_font(13),
    )
    manifest_rows = []
    for index, row in enumerate(selected):
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        with Image.open(job_by_key[key].image_path) as opened:
            rgb = opened.convert("RGB")
        panels = (
            ("RGB", rgb),
            ("Reference box", _mask_overlay(rgb, list(row["routes"]["reference_box"]))),
            ("Automatic box", _mask_overlay(rgb, list(row["routes"]["automatic_box"]))),
        )
        group_column = index % groups_per_row
        group_row = index // groups_per_row
        group_x = group_column * (group_width + group_gap)
        image_y = header_height + group_row * row_height + 24
        for panel_index, (label, image) in enumerate(panels):
            x = group_x + panel_index * (panel_size + panel_gap)
            resized = image.resize((panel_size, panel_size), Image.Resampling.BILINEAR)
            canvas.paste(resized, (x, image_y))
            draw.rectangle(
                (x, image_y, x + panel_size - 1, image_y + panel_size - 1),
                outline="#c8d1da",
            )
            draw.text(
                (x, image_y - 20),
                label,
                fill="#1f2933",
                font=_font(12, bold=True),
            )
        draw.text(
            (group_x, image_y + panel_size + 5),
            f"{key[0]} / {key[1]}",
            fill="#1f2933",
            font=_font(12),
        )
        manifest_rows.append(
            {
                "display_order": index + 1,
                "opaque_scene_id": key[0],
                "view_id": key[1],
                "reference_mask_count": len(row["routes"]["reference_box"]),
                "automatic_mask_count": len(row["routes"]["automatic_box"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    write_csv_rows(
        manifest_path,
        [
            "display_order",
            "opaque_scene_id",
            "view_id",
            "reference_mask_count",
            "automatic_mask_count",
        ],
        manifest_rows,
    )


def run_ds0_sam_workflow(
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
    """Run one-image, pilot, repeat and full gates without network tracking."""

    if run_mode not in {"contract_smoke", "model_inference"}:
        raise ValueError("run_mode must be contract_smoke or model_inference")
    if inference_stage not in {"one_image", "pilot", "full"}:
        raise ValueError("inference_stage must be one_image, pilot or full")
    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    safe_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    jobs = build_sam_jobs(prompt_root, media_root)
    job_by_key = {job.key: job for job in jobs}
    prompt_package_sha256 = jobs[0].prompt_package_sha256
    producer_kind = (
        MODEL_PRODUCER_KIND if run_mode == "model_inference" else CONTRACT_PRODUCER_KIND
    )
    result_store = SamResultStore(
        private_root / "image_results",
        configuration.sha256,
        prompt_package_sha256,
        producer_kind,
    )
    result_store.reconcile_jobs(jobs)
    stress_job = max(jobs, key=lambda job: (len(job.prompts), job.key))
    pilot_jobs = _pilot_jobs(jobs, stress_job)
    repeat_store = SamResultStore(
        private_root / "repeat_image_results",
        configuration.sha256,
        prompt_package_sha256,
        producer_kind,
    )
    repeat_store.reconcile_jobs(pilot_jobs)
    required_main_jobs = (
        [stress_job]
        if inference_stage == "one_image"
        else pilot_jobs
        if inference_stage == "pilot"
        else jobs
    )
    required_repeat_jobs = pilot_jobs if inference_stage in {"pilot", "full"} else []
    needs_worker = any(
        job.key not in result_store.completed_keys for job in required_main_jobs
    ) or any(job.key not in repeat_store.completed_keys for job in required_repeat_jobs)
    worker = None
    if needs_worker and run_mode == "model_inference":
        if checkpoint_path is None or checkpoint_sha256 is None:
            raise ValueError("Pending model inference requires a verified SAM checkpoint")
        if checkpoint_sha256 != configuration.checkpoint_sha256:
            raise ValueError("SAM checkpoint hash differs from the pinned configuration")
        worker = Sam2ImageWorker(
            configuration,
            checkpoint_path,
            progress_callback=lambda stage, details: (
                progress_callback(stage, details) if progress_callback else None
            ),
        )
    elif needs_worker:
        worker = ContractSamWorker(configuration)
    elif run_mode == "model_inference" and checkpoint_sha256 != configuration.checkpoint_sha256:
        raise ValueError("Resumed SAM checkpoint identity differs from the pinned configuration")

    attempts_path = private_root / "attempts.jsonl"
    attempts: list[dict[str, Any]] = []
    if attempts_path.is_file():
        attempts = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    valid_keys = {job.key for job in jobs}
    if any(
        (str(row.get("opaque_scene_id")), str(row.get("view_id"))) not in valid_keys
        for row in attempts
    ):
        raise ValueError("Stored SAM attempt history contains an unexpected image")
    started = time.perf_counter()

    def record_attempt(row: Mapping[str, Any]) -> None:
        attempt = dict(row)
        attempts.append(attempt)
        with attempts_path.open("ab") as handle:
            handle.write(canonical_json_bytes(attempt) + b"\n")
            handle.flush()

    def report(completed: int, total: int, row: Mapping[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(
                "inference_progress",
                {
                    "completed": completed,
                    "total": total,
                    "successful_images": len(result_store.completed_keys),
                    "latest_status": row["status"],
                },
            )

    if worker is not None:
        run_sam_jobs(
            worker=worker,
            jobs=[stress_job],
            result_store=result_store,
            maximum_consecutive_failed_jobs=1,
            progress_callback=report,
            attempt_callback=record_attempt,
        )
    if stress_job.key not in result_store.completed_keys:
        raise RuntimeError("The one-image SAM preflight did not complete")

    if inference_stage in {"pilot", "full"}:
        if worker is not None:
            run_sam_jobs(
                worker=worker,
                jobs=pilot_jobs,
                result_store=result_store,
                maximum_consecutive_failed_jobs=1,
                progress_callback=report,
                attempt_callback=record_attempt,
            )

    repeat_summary = {
        "prompt_count": 0,
        "mask_digest_agreement_rate": 0.0,
        "score_agreement_rate": 0.0,
    }
    if inference_stage in {"pilot", "full"}:
        if worker is not None:
            run_sam_jobs(
                worker=worker,
                jobs=pilot_jobs,
                result_store=repeat_store,
                maximum_consecutive_failed_jobs=1,
                progress_callback=report,
                attempt_callback=record_attempt,
            )
        first_pilot_rows = [
            row for row in result_store.rows if (row["opaque_scene_id"], row["view_id"])
            in {job.key for job in pilot_jobs}
        ]
        repeat_summary = compare_repeat_results(first_pilot_rows, repeat_store.rows)

    pilot_keys = {job.key for job in pilot_jobs}
    pilot_rows = [
        row
        for row in result_store.rows
        if (str(row["opaque_scene_id"]), str(row["view_id"])) in pilot_keys
    ]
    preflight_record = {
        "schema_version": 1,
        "one_image_completed": stress_job.key in result_store.completed_keys,
        "pilot_required": inference_stage in {"pilot", "full"},
        "pilot_image_count": len(pilot_rows),
        "pilot_scene_count": len({str(row["opaque_scene_id"]) for row in pilot_rows}),
        "pilot_view_count": len({str(row["view_id"]) for row in pilot_rows}),
        "repeat_prompt_count": repeat_summary["prompt_count"],
        "repeat_mask_digest_agreement_rate": repeat_summary[
            "mask_digest_agreement_rate"
        ],
        "repeat_score_agreement_rate": repeat_summary["score_agreement_rate"],
    }
    preflight_passed = preflight_record["one_image_completed"] and (
        inference_stage == "one_image"
        or (
            preflight_record["pilot_image_count"] == PILOT_IMAGE_COUNT
            and preflight_record["pilot_scene_count"] == PILOT_IMAGE_COUNT
            and preflight_record["pilot_view_count"] == PILOT_IMAGE_COUNT
            and repeat_summary["prompt_count"]
            == sum(len(job.prompts) for job in pilot_jobs)
            and repeat_summary["mask_digest_agreement_rate"] == 1.0
            and repeat_summary["score_agreement_rate"] == 1.0
        )
    )
    preflight_record["passed"] = preflight_passed
    _write_json(safe_root / "preflight_gates.json", preflight_record)
    if not preflight_passed:
        raise RuntimeError("SAM preflight gates failed before full inference")

    run_error: Exception | None = None
    if inference_stage == "full":
        if worker is not None:
            try:
                run_sam_jobs(
                    worker=worker,
                    jobs=jobs,
                    result_store=result_store,
                    maximum_consecutive_failed_jobs=2,
                    progress_callback=report,
                    attempt_callback=record_attempt,
                )
            except Exception as error:
                run_error = error

    result_store.write_index(private_root / "mask_results.jsonl")
    if repeat_store.rows:
        repeat_store.write_index(private_root / "repeat_mask_results.jsonl")
    rows = result_store.rows
    evidence = collect_sam_evidence(rows)
    reference_prompt_count = _route_count(rows, "reference_box")
    automatic_prompt_count = _route_count(rows, "automatic_box")
    expected_images = {
        "one_image": 1,
        "pilot": PILOT_IMAGE_COUNT,
        "full": EXPECTED_IMAGE_COUNT,
    }[inference_stage]
    expected_reference = (
        EXPECTED_REFERENCE_PROMPTS
        if inference_stage == "full"
        else sum(
            len([prompt for prompt in job.prompts if prompt.route == "reference_box"])
            for job in ({stress_job} if inference_stage == "one_image" else set(pilot_jobs))
        )
    )
    expected_automatic = (
        EXPECTED_AUTOMATIC_PROMPTS
        if inference_stage == "full"
        else sum(
            len([prompt for prompt in job.prompts if prompt.route == "automatic_box"])
            for job in ({stress_job} if inference_stage == "one_image" else set(pilot_jobs))
        )
    )
    expected_jobs = (
        [stress_job]
        if inference_stage == "one_image"
        else pilot_jobs
        if inference_stage == "pilot"
        else jobs
    )
    unresolved_keys = {
        job.key for job in expected_jobs
    } - result_store.completed_keys
    gate_rows = [
        {
            "gate": "image coverage",
            "observed": evidence["image_count"],
            "expected": expected_images,
            "status": "pass" if evidence["image_count"] == expected_images else "fail",
        },
        {
            "gate": "reference-route masks",
            "observed": reference_prompt_count,
            "expected": expected_reference,
            "status": "pass" if reference_prompt_count == expected_reference else "fail",
        },
        {
            "gate": "automatic-route masks",
            "observed": automatic_prompt_count,
            "expected": expected_automatic,
            "status": "pass" if automatic_prompt_count == expected_automatic else "fail",
        },
        {
            "gate": "unresolved failures",
            "observed": len(unresolved_keys),
            "expected": 0,
            "status": "pass" if not unresolved_keys else "fail",
        },
        {
            "gate": "repeat mask digest agreement",
            "observed": repeat_summary["mask_digest_agreement_rate"],
            "expected": 1.0 if inference_stage in {"pilot", "full"} else "not assessed",
            "status": "pass"
            if inference_stage == "one_image"
            or repeat_summary["mask_digest_agreement_rate"] == 1.0
            else "fail",
        },
        {
            "gate": "repeat score agreement",
            "observed": repeat_summary["score_agreement_rate"],
            "expected": 1.0 if inference_stage in {"pilot", "full"} else "not assessed",
            "status": "pass"
            if inference_stage == "one_image"
            or repeat_summary["score_agreement_rate"] == 1.0
            else "fail",
        },
    ]
    gate_passed = all(row["status"] == "pass" for row in gate_rows)

    write_csv_rows(
        safe_root / "completion_gates.csv",
        ["gate", "observed", "expected", "status"],
        gate_rows,
    )
    write_csv_rows(
        safe_root / "route_summary.csv",
        [
            "route",
            "mask_count",
            "empty_mask_count",
            "empty_mask_rate",
            "mean_area_fraction",
            "median_area_fraction",
            "p95_area_fraction",
            "mean_predicted_quality_score",
            "p10_predicted_quality_score",
        ],
        evidence["route_rows"],
    )
    write_csv_rows(
        safe_root / "view_summary.csv",
        [
            "view_id",
            "route",
            "mask_count",
            "empty_mask_count",
            "mean_area_fraction",
            "mean_predicted_quality_score",
        ],
        evidence["view_rows"],
    )
    write_csv_rows(
        safe_root / "attempt_summary.csv",
        ["status", "attempt", "count"],
        _attempt_summary(attempts),
    )
    write_csv_rows(
        safe_root / "failure_summary.csv",
        ["error_type", "count"],
        _failure_summary(attempts),
    )
    _write_json(safe_root / "repeat_agreement.json", repeat_summary)
    _write_json(
        safe_root / "runtime_summary.json",
        {
            "workflow_seconds": round(time.perf_counter() - started, 4),
            "median_image_runtime_seconds": evidence["median_image_runtime_seconds"],
            "p95_image_runtime_seconds": evidence["p95_image_runtime_seconds"],
            "median_prompts_per_image": evidence["median_prompts_per_image"],
        },
    )
    _write_json(
        safe_root / "mask_geometry_summary.json",
        {"routes": evidence["route_rows"]},
    )

    runtime_chart = safe_root / "runtime_distribution.png"
    prompt_chart = safe_root / "mask_count_distribution.png"
    runtime_scatter = safe_root / "runtime_by_prompt_count.png"
    save_histogram(
        evidence["image_runtimes"],
        runtime_chart,
        title="SAM 2.1 runtime distribution",
        x_label="Seconds per image",
        bins=12,
    )
    save_histogram(
        evidence["prompt_counts"],
        prompt_chart,
        title="SAM 2.1 mask count per image",
        x_label="Masks per image",
        bins=10,
    )
    save_scatter_plot(
        evidence["prompt_counts"],
        evidence["image_runtimes"],
        runtime_scatter,
        title="SAM 2.1 runtime by prompt count",
        x_label="Prompts per image",
        y_label="Seconds per image",
    )
    for route, values in evidence["area_values_by_route"].items():
        save_histogram(
            values,
            safe_root / f"{route}_mask_area_distribution.png",
            title=f"{route.replace('_', ' ').title()} mask area distribution",
            x_label="Mask area as a fraction of image area",
            bins=12,
        )
    for route, values in evidence["score_values_by_route"].items():
        save_histogram(
            values,
            safe_root / f"{route}_predicted_score_distribution.png",
            title=f"{route.replace('_', ' ').title()} predicted score distribution",
            x_label="SAM predicted quality score, not ground-truth quality",
            bins=12,
        )
    save_bar_chart(
        [row["route"] for row in evidence["route_rows"]],
        [row["mask_count"] for row in evidence["route_rows"]],
        safe_root / "mask_count_by_route.png",
        title="Completed SAM 2.1 masks by route",
        y_label="Masks",
    )

    pilot_rows = [
        row
        for row in rows
        if (str(row["opaque_scene_id"]), str(row["view_id"]))
        in {job.key for job in pilot_jobs}
    ]
    if pilot_rows:
        _save_private_mask_grid(
            private_root / "reference_mask_overlays.png",
            pilot_rows,
            job_by_key,
            "reference_box",
        )
        _save_private_mask_grid(
            private_root / "automatic_mask_overlays.png",
            pilot_rows,
            job_by_key,
            "automatic_box",
        )
        _save_private_route_comparison_grid(
            private_root / "mask_route_examples.png",
            private_root / "qualitative_example_manifest.csv",
            pilot_rows,
            job_by_key,
        )

    recorded_gpu = dict(gpu_evidence)
    if isinstance(worker, Sam2ImageWorker):
        recorded_gpu.update(
            {
                "peak_allocated_gib": round(
                    worker.torch.cuda.max_memory_allocated() / (1024**3), 4
                ),
                "peak_reserved_gib": round(
                    worker.torch.cuda.max_memory_reserved() / (1024**3), 4
                ),
            }
        )
    run_record = {
        "schema_version": 1,
        "stage": "ds0_sam21_prompted_masks",
        "status": "completed" if gate_passed else "failed",
        "tracking_status": "pending",
        "run_mode": run_mode,
        "inference_stage": inference_stage,
        "producer_kind": producer_kind,
        "python_version": platform.python_version(),
        "configuration": configuration.to_dict(),
        "configuration_sha256": configuration.sha256,
        "prompt_package_sha256": prompt_package_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance": dict(provenance),
        "platform_versions": dict(platform_versions),
        "gpu": recorded_gpu,
        "model_load_checkpoints": list(model_load_rows),
        "summary": {
            key: evidence[key]
            for key in (
                "image_count",
                "scene_count",
                "mask_count",
                "empty_mask_count",
                "median_image_runtime_seconds",
                "p95_image_runtime_seconds",
            )
        },
        "reference_prompt_count": reference_prompt_count,
        "automatic_prompt_count": automatic_prompt_count,
        "repeat_agreement": repeat_summary,
        "completion_gate_passed": gate_passed,
        "error_type": type(run_error).__name__ if run_error is not None else None,
    }
    _write_json(safe_root / "run_record.json", run_record)
    _write_artifact_manifest(private_root)
    _write_artifact_manifest(safe_root)
    return run_record


__all__ = ["run_ds0_sam_workflow"]
