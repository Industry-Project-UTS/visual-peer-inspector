"""Reference-box and automatic-box SAM 2.1 inference for DS2 confirmation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from visual_peer_inspector.data.cohort_contract import DS2_CONFIRM_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)
from visual_peer_inspector.evaluation.confirmation_reference import (
    verify_ds2_geometry_package,
)
from visual_peer_inspector.evidence import save_bar_chart, save_histogram
from visual_peer_inspector.models.sam2 import (
    ContractSamWorker,
    Sam2Configuration,
    SamImageJob,
    SamPrompt,
    binary_mask_sha256,
    compare_repeat_results,
    decode_binary_mask,
    run_sam_jobs,
)


REFERENCE_STAGE = "ds2_reference_box_sam21"
AUTOMATIC_STAGE = "ds2_automatic_box_sam21"
REFERENCE_ROUTE = "reference_box"
AUTOMATIC_ROUTE = "automatic_box"
ROUTE = REFERENCE_ROUTE
PILOT_IMAGE_COUNT = 20
AUTOMATIC_PROVENANCE_FIELDS = (
    "source_status",
    "source_error_type",
    "source_attempt",
    "carried_as_zero_proposal",
    "raw_proposal_count",
    "removed_proposal_count",
    "discarded_untrusted_box_count",
    "proposal_count",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path.name}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("DS2 outputs cannot contain symbolic links")
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


def build_ds2_sam_jobs(rgb_root: Path, geometry_root: Path) -> list[SamImageJob]:
    """Build one reference-box prompt for each visible object in every DS2 image."""

    rgb_root = rgb_root.expanduser().resolve()
    geometry_root = geometry_root.expanduser().resolve()
    verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    verify_ds2_geometry_package(geometry_root)
    prompt_digest = file_sha256(geometry_root / "package_record.json")
    geometry: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in read_jsonl(geometry_root / "reference_manifest.jsonl"):
        if bool(row["visible"]):
            geometry[(str(row["opaque_scene_id"]), str(row["view_id"]))].append(row)

    jobs: list[SamImageJob] = []
    for scene in read_jsonl(rgb_root / "neutral_corpus_manifest.jsonl"):
        scene_id = str(scene["opaque_scene_id"])
        for view_id in map(str, scene["view_ids"]):
            image_path = rgb_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            prompts = tuple(
                SamPrompt(
                    route=REFERENCE_ROUTE,
                    prompt_key=str(row["opaque_object_id"]),
                    box_xyxy=tuple(int(value) for value in row["box_xyxy"]),
                )
                for row in sorted(
                    geometry[(scene_id, view_id)],
                    key=lambda value: str(value["opaque_object_id"]),
                )
            )
            if len(prompts) < 3:
                raise ValueError("Every DS2 image requires at least three visible peer objects")
            jobs.append(
                SamImageJob(
                    opaque_scene_id=scene_id,
                    view_id=view_id,
                    image_path=image_path,
                    image_sha256=file_sha256(image_path),
                    prompts=prompts,
                    prompt_package_sha256=prompt_digest,
                    cohort_id=DS2_CONFIRM_CONTRACT.cohort_id,
                )
            )
    if len(jobs) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 SAM jobs must cover all 6,000 images")
    return sorted(jobs, key=lambda job: job.key)


def _automatic_prompt_key(
    scene_id: str,
    view_id: str,
    index: int,
    box_xyxy: Sequence[int],
) -> str:
    payload = f"{scene_id}|{view_id}|{index}|{','.join(map(str, box_xyxy))}".encode("utf-8")
    return "P" + sha256(payload).hexdigest()[:20]


def build_ds2_automatic_sam_jobs(
    rgb_root: Path,
    proposal_root: Path,
) -> list[SamImageJob]:
    """Build SAM jobs from frozen DS2 proposals without opening evaluator geometry."""

    rgb_root = rgb_root.expanduser().resolve()
    proposal_root = proposal_root.expanduser().resolve()
    verify_model_input_package(rgb_root, contract=DS2_CONFIRM_CONTRACT)
    proposal_path = proposal_root / "automatic_proposals.jsonl"
    package_path = proposal_root / "package_record.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if (
        package.get("package_kind") != "ds2-frozen-automatic-proposals"
        or package.get("cohort_id") != DS2_CONFIRM_CONTRACT.cohort_id
        or package.get("image_count") != DS2_CONFIRM_CONTRACT.image_count
        or package.get("evaluator_opened") is not False
        or package.get("protected_fields_included") is not False
        or package.get("proposal_manifest_sha256") != file_sha256(proposal_path)
    ):
        raise ValueError("DS2 automatic proposal package is not a frozen label-free input")
    proposals = {
        (str(row["opaque_scene_id"]), str(row["view_id"])): row
        for row in read_jsonl(proposal_path)
    }
    if len(proposals) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 automatic proposals do not cover every image")

    jobs: list[SamImageJob] = []
    for scene in read_jsonl(rgb_root / "neutral_corpus_manifest.jsonl"):
        scene_id = str(scene["opaque_scene_id"])
        for view_id in map(str, scene["view_ids"]):
            row = proposals.get((scene_id, view_id))
            if row is None:
                raise ValueError("DS2 automatic proposal image is missing")
            boxes = [[int(value) for value in box] for box in row["boxes_xyxy"]]
            if int(row["proposal_count"]) != len(boxes):
                raise ValueError("DS2 automatic proposal count differs from its boxes")
            prompts = []
            for index, box in enumerate(boxes):
                if len(box) != 4:
                    raise ValueError("DS2 automatic proposal box must contain four values")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
                    raise ValueError("DS2 automatic proposal box is outside the image")
                prompts.append(
                    SamPrompt(
                        route=AUTOMATIC_ROUTE,
                        prompt_key=_automatic_prompt_key(scene_id, view_id, index, box),
                        box_xyxy=tuple(box),
                    )
                )
            image_path = rgb_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            image_digest = file_sha256(image_path)
            if image_digest != str(row["image_sha256"]):
                raise ValueError("DS2 proposal and RGB package identify different images")
            jobs.append(
                SamImageJob(
                    opaque_scene_id=scene_id,
                    view_id=view_id,
                    image_path=image_path,
                    image_sha256=image_digest,
                    prompts=tuple(prompts),
                    prompt_package_sha256=file_sha256(package_path),
                    cohort_id=DS2_CONFIRM_CONTRACT.cohort_id,
                )
            )
    if len(jobs) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 automatic SAM jobs must cover all confirmation images")
    return sorted(jobs, key=lambda job: job.key)


def _automatic_proposal_provenance(
    proposal_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the failure disposition needed by the protected evaluator."""

    rows = read_jsonl(proposal_root / "automatic_proposals.jsonl")
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["opaque_scene_id"]), str(row["view_id"]))
        if key in provenance:
            raise ValueError("DS2 automatic proposal provenance contains duplicate images")
        value = {field: row.get(field) for field in AUTOMATIC_PROVENANCE_FIELDS}
        status = value["source_status"]
        if status not in {"ok", "error"}:
            raise ValueError("DS2 automatic proposal provenance has an invalid status")
        raw_count = int(value["raw_proposal_count"])
        removed_count = int(value["removed_proposal_count"])
        discarded_count = int(value["discarded_untrusted_box_count"])
        proposal_count = int(value["proposal_count"])
        if min(raw_count, removed_count, discarded_count, proposal_count) < 0:
            raise ValueError("DS2 automatic proposal provenance has a negative count")
        if raw_count != removed_count + discarded_count + proposal_count:
            raise ValueError("DS2 automatic proposal provenance accounting differs")
        carried = value["carried_as_zero_proposal"]
        if not isinstance(carried, bool) or carried != (status == "error"):
            raise ValueError("DS2 automatic proposal failure disposition differs")
        if status == "error" and (
            proposal_count != 0
            or removed_count != 0
            or discarded_count != raw_count
            or not value["source_error_type"]
        ):
            raise ValueError("DS2 terminal proposal failure was not carried as zero")
        if status == "ok" and (
            discarded_count != 0 or value["source_error_type"] is not None
        ):
            raise ValueError("Successful DS2 proposal provenance contains failure detail")
        value["source_attempt"] = int(value["source_attempt"])
        if value["source_attempt"] < 1:
            raise ValueError("DS2 automatic proposal attempt must be positive")
        value["raw_proposal_count"] = raw_count
        value["removed_proposal_count"] = removed_count
        value["discarded_untrusted_box_count"] = discarded_count
        value["proposal_count"] = proposal_count
        provenance[key] = value
    if len(provenance) != DS2_CONFIRM_CONTRACT.image_count:
        raise ValueError("DS2 automatic proposal provenance coverage is incomplete")
    return provenance


class ConfirmationSamStore:
    """Atomic DS2 SAM result storage with exact input reconciliation."""

    def __init__(
        self,
        root: Path,
        *,
        configuration_sha256: str,
        prompt_package_sha256: str,
        producer_kind: str,
        route: str = REFERENCE_ROUTE,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.configuration_sha256 = configuration_sha256
        self.prompt_package_sha256 = prompt_package_sha256
        self.producer_kind = producer_kind
        if route not in {REFERENCE_ROUTE, AUTOMATIC_ROUTE}:
            raise ValueError("DS2 SAM result store route is unsupported")
        self.route = route
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        for path in sorted(self.root.glob("S*/V*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            self._validate(row)
            key = (str(row["opaque_scene_id"]), str(row["view_id"]))
            if key in self._rows:
                raise ValueError("Stored DS2 SAM results contain duplicate images")
            self._rows[key] = row

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [self._rows[key] for key in sorted(self._rows)]

    @property
    def completed_keys(self) -> set[tuple[str, str]]:
        return set(self._rows)

    def _validate(self, row: Mapping[str, Any]) -> None:
        if (
            row.get("cohort_id") != DS2_CONFIRM_CONTRACT.cohort_id
            or row.get("configuration_sha256") != self.configuration_sha256
            or row.get("prompt_package_sha256") != self.prompt_package_sha256
            or row.get("producer_kind") != self.producer_kind
            or row.get("status") != "ok"
            or set(row.get("routes", {})) != {self.route}
        ):
            raise ValueError("Stored DS2 SAM result identity differs from this run")
        if int(row.get("attempt", 0)) < 1:
            raise ValueError("Stored DS2 SAM attempt is invalid")
        for name in ("embedding_runtime_seconds", "prompt_runtime_seconds", "total_runtime_seconds"):
            if not math.isfinite(float(row.get(name, -1))) or float(row[name]) < 0:
                raise ValueError("Stored DS2 SAM runtime is invalid")
        if int(row.get("prompt_count", -1)) != len(row["routes"][self.route]):
            raise ValueError("Stored DS2 SAM prompt count differs from its route")
        for item in row["routes"][self.route]:
            mask = decode_binary_mask(item["mask_rle"])
            if binary_mask_sha256(mask) != str(item["mask_sha256"]):
                raise ValueError("Stored DS2 SAM mask digest differs")
            if int(mask.sum()) != int(item["foreground_pixels"]):
                raise ValueError("Stored DS2 SAM mask area differs")

    def reconcile_jobs(self, jobs: Iterable[SamImageJob]) -> None:
        job_list = list(jobs)
        jobs_by_key = {job.key: job for job in job_list}
        if len(jobs_by_key) != len(job_list):
            raise ValueError("DS2 SAM reconciliation received duplicate image jobs")
        if set(self._rows) - set(jobs_by_key):
            raise ValueError("Stored DS2 SAM results contain unexpected images")
        for key, row in self._rows.items():
            job = jobs_by_key[key]
            expected = Counter((prompt.prompt_key, prompt.box_xyxy) for prompt in job.prompts)
            observed = Counter(
                (str(item["prompt_key"]), tuple(item["box_xyxy"]))
                for item in row["routes"][self.route]
            )
            if row["image_sha256"] != job.image_sha256 or expected != observed:
                raise ValueError("Stored DS2 SAM result differs from its image job")

    def write(self, job: SamImageJob, row: Mapping[str, Any]) -> Path:
        self._validate(row)
        if (str(row["opaque_scene_id"]), str(row["view_id"])) != job.key:
            raise ValueError("DS2 SAM result identifies another image")
        expected = Counter((prompt.prompt_key, prompt.box_xyxy) for prompt in job.prompts)
        observed = Counter(
            (str(item["prompt_key"]), tuple(item["box_xyxy"]))
            for item in row["routes"][self.route]
        )
        if expected != observed or str(row["image_sha256"]) != job.image_sha256:
            raise ValueError("DS2 SAM result does not match its prompt job")
        path = self.root / job.opaque_scene_id / f"{job.view_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(dict(row)) + b"\n"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("A completed DS2 SAM result would change")
        if not path.exists():
            temporary = path.with_suffix(".json.incomplete")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        self._rows[job.key] = dict(row)
        return path


def _pilot_jobs(jobs: Sequence[SamImageJob]) -> list[SamImageJob]:
    candidates = [job for job in jobs if job.prompts]
    if not candidates:
        return []
    count = min(PILOT_IMAGE_COUNT, len(candidates))
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    selected = [candidates[int(index)] for index in indices]
    if len(selected) != count:
        raise ValueError("DS2 SAM pilot selection is incomplete")
    return selected


def _mask_overlay_panel(
    jobs: Sequence[SamImageJob],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    route: str,
) -> None:
    rows_by_key = {(str(row["opaque_scene_id"]), str(row["view_id"])): row for row in rows}
    requested_views = ("V00", "V04", "V08", "V12", "V16", "V19")
    scene_ids = sorted({job.opaque_scene_id for job in jobs})
    requested_scenes = [scene_ids[index] for index in np.linspace(0, len(scene_ids) - 1, 6, dtype=int)]
    jobs_by_key = {job.key: job for job in jobs}
    selected = [
        jobs_by_key[(scene_id, view_id)]
        for scene_id, view_id in zip(requested_scenes, requested_views, strict=True)
    ]
    panel = Image.new("RGB", (256 * 3, 300 * 2), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(panel)
    for index, job in enumerate(selected):
        image = Image.open(job.image_path).convert("RGB")
        row = rows_by_key[job.key]
        array = np.asarray(image).copy()
        scores = []
        for item in row["routes"][route]:
            mask = decode_binary_mask(item["mask_rle"])
            edge = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.FIND_EDGES)
            ) > 0
            array[edge] = np.asarray((225, 45, 65), dtype=np.uint8)
            scores.append(float(item["predicted_quality_score"]))
        x = (index % 3) * 256
        y = (index // 3) * 292
        panel.paste(Image.fromarray(array), (x, y))
        draw.text(
            (x + 5, y + 260),
            (
                f"{job.view_id}  {len(scores)} masks  mean score {np.mean(scores):.3f}"
                if scores
                else f"{job.view_id}  no automatic proposals"
            ),
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def _empty_sam_result(
    job: SamImageJob,
    *,
    configuration: Sam2Configuration,
    producer_kind: str,
    route: str,
) -> dict[str, Any]:
    """Record an image with no automatic proposals without invoking SAM."""

    if job.prompts:
        raise ValueError("Only an empty SAM job can use the empty result path")
    return {
        "schema_version": 1,
        "producer_kind": producer_kind,
        "cohort_id": job.cohort_id,
        "opaque_scene_id": job.opaque_scene_id,
        "view_id": job.view_id,
        "image_sha256": job.image_sha256,
        "prompt_package_sha256": job.prompt_package_sha256,
        "configuration_sha256": configuration.sha256,
        "attempt": 1,
        "status": "ok",
        "embedding_runtime_seconds": 0.0,
        "prompt_runtime_seconds": 0.0,
        "total_runtime_seconds": 0.0,
        "prompt_count": 0,
        "routes": {route: []},
    }


def _run_ds2_sam_route(
    *,
    jobs: Sequence[SamImageJob],
    output_root: Path,
    worker: Any | None,
    run_mode: str,
    route: str,
    stage: str,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None,
    image_provenance: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one frozen SAM route with resumable per-image storage."""

    if run_mode not in {"contract", "model_inference"}:
        raise ValueError("DS2 SAM run mode must be contract or model_inference")
    output_root = output_root.expanduser().resolve()
    safe_root = output_root / "safe"
    private_root = output_root / "private"
    configuration = worker.configuration if worker is not None else Sam2Configuration(device="cpu", precision="float32")
    if worker is None:
        if run_mode != "contract":
            raise ValueError("Model inference requires an explicit pinned SAM worker")
        worker = ContractSamWorker(configuration)
    if not jobs:
        raise ValueError("DS2 SAM route requires image jobs")
    prompt_digest = jobs[0].prompt_package_sha256
    if {job.prompt_package_sha256 for job in jobs} != {prompt_digest}:
        raise ValueError("DS2 SAM jobs contain multiple prompt package identities")
    producer_kind = str(worker.producer_kind)

    main_store = ConfirmationSamStore(
        private_root / "checkpoints",
        configuration_sha256=configuration.sha256,
        prompt_package_sha256=prompt_digest,
        producer_kind=producer_kind,
        route=route,
    )
    main_store.reconcile_jobs(jobs)
    for job in jobs:
        if not job.prompts and job.key not in main_store.completed_keys:
            main_store.write(
                job,
                _empty_sam_result(
                    job,
                    configuration=configuration,
                    producer_kind=producer_kind,
                    route=route,
                ),
            )

    pilot = _pilot_jobs(jobs)
    pilot_root = output_root / ".pilot"
    repeat_root = output_root / ".pilot-repeat"
    for transient in (pilot_root, repeat_root):
        if transient.exists():
            shutil.rmtree(transient)
    repeat = {
        "pilot_image_count": 0,
        "repeat_image_count": 0,
        "prompt_count": 0,
        "matching_mask_digest_count": 0,
        "mask_digest_agreement_rate": 1.0,
        "matching_score_count": 0,
        "score_agreement_rate": 1.0,
        "score_tolerance": 1e-5,
    }
    if pilot:
        pilot_store = ConfirmationSamStore(
            pilot_root,
            configuration_sha256=configuration.sha256,
            prompt_package_sha256=prompt_digest,
            producer_kind=producer_kind,
            route=route,
        )
        run_sam_jobs(worker=worker, jobs=pilot, result_store=pilot_store)
        repeat_store = ConfirmationSamStore(
            repeat_root,
            configuration_sha256=configuration.sha256,
            prompt_package_sha256=prompt_digest,
            producer_kind=producer_kind,
            route=route,
        )
        run_sam_jobs(worker=worker, jobs=pilot, result_store=repeat_store)
        repeat = compare_repeat_results(pilot_store.rows, repeat_store.rows)
        repeat["pilot_image_count"] = len(pilot_store.rows)
        repeat["repeat_image_count"] = len(repeat_store.rows)

    started = time.perf_counter()
    attempts = run_sam_jobs(
        worker=worker,
        jobs=jobs,
        result_store=main_store,
        progress_callback=progress_callback,
    )
    main_store.reconcile_jobs(jobs)
    rows = main_store.rows
    if len(rows) != DS2_CONFIRM_CONTRACT.image_count:
        raise RuntimeError("DS2 SAM did not complete every image")
    if image_provenance is not None:
        job_keys = {job.key for job in jobs}
        if route != AUTOMATIC_ROUTE or set(image_provenance) != job_keys:
            raise ValueError("DS2 automatic SAM provenance coverage differs")
        rows = [
            {
                **row,
                "automatic_proposal_provenance": dict(
                    image_provenance[(str(row["opaque_scene_id"]), str(row["view_id"]))]
                ),
            }
            for row in rows
        ]
    for transient in (pilot_root, repeat_root):
        if transient.exists():
            shutil.rmtree(transient)

    results_path = private_root / "sam_results.jsonl"
    _write_jsonl(results_path, rows)
    runtimes = [float(row["total_runtime_seconds"]) for row in rows]
    prompt_counts = [int(row["prompt_count"]) for row in rows]
    route_items = [item for row in rows for item in row["routes"][route]]
    quality_values = [float(item["predicted_quality_score"]) for item in route_items]
    route_summary = [
        {
            "route": route,
            "image_count": len(rows),
            "mask_count": len(route_items),
            "zero_prompt_image_count": sum(count == 0 for count in prompt_counts),
            "empty_mask_count": sum(int(item["foreground_pixels"]) == 0 for item in route_items),
            "mean_predicted_quality_score": float(np.mean(quality_values)) if quality_values else None,
            "median_runtime_seconds": float(np.median(runtimes)),
            "p95_runtime_seconds": float(np.quantile(runtimes, 0.95)),
        }
    ]
    if image_provenance is not None:
        route_summary[0].update(
            {
                "terminal_failed_image_count": sum(
                    row["source_status"] == "error"
                    for row in image_provenance.values()
                ),
                "genuine_zero_proposal_image_count": sum(
                    row["source_status"] == "ok" and int(row["proposal_count"]) == 0
                    for row in image_provenance.values()
                ),
                "discarded_untrusted_box_count": sum(
                    int(row["discarded_untrusted_box_count"])
                    for row in image_provenance.values()
                ),
            }
        )
    checks = [
        {"check": "all DS2 images recorded", "passed": len(rows) == DS2_CONFIRM_CONTRACT.image_count},
        {"check": "stored prompts match the frozen input", "passed": True},
        {
            "check": "pilot repeat completed and was measured",
            "passed": repeat["pilot_image_count"] == len(pilot) and repeat["repeat_image_count"] == len(pilot),
        },
        {"check": "no unresolved SAM image failures", "passed": not any(row["status"] == "error" for row in attempts)},
    ]
    _write_csv(safe_root / "route_summary.csv", route_summary)
    _write_csv(safe_root / "completion_checks.csv", checks)
    _write_json(safe_root / "repeat_agreement.json", repeat)
    _write_json(safe_root / "resolved_configuration.json", configuration.to_dict())
    save_histogram(runtimes, safe_root / "runtime_distribution.png", title="DS2 SAM 2.1 runtime", x_label="Seconds per image", bins=24)
    save_histogram(prompt_counts, safe_root / "prompt_count_distribution.png", title="DS2 prompts per image", x_label="Prompts per image", bins=8)
    save_bar_chart(["completed", "failed"], [len(rows), 0], safe_root / "completion_summary.png", title="DS2 SAM completion", y_label="Images", show_values=True)
    _mask_overlay_panel(
        jobs,
        rows,
        private_root / "figures" / f"{route}_mask_examples.png",
        route=route,
    )
    run_record = {
        "schema_version": 1,
        "stage": stage,
        "status": "completed",
        "run_mode": run_mode,
        "producer_kind": producer_kind,
        "route": route,
        "cohort_id": DS2_CONFIRM_CONTRACT.cohort_id,
        "image_count": len(rows),
        "mask_count": len(route_items),
        "zero_prompt_image_count": route_summary[0]["zero_prompt_image_count"],
        "configuration_sha256": configuration.sha256,
        "prompt_package_sha256": prompt_digest,
        "sam_results_sha256": file_sha256(results_path),
        "runtime_seconds_this_session": time.perf_counter() - started,
        "resumed_image_count": DS2_CONFIRM_CONTRACT.image_count - len(attempts),
        "all_completion_checks_passed": all(bool(row["passed"]) for row in checks),
    }
    if image_provenance is not None:
        run_record.update(
            {
                "terminal_failed_image_count": route_summary[0][
                    "terminal_failed_image_count"
                ],
                "genuine_zero_proposal_image_count": route_summary[0][
                    "genuine_zero_proposal_image_count"
                ],
                "discarded_untrusted_box_count": route_summary[0][
                    "discarded_untrusted_box_count"
                ],
                "per_image_proposal_provenance_included": True,
            }
        )
    _write_json(safe_root / "run_record.json", run_record)
    _artifact_manifest(safe_root)
    _artifact_manifest(private_root)
    return run_record


def run_ds2_reference_sam(
    *,
    rgb_root: Path,
    geometry_root: Path,
    output_root: Path,
    worker: Any | None = None,
    run_mode: str = "contract",
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the supplied-reference box route as a segmentation upper bound."""

    jobs = build_ds2_sam_jobs(rgb_root, geometry_root)
    return _run_ds2_sam_route(
        jobs=jobs,
        output_root=output_root,
        worker=worker,
        run_mode=run_mode,
        route=REFERENCE_ROUTE,
        stage=REFERENCE_STAGE,
        progress_callback=progress_callback,
    )


def run_ds2_automatic_sam(
    *,
    rgb_root: Path,
    proposal_root: Path,
    output_root: Path,
    worker: Any | None = None,
    run_mode: str = "contract",
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run SAM from frozen LocateAnything boxes without evaluator information."""

    jobs = build_ds2_automatic_sam_jobs(rgb_root, proposal_root)
    provenance = _automatic_proposal_provenance(proposal_root.expanduser().resolve())
    return _run_ds2_sam_route(
        jobs=jobs,
        output_root=output_root,
        worker=worker,
        run_mode=run_mode,
        route=AUTOMATIC_ROUTE,
        stage=AUTOMATIC_STAGE,
        progress_callback=progress_callback,
        image_provenance=provenance,
    )


__all__ = [
    "AUTOMATIC_ROUTE",
    "ConfirmationSamStore",
    "REFERENCE_ROUTE",
    "build_ds2_automatic_sam_jobs",
    "build_ds2_sam_jobs",
    "run_ds2_automatic_sam",
    "run_ds2_reference_sam",
]
