"""DS1 LocateAnything proposal generation and operational evidence.

The workflow keeps raw responses and per-image records in ``private``. Only
aggregate, allowlisted evidence is written to ``safe`` for GitHub and ClearML.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import (
    canonical_json_bytes,
    file_sha256,
    membership_sha256,
    verify_model_input_package,
)
from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_histogram,
    save_image_grid,
    save_scatter_plot,
    write_csv_rows,
)
from visual_peer_inspector.models.locateanything import (
    CONTRACT_PRODUCER_KIND,
    MODEL_PRODUCER_KIND,
    LocateAnythingConfiguration,
    ProposalJob,
    ProposalResultStore,
    collect_proposal_evidence,
    repeat_agreement,
    run_proposal_jobs,
    summarize_proposal_rows,
)


RUN_MODES = frozenset({"model_inference", "contract_smoke"})
INFERENCE_STAGES = frozenset({"one_image", "pilot", "full"})
PILOT_IMAGE_COUNT = 20
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FILES = (
    "one_image_results.jsonl",
    "pilot_results.jsonl",
    "pilot_repeat_results.jsonl",
    "proposal_results.jsonl",
)

SAFE_TABLES = frozenset(
    {
        "attempt_summary.csv",
        "completion_gates.csv",
        "failure_summary.csv",
        "progress_checkpoints.csv",
        "proposal_summary_by_view.csv",
    }
)
SAFE_JSON = frozenset(
    {
        "artifact_manifest.json",
        "box_geometry_summary.json",
        "environment_summary.json",
        "input_audit.json",
        "model_load_summary.json",
        "one_image_preflight.json",
        "pilot_summary.json",
        "proposal_summary.json",
        "repeat_agreement.json",
        "resolved_configuration.json",
        "run_record.json",
        "runtime_summary.json",
    }
)
SAFE_CHARTS = frozenset(
    {
        "box_area_fraction_distribution.png",
        "box_aspect_ratio_distribution.png",
        "mean_proposals_by_view.png",
        "proposal_count_distribution.png",
        "runtime_by_proposal_count.png",
        "runtime_distribution.png",
    }
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def write_artifact_manifest(root: Path) -> dict[str, Any]:
    """Write the checksum manifest after every other safe artifact exists."""

    rows = _artifact_rows(root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(rows),
        "artifacts": rows,
        "manifest_content_sha256": sha256(
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        ).hexdigest(),
    }
    _write_json(root / "artifact_manifest.json", manifest)
    return manifest


def verify_ds1_jobs(jobs: Sequence[ProposalJob]) -> dict[str, Any]:
    """Fail closed when the model package differs from the frozen DS1 screen."""

    contract = DS1_SCREEN_CONTRACT
    keys = [job.key for job in jobs]
    scene_ids = {job.opaque_scene_id for job in jobs}
    views = {job.view_id for job in jobs}
    if len(keys) != len(set(keys)):
        raise ValueError("DS1 proposal jobs contain duplicate scene/view pairs")
    if len(jobs) != contract.image_count:
        raise ValueError("DS1 proposal image count differs from the cohort contract")
    if len(scene_ids) != contract.scene_count:
        raise ValueError("DS1 proposal scene count differs from the cohort contract")
    expected_views = {f"V{index:02d}" for index in range(contract.views_per_scene)}
    if views != expected_views:
        raise ValueError("DS1 proposal views differ from the cohort contract")
    counts = Counter(job.opaque_scene_id for job in jobs)
    if set(counts.values()) != {contract.views_per_scene}:
        raise ValueError("Every DS1 scene must contain all declared views")
    observed_membership = membership_sha256(scene_ids)
    if observed_membership != contract.membership_sha256:
        raise ValueError("DS1 proposal membership differs from the frozen screen")
    cohorts = {job.cohort_id for job in jobs}
    if cohorts != {contract.cohort_id}:
        raise ValueError("DS1 proposal jobs identify another cohort")
    manifests = {job.input_manifest_sha256 for job in jobs}
    if len(manifests) != 1:
        raise ValueError("DS1 proposal jobs identify multiple input manifests")
    return {
        "release_id": contract.release_id,
        "cohort_id": contract.cohort_id,
        "role": contract.role,
        "pilot_stage": contract.pilot_stage,
        "membership_sha256": observed_membership,
        "input_manifest_sha256": next(iter(manifests)),
        "scene_count": len(scene_ids),
        "views_per_scene": len(views),
        "image_count": len(jobs),
        "asset_group_count": contract.asset_group_count,
        "all_checks_passed": True,
    }


def select_pilot_jobs(jobs: Sequence[ProposalJob]) -> list[ProposalJob]:
    """Select 20 deterministic jobs covering 20 scenes and all view positions."""

    by_key = {job.key: job for job in jobs}
    scenes = sorted({job.opaque_scene_id for job in jobs})
    views = [f"V{index:02d}" for index in range(PILOT_IMAGE_COUNT)]
    selected = [by_key[(scene, view)] for scene, view in zip(scenes[:20], views)]
    if len(selected) != PILOT_IMAGE_COUNT:
        raise ValueError("DS1 pilot must contain 20 images")
    if len({job.opaque_scene_id for job in selected}) != PILOT_IMAGE_COUNT:
        raise ValueError("DS1 pilot must cover 20 scenes")
    if {job.view_id for job in selected} != set(views):
        raise ValueError("DS1 pilot must cover all 20 view positions")
    return selected


def _run_stage(
    *,
    worker: Any,
    jobs: Sequence[ProposalJob],
    path: Path,
    configuration: LocateAnythingConfiguration,
    producer_kind: str,
    maximum_attempts_per_job: int,
    maximum_consecutive_failed_jobs: int,
    progress_callback: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> tuple[ProposalResultStore, dict[str, Any]]:
    store = ProposalResultStore(path, configuration.sha256, producer_kind)
    session = run_proposal_jobs(
        worker=worker,
        jobs=jobs,
        result_store=store,
        maximum_attempts_per_job=maximum_attempts_per_job,
        maximum_consecutive_failed_jobs=maximum_consecutive_failed_jobs,
        progress_callback=progress_callback,
    )
    return store, session


def _diagnostic_gates(
    summary: Mapping[str, Any], *, expected_jobs: int, expected_scenes: int
) -> list[dict[str, Any]]:
    checks = (
        ("unique jobs", summary["job_count"], expected_jobs),
        ("successful jobs", summary["successful_job_count"], expected_jobs),
        ("failed jobs", summary["failed_job_count"], 0),
        ("scene coverage", summary["scene_count"], expected_scenes),
        ("truncated responses", summary["possibly_truncated_count"], 0),
        ("zero-proposal successes", summary["zero_proposal_image_count"], 0),
    )
    return [
        {
            "gate": name,
            "status": "pass" if observed == expected else "fail",
            "observed": observed,
            "expected": expected,
        }
        for name, observed, expected in checks
    ]


def _attempt_rows(history_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attempts_by_image = Counter(
        (str(row["opaque_scene_id"]), str(row["view_id"])) for row in history_rows
    )
    counts = Counter(attempts_by_image.values())
    return [
        {"attempt_count": attempt_count, "image_count": image_count}
        for attempt_count, image_count in sorted(counts.items())
    ]


def _stage_gate_rows(
    one_image_summary: Mapping[str, Any],
    pilot_summary: Mapping[str, Any],
    repeat_summary: Mapping[str, Any],
    *,
    inference_stage: str,
) -> list[dict[str, Any]]:
    checks: list[tuple[str, Any, Any]] = [
        ("preflight images", one_image_summary.get("job_count"), 1),
        ("preflight successes", one_image_summary.get("successful_job_count"), 1),
        ("preflight failures", one_image_summary.get("failed_job_count"), 0),
        ("preflight truncated responses", one_image_summary.get("possibly_truncated_count"), 0),
        ("preflight zero-proposal successes", one_image_summary.get("zero_proposal_image_count"), 0),
    ]
    if inference_stage in {"pilot", "full"}:
        checks.extend(
            [
                ("pilot images", pilot_summary.get("job_count"), PILOT_IMAGE_COUNT),
                ("pilot successes", pilot_summary.get("successful_job_count"), PILOT_IMAGE_COUNT),
                ("pilot failures", pilot_summary.get("failed_job_count"), 0),
                ("pilot scene coverage", pilot_summary.get("scene_count"), PILOT_IMAGE_COUNT),
                ("pilot truncated responses", pilot_summary.get("possibly_truncated_count"), 0),
                ("pilot zero-proposal successes", pilot_summary.get("zero_proposal_image_count"), 0),
                ("repeat images", repeat_summary.get("job_count"), PILOT_IMAGE_COUNT),
                ("repeat matching images", repeat_summary.get("matching_job_count"), PILOT_IMAGE_COUNT),
                ("repeat box agreement", repeat_summary.get("box_agreement_rate"), 1.0),
            ]
        )
    return [
        {
            "gate": name,
            "status": "pass" if observed == expected else "fail",
            "observed": observed,
            "expected": expected,
        }
        for name, observed, expected in checks
    ]


def _require_passed_stage(gates: Sequence[Mapping[str, Any]], stage: str) -> None:
    failures = [str(row["gate"]) for row in gates if row["status"] != "pass"]
    if failures:
        raise RuntimeError(f"DS1 LocateAnything {stage} gate failed: {', '.join(failures)}")


def _verify_canonical_provenance(provenance: Mapping[str, Any]) -> None:
    if provenance.get("git_dirty") is not False or provenance.get("precommit_candidate") is not False:
        raise ValueError("Canonical DS1 model inference requires a clean committed snapshot")
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_commit", ""))):
        raise ValueError("Canonical DS1 model inference requires a full Git commit")
    for field in ("snapshot_sha256", "notebook_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(field, ""))):
            raise ValueError(f"Canonical DS1 model inference requires {field}")


def export_ds1_checkpoint(
    output_root: Path,
    archive_path: Path,
    *,
    configuration_sha256: str,
    producer_kind: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    """Export private resumable JSONL state for a fresh private Kaggle version."""

    private_root = output_root.expanduser().resolve() / "private"
    available = [name for name in CHECKPOINT_FILES if (private_root / name).is_file()]
    if not available:
        raise ValueError("No DS1 LocateAnything checkpoint files are available")
    record = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "package_kind": "ds1-locateanything-private-checkpoint",
        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
        "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
        "configuration_sha256": configuration_sha256,
        "producer_kind": producer_kind,
        "input_manifest_sha256": input_manifest_sha256,
        "files": {
            name: file_sha256(private_root / name) for name in available
        },
    }
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".incomplete")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "checkpoint_record.json",
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )
        for name in available:
            archive.write(private_root / name, arcname=name)
    temporary.replace(archive_path)
    return {**record, "archive_sha256": file_sha256(archive_path)}


def import_ds1_checkpoint(
    archive_path: Path,
    output_root: Path,
    *,
    configuration_sha256: str,
    producer_kind: str,
    input_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify and restore a private checkpoint without accepting extra files."""

    archive_path = archive_path.expanduser().resolve(strict=True)
    private_root = output_root.expanduser().resolve() / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or "checkpoint_record.json" not in names:
            raise ValueError("DS1 checkpoint archive is malformed")
        if any(
            name != "checkpoint_record.json" and name not in CHECKPOINT_FILES
            for name in names
        ):
            raise ValueError("DS1 checkpoint archive contains an unexpected file")
        if any(member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000 for member in members):
            raise ValueError("DS1 checkpoint archive contains a directory or symlink")
        record = json.loads(archive.read("checkpoint_record.json"))
        if record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("DS1 checkpoint schema version differs")
        expected = {
            "package_kind": "ds1-locateanything-private-checkpoint",
            "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
            "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
            "configuration_sha256": configuration_sha256,
            "producer_kind": producer_kind,
            "input_manifest_sha256": input_manifest_sha256,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"DS1 checkpoint differs: {key}")
        if set(record.get("files", {})) != set(names).difference({"checkpoint_record.json"}):
            raise ValueError("DS1 checkpoint record and archive contents differ")
        for name, expected_sha256 in sorted(record["files"].items()):
            data = archive.read(name)
            if sha256(data).hexdigest() != expected_sha256:
                raise ValueError("DS1 checkpoint content hash mismatch")
            destination = private_root / name
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError("Existing DS1 checkpoint state differs")
            if not destination.exists():
                temporary = destination.with_name(destination.name + ".incomplete")
                temporary.write_bytes(data)
                temporary.replace(destination)
    return {**record, "archive_sha256": file_sha256(archive_path)}


def _failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["error_type"]) for row in rows if row.get("error_type"))
    return [
        {"error_type": error_type, "count": count}
        for error_type, count in sorted(counts.items())
    ]


def _save_safe_evidence(
    *,
    safe_root: Path,
    evidence: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    configuration: LocateAnythingConfiguration,
    run_record: Mapping[str, Any],
    one_image_summary: Mapping[str, Any],
    pilot_summary: Mapping[str, Any],
    repeat_summary: Mapping[str, Any],
    progress_rows: Sequence[Mapping[str, Any]],
    environment_summary: Mapping[str, Any],
    model_load_summary: Mapping[str, Any],
) -> dict[str, Any]:
    safe_root.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        safe_root / "completion_gates.csv",
        ["gate", "status", "observed", "expected"],
        evidence["gate_rows"],
    )
    write_csv_rows(
        safe_root / "proposal_summary_by_view.csv",
        [
            "view_id",
            "image_count",
            "successful_count",
            "failed_count",
            "mean_proposal_count",
            "mean_runtime_seconds",
            "truncated_count",
        ],
        evidence.get("view_rows", []),
    )
    write_csv_rows(
        safe_root / "attempt_summary.csv",
        ["attempt_count", "image_count"],
        evidence["attempt_rows"],
    )
    write_csv_rows(
        safe_root / "failure_summary.csv",
        ["error_type", "count"],
        evidence["failure_rows"],
    )
    write_csv_rows(
        safe_root / "progress_checkpoints.csv",
        ["completed", "total", "successful", "failed", "elapsed_seconds"],
        progress_rows,
    )
    _write_json(safe_root / "resolved_configuration.json", asdict(configuration))
    _write_json(safe_root / "input_audit.json", input_audit)
    _write_json(safe_root / "one_image_preflight.json", one_image_summary)
    _write_json(safe_root / "pilot_summary.json", pilot_summary)
    _write_json(safe_root / "repeat_agreement.json", repeat_summary)
    _write_json(safe_root / "proposal_summary.json", evidence["summary"])
    _write_json(safe_root / "box_geometry_summary.json", evidence["geometry_summary"])
    _write_json(safe_root / "runtime_summary.json", run_record["runtime"])
    _write_json(safe_root / "environment_summary.json", environment_summary)
    _write_json(safe_root / "model_load_summary.json", model_load_summary)
    _write_json(safe_root / "run_record.json", run_record)

    proposal_counts = evidence["proposal_counts"] or [0]
    runtimes = evidence["runtime_seconds"] or [0.0]
    box_areas = evidence["box_area_fractions"] or [0.0]
    box_ratios = evidence["box_aspect_ratios"] or [0.0]
    save_histogram(
        proposal_counts,
        safe_root / "proposal_count_distribution.png",
        title="DS1 proposals per image",
        x_label="Proposal count",
        bins=12,
    )
    save_histogram(
        runtimes,
        safe_root / "runtime_distribution.png",
        title="DS1 LocateAnything runtime",
        x_label="Seconds per image",
        bins=14,
    )
    save_histogram(
        box_areas,
        safe_root / "box_area_fraction_distribution.png",
        title="DS1 proposal box area",
        x_label="Fraction of image area",
        bins=14,
    )
    save_histogram(
        box_ratios,
        safe_root / "box_aspect_ratio_distribution.png",
        title="DS1 proposal box aspect ratio",
        x_label="Width divided by height",
        bins=14,
    )
    view_rows = evidence.get("view_rows", [])
    view_labels = [str(row["view_id"]) for row in view_rows] or ["diagnostic"]
    view_values = [float(row["mean_proposal_count"]) for row in view_rows] or [
        float(evidence["summary"]["mean_proposals_per_successful_image"])
    ]
    save_bar_chart(
        view_labels,
        view_values,
        safe_root / "mean_proposals_by_view.png",
        title="Mean proposals by view",
        y_label="Mean proposals",
    )
    save_scatter_plot(
        proposal_counts,
        runtimes,
        safe_root / "runtime_by_proposal_count.png",
        title="Runtime and proposal count",
        x_label="Proposal count",
        y_label="Seconds per image",
    )
    return write_artifact_manifest(safe_root)


def run_ds1_locateanything_workflow(
    *,
    model_input_root: Path,
    output_root: Path,
    worker: Any,
    configuration: LocateAnythingConfiguration,
    run_mode: str,
    inference_stage: str = "full",
    provenance: Mapping[str, Any] | None = None,
    platform_versions: Mapping[str, Any] | None = None,
    model_load_evidence: Mapping[str, Any] | None = None,
    maximum_attempts_per_job: int = 2,
    maximum_consecutive_failed_jobs: int = 3,
) -> dict[str, Any]:
    """Run DS1 preflight, pilot/repeat and the resumable all-view proposal job."""

    if run_mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {sorted(RUN_MODES)}")
    if inference_stage not in INFERENCE_STAGES:
        raise ValueError(f"inference_stage must be one of {sorted(INFERENCE_STAGES)}")
    if run_mode == "model_inference" and inference_stage == "full":
        _verify_canonical_provenance(dict(provenance or {}))
        load_evidence = dict(model_load_evidence or {})
        if load_evidence.get("disposable_preflight", {}).get("status") != "passed":
            raise ValueError("Canonical DS1 model inference requires the disposable one-image gate")
        if not any(
            row.get("stage") == "model_ready"
            for row in load_evidence.get("checkpoints", [])
        ):
            raise ValueError("Canonical DS1 model inference requires a completed model load")
    worker_configuration = getattr(worker, "configuration", None)
    if worker_configuration is None or worker_configuration.sha256 != configuration.sha256:
        raise ValueError("LocateAnything worker and resolved configuration differ")

    from visual_peer_inspector.models.locateanything import build_proposal_jobs

    output_root = output_root.expanduser().resolve()
    private_root = output_root / "private"
    safe_root = output_root / "safe"
    private_root.mkdir(parents=True, exist_ok=True)
    package_audit = verify_model_input_package(
        model_input_root,
        contract=DS1_SCREEN_CONTRACT,
    )
    jobs = build_proposal_jobs(model_input_root, DS1_SCREEN_CONTRACT.cohort_id)
    input_audit = verify_ds1_jobs(jobs)
    if package_audit.manifest_sha256 != input_audit["input_manifest_sha256"]:
        raise ValueError("Proposal jobs and verified model input use different manifests")
    input_audit["model_input_package"] = package_audit.to_dict()
    pilot_jobs = select_pilot_jobs(jobs)
    producer_kind = (
        MODEL_PRODUCER_KIND if run_mode == "model_inference" else CONTRACT_PRODUCER_KIND
    )
    progress_rows: list[dict[str, Any]] = []
    progress_started = time.perf_counter()
    progress_successful = 0
    progress_failed = 0

    def progress_callback(completed: int, total: int, row: Mapping[str, Any]) -> None:
        nonlocal progress_successful, progress_failed
        if row["status"] == "ok":
            progress_successful += 1
        else:
            progress_failed += 1
        if completed == 1 or completed == total or completed % 100 == 0:
            progress_rows.append(
                {
                    "completed": completed,
                    "total": total,
                    "successful": progress_successful,
                    "failed": progress_failed,
                    "elapsed_seconds": round(time.perf_counter() - progress_started, 3),
                }
            )

    one_store, one_session = _run_stage(
        worker=worker,
        jobs=jobs[:1],
        path=private_root / "one_image_results.jsonl",
        configuration=configuration,
        producer_kind=producer_kind,
        maximum_attempts_per_job=maximum_attempts_per_job,
        maximum_consecutive_failed_jobs=maximum_consecutive_failed_jobs,
    )
    one_image_summary = summarize_proposal_rows(one_store.rows)
    one_image_summary["session"] = one_session
    one_image_gates = _stage_gate_rows(
        one_image_summary,
        {},
        {},
        inference_stage="one_image",
    )
    if inference_stage in {"pilot", "full"}:
        _require_passed_stage(one_image_gates, "one-image preflight")

    pilot_store: ProposalResultStore | None = None
    repeat_store: ProposalResultStore | None = None
    pilot_summary: dict[str, Any] = {}
    repeat_summary: dict[str, Any] = {}
    if inference_stage in {"pilot", "full"}:
        pilot_store, pilot_session = _run_stage(
            worker=worker,
            jobs=pilot_jobs,
            path=private_root / "pilot_results.jsonl",
            configuration=configuration,
            producer_kind=producer_kind,
            maximum_attempts_per_job=maximum_attempts_per_job,
            maximum_consecutive_failed_jobs=maximum_consecutive_failed_jobs,
        )
        repeat_store, repeat_session = _run_stage(
            worker=worker,
            jobs=pilot_jobs,
            path=private_root / "pilot_repeat_results.jsonl",
            configuration=configuration,
            producer_kind=producer_kind,
            maximum_attempts_per_job=maximum_attempts_per_job,
            maximum_consecutive_failed_jobs=maximum_consecutive_failed_jobs,
        )
        pilot_summary = summarize_proposal_rows(pilot_store.rows)
        pilot_summary["session"] = pilot_session
        repeat_summary = repeat_agreement(pilot_store.rows, repeat_store.rows)
        repeat_summary["session"] = repeat_session
        pilot_gates = _stage_gate_rows(
            one_image_summary,
            pilot_summary,
            repeat_summary,
            inference_stage="pilot",
        )
        if inference_stage == "full":
            _require_passed_stage(pilot_gates, "pilot and repeatability")

    if inference_stage == "full":
        main_jobs = jobs
        result_store, run_session = _run_stage(
            worker=worker,
            jobs=main_jobs,
            path=private_root / "proposal_results.jsonl",
            configuration=configuration,
            producer_kind=producer_kind,
            maximum_attempts_per_job=maximum_attempts_per_job,
            maximum_consecutive_failed_jobs=maximum_consecutive_failed_jobs,
            progress_callback=progress_callback,
        )
        evidence = collect_proposal_evidence(
            result_store.rows,
            main_jobs,
            history_rows=result_store.history_rows,
            expected_job_count=DS1_SCREEN_CONTRACT.image_count,
            expected_scene_count=DS1_SCREEN_CONTRACT.scene_count,
            expected_view_count=DS1_SCREEN_CONTRACT.views_per_scene,
        )
        evidence["failure_rows"] = _failure_rows(result_store.history_rows)
    elif inference_stage == "pilot":
        assert pilot_store is not None
        main_jobs = pilot_jobs
        result_store = pilot_store
        run_session = pilot_summary["session"]
        summary = summarize_proposal_rows(result_store.rows)
        evidence = {
            "summary": summary,
            "gate_rows": _diagnostic_gates(
                summary, expected_jobs=PILOT_IMAGE_COUNT, expected_scenes=PILOT_IMAGE_COUNT
            ),
            "view_rows": [],
            "attempt_rows": _attempt_rows(result_store.history_rows),
            "failure_rows": _failure_rows(result_store.rows),
            "geometry_summary": {
                "box_count": sum(int(row["proposal_count"]) for row in result_store.rows),
                "invalid_box_count": 0,
                "duplicate_box_count": 0,
            },
            "proposal_counts": [int(row["proposal_count"]) for row in result_store.rows],
            "runtime_seconds": [float(row["runtime_seconds"]) for row in result_store.rows],
            "box_area_fractions": [0.0],
            "box_aspect_ratios": [0.0],
        }
    else:
        main_jobs = jobs[:1]
        result_store = one_store
        run_session = one_session
        summary = summarize_proposal_rows(result_store.rows)
        evidence = {
            "summary": summary,
            "gate_rows": _diagnostic_gates(summary, expected_jobs=1, expected_scenes=1),
            "view_rows": [],
            "attempt_rows": _attempt_rows(result_store.history_rows),
            "failure_rows": _failure_rows(result_store.rows),
            "geometry_summary": {
                "box_count": sum(int(row["proposal_count"]) for row in result_store.rows),
                "invalid_box_count": 0,
                "duplicate_box_count": 0,
            },
            "proposal_counts": [int(row["proposal_count"]) for row in result_store.rows],
            "runtime_seconds": [float(row["runtime_seconds"]) for row in result_store.rows],
            "box_area_fractions": [0.0],
            "box_aspect_ratios": [0.0],
        }

    stage_gates = _stage_gate_rows(
        one_image_summary,
        pilot_summary,
        repeat_summary,
        inference_stage=inference_stage,
    )
    evidence["gate_rows"] = [*stage_gates, *evidence["gate_rows"]]
    all_gates_passed = all(row["status"] == "pass" for row in evidence["gate_rows"])
    status = (
        "completed"
        if inference_stage == "full" and all_gates_passed
        else "diagnostic_completed"
        if all_gates_passed
        else "failed"
    )
    observed_gpu_allocated = 0.0
    observed_gpu_reserved = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            observed_gpu_allocated = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 4
            )
            observed_gpu_reserved = round(
                torch.cuda.max_memory_reserved() / (1024 ** 3), 4
            )
    except ImportError:
        pass
    model_load_summary = dict(model_load_evidence or {})
    runtime_summary = {
        "session_elapsed_seconds": run_session["elapsed_seconds"],
        "model_download_seconds": float((model_load_evidence or {}).get("download_seconds", 0.0)),
        "model_load_seconds": float((model_load_evidence or {}).get("load_seconds", 0.0)),
        "total_inference_seconds": evidence["summary"]["total_runtime_seconds"],
        "median_image_runtime_seconds": evidence["summary"]["median_runtime_seconds"],
        "p95_image_runtime_seconds": evidence["summary"]["p95_runtime_seconds"],
        "images_per_second": evidence["summary"]["images_per_second"],
        "peak_gpu_allocated_gib": max(
            observed_gpu_allocated,
            float((model_load_evidence or {}).get("peak_gpu_allocated_gib", 0.0)),
        ),
        "peak_gpu_reserved_gib": max(
            observed_gpu_reserved,
            float((model_load_evidence or {}).get("peak_gpu_reserved_gib", 0.0)),
        ),
    }
    environment_summary = {
        "python_version": platform.python_version(),
        "run_mode": run_mode,
        "inference_stage": inference_stage,
        "platform_versions": dict(platform_versions or {}),
        "device": dict((model_load_evidence or {}).get("device", {})),
    }
    main_result_path = (
        private_root / "proposal_results.jsonl"
        if inference_stage == "full"
        else private_root / "pilot_results.jsonl"
        if inference_stage == "pilot"
        else private_root / "one_image_results.jsonl"
    )
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "stage": "ds1_locateanything_screen",
        "status": status,
        "tracking_status": "pending",
        "run_mode": run_mode,
        "inference_stage": inference_stage,
        "producer_kind": producer_kind,
        "cohort": input_audit,
        "configuration": asdict(configuration),
        "configuration_sha256": configuration.sha256,
        "input_identity": {
            "model_input_manifest_sha256": input_audit["input_manifest_sha256"],
        },
        "output_identity": {
            "result_filename": main_result_path.name,
            "result_sha256": file_sha256(main_result_path),
            "proposal_results_sha256": file_sha256(main_result_path),
            "result_count": len(result_store.rows),
        },
        "provenance": dict(provenance or {}),
        "summary": evidence["summary"],
        "one_image_preflight": one_image_summary,
        "pilot_summary": pilot_summary,
        "repeat_agreement": repeat_summary,
        "runtime": runtime_summary,
        "run_session": run_session,
        "completion_gate_passed": status == "completed",
        "error_type": None if status != "failed" else "CompletionGateFailure",
    }
    _save_safe_evidence(
        safe_root=safe_root,
        evidence=evidence,
        input_audit=input_audit,
        configuration=configuration,
        run_record=run_record,
        one_image_summary=one_image_summary,
        pilot_summary=pilot_summary,
        repeat_summary=repeat_summary,
        progress_rows=progress_rows,
        environment_summary=environment_summary,
        model_load_summary=model_load_summary,
    )
    if inference_stage == "full" and evidence.get("overlay_items"):
        visual_title = (
            "Private DS1 LocateAnything model proposal examples"
            if run_mode == "model_inference"
            else "Synthetic contract-smoke boxes, not model predictions"
        )
        save_image_grid(
            list(evidence["overlay_items"])[:20],
            private_root / "proposal_examples.png",
            title=visual_title,
            columns=5,
        )
        if evidence.get("outlier_overlay_items"):
            save_image_grid(
                list(evidence["outlier_overlay_items"]),
                private_root / "proposal_outliers.png",
                title=(
                    "Private model proposal outliers"
                    if run_mode == "model_inference"
                    else "Synthetic contract-smoke outliers, not model predictions"
                ),
                columns=5,
            )
        write_csv_rows(
            private_root / "proposal_outliers.csv",
            [
                "reason",
                "job_index",
                "view_id",
                "status",
                "attempt_count",
                "proposal_count",
                "response_token_count",
                "truncation_check",
                "runtime_seconds",
                "mean_box_area_fraction",
            ],
            evidence.get("outlier_rows", []),
        )
        write_artifact_manifest(private_root)
    return run_record


__all__ = [
    "INFERENCE_STAGES",
    "PILOT_IMAGE_COUNT",
    "RUN_MODES",
    "SAFE_CHARTS",
    "SAFE_JSON",
    "SAFE_TABLES",
    "export_ds1_checkpoint",
    "import_ds1_checkpoint",
    "run_ds1_locateanything_workflow",
    "select_pilot_jobs",
    "verify_ds1_jobs",
    "write_artifact_manifest",
]
