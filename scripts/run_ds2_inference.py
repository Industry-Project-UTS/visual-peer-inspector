#!/usr/bin/env python3
"""Run the frozen label-free DS2 LocateAnything, SAM and DINO stages."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.confirmation_mask_routes import (
    build_ds2_mask_route_package,
    verify_ds2_mask_route_package,
)
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.locateanything import (
    LocateAnythingWorker,
    ProposalContractWorker,
)
from visual_peer_inspector.models.sam2 import Sam2Configuration, Sam2ImageWorker
from visual_peer_inspector.workflows.locateanything_confirmation_ds2 import (
    load_frozen_ds1_configuration,
    resolve_ds2_operational_policy,
    run_ds2_locateanything,
)
from visual_peer_inspector.workflows.representation_confirmation_ds2 import (
    run_ds2_representation,
)
from visual_peer_inspector.workflows.sam_confirmation_ds2 import (
    run_ds2_automatic_sam,
    run_ds2_reference_sam,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--selected-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("contract", "model_inference"), default="contract")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--locate-model-source", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recovery-input-root", type=Path)
    parser.add_argument("--recovery-output-root", type=Path)
    return parser.parse_args()


RECOVERY_PATHS = (
    Path("locateanything/private/raw_proposal_history.jsonl"),
    Path("sam21/reference-box/private/checkpoints"),
    Path("sam21/automatic-box/private/checkpoints"),
    Path("representation/private/score-shards"),
    Path("representation/cache"),
)
RECOVERY_ARCHIVE_NAME = "recovery_snapshot.zip"


def _recovery_path_is_allowed(relative: Path) -> bool:
    canonical = relative.as_posix()
    allowed = tuple(path.as_posix() for path in RECOVERY_PATHS)
    return any(
        canonical == prefix or canonical.startswith(f"{prefix}/")
        for prefix in allowed
    )


def _archive_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    from hashlib import sha256

    digest = sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_recovery_archive(archive_path: Path, output_root: Path) -> int:
    """Verify and restore one atomically published recovery snapshot."""

    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or "recovery_record.json" not in names:
            raise ValueError("DS2 recovery archive members are invalid")
        if archive.testzip() is not None:
            raise ValueError("DS2 recovery archive failed its integrity check")
        record = json.loads(archive.read("recovery_record.json").decode("utf-8"))
        files = record.get("files")
        if (
            record.get("schema_version") != 2
            or record.get("archive_kind") != "ds2-private-recovery-snapshot"
            or not isinstance(files, dict)
            or int(record.get("file_count", -1)) != len(files)
            or set(names) != set(files) | {"recovery_record.json"}
        ):
            raise ValueError("DS2 recovery archive record is invalid")
        copied = 0
        for name, expected_sha256 in sorted(files.items()):
            relative = Path(str(name))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not _recovery_path_is_allowed(relative)
            ):
                raise ValueError("DS2 recovery archive contains an unsupported path")
            if _archive_member_sha256(archive, name) != str(expected_sha256):
                raise ValueError("DS2 recovery checkpoint failed integrity verification")
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".incomplete")
            with archive.open(name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(destination)
            copied += 1
    return copied


def seed_recovery(input_root: Path | None, output_root: Path) -> int:
    """Copy verified private checkpoints into the temporary run directory."""

    if input_root is None:
        return 0
    input_root = input_root.expanduser().resolve()
    archive_path = input_root / RECOVERY_ARCHIVE_NAME
    if archive_path.is_file():
        copied = _seed_recovery_archive(archive_path, output_root)
        print(f"Seeded DS2 recovery files: {copied}", flush=True)
        return copied

    # Accept the first recovery format so an already published checkpoint can resume.
    record_path = input_root / "recovery_record.json"
    if not record_path.is_file():
        raise FileNotFoundError("DS2 recovery input is missing recovery_record.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    files = record.get("files")
    if (
        record.get("schema_version") != 1
        or not isinstance(files, dict)
        or int(record.get("file_count", -1)) != len(files)
    ):
        raise ValueError("DS2 recovery record is invalid")
    actual_files = {
        path.relative_to(input_root).as_posix()
        for path in input_root.rglob("*")
        if path.is_file() and path.name != "recovery_record.json"
    }
    if actual_files != set(files):
        raise ValueError("DS2 recovery file inventory differs from its record")
    copied = 0
    for name, expected_sha256 in sorted(files.items()):
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("DS2 recovery record contains an unsafe path")
        if not _recovery_path_is_allowed(relative):
            raise ValueError("DS2 recovery record contains an unsupported checkpoint path")
        source = input_root / relative
        if not source.is_file() or file_sha256(source) != str(expected_sha256):
            raise ValueError("DS2 recovery checkpoint failed integrity verification")
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    print(f"Seeded DS2 recovery files: {copied}", flush=True)
    return copied


def _recovery_files(output_root: Path) -> list[tuple[Path, Path]]:
    files: dict[Path, Path] = {}
    for relative in RECOVERY_PATHS:
        source = output_root / relative
        if source.is_symlink():
            raise ValueError("DS2 recovery checkpoints cannot contain symbolic links")
        if source.is_file():
            files[relative] = source
        elif source.is_dir():
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise ValueError("DS2 recovery checkpoints cannot contain symbolic links")
                if path.is_file():
                    files[path.relative_to(output_root)] = path
    return sorted(files.items(), key=lambda item: item[0].as_posix())


def _write_recovery_archive(output_root: Path, archive_path: Path) -> int:
    files = _recovery_files(output_root)
    record = {
        "schema_version": 2,
        "archive_kind": "ds2-private-recovery-snapshot",
        "purpose": "Private resume checkpoints for Notebook 08B",
        "file_count": len(files),
        "files": {
            relative.as_posix(): file_sha256(path)
            for relative, path in files
        },
    }
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for relative, path in files:
            archive.write(path, relative.as_posix())
        archive.writestr(
            "recovery_record.json",
            json.dumps(record, indent=2, sort_keys=True) + "\n",
        )
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("New DS2 recovery archive failed its integrity check")
    return len(files)


def save_recovery(output_root: Path, recovery_root: Path | None) -> int:
    """Atomically replace one private checkpoint snapshot after full verification."""

    if recovery_root is None:
        return 0
    recovery_root = recovery_root.expanduser().resolve()
    recovery_root.mkdir(parents=True, exist_ok=True)
    archive_path = recovery_root / RECOVERY_ARCHIVE_NAME
    temporary = recovery_root / f".{RECOVERY_ARCHIVE_NAME}.incomplete"
    if temporary.exists():
        temporary.unlink()
    try:
        file_count = _write_recovery_archive(output_root, temporary)
        os.replace(temporary, archive_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    # Remove legacy expanded copies only after the new snapshot is durable.
    legacy_record = recovery_root / "recovery_record.json"
    if legacy_record.exists():
        legacy_record.unlink()
    for relative in RECOVERY_PATHS:
        legacy = recovery_root / relative
        if legacy.is_dir():
            shutil.rmtree(legacy)
        elif legacy.exists():
            legacy.unlink()
    print(f"Saved DS2 recovery files: {file_count}", flush=True)
    return file_count


def clear_released_gpu_memory() -> None:
    """Collect released objects and return unused CUDA memory to the allocator."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def recovery_progress_callback(
    *,
    output_root: Path,
    recovery_root: Path | None,
    label: str,
    recovery_interval: int = 1_000,
    display_interval: int = 250,
):
    """Return a stage callback that checkpoints completed work at fixed intervals."""

    if recovery_interval < 1 or display_interval < 1:
        raise ValueError("Recovery and display intervals must be positive")

    def report(completed: int, total: int, _row: object) -> None:
        if completed == 1 or completed % display_interval == 0 or completed == total:
            print(f"{label}: {completed}/{total}", flush=True)
        if completed % recovery_interval == 0 or completed == total:
            save_recovery(output_root, recovery_root)

    return report


def main() -> None:
    args = parse_arguments()
    output_root = args.output_root.expanduser().resolve()
    seed_recovery(args.recovery_input_root, output_root)

    def progress_with_recovery(label: str, *, recovery_interval: int = 1_000):
        return recovery_progress_callback(
            output_root=output_root,
            recovery_root=args.recovery_output_root,
            label=label,
            recovery_interval=recovery_interval,
        )
    frozen_locate_configuration, _policy, _freeze_sha256 = load_frozen_ds1_configuration(
        args.selected_prompt
    )
    locate_configuration, locate_execution_policy = resolve_ds2_operational_policy(
        frozen_locate_configuration
    )
    if args.run_mode == "model_inference":
        locate_worker = LocateAnythingWorker(
            locate_configuration,
            device=args.device,
            model_source=args.locate_model_source,
            progress_callback=lambda stage, details: print(
                json.dumps({"component": "LocateAnything", "stage": stage, **details}, sort_keys=True),
                flush=True,
            ),
            execution_policy=locate_execution_policy,
        )
    else:
        locate_worker = ProposalContractWorker(
            locate_configuration,
            execution_policy=locate_execution_policy,
        )

    locate_root = output_root / "locateanything"
    locate_record = run_ds2_locateanything(
        rgb_root=args.rgb_root,
        selected_prompt_path=args.selected_prompt,
        output_root=locate_root,
        worker=locate_worker,
        run_mode=args.run_mode,
        progress_callback=progress_with_recovery("DS2 LocateAnything images"),
    )
    save_recovery(output_root, args.recovery_output_root)
    locate_worker = None
    clear_released_gpu_memory()

    sam_worker = None
    if args.run_mode == "model_inference":
        if args.sam_checkpoint is None:
            raise ValueError("Model inference requires --sam-checkpoint")
        configuration = Sam2Configuration(device=args.device, precision="float16")
        sam_worker = Sam2ImageWorker(configuration, args.sam_checkpoint)

    sam_root = output_root / "sam21"
    reference_sam_root = sam_root / "reference-box"
    reference_sam_record = run_ds2_reference_sam(
        rgb_root=args.rgb_root,
        geometry_root=args.geometry_root,
        output_root=reference_sam_root,
        worker=sam_worker,
        run_mode=args.run_mode,
        progress_callback=progress_with_recovery("DS2 SAM images"),
    )
    save_recovery(output_root, args.recovery_output_root)
    automatic_sam_root = sam_root / "automatic-box"
    automatic_sam_record = run_ds2_automatic_sam(
        rgb_root=args.rgb_root,
        proposal_root=locate_root / "model-input",
        output_root=automatic_sam_root,
        worker=sam_worker,
        run_mode=args.run_mode,
        progress_callback=progress_with_recovery("DS2 automatic-box SAM images"),
    )
    save_recovery(output_root, args.recovery_output_root)
    sam_worker = None
    clear_released_gpu_memory()

    mask_route_root = output_root / "mask-routes"
    if mask_route_root.exists():
        mask_record = verify_ds2_mask_route_package(mask_route_root)
    else:
        mask_record = build_ds2_mask_route_package(
            rgb_root=args.rgb_root,
            geometry_root=args.geometry_root,
            sam_results_path=reference_sam_root / "private" / "sam_results.jsonl",
            automatic_sam_results_path=automatic_sam_root / "private" / "sam_results.jsonl",
            output_root=mask_route_root,
            provenance={
                "reference_sam_run_record_sha256": file_sha256(reference_sam_root / "safe" / "run_record.json"),
                "reference_sam_results_sha256": file_sha256(reference_sam_root / "private" / "sam_results.jsonl"),
                "automatic_sam_run_record_sha256": file_sha256(automatic_sam_root / "safe" / "run_record.json"),
                "automatic_sam_results_sha256": file_sha256(automatic_sam_root / "private" / "sam_results.jsonl"),
                "automatic_proposal_record_sha256": file_sha256(locate_root / "model-input" / "package_record.json"),
                "geometry_record_sha256": file_sha256(args.geometry_root / "package_record.json"),
            },
        )

    representation_record = run_ds2_representation(
        rgb_root=args.rgb_root,
        mask_route_root=mask_route_root,
        output_root=output_root / "representation",
        run_mode=args.run_mode,
        device="cuda" if args.run_mode == "model_inference" else args.device,
        batch_size=args.batch_size,
        progress_callback=progress_with_recovery(
            "DS2 representation scenes",
            recovery_interval=25,
        ),
    )
    save_recovery(output_root, args.recovery_output_root)
    print(json.dumps({
        "locateanything": locate_record,
        "reference_sam": reference_sam_record,
        "automatic_sam": automatic_sam_record,
        "mask_routes": mask_record,
        "representation": representation_record,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
