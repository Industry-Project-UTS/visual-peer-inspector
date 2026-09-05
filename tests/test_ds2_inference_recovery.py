from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_ds2_inference.py"
SPEC = importlib.util.spec_from_file_location("run_ds2_inference", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recovery_round_trip_preserves_supported_checkpoints(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    files = {
        "locateanything/private/raw_proposal_history.jsonl": b"proposal\n",
        "sam21/reference-box/private/checkpoints/S000001/V00.json": b"reference\n",
        "sam21/automatic-box/private/checkpoints/S000001/V00.json": b"automatic\n",
        "representation/private/score-shards/S000001.jsonl": b"scores\n",
        "representation/cache/S000001.complete.json": b"record\n",
        "representation/cache/S000001.npz": b"arrays\n",
    }
    for name, content in files.items():
        path = output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    recovery_root = tmp_path / "recovery"
    assert MODULE.save_recovery(output_root, recovery_root) == len(files)
    archive_path = recovery_root / MODULE.RECOVERY_ARCHIVE_NAME
    with zipfile.ZipFile(archive_path) as archive:
        record = json.loads(archive.read("recovery_record.json").decode("utf-8"))
    assert record["file_count"] == len(files)

    seeded_root = tmp_path / "seeded"
    assert MODULE.seed_recovery(recovery_root, seeded_root) == len(files)
    for name, content in files.items():
        assert (seeded_root / name).read_bytes() == content


def test_recovery_save_removes_stale_stage_files(tmp_path: Path) -> None:
    recovery_root = tmp_path / "recovery"
    stale_output = tmp_path / "stale-run"
    stale = stale_output / "sam21/reference-box/private/checkpoints/S999999/V19.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    MODULE.save_recovery(stale_output, recovery_root)

    output_root = tmp_path / "run"
    current = output_root / "sam21/reference-box/private/checkpoints/S000001/V00.json"
    current.parent.mkdir(parents=True)
    current.write_text("current\n", encoding="utf-8")

    MODULE.save_recovery(output_root, recovery_root)
    seeded_root = tmp_path / "seeded"
    MODULE.seed_recovery(recovery_root, seeded_root)

    assert (seeded_root / current.relative_to(output_root)).read_text(encoding="utf-8") == "current\n"
    assert not (seeded_root / stale.relative_to(stale_output)).exists()


def test_recovery_seed_rejects_changed_checkpoint(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    source = output_root / "locateanything/private/raw_proposal_history.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("proposal\n", encoding="utf-8")
    recovery_root = tmp_path / "recovery"
    MODULE.save_recovery(output_root, recovery_root)
    archive_path = recovery_root / MODULE.RECOVERY_ARCHIVE_NAME
    changed_path = recovery_root / "changed.zip"
    with zipfile.ZipFile(archive_path) as source_archive, zipfile.ZipFile(changed_path, "w") as changed:
        for info in source_archive.infolist():
            value = source_archive.read(info.filename)
            if info.filename == source.relative_to(output_root).as_posix():
                value = b"changed\n"
            changed.writestr(info.filename, value)
    changed_path.replace(archive_path)

    with pytest.raises(ValueError, match="integrity"):
        MODULE.seed_recovery(recovery_root, tmp_path / "seeded")


def test_interrupted_recovery_save_keeps_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    source = output_root / "locateanything/private/raw_proposal_history.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("first\n", encoding="utf-8")
    recovery_root = tmp_path / "recovery"
    MODULE.save_recovery(output_root, recovery_root)
    archive_path = recovery_root / MODULE.RECOVERY_ARCHIVE_NAME
    original_bytes = archive_path.read_bytes()
    source.write_text("second\n", encoding="utf-8")

    def fail_before_replace(_output_root: Path, temporary: Path) -> int:
        temporary.write_bytes(b"partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr(MODULE, "_write_recovery_archive", fail_before_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        MODULE.save_recovery(output_root, recovery_root)

    assert archive_path.read_bytes() == original_bytes
    seeded_root = tmp_path / "seeded"
    MODULE.seed_recovery(recovery_root, seeded_root)
    assert (seeded_root / source.relative_to(output_root)).read_text(encoding="utf-8") == "first\n"


def test_representation_progress_can_checkpoint_before_stage_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def record_save(output_root: Path, recovery_root: Path | None) -> int:
        calls.append((output_root, recovery_root))
        return 0

    monkeypatch.setattr(MODULE, "save_recovery", record_save)
    output_root = tmp_path / "run"
    recovery_root = tmp_path / "recovery"
    callback = MODULE.recovery_progress_callback(
        output_root=output_root,
        recovery_root=recovery_root,
        label="DS2 representation scenes",
        recovery_interval=25,
    )

    callback(24, 300, {})
    assert calls == []
    callback(25, 300, {})
    assert calls == [(output_root, recovery_root)]
    callback(300, 300, {})
    assert calls == [(output_root, recovery_root), (output_root, recovery_root)]


def test_recovery_seed_rejects_unsupported_path(tmp_path: Path) -> None:
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    unsupported = recovery_root / "evaluator/labels.json"
    unsupported.parent.mkdir()
    unsupported.write_text("labels\n", encoding="utf-8")
    (recovery_root / "recovery_record.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "file_count": 1,
                "files": {"evaluator/labels.json": MODULE.file_sha256(unsupported)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported"):
        MODULE.seed_recovery(recovery_root, tmp_path / "seeded")
