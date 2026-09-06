from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from visual_peer_inspector.verification import verify_code_snapshot_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "build_kaggle_evaluation_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_kaggle_evaluation_snapshot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
snapshot_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_builder)


def test_clean_snapshot_contains_complete_ds1_execution_source(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_builder, "REPOSITORY_ROOT", REPOSITORY_ROOT)
    monkeypatch.setattr(snapshot_builder, "_git_state", lambda: ("a" * 40, False))
    output = snapshot_builder.build_snapshot(
        output_root=tmp_path / "snapshot",
        allow_dirty=False,
        code_dataset_version="12",
        notebook_version_05="8",
        notebook_version_06="6",
        notebook_version_07="4",
        notebook_version_03_1="1",
        notebook_version_04_1="1",
        notebook_version_05_1="1",
        notebook_version_06_1="1",
        notebook_version_04_3="candidate",
        notebook_version_08a="candidate",
        notebook_version_08b="candidate",
        notebook_version_08c="candidate",
        notebook_version_08d="candidate",
        notebook_version_08e="candidate",
        notebook_version_08e1="candidate",
        notebook_version_08f="candidate",
        notebook_version_08f1="candidate",
        notebook_version_09a="candidate",
        notebook_version_09b="candidate",
        notebook_version_09b1="candidate",
        notebook_version_09c="candidate",
        notebook_version_09d="candidate",
        notebook_version_10r="candidate",
        notebook_version_10a="candidate",
        notebook_version_10b="candidate",
        notebook_version_11r="candidate",
        notebook_version_11a="candidate",
        notebook_version_11b="candidate",
    )
    code_root = output / "code"
    record = snapshot_builder.verify_snapshot_record(code_root)
    assert verify_code_snapshot_record(code_root, code_root / "snapshot_record.json") == record
    assert record["git_dirty"] is False
    assert record["precommit_candidate"] is False
    assert record["platform_versions"]["kaggle_notebook_version_06_1"] == "1"

    recorded = set(record["files"])
    expected_ds1 = {path.as_posix() for path in snapshot_builder.DS1_NOTEBOOK_PATHS}
    assert expected_ds1.issubset(recorded)
    for relative in expected_ds1:
        assert (code_root / relative).is_file()
    assert record["platform_versions"]["kaggle_notebook_version_04_3"] == "candidate"

    expected_ds2 = {path.as_posix() for path in snapshot_builder.DS2_NOTEBOOK_PATHS}
    assert expected_ds2.issubset(recorded)
    assert record["platform_versions"]["kaggle_notebook_version_08c"] == "candidate"
    for relative in expected_ds2:
        assert (code_root / relative).is_file()

    expected_remaining = {
        path.as_posix() for path in snapshot_builder.REMAINING_NOTEBOOK_PATHS
    }
    assert expected_remaining.issubset(recorded)
    assert record["platform_versions"]["kaggle_notebook_version_08d"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_08e1"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_08f1"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_09b1"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_10b"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_11r"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_11a"] == "candidate"
    assert record["platform_versions"]["kaggle_notebook_version_11b"] == "candidate"
    for relative in expected_remaining:
        assert (code_root / relative).is_file()

    official_remaining = {
        path.as_posix()
        for path in snapshot_builder.REMAINING_NOTEBOOK_PATHS
        if path.parts[:2] == ("notebooks", "official-evaluation")
    }
    assert official_remaining == {
        "notebooks/official-evaluation/10r-toysad8k-official-release.ipynb",
        (
            "notebooks/official-evaluation/"
            "10a-toysad8k-official-seen-frozen-inference.ipynb"
        ),
        (
            "notebooks/official-evaluation/"
            "10a1-toysad8k-official-unseen-frozen-inference.ipynb"
        ),
        (
            "notebooks/official-evaluation/"
            "10a-finalize-toysad8k-official-prediction-summary.ipynb"
        ),
        (
            "notebooks/official-evaluation/"
            "10b-toysad8k-official-seen-unseen-evaluation.ipynb"
        ),
    }

    expected_source = set()
    for source_relative in snapshot_builder.SOURCE_ROOTS:
        source_root = REPOSITORY_ROOT / source_relative
        expected_source.update(
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in snapshot_builder._source_files(
                source_root,
                snapshot_builder.ALLOWED_SUFFIXES[source_relative],
            )
        )
    assert expected_source.issubset(recorded)
    assert any(name.startswith("src/visual_peer_inspector/") for name in recorded)
    assert any(name.startswith("scripts/") for name in recorded)
    assert any(name.startswith("requirements/") for name in recorded)
    assert any(name.startswith("tests/") for name in recorded)

    actual = {
        path.relative_to(code_root).as_posix()
        for path in code_root.rglob("*")
        if path.is_file()
    }
    assert actual == recorded | {"snapshot_record.json"}
    for name in recorded:
        path = Path(name)
        lowered_parts = {part.lower() for part in path.parts}
        assert not lowered_parts.intersection(snapshot_builder.FORBIDDEN_PATH_PARTS)
        assert path.name.lower() not in snapshot_builder.FORBIDDEN_FILE_NAMES
        assert path.suffix in {".py", ".txt", ".ipynb"}
    assert not any("__pycache__" in Path(name).parts for name in recorded)
    assert not any(Path(name).suffix in {".env", ".key", ".pem", ".sqlite", ".zip"} for name in recorded)

    metadata = json.loads((output / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["isPrivate"] is True


def test_snapshot_verifier_rejects_changed_and_undeclared_files(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot_builder, "REPOSITORY_ROOT", REPOSITORY_ROOT)
    monkeypatch.setattr(snapshot_builder, "_git_state", lambda: ("b" * 40, False))
    code_root = snapshot_builder.build_snapshot(
        output_root=tmp_path / "snapshot",
        allow_dirty=False,
        code_dataset_version="candidate",
        notebook_version_05="pending",
        notebook_version_06="pending",
        notebook_version_07="pending",
    ) / "code"
    record = json.loads((code_root / "snapshot_record.json").read_text(encoding="utf-8"))
    first = code_root / min(record["files"])
    original = first.read_bytes()
    first.write_bytes(original + b"\n")
    try:
        snapshot_builder.verify_snapshot_record(code_root)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Changed snapshot file was accepted")
    first.write_bytes(original)
    (code_root / "credential.env").write_text("TOKEN=not-real\n", encoding="utf-8")
    try:
        snapshot_builder.verify_snapshot_record(code_root)
    except ValueError as error:
        assert "undeclared" in str(error)
    else:
        raise AssertionError("Undeclared snapshot file was accepted")
    try:
        verify_code_snapshot_record(code_root, code_root / "snapshot_record.json")
    except ValueError as error:
        assert "undeclared" in str(error)
    else:
        raise AssertionError("Consumer snapshot verifier accepted an undeclared file")
