from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from visual_peer_inspector.notebook_provenance import (
    ENVIRONMENT_KEY,
    runtime_code_provenance,
    verify_notebook_code_snapshot,
)
from visual_peer_inspector.tracking_safe_artifacts import write_tracking_safe_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_every_remaining_notebook_verifies_and_retains_snapshot_provenance() -> None:
    module_path = ROOT / "scripts/validate_remaining_notebooks.py"
    specification = importlib.util.spec_from_file_location(
        "remaining_notebook_validator", module_path
    )
    assert specification is not None and specification.loader is not None
    validator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(validator)

    paths = validator.notebook_paths(ROOT)
    assert set(paths) == set(validator.EXPECTED)
    for stage, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        assert source.count("verify_notebook_code_snapshot(") == 1, stage
        assert "CODE_PROVENANCE = verify_notebook_code_snapshot(" in source, stage
        assert source.index("CODE_ROOT = find_code_root()") < source.index(
            "verify_notebook_code_snapshot("
        ), stage


def test_notebook_snapshot_helper_verifies_exact_notebook_and_exports_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook_relative = "notebooks/example.ipynb"
    notebook_path = tmp_path / notebook_relative
    module_path = tmp_path / "src/visual_peer_inspector/example.py"
    notebook_path.parent.mkdir(parents=True)
    module_path.parent.mkdir(parents=True)
    notebook_path.write_bytes(b"notebook")
    module_path.write_bytes(b"module")
    files = {
        notebook_relative: sha256(notebook_path.read_bytes()).hexdigest(),
        "src/visual_peer_inspector/example.py": sha256(module_path.read_bytes()).hexdigest(),
    }
    aggregate = sha256(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())).encode()
    ).hexdigest()
    (tmp_path / "snapshot_record.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "git_commit": "a" * 40,
                "git_dirty": False,
                "precommit_candidate": False,
                "snapshot_sha256": aggregate,
                "file_count": len(files),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv(ENVIRONMENT_KEY, raising=False)

    provenance = verify_notebook_code_snapshot(tmp_path, notebook_relative)

    assert provenance == {
        "snapshot_sha256": aggregate,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "notebook_sha256": files[notebook_relative],
    }
    assert runtime_code_provenance() == provenance


def test_safe_run_record_is_bound_before_manifest_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "snapshot_sha256": "b" * 64,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "notebook_sha256": "c" * 64,
    }
    monkeypatch.setenv(ENVIRONMENT_KEY, json.dumps(provenance))
    run_path = tmp_path / "run_record.json"
    run_path.write_text(
        json.dumps({"schema_version": 1, "status": "completed", "stage": "example"}),
        encoding="utf-8",
    )

    manifest = write_tracking_safe_manifest(tmp_path)

    assert json.loads(run_path.read_text(encoding="utf-8")) | provenance == json.loads(
        run_path.read_text(encoding="utf-8")
    )
    run_row = next(row for row in manifest["artifacts"] if row["path"] == "run_record.json")
    assert run_row["sha256"] == sha256(run_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    (("git_dirty", True), ("precommit_candidate", True)),
)
def test_notebook_snapshot_helper_rejects_unpinned_provenance(
    tmp_path: Path, field: str, value: bool
) -> None:
    notebook_relative = "notebooks/example.ipynb"
    notebook_path = tmp_path / notebook_relative
    notebook_path.parent.mkdir(parents=True)
    notebook_path.write_bytes(b"notebook")
    digest = sha256(notebook_path.read_bytes()).hexdigest()
    record = {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "precommit_candidate": False,
        "snapshot_sha256": sha256(f"{digest}  {notebook_relative}\n".encode()).hexdigest(),
        "file_count": 1,
        "files": {notebook_relative: digest},
    }
    record[field] = value
    (tmp_path / "snapshot_record.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError):
        verify_notebook_code_snapshot(tmp_path, notebook_relative)
