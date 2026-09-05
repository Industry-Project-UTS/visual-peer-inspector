from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_week8_transfer_package.py"
SPEC = importlib.util.spec_from_file_location("build_week8_transfer_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_rows_sha256(rows: list[dict[str, object]]) -> str:
    return sha256(
        b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            for row in rows
        )
    ).hexdigest()


def _clean_notebook() -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["value = 1\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _executed_notebook() -> dict[str, object]:
    notebook = _clean_notebook()
    cell = notebook["cells"][0]
    assert isinstance(cell, dict)
    cell["execution_count"] = 1
    cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["done\n"]}]
    return notebook


def _artifact_tree(root: Path, number: int) -> list[Path]:
    root.mkdir(parents=True)
    summary = root / "summary.csv"
    summary.write_text(f"metric,value\ncompleted,{number}\n", encoding="utf-8")
    run_record = root / "run_record.json"
    _write_json(run_record, {"schema_version": 1, "status": "completed", "stage_number": number})
    rows = [
        {
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in (run_record, summary)
    ]
    manifest = root / "artifact_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "manifest_content_sha256": _artifact_rows_sha256(rows),
        },
    )
    return [manifest, run_record, summary]


def _stage_bundle(root: Path, stage: str, safe_root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    safe_files = sorted(path for path in safe_root.rglob("*") if path.is_file())
    repository_archive = root / f"{stage}-repository-evidence.zip"
    with zipfile.ZipFile(repository_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in safe_files:
            archive.write(path, f"safe/{path.relative_to(safe_root).as_posix()}")
    review_archive = root / f"{stage}-review-evidence.zip"
    with zipfile.ZipFile(review_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("clearml_receipt.json", '{"status":"completed"}\n')
    downstream_count = int(
        stage
        in {
            "ds1-locateanything-prompt-freeze",
            "ds2-reference-sam21",
            "ds2-automatic-sam21",
            "ds2-dinov2-p1",
        }
    )
    downstream_archive = root / f"{stage}-downstream-input.zip"
    if downstream_count:
        with zipfile.ZipFile(
            downstream_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("downstream/input.json", '{"status":"completed"}\n')
    archive_paths = [repository_archive, review_archive]
    if downstream_count:
        archive_paths.append(downstream_archive)
    verification_path = root / f"{stage}-verification.json"
    verification = {
        "schema_version": 1,
        "stage": stage,
        "status": "completed",
        "safe_manifest_sha256": sha256(
            (safe_root / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
        "safe_file_count": len(safe_files),
        "review_file_count": 1,
        "downstream_file_count": downstream_count,
        "archives": [
            {
                "path": path.name,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "source_file_count": (
                    len(safe_files) if path == repository_archive else 1
                ),
            }
            for path in archive_paths
        ],
    }
    _write_json(verification_path, verification)
    checksums_path = root / "SHA256SUMS"
    checksums_path.write_text(
        "".join(
            f"{sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted((*archive_paths, verification_path), key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    return verification_path, checksums_path


def _fixture(root: Path) -> tuple[Path, Path, dict[str, Path]]:
    sources = root / "sources"
    repository = sources / "repository"
    evidence = sources / "evidence"
    review = sources / "review"
    control = sources / "control"
    for path in (repository, evidence, review, control):
        path.mkdir(parents=True)

    repository_entries = []
    review_entries = []
    for name in sorted(builder.REQUIRED_NOTEBOOKS):
        source = Path("notebooks") / name
        _write_json(repository / source, _clean_notebook())
        repository_entries.append(
            {"root": "repository", "source": source.as_posix(), "destination": source.as_posix()}
        )
        review_source = Path("executed") / name
        _write_json(review / review_source, _executed_notebook())
        review_entries.append(
            {
                "root": "review",
                "source": review_source.as_posix(),
                "destination": (Path("executed-notebooks") / name).as_posix(),
            }
        )

    stage_paths: dict[str, Path] = {}
    required_stages = []
    for number, stage in enumerate(sorted(builder.REQUIRED_STAGES), start=1):
        safe_root = evidence / "safe" / stage
        for path in _artifact_tree(safe_root, number):
            repository_entries.append(
                {
                    "root": "evidence",
                    "source": path.relative_to(evidence).as_posix(),
                    "destination": (Path("reports") / stage / path.name).as_posix(),
                }
            )
        bundle_root = evidence / "bundles" / stage
        verification, checksums = _stage_bundle(bundle_root, stage, safe_root)
        stage_paths[stage] = bundle_root
        required_stages.append(
            {
                "stage": stage,
                "root": "evidence",
                "bundle_root": bundle_root.relative_to(evidence).as_posix(),
            }
        )
        for source in (verification, checksums):
            review_entries.append(
                {
                    "root": "evidence",
                    "source": source.relative_to(evidence).as_posix(),
                    "destination": (
                        Path("stage-verification") / stage / source.name
                    ).as_posix(),
                }
            )

    verification_note = review / "VERIFICATION.md"
    verification_note.write_text("# Verification\n\nAll seven stages were checked.\n", encoding="utf-8")
    review_entries.append(
        {
            "root": "review",
            "source": "VERIFICATION.md",
            "destination": "VERIFICATION.md",
        }
    )
    instructions = control / "upload.md"
    instructions.write_text(
        "# Upload steps\n\n"
        "Copy only `repository-files/`.\n\n"
        "Do not copy `review-only/`.\n\n"
        "Do not add a co-author line.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "package_name": builder.PACKAGE_NAME,
        "source_roots": {
            "repository": str(repository),
            "evidence": str(evidence),
            "review": str(review),
            "control": str(control),
        },
        "instructions": {"root": "control", "source": "upload.md"},
        "required_stages": required_stages,
        "repository_files": repository_entries,
        "review_files": review_entries,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, evidence, stage_paths


class Week8TransferPackageTests(unittest.TestCase):
    def test_builds_deterministic_verified_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _evidence, _stages = _fixture(root)
            first = builder.build_transfer_package(
                manifest_path=manifest,
                output_parent=root / "output-one",
            )
            second = builder.build_transfer_package(
                manifest_path=manifest,
                output_parent=root / "output-two",
            )
            self.assertEqual(first["stage_count"], len(builder.REQUIRED_STAGES))
            self.assertEqual(first["zip_sha256"], second["zip_sha256"])
            package = root / "output-one" / builder.PACKAGE_NAME
            builder.verify_transfer_tree(package)
            with zipfile.ZipFile(root / "output-one" / f"{builder.PACKAGE_NAME}.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(
                    all(name.startswith(f"{builder.PACKAGE_NAME}/") for name in archive.namelist())
                )

    def test_refuses_to_overwrite_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _evidence, _stages = _fixture(root)
            output = root / "output"
            builder.build_transfer_package(manifest_path=manifest, output_parent=output)
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                builder.build_transfer_package(manifest_path=manifest, output_parent=output)

    def test_rejects_saved_output_in_repository_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, _stages = _fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            notebook_entry = next(
                row for row in manifest["repository_files"] if row["source"].endswith(".ipynb")
            )
            notebook = Path(manifest["source_roots"][notebook_entry["root"]]) / notebook_entry["source"]
            _write_json(notebook, _executed_notebook())
            with self.assertRaisesRegex(builder.TransferPackageError, "contains saved output"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_credential_literal_in_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, _stages = _fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            notebook_entry = next(
                row for row in manifest["repository_files"] if row["source"].endswith(".ipynb")
            )
            notebook_path = (
                Path(manifest["source_roots"][notebook_entry["root"]])
                / notebook_entry["source"]
            )
            notebook = _clean_notebook()
            cell = notebook["cells"][0]
            assert isinstance(cell, dict)
            cell["source"] = ["api_" + 'secret_key = "123456789abcdef"\n']
            _write_json(notebook_path, notebook)
            with self.assertRaisesRegex(builder.TransferPackageError, "Possible credential"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_review_notebook_without_execution_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, _stages = _fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            notebook_entry = next(
                row for row in manifest["review_files"] if row["source"].endswith(".ipynb")
            )
            notebook = (
                Path(manifest["source_roots"][notebook_entry["root"]])
                / notebook_entry["source"]
            )
            _write_json(notebook, _clean_notebook())
            with self.assertRaisesRegex(builder.TransferPackageError, "no saved execution evidence"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_executed_notebook_outside_review_execution_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, _stages = _fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            notebook_entry = next(
                row for row in manifest["review_files"] if row["source"].endswith(".ipynb")
            )
            notebook_entry["destination"] = (
                Path("published-source") / Path(notebook_entry["destination"]).name
            ).as_posix()
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                builder.TransferPackageError,
                "under executed-notebooks",
            ):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_repository_evidence_not_linked_to_stage_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, evidence, _stages = _fixture(root)
            safe_root = next((evidence / "safe").iterdir())
            summary = safe_root / "summary.csv"
            summary.write_text("metric,value\ncompleted,99\n", encoding="utf-8")
            manifest = json.loads((safe_root / "artifact_manifest.json").read_text(encoding="utf-8"))
            summary_row = next(row for row in manifest["artifacts"] if row["path"] == "summary.csv")
            summary_row["sha256"] = sha256(summary.read_bytes()).hexdigest()
            summary_row["size_bytes"] = summary.stat().st_size
            manifest["manifest_content_sha256"] = _artifact_rows_sha256(manifest["artifacts"])
            _write_json(safe_root / "artifact_manifest.json", manifest)
            with self.assertRaisesRegex(
                builder.TransferPackageError,
                "does not match the verified stage safe manifests",
            ):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_invalid_artifact_manifest_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, evidence, _stages = _fixture(root)
            artifact_manifest_path = next(
                (evidence / "safe").glob("*/artifact_manifest.json")
            )
            artifact_manifest = json.loads(
                artifact_manifest_path.read_text(encoding="utf-8")
            )
            artifact_manifest["manifest_content_sha256"] = "f" * 64
            _write_json(artifact_manifest_path, artifact_manifest)
            with self.assertRaisesRegex(
                builder.TransferPackageError,
                "Artifact manifest content hash differs",
            ):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_protected_identifier_in_repository_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, evidence, _stages = _fixture(root)
            summary = next((evidence / "safe").glob("*/summary.csv"))
            summary.write_text("scene,value\nS000001,1\n", encoding="utf-8")
            with self.assertRaisesRegex(builder.TransferPackageError, "Protected detail"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_failed_stage_archive_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, stages = _fixture(root)
            stage = sorted(stages)[0]
            archive = stages[stage] / f"{stage}-repository-evidence.zip"
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(builder.TransferPackageError, "size differs"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_unexpected_stage_bundle_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _evidence, stages = _fixture(root)
            (next(iter(stages.values())) / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(builder.TransferPackageError, "unexpected files"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_review_file_in_repository_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, evidence, _stages = _fixture(root)
            receipt = evidence / "clearml_receipt.json"
            _write_json(receipt, {"status": "completed", "task_id": "abc"})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["repository_files"].append(
                {
                    "root": "evidence",
                    "source": "clearml_receipt.json",
                    "destination": "reports/clearml_receipt.json",
                }
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(builder.TransferPackageError, "Review-only file"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )

    def test_rejects_symlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, evidence, _stages = _fixture(root)
            target = evidence / "target.md"
            target.write_text("safe", encoding="utf-8")
            link = evidence / "link.md"
            link.symlink_to(target)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["review_files"].append(
                {
                    "root": "evidence",
                    "source": "link.md",
                    "destination": "link.md",
                }
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(builder.TransferPackageError, "symbolic link"):
                builder.build_transfer_package(
                    manifest_path=manifest_path,
                    output_parent=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
