from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_remaining_notebooks.py"
SPEC = importlib.util.spec_from_file_location("notebook_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_notebook(path: Path, markdown: str, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": markdown},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": (
                    "CODE_ROOT = find_code_root()\n"
                    "CODE_PROVENANCE = verify_notebook_code_snapshot(CODE_ROOT, 'notebooks/example.ipynb')\n"
                    + code
                ),
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_metadata(
    path: Path,
    *,
    code_file: str,
    kernel_sources: list[str],
    dataset_sources: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "esreekarreddy/example",
                "code_file": code_file,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "kernel_sources": kernel_sources,
                "dataset_sources": dataset_sources or [],
            }
        ),
        encoding="utf-8",
    )


class ResearchNotebookValidatorTests(unittest.TestCase):
    def test_discovery_ignores_clean_sources_saved_with_executed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "notebooks/development-freeze/08d-current.ipynb"
            saved = root / "notebooks/executed/development-freeze/08d-kaggle-v2.ipynb"
            for path in (current, saved, saved.with_name("08d-kaggle-v2-executed.ipynb")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            self.assertEqual(validator.notebook_paths(root), {"08d": current})

    def test_accepts_clean_stage_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "08d-example.ipynb"
            write_notebook(
                path,
                "# Release\n\n08d-example.ipynb remaining 600 development calibration "
                "internal-lock ClearML hash failure",
                "status = 'ready'\n",
            )
            self.assertEqual(validator.validate_notebook("08d", path), [])

    def test_rejects_output_secret_and_legacy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "11b-example.ipynb"
            write_notebook(
                path,
                "# Review\n\n11b-example.ipynb full-pool sensitivity official ClearML hash failure",
                "CLEARML_API_SECRET_KEY = 'unsafe'\nacceptance_recall = 0.95\n",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cells"][1]["execution_count"] = 1
            payload["cells"][1]["outputs"] = [{"output_type": "stream", "text": "x"}]
            path.write_text(json.dumps(payload), encoding="utf-8")

            errors = validator.validate_notebook("11b", path)

            self.assertTrue(any("hard-coded" in error for error in errors))
            self.assertTrue(any("legacy 0.95" in error for error in errors))
            self.assertTrue(any("execution count" in error for error in errors))
            self.assertTrue(any("saved output" in error for error in errors))

    def test_rejects_missing_or_late_snapshot_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "08d-example.ipynb"
            write_notebook(
                path,
                "# Release\n\n08d-example.ipynb remaining 600 development calibration "
                "internal-lock ClearML hash failure",
                "status = 'ready'\n",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cells"][1]["source"] = "CODE_ROOT = find_code_root()\nstatus = 'ready'\n"
            path.write_text(json.dumps(payload), encoding="utf-8")

            errors = validator.validate_notebook("08d", path)

            self.assertTrue(any("snapshot verifier" in error for error in errors))
            self.assertTrue(any("provenance is not retained" in error for error in errors))

    def test_metadata_rejects_a_stale_upstream_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / "11b-example.ipynb"
            write_notebook(
                notebook,
                "# Review\n\n11b-example.ipynb full-pool sensitivity official ClearML hash failure",
                "status = 'ready'\n",
            )
            metadata = root / "kernel-metadata-11b.json"
            write_metadata(
                metadata,
                code_file=notebook.name,
                kernel_sources=["esreekarreddy/toysad-8k-development-method-freeze"],
            )

            errors = validator.validate_metadata("11b", notebook)

            self.assertTrue(any("exact dependency contract" in error for error in errors))

    def test_long_stage_requires_cross_version_recovery_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "11a-example.ipynb"
            write_notebook(
                path,
                "# Official\n\n11a-example.ipynb full-pool frozen prediction ClearML hash failure checkpoint",
                "status = 'ready'\n",
            )

            errors = validator.validate_notebook("11a", path)

            self.assertTrue(any("incremental" in error for error in errors))
            self.assertTrue(any("recovery" in error for error in errors))

    def test_resource_stage_names_and_dataset_graph_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook_root = root / "notebooks" / "protected-evaluation"
            notebook = notebook_root / "11r-example.ipynb"
            write_notebook(
                notebook,
                "# Full-pool resource\n\n11r-example.ipynb full-pool post-freeze "
                "evaluator ClearML hash failure checkpoint "
                "incremental recovery",
                "status = 'ready'\n",
            )
            metadata = notebook_root / "kernel-metadata-11r.json"
            expected_datasets = list(validator.EXPECTED_DATASET_SOURCES["11r"])
            write_metadata(
                metadata,
                code_file=notebook.name,
                kernel_sources=list(validator.EXPECTED_KERNEL_SOURCES["11r"]),
                dataset_sources=expected_datasets,
            )

            found = validator.notebook_paths(root)

            self.assertEqual(found, {"11r": notebook})
            self.assertEqual(validator.validate_notebook("11r", notebook), [])
            self.assertEqual(validator.validate_metadata("11r", notebook), [])
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["dataset_sources"] = expected_datasets[:-1]
            metadata.write_text(json.dumps(value), encoding="utf-8")
            errors = validator.validate_metadata("11r", notebook)
            self.assertTrue(any("dataset_sources" in error for error in errors))

    def test_optional_full_pool_stages_are_in_the_shared_contract(self) -> None:
        self.assertEqual(
            validator.EXPECTED_KERNEL_SOURCES["11r"],
            ("esreekarreddy/toysad-8k-official-seen-and-unseen-test-evaluation",),
        )
        self.assertIn("11a", validator.GPU_STAGES)
        self.assertIn("11r", validator.LONG_STAGE_TOPICS)
        self.assertIn("11a", validator.LONG_STAGE_TOPICS)
        self.assertEqual(len(validator.EXPECTED), 18)


if __name__ == "__main__":
    unittest.main()
