from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "finalize_locateanything_ds1_clearml",
    CODE_ROOT / "scripts/finalize_locateanything_ds1_clearml.py",
)
assert SPEC and SPEC.loader
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


class Ds1LocateAnythingClearmlFinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "safe"
        self.root.mkdir()
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(self, name: str, columns: list[str]) -> None:
        (self.root / name).write_text(
            ",".join(columns) + "\n" + ",".join("0" for _ in columns) + "\n",
            encoding="utf-8",
        )

    def _run_record(self) -> dict:
        return {
            "schema_version": 1,
            "stage": "ds1_locateanything_screen",
            "status": "completed",
            "tracking_status": "pending",
            "run_mode": "model_inference",
            "inference_stage": "full",
            "producer_kind": "locateanything_model",
            "cohort": {
                "cohort_id": FINALIZER.EXPECTED_COHORT_ID,
                "release_id": "toysad8k-train-2000-v2",
                "role": "development",
                "pilot_stage": "screen_100",
                "membership_sha256": FINALIZER.EXPECTED_MEMBERSHIP_SHA256,
                "scene_count": 100,
                "views_per_scene": 20,
                "image_count": 2_000,
                "asset_group_count": 27,
            },
            "configuration": {
                "model_id": FINALIZER.EXPECTED_MODEL_ID,
                "model_revision": FINALIZER.EXPECTED_MODEL_REVISION,
                "prompt": "object",
                "seed": 32_040,
                "generation_mode": "hybrid",
                "max_new_tokens": 8_192,
            },
            "retry_policy": {
                "execution_policy_sha256": FINALIZER.EXECUTION_POLICY_SHA256,
                "execution_policy": FINALIZER.locateanything_execution_policy(),
            },
            "generation_mode_counts": {"hybrid": 2_000},
            "provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "precommit_candidate": False,
                "snapshot_sha256": "b" * 64,
                "notebook_sha256": "c" * 64,
                "kaggle_notebook_version_04_1": "2",
                "code_dataset_version": "32",
            },
            "output_identity": {
                "result_filename": "proposal_results.jsonl",
                "result_sha256": "d" * 64,
                "proposal_results_sha256": "d" * 64,
                "result_count": 2_000,
            },
            "summary": {
                "job_count": 2_000,
                "successful_job_count": 2_000,
                "failed_job_count": 0,
                "possibly_truncated_count": 0,
                "zero_proposal_image_count": 0,
            },
            "one_image_preflight": {
                "job_count": 1,
                "successful_job_count": 1,
                "failed_job_count": 0,
            },
            "pilot_summary": {
                "job_count": 20,
                "successful_job_count": 20,
                "failed_job_count": 0,
                "scene_count": 20,
            },
            "repeat_agreement": {
                "job_count": 20,
                "matching_job_count": 20,
                "box_agreement_rate": 1.0,
            },
            "completion_gate_passed": True,
        }

    def _write_manifest(self) -> None:
        rows = []
        for path in sorted(item for item in self.root.iterdir() if item.is_file()):
            if path.name == "artifact_manifest.json":
                continue
            rows.append(
                {
                    "path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        (self.root / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_count": len(rows),
                    "artifacts": rows,
                    "manifest_content_sha256": sha256(
                        b"".join(
                            json.dumps(row, separators=(",", ":"), sort_keys=True).encode("utf-8")
                            + b"\n"
                            for row in rows
                        )
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )

    def _write_fixture(self) -> None:
        for name, columns in FINALIZER.ALLOWED_TABLES.items():
            self._write_csv(name, columns)
        (self.root / "attempt_summary.csv").write_text(
            "attempt_count,image_count\n1,2000\n", encoding="utf-8"
        )
        (self.root / "completion_gates.csv").write_text(
            "gate,status,observed,expected\nfull coverage,pass,2000,2000\n",
            encoding="utf-8",
        )
        (self.root / "generation_mode_summary.csv").write_text(
            "generation_mode,attempt_count\nhybrid,2000\n", encoding="utf-8"
        )
        generic = {
            "box_geometry_summary.json": {"box_count": 8_000},
            "environment_summary.json": {
                "python_version": "3.12",
                "platform_versions": {
                    "python": "3.12",
                    "pillow": "11.1.0",
                    "numpy": "1.25.0",
                    "transformers": "4.57.1",
                    "torch": "2.7.0",
                    "torchvision": "0.22.0",
                    "cuda_runtime": "12.6",
                    "cudnn": "90501",
                    "kaggle_container_image": "test",
                    "kaggle_kernel_run_type": "batch",
                    "kaggle_notebook_version_04_1": "2",
                    "code_dataset_version": "32",
                },
            },
            "execution_policy.json": FINALIZER.locateanything_execution_policy(),
            "input_audit.json": {"image_count": 2_000, "all_checks_passed": True},
            "model_load_summary.json": {
                "load_seconds": 80.0,
                "checkpoints": [{"stage": "model_ready"}],
                "disposable_preflight": {"status": "passed"},
            },
            "one_image_preflight.json": {"successful_job_count": 1},
            "pilot_summary.json": {"successful_job_count": 20},
            "proposal_summary.json": {"successful_job_count": 2_000},
            "repeat_agreement.json": {"box_agreement_rate": 1.0},
            "resolved_configuration.json": {
                "model_id": FINALIZER.EXPECTED_MODEL_ID,
                "model_revision": FINALIZER.EXPECTED_MODEL_REVISION,
                "prompt": "object",
                "seed": 32_040,
            },
            "runtime_summary.json": {"median_image_runtime_seconds": 0.4},
            "run_record.json": self._run_record(),
        }
        for name, value in generic.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        for name in FINALIZER.ALLOWED_CHARTS:
            Image.new("RGB", (8, 8), "white").save(self.root / name)
        self._write_manifest()

    def test_complete_safe_root_passes(self) -> None:
        record = FINALIZER.validate_safe_root(self.root)
        self.assertEqual(record["status"], "completed")

    def test_contract_run_is_rejected(self) -> None:
        record = self._run_record()
        record["run_mode"] = "contract_smoke"
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "completed full DS1 model run"):
            FINALIZER.validate_safe_root(self.root)

    def test_private_identifier_is_rejected(self) -> None:
        path = self.root / "failure_summary.csv"
        path.write_text("error_type,count\nS000001,1\n", encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.root)

    def test_manifest_tampering_is_rejected(self) -> None:
        chart = self.root / next(iter(FINALIZER.ALLOWED_CHARTS))
        chart.write_bytes(chart.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            FINALIZER.validate_safe_root(self.root)

    def test_failed_pilot_is_rejected(self) -> None:
        record = self._run_record()
        record["pilot_summary"]["successful_job_count"] = 19
        record["pilot_summary"]["failed_job_count"] = 1
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "20-image pilot"):
            FINALIZER.validate_safe_root(self.root)

    def test_dirty_provenance_is_rejected(self) -> None:
        record = self._run_record()
        record["provenance"]["git_dirty"] = True
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "canonical provenance"):
            FINALIZER.validate_safe_root(self.root)

    def test_version_provenance_mismatch_is_rejected(self) -> None:
        record = self._run_record()
        record["provenance"]["code_dataset_version"] = "31"
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "version provenance"):
            FINALIZER.validate_safe_root(self.root)

    def test_incomplete_retry_evidence_is_rejected(self) -> None:
        (self.root / "attempt_summary.csv").write_text(
            "attempt_count,image_count\n1,1999\n", encoding="utf-8"
        )
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "retry evidence"):
            FINALIZER.validate_safe_root(self.root)

    def test_uncommitted_proposal_identity_is_rejected(self) -> None:
        record = self._run_record()
        record["output_identity"]["result_sha256"] = "not-a-digest"
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "proposal-result identity"):
            FINALIZER.validate_safe_root(self.root)

    def test_inconsistent_named_proposal_digest_is_rejected(self) -> None:
        record = self._run_record()
        record["output_identity"]["proposal_results_sha256"] = "e" * 64
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "proposal-result identity"):
            FINALIZER.validate_safe_root(self.root)

    def test_notebook_leaves_tracking_setup_to_bounded_child(self) -> None:
        notebook_path = CODE_ROOT / (
            "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
            "04.1-toysad8k-ds1-locateanything-screen.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        tracking_cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "finalize_locateanything_ds1_clearml.py" in "".join(cell["source"])
        )
        self.assertNotIn("pip\", \"install", tracking_cell)
        self.assertNotIn("UserSecretsClient", tracking_cell)
        self.assertIn("subprocess.run", tracking_cell)
        self.assertIn("timeout=210", tracking_cell)


if __name__ == "__main__":
    unittest.main()
