import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_sam_clearml.py"
SPEC = importlib.util.spec_from_file_location("finalize_sam_clearml", SCRIPT)
FINALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FINALIZER)


class SamClearmlValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "safe"
        self.root.mkdir()
        self._write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_csv(self, name, columns):
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerow(["none" if column.endswith("type") else 0 for column in columns])

    def _run_record(self):
        configuration = {
            "code_repository": "https://github.com/facebookresearch/sam2",
            "code_revision": "2" * 40,
            "model_repository": "facebook/sam2.1-hiera-large",
            "model_revision": "6" * 40,
            "checkpoint_filename": "sam2.1_hiera_large.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_size_bytes": 898083611,
            "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "device": "cuda",
            "precision": "float16",
            "multimask_output": False,
            "mask_threshold": 0.0,
            "max_hole_area": 0.0,
            "max_sprinkle_area": 0.0,
            "seed": 20260824,
            "image_height": 256,
            "image_width": 256,
        }
        return {
            "schema_version": 1,
            "stage": "ds0_sam21_prompted_masks",
            "status": "completed",
            "tracking_status": "pending",
            "run_mode": "model_inference",
            "inference_stage": "full",
            "producer_kind": "sam2.1_model",
            "python_version": "3.11",
            "configuration": configuration,
            "configuration_sha256": "b" * 64,
            "prompt_package_sha256": "c" * 64,
            "checkpoint_sha256": "a" * 64,
            "provenance": {"git_commit": "d" * 40, "git_dirty": True},
            "platform_versions": {"python": "3.11"},
            "gpu": {"cuda_available": True, "device_count": 1},
            "model_load_checkpoints": [],
            "summary": {
                "image_count": 400,
                "scene_count": 20,
                "mask_count": 3256,
                "empty_mask_count": 0,
                "median_image_runtime_seconds": 0.1,
                "p95_image_runtime_seconds": 0.2,
            },
            "reference_prompt_count": 1640,
            "automatic_prompt_count": 1616,
            "repeat_agreement": {
                "mask_digest_agreement_rate": 1.0,
                "score_agreement_rate": 1.0,
            },
            "completion_gate_passed": True,
            "error_type": None,
        }

    def _write_manifest(self):
        rows = []
        for path in sorted(self.root.iterdir()):
            if path.name == "artifact_manifest.json":
                continue
            rows.append(
                {
                    "path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        digest = sha256(
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
                for row in rows
            )
        ).hexdigest()
        (self.root / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_count": len(rows),
                    "artifacts": rows,
                    "manifest_content_sha256": digest,
                }
            ),
            encoding="utf-8",
        )

    def _write_fixture(self):
        for name, columns in FINALIZER.ALLOWED_TABLES.items():
            self._write_csv(name, columns)
        generic = {
            "preflight_gates.json": {"passed": True},
            "repeat_agreement.json": {"mask_digest_agreement_rate": 1.0},
            "runtime_summary.json": {"median_seconds": 0.1},
            "mask_geometry_summary.json": {"route_count": 2},
        }
        for name, value in generic.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        (self.root / "run_record.json").write_text(
            json.dumps(self._run_record()), encoding="utf-8"
        )
        for name in FINALIZER.ALLOWED_CHARTS:
            Image.new("RGB", (4, 4), "white").save(self.root / name)
        self._write_manifest()

    def test_complete_model_safe_root_passes(self):
        FINALIZER.validate_safe_root(self.root)

    def test_contract_run_is_rejected(self):
        record = self._run_record()
        record["run_mode"] = "contract_smoke"
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "completed full model run"):
            FINALIZER.validate_safe_root(self.root)

    def test_chart_symlink_is_rejected(self):
        chart = self.root / next(iter(FINALIZER.ALLOWED_CHARTS))
        private_image = Path(self.temporary.name) / "private.png"
        Image.new("RGB", (4, 4), "red").save(private_image)
        chart.unlink()
        chart.symlink_to(private_image)

        with self.assertRaisesRegex(ValueError, "symbolic links"):
            FINALIZER.validate_safe_root(self.root)

    def test_csv_private_identifier_is_rejected(self):
        path = self.root / "failure_summary.csv"
        path.write_text("error_type,count\nS000001,1\n", encoding="utf-8")
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.root)

    def test_manifest_tampering_is_rejected(self):
        chart = self.root / next(iter(FINALIZER.ALLOWED_CHARTS))
        chart.write_bytes(chart.read_bytes() + b"changed")

        with self.assertRaisesRegex(ValueError, "evidence differs"):
            FINALIZER.validate_safe_root(self.root)


if __name__ == "__main__":
    unittest.main()
