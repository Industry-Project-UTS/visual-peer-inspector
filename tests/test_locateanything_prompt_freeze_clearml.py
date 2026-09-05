from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_ds1_locateanything_prompt_freeze_clearml",
    ROOT / "scripts/finalize_ds1_locateanything_prompt_freeze_clearml.py",
)
assert SPEC and SPEC.loader
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


class PromptFreezeClearmlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "safe"
        self.root.mkdir()
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_table(self, name: str, columns: list[str], rows: list[dict]) -> None:
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _write_manifest(self) -> None:
        rows = [
            {
                "path": path.name,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(self.root.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        (self.root / "artifact_manifest.json").write_text(
            json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}),
            encoding="utf-8",
        )

    def _write_fixture(self) -> None:
        self._write_table(
            "completion_checks.csv",
            FINALIZER.ALLOWED_TABLES["completion_checks.csv"],
            [{"check": "complete", "passed": True}],
        )
        self._write_table(
            "prompt_operational_summary.csv",
            FINALIZER.ALLOWED_TABLES["prompt_operational_summary.csv"],
            [
                {
                    "prompt_key": "object",
                    "prompt": "object",
                    "source": "reused_baseline",
                    "image_count": 2_000,
                    "successful_image_count": 2_000,
                    "failed_image_count": 0,
                    "mean_proposals_per_image": 4.2,
                    "median_runtime_seconds": 1.2,
                    "p95_runtime_seconds": 2.1,
                    "total_inference_seconds": 2_400,
                }
            ],
        )
        prompt_rows = []
        for prompt_key, prompt in (
            ("object", "object"),
            ("individual_object", "individual object"),
        ):
            for policy in ("raw", "area_nms_075", "area_nms_075_cap8"):
                prompt_rows.append(
                    {
                        column: (
                            f"{prompt_key}__{policy}" if column == "candidate_key"
                            else prompt_key if column == "prompt_key"
                            else prompt if column == "prompt"
                            else policy if column == "postprocessing_policy"
                            else 2_000 if column == "image_count"
                            else 1
                        )
                        for column in FINALIZER.ALLOWED_TABLES["prompt_policy_summary.csv"]
                    }
                )
        self._write_table(
            "prompt_policy_summary.csv",
            FINALIZER.ALLOWED_TABLES["prompt_policy_summary.csv"],
            prompt_rows,
        )
        self._write_table(
            "prompt_repeatability.csv",
            FINALIZER.ALLOWED_TABLES["prompt_repeatability.csv"],
            [{"prompt_key": "object", "prompt": "object", "pilot_image_count": 20, "exact_box_agreement_rate": 1.0}],
        )
        json_values = {
            "model_load_summary.json": {"model_load_count": 1},
            "run_record.json": {
                "stage": FINALIZER.EXPECTED_STAGE,
                "status": "completed",
                "all_completion_checks_passed": True,
                "baseline_reused": True,
                "model_load_count": 1,
                "new_model_prompt_count": 1,
                "image_count": 2_000,
                "selected_candidate": "object__raw",
            },
            "selected_prompt.json": {
                "selected": {"prompt_key": "object", "postprocessing_policy": "raw"},
                "candidate_lock_sha256": "a" * 64,
                "model_revision": "b" * 40,
                "requires_unchanged_ds2_confirmation": True,
            },
            "study_protocol.json": {"baseline_reused": True},
        }
        for name, value in json_values.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        for name in FINALIZER.ALLOWED_CHARTS:
            Image.new("RGB", (8, 8), "white").save(self.root / name)
        self._write_manifest()

    def test_complete_aggregate_root_passes(self) -> None:
        record = FINALIZER.validate_safe_root(self.root)
        self.assertEqual(record["status"], "completed")

    def test_private_identifier_is_rejected(self) -> None:
        path = self.root / "completion_checks.csv"
        path.write_text("check,passed\nS000001,true\n", encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.root)


if __name__ == "__main__":
    unittest.main()
