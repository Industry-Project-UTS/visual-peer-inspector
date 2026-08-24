import csv
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "finalize_mask_evaluation_clearml.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_mask_evaluation_clearml", SCRIPT
)
FINALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FINALIZER)


class MaskEvaluationClearmlValidationTests(unittest.TestCase):
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
            writer.writerow([0 for _ in columns])

    def _run_record(self):
        return {
            "schema_version": 1,
            "stage": "ds0_sam21_mask_evaluation",
            "status": "completed",
            "tracking_status": "pending",
            "evaluation_mode": "model_evidence",
            "evidence_eligible": True,
            "completion_gate_passed": True,
            "scientific_gates_passed": True,
            "quality_claim": "provisional_ds0_integration_evidence",
            "sam_producer_kind": "sam2.1_model",
            "python_version": "3.11",
            "evaluator_provenance": {
                "git_commit": "b" * 40,
                "git_dirty": True,
                "snapshot_sha256": "c" * 64,
                "precommit_candidate": True,
                "notebook_sha256": "d" * 64,
                "kaggle_notebook_slug": (
                    "esreekarreddy/toysad-8k-ds0-proposal-and-mask-evaluation"
                ),
                "kaggle_notebook_version": "runtime",
                "code_dataset_version": "19",
            },
            "settings": {},
            "coverage": {
                "scene_count": 20,
                "view_count_per_scene": 20,
                "image_count": 400,
                "reference_mask_count": 1640,
                "sam_mask_count": 3256,
                "reference_prompt_count": 1640,
                "automatic_prompt_count": 1616,
            },
            "inputs": {"package_sha256": "a" * 64},
            "proposal_gate_summary": {},
            "route_summary": [],
            "bootstrap_summary": [],
            "candidate_decision": {
                "status": "candidate",
                "proposal_git_dirty": True,
                "sam_git_dirty": True,
                "evaluator_git_dirty": True,
                "proceed_to_ds1": True,
                "final_route_acceptance": False,
                "reason": "DS0 gates support the next development stage",
            },
            "ranking_metrics_status": "deferred_to_notebook_07",
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
        (self.root / "metric_definitions.json").write_text(
            json.dumps({"scope": "aggregate DS0 evaluation"}), encoding="utf-8"
        )
        (self.root / "run_record.json").write_text(
            json.dumps(self._run_record()), encoding="utf-8"
        )
        for name in FINALIZER.ALLOWED_CHARTS:
            Image.new("RGB", (4, 4), "white").save(self.root / name)
        self._write_manifest()

    def test_complete_aggregate_root_passes(self):
        record = FINALIZER.validate_safe_root(self.root)
        self.assertTrue(record["evidence_eligible"])

    def test_contract_evidence_is_rejected(self):
        record = self._run_record()
        record["evaluation_mode"] = "contract_test"
        record["evidence_eligible"] = False
        (self.root / "run_record.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "accepted model evidence"):
            FINALIZER.validate_safe_root(self.root)

    def test_private_identifier_is_rejected(self):
        path = self.root / "completion_gates.csv"
        path.write_text(
            "gate,observed,expected,status\nS000001,1,1,pass\n",
            encoding="utf-8",
        )
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.root)

    def test_private_json_field_is_rejected(self):
        path = self.root / "metric_definitions.json"
        path.write_text(json.dumps({"source_id": 17}), encoding="utf-8")
        self._write_manifest()

        with self.assertRaisesRegex(ValueError, "Protected JSON field"):
            FINALIZER.validate_safe_root(self.root)

    def test_chart_symlink_is_rejected(self):
        chart = self.root / next(iter(FINALIZER.ALLOWED_CHARTS))
        private_image = Path(self.temporary.name) / "private.png"
        Image.new("RGB", (4, 4), "red").save(private_image)
        chart.unlink()
        chart.symlink_to(private_image)

        with self.assertRaisesRegex(ValueError, "symbolic links"):
            FINALIZER.validate_safe_root(self.root)

    def test_safe_root_symlink_is_rejected(self):
        symlink = Path(self.temporary.name) / "safe-link"
        symlink.symlink_to(self.root, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "missing or unsafe"):
            FINALIZER.validate_safe_root(symlink)


if __name__ == "__main__":
    unittest.main()
