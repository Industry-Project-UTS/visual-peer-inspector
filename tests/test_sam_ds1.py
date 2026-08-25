import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT
from visual_peer_inspector.data.model_safe_package import file_sha256
from visual_peer_inspector.models.sam2 import Sam2Configuration
from visual_peer_inspector.workflows.sam_ds1 import (
    _write_artifact_manifest,
    build_ds1_sam_jobs,
    import_ds1_sam_resume,
    run_ds1_sam_workflow,
)


FINALIZER_PATH = Path(__file__).parents[1] / "scripts" / "finalize_ds1_sam_clearml.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_ds1_sam_clearml", FINALIZER_PATH
)
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
assert FINALIZER_SPEC.loader is not None
FINALIZER_SPEC.loader.exec_module(FINALIZER)


class Ds1SamWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media_root = self.root / "media"
        self.media_root.mkdir()
        self.image_path = self.media_root / "shared.png"
        Image.new("RGB", (256, 256), (120, 130, 140)).save(self.image_path)
        self.prompt_root = self.root / "prompts"
        self.prompt_root.mkdir()
        self.output_root = self.root / "output"
        self._write_prompt_package()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_prompt_package(self):
        image_sha256 = file_sha256(self.image_path)
        rows = []
        for scene_index in range(1, DS1_SCREEN_CONTRACT.scene_count + 1):
            scene_id = f"S{scene_index:06d}"
            for view_index in range(DS1_SCREEN_CONTRACT.views_per_scene):
                rows.append(
                    {
                        "schema_version": 1,
                        "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
                        "opaque_scene_id": scene_id,
                        "view_id": f"V{view_index:02d}",
                        "image_path": "shared.png",
                        "image_sha256": image_sha256,
                        "routes": {
                            "reference_box": [
                                {"prompt_key": "R000", "box_xyxy": [8, 10, 80, 90]}
                            ],
                            "automatic_box": [
                                {"prompt_key": "P000", "box_xyxy": [9, 11, 81, 91]}
                            ],
                        },
                    }
                )
        manifest_path = self.prompt_root / "sam_prompt_manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        upstream = {
            "schema_version": 1,
            "decision": "use_frozen_ds1_proposals",
            "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
            "release_id": DS1_SCREEN_CONTRACT.release_id,
            "pilot_stage": DS1_SCREEN_CONTRACT.pilot_stage,
            "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
            "model_manifest_sha256": "1" * 64,
            "model_package_record_sha256": "2" * 64,
            "geometry_manifest_sha256": "3" * 64,
            "geometry_package_record_sha256": "4" * 64,
            "proposal_results_sha256": "5" * 64,
            "proposal_run_record_sha256": "6" * 64,
            "proposal_artifact_manifest_sha256": "7" * 64,
            "proposal_configuration_sha256": "8" * 64,
            "proposal_model_id": "nvidia/LocateAnything-3B",
            "proposal_model_revision": "frozen-test-revision",
        }
        upstream_path = self.prompt_root / "upstream_decision.json"
        upstream_path.write_text(
            json.dumps(upstream, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_path = self.prompt_root / "content_checksums.sha256"
        checksum_path.write_text(
            f"{file_sha256(manifest_path)}  sam_prompt_manifest.jsonl\n"
            f"{file_sha256(upstream_path)}  upstream_decision.json\n",
            encoding="utf-8",
        )
        record = {
            "schema_version": 1,
            "package_kind": "ds1-sam-prompts",
            "cohort_id": DS1_SCREEN_CONTRACT.cohort_id,
            "release_id": DS1_SCREEN_CONTRACT.release_id,
            "role": DS1_SCREEN_CONTRACT.role,
            "pilot_stage": DS1_SCREEN_CONTRACT.pilot_stage,
            "membership_sha256": DS1_SCREEN_CONTRACT.membership_sha256,
            "scene_count": DS1_SCREEN_CONTRACT.scene_count,
            "views_per_scene": DS1_SCREEN_CONTRACT.views_per_scene,
            "image_count": DS1_SCREEN_CONTRACT.image_count,
            "asset_group_count": DS1_SCREEN_CONTRACT.asset_group_count,
            "reference_prompt_count": DS1_SCREEN_CONTRACT.image_count,
            "automatic_prompt_count": DS1_SCREEN_CONTRACT.image_count,
            "model_manifest_sha256": upstream["model_manifest_sha256"],
            "prompt_manifest_sha256": file_sha256(manifest_path),
            "upstream_decision_sha256": file_sha256(upstream_path),
            "content_checksums_sha256": file_sha256(checksum_path),
            "all_checks_passed": True,
        }
        (self.prompt_root / "package_record.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    def _run(self, progress_callback=None):
        return run_ds1_sam_workflow(
            prompt_root=self.prompt_root,
            media_root=self.media_root,
            output_root=self.output_root,
            configuration=Sam2Configuration(device="cpu", precision="float32"),
            run_mode="contract_smoke",
            inference_stage="full",
            checkpoint_path=None,
            checkpoint_sha256=None,
            provenance={},
            platform_versions={"python": "test"},
            gpu_evidence={"cuda_available": False, "device_count": 0},
            model_load_rows=[],
            progress_callback=progress_callback,
        )

    def test_complete_contract_run_is_resumable(self):
        first = self._run()
        attempt_path = self.output_root / "private" / "attempts.jsonl"
        self.assertTrue(attempt_path.is_file())
        original_output = self.output_root
        self.output_root = self.root / "resumed-output"
        imported = import_ds1_sam_resume(original_output, self.output_root)
        second = self._run()

        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["summary"]["scene_count"], 100)
        self.assertEqual(first["summary"]["image_count"], 2000)
        self.assertEqual(first["summary"]["mask_count"], 4000)
        self.assertEqual(first["repeat_agreement"]["mask_digest_agreement_rate"], 1.0)
        self.assertEqual(second["summary"], first["summary"])
        self.assertGreater(imported["copied_result_files"], 0)
        self.assertFalse((self.output_root / "private" / "attempts.jsonl").exists())
        self.assertFalse(imported["attempt_history_copied"])
        self.assertEqual(
            imported["attempt_history_policy"],
            "not_imported_without_bound_run_identity",
        )
        self.assertTrue(
            (self.output_root / "private" / "mask_route_examples.png").is_file()
        )
        private_row = json.loads(
            (self.output_root / "private" / "mask_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(private_row["cohort_id"], DS1_SCREEN_CONTRACT.cohort_id)
        self.assertEqual(
            second["output_identity"]["mask_results_sha256"],
            file_sha256(self.output_root / "private" / "mask_results.jsonl"),
        )
        self.assertEqual(
            second["output_identity"]["repeat_mask_results_sha256"],
            file_sha256(self.output_root / "private" / "repeat_mask_results.jsonl"),
        )

    def test_prompt_package_membership_is_fail_closed(self):
        record_path = self.prompt_root / "package_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["membership_sha256"] = "0" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "membership_sha256"):
            build_ds1_sam_jobs(self.prompt_root, self.media_root)

    def test_full_run_reports_phases_without_per_image_log_noise(self):
        events = []

        def capture(stage, details):
            events.append((stage, details))

        record = self._run(progress_callback=capture)
        phase_names = [
            stage for stage, _ in events if stage != "inference_progress"
        ]
        self.assertEqual(
            phase_names,
            [
                "one_image_preflight_started",
                "one_image_preflight_completed",
                "pilot_started",
                "pilot_completed",
                "repeatability_started",
                "repeatability_completed",
                "full_inference_started",
                "full_inference_completed",
                "acceptance_gates_completed",
            ],
        )
        progress_events = [
            details for stage, details in events if stage == "inference_progress"
        ]
        self.assertLessEqual(len(progress_events), 26)
        self.assertEqual(progress_events[-1]["completed"], progress_events[-1]["total"])
        self.assertTrue(events[-1][1]["passed"])
        self.assertTrue(record["completion_gate_passed"])

    def test_clearml_validator_accepts_only_completed_model_evidence(self):
        self._run()
        safe_root = self.output_root / "safe"
        record_path = safe_root / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update(
            {
                "run_mode": "model_inference",
                "producer_kind": "sam2.1_model",
                "checkpoint_sha256": record["configuration"]["checkpoint_sha256"],
                "provenance": {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "precommit_candidate": False,
                    "snapshot_sha256": "b" * 64,
                    "notebook_sha256": "c" * 64,
                },
            }
        )
        record_path.write_text(json.dumps(record), encoding="utf-8")
        _write_artifact_manifest(safe_root)

        validated = FINALIZER.validate_safe_root(safe_root)

        self.assertEqual(validated["cohort"]["scene_count"], 100)
        record["provenance"]["git_dirty"] = True
        record_path.write_text(json.dumps(record), encoding="utf-8")
        _write_artifact_manifest(safe_root)
        with self.assertRaisesRegex(ValueError, "clean committed"):
            FINALIZER.validate_safe_root(safe_root)

    def test_clearml_validator_rejects_private_identifiers(self):
        self._run()
        path = self.output_root / "safe" / "failure_summary.csv"
        path.write_text("error_type,count\nS000001,1\n", encoding="utf-8")
        _write_artifact_manifest(self.output_root / "safe")

        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.output_root / "safe")

    def test_clearml_validator_rejects_uncommitted_mask_identity(self):
        self._run()
        safe_root = self.output_root / "safe"
        record_path = safe_root / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update(
            {
                "run_mode": "model_inference",
                "producer_kind": "sam2.1_model",
                "checkpoint_sha256": record["configuration"]["checkpoint_sha256"],
                "provenance": {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "precommit_candidate": False,
                    "snapshot_sha256": "b" * 64,
                    "notebook_sha256": "c" * 64,
                },
            }
        )
        record["output_identity"]["mask_results_sha256"] = "not-a-digest"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        _write_artifact_manifest(safe_root)

        with self.assertRaisesRegex(ValueError, "SAM result identity"):
            FINALIZER.validate_safe_root(safe_root)

    def test_clearml_validator_rejects_uncommitted_input_identity(self):
        self._run()
        safe_root = self.output_root / "safe"
        record_path = safe_root / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update(
            {
                "run_mode": "model_inference",
                "producer_kind": "sam2.1_model",
                "checkpoint_sha256": record["configuration"]["checkpoint_sha256"],
                "provenance": {
                    "git_commit": "a" * 40,
                    "git_dirty": False,
                    "precommit_candidate": False,
                    "snapshot_sha256": "b" * 64,
                    "notebook_sha256": "c" * 64,
                },
            }
        )
        record["input_identity"]["prompt_manifest_sha256"] = "not-a-digest"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        _write_artifact_manifest(safe_root)

        with self.assertRaisesRegex(ValueError, "SAM input identity"):
            FINALIZER.validate_safe_root(safe_root)


if __name__ == "__main__":
    unittest.main()
