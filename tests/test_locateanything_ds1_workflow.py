from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.cohort_contract import (  # noqa: E402
    CohortContract,
    DS1_SCREEN_CONTRACT,
)
from visual_peer_inspector.data.model_safe_package import (  # noqa: E402
    membership_sha256,
)
from visual_peer_inspector.models.locateanything import (  # noqa: E402
    LocateAnythingConfiguration,
    ProposalContractWorker,
    ProposalResultStore,
    build_proposal_jobs,
)
from visual_peer_inspector.workflows import locateanything_ds1 as WORKFLOW  # noqa: E402


class Ds1LocateAnythingWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_root = self.root / "model-input"
        self.output_root = self.root / "output"
        scene_ids = [f"S{index:06d}" for index in range(1, 21)]
        self.contract = CohortContract(
            cohort_id="test-development-screen-20-v1",
            release_id="test-release-v1",
            role="development",
            pilot_stage="screen_20",
            membership_sha256=membership_sha256(scene_ids),
            scene_count=20,
            views_per_scene=20,
            image_count=400,
            asset_group_count=20,
        )
        rows = []
        for scene_index, scene_id in enumerate(scene_ids, 1):
            media_root = Path("media") / scene_id
            view_ids = [f"V{index:02d}" for index in range(20)]
            rows.append(
                {
                    "opaque_scene_id": scene_id,
                    "media_root": media_root.as_posix(),
                    "view_ids": view_ids,
                }
            )
            for view_index, view_id in enumerate(view_ids):
                image_path = self.model_root / media_root / "rgb" / f"{view_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (32, 32),
                    (scene_index % 255, view_index * 8 % 255, 120),
                ).save(image_path)
        manifest = self.model_root / "neutral_corpus_manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.manifest_sha256 = sha256(manifest.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _package_audit(self) -> SimpleNamespace:
        values = {
            "cohort_id": self.contract.cohort_id,
            "release_id": self.contract.release_id,
            "scene_count": self.contract.scene_count,
            "view_count": self.contract.views_per_scene,
            "image_count": self.contract.image_count,
            "manifest_sha256": self.manifest_sha256,
            "membership_sha256": self.contract.membership_sha256,
            "verified_image_count": self.contract.image_count,
            "all_checks_passed": True,
        }
        return SimpleNamespace(
            manifest_sha256=self.manifest_sha256,
            to_dict=lambda: dict(values),
        )

    def test_frozen_ds1_contract_has_expected_cardinality(self) -> None:
        self.assertEqual(DS1_SCREEN_CONTRACT.scene_count, 100)
        self.assertEqual(DS1_SCREEN_CONTRACT.views_per_scene, 20)
        self.assertEqual(DS1_SCREEN_CONTRACT.image_count, 2_000)
        self.assertEqual(DS1_SCREEN_CONTRACT.asset_group_count, 27)

    def test_pilot_covers_twenty_scenes_and_views(self) -> None:
        jobs = build_proposal_jobs(self.model_root, self.contract.cohort_id)
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract):
            WORKFLOW.verify_ds1_jobs(jobs)
            pilot = WORKFLOW.select_pilot_jobs(jobs)
        self.assertEqual(len(pilot), 20)
        self.assertEqual(len({job.opaque_scene_id for job in pilot}), 20)
        self.assertEqual({job.view_id for job in pilot}, {f"V{i:02d}" for i in range(20)})

    def test_full_contract_run_is_complete_and_resumable(self) -> None:
        configuration = LocateAnythingConfiguration()
        worker = ProposalContractWorker(configuration)
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract), patch.object(
            WORKFLOW,
            "verify_model_input_package",
            return_value=self._package_audit(),
        ):
            first = WORKFLOW.run_ds1_locateanything_workflow(
                model_input_root=self.model_root,
                output_root=self.output_root,
                worker=worker,
                configuration=configuration,
                run_mode="contract_smoke",
                inference_stage="full",
            )
            proposal_path = self.output_root / "private/proposal_results.jsonl"
            first_digest = sha256(proposal_path.read_bytes()).hexdigest()
            second = WORKFLOW.run_ds1_locateanything_workflow(
                model_input_root=self.model_root,
                output_root=self.output_root,
                worker=worker,
                configuration=configuration,
                run_mode="contract_smoke",
                inference_stage="full",
            )
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["summary"]["job_count"], 400)
        self.assertEqual(first["repeat_agreement"]["box_agreement_rate"], 1.0)
        self.assertEqual(first["output_identity"]["result_filename"], "proposal_results.jsonl")
        self.assertEqual(first["output_identity"]["result_sha256"], first_digest)
        self.assertEqual(
            first["output_identity"]["proposal_results_sha256"], first_digest
        )
        self.assertEqual(first["output_identity"]["result_count"], 400)
        self.assertEqual(first["retry_policy"]["maximum_attempts_per_job"], 4)
        self.assertEqual(first["retry_policy"]["seed_scope"], "scene_view_attempt")
        self.assertEqual(second["run_session"]["pending_job_count_at_start"], 0)
        self.assertEqual(sha256(proposal_path.read_bytes()).hexdigest(), first_digest)
        self.assertTrue((self.output_root / "safe/artifact_manifest.json").is_file())
        self.assertTrue((self.output_root / "private/proposal_examples.png").is_file())

    def test_one_image_and_pilot_stages_produce_diagnostic_evidence(self) -> None:
        configuration = LocateAnythingConfiguration()
        worker = ProposalContractWorker(configuration)
        for stage, expected in (("one_image", 1), ("pilot", 20)):
            with self.subTest(stage=stage), patch.object(
                WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract
            ), patch.object(
                WORKFLOW,
                "verify_model_input_package",
                return_value=self._package_audit(),
            ):
                output = self.root / f"output-{stage}"
                record = WORKFLOW.run_ds1_locateanything_workflow(
                    model_input_root=self.model_root,
                    output_root=output,
                    worker=worker,
                    configuration=configuration,
                    run_mode="contract_smoke",
                    inference_stage=stage,
                )
                self.assertEqual(record["status"], "diagnostic_completed")
                self.assertEqual(record["summary"]["job_count"], expected)
                self.assertTrue((output / "safe/mean_proposals_by_view.png").is_file())

    def test_membership_mismatch_is_rejected(self) -> None:
        jobs = build_proposal_jobs(self.model_root, self.contract.cohort_id)
        wrong = CohortContract(
            **{
                **self.contract.__dict__,
                "membership_sha256": "0" * 64,
            }
        )
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", wrong):
            with self.assertRaisesRegex(ValueError, "membership"):
                WORKFLOW.verify_ds1_jobs(jobs)

    def test_failed_preflight_stops_before_the_pilot(self) -> None:
        configuration = LocateAnythingConfiguration()

        class FailedWorker(ProposalContractWorker):
            def run_job(self, job, *, attempt=1):
                row = super().run_job(job, attempt=attempt)
                row.update(
                    {
                        "proposal_count": 0,
                        "boxes_xyxy": [],
                        "status": "error",
                        "error_type": "TestFailure",
                        "error_message": "Sanitized test failure",
                    }
                )
                return row

        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract), patch.object(
            WORKFLOW,
            "verify_model_input_package",
            return_value=self._package_audit(),
        ):
            with self.assertRaisesRegex(RuntimeError, "one-image preflight"):
                WORKFLOW.run_ds1_locateanything_workflow(
                    model_input_root=self.model_root,
                    output_root=self.output_root,
                    worker=FailedWorker(configuration),
                    configuration=configuration,
                    run_mode="contract_smoke",
                    inference_stage="full",
                )
        self.assertFalse((self.output_root / "private/pilot_results.jsonl").exists())
        self.assertFalse((self.output_root / "private/proposal_results.jsonl").exists())

    def test_checkpoint_round_trip_preserves_private_resume_state(self) -> None:
        configuration = LocateAnythingConfiguration()
        jobs = build_proposal_jobs(self.model_root, self.contract.cohort_id)
        private_root = self.output_root / "private"
        store = ProposalResultStore(
            private_root / "one_image_results.jsonl",
            configuration.sha256,
            "contract_smoke",
        )
        store.append(ProposalContractWorker(configuration).run_job(jobs[0]))
        archive = self.root / "checkpoint.zip"
        restored = self.root / "restored"
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract):
            exported = WORKFLOW.export_ds1_checkpoint(
                self.output_root,
                archive,
                configuration_sha256=configuration.sha256,
                producer_kind="contract_smoke",
                input_manifest_sha256=self.manifest_sha256,
            )
            imported = WORKFLOW.import_ds1_checkpoint(
                archive,
                restored,
                configuration_sha256=configuration.sha256,
                producer_kind="contract_smoke",
                input_manifest_sha256=self.manifest_sha256,
            )
        self.assertEqual(exported["archive_sha256"], imported["archive_sha256"])
        self.assertEqual(
            (private_root / "one_image_results.jsonl").read_bytes(),
            (restored / "private/one_image_results.jsonl").read_bytes(),
        )

    def test_canonical_model_run_requires_clean_provenance(self) -> None:
        configuration = LocateAnythingConfiguration()
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract):
            with self.assertRaisesRegex(ValueError, "clean committed snapshot"):
                WORKFLOW.run_ds1_locateanything_workflow(
                    model_input_root=self.model_root,
                    output_root=self.output_root,
                    worker=ProposalContractWorker(configuration),
                    configuration=configuration,
                    run_mode="model_inference",
                    inference_stage="full",
                    provenance={"git_dirty": True},
                )

    def test_canonical_model_run_requires_disposable_gpu_gate(self) -> None:
        configuration = LocateAnythingConfiguration()
        provenance = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "precommit_candidate": False,
            "snapshot_sha256": "b" * 64,
            "notebook_sha256": "c" * 64,
        }
        with patch.object(WORKFLOW, "DS1_SCREEN_CONTRACT", self.contract):
            with self.assertRaisesRegex(ValueError, "disposable one-image gate"):
                WORKFLOW.run_ds1_locateanything_workflow(
                    model_input_root=self.model_root,
                    output_root=self.output_root,
                    worker=ProposalContractWorker(configuration),
                    configuration=configuration,
                    run_mode="model_inference",
                    inference_stage="full",
                    provenance=provenance,
                    model_load_evidence={},
                )


if __name__ == "__main__":
    unittest.main()
