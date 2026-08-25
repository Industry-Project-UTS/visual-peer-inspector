from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.models.locateanything import (
    CONTRACT_PRODUCER_KIND,
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    MODEL_PRODUCER_KIND,
    ProposalContractWorker,
    ProposalJob,
    ProposalResultStore,
    build_proposal_jobs,
    collect_proposal_evidence,
    repeat_agreement,
    run_proposal_jobs,
    parse_locateanything_boxes,
    summarize_proposal_rows,
)


class LocateAnythingTests(unittest.TestCase):
    def test_parser_supports_both_box_token_formats(self) -> None:
        response = (
            "<box><100><200><500><800></box> "
            "<box> 600, 100, 900, 400 </box> "
            "<box><100><200><500><800></box>"
        )

        boxes = parse_locateanything_boxes(response, 200, 100)

        self.assertEqual(boxes, [[20, 20, 100, 80], [120, 10, 180, 40]])

    def test_job_builder_uses_only_opaque_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene_id = "S000001"
            view_ids = ["V00", "V01"]
            row = {
                "opaque_scene_id": scene_id,
                "view_ids": view_ids,
                "media_root": f"media/{scene_id}",
            }
            (root / "neutral_corpus_manifest.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            for view_id in view_ids:
                path = root / "media" / scene_id / "rgb" / f"{view_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), "white").save(path)

            jobs = build_proposal_jobs(root, "smoke")

            self.assertEqual([job.key for job in jobs], [(scene_id, "V00"), (scene_id, "V01")])

    def test_result_store_resumes_and_rejects_duplicate_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "proposals.jsonl"
            configuration = LocateAnythingConfiguration()
            store = ProposalResultStore(
                path, configuration.sha256, CONTRACT_PRODUCER_KIND
            )
            row = {
                "schema_version": 1,
                "cohort_id": "smoke",
                "input_manifest_sha256": "b" * 64,
                "opaque_scene_id": "S000001",
                "view_id": "V00",
                "attempt": 1,
                "image_sha256": "a" * 64,
                "configuration_sha256": configuration.sha256,
                "producer_kind": CONTRACT_PRODUCER_KIND,
                "model_id": configuration.model_id,
                "model_revision": configuration.model_revision,
                "prompt": configuration.prompt,
                "proposal_count": 1,
                "boxes_xyxy": [[1, 2, 5, 6]],
                "raw_response": "<box><100><200><500><600></box>",
                "generation_stats": {},
                "response_token_count": 6,
                "possibly_truncated": False,
                "truncation_check": "not_applicable",
                "runtime_seconds": 0.25,
                "status": "ok",
                "error_type": None,
                "error_message": None,
            }
            store.append(row)

            resumed = ProposalResultStore(
                path, configuration.sha256, CONTRACT_PRODUCER_KIND
            )

            self.assertEqual(resumed.completed_keys, {("S000001", "V00")})
            self.assertEqual(summarize_proposal_rows(resumed.rows)["successful_job_count"], 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resumed.append(row)

    def test_error_attempt_is_retried_and_reconciled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )
            store = ProposalResultStore(
                root / "results.jsonl",
                configuration.sha256,
                CONTRACT_PRODUCER_KIND,
            )
            failed = {
                "schema_version": 1,
                "cohort_id": job.cohort_id,
                "input_manifest_sha256": job.input_manifest_sha256,
                "opaque_scene_id": job.opaque_scene_id,
                "view_id": job.view_id,
                "attempt": 1,
                "image_sha256": job.image_sha256,
                "configuration_sha256": configuration.sha256,
                "producer_kind": CONTRACT_PRODUCER_KIND,
                "model_id": configuration.model_id,
                "model_revision": configuration.model_revision,
                "prompt": configuration.prompt,
                "proposal_count": 0,
                "boxes_xyxy": [],
                "raw_response": "",
                "generation_stats": {},
                "response_token_count": 0,
                "possibly_truncated": False,
                "truncation_check": "not_applicable",
                "runtime_seconds": 0.1,
                "status": "error",
                "error_type": "RuntimeError",
                "error_message": "Synthetic retry test failure",
            }
            store.append(failed)

            run_proposal_jobs(
                worker=ProposalContractWorker(configuration),
                jobs=[job],
                result_store=store,
                maximum_attempts_per_job=2,
            )

            self.assertEqual(store.completed_keys, {job.key})
            self.assertEqual(len(store.history_rows), 2)
            self.assertEqual(store.rows[0]["attempt"], 2)
            changed_job = ProposalJob(
                cohort_id=job.cohort_id,
                input_manifest_sha256=job.input_manifest_sha256,
                opaque_scene_id=job.opaque_scene_id,
                view_id=job.view_id,
                image_path=job.image_path,
                image_sha256="c" * 64,
            )
            with self.assertRaisesRegex(ValueError, "image checksum"):
                store.reconcile_jobs([changed_job])

    def test_contract_worker_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )
            worker = ProposalContractWorker(configuration)
            first = worker.run_job(job)
            second = worker.run_job(job)

            agreement = repeat_agreement([first], [second])

            self.assertEqual(agreement["box_agreement_rate"], 1.0)

    def test_one_image_evidence_keeps_geometry_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )
            row = ProposalContractWorker(configuration).run_job(job)

            evidence = collect_proposal_evidence(
                [row],
                [job],
                expected_job_count=1,
                expected_scene_count=1,
                expected_view_count=1,
            )

            self.assertTrue(all(gate["status"] == "pass" for gate in evidence["gate_rows"]))
            self.assertEqual(len(evidence["overlay_items"]), 1)
            self.assertEqual(evidence["geometry_summary"]["invalid_box_count"], 0)
            self.assertEqual(len(evidence["operational_rows"]), 1)

    def test_result_store_rejects_another_producer_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "proposals.jsonl"
            image_path = root / "V00.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )
            store = ProposalResultStore(
                path, configuration.sha256, CONTRACT_PRODUCER_KIND
            )
            store.append(ProposalContractWorker(configuration).run_job(job))
            with self.assertRaisesRegex(ValueError, "producer kind"):
                ProposalResultStore(
                    path, configuration.sha256, MODEL_PRODUCER_KIND
                )

    def test_real_worker_rejects_response_without_valid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )

            class FakeTokenizer:
                @staticmethod
                def encode(value: str, add_special_tokens: bool = False) -> list[int]:
                    return [1, 2, 3]

            worker = object.__new__(LocateAnythingWorker)
            worker.configuration = configuration
            worker.tokenizer = FakeTokenizer()
            worker.predict = lambda image, scene_id, view_id, attempt=1: "no boxes found"

            row = worker.run_job(job)

            self.assertEqual(row["producer_kind"], MODEL_PRODUCER_KIND)
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error_type"], "NoValidBoxes")
            self.assertEqual(row["proposal_count"], 0)

    def test_real_worker_prefers_generation_stats_for_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            configuration = LocateAnythingConfiguration()
            job = ProposalJob(
                cohort_id="smoke",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )

            class FakeTokenizer:
                @staticmethod
                def encode(value: str, add_special_tokens: bool = False) -> list[int]:
                    return [1, 2, 3]

            worker = object.__new__(LocateAnythingWorker)
            worker.configuration = configuration
            worker.tokenizer = FakeTokenizer()
            worker.predict = lambda image, scene_id, view_id, attempt=1: {
                "answer": "<box><100><100><900><900></box>",
                "stats": {"finish_reason": "max_token_limit"},
            }

            row = worker.run_job(job)

            self.assertEqual(row["status"], "error")
            self.assertTrue(row["possibly_truncated"])
            self.assertEqual(row["truncation_check"], "generation_stats")

    def test_retry_seed_is_repeatable_per_attempt_and_changes_between_attempts(self) -> None:
        worker = object.__new__(LocateAnythingWorker)
        worker.configuration = LocateAnythingConfiguration()

        first = worker._job_seed("S000001", "V13", attempt=1)
        repeated = worker._job_seed("S000001", "V13", attempt=1)
        retry = worker._job_seed("S000001", "V13", attempt=2)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, retry)
        with self.assertRaisesRegex(ValueError, "positive"):
            worker._job_seed("S000001", "V13", attempt=0)


if __name__ == "__main__":
    unittest.main()
