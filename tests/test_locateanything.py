from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.models.locateanything import (
    CONTRACT_PRODUCER_KIND,
    EXECUTION_POLICY_SHA256,
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    MODEL_PRODUCER_KIND,
    ProposalContractWorker,
    ProposalJob,
    ProposalResultStore,
    bounded_locateanything_execution_policy,
    build_proposal_jobs,
    collect_proposal_evidence,
    execution_policy_sha256,
    generation_wall_time_limit,
    locateanything_execution_policy,
    resolve_generation_policy,
    repeat_agreement,
    run_proposal_jobs,
    parse_locateanything_boxes,
    summarize_proposal_rows,
    validate_proposal_result,
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
                "execution_policy_sha256": EXECUTION_POLICY_SHA256,
                "effective_generation_mode": "hybrid",
                "effective_max_new_tokens": 8192,
                "producer_kind": CONTRACT_PRODUCER_KIND,
                "model_id": configuration.model_id,
                "model_revision": configuration.model_revision,
                "prompt": configuration.prompt,
                "proposal_count": 1,
                "boxes_xyxy": [[1, 2, 5, 6]],
                "raw_response": "<box><100><200><500><600></box>",
                "generation_stats": {
                    "effective_generation_mode": "hybrid",
                    "effective_max_new_tokens": 8192,
                },
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
                "execution_policy_sha256": EXECUTION_POLICY_SHA256,
                "effective_generation_mode": "hybrid",
                "effective_max_new_tokens": 8192,
                "producer_kind": CONTRACT_PRODUCER_KIND,
                "model_id": configuration.model_id,
                "model_revision": configuration.model_revision,
                "prompt": configuration.prompt,
                "proposal_count": 0,
                "boxes_xyxy": [],
                "raw_response": "",
                "generation_stats": {
                    "effective_generation_mode": "hybrid",
                    "effective_max_new_tokens": 8192,
                },
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

    def test_bounded_policy_keeps_the_legacy_policy_identity_unchanged(self) -> None:
        legacy_digest = execution_policy_sha256(locateanything_execution_policy())
        configuration = LocateAnythingConfiguration(max_new_tokens=4096)
        bounded = bounded_locateanything_execution_policy(
            configuration,
            maximum_attempts_per_job=2,
            terminal_failure_disposition="carry_as_zero_proposal",
        )

        self.assertEqual(
            legacy_digest,
            "1477c554722790d78df5979555bb78e22e87eb60a4c75d03e9eb885bba2d6f9e",
        )
        self.assertEqual(legacy_digest, EXECUTION_POLICY_SHA256)
        self.assertNotEqual(execution_policy_sha256(bounded), EXECUTION_POLICY_SHA256)
        self.assertEqual(bounded["primary"]["attempts"], [1, 2])
        self.assertEqual(bounded["primary"]["max_new_tokens"], 4096)

    def test_generation_wall_time_limit_interrupts_a_slow_attempt(self) -> None:
        with self.assertRaises(TimeoutError):
            with generation_wall_time_limit(0.02) as active:
                self.assertTrue(active)
                time.sleep(0.20)

    def test_bounded_policy_rows_validate_and_resume_only_with_that_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "V00.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            configuration = LocateAnythingConfiguration(max_new_tokens=4096)
            policy = bounded_locateanything_execution_policy(
                configuration,
                maximum_attempts_per_job=2,
                terminal_failure_disposition="carry_as_zero_proposal",
            )
            job = ProposalJob(
                cohort_id="confirm_300",
                input_manifest_sha256="b" * 64,
                opaque_scene_id="S000001",
                view_id="V00",
                image_path=image_path,
                image_sha256="a" * 64,
            )
            worker = ProposalContractWorker(
                configuration,
                execution_policy=policy,
            )
            row = worker.run_job(job)
            store_path = root / "results.jsonl"
            store = ProposalResultStore(
                store_path,
                configuration.sha256,
                CONTRACT_PRODUCER_KIND,
                policy,
            )
            store.append(row)

            self.assertEqual(row["effective_max_new_tokens"], 4096)
            self.assertEqual(
                row["execution_policy_sha256"], execution_policy_sha256(policy)
            )
            ProposalResultStore(
                store_path,
                configuration.sha256,
                CONTRACT_PRODUCER_KIND,
                policy,
            )
            with self.assertRaisesRegex(ValueError, "execution policy"):
                ProposalResultStore(
                    store_path,
                    configuration.sha256,
                    CONTRACT_PRODUCER_KIND,
                )
            with self.assertRaisesRegex(ValueError, "active execution policy"):
                worker.run_job(job, attempt=3)

    def test_terminal_failures_can_be_kept_without_a_consecutive_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            configuration = LocateAnythingConfiguration(max_new_tokens=4096)
            policy = bounded_locateanything_execution_policy(
                configuration,
                maximum_attempts_per_job=2,
                terminal_failure_disposition="carry_as_zero_proposal",
            )
            jobs = []
            for index in range(2):
                image_path = root / f"V{index:02d}.png"
                Image.new("RGB", (64, 64), "white").save(image_path)
                jobs.append(
                    ProposalJob(
                        cohort_id="confirm_300",
                        input_manifest_sha256="b" * 64,
                        opaque_scene_id="S000001",
                        view_id=f"V{index:02d}",
                        image_path=image_path,
                        image_sha256=chr(ord("a") + index) * 64,
                    )
                )

            class FailingWorker:
                producer_kind = CONTRACT_PRODUCER_KIND

                def __init__(self) -> None:
                    self.configuration = configuration
                    self.execution_policy = policy

                def run_job(self, job: ProposalJob, *, attempt: int = 1) -> dict:
                    row = ProposalContractWorker(
                        configuration,
                        execution_policy=policy,
                    ).run_job(job, attempt=attempt)
                    row.update(
                        {
                            "proposal_count": 0,
                            "boxes_xyxy": [],
                            "raw_response": "",
                            "status": "error",
                            "error_type": "NoValidBoxes",
                            "error_message": "No valid boxes",
                        }
                    )
                    return row

            store = ProposalResultStore(
                root / "failures.jsonl",
                configuration.sha256,
                CONTRACT_PRODUCER_KIND,
                policy,
            )
            session = run_proposal_jobs(
                worker=FailingWorker(),
                jobs=jobs,
                result_store=store,
                maximum_attempts_per_job=2,
                maximum_consecutive_failed_jobs=None,
            )

            self.assertEqual(session["latest_row_count"], 2)
            self.assertEqual(session["successful_job_count"], 0)
            self.assertEqual([row["attempt"] for row in store.rows], [2, 2])

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

    def test_retry_policy_uses_hybrid_then_supported_slow_mode(self) -> None:
        configuration = LocateAnythingConfiguration()
        configuration_sha256 = configuration.sha256

        policies = [resolve_generation_policy(configuration, attempt) for attempt in range(1, 7)]

        self.assertEqual(
            [policy.generation_mode for policy in policies],
            ["hybrid", "hybrid", "hybrid", "hybrid", "slow", "slow"],
        )
        self.assertEqual(
            [policy.max_new_tokens for policy in policies],
            [8192, 8192, 8192, 8192, 8192, 8192],
        )
        self.assertEqual(configuration.sha256, configuration_sha256)

    def test_predict_passes_the_effective_policy_to_model_generation(self) -> None:
        calls = []

        class FakeTensor:
            def to(self, _value):
                return self

        class FakeInputs(dict):
            def to(self, _value):
                return self

        class FakeProcessor:
            @staticmethod
            def py_apply_chat_template(*_args, **_kwargs):
                return "prompt"

            @staticmethod
            def process_vision_info(*_args, **_kwargs):
                return ["image"], []

            def __call__(self, **_kwargs):
                return FakeInputs(
                    pixel_values=FakeTensor(),
                    input_ids=FakeTensor(),
                    attention_mask=FakeTensor(),
                )

        class FakeModel:
            @staticmethod
            def generate(**kwargs):
                calls.append(kwargs)
                return "<box><100><100><900><900></box>"

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def manual_seed_all(_seed):
                return None

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": FakeCuda(),
                "manual_seed": staticmethod(lambda _seed: None),
                "inference_mode": staticmethod(nullcontext),
            },
        )()
        worker = object.__new__(LocateAnythingWorker)
        worker.configuration = LocateAnythingConfiguration()
        worker.processor = FakeProcessor()
        worker.model = FakeModel()
        worker.tokenizer = object()
        worker.device = "cpu"
        worker.dtype = "float32"

        with patch.dict(sys.modules, {"torch": fake_torch}):
            worker.predict(Image.new("RGB", (8, 8)), "S000001", "V00", attempt=1)
            worker.predict(Image.new("RGB", (8, 8)), "S000001", "V00", attempt=5)

        self.assertEqual(calls[0]["generation_mode"], "hybrid")
        self.assertEqual(calls[1]["generation_mode"], "slow")
        self.assertEqual([call["max_new_tokens"] for call in calls], [8192, 8192])

    def test_result_rows_record_effective_retry_generation_settings(self) -> None:
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

            primary = worker.run_job(job, attempt=4)
            fallback = worker.run_job(job, attempt=5)

            self.assertEqual(primary["generation_stats"]["effective_generation_mode"], "hybrid")
            self.assertEqual(fallback["generation_stats"]["effective_generation_mode"], "slow")
            self.assertEqual(primary["generation_stats"]["effective_max_new_tokens"], 8192)
            self.assertEqual(fallback["generation_stats"]["effective_max_new_tokens"], 8192)

    def test_four_truncations_are_followed_by_slow_attempt_five(self) -> None:
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

            class RecoveryWorker(ProposalContractWorker):
                def run_job(self, current_job, *, attempt=1):
                    row = super().run_job(current_job, attempt=attempt)
                    if attempt <= 4:
                        row.update(
                            status="error",
                            error_type="PossibleTruncation",
                            error_message="Synthetic truncation",
                            possibly_truncated=True,
                            truncation_check="generation_stats",
                            proposal_count=0,
                            boxes_xyxy=[],
                        )
                    return row

            store = ProposalResultStore(
                root / "results.jsonl", configuration.sha256, CONTRACT_PRODUCER_KIND
            )
            run_proposal_jobs(
                worker=RecoveryWorker(configuration),
                jobs=[job],
                result_store=store,
                maximum_attempts_per_job=6,
            )

            self.assertEqual([row["attempt"] for row in store.history_rows], [1, 2, 3, 4, 5])
            self.assertEqual(
                [row["effective_generation_mode"] for row in store.history_rows],
                ["hybrid", "hybrid", "hybrid", "hybrid", "slow"],
            )
            self.assertEqual(store.rows[0]["status"], "ok")

    def test_result_validation_rejects_policy_mode_and_token_tampering(self) -> None:
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
            baseline = ProposalContractWorker(configuration).run_job(job)
            cases = (
                ("execution_policy_sha256", "0" * 64, "execution policy"),
                ("effective_generation_mode", "slow", "generation mode"),
                ("effective_max_new_tokens", 16384, "token limit"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    tampered = dict(baseline)
                    tampered[field] = value
                    with self.assertRaisesRegex(ValueError, message):
                        validate_proposal_result(tampered)

    def test_attempt_six_truncation_stops_without_promotion(self) -> None:
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

            class AlwaysTruncatedWorker(ProposalContractWorker):
                def run_job(self, current_job, *, attempt=1):
                    row = super().run_job(current_job, attempt=attempt)
                    row.update(
                        status="error",
                        error_type="PossibleTruncation",
                        error_message="Synthetic truncation",
                        possibly_truncated=True,
                        truncation_check="generation_stats",
                        proposal_count=0,
                        boxes_xyxy=[],
                    )
                    return row

            store = ProposalResultStore(
                root / "results.jsonl", configuration.sha256, CONTRACT_PRODUCER_KIND
            )
            session = run_proposal_jobs(
                worker=AlwaysTruncatedWorker(configuration),
                jobs=[job],
                result_store=store,
                maximum_attempts_per_job=6,
            )

            self.assertEqual(len(store.history_rows), 6)
            self.assertEqual(store.rows[0]["attempt"], 6)
            self.assertEqual(store.rows[0]["status"], "error")
            self.assertEqual(session["successful_job_count"], 0)

    def test_slow_fallback_rejects_a_non_truncation_predecessor(self) -> None:
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
            worker = ProposalContractWorker(configuration)
            store = ProposalResultStore(
                root / "results.jsonl", configuration.sha256, CONTRACT_PRODUCER_KIND
            )
            for attempt in range(1, 5):
                row = worker.run_job(job, attempt=attempt)
                row.update(
                    status="error",
                    error_type="NoValidBoxes",
                    error_message="Synthetic non-truncation failure",
                    proposal_count=0,
                    boxes_xyxy=[],
                )
                store.append(row)
            with self.assertRaisesRegex(ValueError, "preceding possible truncation"):
                store.append(worker.run_job(job, attempt=5))


if __name__ == "__main__":
    unittest.main()
