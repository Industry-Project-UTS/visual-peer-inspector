from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np

from visual_peer_inspector.qualitative_reporting import (
    PredictedObject,
    QualitativeResult,
    QualitativeView,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/visual_peer_inspector/workflows/frozen_development_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("vpi_frozen_development_pipeline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EmptyMaskFailure = MODULE.EmptyMaskFailure
FeatureFailure = MODULE.FeatureFailure
FrozenPipelineIdentity = MODULE.FrozenPipelineIdentity
StageOutcome = MODULE.StageOutcome
ZeroProposalFailure = MODULE.ZeroProposalFailure
bounded_tracking_call = MODULE.bounded_tracking_call
combine_development_locks = MODULE.combine_development_locks
load_development_evaluator = MODULE.load_development_evaluator
lock_predictions = MODULE.lock_predictions
publish_prediction_bundle = MODULE.publish_prediction_bundle
read_completed_stage = MODULE.read_completed_stage
reference_only_fusion = MODULE.reference_only_fusion
render_mask_evidence_panel = MODULE.render_mask_evidence_panel
run_sharded_stage = MODULE.run_sharded_stage
select_development_audit = MODULE.select_development_audit
stage_global_embeddings = MODULE.stage_global_embeddings
verify_prediction_lock = MODULE.verify_prediction_lock
validate_ds2_backfill_inputs = MODULE.validate_ds2_backfill_inputs


def identity(seed: str = "a") -> FrozenPipelineIdentity:
    digests = [sha256(f"{seed}-{index}".encode()).hexdigest() for index in range(7)]
    return FrozenPipelineIdentity(*digests)


def records(count: int = 4) -> list[dict]:
    return [
        {"opaque_scene_id": f"S{index // 2:04d}", "view_id": f"V{index % 2:02d}", "value": index}
        for index in range(count)
    ]


class FrozenDevelopmentPipelineTests(unittest.TestCase):
    def test_stage_resumes_only_verified_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def worker(row):
                calls.append(row["value"])
                return {"status": "ok", "value": row["value"]}

            first = run_sharded_stage(
                stage="locateanything",
                input_records=records(),
                output_root=root,
                identity=identity(),
                worker=worker,
                shard_size=2,
            )
            second = run_sharded_stage(
                stage="locateanything",
                input_records=records(),
                output_root=root,
                identity=identity(),
                worker=worker,
                shard_size=2,
            )
            self.assertEqual(first["processed_this_run"], 4)
            self.assertEqual(second["resumed"], 4)
            self.assertEqual(len(calls), 4)

            output = root / "locateanything/shard-00000.jsonl"
            output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            third = run_sharded_stage(
                stage="locateanything",
                input_records=records(),
                output_root=root,
                identity=identity(),
                worker=worker,
                shard_size=2,
            )
            self.assertEqual(third["processed_this_run"], 2)
            self.assertEqual(third["resumed"], 2)

    def test_bounded_runs_match_one_uninterrupted_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def worker(row):
                return {"status": "ok", "value": row["value"]}

            first = run_sharded_stage(
                stage="peer_scoring",
                input_records=records(7),
                output_root=root / "bounded",
                identity=identity(),
                worker=worker,
                shard_size=2,
                maximum_new_shards=1,
            )
            self.assertEqual(first["completed_shards"], 1)
            self.assertEqual(first["covered_input_count"], 2)
            self.assertFalse(first["complete"])
            for _ in range(3):
                bounded = run_sharded_stage(
                    stage="peer_scoring",
                    input_records=records(7),
                    output_root=root / "bounded",
                    identity=identity(),
                    worker=worker,
                    shard_size=2,
                    maximum_new_shards=1,
                )
            uninterrupted = run_sharded_stage(
                stage="peer_scoring",
                input_records=records(7),
                output_root=root / "uninterrupted",
                identity=identity(),
                worker=worker,
                shard_size=2,
            )
            self.assertTrue(bounded["complete"])
            self.assertTrue(uninterrupted["complete"])
            self.assertEqual(bounded["covered_input_count"], 7)
            self.assertEqual(
                read_completed_stage(root / "bounded/peer_scoring"),
                read_completed_stage(root / "uninterrupted/peer_scoring"),
            )

    def test_terminal_failures_are_explicit(self) -> None:
        errors = (ZeroProposalFailure("x"), EmptyMaskFailure("x"), FeatureFailure("x"))
        with tempfile.TemporaryDirectory() as directory:
            def worker(row):
                raise errors[int(row["value"])]

            summary = run_sharded_stage(
                stage="sam21",
                input_records=records(3),
                output_root=Path(directory),
                identity=identity(),
                worker=worker,
                shard_size=3,
            )
            rows = read_completed_stage(Path(directory) / "sam21")
            self.assertEqual(summary["terminal_failures"], 3)
            self.assertEqual(
                {row["terminal_failure"] for row in rows},
                {"zero_proposal", "empty_mask", "feature_failure"},
            )
            resumed = run_sharded_stage(
                stage="sam21",
                input_records=records(3),
                output_root=Path(directory),
                identity=identity(),
                worker=worker,
                shard_size=3,
            )
            self.assertEqual(resumed["terminal_failures"], 3)
            self.assertEqual(resumed["failure_counts"], summary["failure_counts"])

    def test_unknown_exception_is_not_misclassified_as_feature_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_sharded_stage(
                stage="sam21",
                input_records=records(1),
                output_root=Path(directory),
                identity=identity(),
                worker=lambda _row: (_ for _ in ()).throw(RuntimeError("unexpected")),
                shard_size=1,
            )
            self.assertEqual(summary["failure_counts"]["stage_failure"], 1)

    def test_embeddings_are_float16_hashed_and_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_sharded_stage(
                stage="dinov2_small",
                input_records=records(3),
                output_root=root,
                identity=identity(),
                worker=lambda row: StageOutcome(
                    {"status": "ok"},
                    np.asarray([[row["value"] + 1.0, 2.0]], dtype=np.float32),
                ),
                shard_size=2,
            )
            embeddings, index_rows = stage_global_embeddings(root / "dinov2_small")
            self.assertEqual(embeddings.dtype, np.float16)
            self.assertEqual(embeddings.shape, (3, 2))
            self.assertEqual([row["start"] for row in index_rows], [0, 1, 2])

    def test_lock_precedes_evaluator_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator.json"
            evaluator.write_text('{"reference": 1}\n', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_development_evaluator(evaluator, root / "prediction_lock.json")
            artifact = root / "prediction_manifest.jsonl"
            artifact.write_text("{}\n", encoding="utf-8")
            lock_predictions(
                prediction_root=root,
                identity=identity(),
                cohort_id="cohort",
                cohort_membership_sha256="f" * 64,
                scene_count=1,
                image_count=20,
            )
            self.assertEqual(load_development_evaluator(evaluator, root / "prediction_lock.json")["reference"], 1)
            artifact.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_prediction_lock(root / "prediction_lock.json")

    def test_combine_requires_exact_frozen_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            details = (
                ("toysad8k-development-screen-100-v1", 100),
                ("toysad8k-development-remainder-600-v1", 600),
                ("toysad8k-development-confirm-300-v1", 300),
            )
            locks = []
            for index, (cohort, count) in enumerate(details):
                part = root / str(index)
                part.mkdir()
                (part / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
                lock_predictions(
                    prediction_root=part,
                    identity=identity(),
                    cohort_id=cohort,
                    cohort_membership_sha256=chr(ord("a") + index) * 64,
                    scene_count=count,
                    image_count=count * 20,
                )
                locks.append(part / "prediction_lock.json")
            combined = combine_development_locks(
                ds1_lock=locks[0],
                remainder_lock=locks[1],
                ds2_lock=locks[2],
                output_path=root / "combined.json",
            )
            self.assertEqual(combined["scene_count"], 1_000)
            self.assertEqual(combined["image_count"], 20_000)

            ds2_root = locks[2].parent
            (ds2_root / "prediction_lock.json").unlink()
            lock_predictions(
                prediction_root=ds2_root,
                identity=identity("b"),
                cohort_id=details[2][0],
                cohort_membership_sha256="c" * 64,
                scene_count=300,
                image_count=6_000,
            )
            with self.assertRaises(ValueError):
                combine_development_locks(
                    ds1_lock=locks[0],
                    remainder_lock=locks[1],
                    ds2_lock=locks[2],
                    output_path=root / "combined-invalid.json",
                )

    def test_audit_selection_is_deterministic_and_keeps_failures(self) -> None:
        values = [
            {"opaque_scene_id": f"S{index:04d}", "view_id": "V00", "terminal_failure": None}
            for index in range(25)
        ]
        for index, failure in enumerate(("zero_proposal", "empty_mask", "feature_failure")):
            values[index]["terminal_failure"] = failure
        first = select_development_audit(values, base_count=15)
        second = select_development_audit(list(reversed(values)), base_count=15)
        self.assertEqual(first, second)
        self.assertTrue({"zero_proposal", "empty_mask", "feature_failure"}.issubset(
            {row.get("terminal_failure") for row in first}
        ))
        self.assertGreaterEqual(len(first), 15)

    def test_tracking_timeout_cannot_block_the_caller(self) -> None:
        started = time.perf_counter()
        result = bounded_tracking_call(lambda: time.sleep(0.5), timeout_seconds=0.02)
        self.assertEqual(result["status"], "timed_out")
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_reference_fusion_is_never_deployment_valid(self) -> None:
        result = reference_only_fusion([{"score": 1.0}], {"mapping": [1]})
        self.assertEqual(result["claim_scope"], "reference_only")
        self.assertFalse(result["deployment_valid"])

    def test_publication_bundle_has_strict_08f_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = records(2)

            def representation(row):
                object_id = f"{row['view_id']}-P01"
                return StageOutcome(
                    {
                        "status": "insufficient_peers",
                        "image_sha256": "1" * 64,
                        "objects": [{
                            "object_id": object_id,
                            "box_xyxy": [1, 2, 8, 9],
                            "mask_rle": {"size": [10, 10], "counts": [21, 4, 75]},
                            "mask_sha256": "2" * 64,
                            "foreground_pixels": 4,
                            "predicted_quality_score": 0.8,
                            "b0_status": "insufficient_peers",
                            "b0_score": None,
                            "p1_status": "insufficient_peers",
                            "p1_score": None,
                            "p1_rank": None,
                            "p1_heatmap": None,
                        }],
                    },
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                )

            run_sharded_stage(
                stage="dinov2_small",
                input_records=source,
                output_root=root,
                identity=identity(),
                worker=representation,
                shard_size=1,
            )
            representation_rows = read_completed_stage(root / "dinov2_small")
            run_sharded_stage(
                stage="peer_scoring",
                input_records=representation_rows,
                output_root=root,
                identity=identity(),
                worker=lambda row: row,
                shard_size=1,
            )
            bundle = publish_prediction_bundle(
                cohort_root=root,
                identity=identity(),
                cohort_id="test-cohort",
                cohort_membership_sha256="f" * 64,
                scene_count=1,
                image_count=2,
            )
            self.assertTrue(Path(bundle["prediction_manifest"]).is_file())
            self.assertEqual(np.load(bundle["global_embeddings"], allow_pickle=False).dtype, np.float16)
            index = [json.loads(line) for line in Path(bundle["embedding_index"]).read_text().splitlines()]
            self.assertEqual([row["row_index"] for row in index], [0, 1])
            verify_prediction_lock(Path(bundle["prediction_lock"]))

    def test_mask_evidence_contains_three_views(self) -> None:
        rgb = np.full((32, 32, 3), 160, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        views = tuple(
            QualitativeView(
                view_id=f"V{index:02d}",
                rgb=rgb,
                objects=(PredictedObject(f"V{index:02d}-P01", (8, 8, 24, 24), mask, 0.5, 1),),
            )
            for index in range(3)
        )
        result = QualitativeResult("S0001", views, "V00-P01", 0.8, 0.1, "prediction_only")
        panel = render_mask_evidence_panel(result, panel_size=128)
        self.assertGreater(panel.width, panel.height // 2)

    def test_ds2_backfill_fails_closed_on_identity_or_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = {
                "schema_version": 1,
                "package_kind": "ds2-frozen-automatic-proposals",
                "cohort_id": "toysad8k-development-confirm-300-v1",
                "model_id": "nvidia/LocateAnything-3B",
                "model_revision": "a" * 40,
                "prompt": "individual object",
                "postprocessing_policy": "area_nms_075_cap8",
                "effective_generation_configuration_sha256": "b" * 64,
                "proposal_manifest_sha256": "c" * 64,
            }
            sam = {
                "schema_version": 1,
                "stage": "ds2_automatic_sam",
                "status": "completed",
                "route": "automatic_box",
                "cohort_id": "toysad8k-development-confirm-300-v1",
                "configuration_sha256": "d" * 64,
                "prompt_package_sha256": "c" * 64,
                "sam_results_sha256": "e" * 64,
            }
            routes = {
                "schema_version": 1,
                "package_kind": "ds2-confirmation-mask-routes",
                "cohort_id": "toysad8k-development-confirm-300-v1",
                "mask_routes_sha256": "f" * 64,
                "automatic_sam_results_sha256": "e" * 64,
            }
            paths = []
            for name, value in (("proposal.json", proposal), ("sam.json", sam), ("routes.json", routes)):
                path = root / name
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            verified = validate_ds2_backfill_inputs(
                proposal_record_path=paths[0],
                automatic_sam_record_path=paths[1],
                mask_route_record_path=paths[2],
                expected_locate_model_id=proposal["model_id"],
                expected_locate_revision=proposal["model_revision"],
                expected_locate_configuration_sha256=proposal["effective_generation_configuration_sha256"],
                expected_prompt=proposal["prompt"],
                expected_proposal_policy=proposal["postprocessing_policy"],
                expected_sam_configuration_sha256=sam["configuration_sha256"],
            )
            self.assertTrue(verified["requires_dinov2_and_scoring_rebuild"])
            proposal.pop("model_revision")
            paths[0].write_text(json.dumps(proposal), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ds2_backfill_inputs(
                    proposal_record_path=paths[0],
                    automatic_sam_record_path=paths[1],
                    mask_route_record_path=paths[2],
                    expected_locate_model_id="nvidia/LocateAnything-3B",
                    expected_locate_revision="a" * 40,
                    expected_locate_configuration_sha256="b" * 64,
                    expected_prompt="individual object",
                    expected_proposal_policy="area_nms_075_cap8",
                    expected_sam_configuration_sha256="d" * 64,
                )


if __name__ == "__main__":
    unittest.main()
