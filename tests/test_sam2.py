import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import (
    CONTRACT_PRODUCER_KIND,
    ContractSamWorker,
    Sam2Configuration,
    SamImageJob,
    SamPrompt,
    SamResultStore,
    binary_mask_sha256,
    collect_sam_evidence,
    compare_repeat_results,
    decode_binary_mask,
    encode_binary_mask,
    run_sam_jobs,
)
from visual_peer_inspector.workflows.sam_ds0 import _evenly_spaced_rows


class SamMaskTests(unittest.TestCase):
    def test_rle_round_trip_and_digest_are_stable(self):
        mask = np.zeros((256, 256), dtype=bool)
        mask[4:21, 7:31] = True

        encoded = encode_binary_mask(mask)
        decoded = decode_binary_mask(encoded)

        np.testing.assert_array_equal(decoded, mask)
        self.assertEqual(binary_mask_sha256(decoded), binary_mask_sha256(mask))

    def test_invalid_rle_length_fails(self):
        with self.assertRaisesRegex(ValueError, "length differs"):
            decode_binary_mask({"size": [2, 2], "counts": [1, 1]})

    def test_oversized_rle_fails_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            decode_binary_mask({"size": [2, 2], "counts": [10**12]})

    def test_non_integer_rle_count_fails(self):
        with self.assertRaisesRegex(ValueError, "must be integers"):
            decode_binary_mask({"size": [2, 2], "counts": [4.0]})


class SamResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "image.png"
        Image.new("RGB", (256, 256), color=(120, 130, 140)).save(self.image_path)
        from visual_peer_inspector.data.model_safe_package import file_sha256

        self.configuration = Sam2Configuration(device="cpu", precision="float32")
        self.job = SamImageJob(
            opaque_scene_id="S000001",
            view_id="V00",
            image_path=self.image_path,
            image_sha256=file_sha256(self.image_path),
            prompts=(
                SamPrompt("reference_box", "R000", (2, 3, 20, 30)),
                SamPrompt("automatic_box", "P000", (5, 7, 22, 32)),
                SamPrompt("automatic_box", "P001", (40, 50, 70, 80)),
            ),
            prompt_package_sha256="a" * 64,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_store(self, name="results"):
        return SamResultStore(
            self.root / name,
            self.configuration.sha256,
            self.job.prompt_package_sha256,
            CONTRACT_PRODUCER_KIND,
        )

    def test_contract_worker_preserves_every_prompt_and_resumes(self):
        worker = ContractSamWorker(self.configuration)
        store = self.make_store()

        attempts = run_sam_jobs(worker=worker, jobs=[self.job], result_store=store)
        store.reconcile_jobs([self.job])
        index_path = store.write_index(self.root / "mask_results.jsonl")
        resumed = self.make_store()
        resumed.reconcile_jobs([self.job])

        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(store.rows), 1)
        self.assertEqual(store.rows[0]["prompt_count"], 3)
        self.assertEqual(len(index_path.read_text().splitlines()), 1)
        self.assertEqual(len(resumed.rows), 1)

    def test_repeat_comparison_requires_exact_mask_digests(self):
        worker = ContractSamWorker(self.configuration)
        first = worker.process(self.job, 1)
        second = worker.process(self.job, 1)

        summary = compare_repeat_results([first], [second])

        self.assertEqual(summary["mask_digest_agreement_rate"], 1.0)
        self.assertEqual(summary["score_agreement_rate"], 1.0)

    def test_tampered_mask_fails_store_validation(self):
        worker = ContractSamWorker(self.configuration)
        row = worker.process(self.job, 1)
        row["routes"]["automatic_box"][0]["mask_sha256"] = "0" * 64
        store = self.make_store()

        with self.assertRaisesRegex(ValueError, "hash differs"):
            store.write(self.job, row)

    def test_empty_mask_is_preserved_as_an_evaluation_outcome(self):
        row = ContractSamWorker(self.configuration).process(self.job, 1)
        item = row["routes"]["automatic_box"][0]
        empty = np.zeros((256, 256), dtype=bool)
        item["mask_rle"] = encode_binary_mask(empty)
        item["mask_sha256"] = binary_mask_sha256(empty)
        item["foreground_pixels"] = 0
        item["mask_status"] = "empty"

        store = self.make_store()
        store.write(self.job, row)

        self.assertEqual(store.rows[0]["routes"]["automatic_box"][0]["mask_status"], "empty")

    def test_tampered_prompt_box_fails_store_validation(self):
        row = ContractSamWorker(self.configuration).process(self.job, 1)
        row["routes"]["automatic_box"][0]["box_xyxy"] = [1, 1, 2, 2]

        with self.assertRaisesRegex(ValueError, "prompt keys differ"):
            self.make_store().write(self.job, row)

    def test_tampered_foreground_count_fails_store_validation(self):
        row = ContractSamWorker(self.configuration).process(self.job, 1)
        row["routes"]["automatic_box"][0]["foreground_pixels"] += 1

        with self.assertRaisesRegex(ValueError, "foreground count"):
            self.make_store().write(self.job, row)

    def test_incomplete_file_does_not_mark_image_complete(self):
        incomplete = self.root / "results" / "S000001" / "V00.json.incomplete"
        incomplete.parent.mkdir(parents=True)
        incomplete.write_text("partial", encoding="utf-8")

        store = self.make_store()

        self.assertEqual(store.completed_keys, set())

    def test_evidence_contains_only_aggregate_rows(self):
        row = ContractSamWorker(self.configuration).process(self.job, 1)

        evidence = collect_sam_evidence([row])
        serialized = json.dumps(evidence["route_rows"])

        self.assertEqual(evidence["image_count"], 1)
        self.assertNotIn("S000001", serialized)
        self.assertNotIn("opaque_scene_id", serialized)


class SamQualitativeSelectionTests(unittest.TestCase):
    def test_examples_are_evenly_spaced_and_deterministic(self):
        rows = [
            {"opaque_scene_id": f"S{index:06d}", "view_id": f"V{index:02d}"}
            for index in range(20)
        ]

        selected = _evenly_spaced_rows(list(reversed(rows)), 6)

        self.assertEqual(selected[0], rows[0])
        self.assertEqual(selected[-1], rows[-1])
        self.assertEqual(len({row["opaque_scene_id"] for row in selected}), 6)

    def test_invalid_example_count_is_rejected(self):
        rows = [{"opaque_scene_id": "S000001", "view_id": "V00"}]

        with self.assertRaisesRegex(ValueError, "outside"):
            _evenly_spaced_rows(rows, 2)


if __name__ == "__main__":
    unittest.main()
