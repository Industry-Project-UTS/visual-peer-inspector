from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ds1_locateanything_prompt_freeze",
    ROOT / "scripts/run_ds1_locateanything_prompt_freeze.py",
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PromptFreezeRunnerTests(unittest.TestCase):
    def test_plan_reuses_object_and_generates_only_one_new_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline/private/proposal_results.jsonl"
            paths = RUNNER.candidate_result_paths(root / "study", baseline)

        self.assertEqual(paths["object"], baseline.resolve())
        self.assertEqual(set(paths).difference({"object"}), set(RUNNER.NEW_PROMPT_KEYS))
        self.assertTrue(
            paths["individual_object"].as_posix().endswith(
                "prompt_runs/individual_object/private/proposal_results.jsonl"
            )
        )

    def test_integrity_gate_does_not_apply_a_zero_box_rate_threshold(self) -> None:
        summary = {
            "job_count": 2_000,
            "successful_job_count": 2_000,
            "failed_job_count": 0,
            "possibly_truncated_count": 0,
            "zero_proposal_image_count": 7,
        }
        RUNNER._require_complete(summary, 2_000, "test")

    def test_loaded_worker_can_change_only_the_prompt_configuration(self) -> None:
        worker = object.__new__(RUNNER.LocateAnythingWorker)
        worker.configuration = RUNNER.prompt_configurations()["object"]
        model_marker = object()
        worker.model = model_marker

        worker.use_configuration(RUNNER.prompt_configurations()["individual_object"])

        self.assertIs(worker.model, model_marker)
        self.assertEqual(worker.configuration.prompt, "individual object")
        with self.assertRaisesRegex(ValueError, "model identity"):
            worker.use_configuration(replace(worker.configuration, model_id="another/model"))


if __name__ == "__main__":
    unittest.main()
