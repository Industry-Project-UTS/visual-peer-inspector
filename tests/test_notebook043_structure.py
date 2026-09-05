from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything-prompt-freeze/"
    "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb"
)
METADATA = NOTEBOOK.parent / "kernel-metadata.json"


class Notebook043StructureTests(unittest.TestCase):
    def test_notebook_is_clean_and_covers_the_bounded_study(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        for phrase in (
            "04.3-toysad8k-ds1-locateanything-prompt-freeze.ipynb",
            "The model is loaded once",
            "one-image check",
            "20-image pilot",
            "full 2,000-image DS1 screen",
            "Private visual review",
            "Safe ClearML finalisation",
            "requires a Kaggle GPU",
            "must be confirmed unchanged on DS2",
            "matches_by_identity",
            '("neutral_corpus_manifest.jsonl", "media")',
            '("evaluator_crosswalk.jsonl", "masks")',
        ):
            self.assertIn(phrase, source)
        self.assertNotIn("—", source)
        self.assertTrue(
            all(
                not cell.get("outputs")
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")

    def test_kaggle_metadata_attaches_existing_inputs_and_gpu(self) -> None:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            metadata["kernel_sources"],
            [],
        )
        self.assertIn("esreekarreddy/vpi-toysad-ds1-mask-evaluator", metadata["dataset_sources"])
        self.assertIn("esreekarreddy/vpi-toysad-ds1-locateanything-output", metadata["dataset_sources"])


if __name__ == "__main__":
    unittest.main()
