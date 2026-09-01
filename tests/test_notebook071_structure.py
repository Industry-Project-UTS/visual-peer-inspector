from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "notebooks/representation-and-scoring/toysad8k-ds1-peer-scores/"
    "07.1-toysad8k-ds1-representation-and-peer-scores.ipynb"
)


class Notebook071StructureTests(unittest.TestCase):
    def test_notebook_is_clean_and_contains_the_complete_screen(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("07.1-toysad8k-ds1-representation-and-peer-scores.ipynb", source)
        self.assertIn("B0, B1, and P1", source)
        self.assertIn("DINOv3 access is gated", source)
        self.assertIn("Empty masks and masks with fewer than four", source)
        self.assertIn("Safe ClearML finalisation", source)
        self.assertIn('"content_checksums_sha256" in record', source)
        self.assertIn('("neutral_corpus_manifest.jsonl", "content_checksums.sha256", "media")', source)
        self.assertNotIn("—", source)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"))

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
