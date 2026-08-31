from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / (
    "notebooks/detection-and-segmentation/toysad8k-ds1-proposal-refinement/"
    "04.2-toysad8k-ds1-proposal-refinement.ipynb"
)


class Notebook042StructureTests(unittest.TestCase):
    def test_notebook_is_clean_and_explains_the_full_experiment(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("04.2-toysad8k-ds1-proposal-refinement.ipynb", source)
        self.assertIn("all 100 DS1 scenes and all 2,000 images", source)
        self.assertIn("paired confidence intervals", source)
        self.assertIn("Private before-and-after examples", source)
        self.assertIn("Safe ClearML finalisation", source)
        self.assertNotIn("—", source)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code"))

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
