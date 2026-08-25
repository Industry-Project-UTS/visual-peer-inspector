from __future__ import annotations

import json
from pathlib import Path
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / (
    "notebooks/detection-and-segmentation/toysad8k-ds1-locateanything/"
    "04.1-toysad8k-ds1-locateanything-screen.ipynb"
)


class Notebook041StructureTests(unittest.TestCase):
    def test_failed_acceptance_preserves_downloadable_evidence(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertNotIn(
            "raise RuntimeError(\"Notebook 04.1 did not pass the full DS1 operational contract\")",
            source,
        )
        self.assertIn("this run is not promotable", source)
        self.assertIn("must not feed the SAM stage", source)
        self.assertIn("different deterministic seed", source)


if __name__ == "__main__":
    unittest.main()
