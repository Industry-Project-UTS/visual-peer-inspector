from __future__ import annotations

import json
from pathlib import Path
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    CODE_ROOT
    / "notebooks/data-preparation/toysad8k-ds1-opaque-package/"
    / "03.1-toysad8k-ds1-opaque-package.ipynb"
)


class Notebook031StructureTests(unittest.TestCase):
    def test_notebook_is_clean_compilable_and_privacy_separated(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        source = "\n".join("".join(cell["source"]) for cell in code_cells)
        for index, cell in enumerate(code_cells, 1):
            compile(
                "".join(cell["source"]),
                f"{NOTEBOOK_PATH.name}:code-cell-{index}",
                "exec",
            )
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
        self.assertIn('SAFE_ROOT = OUTPUT_ROOT / "safe"', source)
        self.assertIn('PRIVATE_ROOT = OUTPUT_ROOT / "private"', source)
        self.assertIn("finalize_ds1_opaque_package_clearml.py", source)
        self.assertIn("communicate(timeout=210)", source)
        self.assertNotIn("prepare_clearml_tracking", source)
        self.assertNotIn("ExperimentTracker.start", source)
        self.assertNotIn("display(Image.open(contact_sheet))", source)
        self.assertNotIn("if finalizer_return_code != 0:\n    raise RuntimeError", source)
        self.assertIn("must not be promoted until tracking succeeds", source)

    def test_clearml_finalizer_occurs_after_safe_manifest(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertLess(
            source.index("safe_manifest = write_strict_manifest"),
            source.index("finalize_ds1_opaque_package_clearml.py"),
        )


if __name__ == "__main__":
    unittest.main()
