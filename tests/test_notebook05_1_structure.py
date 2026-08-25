from __future__ import annotations

import json
from pathlib import Path
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = (
    CODE_ROOT
    / "notebooks/detection-and-segmentation/toysad8k-ds1-sam21"
)
NOTEBOOK_PATH = NOTEBOOK_ROOT / "05.1-toysad8k-ds1-sam21-segmentation.ipynb"


class Notebook051StructureTests(unittest.TestCase):
    def test_notebook_is_clean_compilable_and_fail_closed(self) -> None:
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
        self.assertIn("verify_ds1_sam_prompt_package", source)
        self.assertIn("verify_model_input_package", source)
        self.assertIn("Notebook 05.1 requires a clean committed code snapshot", source)
        self.assertIn("NOTEBOOK_RELATIVE", source)
        self.assertIn("import_ds1_sam_resume", source)
        self.assertNotIn("UserSecretsClient", source)
        self.assertNotIn("get_secret", source)

    def test_tracking_is_after_compute_and_bounded(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertLess(
            source.index("run_ds1_sam_workflow("),
            source.index("finalize_ds1_sam_clearml.py"),
        )
        self.assertIn('"--credentials-source"', source)
        self.assertIn("timeout=240", source)
        self.assertIn('OUTPUT_ROOT / "private" / "tracking_receipt.json"', source)

    def test_kernel_uses_the_frozen_ds1_model_input(self) -> None:
        metadata = json.loads(
            (NOTEBOOK_ROOT / "kernel-metadata.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "esreekarreddy/vpi-toysad-ds1-screen100-model-input",
            metadata["dataset_sources"],
        )
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")


if __name__ == "__main__":
    unittest.main()
