from __future__ import annotations

import json
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/development-freeze/08e-toysad8k-frozen-development-pipeline.ipynb"
METADATA = ROOT / "notebooks/development-freeze/kernel-metadata-08e.json"


class Notebook08EStructureTests(unittest.TestCase):
    def test_notebook_contains_the_complete_frozen_route(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join("".join(cell.source) for cell in notebook.cells)
        for phrase in (
            "100-scene screen",
            "600-scene development remainder",
            "LocateAnything-3B",
            "SAM 2.1",
            "DINOv2-small",
            "B0 and P1",
            "prediction_manifest.jsonl",
            "global_embeddings.npy",
            "embedding_index.jsonl",
            "prediction_lock.json",
            "zero-proposal",
            "empty-mask",
            "feature failures",
            "reference-only",
            "Backfill the confirmed 300-scene route",
            "requires a complete confirmation rerun",
            "ClearML is not active during model loading or inference",
            "VPI_RESUME_ROOT",
            "verified recovery ZIP",
            "maximum new shards per stage",
            "not published before exact full coverage",
            "VPI_DS2_RESUME_ROOT",
            "6,000-view DS2 representation backfill",
            "bounded model preflight",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            self.assertIn(phrase.lower(), source.lower())
        self.assertNotIn("0.95", source)
        self.assertNotIn("legacy", source.lower())
        self.assertIn('"--run-mode", RUN_PHASE', source)
        self.assertIn('"--recovery-output-directory"', source)
        self.assertIn('"--run-mode", DS2_BACKFILL_PHASE', source)
        self.assertIn('recovery_safe = OUTPUT_ROOT / "tracking-safe"', source)
        self.assertNotIn("recovery_summary.json", source)
        self.assertIn('OUTPUT_ROOT / "tracking/clearml_recovery_receipt.json"', source)
        self.assertNotIn("write_safe_artifact_manifest(recovery_safe)", source)
        self.assertIn("timeout=1800", source)
        self.assertLess(
            source.index("subprocess.run(preflight_command"),
            source.index("subprocess.run(command, check=True, env=inference_environment)"),
        )

    def test_notebook_has_professional_cell_structure(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        headings = [
            line
            for cell in notebook.cells
            if cell.cell_type == "markdown"
            for line in "".join(cell.source).splitlines()
            if line.startswith("## ")
        ]
        self.assertGreaterEqual(len(headings), 9)
        self.assertGreaterEqual(len(notebook.cells), 16)
        self.assertEqual(notebook.cells[0].cell_type, "markdown")
        self.assertTrue(all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code"))

    def test_kaggle_metadata_enables_one_t4_gpu(self) -> None:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        self.assertEqual(metadata["code_file"], NOTEBOOK.name)
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["enable_internet"])
        self.assertIn(
            "esreekarreddy/toysad-8k-ds2-frozen-pipeline",
            metadata["kernel_sources"],
        )

    def test_runner_and_workflow_compile(self) -> None:
        compile(
            (ROOT / "scripts/run_frozen_development_pipeline.py").read_text(encoding="utf-8"),
            "run_frozen_development_pipeline.py",
            "exec",
        )
        compile(
            (ROOT / "src/visual_peer_inspector/workflows/frozen_development_pipeline.py").read_text(encoding="utf-8"),
            "frozen_development_pipeline.py",
            "exec",
        )
        runner = (ROOT / "scripts/run_frozen_development_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"stage": "frozen_development_pipeline"', runner)
        self.assertIn("write_safe_artifact_manifest", runner)
        self.assertIn("write_recovery_tracking_root", runner)
        self.assertIn("def run_preflight", runner)
        self.assertIn("models_loaded_sequentially", runner)


if __name__ == "__main__":
    unittest.main()
