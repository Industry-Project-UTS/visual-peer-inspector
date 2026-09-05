from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_executed_notebook.py"
SPEC = importlib.util.spec_from_file_location("publish_executed_notebook", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def _notebook(*, executed: bool, private_visual: bool = False) -> dict[str, object]:
    source = "display(private / 'example.png')\n" if private_visual else "print('done')\n"
    output = []
    if executed:
        output = [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": "aGVsbG8=",
                    "text/plain": ["/Users/student/private S000001 O01\n"],
                },
            }
        ]
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Evidence\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 1 if executed else None,
                "metadata": {},
                "outputs": output,
                "source": [source],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class PublishExecutedNotebookTests(unittest.TestCase):
    def test_private_visuals_and_protected_text_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_path = root / "clean.ipynb"
            executed_path = root / "executed.ipynb"
            output_path = root / "published.ipynb"
            log_path = root / "run.log"
            clean_path.write_text(
                json.dumps(_notebook(executed=False, private_visual=True)), encoding="utf-8"
            )
            executed_path.write_text(
                json.dumps(_notebook(executed=True, private_visual=True)), encoding="utf-8"
            )

            receipt = publisher.publish_notebook(
                clean_path=clean_path,
                executed_path=executed_path,
                output_path=output_path,
                log_path=log_path,
                label="test-stage",
            )

            published_text = output_path.read_text(encoding="utf-8")
            self.assertNotIn("aGVsbG8=", published_text)
            self.assertNotIn("/Users/student", published_text)
            self.assertNotIn("S000001", published_text)
            self.assertNotIn("O01", published_text)
            self.assertEqual(receipt["removed_private_visuals"], 1)
            self.assertEqual(receipt["retained_safe_visuals"], 0)

    def test_safe_aggregate_visual_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_path = root / "clean.ipynb"
            executed_path = root / "executed.ipynb"
            output_path = root / "published.ipynb"
            log_path = root / "run.log"
            clean_path.write_text(json.dumps(_notebook(executed=False)), encoding="utf-8")
            executed_path.write_text(json.dumps(_notebook(executed=True)), encoding="utf-8")

            receipt = publisher.publish_notebook(
                clean_path=clean_path,
                executed_path=executed_path,
                output_path=output_path,
                log_path=log_path,
                label="test-stage",
            )

            self.assertIn("aGVsbG8=", output_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["retained_safe_visuals"], 1)

    def test_source_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = _notebook(executed=False)
            executed = _notebook(executed=True)
            executed["cells"][1]["source"] = ["print('changed')\n"]
            clean_path = root / "clean.ipynb"
            executed_path = root / "executed.ipynb"
            clean_path.write_text(json.dumps(clean), encoding="utf-8")
            executed_path.write_text(json.dumps(executed), encoding="utf-8")

            with self.assertRaisesRegex(publisher.PublicationError, "cell sources differ"):
                publisher.publish_notebook(
                    clean_path=clean_path,
                    executed_path=executed_path,
                    output_path=root / "published.ipynb",
                    log_path=root / "run.log",
                    label="test-stage",
                )


if __name__ == "__main__":
    unittest.main()
