from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_repository_run_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_repository_run_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class RepositoryRunEvidenceTests(unittest.TestCase):
    def test_later_evidence_sets_do_not_change_the_original_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "reports/run-evidence"
            evidence.mkdir(parents=True)
            old = root / "notebooks/earlier/run-executed.ipynb"
            new = root / "notebooks/executed/development-freeze/08f-executed.ipynb"
            for path in (old, new):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            (evidence / "evidence_manifest.json").write_text(json.dumps({
                "files": [{"path": old.relative_to(root).as_posix()}],
            }))
            with patch.object(verifier, "ROOT", root), patch.object(verifier, "EVIDENCE_ROOT", evidence):
                self.assertEqual(verifier._registered_publications(), [old])

    def test_rejects_duplicate_notebooks_in_the_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "reports/run-evidence"
            evidence.mkdir(parents=True)
            (evidence / "evidence_manifest.json").write_text(json.dumps({
                "files": [{"path": "notebooks/run-executed.ipynb"}] * 2,
            }))
            with (
                patch.object(verifier, "ROOT", root),
                patch.object(verifier, "EVIDENCE_ROOT", evidence),
                self.assertRaisesRegex(verifier.EvidenceVerificationError, "duplicates"),
            ):
                verifier._registered_publications()

    def test_repository_evidence_is_complete_and_safe(self) -> None:
        summary = verifier.verify()
        self.assertEqual(summary["publication_notebooks"], 10)
        self.assertEqual(summary["execution_transcripts"], 10)
        self.assertEqual(summary["publication_receipts"], 10)
        self.assertEqual(summary["clearml_receipts"], 7)


if __name__ == "__main__":
    unittest.main()
