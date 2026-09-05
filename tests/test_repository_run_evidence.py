from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_repository_run_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_repository_run_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class RepositoryRunEvidenceTests(unittest.TestCase):
    def test_repository_evidence_is_complete_and_safe(self) -> None:
        summary = verifier.verify()
        self.assertEqual(summary["publication_notebooks"], 10)
        self.assertEqual(summary["execution_transcripts"], 10)
        self.assertEqual(summary["publication_receipts"], 10)
        self.assertEqual(summary["clearml_receipts"], 7)


if __name__ == "__main__":
    unittest.main()
