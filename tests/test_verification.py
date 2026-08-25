from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.verification import (
    file_sha256,
    verify_artifact_manifest,
    verify_code_snapshot_record,
)


class VerificationTests(unittest.TestCase):
    def test_code_snapshot_record_detects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            digest = file_sha256(source)
            payload = f"{digest}  module.py\n".encode("utf-8")
            record = {
                "file_count": 1,
                "files": {"module.py": digest},
                "snapshot_sha256": sha256(payload).hexdigest(),
            }
            record_path = root / "snapshot_record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            verify_code_snapshot_record(root, record_path)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify_code_snapshot_record(root, record_path)

    def test_code_snapshot_record_rejects_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = {
                "file_count": 1,
                "files": {"../outside.py": "0" * 64},
                "snapshot_sha256": "0" * 64,
            }
            record_path = root / "snapshot_record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
                verify_code_snapshot_record(root, record_path)

    def test_code_snapshot_record_rejects_undeclared_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            digest = file_sha256(source)
            payload = f"{digest}  module.py\n".encode("utf-8")
            record_path = root / "snapshot_record.json"
            record_path.write_text(
                json.dumps(
                    {
                        "file_count": 1,
                        "files": {"module.py": digest},
                        "snapshot_sha256": sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            (root / "undeclared.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "undeclared"):
                verify_code_snapshot_record(root, record_path)

    def test_code_snapshot_record_rejects_nested_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            record_path = nested / "snapshot_record.json"
            record_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root snapshot_record"):
                verify_code_snapshot_record(root, record_path)

    def test_artifact_manifest_requires_complete_report_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report"
            report.mkdir()
            table = report / "table.csv"
            table.write_text("status\npass\n", encoding="utf-8")
            manifest_path = report / "artifact_manifest.json"
            manifest_path.write_text(
                json.dumps({"report/table.csv": file_sha256(table)}),
                encoding="utf-8",
            )
            verify_artifact_manifest(root)
            (report / "unlisted.png").write_bytes(b"not-an-image")
            with self.assertRaisesRegex(ValueError, "absent from"):
                verify_artifact_manifest(root)


if __name__ == "__main__":
    unittest.main()
