from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.integrity import verify_chunk_directory


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


class VerifyChunkDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.first = b"first test segment"
        self.second = b"second test segment"
        self.manifest = {
            "dataset": "test-dataset",
            "segment_prefix": "segment_",
            "expected_total_bytes": len(self.first) + len(self.second),
            "files": [
                {
                    "name": "segment_aa",
                    "size_bytes": len(self.first),
                    "sha256": digest(self.first),
                },
                {
                    "name": "segment_ab",
                    "size_bytes": len(self.second),
                    "sha256": digest(self.second),
                },
            ],
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_valid_files(self) -> None:
        (self.directory / "segment_aa").write_bytes(self.first)
        (self.directory / "segment_ab").write_bytes(self.second)

    def test_valid_directory_passes(self) -> None:
        self.write_valid_files()

        report = verify_chunk_directory(self.directory, self.manifest)

        self.assertTrue(report.ok)
        self.assertEqual(report.actual_total_bytes, report.expected_total_bytes)

    def test_missing_segment_fails(self) -> None:
        (self.directory / "segment_aa").write_bytes(self.first)

        report = verify_chunk_directory(self.directory, self.manifest)

        self.assertFalse(report.ok)
        self.assertIn("missing", report.files[1].errors)

    def test_same_size_corruption_fails_hash_check(self) -> None:
        self.write_valid_files()
        corrupted = b"X" * len(self.second)
        (self.directory / "segment_ab").write_bytes(corrupted)

        report = verify_chunk_directory(self.directory, self.manifest)

        self.assertFalse(report.ok)
        self.assertIn("SHA-256 mismatch", report.files[1].errors)

    def test_unexpected_and_incomplete_files_fail(self) -> None:
        self.write_valid_files()
        (self.directory / "segment_ac").write_bytes(b"unexpected")
        (self.directory / "copy.incomplete").write_bytes(b"partial")

        report = verify_chunk_directory(self.directory, self.manifest)

        self.assertFalse(report.ok)
        self.assertEqual(report.unexpected_segments, ("segment_ac",))
        self.assertEqual(report.incomplete_files, ("copy.incomplete",))


if __name__ == "__main__":
    unittest.main()
