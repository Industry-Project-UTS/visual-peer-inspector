from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector.chunks import verify_chunks


class ChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.contents = [b"first", b"second"]
        for suffix, content in zip(("aa", "ab"), self.contents, strict=True):
            (self.directory / f"part_{suffix}").write_bytes(content)
        self.manifest = {
            "dataset": "sample",
            "segment_prefix": "part_",
            "expected_total_bytes": sum(map(len, self.contents)),
            "files": [
                {
                    "name": f"part_{suffix}",
                    "size_bytes": len(content),
                    "sha256": sha256(content).hexdigest(),
                }
                for suffix, content in zip(("aa", "ab"), self.contents, strict=True)
            ],
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_chunks_pass(self) -> None:
        report = verify_chunks(self.directory, self.manifest)
        self.assertTrue(report.ok)

    def test_changed_chunk_fails(self) -> None:
        (self.directory / "part_ab").write_bytes(b"changed")
        report = verify_chunks(self.directory, self.manifest)
        self.assertFalse(report.ok)
        self.assertIn("SHA-256 mismatch", report.files[1].errors)

    def test_incomplete_download_fails(self) -> None:
        (self.directory / "part_ac.incomplete").write_bytes(b"")
        report = verify_chunks(self.directory, self.manifest, check_hashes=False)
        self.assertFalse(report.ok)
        self.assertEqual(report.incomplete_files, ("part_ac.incomplete",))


if __name__ == "__main__":
    unittest.main()
