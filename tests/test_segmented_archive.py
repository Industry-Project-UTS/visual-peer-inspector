from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data.archive_index import (
    build_archive_index,
    read_central_directory_record,
)
from visual_peer_inspector.data.archive_members import (
    extract_indexed_member,
    load_indexed_member,
    read_indexed_member,
)
from visual_peer_inspector.data.segmented_file import SegmentedFile
from visual_peer_inspector.data.scene_catalog import (
    anomaly_variant,
    load_split_mapping,
    scene2d_anomaly_flags,
)


class SegmentedArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def split_bytes(self, data: bytes, segment_size: int) -> list[Path]:
        paths: list[Path] = []
        for index, start in enumerate(range(0, len(data), segment_size)):
            path = self.directory / f"part_{index:02d}"
            path.write_bytes(data[start : start + segment_size])
            paths.append(path)
        return paths

    def make_zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("toysad8k/train/scene-001/RGB/0001.png", b"fake-png")
            archive.writestr(
                "toysad8k/train/scene-001/metadata.json",
                json.dumps({"scene_id": "scene-001"}),
            )
        return buffer.getvalue()

    def test_reads_and_seeks_across_boundaries(self) -> None:
        original = b"abcdefghijklmnopqrstuvwxyz"
        paths = self.split_bytes(original, 5)

        with SegmentedFile(paths) as stream:
            stream.seek(3)
            self.assertEqual(stream.read(9), original[3:12])
            stream.seek(-4, io.SEEK_END)
            self.assertEqual(stream.read(), original[-4:])
            stream.seek(5)
            target = bytearray(7)
            self.assertEqual(stream.readinto(target), 7)
            self.assertEqual(bytes(target), original[5:12])

    def test_expected_size_mismatch_is_rejected(self) -> None:
        paths = self.split_bytes(b"abcdef", 3)

        with self.assertRaisesRegex(ValueError, "Segment size mismatch"):
            SegmentedFile(paths, expected_sizes=[3, 4])

    def test_anomaly_variant_matches_official_hyphen_contract(self) -> None:
        self.assertIsNone(anomaly_variant("donut_025/donut_025.blend"))
        self.assertEqual(
            anomaly_variant("donut_025/donut_025-disc-001.blend"),
            "disc-001",
        )

    def test_scene2d_zero_label_means_anomaly(self) -> None:
        self.assertEqual(scene2d_anomaly_flags([0, 1, 1]), [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "binary"):
            scene2d_anomaly_flags([2])

    def test_split_mapping_rejects_cross_split_duplicates(self) -> None:
        split_path = self.directory / "split.json"
        split_path.write_text(
            json.dumps({"train": ["scene-a"], "seen-test": ["scene-a"]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "multiple official splits"):
            load_split_mapping(split_path)

    def test_python_zipfile_reads_virtual_concatenation(self) -> None:
        archive_bytes = self.make_zip()
        paths = self.split_bytes(archive_bytes, 17)

        with SegmentedFile(paths) as stream, zipfile.ZipFile(stream) as archive:
            self.assertEqual(
                archive.read("toysad8k/train/scene-001/RGB/0001.png"),
                b"fake-png",
            )
            self.assertIsNone(archive.testzip())

    def test_streaming_index_matches_synthetic_zip(self) -> None:
        archive_bytes = self.make_zip()
        paths = self.split_bytes(archive_bytes, 23)
        database_path = self.directory / "index.sqlite3"
        summary_path = self.directory / "summary.json"

        with SegmentedFile(paths) as stream:
            record = read_central_directory_record(stream)
            self.assertEqual(record.total_entries, 2)
            summary = build_archive_index(stream, database_path, summary_path)

        self.assertEqual(summary["entries_indexed"], 2)
        self.assertEqual(summary["unsafe_path_entries"], 0)
        self.assertEqual(summary["duplicate_name_groups"], 0)
        with sqlite3.connect(database_path) as connection:
            names = [row[0] for row in connection.execute("SELECT name FROM members")]
        self.assertEqual(
            names,
            [
                "toysad8k/train/scene-001/RGB/0001.png",
                "toysad8k/train/scene-001/metadata.json",
            ],
        )

    def test_indexed_member_read_and_safe_extraction(self) -> None:
        archive_bytes = self.make_zip()
        paths = self.split_bytes(archive_bytes, 19)
        database_path = self.directory / "member-index.sqlite3"
        summary_path = self.directory / "member-summary.json"

        with SegmentedFile(paths) as stream:
            build_archive_index(stream, database_path, summary_path)
            member = load_indexed_member(
                database_path,
                "toysad8k/train/scene-001/metadata.json",
            )
            content = read_indexed_member(stream, member)
            self.assertEqual(json.loads(content), {"scene_id": "scene-001"})
            output_root = self.directory / "selected"
            output = extract_indexed_member(stream, member, output_root)

        self.assertTrue(output.is_file())
        self.assertEqual(json.loads(output.read_bytes()), {"scene_id": "scene-001"})
        with SegmentedFile(paths) as stream:
            with self.assertRaises(FileExistsError):
                extract_indexed_member(stream, member, output_root)


if __name__ == "__main__":
    unittest.main()
