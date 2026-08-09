from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector import load_scene_view
from visual_peer_inspector.extraction import extract_cohort


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _archive_bytes(self) -> bytes:
        scene = "toysad8k/widget_001-001"
        scene2d = {
            "anomaly_labels": [0, 1],
            "blend_paths": [
                "widget_001/widget_001-miss-001.blend",
                "widget_001/widget_001.blend",
            ],
        }
        scene3d = {
            "objects": [
                {
                    "name": "object",
                    "path": "widget_001/widget_001-miss-001.blend",
                },
                {
                    "name": "object.001",
                    "path": "widget_001/widget_001.blend",
                },
            ]
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{scene}/RGB/0000.png", b"image")
            archive.writestr(f"{scene}/scene2d.metadata.json", json.dumps(scene2d))
            archive.writestr(f"{scene}/scene3d.metadata.json", json.dumps(scene3d))
            archive.writestr(
                f"{scene}/segmentations/object/object_0000.png",
                b"mask-one",
            )
            archive.writestr(
                f"{scene}/segmentations/object.001/object.001_0000.png",
                b"mask-two",
            )
        return buffer.getvalue()

    def test_extracts_a_cohort_from_archive_chunks(self) -> None:
        archive = self._archive_bytes()
        split_at = len(archive) // 2
        contents = [archive[:split_at], archive[split_at:]]
        chunks = self.root / "chunks"
        chunks.mkdir()
        files = []
        for suffix, content in zip(("aa", "ab"), contents, strict=True):
            name = f"part_{suffix}"
            (chunks / name).write_bytes(content)
            files.append(
                {
                    "name": name,
                    "size_bytes": len(content),
                    "sha256": sha256(content).hexdigest(),
                }
            )

        chunk_manifest = {
            "dataset": "sample",
            "segment_prefix": "part_",
            "expected_total_bytes": len(archive),
            "files": files,
        }
        cohort = {
            "id": "sample-cohort",
            "dataset": "sample",
            "dataset_revision": "test",
            "split": "train",
            "view_id": "0000",
            "scenes": ["widget_001-001"],
        }
        output = self.root / "selected"
        report = extract_cohort(chunks, chunk_manifest, cohort, output)

        self.assertEqual(report["scene_count"], 1)
        self.assertEqual(report["file_count"], 5)
        scene = load_scene_view(output, "widget_001-001", split="train")
        self.assertEqual(scene.anomaly_count, 1)


if __name__ == "__main__":
    unittest.main()
