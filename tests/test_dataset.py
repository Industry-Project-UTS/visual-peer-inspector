from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector import load_scene_view


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.scene = self.root / "toysad8k" / "widget_001-001"
        (self.scene / "RGB").mkdir(parents=True)
        (self.scene / "RGB" / "0000.png").write_bytes(b"image")
        for name in ("object", "object.001"):
            mask_directory = self.scene / "segmentations" / name
            mask_directory.mkdir(parents=True)
            (mask_directory / f"{name}_0000.png").write_bytes(b"mask")

        self.scene2d = {
            "anomaly_labels": [0, 1],
            "blend_paths": [
                "widget_001/widget_001-miss-001.blend",
                "widget_001/widget_001.blend",
            ],
        }
        self.scene3d = {
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
        self._write_metadata()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_metadata(self) -> None:
        (self.scene / "scene2d.metadata.json").write_text(
            json.dumps(self.scene2d),
            encoding="utf-8",
        )
        (self.scene / "scene3d.metadata.json").write_text(
            json.dumps(self.scene3d),
            encoding="utf-8",
        )

    def test_loads_aligned_objects_and_labels(self) -> None:
        scene = load_scene_view(
            self.root,
            "widget_001-001",
            split="train",
        )
        self.assertEqual(scene.anomaly_count, 1)
        self.assertTrue(scene.objects[0].is_anomaly)
        self.assertEqual(scene.objects[0].anomaly_variant, "miss-001")
        self.assertFalse(scene.objects[1].is_anomaly)

    def test_rejects_metadata_order_mismatch(self) -> None:
        self.scene3d["objects"].reverse()
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, "path mismatch"):
            load_scene_view(self.root, "widget_001-001", split="train")


if __name__ == "__main__":
    unittest.main()
