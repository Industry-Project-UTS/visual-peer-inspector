from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.data import load_toys_scene_view


class SelectedExampleLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.scene = self.root / "toysad8k" / "widget_001-001"
        (self.scene / "RGB").mkdir(parents=True)
        (self.scene / "masks").mkdir()
        for object_name in ("object", "object.001"):
            path = self.scene / "segmentations" / object_name
            path.mkdir(parents=True)
            (path / f"{object_name}_0000.png").write_bytes(b"png")
        (self.scene / "RGB" / "0000.png").write_bytes(b"png")
        (self.scene / "masks" / "0000.png").write_bytes(b"png")
        (self.scene / "scene2d.metadata.json").write_text(
            json.dumps(
                {
                    "anomaly_labels": [0, 1],
                    "blend_paths": [
                        "widget_001/widget_001-miss-001.blend",
                        "widget_001/widget_001.blend",
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.scene3d_path = self.scene / "scene3d.metadata.json"
        self.scene3d_path.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_one_view_with_aligned_labels(self) -> None:
        loaded = load_toys_scene_view(
            self.root,
            "widget_001-001",
            split="train",
            view_id="0000",
        )

        self.assertEqual(loaded.anomaly_count, 1)
        self.assertTrue(loaded.objects[0].is_anomaly)
        self.assertEqual(loaded.objects[0].anomaly_variant, "miss-001")
        self.assertFalse(loaded.objects[1].is_anomaly)
        self.assertIsNone(loaded.objects[1].anomaly_variant)

    def test_rejects_scene2d_scene3d_order_mismatch(self) -> None:
        metadata = json.loads(self.scene3d_path.read_text(encoding="utf-8"))
        metadata["objects"].reverse()
        self.scene3d_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "path mismatch"):
            load_toys_scene_view(
                self.root,
                "widget_001-001",
                split="train",
                view_id="0000",
            )


if __name__ == "__main__":
    unittest.main()
