from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.evidence import (
    save_bar_chart,
    save_histogram,
    save_image_grid,
    save_scatter_plot,
    write_csv_rows,
)


class EvidenceTests(unittest.TestCase):
    def test_tables_and_figures_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "gates.csv"
            write_csv_rows(
                csv_path,
                ["gate", "status"],
                [{"gate": "image count", "status": "pass"}],
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["status"], "pass")

            bar_path = root / "bars.png"
            histogram_path = root / "histogram.png"
            source_path = root / "source.png"
            grid_path = root / "grid.png"
            scatter_path = root / "scatter.png"
            Image.new("RGB", (32, 32), "white").save(source_path)
            save_bar_chart(
                ["V00", "V01"],
                [20, 20],
                bar_path,
                title="View coverage",
                y_label="Scenes",
            )
            save_histogram(
                [1, 2, 2, 3],
                histogram_path,
                title="Image size",
                x_label="KiB",
            )
            save_image_grid(
                [{"label": "Scene 01 / V00", "path": source_path, "boxes": [[2, 2, 20, 20]]}],
                grid_path,
                title="Private proposal examples",
            )
            save_scatter_plot(
                [1, 2, 3],
                [0.2, 0.6, 0.4],
                scatter_path,
                title="Runtime by proposal count",
                x_label="Proposals",
                y_label="Seconds",
            )
            for path in (bar_path, histogram_path, grid_path, scatter_path):
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 100)


if __name__ == "__main__":
    unittest.main()
