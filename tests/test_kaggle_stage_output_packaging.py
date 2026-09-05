from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_kaggle_stage_output.py"
SPEC = importlib.util.spec_from_file_location("package_kaggle_stage_output", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def _stage_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage"
    safe = root / "safe"
    safe.mkdir(parents=True)
    (safe / "summary.csv").write_text("metric,value\nmean_ap,0.7\n", encoding="utf-8")
    (safe / "run_record.json").write_text(
        json.dumps({"schema_version": 1, "stage": "test", "status": "completed"}),
        encoding="utf-8",
    )
    rows = []
    for path in sorted(safe.iterdir()):
        rows.append({
            "path": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
    (safe / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}),
        encoding="utf-8",
    )
    private = root / "private" / "figures"
    private.mkdir(parents=True)
    (private / "cases.png").write_bytes(b"private review panel")
    downstream = root / "model-input"
    downstream.mkdir()
    (downstream / "package_record.json").write_text('{"status":"complete"}', encoding="utf-8")
    return root


class KaggleStageOutputPackagingTests(unittest.TestCase):
    def test_bounded_kaggle_roots_supports_nested_dataset_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "datasets" / "owner" / "dataset" / "versions" / "3"
            package.mkdir(parents=True)
            (package / "package_record.json").write_text("{}", encoding="utf-8")
            roots = packager.bounded_kaggle_roots(root)
            self.assertIn(package.resolve(), roots)
            self.assertEqual(
                packager.find_bounded_marker_root(root, "package_record.json"),
                package.resolve(),
            )

    def test_build_bundles_writes_compact_verified_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = _stage_root(root)
            output = root / "working"
            record = packager.build_bundles(
                stage_root=stage,
                output_root=output,
                stage_name="test-stage",
                review_paths=["private/figures/cases.png"],
                downstream_paths=["model-input"],
            )
            self.assertEqual(record["safe_file_count"], 3)
            self.assertEqual(record["review_file_count"], 1)
            self.assertEqual(record["downstream_file_count"], 1)
            self.assertEqual(len(record["archives"]), 3)
            for row in record["archives"]:
                archive_path = output / row["path"]
                self.assertEqual(packager.file_sha256(archive_path), row["sha256"])
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertIsNone(archive.testzip())
            checksum_lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(checksum_lines), 4)

    def test_safe_bundle_rejects_protected_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = _stage_root(root)
            safe = stage / "safe"
            (safe / "summary.csv").write_text("opaque_scene_id,value\nS000001,1\n", encoding="utf-8")
            rows = []
            for path in sorted(item for item in safe.iterdir() if item.name != "artifact_manifest.json"):
                rows.append({
                    "path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                })
            (safe / "artifact_manifest.json").write_text(
                json.dumps({"schema_version": 1, "artifact_count": len(rows), "artifacts": rows}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Protected detail"):
                packager.build_bundles(
                    stage_root=stage,
                    output_root=root / "working",
                    stage_name="test-stage",
                )

    def test_confirmation_lock_does_not_replace_kaggle_cuda_stack(self) -> None:
        model_lines = {
            line.strip().split("==", 1)[0].lower()
            for line in (ROOT / "requirements/confirmation-lock.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(model_lines.isdisjoint({"torch", "torchvision", "numpy", "pillow"}))
        tracking_lines = [
            line.strip()
            for line in (ROOT / "requirements/confirmation-tracking-lock.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(tracking_lines, ["clearml==2.1.10"])


if __name__ == "__main__":
    unittest.main()
