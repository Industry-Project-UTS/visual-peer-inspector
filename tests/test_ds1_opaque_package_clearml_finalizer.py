from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "finalize_ds1_opaque_package_clearml",
    CODE_ROOT / "scripts/finalize_ds1_opaque_package_clearml.py",
)
assert SPEC and SPEC.loader
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


class Ds1OpaquePackageClearmlFinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "safe"
        self.root.mkdir()
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_record(self) -> dict:
        return {
            "schema_version": 1,
            "stage": "ds1_opaque_package",
            "status": "completed",
            "tracking_status": "handled_by_bounded_finalizer",
            "execution_environment": "kaggle",
            "provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "precommit_candidate": False,
                "snapshot_sha256": "b" * 64,
            },
            "configuration": {
                "scene_count": 100,
                "views_per_scene": 20,
                "expected_image_count": 2_000,
            },
            "package": {
                "cohort_id": FINALIZER.EXPECTED_COHORT_ID,
                "release_id": FINALIZER.EXPECTED_RELEASE_ID,
                "scene_count": 100,
                "view_count": 20,
                "image_count": 2_000,
                "verified_image_count": 2_000,
                "decoded_image_failure_count": 0,
                "all_checks_passed": True,
            },
            "cohort_summary": {
                "cohort_id": FINALIZER.EXPECTED_COHORT_ID,
                "membership_sha256": FINALIZER.EXPECTED_MEMBERSHIP_SHA256,
                "scene_count": 100,
                "asset_group_count": 27,
                "all_checks_passed": True,
            },
            "resume_summary": {
                "first_elapsed_seconds": 34.0,
                "second_elapsed_seconds": 10.0,
            },
        }

    def _write_manifest(self) -> None:
        rows = []
        for path in sorted(item for item in self.root.iterdir() if item.is_file()):
            if path.name == "artifact_manifest.json":
                continue
            rows.append(
                {
                    "path": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        (self.root / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_count": len(rows),
                    "artifacts": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_fixture(self) -> None:
        (self.root / "validation_gates.csv").write_text(
            "gate,status,observed,expected\nimage count,pass,2000,2000\n",
            encoding="utf-8",
        )
        (self.root / "view_coverage.csv").write_text(
            "view_id,scene_count,expected_scene_count,complete\nV00,100,100,True\n",
            encoding="utf-8",
        )
        (self.root / "media_summary.json").write_text(
            json.dumps({"image_count": 2_000, "total_media_mib": 246.0}),
            encoding="utf-8",
        )
        (self.root / "run_record.json").write_text(
            json.dumps(self._run_record()),
            encoding="utf-8",
        )
        for name in FINALIZER.ALLOWED_CHARTS:
            Image.new("RGB", (8, 8), "white").save(self.root / name)
        self._write_manifest()

    def test_complete_safe_root_passes(self) -> None:
        record = FINALIZER.validate_safe_root(self.root)
        self.assertEqual(record["status"], "completed")

    def test_dirty_snapshot_is_rejected_for_tracking(self) -> None:
        record = self._run_record()
        record["provenance"]["git_dirty"] = True
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "clean committed"):
            FINALIZER.validate_safe_root(self.root)

    def test_disabled_validation_can_review_a_dirty_local_run(self) -> None:
        record = self._run_record()
        record["provenance"]["git_dirty"] = True
        record["provenance"]["snapshot_sha256"] = "local-working-tree"
        (self.root / "run_record.json").write_text(json.dumps(record), encoding="utf-8")
        self._write_manifest()
        observed = FINALIZER.validate_safe_root(
            self.root,
            require_clean_snapshot=False,
        )
        self.assertEqual(observed["status"], "completed")

    def test_private_identifier_is_rejected(self) -> None:
        (self.root / "media_summary.json").write_text(
            json.dumps({"example": "S000021"}),
            encoding="utf-8",
        )
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Protected detail"):
            FINALIZER.validate_safe_root(self.root)

    def test_unexpected_file_is_rejected(self) -> None:
        (self.root / "private_contact_sheet.png").write_bytes(b"not-safe")
        with self.assertRaisesRegex(ValueError, "allowlist differs"):
            FINALIZER.validate_safe_root(self.root)

    def test_manifest_tampering_is_rejected(self) -> None:
        chart = self.root / "view_coverage.png"
        chart.write_bytes(chart.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            FINALIZER.validate_safe_root(self.root)


if __name__ == "__main__":
    unittest.main()
