from __future__ import annotations

from pathlib import Path
import sys
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.tracking import (
    ExperimentTracker,
    validate_safe_tracking_payload,
)


class SafeTrackingTests(unittest.TestCase):
    def test_disabled_tracking_needs_no_clearml_installation(self) -> None:
        tracker = ExperimentTracker.start(
            project_name="Visual Peer Inspector/01 Data Preparation",
            task_name="selection-audit-local",
            mode="disabled",
            properties={"release_id": "toysad8k-train-2000-v2"},
            configuration={"selected_scenes": 2000, "views_per_scene": 20},
        )

        tracker.report_scalar(
            title="Selection audit", series="selected scenes", value=2000
        )
        tracker.upload_summary("audit summary", {"all_checks_passed": True})
        tracker.close()

        self.assertEqual(tracker.mode, "disabled")
        self.assertIsNone(tracker.task_id)

    def test_unknown_tracking_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "VPI_TRACKING_MODE"):
            ExperimentTracker.start(
                project_name="Visual Peer Inspector/01 Data Preparation",
                task_name="selection-audit-local",
                mode="automatic",
            )

    def test_private_source_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe tracking value"):
            validate_safe_tracking_payload(
                {"input": "/kaggle/input/private-toysad/file.json"}
            )

    def test_source_object_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe tracking key"):
            validate_safe_tracking_payload({"source_object_index": 0})

    def test_private_label_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe tracking value"):
            validate_safe_tracking_payload(
                {"evidence_name": "labels_internal_lock.json"}
            )

    def test_aggregate_provenance_is_allowed(self) -> None:
        validate_safe_tracking_payload(
            {
                "release_id": "toysad8k-train-2000-v2",
                "manifest_sha256": "a" * 64,
                "selected_scenes": 2000,
                "required_indexed_path_count": 238000,
                "all_checks_passed": True,
                "kaggle_dataset_version": "team/toysad-safe/3",
            }
        )


if __name__ == "__main__":
    unittest.main()
