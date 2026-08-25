from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.tracking import (
    ClearMLTrackingFailure,
    ExperimentTracker,
    OFFICIAL_CLEARML_HOSTS,
    configure_official_clearml_hosts,
    initialize_clearml_task_with_retry,
    sanitized_tracking_failure_fields,
    validate_safe_tracking_payload,
)


class SafeTrackingTests(unittest.TestCase):
    def test_official_hosts_replace_ambient_clearml_hosts(self) -> None:
        poisoned = {
            name: "https://invalid.example.test" for name in OFFICIAL_CLEARML_HOSTS
        }
        with patch.dict("os.environ", poisoned, clear=False):
            observed = configure_official_clearml_hosts()
            self.assertEqual(observed, OFFICIAL_CLEARML_HOSTS)
            for name, expected in OFFICIAL_CLEARML_HOSTS.items():
                self.assertEqual(os.environ[name], expected)

    def test_transient_initialization_is_retried_and_can_recover(self) -> None:
        calls = []
        delays = []

        def initialise():
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                raise ConnectionError("temporary connection error")
            return "task"

        task, outcome = initialize_clearml_task_with_retry(
            initialise,
            online=True,
            retry_delays_seconds=(0.1, 0.2),
            sleep=delays.append,
        )
        self.assertEqual(task, "task")
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(delays, [0.1, 0.2])
        self.assertEqual(
            outcome.to_receipt_fields(),
            {
                "tracking_attempt_count": 3,
                "tracking_retry_count": 2,
                "tracking_transient_failure_count": 2,
            },
        )

    def test_non_transient_initialization_is_not_retried(self) -> None:
        calls = []

        def initialise():
            calls.append(1)
            raise ValueError("invalid task configuration")

        with self.assertRaises(ClearMLTrackingFailure) as captured:
            initialize_clearml_task_with_retry(
                initialise,
                online=True,
                retry_delays_seconds=(0.1, 0.2),
                sleep=lambda _: None,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(captured.exception.category, "non_transient_error")

    def test_exhausted_failure_diagnostics_do_not_persist_exception_text(self) -> None:
        secret_text = (
            "Connection error for CLEARML_API_SECRET_KEY=top-secret "
            "at /Users/person/private"
        )

        def initialise():
            raise ConnectionError(secret_text)

        with self.assertRaises(ClearMLTrackingFailure) as captured:
            initialize_clearml_task_with_retry(
                initialise,
                online=True,
                retry_delays_seconds=(0.0,),
                sleep=lambda _: None,
            )
        fields = sanitized_tracking_failure_fields(captured.exception)
        serialized = str(fields) + str(captured.exception)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("error_message", fields)
        self.assertEqual(fields["tracking_attempt_count"], 2)
        self.assertEqual(
            fields["tracking_error_category"], "transient_connection_error"
        )

    def test_finalizer_deadline_is_never_retried(self) -> None:
        calls = []

        def initialise():
            calls.append(1)
            raise TimeoutError("ClearML finalisation exceeded its wall-time limit")

        with self.assertRaises(ClearMLTrackingFailure) as captured:
            initialize_clearml_task_with_retry(
                initialise,
                online=True,
                sleep=lambda _: None,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(captured.exception.category, "deadline_exceeded")

    def test_deadline_during_retry_backoff_stops_immediately(self) -> None:
        calls = []

        def initialise():
            calls.append(1)
            raise ConnectionError("temporary connection error")

        def deadline(_seconds: float) -> None:
            raise TimeoutError("ClearML finalisation exceeded its wall-time limit")

        with self.assertRaises(ClearMLTrackingFailure) as captured:
            initialize_clearml_task_with_retry(
                initialise,
                online=True,
                sleep=deadline,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(captured.exception.category, "deadline_exceeded")

    def test_enabled_tracker_reports_and_closes_with_upload_flush(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.scalars = []
                self.tables = []
                self.images = []

            def report_scalar(self, **kwargs) -> None:
                self.scalars.append(kwargs)

            def report_table(self, **kwargs) -> None:
                self.tables.append(kwargs)

            def report_image(self, **kwargs) -> None:
                self.images.append(kwargs)

        class FakeTask:
            id = "task-123"

            def __init__(self) -> None:
                self.logger = FakeLogger()
                self.artifacts = []
                self.tags = []
                self.failed = None
                self.flush_wait = None
                self.closed = False

            def get_logger(self):
                return self.logger

            def upload_artifact(self, **kwargs) -> None:
                self.artifacts.append(kwargs)

            def add_tags(self, tags) -> None:
                self.tags.extend(tags)

            def mark_failed(self, status_reason) -> None:
                self.failed = status_reason

            def flush(self, wait_for_uploads) -> None:
                self.flush_wait = wait_for_uploads

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            chart = Path(temporary_directory) / "chart.png"
            Image.new("RGB", (8, 8), "white").save(chart)
            task = FakeTask()
            tracker = ExperimentTracker(mode="online", task=task)
            tracker.report_scalar(title="Gate", series="passed", value=1)
            tracker.report_table(
                title="Gate",
                series="table",
                columns=["gate", "status"],
                rows=[{"gate": "count", "status": "pass"}],
            )
            tracker.report_image_file(
                title="Gate",
                series="chart",
                path=chart,
                evidence_kind="aggregate_chart",
            )
            tracker.upload_summary("summary", {"all_checks_passed": True})
            tracker.add_tags(["candidate"])
            tracker.close()

            self.assertEqual(tracker.task_id, "task-123")
            self.assertEqual(len(task.logger.scalars), 1)
            self.assertEqual(len(task.logger.tables), 1)
            self.assertEqual(len(task.logger.images), 1)
            self.assertEqual(len(task.artifacts), 1)
            self.assertTrue(task.flush_wait)
            self.assertTrue(task.closed)

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
        tracker.report_table(
            title="Integrity",
            series="gates",
            columns=["gate", "status"],
            rows=[{"gate": "image count", "status": "pass"}],
        )
        tracker.close()

        self.assertEqual(tracker.mode, "disabled")
        self.assertIsNone(tracker.task_id)

    def test_online_tracking_disables_repository_detection(self) -> None:
        init_arguments = {}

        class FakeTaskInstance:
            id = "task-456"

            def set_user_properties(self, **properties) -> None:
                self.properties = properties

            def connect(self, configuration, **kwargs) -> None:
                self.configuration = configuration

        class FakeTaskApi:
            TaskTypes = types.SimpleNamespace(inference="inference")

            @classmethod
            def init(cls, **kwargs):
                init_arguments.update(kwargs)
                return FakeTaskInstance()

        fake_clearml = types.SimpleNamespace(Task=FakeTaskApi)
        with patch.dict(sys.modules, {"clearml": fake_clearml}):
            tracker = ExperimentTracker.start(
                project_name="Visual Peer Inspector/02 Detection and Segmentation",
                task_name="proposal-pilot",
                task_type="inference",
                mode="online",
                properties={"stage": "pilot"},
                configuration={"image_count": 20},
            )

        self.assertEqual(tracker.task_id, "task-456")
        self.assertEqual(
            init_arguments["auto_connect_frameworks"],
            {"detect_repository": False},
        )

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

    def test_raw_image_upload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chart.png"
            Image.new("RGB", (8, 8), "white").save(path)
            tracker = ExperimentTracker(mode="disabled")
            with self.assertRaisesRegex(ValueError, "aggregate charts"):
                tracker.report_image_file(
                    title="Samples",
                    series="raw inputs",
                    path=path,
                    evidence_kind="raw_image",
                )

    def test_aggregate_chart_is_allowed_when_tracking_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chart.png"
            Image.new("RGB", (8, 8), "white").save(path)
            tracker = ExperimentTracker(mode="disabled")
            tracker.report_image_file(
                title="Integrity",
                series="view coverage",
                path=path,
                evidence_kind="aggregate_chart",
            )


if __name__ == "__main__":
    unittest.main()
