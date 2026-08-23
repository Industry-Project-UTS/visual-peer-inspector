"""Small, fail-closed ClearML adapter for aggregate experiment evidence.

ClearML is optional at import time. Tracking defaults to disabled, and this
module deliberately avoids automatic argument, framework, and stream capture.
The caller must provide an allowlisted aggregate payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping


TRACKING_MODES = frozenset({"disabled", "offline", "online"})


def _configure_clearml_request_timeouts() -> None:
    """Keep one slow tracking request from blocking an experiment run."""

    try:
        from clearml.backend_api.session.session import Session
    except (ImportError, AttributeError):
        return

    timeout_seconds = float(
        os.getenv("VPI_CLEARML_REQUEST_TIMEOUT_SECONDS", "30")
    )
    timeout_seconds = max(5.0, min(timeout_seconds, 120.0))
    connect_seconds = min(5.0, timeout_seconds)
    bounded_timeout = (connect_seconds, timeout_seconds)
    Session._session_initial_timeout = bounded_timeout
    Session._session_timeout = bounded_timeout
    Session._write_session_timeout = bounded_timeout

_FORBIDDEN_KEY_PARTS = (
    "source_path",
    "source_scene",
    "source_object",
    "source_filename",
    "raw_path",
    "local_path",
    "object_order",
    "source_index",
    "anomaly_count",
    "anomaly_label",
    "labels_by_object",
    "base_asset",
    "crosswalk",
    "permutation",
    "evaluator_label",
    "internal_lock_label",
    "calibration_label",
)

_FORBIDDEN_VALUE_PARTS = (
    "/users/",
    "/kaggle/input/",
    "\\users\\",
    "crosswalk_private",
    "labels_internal_lock",
    "labels_calibration",
)


def validate_safe_tracking_payload(payload: Any, location: str = "payload") -> None:
    """Reject private identifiers, paths, labels, and unsupported values.

    This check is intentionally conservative. Only aggregate metadata and
    metrics should reach the tracking service.
    """

    if payload is None or isinstance(payload, (bool, int, float)):
        return

    if isinstance(payload, str):
        lowered = payload.lower()
        if any(part in lowered for part in _FORBIDDEN_VALUE_PARTS):
            raise ValueError(f"Unsafe tracking value at {location}")
        return

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"Tracking key at {location} must be a string")
            lowered_key = key.lower()
            if any(part in lowered_key for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"Unsafe tracking key at {location}.{key}")
            validate_safe_tracking_payload(value, f"{location}.{key}")
        return

    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_safe_tracking_payload(value, f"{location}[{index}]")
        return

    raise TypeError(
        f"Unsupported tracking value at {location}: {type(payload).__name__}"
    )


@dataclass
class ExperimentTracker:
    """ClearML wrapper with a no-op disabled mode."""

    mode: str
    task: Any | None = None

    @classmethod
    def start(
        cls,
        *,
        project_name: str,
        task_name: str,
        task_type: str = "data_processing",
        properties: Mapping[str, Any] | None = None,
        configuration: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> "ExperimentTracker":
        resolved_mode = (mode or os.getenv("VPI_TRACKING_MODE", "disabled")).lower()
        if resolved_mode not in TRACKING_MODES:
            choices = ", ".join(sorted(TRACKING_MODES))
            raise ValueError(f"VPI_TRACKING_MODE must be one of: {choices}")

        safe_properties = dict(properties or {})
        safe_configuration = dict(configuration or {})
        validate_safe_tracking_payload(safe_properties, "properties")
        validate_safe_tracking_payload(safe_configuration, "configuration")

        if resolved_mode == "disabled":
            return cls(mode=resolved_mode)

        try:
            from clearml import Task
        except ImportError as error:
            raise RuntimeError(
                "ClearML tracking was requested but clearml is not installed. "
                "Install code/requirements/tracking-lock.txt or set "
                "VPI_TRACKING_MODE=disabled."
            ) from error

        if resolved_mode == "offline":
            Task.set_offline(True)

        _configure_clearml_request_timeouts()

        clearml_task_type = getattr(Task.TaskTypes, task_type, task_type)
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=clearml_task_type,
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks={"detect_repository": False},
            auto_connect_streams=False,
            auto_resource_monitoring=False,
        )

        if safe_properties:
            task.set_user_properties(**safe_properties)
        if safe_configuration:
            task.connect(
                safe_configuration,
                name="Resolved Configuration",
                ignore_remote_overrides=True,
            )

        return cls(mode=resolved_mode, task=task)

    @property
    def task_id(self) -> str | None:
        if self.task is None:
            return None
        return str(self.task.id)

    def report_scalar(
        self,
        *,
        title: str,
        series: str,
        value: int | float,
        iteration: int = 0,
    ) -> None:
        validate_safe_tracking_payload(
            {"title": title, "series": series, "value": value}, "scalar"
        )
        if self.task is not None:
            self.task.get_logger().report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=iteration,
            )

    def upload_summary(self, name: str, summary: Mapping[str, Any]) -> None:
        """Upload one small sanitized dictionary, never a file or data array."""

        safe_summary = dict(summary)
        validate_safe_tracking_payload(
            {"artifact_name": name, "summary": safe_summary}, "artifact"
        )
        if self.task is not None:
            self.task.upload_artifact(name=name, artifact_object=safe_summary)

    def report_table(
        self,
        *,
        title: str,
        series: str,
        columns: list[str],
        rows: list[Mapping[str, Any]],
        iteration: int = 0,
    ) -> None:
        """Report one sanitized table without requiring pandas."""

        safe_rows = [dict(row) for row in rows]
        validate_safe_tracking_payload(
            {
                "title": title,
                "series": series,
                "columns": columns,
                "rows": safe_rows,
            },
            "table",
        )
        missing = [
            column
            for row in safe_rows
            for column in columns
            if column not in row
        ]
        if missing:
            raise ValueError(f"Tracking table is missing columns: {sorted(set(missing))}")
        if self.task is not None:
            table = [list(columns)] + [
                [row[column] for column in columns] for row in safe_rows
            ]
            self.task.get_logger().report_table(
                title=title,
                series=series,
                iteration=iteration,
                table_plot=table,
            )

    def report_image_file(
        self,
        *,
        title: str,
        series: str,
        path: Path,
        evidence_kind: str,
        iteration: int = 0,
    ) -> None:
        """Report an aggregate chart while rejecting raw-image uploads."""

        validate_safe_tracking_payload(
            {"title": title, "series": series, "evidence_kind": evidence_kind},
            "image",
        )
        if evidence_kind != "aggregate_chart":
            raise ValueError("ClearML image uploads are limited to aggregate charts")
        path = path.expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Tracking image must be an existing PNG or JPEG")
        if self.task is not None:
            self.task.get_logger().report_image(
                title=title,
                series=series,
                iteration=iteration,
                local_path=str(path),
                max_image_history=-1,
                delete_after_upload=False,
            )

    def add_tags(self, tags: list[str]) -> None:
        validate_safe_tracking_payload(tags, "tags")
        if self.task is not None:
            self.task.add_tags(tags)

    def mark_failed(self, reason: str) -> None:
        validate_safe_tracking_payload(reason, "failure_reason")
        if self.task is not None:
            self.task.mark_failed(status_reason=reason[:512])

    def close(self) -> None:
        if self.task is not None:
            self.task.flush(wait_for_uploads=True)
            self.task.close()


__all__ = [
    "ExperimentTracker",
    "TRACKING_MODES",
    "validate_safe_tracking_payload",
]
