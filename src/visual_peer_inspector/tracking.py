"""Small, fail-closed ClearML adapter for aggregate experiment evidence.

ClearML is optional at import time. Tracking defaults to disabled, and this
module deliberately avoids automatic argument, framework, and stream capture.
The caller must provide an allowlisted aggregate payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Mapping


TRACKING_MODES = frozenset({"disabled", "offline", "online"})
OFFICIAL_CLEARML_HOSTS = {
    "CLEARML_API_HOST": "https://api.clear.ml",
    "CLEARML_WEB_HOST": "https://app.clear.ml",
    "CLEARML_FILES_HOST": "https://files.clear.ml",
}
DEFAULT_CLEARML_RETRY_DELAYS_SECONDS = (2.0, 5.0)


@dataclass(frozen=True)
class ClearMLRetryOutcome:
    """Bounded Task.init result without retaining exception text."""

    attempt_count: int
    retry_count: int
    transient_failure_count: int

    def to_receipt_fields(self) -> dict[str, int]:
        return {
            "tracking_attempt_count": self.attempt_count,
            "tracking_retry_count": self.retry_count,
            "tracking_transient_failure_count": self.transient_failure_count,
        }


class ClearMLTrackingFailure(RuntimeError):
    """Sanitized failure raised after a bounded ClearML initialization attempt."""

    def __init__(
        self,
        *,
        category: str,
        error_type: str,
        attempt_count: int,
        retry_count: int,
        transient_failure_count: int,
    ) -> None:
        self.category = category
        self.error_type = _safe_error_type(error_type)
        self.attempt_count = attempt_count
        self.retry_count = retry_count
        self.transient_failure_count = transient_failure_count
        super().__init__(
            f"ClearML {category} after {attempt_count} bounded attempt(s)"
        )


def configure_official_clearml_hosts() -> dict[str, str]:
    """Force the hosted ClearML endpoints inside the isolated finalizer."""

    for name, value in OFFICIAL_CLEARML_HOSTS.items():
        os.environ[name] = value
    os.environ.setdefault("CLEARML_LOG_ENVIRONMENT", "")
    os.environ.setdefault("VPI_CLEARML_REQUEST_TIMEOUT_SECONDS", "20")
    return dict(OFFICIAL_CLEARML_HOSTS)


def _safe_error_type(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.]", "", value)[:80]
    return cleaned or "UnknownError"


def _tracking_error_category(error: Exception) -> tuple[str, bool]:
    """Classify an exception without returning or persisting its message."""

    name = type(error).__name__
    lowered_name = name.lower()
    message = str(error).lower()
    if isinstance(error, TimeoutError) and "finalisation exceeded" in message:
        return "deadline_exceeded", False
    authentication_markers = (
        "unauthorized",
        "forbidden",
        "invalid credential",
        "authentication",
        "status code 401",
        "status code 403",
    )
    if any(marker in message for marker in authentication_markers):
        return "authentication_error", False
    transient_names = (
        "connectionerror",
        "connecttimeout",
        "readtimeout",
        "proxyerror",
        "retryerror",
        "gaierror",
    )
    transient_messages = (
        "connection error",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "temporary failure",
        "timed out",
        "timeout",
        "name resolution",
        "failed to resolve",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "status code 502",
        "status code 503",
        "status code 504",
    )
    transient = (
        isinstance(error, ConnectionError)
        or any(marker in lowered_name for marker in transient_names)
        or any(marker in message for marker in transient_messages)
    )
    return ("transient_connection_error", True) if transient else (
        "non_transient_error",
        False,
    )


def initialize_clearml_task_with_retry(
    operation: Callable[[], Any],
    *,
    online: bool,
    retry_delays_seconds: Sequence[float] = DEFAULT_CLEARML_RETRY_DELAYS_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, ClearMLRetryOutcome]:
    """Retry only transient Task.init failures within a small fixed budget."""

    delays = tuple(float(value) for value in retry_delays_seconds) if online else ()
    if any(value < 0 or value > 30 for value in delays) or len(delays) > 4:
        raise ValueError("ClearML retry delays are outside the bounded policy")
    attempts = 0
    transient_failures = 0
    while True:
        attempts += 1
        try:
            task = operation()
            return task, ClearMLRetryOutcome(
                attempt_count=attempts,
                retry_count=attempts - 1,
                transient_failure_count=transient_failures,
            )
        except Exception as error:
            category, transient = _tracking_error_category(error)
            if transient:
                transient_failures += 1
            retry_index = attempts - 1
            if transient and retry_index < len(delays):
                try:
                    sleep(delays[retry_index])
                except TimeoutError:
                    raise ClearMLTrackingFailure(
                        category="deadline_exceeded",
                        error_type="TimeoutError",
                        attempt_count=attempts,
                        retry_count=max(0, attempts - 1),
                        transient_failure_count=transient_failures,
                    ) from None
                continue
            raise ClearMLTrackingFailure(
                category=category,
                error_type=type(error).__name__,
                attempt_count=attempts,
                retry_count=max(0, attempts - 1),
                transient_failure_count=transient_failures,
            ) from None


def sanitized_tracking_failure_fields(error: Exception) -> dict[str, Any]:
    """Return receipt-safe diagnostics without exception messages or secrets."""

    if isinstance(error, ClearMLTrackingFailure):
        return {
            "tracking_error_category": error.category,
            "tracking_error_type": error.error_type,
            "tracking_attempt_count": error.attempt_count,
            "tracking_retry_count": error.retry_count,
            "tracking_transient_failure_count": error.transient_failure_count,
        }
    if isinstance(error, TimeoutError):
        category = "deadline_exceeded"
    elif isinstance(error, ValueError):
        category = "validation_error"
    elif isinstance(error, RuntimeError):
        category = "configuration_error"
    else:
        category = "internal_error"
    return {
        "tracking_error_category": category,
        "tracking_error_type": _safe_error_type(type(error).__name__),
        "tracking_attempt_count": 0,
        "tracking_retry_count": 0,
        "tracking_transient_failure_count": 0,
    }


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
