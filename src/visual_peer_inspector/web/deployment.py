"""Deployment configuration. Importing this module never loads model libraries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")


def pipeline_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def pipeline_fingerprint() -> str:
    payload = json.dumps(pipeline_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class WorkerSettings:
    python: str
    timeout_seconds: float = 1800
    allow_downloads: bool = False
    review_policy_path: str | None = None

    def __post_init__(self):
        if not Path(self.python).is_file():
            raise ValueError(
                "VPI_WORKER_PYTHON must point to the inference environment's Python."
            )
        if not 1 <= self.timeout_seconds <= 7200:
            raise ValueError(
                "VPI_INSPECTION_TIMEOUT must be between 1 and 7200 seconds."
            )

    @classmethod
    def from_environment(cls):
        return cls(
            python=os.environ.get("VPI_WORKER_PYTHON", ""),
            timeout_seconds=float(os.environ.get("VPI_INSPECTION_TIMEOUT", "1800")),
            allow_downloads=os.environ.get("VPI_ALLOW_MODEL_DOWNLOADS") == "1",
            review_policy_path=os.environ.get("VPI_REVIEW_POLICY") or None,
        )


def load_review_policy(path: str | None):
    """Require an explicit policy bound to this upload pipeline, never guess thresholds."""
    if path is None:
        return None, pipeline_config()["default_review_policy"]
    from visual_peer_inspector.review_policy import ReviewPolicy

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"schema_version", "pipeline_sha256", "policy_configuration"}:
        raise ValueError(
            "Review policy envelope fields differ from vpi-web-review-policy/v1."
        )
    if value["schema_version"] != "vpi-web-review-policy/v1":
        raise ValueError("Unsupported web review policy schema.")
    if value["pipeline_sha256"] != pipeline_fingerprint():
        raise ValueError("Review policy belongs to a different upload pipeline.")
    policy = ReviewPolicy.from_mapping(value["policy_configuration"])
    return policy, f"review-policy-sha256:{policy.sha256}"
