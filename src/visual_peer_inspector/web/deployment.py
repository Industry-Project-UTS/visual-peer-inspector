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
    # Bind exported decisions and any later policy to the executable method too.
    base = Path(__file__).resolve().parents[2]
    paths = [
        'visual_peer_inspector/web/' + name for name in
        ('pipeline.py', 'decision.py', 'core.py', 'deployment.py', 'model_runtime.py', 'encoder_worker.py')
    ]
    paths += [
        'visual_peer_inspector/workflows/learned_peer_features.py',
        'visual_peer_inspector/representation/cross_view_identity.py',
        'visual_peer_inspector/representation/crops.py',
        'visual_peer_inspector/models/visual_encoder.py',
        'visual_peer_inspector/scoring/forward_patch.py',
        'visual_peer_inspector/scoring/peer.py',
        'visual_peer_inspector/scoring/peer_support.py',
        'vpi_object_bank/scoring.py', 'vpi_object_bank/prediction.py',
    ]
    sources = {path: hashlib.sha256((base / path).read_bytes()).hexdigest() for path in paths}
    payload = json.dumps({'configuration': pipeline_config(), 'sources': sources}, sort_keys=True, separators=(",", ":"))
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
    """Load only the byte-identical policy frozen by calibration stage 09B."""
    from .frozen_policy import load_frozen_policy

    policy = load_frozen_policy(path)
    return policy, policy.version
