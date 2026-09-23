"""Offline deployment checks. Does not import or download models."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visual_peer_inspector.web.deployment import pipeline_config, pipeline_fingerprint
from visual_peer_inspector.web.frozen_policy import load_frozen_policy
from visual_peer_inspector.web.ui import build_app


def main():
    config = pipeline_config()
    for key in ("locateanything", "sam2", "encoder"):
        assert len(config[key]["revision"]) == 40
    app = build_app()
    policy = load_frozen_policy()
    assert config["default_review_policy"] == policy.version
    assert app is not None
    assert (ROOT / "apps/web/favicon.svg").is_file()
    print(
        json.dumps(
            {
                "status": "code_checks_passed",
                "pipeline_sha256": pipeline_fingerprint(),
                "review_policy": policy.version,
                "review_policy_sha256": policy.source_sha256,
                "models_loaded": False,
                "gpu_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
