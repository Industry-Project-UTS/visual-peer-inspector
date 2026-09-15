#!/usr/bin/env python3
"""Run Notebook 08L's label-free feature preparation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.workflows.learned_peer_features import prepare_feature_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_feature_package(args.scoring_root, args.output)
    print(json.dumps({"event": "feature_preparation_complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

