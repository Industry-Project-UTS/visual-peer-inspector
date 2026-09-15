#!/usr/bin/env python3
"""Run Notebook 08M's grouped development metric evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.workflows.learned_peer_evaluation import evaluate_metric_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--scoring-root", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_metric_candidates(
        args.feature_root,
        args.scoring_root,
        args.evaluator_root,
        args.model_input_root,
        args.output,
    )
    print(json.dumps({"event": "metric_evaluation_complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

