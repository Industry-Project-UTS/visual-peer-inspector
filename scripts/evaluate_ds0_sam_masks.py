#!/usr/bin/env python3
"""Evaluate complete DS0 SAM routes after all model inference has finished."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_peer_inspector.workflows.mask_evaluation_ds0 import (
    CONTRACT_TEST_MODE,
    MODEL_EVIDENCE_MODE,
    run_ds0_mask_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--proposal-output-root", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator-provenance", type=Path, required=True)
    parser.add_argument(
        "--evaluation-mode",
        choices=(MODEL_EVIDENCE_MODE, CONTRACT_TEST_MODE),
        default=MODEL_EVIDENCE_MODE,
    )
    parser.add_argument("--mask-iou-threshold", type=float, default=0.75)
    parser.add_argument("--boundary-tolerance-pixels", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=32_040)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator_provenance = json.loads(
        args.evaluator_provenance.read_text(encoding="utf-8")
    )
    record = run_ds0_mask_evaluation(
        evaluator_root=args.evaluator_root,
        model_input_root=args.model_input_root,
        proposal_output_root=args.proposal_output_root,
        sam_output_root=args.sam_output_root,
        output_root=args.output_root,
        evaluator_provenance=evaluator_provenance,
        evaluation_mode=args.evaluation_mode,
        mask_iou_threshold=args.mask_iou_threshold,
        boundary_tolerance_pixels=args.boundary_tolerance_pixels,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "evaluation_mode": record["evaluation_mode"],
                "evidence_eligible": record["evidence_eligible"],
                "scientific_gates_passed": record["scientific_gates_passed"],
                "decision": record["candidate_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
