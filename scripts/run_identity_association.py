"""Run Notebook 08J prediction, evaluation and verification stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "predict", "evaluate", "verify"))
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model-input", type=Path)
    parser.add_argument("--evaluator", type=Path)
    args = parser.parse_args()

    from visual_peer_inspector.workflows.comparison_io import verify_seal
    from visual_peer_inspector.workflows.identity_association_study import evaluate, predict, preflight

    if args.stage == "preflight":
        print(preflight(args.scoring))
    elif args.stage == "predict":
        print(predict(args.scoring, args.output, SOURCE))
    elif args.stage == "evaluate":
        if args.model_input is None or args.evaluator is None or args.predictions is None:
            parser.error("Evaluation requires --model-input, --evaluator and --predictions")
        print(evaluate(args.model_input, args.scoring, args.evaluator, args.predictions, args.output).to_string(index=False))
    else:
        lock_name = "prediction_lock.json" if (args.output / "prediction_lock.json").is_file() else "artifact_lock.json"
        print(verify_seal(args.output, name=lock_name))


if __name__ == "__main__":
    main()
