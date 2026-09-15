#!/usr/bin/env python3
"""Local CPU execution over sealed object-bank artifacts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['predict', 'evaluate', 'report', 'verify'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-scenes', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.stage == 'predict':
        from vpi_object_mapping.prediction import predict
        result = predict(cfg, ROOT, max_scenes=args.max_scenes)
    elif args.stage == 'evaluate':
        from vpi_object_mapping.evaluation import evaluate
        result = evaluate(cfg['prediction_root'], cfg['previous_prediction_root'], cfg['prepared_root'],
                          cfg['scoring_root'], cfg['evaluator_root'], cfg['evaluation_root'], ROOT)
    elif args.stage == 'report':
        from vpi_object_mapping.reporting import build_report
        result = build_report(cfg, ROOT)
    else:
        from vpi_object_bank.storage import verify_seal
        for key, name in [('prediction_root', 'prediction_lock.json'), ('evaluation_root', 'evaluation_lock.json'),
                          ('report_root', 'report_lock.json')]:
            verify_seal(cfg[key], name=name)
        result = {'status': 'all scientific and report seals verified'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
