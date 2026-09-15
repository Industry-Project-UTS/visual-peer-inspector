#!/usr/bin/env python3
"""Run each notebook stage independently so failures never require model reruns."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.local/matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('preflight', 'extract', 'predict', 'evaluate', 'report', 'verify'))
    parser.add_argument('--config', type=Path, default=ROOT / '.local/config.json')
    parser.add_argument('--max-units', type=int)
    parser.add_argument('--max-scenes', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    from vpi_object_bank.protocol import PROTOCOL
    from vpi_object_bank.storage import atomic_json, digest, read_json, verify_seal
    if args.stage == 'preflight':
        from vpi_object_bank.inputs import prepare_inputs, _source_correspondence
        from vpi_object_bank import inputs as input_module
        from vpi_object_bank.storage import digest_file
        prepared = Path(cfg['prepared_root'])
        if prepared.exists():
            prepared_lock = verify_seal(prepared)
            if 'input_manifest.json' not in prepared_lock['files']:
                raise ValueError('Prepared seal omits its manifest')
            result = read_json(prepared / 'input_manifest.json')
            if result['view_sets'] != PROTOCOL['view_sets'] or result['implementation_sha256'] != digest_file(Path(input_module.__file__)):
                raise ValueError('Prepared inputs belong to changed source or view configuration')
            for key, name, field in [('model_root','artifact_lock.json','model_input_lock_sha256'),
                                     ('scoring_root','prediction_lock.json','scoring_prediction_lock_sha256'),
                                     ('scoring_root','feature_lock.json','scoring_feature_lock_sha256')]:
                verify_seal(Path(cfg[key]), name=name)
                if digest_file(Path(cfg[key])/name) != result[field]:
                    raise ValueError('Prepared input binding does not match configured source')
            for route, key in [('supplied','neutral_supplied'),('automatic','neutral_automatic')]:
                _, receipt = _source_correspondence(Path(cfg[key]), route)
                if receipt != result['neutral_sources'][route]:
                    raise ValueError('Neutral source changed since preparation')
        else:
            result = prepare_inputs(Path(cfg['model_root']), Path(cfg['scoring_root']),
                                    Path(cfg['neutral_supplied']), Path(cfg['neutral_automatic']),
                                    prepared, view_sets=PROTOCOL['view_sets'])
        frozen = Path(cfg.get('protocol_path', str(Path(cfg['prepared_root']).parent / 'frozen_protocol.json')))
        if frozen.exists() and read_json(frozen) != PROTOCOL:
            raise ValueError('Scientific protocol has changed; preserve current run and choose a new stage')
        atomic_json(frozen, PROTOCOL)
        print(json.dumps({'stage': args.stage, 'scenes': len(result['scenes']),
                          'protocol_sha256': digest(PROTOCOL), 'status': 'verified'}), flush=True)
        return
    if args.stage == 'extract':
        from vpi_object_bank.extraction import extract_features
        result = extract_features(Path(cfg['model_root']), Path(cfg['prepared_root']), Path(cfg['feature_root']), ROOT,
                                  device=cfg.get('device', 'cpu'), model_source=Path(cfg['model_source']),
                                  max_units=args.max_units)
    elif args.stage == 'predict':
        from vpi_object_bank.prediction import predict
        result = predict(Path(cfg['prepared_root']), Path(cfg['feature_root']), Path(cfg['scoring_root']),
                         Path(cfg['prediction_root']), ROOT, max_scenes=args.max_scenes)
    elif args.stage == 'evaluate':
        from vpi_object_bank.evaluation import evaluate
        result = evaluate(Path(cfg['prediction_root']), Path(cfg['prepared_root']), Path(cfg['scoring_root']),
                          Path(cfg['evaluator_root']), Path(cfg['evaluation_root']), ROOT)
    elif args.stage == 'report':
        from vpi_object_bank.reporting import build_report
        result = build_report(cfg, ROOT)
    else:
        result = {}
        for key, lock_name in [('prepared_root', 'artifact_lock.json'), ('feature_root', 'feature_lock.json'),
                               ('prediction_root', 'prediction_lock.json'), ('evaluation_root', 'evaluation_lock.json'),
                               ('report_root', 'report_lock.json')]:
            root = Path(cfg[key])
            if (root / lock_name).exists():
                lock = verify_seal(root, name=lock_name)
                result[key] = {'verified_files': len(lock['files'])}
        if 'prediction_root' not in result:
            raise ValueError('Notebook 08N has not produced a sealed scientific prediction')
    print(json.dumps({'stage': args.stage, 'result': result}, sort_keys=True, default=str), flush=True)


if __name__ == '__main__':
    main()
