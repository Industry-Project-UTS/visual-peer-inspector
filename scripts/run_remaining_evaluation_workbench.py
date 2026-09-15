#!/usr/bin/env python3
"""Prepare or repair existing development evidence without protected evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','report','verify'])
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text())
    if args.stage=='prepare':
        from vpi_remaining_evaluation.workbench import prepare_workbench
        result=prepare_workbench(cfg,cfg['scientific_root'],ROOT)
    elif args.stage=='report':
        from vpi_remaining_evaluation.reporting import render_workbench
        result=render_workbench(cfg,cfg['scientific_root'],cfg['report_root'],ROOT)
    else:
        from vpi_object_bank.storage import verify_seal
        verify_seal(cfg['scientific_root'],name='scientific_lock.json')
        verify_seal(cfg['report_root'],name='report_lock.json')
        result={'status':'scientific and report seals verified','new_protected_evaluations':0}
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    main()
