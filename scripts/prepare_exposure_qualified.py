#!/usr/bin/env python3
"""Prepare a truthfully qualified audit from neutral administrative membership."""
import argparse,json
from pathlib import Path
from vpi_next_evaluation.exposure import prepare_audit
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();print(json.dumps(prepare_audit(json.loads(Path(args.config).read_text())),indent=2))
