#!/usr/bin/env python3
"""Audit or byte-preserving import of 08N features; JSON config uses explicit paths/hashes."""
import argparse
import json
from pathlib import Path
from vpi_next_evaluation.feature_remap import audit_remap, import_remapped

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit','import'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((audit_remap if args.action == 'audit' else import_remapped)(json.loads(args.config.read_text())), indent=2))
