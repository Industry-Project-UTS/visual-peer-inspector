#!/usr/bin/env python3
"""Join current native cache shards without inference or evaluator access."""
import argparse
import json
from pathlib import Path
from vpi_next_evaluation.cache_shards import merge_cache_shards

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(merge_cache_shards(json.loads(args.config.read_text()),args.source_root),indent=2))
