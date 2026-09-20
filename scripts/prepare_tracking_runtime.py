#!/usr/bin/env python3
"""Install verified publication dependencies before the bounded ClearML worker."""
import argparse
import json
from current_notebook_runtime import configure_tracking_runtime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    args = parser.parse_args()
    print(json.dumps(configure_tracking_runtime(args.source_root, allow_install=True), indent=2))
