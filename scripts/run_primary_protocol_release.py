#!/usr/bin/env python3
"""Verify the primary split and release the remaining development RGB package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.workflows.primary_protocol_release import (
    run_primary_protocol_release,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--administrative-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-archive-byte-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    record = run_primary_protocol_release(
        selection_root=arguments.selection_root,
        development_root=arguments.development_root,
        administrative_root=arguments.administrative_root,
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        verify_archive_bytes=not arguments.skip_archive_byte_check,
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
