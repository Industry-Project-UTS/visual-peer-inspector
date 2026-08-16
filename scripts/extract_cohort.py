#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector.chunks import load_json
from visual_peer_inspector.extraction import extract_cohort


DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "manifests" / "toysad_chunks.json"
DEFAULT_COHORT = PROJECT_ROOT / "data" / "manifests" / "development_40.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract the fixed development cohort")
    parser.add_argument("chunk_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-manifest", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = extract_cohort(
            args.chunk_directory,
            load_json(args.chunk_manifest),
            load_json(args.cohort),
            args.output,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Scenes: {report['scene_count']}")
    print(f"Files: {report['file_count']}")
    print(f"Bytes: {report['total_bytes']}")
    print(f"Output: {args.output.expanduser().resolve()}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
