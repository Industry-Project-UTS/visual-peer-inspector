#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_peer_inspector.chunks import load_json, verify_chunks


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "toysad_chunks.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the ToysAD-8K archive chunks")
    parser.add_argument("chunk_directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sizes-only",
        action="store_true",
        help="Skip SHA-256 checks",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_json(args.manifest)
        report = verify_chunks(
            args.chunk_directory,
            manifest,
            check_hashes=not args.sizes_only,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    for item in report.files:
        status = "OK" if item.ok else "FAIL"
        detail = ", ".join(item.errors) if item.errors else "verified"
        print(f"[{status}] {item.name}: {detail}")
    for name in report.unexpected_files:
        print(f"[FAIL] unexpected file: {name}")
    for name in report.incomplete_files:
        print(f"[FAIL] incomplete file: {name}")
    for error in report.manifest_errors:
        print(f"[FAIL] manifest: {error}")

    print(f"Bytes: {report.actual_bytes}/{report.expected_bytes}")
    print("RESULT: PASS" if report.ok else "RESULT: FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
