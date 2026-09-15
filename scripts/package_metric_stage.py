#!/usr/bin/env python3
"""Build deterministic evidence archives for one completed metric stage."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_archive(root: Path, output: Path, allowed_roots: tuple[str, ...]) -> dict[str, object]:
    members = sorted(
        path for path in root.rglob("*")
        if path.is_file() and any(path.relative_to(root).parts[0] == name for name in allowed_roots)
    )
    if not members:
        raise ValueError("No files matched the requested archive roots")
    temporary = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in members:
            info = zipfile.ZipInfo(str(path.relative_to(root)))
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Archive CRC failed for {bad}")
    temporary.replace(output)
    return {"file": output.name, "sha256": digest(output), "bytes": output.stat().st_size, "members": len(members)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    repository = write_archive(args.stage_root, args.output_root / f"{args.prefix}-repository-evidence.zip", ("shareable",))
    review = write_archive(args.stage_root, args.output_root / f"{args.prefix}-review-evidence.zip", ("private", "shareable", "fold-models"))
    receipt = {"schema_version": "metric-stage-package/v1", "repository": repository, "review": review}
    (args.output_root / f"{args.prefix}-package-receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    print(json.dumps({"event": "metric_stage_packaged", **receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

