#!/usr/bin/env python3
"""Seal and seed verified LocateAnything checkpoints without rerunning inference.

This transport helper deliberately lives outside the frozen scientific source
tree.  It never changes a run binding: the destination must use the exact
binding under which the recovered proposal units were produced.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from uuid import uuid4

from vpi_object_bank.storage import (
    atomic_json,
    atomic_unit,
    bind,
    digest,
    digest_file,
    read_json,
    seal,
    verified_unit,
    verify_seal,
)
from vpi_remaining_evaluation.workbench import resolve_input_root


SCHEMA = "vpi-current-proposal-recovery/v1"


def _validate_proposal(value):
    if not isinstance(value, dict) or set(value) != {"boxes", "attempts", "status", "terminal_failure"}:
        raise ValueError("Proposal recovery record schema differs")
    boxes = value["boxes"]
    if not isinstance(boxes, list):
        raise ValueError("Proposal recovery boxes are not a list")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in box
        ):
            raise ValueError("Proposal recovery box geometry differs")
        if not (0 <= box[0] < box[2] <= 256 and 0 <= box[1] < box[3] <= 256):
            raise ValueError("Proposal recovery box is outside the model input")
    attempts = value["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise ValueError("Proposal recovery attempt history differs")
    for position, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != {
            "attempt", "status", "error_type", "runtime_seconds"
        }:
            raise ValueError("Proposal recovery attempt schema differs")
        if attempt["attempt"] != position or attempt["status"] not in {"ok", "error"}:
            raise ValueError("Proposal recovery attempt ordering or status differs")
        runtime = attempt["runtime_seconds"]
        if runtime is not None and (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not math.isfinite(runtime)
            or runtime < 0
        ):
            raise ValueError("Proposal recovery runtime differs")
    if boxes:
        if value["status"] != "ok" or value["terminal_failure"] is not None or attempts[-1]["status"] != "ok":
            raise ValueError("Successful proposal recovery status differs")
    elif value["status"] != "zero_proposals" or not isinstance(value["terminal_failure"], str):
        raise ValueError("Empty proposal recovery status differs")
    return value


def _proposal_units(root, binding_sha256):
    """Validate every completed unit before returning any transport payload."""
    root = Path(root)
    units = []
    for lock_path in sorted((root / "checkpoints/proposals").glob("*/*/artifact_lock.json")):
        unit = lock_path.parent
        parts = unit.relative_to(root / "checkpoints/proposals").parts
        if len(parts) != 2 or not verified_unit(unit, binding_sha256):
            raise ValueError("Proposal recovery checkpoint inventory differs")
        relative = unit.relative_to(root).as_posix()
        units.append(((parts[0], parts[1]), relative, _validate_proposal(read_json(unit / "proposal.json"))))
    if not units:
        raise ValueError("Proposal recovery contains no completed checkpoints")
    if len({key for key, _, _ in units}) != len(units):
        raise ValueError("Proposal recovery contains duplicate checkpoints")
    return units


def export_recovery(source_output_root, recovery_root, expected_proposal_views):
    """Create one immutable package from completed units in a failed run."""
    source = Path(source_output_root)
    destination = Path(recovery_root)
    if destination.exists():
        raise FileExistsError("Choose a fresh proposal recovery destination")
    binding_path = source / "run_binding.json"
    if not binding_path.is_file():
        raise ValueError("Proposal recovery source has no run binding")
    binding = read_json(binding_path)
    signature = digest(binding)
    units = _proposal_units(source, signature)
    if (
        isinstance(expected_proposal_views, bool)
        or not isinstance(expected_proposal_views, int)
        or expected_proposal_views < 1
        or len(units) != expected_proposal_views
    ):
        raise ValueError("Proposal recovery is incomplete for the declared expected view count")
    native_source = binding.get("source_sha256", {}).get("native_regions.py")
    if not isinstance(native_source, str) or len(native_source) != 64:
        raise ValueError("Proposal recovery source code hash is absent")

    temporary = destination.with_name(destination.name + ".partial-" + uuid4().hex[:8])
    temporary.mkdir(parents=True)
    try:
        shutil.copyfile(binding_path, temporary / "run_binding.json")
        copied = [temporary / "run_binding.json"]
        for _, relative, _ in units:
            target = temporary / relative
            target.mkdir(parents=True)
            for name in ("proposal.json", "artifact_lock.json"):
                shutil.copyfile(source / relative / name, target / name)
                copied.append(target / name)
        manifest = {
            "schema": SCHEMA,
            "source_binding_sha256": signature,
            "source_native_regions_sha256": native_source,
            "expected_proposal_views": expected_proposal_views,
            "completed_proposal_views": len(units),
            "units": [relative for _, relative, _ in units],
        }
        atomic_json(temporary / "proposal_recovery_manifest.json", manifest)
        copied.append(temporary / "proposal_recovery_manifest.json")
        seal(
            temporary,
            copied,
            name="proposal_recovery_lock.json",
            metadata={"schema": SCHEMA, "source_binding_sha256": signature},
        )
        verify_seal(temporary, name="proposal_recovery_lock.json")
        temporary.rename(destination)
    except BaseException:
        # Keep the staging directory for diagnosis. It is never accepted as an input.
        raise
    return {
        "schema": SCHEMA,
        "status": "complete",
        "expected_proposal_views": expected_proposal_views,
        "completed_proposal_views": len(units),
        "source_binding_sha256": signature,
        "source_native_regions_sha256": native_source,
        "proposal_recovery_lock_sha256": digest_file(destination / "proposal_recovery_lock.json"),
    }


def _read_recovery(recovery_root, expected_lock_sha256, expected_binding):
    root = resolve_input_root(Path(recovery_root), "proposal_recovery_manifest.json")
    if (
        not isinstance(expected_lock_sha256, str)
        or len(expected_lock_sha256) != 64
        or digest_file(root / "proposal_recovery_lock.json") != expected_lock_sha256
    ):
        raise ValueError("Proposal recovery lock is absent or differs from the approved package")
    lock = verify_seal(root, name="proposal_recovery_lock.json")
    manifest = read_json(root / "proposal_recovery_manifest.json")
    binding = read_json(root / "run_binding.json")
    signature = digest(binding)
    if binding != expected_binding:
        raise ValueError("Proposal recovery input, model, configuration, code or partition differs")
    native_source = binding.get("source_sha256", {}).get("native_regions.py")
    expected_manifest = {
        "schema": SCHEMA,
        "source_binding_sha256": signature,
        "source_native_regions_sha256": native_source,
        "expected_proposal_views": manifest.get("expected_proposal_views"),
        "completed_proposal_views": manifest.get("completed_proposal_views"),
        "units": manifest.get("units"),
    }
    if manifest != expected_manifest:
        raise ValueError("Proposal recovery manifest differs")
    if lock.get("metadata") != {"schema": SCHEMA, "source_binding_sha256": signature}:
        raise ValueError("Proposal recovery seal metadata differs")
    units = _proposal_units(root, signature)
    relative_units = [relative for _, relative, _ in units]
    if (
        manifest.get("expected_proposal_views") != len(units)
        or manifest.get("completed_proposal_views") != len(units)
        or manifest.get("units") != relative_units
    ):
        raise ValueError("Proposal recovery manifest checkpoint inventory differs")
    required = {"run_binding.json", "proposal_recovery_manifest.json"} | {
        f"{relative}/{name}"
        for _, relative, _ in units
        for name in ("proposal.json", "artifact_lock.json")
    }
    if set(lock["files"]) != required:
        raise ValueError("Proposal recovery seal inventory differs")
    return root, signature, manifest, units


def seed_recovery(recovery_root, output_root, expected_lock_sha256, expected_binding):
    """Atomically seed only after the package and every existing unit validate."""
    _, signature, manifest, units = _read_recovery(
        recovery_root, expected_lock_sha256, expected_binding
    )
    output = Path(output_root)

    # Check all existing target units before binding or copying the first payload.
    existing = {}
    existing_locks = sorted((output / "checkpoints/proposals").glob("*/*/artifact_lock.json"))
    if existing_locks:
        for key, _, proposal in _proposal_units(output, signature):
            existing[key] = proposal
        selected_scenes = set(expected_binding.get("partition", {}).get("scenes", []))
        if not selected_scenes or any(key[0] not in selected_scenes for key in existing):
            raise ValueError("Existing target proposal is outside the bound partition")
    pending = []
    already = 0
    for key, _, proposal in units:
        target = output / "checkpoints/proposals" / key[0] / key[1]
        if key in existing:
            if existing[key] != proposal:
                raise ValueError("Existing target proposal payload differs from recovery")
            already += 1
        else:
            pending.append((key, proposal))

    bind(output, expected_binding)
    for key, proposal in pending:
        target = output / "checkpoints/proposals" / key[0] / key[1]
        with atomic_unit(target, signature) as temporary:
            atomic_json(temporary / "proposal.json", proposal)

    receipt = {
        "schema": SCHEMA,
        "status": "seeded",
        "proposal_recovery_lock_sha256": expected_lock_sha256,
        "source_binding_sha256": signature,
        "source_native_regions_sha256": manifest["source_native_regions_sha256"],
        "eligible_views": len(units),
        "seeded_views_this_call": len(pending),
        "already_seeded_views": already,
        "locateanything_calls": 0,
    }
    atomic_json(output / "proposal_recovery_receipt.json", receipt)
    return receipt


def expected_binding(config, source_root):
    """Recompute the unchanged scientific binding with the frozen implementation."""
    from vpi_next_evaluation.native_regions import _preflight

    _, _, _, binding = _preflight(config, Path(source_root))
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--source", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--expected-proposal-views", required=True, type=int)
    seed = subparsers.add_parser("seed")
    seed.add_argument("--recovery", required=True)
    seed.add_argument("--output", required=True)
    seed.add_argument("--expected-lock-sha256", required=True)
    seed.add_argument("--config", required=True)
    seed.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_recovery(
            args.source, args.destination, args.expected_proposal_views
        )
    else:
        config = json.loads(Path(args.config).read_text())
        binding = expected_binding(config, args.source_root)
        result = seed_recovery(
            args.recovery, args.output, args.expected_lock_sha256, binding
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
