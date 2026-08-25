#!/usr/bin/env python3
"""Run the DS1 LocateAnything workflow locally or inside a notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from visual_peer_inspector.models.locateanything import (  # noqa: E402
    LocateAnythingConfiguration,
    LocateAnythingWorker,
    ProposalContractWorker,
)
from visual_peer_inspector.workflows.locateanything_ds1 import (  # noqa: E402
    run_ds1_locateanything_workflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-mode",
        choices=("contract_smoke", "model_inference"),
        default="contract_smoke",
    )
    parser.add_argument(
        "--inference-stage",
        choices=("one_image", "pilot", "full"),
        default="one_image",
    )
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--provenance", type=Path)
    return parser.parse_args()


def _read_optional_json(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Provenance must be a JSON object")
    return value


def main() -> None:
    args = parse_args()
    configuration = LocateAnythingConfiguration()
    load_started = time.perf_counter()
    if args.run_mode == "model_inference":
        worker = LocateAnythingWorker(
            configuration,
            model_source=args.model_source,
        )
    else:
        worker = ProposalContractWorker(configuration)
    load_seconds = round(time.perf_counter() - load_started, 4)
    record = run_ds1_locateanything_workflow(
        model_input_root=args.model_input_root,
        output_root=args.output_root,
        worker=worker,
        configuration=configuration,
        run_mode=args.run_mode,
        inference_stage=args.inference_stage,
        provenance=_read_optional_json(args.provenance),
        model_load_evidence={"load_seconds": load_seconds},
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    if record["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
