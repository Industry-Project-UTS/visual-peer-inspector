#!/usr/bin/env python3
"""Build the clean, development-only DINO CPU/GPU compatibility notebook."""
from pathlib import Path
import json

import nbformat

from current_notebook_cells import bootstrap_cell


ROOT = Path(__file__).resolve().parents[1]
NAME = "current-dino-gpu-compatibility.ipynb"


def build():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        md("""# Validate strict FP32 CUDA DINO against the sealed CPU cache

This development-only job regenerates a fixed, label-free sample of DINOv2-small features on CUDA and compares full patch tensors, global vectors, peer-score components and final automatic decisions with the verified 100-scene CPU cache. It opens no evaluator or anomaly labels and produces no accuracy result.

A passing receipt permits a later homogeneous GPU cache run. It never permits one assembled cache to mix CPU- and GPU-bound feature roots."""),
        bootstrap_cell(NAME, runtime_profile="requirements/profiles/linux-dino-fp32-cuda126.json"),
        md("""## Bound inputs

The launch configuration reuses the same sealed development regions, immutable DINO snapshot, current method freeze and 100-scene CPU feature cache. The sample is fixed in source: ten scene IDs selected by a label-free hash and views V00, V07 and V14. Numerical thresholds are fixed before execution."""),
        code("CONFIG = _runtime.resolve_config('dino-gpu-compatibility', ROOT)\nCONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\nfrom vpi_next_evaluation.gpu_compatibility import run_gpu_compatibility\nRESULT = run_gpu_compatibility(CONFIG, ROOT)\ndisplay(RESULT)"),
        md("""## Release rule

Keep the sealed compatibility receipt and detailed numerical summary. If the receipt says incompatible, retain the evidence and continue with the verified CPU extractor. Do not adjust thresholds after seeing this result. A production GPU queue must regenerate every feature shard under one GPU binding and must gate on this exact sealed receipt."""),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"name": "python3", "display_name": "Python3", "language": "python"}},
    )
    for index, cell in enumerate(cells):
        cell["id"] = __import__("hashlib").sha256((NAME + str(index) + cell.source).encode()).hexdigest()[:12]
    target = ROOT / "notebooks/input-preparation"
    target.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target / NAME)
    (target / "kernel-metadata-dino-gpu-compatibility.json").write_text(json.dumps({
        "id": "esreekarreddy/vpi-current-dino-gpu-compatibility",
        "title": "VPI current DINO GPU FP32 compatibility",
        "code_file": NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }, indent=2) + "\n")
    return target / NAME


if __name__ == "__main__":
    print(build())
