#!/usr/bin/env python3
"""Build current09B fixed-ranking evaluation; no readiness policy is fitted."""
import nbformat
from notebooks09_current import ROOT, STAGES, build_cells, write_stage

NOTEBOOK_ROOT = ROOT / "notebooks/protected-evaluation"
NOTEBOOK_PATH = NOTEBOOK_ROOT / STAGES["09b"][0]
METADATA_PATH = NOTEBOOK_ROOT / "kernel-metadata-09b.json"


def build_notebook():
    return nbformat.v4.new_notebook(cells=build_cells("09b"), metadata={"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}})


def main():
    write_stage("09b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
