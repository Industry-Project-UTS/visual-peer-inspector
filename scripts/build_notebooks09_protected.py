#!/usr/bin/env python3
"""Build current09A/09C CPU cached prediction notebooks."""
from notebooks09_current import ROOT, STAGES, build_cells as current_cells, write_stage

NOTEBOOK_ROOT = ROOT / "notebooks/protected-evaluation"
CONFIGURATIONS = tuple({"code": stage, "filename": STAGES[stage][0], "title": STAGES[stage][1]} for stage in ("09a", "09c"))


def build_cells(configuration):
    return current_cells(configuration["code"])


def build():
    return [write_stage(stage) for stage in ("09a", "09c")]


if __name__ == "__main__":
    for path in build():
        print(path)
