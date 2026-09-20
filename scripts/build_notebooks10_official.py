#!/usr/bin/env python3
"""Build all current10 notebooks; legacy runners remain historical source only."""
import json
from pathlib import Path
import nbformat
from notebook_runtime_cells import harden_notebook
from notebooks10_current import (build_release_cells, build_prediction_cells,
    build_finalize_cells, build_evaluation_cells, clean_notebook)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / 'notebooks/official-evaluation'


def build():
    entries = [('10r', '10r-toysad8k-official-release.ipynb', build_release_cells()),
               ('10a', '10a-toysad8k-official-seen-frozen-inference.ipynb', build_prediction_cells('seen')),
               ('10a1', '10a1-toysad8k-official-unseen-frozen-inference.ipynb', build_prediction_cells('unseen')),
               ('10a-finalize', '10a-finalize-toysad8k-official-prediction-summary.ipynb', build_finalize_cells()),
               ('10b', '10b-toysad8k-official-seen-unseen-evaluation.ipynb', build_evaluation_cells())]
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for stage, filename, cells in entries:
        nbformat.write(harden_notebook(clean_notebook(cells, filename)), NOTEBOOK_ROOT / filename)
        path = NOTEBOOK_ROOT / f'kernel-metadata-{stage}.json'
        metadata = json.loads(path.read_text())
        metadata['enable_gpu'] = False
        metadata['enable_internet'] = False
        metadata['dataset_sources'] = []
        metadata['kernel_sources'] = []
        metadata.pop('machine_shape', None)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    build()
