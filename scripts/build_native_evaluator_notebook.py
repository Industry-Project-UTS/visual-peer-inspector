#!/usr/bin/env python3
"""Generate explicit neutral-reference and post-prediction truth conversion."""
from pathlib import Path
import json
import nbformat
from current_notebook_cells import bootstrap_cell

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'notebooks/input-preparation/current-native-reference-conversion.ipynb'


def build():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# Convert native references for the fixed multi-view pipeline

Run this CPU preparation separately from model extraction. Existing native evaluator resources contain registered crosswalk, geometry and label files. `geometry` reads only crosswalk and geometry, strips raw metadata, and produces neutral object IDs, boxes and a scene-to-asset index. `neutral` matches predicted boxes to reference boxes with the existing one-to-one IoU≥0.5 policy. Unmatched candidates retain null identities. No anomaly answer chooses a match.

`truth` is a later operation: both automatic and assisted predictions must be completely sealed before the converter opens any outcome file. Its output is the typed evaluator consumed by09B/09D/10B. A dataset-reference identity map must be reported as assisted. The scorer still chooses the unusual object.'''), bootstrap_cell(RELATIVE),
        md('''## Use the generated stage configuration

Set `VPI_NEXT_CONFIG` to the launch kit JSON. Geometry requires the native evaluator root and its exact manifest checksum. Neutral conversion additionally requires the verified native-region configuration and sealed geometry root. Truth conversion requires current cache, freeze and complete prediction roots. Every action writes to a separate writable output directory. Never mount the truth resource in a prediction notebook.'''),
        md('''## Reuse the existing full Kaggle dataset directly

When a typed native evaluator resource does not yet exist, `raw_geometry` uses the sealed private RGB source index and the existing indexed ToysAD archive. It extracts only reference boxes and identities from supplied masks (original roles) or supplied2D boxes (official roles). The trusted preparer retains source metadata privately. It exports no anomaly answers and does no RGB re-upload. `raw_truth` runs later, after complete predictions, and applies the exact original opaque-object permutation to the dataset's normalized anomaly flags. Geometry and truth bind the same source index and administrative checksums.'''),
        code("CONFIG = _runtime.resolve_config('native-reference', ROOT)\nCONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\nfrom vpi_next_evaluation.native_evaluator import prepare_geometry, prepare_neutral, convert_evaluator\nfrom vpi_next_evaluation.raw_reference import prepare_raw_geometry, convert_raw_evaluator\nACTION = CONFIG['action']\nOPERATIONS = {'geometry': prepare_geometry, 'neutral': prepare_neutral, 'truth': convert_evaluator, 'raw_geometry': prepare_raw_geometry, 'raw_truth': convert_raw_evaluator}\nif ACTION not in OPERATIONS:\n    raise ValueError('Choose a declared reference-conversion action')\nfrom current_stage_execution import run_stage\nRESULT = run_stage('native-reference', CONFIG, lambda: OPERATIONS[ACTION](CONFIG), CODE_PROVENANCE)\ndisplay(RESULT)"),
        md('''## Retain and verify

The sealed scene-to-asset index is the input to the exposure audit. Keep native labels private; the neutral map seals only the whitelisted identity observations. A final evaluator seal covers all original scenes, including missing or unrankable candidates. Per-scene checkpoints can resume only with identical source, geometry, cache and prediction bindings. Download the resulting resource and verify its seal before continuing the queue.''')]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python3'}})
    path=ROOT/RELATIVE; path.parent.mkdir(parents=True,exist_ok=True); nbformat.write(nb,path)
    (path.parent/'kernel-metadata-native-reference.json').write_text(json.dumps({
        'id':'esreekarreddy/current-native-reference-conversion','title':'Current Native Reference Conversion',
        'code_file':path.name,'language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':False,'enable_internet':False,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]},indent=2)+'\n')
    return path


if __name__ == '__main__':
    print(build())
