#!/usr/bin/env python3
"""Build the completed 09E notebook without editing future-stage notebooks."""
from pathlib import Path
from hashlib import sha256
import nbformat

ROOT = Path(__file__).resolve().parents[1]
TAG = 'vpi-current-method-guard'
DIRECTORY = ROOT / 'notebooks/protected-evaluation'


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


def apply_current_guard(notebook, stage):
    if stage not in ('09a','09b','09b1','09c','09d'):
        return notebook
    notebook.cells = [cell for cell in notebook.cells if TAG not in cell.metadata.get('tags', [])]
    note = md(f"## Current-method readiness before {stage.upper()}\n\n"
        "The current frozen modes are automatic **winner consensus** and assisted **object-bank strongest-two fusion**. "
        "The preserved body below implements the earlier 08F/P1 contract and is not a current-method adapter.\n\n"
        "Run **09E Remaining-cohort readiness and evidence workbench** first. Its resource recipe preserves "
        "1,000 development, 600 calibration and 400 internal-lock scenes. The executable guard requires a resolved, "
        "sealed asset-exposure audit and then stops on the missing current-method adapter before model or evaluator setup. "
        "An empty or unresolved protected cohort cannot pass. Protected labels cannot choose development methods.")
    guard = code("\n".join([
        "from pathlib import Path", "import os", "import sys",
        "_candidates = [Path(os.environ['VPI_CODE_ROOT'])] if os.getenv('VPI_CODE_ROOT') else []",
        "_candidates += [Path.cwd(), *Path.cwd().parents]",
        "_current_root = next((root for root in _candidates if (root / 'src/vpi_remaining_evaluation/gates.py').is_file()), None)",
        "if _current_root is None:",
        "    raise RuntimeError('CURRENT_CODE_SNAPSHOT_REQUIRED: mount the updated source tree containing vpi_remaining_evaluation before this stage.')",
        "sys.path.insert(0, str(_current_root / 'src'))",
        "from vpi_remaining_evaluation.gates import require_stage_adapter",
        f"require_stage_adapter('{stage}')",
    ]))
    for cell in (note, guard):
        cell.metadata['tags'] = [TAG]
        cell['id'] = sha256((stage + cell.source).encode()).hexdigest()[:8]
    notebook.cells[1:1] = [note, guard]
    return notebook


def protect_legacy():
    paths=[]
    for stage in ('09a','09b','09b1','09c','09d'):
        found=list(DIRECTORY.glob(stage+'-*.ipynb'))
        if len(found)!=1:
            raise ValueError('Expected one exact legacy protected notebook per stage')
        path=found[0]
        notebook=apply_current_guard(nbformat.read(path,as_version=4),stage)
        nbformat.write(notebook,path)
        paths.append(path)
    return paths

def build():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata['kernelspec'] = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
    notebook.cells = [
        md('''# 09E: Remaining-cohort readiness and evidence workbench

**Notebook file:** `09e-remaining-cohort-readiness-and-evidence.ipynb`

Prepare the next stage on the original 2,000-scene assignment while keeping the completed 08N/08O/08P scientific artifacts unchanged. This run verifies and reports the existing **100 development scenes / 27 assets**. It does not claim 2,000 evaluated scenes, run neural inference, open protected labels or select another scorer.

Frozen modes: **winner consensus** for the automatic development baseline; **object-bank strongest-two fusion** for supplied or user-confirmed correspondence. Dataset-reference and human-confirmed results require separate provenance and accuracy claims.'''),
        md('''## Resolve the local configuration and exact source tree

Use the complete current source snapshot. Input roots may point to nested or flat extracted packages, but each configured root must contain its exact scientific lock. Missing, conflicting or altered locks stop before tables or galleries are prepared.'''),
        code('''from pathlib import Path
from hashlib import sha256
import json
import os
import sys
import pandas as pd
from IPython.display import Image, display

ROOT = Path(os.environ.get('VPI_OBJECT_BANK_ROOT', Path.cwd())).absolute()
sys.path.insert(0, str(ROOT / 'src'))
CONFIG_PATH = Path(os.environ.get('VPI_OBJECT_BANK_CONFIG', ROOT / '.local/remaining-evaluation/config.json'))
cfg = json.loads(CONFIG_PATH.read_text())
SCIENTIFIC_ROOT = Path(cfg['scientific_root'])
REPORT_ROOT = Path(cfg['report_root'])
from vpi_remaining_evaluation.workbench import prepare_workbench
from vpi_remaining_evaluation.reporting import render_workbench
from vpi_remaining_evaluation.gates import PrerequisiteBlocked, require_stage_adapter
from vpi_object_bank.storage import verify_seal, atomic_json'''),
        md('''## Verify evidence and seal the complete technique matrix

All registered compatible 08O and 08P scores remain visible. The tables distinguish masks, correspondence, methods and view schedules. Primary schedules and scene-averaged schedule summaries are separate. Confidence intervals use asset groups, and missing or unrankable cases remain in the scientific denominators.

The inherited `identity_reference_coverage` statistic has a bookkeeping defect. Its original value is retained for audit, but the usable mean is blank and explicitly marked invalid. Ranking metrics and failure counts are unchanged. A separate app-runtime revalidation is not merged into these historical experiment conditions.'''),
        code('''receipt = prepare_workbench(cfg, SCIENTIFIC_ROOT, ROOT)
display(receipt)
registry = pd.read_csv(SCIENTIFIC_ROOT / 'shareable/technique_feature_matrix.csv')
display(registry)
verify_seal(SCIENTIFIC_ROOT, name='scientific_lock.json')'''),
        md('''## Original roles and executable protected-stage prerequisites

The 1,000/600/400 roles are unchanged. Earlier asset exposure remains unresolved. A completed development reporting run cannot unlock calibration or internal-lock evaluation. Current-method cached prediction hooks are separate from the older P1 resource and policy contracts; their full-role adapters must be verified before protected execution.'''),
        code('''display(pd.read_csv(SCIENTIFIC_ROOT / 'shareable/cohort_readiness.csv'))
display(json.loads((SCIENTIFIC_ROOT / 'shareable/resource_recipe.json').read_text()))
blocked_stages = []
for stage in ('09a', '09b', '09b1', '09c', '09d'):
    try:
        require_stage_adapter(stage, cfg.get('exposure_root'))
    except PrerequisiteBlocked as error:
        blocked_stages.append({'stage': stage, 'status': 'blocked', 'code': error.code, 'detail': str(error)})
if len(blocked_stages) != 5:
    raise RuntimeError('A protected stage unexpectedly passed an unimplemented current-method bridge')
display(pd.DataFrame(blocked_stages))'''),
        md('''## Render rich aggregate evidence after the scientific seal

Rendering is repairable. It never reruns LocateAnything, SAM or DINO. Charts retain every registered development comparison: view-count trends, uncertainty, primary-schedule matrices, schedule sensitivity and failure stages. Select report figures later without changing the evaluated cohort.'''),
        code('''report = render_workbench(cfg, SCIENTIFIC_ROOT, REPORT_ROOT, ROOT)
display(report)
for name in report['aggregate_figures']:
    display(Image(filename=str(REPORT_ROOT / 'shareable/figures' / name)))'''),
        md('''## Private multi-view objects, masks and scores

Panels show original images, boxes, consistent object IDs, SAM masks, objects on black, projected coarse patch evidence, all retained object scores and one combined ranking. A tied ranking has no unique selection. Missing views remain blank and explained. Red boxes locate the selected object region; they are not validated defect boundaries.

The first scene is fixed without outcomes for 1, 2, 3 and 5 views. Additional failure, tie and missing-view examples are explicitly post-hoc development illustrations. They are not a selected evaluation subset. These images and scene-level tables remain private.'''),
        code('''gallery_manifest = json.loads((REPORT_ROOT / 'private/gallery_manifest.json').read_text())
display(pd.DataFrame(gallery_manifest)[['route','view_set_id','method','reason','top_tie_count','unlocated_views']])
for index, item in enumerate(gallery_manifest):
    unit = REPORT_ROOT / 'private/galleries' / f'example-{index+1:02d}'
    display(Image(filename=str(unit / 'views_masks_selection.png')))
    display(Image(filename=str(unit / 'score_matrix.png')))
    display(pd.read_csv(unit / 'object_scores.csv'))'''),
        md('''## Publication and next-stage handoff

Scientific tables and report outputs are independently sealed before any external tracking. Publication is a separate bounded process over aggregate allowlisted files. Private galleries, raw source identities and recovery data are excluded. A tracking failure preserves successful computation.

The next protected run remains blocked by the exposure audit and current-method adapters. The cached cohort hook recomputes the canonical membership hash from actual opaque scene IDs, restricts selected views and writes atomic prediction checkpoints. Its 1,000/600/400-size tests use explicitly synthetic membership contracts; they are not real calibration or internal-lock results. Native scene-to-asset attribution still requires the resource/evaluator bridge.'''),
        code('''verify_seal(SCIENTIFIC_ROOT, name='scientific_lock.json')
verify_seal(REPORT_ROOT, name='report_lock.json')
tracking_receipt = {'status':'pending_separate_publication', 'scientific_results_preserved':True,
                    'report_sealed':True, 'private_artifacts_allowed':False,
                    'note':'Run a bounded publisher over verified shareable artifacts; never repeat inference for tracking.'}
atomic_json(REPORT_ROOT.parent / 'tracking/pending_receipt.json', tracking_receipt)
display(tracking_receipt)
display({'completed':'existing development evidence and protected compatibility preflight',
         'new_protected_scenes_evaluated':0, 'protected_stages':'blocked; see explicit prerequisite records'})'''),
    ]
    path = DIRECTORY / '09e-remaining-cohort-readiness-and-evidence.ipynb'
    for index, cell in enumerate(notebook.cells):
        cell['id'] = sha256(f'09e/{index}/{cell.source}'.encode()).hexdigest()[:8]
    nbformat.write(notebook, path)
    return [path]


if __name__ == '__main__':
    for path in build():
        print(path.relative_to(ROOT))
