"""Prepared 09 sources for the current two-mode cohort contract."""
from pathlib import Path
from hashlib import sha256
import nbformat
from current_notebook_cells import bootstrap_cell

ROOT=Path(__file__).resolve().parents[1]
STAGES={
 '08f1':('08f1-toysad8k-calibration-resource-release.ipynb','Calibration current-cache release','calibration',600,'release'),
 '09a':('09a-toysad8k-protected-calibration-inference.ipynb','Calibration cached prediction','calibration',600,'prediction'),
 '09b':('09b-toysad8k-calibration-review-policy.ipynb','Calibration fixed-ranking evaluation','calibration',600,'evaluation'),
 '09b1':('09b1-toysad8k-internal-lock-resource-release.ipynb','Internal-lock current-cache release','internal_lock',400,'release'),
 '09c':('09c-toysad8k-protected-internal-lock-inference.ipynb','Internal-lock cached prediction','internal_lock',400,'prediction'),
 '09d':('09d-toysad8k-internal-lock-one-shot-evaluation.ipynb','Internal-lock one-shot ranking evaluation','internal_lock',400,'evaluation'),
}


def build_cells(stage):
    filename,title,role,count,kind=STAGES[stage]
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    cells=[md(f'''# {stage.upper()}: {title}

**Notebook file:** `{filename}`

This prepared notebook uses the frozen **automatic winner consensus** and **correspondence-assisted foreground patch-bank strongest-two fusion**. It preserves all {count} original `{role}` scenes. Two, three and five views are core settings; one view is diagnostic. The nineteen schedules are fixed before evaluator access.

This is a current-method adapter, not the old P1-only body. Preparation and synthetic tests do not establish a completed protected run. The source snapshot, current runtime freeze, the declared asset-exposure policy and exact native cache/evaluator resources must pass the executable preflight.'''),
        md('''## Resolve the complete source snapshot and explicit stage configuration

Mount the prepared source package or set `VPI_CODE_ROOT`. The standard-library bootstrap verifies the source manifest and exact Python/dependencies before scientific imports. Set `VPI_NEXT_CONFIG` or mount the unambiguous stage configuration. Nested Kaggle inputs resolve to their sealed package roots. Do not attach a full old evaluator notebook output to prediction stages. This notebook contains no credentials, model download or inference command.'''),
        bootstrap_cell('notebooks/protected-evaluation/'+filename),
        code(f"cfg=_runtime.resolve_config('{stage}',ROOT)\n"+'''from vpi_next_evaluation.contracts import current_method_contract
from vpi_next_evaluation.original import readiness, predict_current, evaluate_predictions, release_internal, release_calibration
from vpi_remaining_evaluation.gates import PrerequisiteBlocked
from vpi_object_bank.storage import read_json, verify_seal, atomic_json
from current_stage_execution import run_stage
display(current_method_contract())'''),
        md('''## Verify native resources, exposure and the current runtime freeze

The method freeze is produced from the completed 09E decision plus the current application's source and runtime versions. The old 08F/P1 freeze cannot pass. Before this sequence, inventory compatible outputs for the remaining 900 development scenes, verify any defensible confirmation cohort, and prepare the calibration cache/resource bridge. CPU processing reuses saved scores and patches. The separate current input-preparation adapter extracts missing DINO patches with a pinned CPU FP32 neural runtime; genuinely missing LocateAnything or SAM regions use the separate current-native-region-inference notebook.

An absent or invalid asset-exposure ledger stops execution. The explicitly exposure-qualified policy retains unknown access history and forbids independent or untouched claims. Correct scene counts without the canonical opaque-scene membership hash cannot pass. Native cache preparation and reviewed scene-to-asset attribution are real prerequisites; no readiness flag replaces them.'''),
        code(f'''def check_stage_resources():
    preflight=readiness('{stage}',cfg,ROOT)
    if preflight['status']!='prerequisites_present':
        raise PrerequisiteBlocked('CURRENT_STAGE_BLOCKED',preflight['detail'])
    return preflight
preflight=run_stage('{stage}-preflight',cfg,check_stage_resources,CODE_PROVENANCE)
display(preflight)''')]
    if kind=='prediction':
        cells += [md('''## Produce both fixed decisions from selected-view cached evidence

This CPU stage receives no anomaly labels. It preserves every valid observation for assisted fusion and records the automatic winner-consensus retention separately. Unknown assisted IDs stay unknown; each ranking carries support and per-view component scores. Same-view duplicate assignments and unselected observations are rejected. Empty views remain explicit.

Scene checkpoints bind cache, method, exposure, source and schedule hashes. Interrupted units remain recovery evidence. Only a complete original-role denominator can create the final prediction lock.'''),
            code(f"receipt=run_stage('{stage}',cfg,lambda:predict_current('{stage}',cfg,ROOT),CODE_PROVENANCE)\ndisplay(receipt)\nverify_seal(Path(cfg['output_root']),name='prediction_lock.json')"),
            md('''## Inspect sealed prediction-only scores and hand off

Per-scene predictions and `per_view_scores.csv` are private. They include one combined ranking for each fixed setting. Rich RGB, masks, boxes, score matrices and projected coarse evidence are rendered after the evaluator boundary by the report adapter; missing cached media is reported explicitly. No evaluator should be mounted here.'''),
            code("display({'prediction_lock':'verified','private_score_exports':'scenes/<opaque ID>/per_view_scores.csv','evaluation_opened':False,'next':'release this model-only prediction resource for its fixed evaluator'})")]
    elif kind=='release':
        cells += [md('''## Release the validated current model cache

Calibration release requires the current runtime freeze and the declared exposure policy. Internal-lock release also requires the completed current calibration protocol receipt and the same exposure ledger. This adapter copies an already prepared current cache and verifies every byte. It does not convert old P1/raw archives, create missing neural features, or open/copy evaluator labels. Produce the new evaluator contract separately through the reviewed native release bridge.'''),
            code("receipt=run_stage('"+stage+"',cfg,lambda:"+("release_calibration" if stage=="08f1" else "release_internal")+"(cfg,ROOT),CODE_PROVENANCE)\ndisplay(receipt)\nverify_seal(Path(cfg['output_root']),name='release_lock.json')")]
    else:
        policy_note='''

09B currently evaluates the two frozen ranking modes. It does **not** fit a calibrated probability or readiness policy. The application keeps its provisional review status; these notebooks report ranking evidence and make no calibrated application-readiness claim.''' if stage=='09b' else ''
        cells += [md('''## Verify the complete prediction boundary, then evaluate once

Every mode and selected schedule is checked before the evaluator package is opened. The native evaluator must bind the exact cache, method and original membership. Its neutral scene-to-asset index must be explicitly reviewed in the exposure ledger. Old P1 evaluator files cannot pass by renaming them.

Results use tie-aware top-1, Recall@2, AP and MRR. Asset-group bootstrap intervals use 2,000 draws with seed 32040. Schedule averages are formed per scene before bootstrap. Candidate, pruning, association, unavailable-score and ranking losses retain the original scene denominator. The inherited identity-reference-coverage bookkeeping is corrected only in this new adapter.'''+policy_note),
            code(f"receipt=run_stage('{stage}',cfg,lambda:evaluate_predictions('{stage}',cfg,ROOT),CODE_PROVENANCE)\ndisplay(receipt)\nverify_seal(Path(cfg['output_root']),name='evaluation_lock.json')\nverify_seal(Path(cfg['output_root']),name='protocol_lock.json')"),
            md('''## Render graphs and private object evidence after scientific sealing

The reporting stage can be repaired independently. It exports aggregate metric/view/failure/identity/anomaly-count tables and confidence-interval charts. Private panels show original RGB, boxes, SAM masks, objects on black, matched IDs, per-view scores, combined ranking, support and correctly projected coarse patch evidence where cached geometry permits it. A tie has no unique winner; missing masks, views or heatmaps are recorded rather than invented.

The fixed first scene illustrates both modes at 2/3/5 views. One clearly labelled post-hoc failure is a report illustration, not a filtered evaluation subset.'''),
            code('''from vpi_next_evaluation.reporting import render_current_report
report_cfg={**cfg,'evaluation_root':cfg['output_root']}
report=run_stage('report',{**cfg,'output_root':cfg['report_root']},lambda:render_current_report(report_cfg),CODE_PROVENANCE)
display(pd.read_csv(Path(cfg['output_root'])/'shareable/metric_summary.csv'))
display(pd.read_csv(Path(cfg['output_root'])/'shareable/view_schedule_summary.csv'))
for name in report['aggregate_figures']:
    display(Image(filename=str(Path(cfg['report_root'])/'shareable/figures'/name)))
display(report)'''),
            md('''## Private galleries and scores

These outputs remain private and are removed entirely from any published executed notebook. Selected-object boxes describe object regions, not validated defect boundaries.'''),
            code('''private_root=Path(cfg['report_root'])/'private/galleries'
for gallery in sorted(private_root.glob('example-*')):
    display(Image(filename=str(gallery/'objects_masks_selection.png')))
    display(Image(filename=str(gallery/'score_matrix.png')))''')]
    cells += [md('''## Recovery and bounded publication

Scientific results are sealed before plotting or tracking. `scripts/publish_next_stage.py` publishes the explicit aggregate table/figure allowlist in a 180-second child process. Run it separately with `--stage`, `--evaluation-root`, `--report-root` and `--receipt` after verification. Keep private RGB/masks, object scores, raw IDs and evaluator records out of ClearML. A tracking failure leaves a pending receipt and never reruns prediction or inference. Kaggle interruption recovery requires exported checkpoint files; local checkpoint existence alone does not survive a lost session.'''),
        code("tracking={'status':'pending_separate_publication','private_artifacts_allowed':False,'scientific_results_preserved':True}\natomic_json(Path(cfg['output_root']).parent/'tracking'/'"+stage+"-pending.json',tracking)\ndisplay(tracking)")]
    for index,cell in enumerate(cells):
        if cell.cell_type == 'code' and cell.source.startswith("private_root="):
            cell.metadata['tags']=['private']
        cell['id']=sha256(f'{stage}/{index}/{cell.source}'.encode()).hexdigest()[:8]
    return cells


def write_stage(stage):
    filename,title,_,_,_=STAGES[stage]
    notebook=nbformat.v4.new_notebook(cells=build_cells(stage),metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'}})
    path=ROOT/'notebooks/protected-evaluation'/filename;nbformat.write(notebook,path)
    metadata_path=path.parent/f'kernel-metadata-{stage}.json'
    metadata={'id':f'esreekarreddy/{filename[:-6]}','title':f'{stage.upper()} current two-mode {title}',
        'code_file':filename,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,
        'enable_internet':False,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
    metadata_path.write_text(__import__('json').dumps(metadata,indent=2)+'\n')
    return path


if __name__=='__main__':
    for stage in STAGES:print(write_stage(stage))
