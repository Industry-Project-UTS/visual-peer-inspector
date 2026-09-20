"""Current two-mode notebook cells; no legacy P1 execution body is retained."""
from hashlib import sha256
import nbformat
from current_notebook_cells import bootstrap_cell

def md(value):
    return nbformat.v4.new_markdown_cell(value.strip())

def code(value):
    return nbformat.v4.new_code_cell(value.strip())

COMMON = '''from current_stage_execution import run_stage
from vpi_next_evaluation.contracts import current_method_contract
from vpi_next_evaluation.official import verify_prerequisites
CONTRACT = current_method_contract()
display({'methods': CONTRACT['methods'], 'selection_closed': CONTRACT['selection_closed'],
         'execution_status': 'prepared source; prerequisite artifacts still required'})
display(pd.DataFrame([{'view_set_id': name, 'view_count': len(views), 'selected_views': ', '.join(views)}
                      for name, views in CONTRACT['view_sets'].items()]))

def load_config():
    return _runtime.resolve_config(STAGE, CODE_ROOT)
'''

CONTRACT_TEXT = '''## Frozen scientific contract

- Automatic: `winner_cluster_consensus`, using forward/P1/P2 normalized-rank consensus and the existing 0.20 winner association threshold.
- Assisted: `object_bank_top2`, excluding every selected observation of the target identity from its peer bank; foreground patch nearest distances, lower-median peer reduction, floor-rounded strongest 2% tail, within-view ranks, strongest-two fusion.
- Predicted LocateAnything-3B boxes, SAM 2.1 masks and DINOv2-small remain fixed. Supplied masks are a separate diagnostic, not this automatic official route.
- The shared contract prints every exact frozen schedule. Two, three and five views are core; one view is diagnostic. A restricted-view decision receives only its selected views and derived features. No 20-view identity map may assist a three-view run.
- `dataset_reference` neutral correspondence and `user_confirmed` correspondence are different assistance sources. Neither supplies the unusual-object answer. Report both separately from automatic RGB-only results.
- One scientific unit per scene; related scenes stay grouped by asset. Retain empty/missing/unrankable scenes and ties. Report tie-aware top-1, Recall@2, AP, MRR and asset-grouped 95% intervals. Schedules do not multiply the scene denominator.
- Official seen-test (715 scenes / 113 assets) and unseen-test (2,437 scenes / 316 assets) stay separate from each other and from the original 1,000 development / 600 calibration / 400 internal-lock roles. Audit prior access; renaming a split does not make it untouched.

## Execution readiness

Prepared current-method source is not an executed official experiment. Passing guards is not full readiness. Required inputs are a new exact runtime freeze, a completed current 09D receipt for both modes, the resolved original asset-access audit, and a separate official membership/exposure audit. Existing 08F/P1 freezes, caches and prediction locks are incompatible.

Compatible cache release, CPU decisions, finalization and typed official evaluation are implemented. Native official RGB/cache extraction and evaluator conversion still need validated bridges. The evaluator accepts only a sealed current-schema truth package with reviewed asset mapping and exact candidate/cache correspondence. Preparation opens no source archive, protected label or model host.
'''

REPORT_TEXT = '''## Reporting and recovery

Seal predictions before evaluator access. CPU decisions save atomic scene checkpoints and resume only with matching input, source, runtime and schedule hashes. Corrupt or incompatible checkpoints stop. Use `scripts/current_stage_recovery.py` to export a versioned archive and sidecar after a checkpoint; both files must actually be retained or downloaded. Explicit restore verifies the archive and exact binding before normal scene-resume checks. No automatic cross-session persistence is claimed.

Report seen and unseen separately, with per-mode/per-schedule denominators and asset-grouped intervals. Do not select methods using official outcomes or promise the 60% goal. Rich evidence must include selected original views, consistent IDs, boxes, masks, objects on black, coarse correctly projected heatmaps, raw component scores, normalized ranks, fusion contributors, ties and missing links. The report adapter checks sealed image/mask coordinates before drawing, saves per-object score tables and reports missing media explicitly.

Keep private galleries/IDs outside aggregate publication. Seal science first, then publish ClearML in a separate bounded process using allowlisted aggregates. Tracking failure must retain science and a pending receipt. Official image reports require compatible cached media. The bounded publisher and artifact allowlist are locally tested with synthetic outputs; live ClearML publication remains unverified for these stages. Never rerun inference to repair reporting.
'''

def base_cells(stage, filename, title):
    return [md(f'# {title}\n\n**Notebook file:** `{filename}`\n\nCurrent two-mode preparation. Previous P1 execution is not part of this notebook.'),
            md(CONTRACT_TEXT), code(f'NOTEBOOK_FILE = {filename!r}\nSTAGE = {stage!r}'), bootstrap_cell('notebooks/official-evaluation/' + filename), code(COMMON)]

def build_release_cells():
    cells = base_cells('10r', '10r-toysad8k-official-release.ipynb', '10R: Current Official Resource Release Preparation')
    return cells + [md('## Verify gates before locating source archives\n\nRelease one prevalidated current official cache at a time. Keep seen/unseen model-facing resources and evaluator-only resources separate. This operation checks every neutral scene, exact membership, media allowlist, storage and current 09D/exposure before a byte-preserving atomic copy. It does not convert an unknown raw source archive.'),
        code("CONFIG = load_config()\nfrom vpi_next_evaluation.official import release_verified_cache\nRELEASE_RECEIPT = run_stage(STAGE, CONFIG, lambda: release_verified_cache(CONFIG, CODE_ROOT), CODE_PROVENANCE)\ndisplay(RELEASE_RECEIPT)"), md(REPORT_TEXT)]

def build_prediction_cells(role):
    stage = '10a' if role == 'seen' else '10a1'
    filename = f'{stage}-toysad8k-official-{role}-frozen-inference.ipynb'
    cells = base_cells(stage, filename, f'{stage.upper()}: Official {role.title()} Current-Method Cached Predictions')
    return cells + [md('## Verify role and complete cache\n\nThis CPU stage reuses compatible saved regions/features. It does not run the legacy GPU/P1 body. A missing cache needs a validated current extractor. Each mode uses a separate output directory. No accuracy metrics are computed.'),
        code(f"PINNED_OFFICIAL_COHORT = {role!r}\nREQUESTED_OFFICIAL_COHORT = os.getenv('VPI_OFFICIAL_COHORT', PINNED_OFFICIAL_COHORT)\nif REQUESTED_OFFICIAL_COHORT != PINNED_OFFICIAL_COHORT:\n    raise ValueError('This notebook is pinned to {role}')\nCONFIG = load_config()\nif CONFIG.get('role') != PINNED_OFFICIAL_COHORT:\n    raise ValueError('Configuration role differs from this notebook')\nFREEZE, PREREQUISITES = verify_prerequisites(CONFIG, CODE_ROOT)\nfrom vpi_next_evaluation.official import validate_cache\nMANIFEST = validate_cache(CONFIG['cache_root'], PINNED_OFFICIAL_COHORT, PREREQUISITES['method_freeze_sha256'])\ndisplay({{'role': MANIFEST['role'], 'scenes': MANIFEST['scene_count'], 'mode': CONFIG.get('mode', 'automatic'), 'labels_opened': False}})"),
        md('## Apply the frozen selected-view decision'),
        code("from vpi_next_evaluation.official import predict_official\nPREDICTION_RECEIPT = run_stage(STAGE, CONFIG, lambda: predict_official(CONFIG, CODE_ROOT), CODE_PROVENANCE)\ndisplay(PREDICTION_RECEIPT)"),
        md('## Verify the prediction seal before evaluator access'),
        code("from vpi_next_evaluation.official import validate_prediction\nVERIFIED = validate_prediction(CONFIG['output_root'], PINNED_OFFICIAL_COHORT, cache_root=CONFIG['cache_root'])\ndisplay({'prediction_lock_sha256': VERIFIED['prediction_lock_sha256'], 'accuracy_evaluated': False, 'tracking': 'not published'})"), md(REPORT_TEXT)]

def build_finalize_cells():
    cells = base_cells('10a-finalize', '10a-finalize-toysad8k-official-prediction-summary.ipynb', '10A Finalize: Separate Official Prediction Locks')
    return cells + [md('## Join compatible locks without pooling results\n\nBoth complete roles must share the method, mode, correspondence provenance, schedules and current 09D/exposure bindings. Finalize automatic and assisted pairs separately. This label-free operation verifies coverage and computes no accuracy.'),
        code("CONFIG = load_config()\nfrom vpi_next_evaluation.official import finalize_official\nSUMMARY = run_stage(STAGE, CONFIG, lambda: finalize_official(CONFIG, CODE_ROOT), CODE_PROVENANCE)\ndisplay(pd.DataFrame(SUMMARY['roles']))\ndisplay({key: SUMMARY[key] for key in ['status', 'mode', 'correspondence_provenance', 'accuracy_evaluated', 'prediction_locks']})"), md(REPORT_TEXT)]

def build_evaluation_cells(configuration=None):
    cells = base_cells('10b', '10b-toysad8k-official-seen-unseen-evaluation.ipynb', '10B: Current Official Evaluation Contract and Sealed Reports')
    cells += [md('## Verify predictions before evaluator access\n\nThe typed evaluator requires both modes and both complete official roles. It verifies every prediction, current freeze, 09D, exposure audit, reviewed asset index and cache/candidate binding before opening scene truth. Native evaluator files need an explicit conversion bridge; legacy P1 outputs cannot be relabelled. Report-only mode repairs displays from sealed results.'),
        code("CONFIG = load_config()\nif CONFIG.get('action', 'evaluate') not in {'evaluate', 'report_only'}:\n    raise ValueError('Choose evaluate or report_only')\nif CONFIG.get('action', 'evaluate') == 'evaluate':\n    from vpi_next_evaluation.official_evaluation import evaluate_official\n    EVALUATION_RECEIPT = run_stage(STAGE, CONFIG, lambda: evaluate_official(CONFIG, CODE_ROOT), CODE_PROVENANCE)\n    CONFIG['evaluation_root'] = CONFIG['output_root']\nfrom vpi_next_evaluation.official_reporting import read_official_aggregates\nTABLE, EVALUATION_RECEIPT = read_official_aggregates(CONFIG, CODE_ROOT)\ndisplay(EVALUATION_RECEIPT)\ndisplay(TABLE)"),
        md('## Primary schedules and uncertainty\n\nIntervals use asset groups. Pooled descriptive rows cannot replace either split; supplied-ID performance does not measure human-confirmed assistance.'),
        code("primary = TABLE[TABLE['view_set_id'].str.endswith('primary')]\ndisplay(primary.pivot_table(index=['role', 'mode', 'correspondence_provenance', 'view_count'], columns='metric', values='mean'))\ndisplay(primary[['role', 'mode', 'view_count', 'metric', 'mean', 'ci_low', 'ci_high', 'scene_count', 'asset_group_count']])"), md('## Repairable aggregate and private visual evidence'),
        code("if CONFIG.get('report_root'):\n    from vpi_next_evaluation.official_reporting import render_official_report\n    from IPython.display import Image as NotebookImage\n    REPORT_RECEIPT = run_stage(STAGE + '-report', {**CONFIG, 'output_root': CONFIG['report_root']}, lambda: render_official_report(CONFIG, CODE_ROOT), CODE_PROVENANCE)\n    display(REPORT_RECEIPT)\n    for name in REPORT_RECEIPT['aggregate_figures']:\n        display(NotebookImage(filename=str(Path(CONFIG['report_root']) / 'shareable/figures' / name)))\nelse:\n    display({'report_status': 'not requested; configure report_root for charts and private galleries'})"),
        md('## Private scene evidence\n\nThe fixed first-scene illustration is private review material. Publication sanitization must strip this cell and its outputs.'),
        code("if CONFIG.get('report_root'):\n    private_examples = sorted((Path(CONFIG['report_root']) / 'private/galleries').rglob('objects_masks_selection.png'))\n    for path in private_examples[:2]:\n        display(NotebookImage(filename=str(path)))"), md(REPORT_TEXT)]
    cells[-2].metadata['tags']=['private']
    cells += [md('## Optional aggregate publication after sealing\n\nSet `publish_tracking=true` only after review. This bounded process leaves a pending receipt on failure. `action=report_only` reuses sealed science and may retry publication without evaluator access or inference.'),
        code("if CONFIG.get('publish_tracking', False):\n    import subprocess\n    if not CONFIG.get('report_root'):\n        raise ValueError('Aggregate publication requires a separately sealed report_root')\n    tracking_receipt = Path(CONFIG.get('tracking_receipt', str(Path(CONFIG['report_root']).parent / 'tracking' / '10b.json')))\n    tracking_receipt = _runtime.writable_output(tracking_receipt)\n    if any(Path(CONFIG[name]).resolve() in tracking_receipt.parents for name in ('evaluation_root', 'report_root')):\n        raise ValueError('Tracking receipts must be outside sealed science and reports')\n    command = [sys.executable, str(CODE_ROOT / 'scripts/publish_next_stage.py'), '--stage', '10b', '--evaluation-root', str(CONFIG['evaluation_root']), '--report-root', str(CONFIG['report_root']), '--receipt', str(tracking_receipt)]\n    try:\n        subprocess.run(command, timeout=200, check=True)\n    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:\n        pending = json.loads(tracking_receipt.read_text()) if tracking_receipt.is_file() else {}\n        pending.update(status='pending', error_type=type(error).__name__, science_preserved=True)\n        _runtime.atomic_json(tracking_receipt, pending)\n    display({'tracking': 'inspect receipt', 'scientific_results_preserved': True})\nelse:\n    display({'tracking': 'not requested', 'scientific_results_preserved': True})")]
    return cells

def clean_notebook(cells, filename):
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'}}
    for index, cell in enumerate(cells):
        cell['id'] = sha256(f'{filename}/{index}/{cell.source}'.encode()).hexdigest()[:12]
    return notebook
