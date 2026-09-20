#!/usr/bin/env python3
"""Build the clean, unexecuted native-cache preparation notebook."""
from pathlib import Path
import json
import nbformat
from current_notebook_cells import bootstrap_cell

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = 'notebooks/input-preparation/current-native-regions-to-object-cache.ipynb'


def build():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# Prepare the current multi-view cache from saved native regions

The preparation and prediction actions reuse saved LocateAnything boxes and SAM masks without reading anomaly labels. Only the explicit `dev-evaluate` action opens the separately bound development evaluator after complete prediction seals pass. No action trains a scorer or runs the old P1 pipeline. **CPU FP32 DINO** matches the application feature contract. No scientific dataset run has been performed by preparing this notebook.

Supported regions: a sealed08G comparison-region package; one native08E/protected `prediction_manifest.jsonl` plus its verified RGB package; the exact100+600+300 native development components and combined development lock; or a current-native-regions resource. Membership must match an existing1000/600/400 original role or715/2437 official role. Old P1 scores, heatmaps and float16 global embeddings are ignored. Full grid08N features can be reused only when region IDs, crop bytes, CPU runtime, model and config hashes match.

When LA/SAM outputs are missing, the plan records that prerequisite. The separate native-region preparation notebook supplies only missing LA/SAM outputs under the current freeze. Raw archive conversion and anomaly-label/evaluator conversion are outside this notebook. The old end-to-end09A runner is incompatible with the current score contract.'''),
        bootstrap_cell(RELATIVE),
        md('''## Explicit inputs and prediction boundary

Set `VPI_NEXT_CONFIG` to a private JSON configuration. Required keys: `role`, `region_format`, `region_root` (or `components` plus `full_development_lock`), `expected_region_lock_sha256`, `feature_root`, `neutral_root`, `output_root`, and current freeze/exposure/prerequisite roots. Native prediction manifests also need `rgb_root`. Development components use cohort IDs as keys, each with `region_root` and `rgb_root`.

`neutral_root` contains exactly a sealed `neutral_correspondence.json`: schema `vpi-current-neutral-correspondence/v1`, the exact region-lock hash, provenance `dataset_reference` or `user_confirmed`, and `observations` with only scene, view_id, candidate_id and object_id. Every candidate requires an explicit ID or null; duplicate IDs in one view are rejected. This map supplies identities, never the unusual-object answer.'''),
        code("CONFIG = _runtime.resolve_config('native-cache', ROOT)\nfrom vpi_next_evaluation.native_cache import plan_native, extract_missing, assemble_native, prepare_shard\nACTION = CONFIG.get('action', 'plan')\nPLAN = plan_native(CONFIG, ROOT) if ACTION in {'plan', 'extract_missing', 'assemble', 'prepare_shard'} else {'status':'typed cached development continuation; no region/model work'}\ndisplay(PLAN)"),
        md('''## Choose one explicit operation

`action=plan` is the default and performs no neural work. `action=extract_missing` requires `model_source` and the separately pinned DINO environment (torch2.13.0, Transformers5.14.1, NumPy2.5.3, Pillow12.3.0). It encodes only missing automatic-region crops, at most four per batch. Corrupt checkpoints and changed bindings stop. A completed08N cache is reusable; finish an incomplete08N run with its original source or use a fresh feature directory.

`action=assemble` recomputes the fixed forward/P1/P2 components from existing full grids and saves foreground banks as compressed FP32 NPZ, with selected-view lazy loading. It saves per-scene atomic checkpoints, all-candidate geometry and reusable images. No full grids enter JSON. The conservative output budget includes checkpoints and released copies. Set `maximum_output_bytes` only to a verified available volume. `max_new_scenes` supports incremental assembly; a final cache seal requires the complete original cohort.

`remap_audit` verifies the historical08N automatic tensors against the native100+600+300 region lineage. It checks the exact hashed-ID crosswalk, RGB bytes, packed/native and raw/08N mask hashes, crop coordinates and FP32 arrays. `remap_import` copies only the verified automatic NPZ files byte-for-byte into a new full-cohort binding; it never renames tensor indices or changes feature values. The original features remain sealed.

Use `shard_count` and zero-based `shard_index` for deterministic lexical scene partitions. `max_new_units` and `max_new_scenes` bound each continuation. `prepare_shard` performs missing CPU DINO extraction and cache assembly in one job. Set `feature_roots` to immutable compatible mounted reuse shards; `feature_root` remains the writable destination for genuinely missing units. A partial feature directory is exported with `feature_shard_lock.json`; completed scene partitions use `cache_shard_lock.json` and cannot be evaluated independently as a complete cohort.

`merge_cache_shards` accepts `cache_shards` plus the normal role/gate/output configuration. It verifies every shard, rejects overlaps and incomplete original membership, copies only released RGB/foreground tensors and metadata, and emits the final full-cohort seal. It does not copy raw full-grid checkpoints. Size partitions so every job fits its reviewed output limit and the final merge fits the platform input-mount limit; retain all shards before ending each session.

For the complete original1000-scene development role only, `dev-predict` produces both fixed methods over the same nineteen schedules from the typed cache; `dev-evaluate` verifies those predictions before opening the separately bound development evaluator and renders reusable reports. These actions never create calibration/internal-lock protocol receipts or an independent-confirmation claim. They use the same current adapters, without another method search. Configure cache_root; dev-evaluate additionally needs prediction_root, evaluator_root and report_root. Use separate output directories for assembly, prediction, evaluation and reports.'''),
        code("if ACTION not in {'plan', 'extract_missing', 'assemble', 'prepare_shard', 'remap_audit', 'remap_import', 'merge_cache_shards', 'dev-predict', 'dev-evaluate'}:\n    raise ValueError('Unsupported preparation or fixed-development action')\nif ACTION == 'extract_missing':\n    CONFIG['feature_root'] = str(_runtime.writable_output(CONFIG['feature_root']))\n    RESULT = extract_missing(CONFIG, ROOT)\nelif ACTION == 'assemble':\n    CONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\n    RESULT = assemble_native(CONFIG, ROOT)\nelif ACTION == 'prepare_shard':\n    CONFIG['feature_root'] = str(_runtime.writable_output(CONFIG['feature_root']))\n    CONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\n    RESULT = prepare_shard(CONFIG, ROOT)\nelif ACTION == 'remap_audit':\n    from vpi_next_evaluation.feature_remap import audit_remap\n    CONFIG['remap_root'] = str(_runtime.writable_output(CONFIG['remap_root']))\n    RESULT = audit_remap(CONFIG)\nelif ACTION == 'remap_import':\n    from vpi_next_evaluation.feature_remap import import_remapped\n    CONFIG['feature_root'] = str(_runtime.writable_output(CONFIG['feature_root']))\n    RESULT = import_remapped(CONFIG)\nelif ACTION == 'merge_cache_shards':\n    from vpi_next_evaluation.cache_shards import merge_cache_shards\n    CONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\n    RESULT = merge_cache_shards(CONFIG, ROOT)\nelif ACTION == 'dev-predict':\n    CONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\n    from vpi_next_evaluation.original import predict_current\n    RESULT = predict_current('dev-predict', CONFIG, ROOT)\nelif ACTION == 'dev-evaluate':\n    CONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\n    CONFIG['report_root'] = str(_runtime.writable_output(CONFIG['report_root']))\n    from vpi_next_evaluation.original import evaluate_predictions\n    RESULT = evaluate_predictions('dev-evaluate', CONFIG, ROOT)\n    from vpi_next_evaluation.reporting import render_current_report\n    REPORT = render_current_report({**CONFIG, 'evaluation_root': CONFIG['output_root']})\n    display(REPORT)\n    for name in REPORT['aggregate_figures']:\n        display(Image(filename=str(Path(CONFIG['report_root']) / 'shareable/figures' / name)))\nelse:\n    RESULT = PLAN\ndisplay(RESULT)"),
        md('''## Retain outputs and hand off

Keep full DINO grids, atomic checkpoints and the assembled model cache as separate retained resources. The model cache seal includes only its cohort manifest, scene metadata, RGB and compressed foreground tensors. Preparation receipts and recovery state are private. A successful cache conversion is not a completed accuracy evaluation.

Mount only the model cache in09A/09C/10A/10A1. Typed evaluator packages and reviewed asset maps remain separate until complete predictions are sealed. Use the shared verified recovery exporter described in the Kaggle preparation guide; checkpoint files must actually be retained or exported before ending a session. Never rerun neural extraction to repair reports.''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3','language':'python','display_name':'Python3'}})
    path = ROOT / RELATIVE; path.parent.mkdir(parents=True, exist_ok=True); nbformat.write(notebook, path)
    metadata = {'id':'esreekarreddy/current-native-regions-to-object-cache', 'title':'Current Native Regions to Object Cache',
        'code_file':path.name, 'language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':False,'enable_internet':False,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]}
    (path.parent / 'kernel-metadata-native-cache.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path


if __name__ == '__main__':
    print(build())
