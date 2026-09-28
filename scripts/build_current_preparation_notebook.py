#!/usr/bin/env python3
"""Build one reusable CPU entry point for the launch graph's preparation jobs."""
from pathlib import Path
import json
import nbformat
from current_notebook_cells import bootstrap_cell

ROOT=Path(__file__).resolve().parents[1]
RELATIVE='notebooks/input-preparation/current-kaggle-preparation.ipynb'


def build():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# Prepare existing resources for the fixed multi-view evaluation

This reusable entry point executes one configured preparation operation. The launch kit binds existing Kaggle datasets and upstream outputs automatically; it does not re-upload the raw dataset. Models and the two selected scoring methods stay fixed. All scientific output, private checkpoints and reports remain separate.

Operations are explicit: method freeze, private RGB index, model-only RGB release, neutral reference geometry/correspondence, missing-feature and cache shards, cache merge, and post-prediction truth conversion. Only truth conversion opens anomaly outcomes, after complete model predictions verify. The trusted raw resource preparer may read mixed source metadata but exports only whitelisted neutral fields.'''),bootstrap_cell(RELATIVE),
        code("CONFIG = _runtime.resolve_config('preparation', ROOT)\nCONFIG['output_root'] = str(_runtime.writable_output(CONFIG['output_root']))\nfrom current_stage_execution import run_stage\nACTION = CONFIG['action']"),
        code('''def prepare():
    if ACTION == 'freeze':
        from vpi_next_evaluation.contracts import prepare_method_freeze
        return prepare_method_freeze(CONFIG['reference_root'], CONFIG['output_root'], ROOT)
    if ACTION in {'rgb_index','rgb_release'}:
        from vpi_next_evaluation.native_rgb import prepare_rgb_index, release_rgb
        if ACTION == 'rgb_release' and CONFIG.get('role') == 'unseen':
            from current_rgb_transport import reuse_completed_transport
            retained = reuse_completed_transport(CONFIG['output_root'])
            if retained is not None:
                return retained
        return (prepare_rgb_index if ACTION=='rgb_index' else release_rgb)(CONFIG,ROOT)
    if ACTION == 'raw_geometry':
        from current_official_geometry import prepare_official_mask_geometry
        return prepare_official_mask_geometry(CONFIG)
    if ACTION == 'raw_truth':
        from current_truth_conversion import convert_raw_evaluator_with_typed_rejections
        return convert_raw_evaluator_with_typed_rejections(CONFIG)
    if ACTION == 'neutral':
        from vpi_next_evaluation.native_evaluator import prepare_neutral
        return prepare_neutral(CONFIG)
    if ACTION == 'cache_merge':
        from vpi_next_evaluation.cache_shards import merge_cache_shards
        return merge_cache_shards(CONFIG,ROOT)
    if ACTION == 'region_merge':
        from vpi_next_evaluation.region_shards import merge_region_shards
        return merge_region_shards(CONFIG,ROOT)
    if ACTION == 'prepare_shard':
        from vpi_next_evaluation.native_cache import prepare_shard
        return prepare_shard(CONFIG,ROOT)
    if ACTION == 'official_exposure':
        from current_official_exposure import prepare_official_exposure_from_geometry
        return prepare_official_exposure_from_geometry(CONFIG)
    raise ValueError('Unknown prepared launch operation')

RESULT = run_stage('preparation-'+ACTION,CONFIG,prepare,CODE_PROVENANCE)
if ACTION == 'rgb_release' and CONFIG.get('role') == 'unseen':
    from current_rgb_transport import compact_completed_rgb
    RGB_TRANSPORT = compact_completed_rgb(CONFIG['output_root'],RESULT,role='unseen')
    display({'rgb_transport_status':RGB_TRANSPORT['status'],
             'archive_sha256':RGB_TRANSPORT['archive_sha256'],
             'scene_count':RGB_TRANSPORT['scene_count'],
             'model_file_count':RGB_TRANSPORT['file_count']})
display(RESULT)'''),md('''## Completion and recovery

A checkpointed result is an incomplete job. Keep its retained output and resume with the same configuration; never let a downstream job consume a partial cohort. Cache merging verifies exact membership and rejects overlapping or missing shards. The final decision still uses the complete declared cohort denominator.

Download each completed job once and verify all registered hashes. Keep source/checkpoints and private evaluator resources outside the later GitHub overlay. Tracking and figure repair use saved outputs without another model run.''')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python3'}})
    path=ROOT/RELATIVE;path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,path)
    return path


if __name__=='__main__':print(build())
