#!/usr/bin/env python3
"""Generate separate CPU raw-resource and GPU region-only preparation notebooks."""
from pathlib import Path
import json
import nbformat
from current_notebook_cells import bootstrap_cell
ROOT=Path(__file__).resolve().parents[1]
RGB='notebooks/input-preparation/current-native-rgb-release.ipynb'
REGIONS='notebooks/input-preparation/current-native-region-inference.ipynb'


def _save(relative,cells,gpu):
    path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.write(notebook,path)
    slug=path.stem
    meta={'id':f'esreekarreddy/{slug}','title':slug.replace('-',' ').title(),'code_file':path.name,'language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':gpu,'enable_internet':False,
        **({'machine_shape':'NvidiaTeslaT4'} if gpu else {}),
        'dataset_sources':[],'kernel_sources':[],'competition_sources':[]}
    (path.parent/f'kernel-metadata-{slug}.json').write_text(json.dumps(meta,indent=2)+'\n')
    return path


def build():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    rgb=[md('''# Prepare model-only RGB resources from the existing dataset

This administrative preparation preserves the original development1000, calibration600, internal-lock400 and official seen/unseen memberships. It reads the existing mounted indexed dataset. It creates a private source index and a physically separate opaque RGB package. No neural inference or scoring occurs.

The official source master is mixed administrative metadata. This trusted adapter projects scene identity, object count and RGB addresses only; it never reads anomaly-label files or scene2D/3D metadata. Keep the private index and source vault out of the model package and public reports.'''),
        bootstrap_cell(RGB),
        md('''## Resolve inputs and select one resource operation

Set VPI_NEXT_CONFIG to the stage JSON. `role` is development, calibration, internal_lock, seen or unseen. `action=index` requires raw_root and, for original roles, selection_root and administrative_root. Exact source/admin checksums and memberships are checked. Indexing precedes exposure preparation and performs no RGB extraction.

`action=release` requires rgb_index_root, raw_root, output_root and current method/exposure prerequisites. Original internal-lock additionally needs calibration_receipt_root; official roles need freeze_root, decision_root and official_exposure_root. The release contains only canonical RGB files and an opaque manifest. `maximum_new_scenes` bounds work per call.'''),
        code("CONFIG=_runtime.resolve_config('native-rgb',ROOT)\nACTION=CONFIG.get('action','index')\nfrom current_stage_execution import run_stage\nfrom vpi_next_evaluation.native_rgb import prepare_rgb_index,release_rgb\nif ACTION not in {'index','release'}: raise ValueError('Choose index or release')\nOPERATION=prepare_rgb_index if ACTION=='index' else release_rgb\nRESULT=run_stage('native-rgb-'+ACTION,CONFIG,lambda:OPERATION(CONFIG,ROOT),CODE_PROVENANCE)\ndisplay(RESULT)"),
        md('''## Retain the correct resource

For index, retain the private rgb_index.json plus artifact_lock.json and run_binding.json. Its raw source paths are administrative only. For release, mount output_root/model-input in the next model stage. Keep release receipts and atomic checkpoints in the private parent directory. A final release_lock.json appears only when every original scene is complete.

The release checks free disk space and maximum_output_bytes (default18GiB) for retained checkpoints, released copies and metadata before reading RGB bytes. Completed/resumed outputs must also fit that retained-output budget. Changed input hashes, corrupt media, wrong membership, extra model-package files and failed exposure prerequisites stop. Recovery exports must actually be retained with their sidecars; local checkpoints alone do not guarantee cross-session recovery.''')]
    region=[md('''# Prepare missing LocateAnything and SAM regions

This is a region-only GPU preparation stage. The models, prompt and proposal policy are fixed. It produces boxes and SAM masks, including explicit empty outcomes, and makes no unusual-object decision. It never loads DINO, computes P1, or opens evaluator labels.

Existing complete development regions should be imported directly into the cache preparation stage. Use this producer only for roles whose region outputs are missing, or resume its own incomplete region run.'''),
        bootstrap_cell(REGIONS,runtime_profile='requirements/profiles/linux-regions-cuda126.json'),
        md('''## Validate current gates and model inputs

Set VPI_NEXT_CONFIG. Required: role, rgb_root, output_root and current method/exposure prerequisites. RGB must be the verified canonical model-input package. All20 views are prepared because the fixed nineteen1/2/3/5-view schedules use their union; every later decision still reads only its selected views.

`action=plan` performs no inference. `action=extract` additionally requires locate_model_root with a verified immutable model_snapshot.json and sam_checkpoint. The exact Linux GPU profile must pass the setup probe first. Pin LocateAnything revision and SAM checkpoint/code revision. No automatic internet download or runtime repinning occurs.'''),
        code("CONFIG=_runtime.resolve_config('native-regions',ROOT)\nACTION=CONFIG.get('action','plan')\nfrom current_stage_execution import run_stage\nfrom vpi_next_evaluation.native_regions import plan_regions,extract_regions\nif ACTION not in {'plan','extract'}: raise ValueError('Choose plan or extract')\nOPERATION=plan_regions if ACTION=='plan' else extract_regions\nRESULT=run_stage('native-regions-'+ACTION,CONFIG,lambda:OPERATION(CONFIG,ROOT),CODE_PROVENANCE)\ndisplay(RESULT)"),
        md('''## Checkpoints, failure accounting and reuse

maximum_new_views bounds each invocation. The launch queue also sets shard_count/shard_index for deterministic partitions of at most100 scenes. A completed partition emits region_shard_lock.json and region_shard_manifest.json; the CPU region_merge operation verifies every partition before producing the complete role artifact_lock.json. Partial shards cannot be used as a complete cohort.

Proposal and region checkpoints are independent: a SAM failure does not rerun completed LocateAnything proposals. Two bounded LocateAnything attempts are allowed for no-valid-box, truncation or generation-timeout outcomes. OOM, permission, corrupt input and incompatible model/configuration errors stop; they are never counted as empty images. Empty proposals and masks remain explicit records.

SAM uses a bounded main-thread timeout. A stuck native GPU kernel can still require stopping the Kaggle process; retained completed checkpoints survive only if their files are exported or saved. The final artifact_lock.json seals all scene/view region records and provenance after complete coverage. Normalized mask hashes are computed from original256×256 uint8 pixels; the historical SAM hash is retained separately.

Pass region_format=current-native-regions, region_root=output_root, rgb_root and the exact artifact_lock.json SHA256 to the current cache preparation notebook. That separate stage reuses these masks and extracts only genuinely missing DINO outputs in its own CPU FP32 runtime. Do not rerun regions to repair charts or tracking.''')]
    return [_save(RGB,rgb,False),_save(REGIONS,region,True)]


if __name__=='__main__':
    for path in build():print(path)
