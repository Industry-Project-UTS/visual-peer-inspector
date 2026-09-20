#!/usr/bin/env python3
"""Build a clean setup-only Kaggle notebook; no scene or model inference."""
import json
from pathlib import Path
from textwrap import indent
import nbformat
from current_notebook_cells import LOCATOR
ROOT=Path(__file__).resolve().parents[1]
NAME='current-kaggle-runtime-setup.ipynb'

def build():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    locator=LOCATOR.split('BOOTSTRAP = _runtime.bootstrap')[0]
    setup_source = 'NOTEBOOK_RELATIVE = '+repr('notebooks/input-preparation/'+NAME)+'\n'+locator+"\nSOURCE = _runtime.verify_source(CODE_ROOT, NOTEBOOK_RELATIVE)\nsys.path.insert(0, str(CODE_ROOT / 'scripts'))\nprint({'source_snapshot_sha256': SOURCE['snapshot_sha256'], 'scientific_imports_started': False})"
    setup_source = "from pathlib import Path\nimport json, os\ntry:\n" + indent(setup_source, '    ') + "\nexcept BaseException as error:\n    receipt_root = Path('/kaggle/working/runtime-receipts') if Path('/kaggle').is_dir() else Path.cwd()/'.local/runtime-receipts'\n    receipt_root.mkdir(parents=True, exist_ok=True)\n    temporary = receipt_root/'setup-bootstrap.json.partial'\n    temporary.write_text(json.dumps({'status':'setup_bootstrap_failed','error_type':type(error).__name__,'scientific_run_started':False})+'\\n')\n    os.replace(temporary, receipt_root/'setup-bootstrap.json')\n    raise"
    cells=[md('''# Current Kaggle runtime setup

This setup-only notebook inventories the actual Linux kernel and prepares one isolated CPU, DINO, LocateAnything/SAM or publication environment. It never opens a scene, anomaly label or scientific evaluator and never calls a model. It preserves exact package pins while recording the observed Python 3.12 patch before the new method freeze.

The source remains a verified working-directory snapshot. Use a separate job for each profile. CPU setup does not need a GPU; CUDA wheels can also be prepared without allocating one. Region inference later requires the configured T4 accelerator. Retain the wheelhouse and runtime-profile output before starting downstream stages.'''),
        code(setup_source),
        md('''## Reviewed setup configuration

Set `VPI_NEXT_CONFIG` to a setup JSON containing `profile` (`cpu`, `dino`, `dino-gpu`, `regions` or `tracking`), `action` (`probe` or `prepare`), `output_root`, `target`, and `wheelhouse`. Set `build_wheelhouse=true` to build the exact offline artifacts on Linux from the source lock. The default action only records metadata. Retained outputs belong under `/kaggle/working`; dependency-only targets use fresh `/tmp/vpi-runtime-*` directories so thousands of installed files are not retained as notebook output. GPU and DINO use separate targets because their Torch/Transformers versions differ.'''),
        code("CONFIG = _runtime.resolve_config('runtime-setup', CODE_ROOT)\nPROFILE_NAMES = {'cpu': 'linux-cpu-cache-cp312', 'dino': 'linux-dino-fp32-cpu', 'dino-gpu': 'linux-dino-fp32-cuda126', 'regions': 'linux-regions-cuda126', 'tracking': 'linux-tracking-cp312'}\nif CONFIG.get('profile') not in PROFILE_NAMES or CONFIG.get('action', 'probe') not in {'probe', 'prepare'}:\n    raise ValueError('Choose one registered profile and probe/prepare action')\nprofile_path = CODE_ROOT / 'requirements/profiles' / (PROFILE_NAMES[CONFIG['profile']] + '.json')\noutput_root = _runtime.writable_output(CONFIG['output_root'])\nfrom kaggle_setup_probe import probe\nPROBE = probe([profile_path])\n_runtime.atomic_json(output_root / 'initial_setup_probe.json', PROBE)\nprint(PROBE)"),
        md('''## Build verified wheels and private runtime

Wheel builds are bounded and use official PyPI/PyTorch indexes. Only dependencies are resolved or built; nothing is installed into the notebook kernel. The next step installs the verified wheelhouse into a fresh private target and seals its exact package metadata and observed Python patch. This is setup verification, not scientific or model compatibility evidence.'''),
        code("if CONFIG.get('action', 'probe') == 'prepare':\n    wheelhouse = Path(CONFIG['wheelhouse'])\n    if CONFIG.get('build_wheelhouse', False):\n        wheelhouse = _runtime.writable_output(wheelhouse)\n        if CONFIG['profile'] == 'cpu':\n            from prepare_linux_wheelhouse import prepare as build_wheels\n        else:\n            from prepare_neural_wheelhouse import prepare as build_wheels\n        print(build_wheels(profile_path, wheelhouse))\n    from prepare_runtime_environment import prepare_isolated\n    base_profile = CODE_ROOT / 'requirements/current-evaluation-runtime.json' if CONFIG['profile'] == 'cpu' else profile_path\n    RECEIPT = prepare_isolated(base_profile, wheelhouse, _runtime.runtime_target(CONFIG['target']), output_root / 'runtime-profile')\n    print(RECEIPT)\nelse:\n    print({'status': 'metadata_only', 'dependencies_installed': False})"),
        md('''## Optional pinned model assets

Set `download_models=true` only in a DINO or regions setup job, plus `model_asset_root` and `models_only=all|regions|dino`. The bounded model downloader uses the existing immutable model revisions and checkpoint hashes. The outer timeout covers each selected model's 1,800-second download budget plus checksum time; timeout cleanup stops the whole child process group. It downloads assets without constructing or calling a model. Reuse its output in inference jobs.'''),
        code("if CONFIG.get('download_models', False):\n    if CONFIG.get('action') != 'prepare' or CONFIG['profile'] not in {'dino', 'dino-gpu', 'regions'}:\n        raise ValueError('Model asset download requires a prepared DINO or regions setup profile')\n    model_root = _runtime.writable_output(CONFIG['model_asset_root'])\n    print(_runtime.run_model_asset_preparation(CODE_ROOT, model_root, CONFIG.get('models_only', 'dino' if CONFIG['profile'] == 'dino-gpu' else CONFIG['profile']), CONFIG['target'], per_model_timeout=1800))\nelse:\n    print({'model_asset_download': 'not requested'})"),
        md('''## Downstream binding

A downstream job sets `VPI_RUNTIME_PROFILE_ROOT` to the sealed `runtime-profile` resource, `VPI_WHEELHOUSE` to the matching wheelhouse, `VPI_PACKAGE_TARGET` to a fresh target and `VPI_INSTALL_DEPENDENCIES=1` before its normal bootstrap. DINO CPU/GPU jobs set their matching exact `VPI_BASE_RUNTIME_PROFILE`; the region notebook selects its GPU profile explicitly.

The bootstrap verifies unchanged package/device pins, the base-profile hash and the observed Linux Python patch. The subsequent method freeze records the actual scientific runtime. A later runtime change cannot bypass a completed freeze. Run development compatibility checks before protected evaluation. Pinned model assets may be prepared by the optional bounded download cell above; no model inference is run.''')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}})
    for i,c in enumerate(cells):c['id']=__import__('hashlib').sha256((NAME+str(i)+c.source).encode()).hexdigest()[:12]
    target=ROOT/'notebooks/input-preparation';target.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,target/NAME)
    (target/'kernel-metadata-runtime-setup.json').write_text(json.dumps({'id':'esreekarreddy/vpi-current-runtime-setup','title':'VPI current runtime setup only','code_file':NAME,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':True,'dataset_sources':[],'kernel_sources':[],'competition_sources':[]},indent=2)+'\n')
    return target/NAME
if __name__=='__main__':print(build())
