#!/usr/bin/env python3
"""Build a deterministic source-only transport; no inputs, outputs or credentials."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from current_notebook_runtime import MANIFEST, SCHEMA, canonical, digest, verify_source, atomic_json
import hashlib


# This module and its builders belong to the already completed, label-free GPU
# compatibility check.  They were added after the method freeze and are not
# dependencies of the remaining 09/10 pipeline.  Keeping them in the local
# workspace is useful; transporting them inside ``src/vpi_next_evaluation``
# would make the frozen scientific source inventory appear to have changed.
LOCAL_ONLY_GPU_COMPATIBILITY = {
    'src/vpi_next_evaluation/gpu_compatibility.py',
    'scripts/build_gpu_compatibility_notebook.py',
    'scripts/build_gpu_compatibility_queue.py',
    'notebooks/input-preparation/current-dino-gpu-compatibility.ipynb',
    'notebooks/input-preparation/kernel-metadata-dino-gpu-compatibility.json',
}


def source_inventory(root):
    root = Path(root).resolve(strict=True)
    files = set()
    # The whole Python source tree includes transitive decision, scoring and I/O dependencies.
    for directory in ('src', 'scripts'):
        for path in (root / directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.css', '.html', '.yaml', '.yml', '.toml'} and not any(p.startswith('.') or p == '__pycache__' for p in path.relative_to(root).parts):
                if path.is_symlink():
                    raise ValueError('Source packages cannot contain symbolic links')
                files.add(path.relative_to(root).as_posix())
    for directory, patterns in [('notebooks/protected-evaluation', ('09*.ipynb','08f1-*.ipynb','kernel-metadata-09*.json','kernel-metadata-08f1.json')),
                                 ('notebooks/official-evaluation', ('10*.ipynb','kernel-metadata-10*.json')),
                                 ('notebooks/input-preparation', ('*.ipynb','kernel-metadata-*.json'))]:
        for pattern in patterns:
            files.update(p.relative_to(root).as_posix() for p in (root / directory).glob(pattern))
    files.add('requirements/current-evaluation-runtime.json')
    files.update(p.relative_to(root).as_posix() for p in (root/'requirements/profiles').glob('*') if p.suffix in {'.json','.in','.txt','.md'})
    files -= LOCAL_ONLY_GPU_COMPATIBILITY
    for name in files:
        if name.endswith('.ipynb'):
            notebook = json.loads((root / name).read_text())
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells'] if c['cell_type'] == 'code'):
                raise ValueError('Source transport requires clean unexecuted notebooks')
    required = {'src/vpi_next_evaluation/contracts.py', 'src/vpi_next_evaluation/original.py',
                'src/vpi_next_evaluation/official.py', 'src/vpi_object_bank/scoring.py',
                'src/visual_peer_inspector/web/decision.py', 'src/vpi_remaining_evaluation/cohort.py',
                'scripts/current_notebook_runtime.py', 'scripts/current_notebook_cells.py',
                'scripts/publish_next_stage.py'}
    if not required <= files:
        raise ValueError('Complete current source dependency tree is missing')
    return {name: digest(root / name) for name in sorted(files)}


def build_package(source_root, output_root, *, base_commit=None):
    root, target = Path(source_root).resolve(strict=True), Path(output_root).absolute()
    if target.exists():
        raise ValueError('Package output already exists; preserve it and choose a fresh destination')
    inventory = source_inventory(root)
    record = {'schema': SCHEMA, 'source_state': 'working_directory_snapshot', 'git_clean_claim': False,
              'base_commit': base_commit, 'snapshot_sha256': hashlib.sha256(canonical(inventory)).hexdigest(),
              'files': inventory, 'scope': 'Source only; prepared producers are included, but actual inputs and scientific execution are not implied'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=target.name + '.partial-', dir=target.parent))
    try:
        prepared = temporary / 'source'
        for name in inventory:
            destination = prepared / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        atomic_json(prepared / MANIFEST, record)
        verify_source(prepared)
        archive = temporary / 'current-evaluation-source.zip.bin'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in [MANIFEST, *sorted(inventory)]:
                info = zipfile.ZipInfo(name, date_time=(2020,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
                bundle.writestr(info, (prepared / name).read_bytes())
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() or set(bundle.namelist()) != {MANIFEST, *inventory}:
                raise ValueError('Source ZIP verification failed')
        receipt = {'schema': 'vpi-current-source-transport/v1', 'status': 'source_prepared',
                   'archive_sha256': digest(archive), 'snapshot_sha256': record['snapshot_sha256'],
                   'source_file_count': len(inventory), 'native_inputs_included': False,
                   'kaggle_execution_verified': False, 'linux_wheelhouse_included': False}
        atomic_json(temporary / 'package_receipt.json', receipt)
        os.replace(temporary, target)
        return receipt
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True); parser.add_argument('--output-root', required=True)
    parser.add_argument('--base-commit')
    args = parser.parse_args()
    print(json.dumps(build_package(args.source_root, args.output_root, base_commit=args.base_commit), indent=2))


if __name__ == '__main__':
    main()
