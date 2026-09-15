#!/usr/bin/env python3
"""Execute unchanged notebook sources with this interpreter and save each cell."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def save_notebook(path, notebook):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    nbformat.write(notebook, temporary)
    temporary.replace(path)


def signatures(notebook):
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def execute(notebook_path, config_path, output_path, *, cell_timeout=21600):
    notebook_path, config_path, output_path = map(lambda p: Path(p).resolve(), (notebook_path, config_path, output_path))
    if output_path == notebook_path:
        raise ValueError('Raw executed evidence must not overwrite the clean notebook')
    if not config_path.is_file():
        raise ValueError('The configuration file does not exist')
    notebook = nbformat.read(notebook_path, as_version=4)
    original = signatures(notebook)
    clean_sha = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    receipt_path = output_path.with_suffix('.execution.json')
    receipt = {'status': 'running', 'clean_notebook_sha256': clean_sha,
               'configuration_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    atomic_json(receipt_path, receipt)
    save_notebook(output_path, notebook)
    try:
        with tempfile.TemporaryDirectory(prefix='vpi-object-bank-kernel-') as temporary:
            kernels = Path(temporary)
            kernel_dir = kernels / 'vpi-object-bank'
            kernel_dir.mkdir()
            atomic_json(kernel_dir / 'kernel.json', {
                'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Visual Peer Inspector local runtime', 'language': 'python',
            })
            manager = AsyncKernelManager(kernel_name='vpi-object-bank',
                                         kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            client = NotebookClient(notebook, km=manager, timeout=cell_timeout, allow_errors=False,
                                    resources={'metadata': {'path': str(ROOT)}})
            def completed(**kwargs):
                save_notebook(output_path, notebook)
            client.on_cell_executed = completed
            environment = os.environ.copy()
            environment.update({'VPI_OBJECT_BANK_ROOT': str(ROOT), 'VPI_OBJECT_BANK_CONFIG': str(config_path),
                                'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'})
            try:
                client.execute(cwd=str(ROOT), env=environment, cleanup_kc=True)
            finally:
                # Also release partial startup state when a local socket cannot open.
                if client.km is not None:
                    client._cleanup_kernel()
        if signatures(notebook) != original:
            raise ValueError('Notebook source changed during execution')
        receipt['status'] = 'completed'
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        save_notebook(output_path, notebook)
        receipt['executed_notebook_sha256'] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        receipt['source_cells_unchanged'] = signatures(notebook) == original
        atomic_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cell-timeout', type=int, default=21600)
    args = parser.parse_args()
    output = args.output or ROOT / '.local/notebooks' / (args.notebook.stem + '-executed.ipynb')
    try:
        receipt = execute(args.notebook, args.config, output, cell_timeout=args.cell_timeout)
        print(json.dumps({'event': 'notebook_execution', 'status': receipt['status'], 'source_cells_unchanged': True}))
    except BaseException as error:
        print(json.dumps({'event': 'notebook_execution', 'status': 'failed', 'error_type': type(error).__name__}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
