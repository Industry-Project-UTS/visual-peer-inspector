#!/usr/bin/env python3
"""Publish only sealed 09E aggregates and a strictly sanitized executed notebook."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import shutil
import sys

import publish_clearml as tracking
import publish_executed_notebook as notebooks
from vpi_object_bank.storage import atomic_json, bind, digest_file, read_json, seal, verify_seal

TABLES = frozenset({
    'metric_summary.csv', 'view_schedule_summary.csv', 'paired_asset_intervals.csv',
    'anomaly_count_summary.csv', 'majority_summary.csv', 'failure_summary.csv',
    'identity_summary.csv', 'technique_feature_matrix.csv', 'cohort_readiness.csv',
    'resource_recipe.json', 'current_method_freeze.json', 'workbench_receipt.json',
})
FIGURES = frozenset({
    'top1_tie_expected.png', 'ap_tie_expected.png', 'recall_at_2_tie_expected.png',
    'reciprocal_rank_tie_expected.png', 'technique_primary_matrix.png',
    'schedule_sensitivity.png', 'failure_decomposition.png',
})
NOTEBOOK = '09e-remaining-cohort-readiness-and-evidence-executed.ipynb'
LOG = '09e-execution.log'
RECEIPT = '09e-publication-receipt.json'
SCHEMA = 'vpi-remaining-evaluation-publication/v1'


def validate_notebook(path, figure_hashes):
    notebook = read_json(path)
    images = 0
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        if cell.get('execution_count') is None:
            raise ValueError('Published code cell was not executed')
        if notebooks._private_visual_cell(''.join(cell.get('source', []))) and cell.get('outputs'):
            raise ValueError('A private notebook cell still has outputs')
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                raise ValueError('Published notebook contains an error')
            for mime, value in output.get('data', {}).items():
                if mime in notebooks.VISUAL_MIME_TYPES:
                    if mime != 'image/png':
                        raise ValueError('Only verified aggregate PNGs can be embedded')
                    encoded = ''.join(value) if isinstance(value, list) else value
                    if hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest() not in figure_hashes:
                        raise ValueError('Notebook embeds an image outside the aggregate report lock')
                    images += 1
            text = json.dumps({key: value for key, value in output.items() if key != 'data'})
            text += json.dumps({key: value for key, value in output.get('data', {}).items() if key.startswith('text/')})
            if notebooks.SECRET_TEXT.search(text) or notebooks.LOCAL_PATH_TEXT.search(text) or notebooks.PROTECTED_TEXT.search(text):
                raise ValueError('Notebook output retains private or credential-bearing text')
    if images != len(FIGURES):
        raise ValueError('Expected every aggregate figure, without private images')
    return images


def prepare_bundle(scientific_root, report_root, clean_path, executed_path, output_root):
    scientific_root, report_root, root = map(Path, (scientific_root, report_root, output_root))
    scientific = verify_seal(scientific_root, name='scientific_lock.json')
    report = verify_seal(report_root, name='report_lock.json')
    if 'run_binding.json' not in report['files'] or read_json(report_root / 'run_binding.json').get('scientific_lock_sha256') != digest_file(scientific_root / 'scientific_lock.json'):
        raise ValueError('Report does not belong to these sealed scientific results')
    binding = {'schema': SCHEMA, 'scientific_lock_sha256': digest_file(scientific_root / 'scientific_lock.json'),
        'report_lock_sha256': digest_file(report_root / 'report_lock.json'),
        'clean_notebook_sha256': digest_file(clean_path), 'raw_executed_notebook_sha256': digest_file(executed_path),
        'publisher_sha256': digest_file(__file__), 'sanitizer_sha256': digest_file(notebooks.__file__)}
    bind(root, binding)
    if (root / 'publication_lock.json').is_file():
        verify_seal(root, name='publication_lock.json')
        return read_json(root / RECEIPT)
    table_root, figure_root = scientific_root / 'shareable', report_root / 'shareable/figures'
    if {p.name for p in table_root.iterdir()} != TABLES or {p.name for p in figure_root.iterdir()} != FIGURES:
        raise ValueError('09E publication requires the exact aggregate allowlist')
    for source_root, directory, names, lock in (
            (scientific_root, table_root, TABLES, scientific), (report_root, figure_root, FIGURES, report)):
        for name in sorted(names):
            source = directory / name
            if source.is_symlink() or lock['files'].get(source.relative_to(source_root).as_posix()) != digest_file(source):
                raise ValueError('Aggregate artifact is outside its verified lock')
            target = root / ('figures' if name in FIGURES else 'tables') / name
            if name in TABLES:
                tracking.validate_text_artifact(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    # Reuse the established sanitizer while dropping the whole private cell output,
    # including anonymized per-object score tables. No old publisher source changes.
    original = notebooks._sanitize_output
    def aggregate_only(output, *, remove_visuals):
        if remove_visuals:
            if output.get('output_type') == 'error':
                raise notebooks.PublicationError('Private cell execution failed')
            removed = sum(mime in notebooks.VISUAL_MIME_TYPES for mime in output.get('data', {}))
            return None, removed, 0
        return original(output, remove_visuals=False)
    try:
        notebooks._sanitize_output = aggregate_only
        receipt = notebooks.publish_notebook(clean_path=Path(clean_path), executed_path=Path(executed_path),
            output_path=root / NOTEBOOK, log_path=root / LOG, label='09E existing development100 and remaining-cohort readiness')
    finally:
        notebooks._sanitize_output = original
    figures = {digest_file(root / 'figures' / name) for name in FIGURES}
    validate_notebook(root / NOTEBOOK, figures)
    receipt.update(schema=SCHEMA, private_cell_outputs_removed=True, protected_scenes_evaluated=0,
        scientific_lock_sha256=binding['scientific_lock_sha256'], report_lock_sha256=binding['report_lock_sha256'],
        aggregate_table_count=len(TABLES), aggregate_figure_count=len(FIGURES),
        scope='100 development scenes / 27 assets; original 1000/600/400 roles preserved; protected stages blocked')
    atomic_json(root / RECEIPT, receipt)
    seal(root, [p for p in root.rglob('*') if p.is_file()], name='publication_lock.json', metadata={'schema': SCHEMA})
    verify_seal(root, name='publication_lock.json')
    return receipt


def collect_artifacts(scientific_root, shareable_root, lock_name, report_root=None):
    """The tracking worker receives a self-contained aggregate publication bundle."""
    root = Path(scientific_root)
    lock, lock_hash = tracking.verified_lock(root, 'publication_lock.json')
    expected = {'run_binding.json', NOTEBOOK, LOG, RECEIPT} | {'tables/' + n for n in TABLES} | {'figures/' + n for n in FIGURES}
    if set(lock['files']) != expected:
        raise ValueError('Publication bundle contains missing or unapproved artifacts')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected | {'publication_lock.json'}:
        raise ValueError('Publication bundle contains unsealed files')
    for name in TABLES:
        tracking.validate_text_artifact(root / 'tables' / name)
    figures = {digest_file(root / 'figures' / name) for name in FIGURES}
    validate_notebook(root / NOTEBOOK, figures)
    tracking.validate_text_artifact(root / LOG)
    tracking.validate_text_artifact(root / RECEIPT)
    files = {name: root / name for name in sorted(expected - {'run_binding.json'})}
    manifest = {'locks': {'09e_publication': lock_hash}, 'artifacts': {name: digest_file(path) for name, path in files.items()}}
    return files, manifest


def publish_bundle(root, receipt_path, *, task_api=None):
    original = tracking.collect_artifacts
    try:
        tracking.collect_artifacts = collect_artifacts
        return tracking.publish(root, root, receipt_path, lock_name='publication_lock.json',
            task_name='09E remaining-cohort readiness and development evidence', task_api=task_api)
    finally:
        tracking.collect_artifacts = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'publish'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--clean', type=Path)
    parser.add_argument('--executed', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    if args.action == 'prepare':
        cfg = read_json(args.config)
        receipt = prepare_bundle(cfg['scientific_root'], cfg['report_root'], args.clean, args.executed, args.bundle)
    elif args.bounded:
        if args.receipt is None:
            parser.error('--receipt is required for tracking')
        command = [sys.executable, '-B', str(Path(__file__).absolute()), 'publish', '--bundle', str(args.bundle), '--receipt', str(args.receipt)]
        receipt = tracking.bounded(command, args.receipt, args.timeout)
    else:
        if args.receipt is None:
            parser.error('--receipt is required for tracking')
        try:
            with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
                receipt = publish_bundle(args.bundle, args.receipt)
        except BaseException as error:
            receipt = tracking.read_receipt(args.receipt)
            receipt.update(status='pending', error_type=type(error).__name__)
            tracking.atomic_json(args.receipt, receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'task_id', 'task_url', 'error_type', 'retained_safe_visuals', 'removed_private_visuals') if key in receipt}))


if __name__ == '__main__':
    main()
