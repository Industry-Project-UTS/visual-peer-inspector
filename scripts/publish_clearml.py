#!/usr/bin/env python3
"""Idempotent aggregate-only ClearML publication after scientific sealing."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'Visual Peer Inspector/Development Scoring'
TABLES = frozenset({
    'metric_summary.csv', 'view_schedule_summary.csv', 'paired_asset_intervals.csv',
    'anomaly_count_summary.csv', 'majority_summary.csv', 'failure_summary.csv',
    'identity_summary.csv', 'decision.json', 'evaluation_receipt.json',
})
FIGURES = frozenset({'view_count_accuracy.png', 'paired_improvement.png', 'failure_waterfall.png'})
PRIVATE_HEADERS = frozenset({'scene', 'scene_id', 'scene_ids', 'opaque_scene_id', 'source_scene_id',
                             'object_id', 'candidate_id', 'base_asset', 'asset_group', 'asset_group_id',
                             'api_key', 'api_secret_key', 'api_access_key', 'password', 'token'})
PRIVATE_TEXT = re.compile(
    r'\bS\d{6}\b|\bO\d{2}\b|/(?:Users|home|kaggle)/|file://|'
    r'\b(?:opaque_scene_id|source_scene_id|labels_by_object|mask_rle|rgb_path)\b|'
    r'\bhf_[A-Za-z0-9]{20,}\b|\bKGAT_[A-Za-z0-9_-]{20,}\b|'
    r'(?i:(?:api[_-]?(?:secret|access)?[_-]?key|password|token)[\"\']?\s*[:=]\s*[\"\'][^\"\'${}\s]{8,}[\"\'])'
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_receipt(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def verified_lock(root, name):
    root = Path(root).resolve()
    if Path(name).name != name:
        raise ValueError('The scientific lock must be directly inside its root')
    path = root / name
    if not path.is_file():
        raise ValueError('Scientific outputs must be sealed before publication')
    lock = json.loads(path.read_text())
    if not lock.get('files'):
        raise ValueError('An empty scientific lock cannot authorize publication')
    for relative, expected in lock['files'].items():
        candidate = root / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError('Unsafe scientific artifact path')
        if not candidate.is_file() or digest(candidate) != expected:
            raise ValueError('Scientific artifact verification failed')
    return lock, digest(path)


def validate_text_artifact(path):
    text = path.read_text(encoding='utf-8')
    if PRIVATE_TEXT.search(text):
        raise ValueError('Aggregate evidence contains private or credential-bearing text')
    if path.suffix == '.csv':
        headers = next(csv.reader(io.StringIO(text)), [])
        if any(value.lower() in PRIVATE_HEADERS for value in headers):
            raise ValueError('Per-scene or per-object evidence cannot be published')
    elif path.suffix == '.json':
        def check(value):
            if isinstance(value, dict):
                if any(str(key).lower() in PRIVATE_HEADERS for key in value):
                    raise ValueError('Per-scene or per-object evidence cannot be published')
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(json.loads(text))


def collect_artifacts(scientific_root, shareable_root, lock_name, report_root=None):
    scientific_root = Path(scientific_root).resolve()
    shareable_root = Path(shareable_root).resolve()
    if not shareable_root.is_relative_to(scientific_root):
        raise ValueError('Shareable tables must belong to the verified scientific root')
    lock, lock_hash = verified_lock(scientific_root, lock_name)
    files = {}
    present = list(shareable_root.iterdir())
    if any(path.is_symlink() or not path.is_file() or path.name not in TABLES for path in present):
        raise ValueError('The aggregate directory contains an unapproved file or directory')
    if not present:
        raise ValueError('No aggregate artifacts are available')
    for path in sorted(present):
        relative = path.relative_to(scientific_root).as_posix()
        if lock['files'].get(relative) != digest(path):
            raise ValueError('An aggregate artifact is not covered by the scientific lock')
        validate_text_artifact(path)
        files['evaluation/' + path.name] = path
    lock_hashes = {'evaluation': lock_hash}
    if report_root is not None:
        report_root = Path(report_root).resolve()
        report_lock, report_hash = verified_lock(report_root, 'report_lock.json')
        figure_root = report_root / 'shareable/figures'
        if figure_root.is_dir():
            for path in sorted(figure_root.iterdir()):
                if path.is_symlink() or not path.is_file() or path.name not in FIGURES:
                    raise ValueError('The figure directory contains an unapproved artifact')
                if report_lock['files'].get(path.relative_to(report_root).as_posix()) != digest(path):
                    raise ValueError('An aggregate figure is outside the report lock')
                from PIL import Image
                with Image.open(path) as image:
                    if image.format != 'PNG':
                        raise ValueError('Aggregate figures must be PNG files')
                    image.verify()
                files['report/' + path.name] = path
        lock_hashes['report'] = report_hash
    manifest = {'locks': lock_hashes, 'artifacts': {name: digest(path) for name, path in files.items()}}
    return files, manifest


def publish(scientific_root, shareable_root, receipt_path, *, lock_name='evaluation_lock.json',
            report_root=None, task_name='08O multi-view object-bank evaluation', task_api=None):
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE_TEXT.search(task_name):
        raise ValueError('Task name must not contain sensitive text')
    # The operating system releases this lock if the bounded worker is terminated.
    with receipt_path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files, manifest = collect_artifacts(scientific_root, shareable_root, lock_name, report_root)
        binding = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        receipt = read_receipt(receipt_path)
        if receipt.get('binding_sha256') not in (None, binding):
            raise ValueError('Publication receipt belongs to different scientific evidence')
        if receipt.get('status') == 'completed':
            return receipt
        remote_name = task_name + ' [' + binding[:16] + ']'
        receipt.update(status='pending', binding_sha256=binding, task_name=remote_name,
                       artifact_hashes=manifest['artifacts'], scientific_lock_hashes=manifest['locks'])
        atomic_json(receipt_path, receipt)
        try:
            if task_api is None:
                from clearml import Task
                task_api = Task
            if receipt.get('task_id'):
                task = task_api.get_task(task_id=receipt['task_id'])
            else:
                # Recover the narrow create/receipt crash window through a deterministic remote name.
                matches = task_api.get_tasks(project_name=PROJECT, task_filter={'name': '^' + re.escape(remote_name) + '$'})
                matches = [task for task in matches if task.name == remote_name]
                if len(matches) > 1:
                    raise ValueError('More than one remote task matches this publication')
                task = matches[0] if matches else task_api.create(
                    project_name=PROJECT, task_name=remote_name, task_type='data_processing',
                    packages=False, detect_repository=False, add_task_init_call=False,
                )
                receipt['task_id'] = task.id
                atomic_json(receipt_path, receipt)
            task_url = task.get_output_log_web_page()
            parsed = urlsplit(task_url)
            if parsed.scheme in ('https', 'http') and parsed.netloc and not (parsed.username or parsed.password or parsed.query):
                receipt['task_url'] = task_url
            task.mark_started(force=True)
            receipt.update(status='uploading')
            atomic_json(receipt_path, receipt)
            for name, path in files.items():
                artifact_name = name.replace('/', '__').replace('.', '_')
                if task.upload_artifact(artifact_name, artifact_object=str(path), wait_on_upload=True,
                                        auto_pickle=False, retries=0) is not True:
                    raise RuntimeError('An aggregate upload was not confirmed')
                receipt.setdefault('uploaded', {})[name] = manifest['artifacts'][name]
                atomic_json(receipt_path, receipt)
            # Re-read remote artifact names instead of equating initiation with completed upload.
            remote = task_api.get_task(task_id=task.id)
            expected = {name.replace('/', '__').replace('.', '_') for name in files}
            if not expected.issubset(set(remote.artifacts)):
                raise RuntimeError('Remote artifact inventory is incomplete')
            task.mark_completed(ignore_errors=False)
            task.close()
            receipt.update(status='completed', remote_artifacts_verified=True)
            receipt.pop('error_type', None)
            atomic_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(status='pending', error_type=type(error).__name__)
            atomic_json(receipt_path, receipt)
            return receipt


def bounded(command, receipt_path, timeout=180):
    """Run only publication; suppress external-service output and preserve task identity."""
    if timeout <= 0:
        raise ValueError('Publication timeout must be positive')
    worker = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
    try:
        worker.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait()
        receipt = read_receipt(receipt_path)
        receipt.update(status='pending', error_type='TimeoutExpired')
        atomic_json(receipt_path, receipt)
    receipt = read_receipt(receipt_path)
    if not receipt:
        receipt = {'status': 'pending', 'error_type': 'PublicationWorkerIncomplete'}
        atomic_json(receipt_path, receipt)
    return receipt


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--scientific-root', type=Path)
    parser.add_argument('--shareable-root', type=Path)
    parser.add_argument('--report-root', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--lock-name', default='evaluation_lock.json')
    parser.add_argument('--task-name', default='08O multi-view object-bank evaluation')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args(argv)
    configuration = json.loads(args.config.read_text()) if args.config else {}
    options = configuration.get('publication', {})
    args.scientific_root = args.scientific_root or Path(options.get('scientific_root', configuration.get('evaluation_root', ROOT / '.local/evaluation')))
    args.shareable_root = args.shareable_root or Path(options.get('shareable_root', args.scientific_root / 'shareable'))
    args.report_root = args.report_root or (Path(options['report_root']) if options.get('report_root') else None)
    args.receipt = args.receipt or Path(options.get('receipt', ROOT / '.local/tracking/08o-publication.json'))
    args.lock_name = options.get('lock_name', args.lock_name)
    args.task_name = options.get('task_name', args.task_name)
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        if args.bounded:
            command = [sys.executable, str(Path(__file__).resolve()), '--scientific-root', str(args.scientific_root),
                       '--shareable-root', str(args.shareable_root), '--receipt', str(args.receipt),
                       '--lock-name', args.lock_name, '--task-name', args.task_name]
            if args.report_root:
                command += ['--report-root', str(args.report_root)]
            receipt = bounded(command, args.receipt, args.timeout)
        else:
            # SDK output can contain local paths or provider diagnostics. Receipts keep only error types.
            with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
                receipt = publish(args.scientific_root, args.shareable_root, args.receipt,
                                  lock_name=args.lock_name, report_root=args.report_root, task_name=args.task_name)
    except BaseException as error:
        receipt = read_receipt(args.receipt)
        receipt.update(status='pending', error_type=type(error).__name__)
        atomic_json(args.receipt, receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'task_id', 'error_type') if key in receipt}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
