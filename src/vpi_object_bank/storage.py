"""File-backed atomic checkpoints; no service or database is required."""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import os
import shutil
import time
import uuid

from visual_peer_inspector.workflows.comparison_io import (
    atomic_json, canonical, digest_file, read_json, seal, verify_seal,
)


def digest(value):
    return sha256(canonical(value)).hexdigest()


def bind(root, binding):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'run_binding.json'
    if path.exists():
        if read_json(path) != binding:
            raise ValueError('Output belongs to different inputs, configuration, code or runtime')
    else:
        if any(root.iterdir()):
            raise ValueError('Nonempty output has no binding; preserve it and choose a fresh directory')
        atomic_json(path, binding)
    return digest(binding)


def verified_unit(path, binding_sha256):
    path = Path(path)
    if not path.exists():
        return False
    lock = verify_seal(path)
    if lock.get('metadata', {}).get('binding_sha256') != binding_sha256:
        raise ValueError('Checkpoint binding does not match this run')
    expected = set(lock['files']) | {'artifact_lock.json'}
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Checkpoint has unexpected or missing files')
    return True


@contextmanager
def atomic_unit(path, binding_sha256):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite a completed checkpoint')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    if temporary.exists():
        temporary.rename(path.with_name(path.name + '.interrupted-' + uuid.uuid4().hex[:8]))
    temporary.mkdir()
    try:
        yield temporary
        files = [p for p in temporary.rglob('*') if p.is_file()]
        seal(temporary, files, metadata={'binding_sha256': binding_sha256})
        verify_seal(temporary)
        temporary.rename(path)
    except BaseException:
        # The incomplete unit remains available for investigation; no completed unit changes.
        raise


@contextmanager
def terminal_status(root, stage):
    root = Path(root)
    start = time.monotonic()
    atomic_json(root / 'run_status.json', {'stage': stage, 'status': 'running'})
    try:
        yield
    except BaseException as error:
        record = {'stage': stage, 'status': 'failed', 'error_type': type(error).__name__,
                  'elapsed_seconds': time.monotonic() - start}
        # Avoid exception text here: external libraries can include credential-bearing URLs.
        atomic_json(root / 'run_status.json', record)
        failures = root / 'failures'
        atomic_json(failures / (uuid.uuid4().hex + '.json'), record)
        raise
    else:
        atomic_json(root / 'run_status.json', {'stage': stage, 'status': 'completed',
                                              'elapsed_seconds': time.monotonic() - start})


def source_hashes(root, relatives):
    root = Path(root)
    return {p: digest_file(root / p) for p in relatives}


def free_space(root, required_bytes):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < required_bytes:
        raise OSError('Insufficient free space for reusable feature checkpoints')
    return free


def progress(root, **fields):
    atomic_json(Path(root) / 'progress.json', fields)
    print(canonical({'event': 'progress', **fields}).decode(), flush=True)
