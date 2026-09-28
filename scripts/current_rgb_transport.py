"""Retain a completed large RGB release as one verified model-only archive.

The scientific RGB package is built and sealed by native_rgb first. This
adapter changes only how its bytes cross Kaggle notebook sessions. The archive
is restored in /tmp before a model-only consumer verifies the original package.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import uuid


SCHEMA = 'vpi-current-rgb-transport/v1'
ARCHIVE = 'model-input.tar'
MANIFEST = 'rgb_transport.json'
MAX_FILES = 50000
MAX_BYTES = 10 * 1024**3


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def _member(name):
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or '\\' in name or '\x00' in name
            or any(part in {'', '.', '..'} for part in name.split('/'))
            or value.parts[0] != 'model-input'):
        raise ValueError('RGB transport member is unsafe')
    return value


def _release(root):
    root = Path(root).resolve(strict=True)
    receipt = _json(root / 'release_receipt.json')
    lock = _json(root / 'release_lock.json')
    files = lock.get('files')
    if (receipt.get('status') != 'rgb_complete' or receipt.get('remaining_scenes') != 0
            or receipt.get('completed_scenes') != receipt.get('scene_count')
            or not isinstance(files, dict) or not files
            or any(name.startswith('checkpoints/') for name in files)):
        raise ValueError('Only a complete release without sealed checkpoints may be transported')
    required = {'run_binding.json', 'release_receipt.json',
                'model-input/package_record.json', 'model-input/content_checksums.sha256',
                'model-input/neutral_corpus_manifest.jsonl'}
    if not required <= set(files):
        raise ValueError('Sealed RGB release omits canonical model input')
    marker = root / 'model-input/package_record.json'
    if (_digest(marker) != receipt.get('model_package_record_sha256')
            or _digest(marker) != files['model-input/package_record.json']):
        raise ValueError('RGB model package marker differs from release seal')
    return root, receipt, files


def _transport_record(root):
    record = _json(root / MANIFEST)
    archive = root / ARCHIVE
    if (record.get('schema') != SCHEMA or record.get('status') != 'complete'
            or not isinstance(record.get('file_count'), int)
            or not 3 <= record['file_count'] <= MAX_FILES
            or not isinstance(record.get('total_member_bytes'), int)
            or not 0 < record['total_member_bytes'] <= MAX_BYTES
            or record.get('archive_sha256') != _digest(archive)
            or record.get('archive_bytes') != archive.stat().st_size
            or record.get('package_record_sha256') != _digest(root / 'model-input/package_record.json')
            or record.get('release_lock_sha256') != _digest(root / 'release_lock.json')):
        raise ValueError('RGB transport archive or receipt differs')
    return record


def _remove_duplicates(root):
    # The verified archive and package marker survive; only redundant loose
    # media and unsealed per-scene checkpoints are removed.
    for relative in ('checkpoints', 'model-input/media'):
        target = root / relative
        if target.is_symlink():
            raise ValueError('RGB transport cleanup refuses symlinks')
        if target.exists():
            if not target.is_dir():
                raise ValueError('RGB transport cleanup target is not a directory')
            shutil.rmtree(target)


def compact_completed_rgb(root, result, *, role):
    """Archive a complete unseen release and remove duplicated loose media."""
    if role != 'unseen':
        raise ValueError('The compact transport is scoped to official unseen RGB')
    root, receipt, sealed = _release(root)
    if result != receipt:
        raise ValueError('Notebook result differs from sealed RGB release receipt')
    if (root / MANIFEST).exists():
        record = _transport_record(root)
        _remove_duplicates(root)
        return record
    model = root / 'model-input'
    paths = sorted(path for path in model.rglob('*') if path.is_file())
    if not 3 <= len(paths) <= MAX_FILES or any(path.is_symlink() for path in paths):
        raise ValueError('RGB transport has an invalid model file inventory')
    names = {path.relative_to(root).as_posix() for path in paths}
    if names != {name for name in sealed if name.startswith('model-input/')}:
        raise ValueError('RGB transport does not match the sealed model file inventory')
    total = sum(path.stat().st_size for path in paths)
    if not 0 < total <= MAX_BYTES:
        raise ValueError('RGB transport exceeds its declared byte bound')
    # Checkpoints are outside release_lock and duplicate the complete model
    # package. Removing them first avoids exceeding Kaggle's working disk.
    checkpoints = root / 'checkpoints'
    if checkpoints.is_symlink():
        raise ValueError('RGB checkpoint directory is a symlink')
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    temporary = root / (ARCHIVE + '.partial-' + uuid.uuid4().hex)
    archive = root / ARCHIVE
    try:
        if not archive.exists():
            with tarfile.open(temporary, mode='w') as bundle:
                for path in paths:
                    relative = path.relative_to(root).as_posix()
                    _member(relative)
                    bundle.add(path, arcname=relative, recursive=False)
            os.replace(temporary, archive)
        # An interrupted run may have finished the archive but not its receipt.
        # Loose model files remain until both are verified, so this resumes safely.
        with tarfile.open(archive, mode='r:') as bundle:
            members = bundle.getmembers()
        if (len(members) != len(paths) or sum(member.size for member in members) != total
                or {member.name for member in members} != names
                or any(not member.isfile() for member in members)):
            raise ValueError('RGB transport archive inventory differs after writing')
        archive_hash = _digest(archive)
        record = {'schema': SCHEMA, 'status': 'complete', 'role': role,
                  'scene_count': receipt['scene_count'], 'file_count': len(paths),
                  'total_member_bytes': total, 'archive_bytes': archive.stat().st_size,
                  'archive_sha256': archive_hash,
                  'package_record_sha256': receipt['model_package_record_sha256'],
                  'release_lock_sha256': _digest(root / 'release_lock.json')}
        _atomic_json(root / MANIFEST, record)
        _transport_record(root)
        _remove_duplicates(root)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def reuse_completed_transport(root):
    """Recover a rerun interrupted after archive creation and before completion."""
    root = Path(root)
    if not (root / MANIFEST).is_file():
        return None
    record = _transport_record(root)
    receipt = _json(root / 'release_receipt.json')
    if receipt.get('status') != 'rgb_complete' or receipt.get('scene_count') != record.get('scene_count'):
        raise ValueError('Retained RGB transport differs from scientific receipt')
    _remove_duplicates(root)
    return receipt


def restore_rgb_transport(model_root, *, cache_base=None):
    """Restore model-only bytes outside Kaggle's retained output directory."""
    model_root = Path(model_root).resolve(strict=True)
    root = model_root.parent
    record = _transport_record(root)
    cache_base = Path(tempfile.gettempdir() if cache_base is None else cache_base).resolve(strict=True)
    destination = cache_base / ('vpi-unseen-rgb-' + record['archive_sha256'][:20])
    ready = destination / 'transport_restored.json'
    if ready.is_file():
        if _json(ready) != record or _digest(destination / 'model-input/package_record.json') != record['package_record_sha256']:
            raise ValueError('Existing RGB transport restoration differs')
        return destination / 'model-input'
    if destination.exists():
        raise ValueError('Incomplete RGB restoration exists; preserve it for inspection')
    temporary = cache_base / (destination.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        count = total = 0
        names = set()
        with tarfile.open(root / ARCHIVE, mode='r:') as bundle:
            for member in bundle:
                relative = _member(member.name)
                if not member.isfile() or member.name in names:
                    raise ValueError('RGB transport contains a duplicate or non-file member')
                names.add(member.name)
                count += 1
                total += member.size
                if count > MAX_FILES or total > MAX_BYTES:
                    raise ValueError('RGB transport extraction exceeds bounds')
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError('RGB transport member is unreadable')
                with target.open('xb') as output:
                    shutil.copyfileobj(source, output, length=4 * 1024**2)
                if target.stat().st_size != member.size:
                    raise ValueError('RGB transport member size changed')
        if count != record['file_count'] or total != record['total_member_bytes']:
            raise ValueError('RGB transport extraction inventory differs')
        if _digest(temporary / 'model-input/package_record.json') != record['package_record_sha256']:
            raise ValueError('Restored RGB package marker differs')
        _atomic_json(temporary / 'transport_restored.json', record)
        os.replace(temporary, destination)
        return destination / 'model-input'
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
