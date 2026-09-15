"""Private session-capability cache; tensor files never use pickle."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import time
import uuid
import numpy as np
from .core import InspectionError, TOKEN


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class SessionCache:
    def __init__(self, root, *, ttl_seconds=1800):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ttl = ttl_seconds

    def path(self, session_id):
        if not isinstance(session_id, str) or not TOKEN.fullmatch(session_id):
            raise InspectionError('Invalid inspection session.')
        path = self.root / hashlib.sha256(session_id.encode()).hexdigest()
        if path.is_symlink():
            raise InspectionError('The session cache is invalid. Reset this session.')
        return path

    def _read(self, path):
        try:
            record = json.loads((path / 'session.json').read_text())
            if digest(record['payload']) != record['sha256']:
                raise ValueError('checksum')
            return record['payload']
        except (OSError, ValueError, KeyError) as error:
            raise InspectionError('The candidate cache is corrupt. Reset this session; inference was not restarted.') from error

    def open(self, session_id, token, binding, *, allow_create=False):
        path = self.path(session_id)
        if token is None or (allow_create and not path.exists()):
            if path.exists():
                raise InspectionError('This session already has a cache. Use its current result or reset it.')
            token = token or secrets.token_hex(32)
            path.mkdir(mode=0o700)
            state = {'token_hash': digest(token), 'binding': binding, 'expires_at': time.time()+self.ttl,
                     'views': {}, 'entries': {}}
            self.save(path, state)
        else:
            if not path.is_dir():
                raise InspectionError('The candidate cache has expired. Reset and inspect the images again.')
            state = self._read(path)
            if not isinstance(token, str) or not hmac.compare_digest(state['token_hash'], digest(token)):
                raise InspectionError('This candidate cache belongs to another session.')
            if state['expires_at'] < time.time():
                shutil.rmtree(path)
                raise InspectionError('The candidate cache has expired. Reset and inspect the images again.')
            if state['binding'] != binding:
                raise InspectionError('The candidate cache uses different model or source settings. Reset this session.')
        return path, state, token

    def save(self, path, state):
        atomic_json(path / 'session.json', {'payload': state, 'sha256': digest(state)})

    def load(self, path, state, key):
        if key not in state['entries']:
            return None
        entry = state['entries'][key]
        file = path / (key + '.npz')
        try:
            if file.is_symlink() or file_hash(file) != entry['sha256']:
                raise ValueError('checksum')
            with np.load(file, allow_pickle=False) as values:
                if set(values.files) != {'mask', 'patches', 'embedding'}:
                    raise ValueError('schema')
                arrays = {name: values[name].copy() for name in values.files}
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise ValueError('nonfinite')
            if (arrays['mask'].ndim != 2 or not np.isin(arrays['mask'], [0,1]).all()
                    or arrays['patches'].ndim != 2 or arrays['embedding'].ndim != 1
                    or (len(arrays['patches']) and (len(arrays['patches']) != 256 or arrays['patches'].shape[1] != len(arrays['embedding'])))):
                raise ValueError('tensor schema')
            return {**entry['metadata'], **arrays}
        except (OSError, ValueError, KeyError) as error:
            raise InspectionError('A cached candidate is corrupt. Reset this session; inference was not restarted.') from error

    def put(self, path, state, key, metadata, arrays):
        if key in state['entries']:
            self.load(path, state, key)
            return
        temporary = path / (key + '.tmp-' + uuid.uuid4().hex)
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        target = path / (key + '.npz')
        temporary.replace(target)
        state['entries'][key] = {'sha256': file_hash(target), 'metadata': metadata}
        self.save(path, state)

    def clear(self, session_id, token=None):
        path = self.path(session_id)
        if not path.exists():
            return
        if token is not None:
            state = self._read(path)
            if not hmac.compare_digest(state['token_hash'], digest(token)):
                raise InspectionError('This candidate cache belongs to another session.')
        shutil.rmtree(path)

    def cleanup_expired(self):
        removed = set()
        for path in self.root.iterdir():
            if path.is_dir() and not path.is_symlink():
                try:
                    if self._read(path)['expires_at'] < time.time():
                        shutil.rmtree(path)
                        removed.add(path.name)
                except InspectionError:
                    # Corruption must remain a visible error for its owner.
                    continue
        return removed
