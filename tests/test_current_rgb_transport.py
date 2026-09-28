"""Transport checks for the completed official unseen RGB package."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from current_rgb_transport import (ARCHIVE, MANIFEST, compact_completed_rgb,
                                   restore_rgb_transport, reuse_completed_transport)
from current_launch import resolve_resource


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def release(tmp_path):
    root = tmp_path / 'unseen-rgb'
    model = root / 'model-input'
    media = model / 'media/S1/rgb/V00.png'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'fixed-rgb-bytes')
    (model / 'neutral_corpus_manifest.jsonl').write_text('{"opaque_scene_id":"S1"}\n')
    (model / 'content_checksums.sha256').write_text(digest(media) + '  media/S1/rgb/V00.png\n')
    write_json(model / 'package_record.json', {'scene_count': 1})
    receipt = {'status': 'rgb_complete', 'scene_count': 1, 'completed_scenes': 1,
               'remaining_scenes': 0, 'model_package_record_sha256': digest(model / 'package_record.json')}
    write_json(root / 'run_binding.json', {'schema': 'test'})
    write_json(root / 'release_receipt.json', receipt)
    checkpoint = root / 'checkpoints/S1/scene.json'
    write_json(checkpoint, {'duplicate': True})
    sealed = [root / 'run_binding.json', root / 'release_receipt.json',
              *sorted(path for path in model.rglob('*') if path.is_file())]
    write_json(root / 'release_lock.json', {'files': {path.relative_to(root).as_posix(): digest(path) for path in sealed}})
    return root, receipt


def test_compact_restore_and_resume(tmp_path):
    root, receipt = release(tmp_path)
    record = compact_completed_rgb(root, receipt, role='unseen')
    assert record['file_count'] == 4
    assert (root / ARCHIVE).is_file()
    assert not (root / 'checkpoints').exists()
    assert not (root / 'model-input/media').exists()
    assert (root / 'model-input/package_record.json').is_file()
    assert reuse_completed_transport(root) == receipt
    assert compact_completed_rgb(root, receipt, role='unseen') == record
    restored = restore_rgb_transport(root / 'model-input', cache_base=tmp_path)
    assert (restored / 'media/S1/rgb/V00.png').read_bytes() == b'fixed-rgb-bytes'
    assert restore_rgb_transport(root / 'model-input', cache_base=tmp_path) == restored


def test_archive_finished_before_receipt_is_resumable(tmp_path):
    root, receipt = release(tmp_path)
    compact_completed_rgb(root, receipt, role='unseen')
    restored = restore_rgb_transport(root / 'model-input', cache_base=tmp_path)
    media = root / 'model-input/media/S1/rgb/V00.png'
    media.parent.mkdir(parents=True)
    media.write_bytes((restored / 'media/S1/rgb/V00.png').read_bytes())
    (root / MANIFEST).unlink()
    record = compact_completed_rgb(root, receipt, role='unseen')
    assert record['status'] == 'complete'
    assert not media.exists()


def test_corrupt_archive_fails_closed(tmp_path):
    root, receipt = release(tmp_path)
    compact_completed_rgb(root, receipt, role='unseen')
    with (root / ARCHIVE).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='archive or receipt differs'):
        restore_rgb_transport(root / 'model-input', cache_base=tmp_path)
    with pytest.raises(ValueError, match='archive or receipt differs'):
        reuse_completed_transport(root)


def test_unsafe_archive_member_is_rejected(tmp_path):
    root, receipt = release(tmp_path)
    record = compact_completed_rgb(root, receipt, role='unseen')
    archive = root / ARCHIVE
    with tarfile.open(archive, 'w') as bundle:
        data = b'unsafe'
        info = tarfile.TarInfo('../outside')
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    record.update(archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                  file_count=3, total_member_bytes=6)
    write_json(root / MANIFEST, record)
    with pytest.raises(ValueError):
        restore_rgb_transport(root / 'model-input', cache_base=tmp_path)
    assert not (tmp_path / 'outside').exists()


def test_incomplete_release_is_not_compacted(tmp_path):
    root, receipt = release(tmp_path)
    receipt['remaining_scenes'] = 1
    write_json(root / 'release_receipt.json', receipt)
    with pytest.raises(ValueError, match='Only a complete release'):
        compact_completed_rgb(root, receipt, role='unseen')
    assert (root / 'checkpoints').exists()


def test_model_only_resource_resolver_restores_transport(tmp_path, monkeypatch):
    root, receipt = release(tmp_path)
    compact_completed_rgb(root, receipt, role='unseen')
    monkeypatch.setattr('current_rgb_transport.tempfile.gettempdir', lambda: str(tmp_path))
    spec = {'ref': 'esreekarreddy/vpi-current-unseen-rgb',
            'marker': 'package_record.json', 'suffixes': ['']}
    found = resolve_resource(spec, overrides={'unseen-rgb': str(root / 'model-input')},
                             name='unseen-rgb')
    assert found != root / 'model-input'
    assert (found / 'media/S1/rgb/V00.png').read_bytes() == b'fixed-rgb-bytes'
