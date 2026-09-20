"""Only synthetic files are compacted; no existing outputs are removed."""
from pathlib import Path
import copy
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import post_completion_retention as retention
from current_launch import complete_job
from vpi_object_bank.storage import atomic_json, digest_file, seal, read_json


@pytest.fixture
def fixture(tmp_path):
    job = 'fixture-cache-00'
    plan = {'jobs': [{'id': job, 'compact_checkpoints': True, 'config': {'action': 'prepare_shard'}}],
            'resources': {'cache': {'producer_job': job, 'suffixes': [job], 'marker': 'cache_shard_lock.json'}}}
    cache, features = tmp_path / job, tmp_path / f'{job}-features'
    cfg = {'output_root': str(cache), 'feature_root': str(features)}
    atomic_json(tmp_path / 'launch-config' / f'{job}.json', cfg)
    assert not retention.begin_cache_job(plan, job, 'a' * 64, output_base=tmp_path)
    atomic_json(cache / 'run_binding.json', {'schema': 'synthetic-science-only'})
    atomic_json(cache / 'shard_manifest.json', {'scene_count': 1})
    atomic_json(cache / 'scenes' / 'fixture.json', {'value': 'canonical'})
    (cache / 'tensors').mkdir()
    (cache / 'tensors' / 'fixture.npz').write_bytes(b'canonical tensor')
    atomic_json(cache / 'checkpoints' / 'fixture' / 'scene.json', {'value': 'duplicate'})
    (cache / 'checkpoints' / 'fixture' / 'tensor.npz').write_bytes(b'duplicate tensor')
    atomic_json(features / 'run_binding.json', {'fixture': 'features'})
    (features / 'views').mkdir()
    (features / 'views' / 'fixture.npz').write_bytes(b'full-grid checkpoint')
    seal(features, [features / 'run_binding.json', features / 'views' / 'fixture.npz'], name='feature_shard_lock.json')
    seal(cache, [cache / 'shard_manifest.json', cache / 'scenes' / 'fixture.json', cache / 'tensors' / 'fixture.npz'], name='cache_shard_lock.json')
    atomic_json(cache / 'preparation_receipt.json', {'status': 'shard_complete'})
    seal(cache, [cache / 'run_binding.json', cache / 'preparation_receipt.json', cache / 'cache_shard_lock.json'], name='assembly_lock.json')
    return plan, job, tmp_path, cache, features


def call(fixture):
    plan, job, base, _, _ = fixture
    return retention.retain_completed_cache(plan, job, 'a' * 64, output_base=base)


def test_compacts_exact_duplicates_preserves_cache_and_fast_resume(fixture):
    plan, job, base, cache, features = fixture
    before = retention._science(cache)
    result = call(fixture)
    assert result['status'] == 'retained' and result['canonical_cache_preserved']
    assert len(result['parts']) == 2 and not result['external_retention_verified']
    assert not list(features.rglob('*'))
    assert not (cache / 'checkpoints').exists()
    assert retention._science(cache) == before
    assert retention.begin_cache_job(plan, job, 'a' * 64, output_base=base)
    receipt_hash = digest_file(base / 'retention' / job / 'retention_receipt.json')
    assert call(fixture) == result
    assert digest_file(base / 'retention' / job / 'retention_receipt.json') == receipt_hash
    receipt = complete_job(plan, job, 'a' * 64, output_base=base)
    assert receipt['checkpoint_retention']['status'] == 'retained'


def test_incomplete_science_preserves_every_loose_file_and_blocks_completion(fixture):
    plan, job, base, cache, features = fixture
    (cache / 'assembly_lock.json').unlink()
    with pytest.raises(FileNotFoundError): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()
    assert (cache / 'checkpoints' / 'fixture' / 'tensor.npz').is_file()
    assert not list((base / 'retention' / job).glob('*.zip.bin'))
    with pytest.raises(FileNotFoundError): complete_job(plan, job, 'a' * 64, output_base=base)
    assert not (base / 'job_completion.json').exists()


def test_second_archive_corruption_prevents_any_prune(fixture, monkeypatch):
    _, job, base, cache, features = fixture
    original = retention.create_recovery_bundle
    def corrupt_second(**kwargs):
        result = original(**kwargs)
        if kwargs['bundle_path'].name.startswith('cache-checkpoints'):
            kwargs['bundle_path'].write_bytes(b'corrupt archive')
        return result
    monkeypatch.setattr(retention, 'create_recovery_bundle', corrupt_second)
    with pytest.raises(ValueError): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()
    assert (cache / 'checkpoints' / 'fixture' / 'tensor.npz').is_file()
    assert read_json(base / 'retention' / job / 'retention_receipt.json')['status'] == 'pending'


def test_changed_source_after_archiving_stops_before_any_prune(fixture, monkeypatch):
    _, _, _, cache, features = fixture
    original = retention.create_recovery_bundle
    def mutate_after_second(**kwargs):
        result = original(**kwargs)
        if kwargs['bundle_path'].name.startswith('cache-checkpoints'):
            (cache / 'checkpoints' / 'fixture' / 'tensor.npz').write_bytes(b'changed')
        return result
    monkeypatch.setattr(retention, 'create_recovery_bundle', mutate_after_second)
    with pytest.raises(ValueError, match='source changed'): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()


def test_interrupted_prune_resumes_from_verified_archives_without_restore(fixture, monkeypatch):
    _, _, _, cache, features = fixture
    original = retention.prune_verified_state
    def interrupted(**kwargs):
        manifest = read_json(kwargs['manifest_path'])
        (kwargs['source_root'] / manifest['artifacts'][0]['path']).unlink()
        raise KeyboardInterrupt('synthetic interruption')
    monkeypatch.setattr(retention, 'prune_verified_state', interrupted)
    with pytest.raises(KeyboardInterrupt): call(fixture)
    assert (cache / 'scenes' / 'fixture.json').is_file()
    monkeypatch.setattr(retention, 'prune_verified_state', original)
    assert call(fixture)['status'] == 'retained'
    assert not [p for p in features.rglob('*') if p.is_file()]
    assert not (cache / 'checkpoints').exists()


def test_archival_write_failure_leaves_loose_members_and_no_completion(fixture, monkeypatch):
    plan, job, base, cache, features = fixture
    def fail(**kwargs): raise OSError('synthetic disk error')
    monkeypatch.setattr(retention, 'create_recovery_bundle', fail)
    with pytest.raises(OSError): call(fixture)
    with pytest.raises(ValueError, match='pending'): complete_job(plan, job, 'a' * 64, output_base=base)
    assert (features / 'views' / 'fixture.npz').is_file()
    assert (cache / 'checkpoints' / 'fixture' / 'tensor.npz').is_file()
    assert not (base / 'job_completion.json').exists()


def test_sidecar_interruption_preserves_orphan_and_rebuilds_without_science(fixture, monkeypatch):
    plan, job, base, _, features = fixture
    from visual_peer_inspector.workflows import kaggle_recovery
    original = kaggle_recovery._atomic_bytes
    def interrupt_sidecar(path, content):
        if path.name == 'features.manifest.json':
            raise KeyboardInterrupt('synthetic interruption after ZIP rename')
        return original(path, content)
    with monkeypatch.context() as scoped:
        scoped.setattr(kaggle_recovery, '_atomic_bytes', interrupt_sidecar)
        with pytest.raises(KeyboardInterrupt): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()
    assert retention.begin_cache_job(plan, job, 'a' * 64, output_base=base)
    assert call(fixture)['status'] == 'retained'
    destination = base / 'retention' / job
    assert len(list(destination.glob('features.zip.bin.orphan-*'))) == 1
    assert (destination / 'features.zip.bin').is_file()


def test_orphan_repair_refuses_missing_loose_bytes(fixture, monkeypatch):
    _, job, base, _, features = fixture
    from visual_peer_inspector.workflows import kaggle_recovery
    original = kaggle_recovery._atomic_bytes
    def interrupted(path, content):
        if path.name == 'features.manifest.json':raise OSError('fixture sidecar write failure')
        return original(path, content)
    with monkeypatch.context() as scoped:
        scoped.setattr(kaggle_recovery, '_atomic_bytes', interrupted)
        with pytest.raises(OSError):call(fixture)
    (features / 'views' / 'fixture.npz').unlink()
    with pytest.raises(ValueError, match='all original loose source bytes'):call(fixture)
    assert (base / 'retention' / job / 'features.zip.bin').is_file()


def test_existing_or_changed_launch_cannot_claim_scientific_reuse(fixture):
    plan, job, base, _, _ = fixture
    with pytest.raises(ValueError, match='binding changed'):
        retention.begin_cache_job(plan, job, 'b' * 64, output_base=base)
    (base / 'launch-science' / f'{job}.json').unlink()
    with pytest.raises(ValueError, match='no matching launch science binding'):
        retention.begin_cache_job(plan, job, 'a' * 64, output_base=base)


def test_mounted_features_and_symlinks_are_never_prune_targets(fixture, tmp_path):
    plan, job, base, _, features = fixture
    path = base / 'launch-config' / f'{job}.json'
    cfg = read_json(path); cfg['feature_root'] = '/kaggle/input/mounted-features'; atomic_json(path, cfg)
    with pytest.raises(ValueError, match='local output paths'): call(fixture)
    cfg['feature_root'] = str(features); atomic_json(path, cfg)
    (features / 'views' / 'unsafe').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError): call(fixture)
    assert (features / 'views' / 'fixture.npz').exists()


def test_unlisted_incomplete_files_remain_loose(fixture):
    _, _, _, cache, features = fixture
    extra = features / 'views' / 'partial.incomplete'; extra.write_bytes(b'preserve incomplete')
    call(fixture)
    assert extra.read_bytes() == b'preserve incomplete'
    assert (cache / 'tensors' / 'fixture.npz').is_file()


def test_canonical_checkpoint_reference_is_rejected_before_deletion(fixture):
    _, _, _, cache, features = fixture
    lock = read_json(cache / 'cache_shard_lock.json')
    lock['files']['checkpoints/fixture/tensor.npz'] = digest_file(cache / 'checkpoints' / 'fixture' / 'tensor.npz')
    atomic_json(cache / 'cache_shard_lock.json', lock)
    seal(cache, [cache / 'run_binding.json', cache / 'preparation_receipt.json', cache / 'cache_shard_lock.json'], name='assembly_lock.json')
    with pytest.raises(ValueError, match='unexpected checkpoint paths'): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()


def test_retained_output_budget_fails_safely(fixture):
    plan, job, base, cache, features = fixture
    cfg_path = base / 'launch-config' / f'{job}.json'
    cfg = read_json(cfg_path); cfg['retention_maximum_bytes'] = 1; atomic_json(cfg_path, cfg)
    state = base / 'launch-science' / f'{job}.json'
    binding = read_json(state); binding['config_sha256'] = retention.digest(cfg); atomic_json(state, binding)
    with pytest.raises(ValueError, match='retained-output budget'): call(fixture)
    assert (features / 'views' / 'fixture.npz').is_file()
    assert (cache / 'checkpoints' / 'fixture' / 'tensor.npz').is_file()
