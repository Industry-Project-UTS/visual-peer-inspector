"""Compact only a completed launch job's duplicate local checkpoints.

The released cache is never pruned. Archives use a .zip.bin suffix so a later
Kaggle attachment does not automatically expand these optional recovery files.
"""
from pathlib import Path
import uuid

from vpi_object_bank.storage import atomic_json, digest, digest_file, free_space, read_json, verify_seal
from visual_peer_inspector.workflows.kaggle_recovery import (
    create_recovery_bundle, verify_recovery_bundle, prune_verified_state,
)

SCHEMA = 'vpi-current-completed-cache-retention/v1'
FEATURE_TOP_LEVELS = ('views', 'run_binding.json', 'run_status.json', 'progress.json',
                      'feature_lock.json', 'feature_shard_lock.json', 'failures')


def _paths(plan, job_id, snapshot, output_base):
    base = Path(output_base).absolute()
    jobs = [row for row in plan['jobs'] if row['id'] == job_id]
    if len(jobs) != 1 or jobs[0]['config'].get('action') != 'prepare_shard' or not jobs[0].get('compact_checkpoints'):
        raise ValueError('Retention is restricted to registered completed cache shard jobs')
    if len(snapshot) != 64 or any(c not in '0123456789abcdef' for c in snapshot):
        raise ValueError('Retention requires the verified source snapshot digest')
    cfg_path = base / 'launch-config' / f'{job_id}.json'
    cfg = read_json(cfg_path)
    cache, features = Path(cfg['output_root']).absolute(), Path(cfg['feature_root']).absolute()
    if cache != base / job_id or features != base / f'{job_id}-features':
        raise ValueError('Retention only accepts this job\'s local output paths')
    # An existing symlink in any path component could redirect pruning to a mount.
    for path in (base, cache, features, base / 'retention' / job_id, base / 'launch-science'):
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise ValueError('Retention paths must not contain symlinks')
    if str(base).startswith('/kaggle/') and base != Path('/kaggle/working'):
        raise ValueError('Kaggle retention must use the writable output directory')
    binding = {'schema': SCHEMA, 'job_id': job_id, 'source_snapshot_sha256': snapshot,
               'plan_sha256': digest(plan), 'config_sha256': digest(cfg)}
    return cfg, cache, features, base / 'retention' / job_id, base / 'launch-science' / f'{job_id}.json', binding


def _science(cache):
    cache_lock = verify_seal(cache, name='cache_shard_lock.json')
    assembly_lock = verify_seal(cache, name='assembly_lock.json')
    if any(Path(relative).parts[0] not in {'shard_manifest.json', 'scenes', 'media', 'tensors'} for relative in cache_lock['files']):
        raise ValueError('Canonical cache seal contains unexpected checkpoint paths')
    if set(assembly_lock['files']) != {'run_binding.json', 'preparation_receipt.json', 'cache_shard_lock.json'}:
        raise ValueError('Assembly seal has an unexpected inventory')
    receipt = read_json(cache / 'preparation_receipt.json')
    if receipt.get('status') != 'shard_complete':
        raise ValueError('Retention requires complete sealed cache science')
    return {name: digest_file(cache / name) for name in
            ('cache_shard_lock.json', 'assembly_lock.json', 'run_binding.json')}


def begin_cache_job(plan, job_id, source_snapshot_sha256, *, output_base=Path('/kaggle/working')):
    """Return True only for the same launch binding with verified complete science.

This gate runs before scientific cells. It avoids extracting archived features
again when retrying retention or downloading an already completed cache job.
"""
    _, cache, features, _, state, binding = _paths(plan, job_id, source_snapshot_sha256, output_base)
    if state.exists():
        if read_json(state) != binding:
            raise ValueError('Launch science binding changed; preserve outputs and choose a new job')
    else:
        if any(root.exists() and any(root.iterdir()) for root in (cache, features)):
            raise ValueError('Existing local checkpoints have no matching launch science binding')
        atomic_json(state, binding)
    if (cache / 'cache_shard_lock.json').exists() and (cache / 'assembly_lock.json').exists():
        _science(cache)
        return True
    return False


def _source_members(root, manifest):
    """Preflight every surviving member before any deletion, including retries."""
    members = []
    for item in manifest['artifacts']:
        path = root / item['path']
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise ValueError('Recovery source contains a symlink')
        if not path.exists():
            continue  # A verified prior prune may have stopped after this member.
        if not path.is_file() or path.stat().st_size != item['size_bytes'] or digest_file(path) != item['sha256']:
            raise ValueError('Recovery source changed before pruning')
        members.append(path)
    return members


def _inventory(root, allowed):
    entries = []
    for top in allowed:
        source = root / top
        paths = [source] if source.is_file() else source.rglob('*')
        for path in paths:
            if path.is_symlink():
                raise ValueError('Recovery source contains a symlink')
            if path.is_file() and not path.name.endswith('.incomplete'):
                entries.append({'path': path.relative_to(root).as_posix(),
                                'sha256': digest_file(path), 'size_bytes': path.stat().st_size})
    return sorted(entries, key=lambda row: row['path'])


def _prune_part(part, manifest):
    kwargs = {key: part[key] for key in ('source_root', 'bundle_path', 'manifest_path', 'allowed_top_levels')}
    kwargs['expected_binding'] = part['binding']
    verify_recovery_bundle(**{k: v for k, v in kwargs.items() if k != 'source_root'})
    members = _source_members(part['source_root'], manifest)
    if len(members) == manifest['artifact_count']:
        return prune_verified_state(**kwargs)
    # Historical prune is deliberately strict about absent files. On an
    # interrupted prune, revalidate the entire archive and all surviving bytes,
    # then remove only surviving exact manifest members; do not restore/recompute.
    for path in members:
        path.unlink()
    return len(members)


def retain_completed_cache(plan, job_id, source_snapshot_sha256, *, output_base=Path('/kaggle/working')):
    """Archive both duplicate roots before pruning either; failures remain retryable."""
    cfg, cache, features, destination, state, binding = _paths(plan, job_id, source_snapshot_sha256, output_base)
    if not state.is_file() or read_json(state) != binding:
        raise ValueError('Retention requires the matching pre-science launch binding')
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / 'retention_receipt.json'
    try:
        science = _science(cache)
        archive_binding = {**binding, 'scientific_seals': science}
        specs_path = destination / 'archive_plan.json'
        if specs_path.exists():
            specs = read_json(specs_path)
            if specs['binding'] != archive_binding:
                raise ValueError('Completed scientific artifacts changed after retention began')
        else:
            if features.exists():
                extra = {p.name for p in features.iterdir()} - set(FEATURE_TOP_LEVELS)
                if extra:
                    raise ValueError('Feature root contains unexpected state; retain it for review')
                feature_allowed = [name for name in FEATURE_TOP_LEVELS if (features / name).exists()]
            else:
                feature_allowed = []
            specs = {'binding': archive_binding, 'parts': []}
            if feature_allowed:
                specs['parts'].append({'name': 'features', 'allowed_top_levels': feature_allowed})
            if (cache / 'checkpoints').exists():
                specs['parts'].append({'name': 'cache-checkpoints', 'allowed_top_levels': ['checkpoints']})
            for spec in specs['parts']:
                spec['source_artifacts'] = _inventory(features if spec['name'] == 'features' else cache, spec['allowed_top_levels'])
            atomic_json(specs_path, specs)
        parts = []
        for spec in specs['parts']:
            name = spec['name']
            part = {'source_root': features if name == 'features' else cache,
                    'bundle_path': destination / f'{name}.zip.bin',
                    'manifest_path': destination / f'{name}.manifest.json',
                    'allowed_top_levels': spec['allowed_top_levels'],
                    'source_artifacts': spec['source_artifacts'],
                    'binding': {**archive_binding, 'part': name, 'source_inventory_sha256': digest(spec['source_artifacts'])}}
            parts.append(part)
        required = sum(p.stat().st_size for part in parts for top in part['allowed_top_levels']
                       for p in ([part['source_root'] / top] if (part['source_root'] / top).is_file()
                                 else (part['source_root'] / top).rglob('*')) if p.is_file())
        maximum = cfg.get('retention_maximum_bytes', 18 * 1024 ** 3)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError('retention_maximum_bytes must be a positive integer')
        # Archive construction temporarily needs a second copy. Final output
        # accounting below removes only exact verified manifest members.
        free_space(destination, required + 64 * 1024 ** 2)
        manifests = []
        for part in parts:
            bundle, sidecar = part['bundle_path'], part['manifest_path']
            kwargs = {k: part[k] for k in ('bundle_path', 'manifest_path', 'allowed_top_levels')}
            temporary = bundle.with_name(bundle.name + '.incomplete')
            if (bundle.exists() != sidecar.exists()) or temporary.exists():
                # An interrupted sidecar write must not force feature extraction.
                # Rebuild only when every original loose source byte still exists;
                # preserve each orphan under a unique name before creating a pair.
                if _inventory(part['source_root'], part['allowed_top_levels']) != part['source_artifacts']:
                    raise ValueError('Orphan archive repair requires all original loose source bytes')
                for orphan in (bundle, sidecar, temporary):
                    if orphan.exists():
                        orphan.rename(orphan.with_name(orphan.name + '.orphan-' + uuid.uuid4().hex))
            if bundle.exists() or sidecar.exists():
                if not bundle.is_file() or not sidecar.is_file():
                    raise ValueError('Incomplete retained archive pair; preserve it and its loose checkpoints')
            else:
                if _inventory(part['source_root'], part['allowed_top_levels']) != part['source_artifacts']:
                    raise ValueError('Recovery source inventory changed before archiving')
                # create_recovery_bundle uses a temporary .incomplete file and
                # writes the sidecar only after ZIP integrity validation.
                create_recovery_bundle(source_root=part['source_root'], binding=part['binding'], **kwargs)
            manifest = verify_recovery_bundle(expected_binding=part['binding'], **kwargs)
            if manifest['artifacts'] != part['source_artifacts']:
                raise ValueError('Recovery archive differs from the initial source inventory')
            manifests.append(manifest)
        # Both complete archives AND all current source bytes must pass before
        # the first unlink. Corruption of the second part cannot remove the first.
        surviving = [_source_members(part['source_root'], manifest) for part, manifest in zip(parts, manifests)]
        retained = sum(p.stat().st_size for root in (cache, features, destination) if root.exists()
                       for p in root.rglob('*') if p.is_file())
        final_size = retained - sum(p.stat().st_size for members in surviving for p in members)
        if final_size > maximum:
            raise ValueError('Completed cache output exceeds the retained-output budget')
        if receipt_path.is_file() and not any(surviving):
            previous = read_json(receipt_path)
            if previous.get('status') == 'retained' and previous.get('binding') == archive_binding:
                return previous
        removed = sum(_prune_part(part, manifest) for part, manifest in zip(parts, manifests))
        if _science(cache) != science:
            raise ValueError('Canonical released cache changed during retention')
        receipt = {'schema': SCHEMA, 'status': 'retained', 'binding': archive_binding,
                   'parts': [{'archive': part['bundle_path'].name, 'manifest': part['manifest_path'].name,
                              'bundle_sha256': manifest['bundle_sha256'], 'artifact_count': manifest['artifact_count']}
                             for part, manifest in zip(parts, manifests)],
                   'removed_members_this_call': removed, 'canonical_cache_preserved': True,
                   'retained_bytes_before_receipt': final_size,
                   'external_retention_verified': False}
        atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        atomic_json(receipt_path, {'schema': SCHEMA, 'status': 'pending', 'binding': binding,
                    'error_type': type(error).__name__, 'scientific_results_preserved': True})
        raise


def verify_retention_receipt(plan, job_id, snapshot, *, output_base=Path('/kaggle/working')):
    """Completion uses this read-only check; it never starts cleanup implicitly."""
    _, cache, _, destination, _, binding = _paths(plan, job_id, snapshot, output_base)
    receipt = read_json(destination / 'retention_receipt.json')
    expected = {**binding, 'scientific_seals': _science(cache)}
    if receipt.get('status') != 'retained' or receipt.get('binding') != expected:
        raise ValueError('Checkpoint retention is pending; scientific cache remains available')
    specs = read_json(destination / 'archive_plan.json')
    if specs['binding'] != expected or len(specs['parts']) != len(receipt['parts']):
        raise ValueError('Retained archive inventory changed')
    for spec, item in zip(specs['parts'], receipt['parts']):
        manifest = verify_recovery_bundle(bundle_path=destination / item['archive'],
            manifest_path=destination / item['manifest'], expected_binding={**expected, 'part': spec['name'], 'source_inventory_sha256': digest(spec['source_artifacts'])},
            allowed_top_levels=spec['allowed_top_levels'])
        if manifest['bundle_sha256'] != item['bundle_sha256']:
            raise ValueError('Retained archive checksum changed')
    return {'status': 'retained', 'receipt_sha256': digest_file(destination / 'retention_receipt.json'),
            'canonical_cache_preserved': True, 'external_retention_verified': False}
