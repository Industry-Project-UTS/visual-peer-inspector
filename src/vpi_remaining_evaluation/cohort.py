"""Cohort-size-independent cached prediction hook; never loads evaluator labels."""
from copy import deepcopy
from pathlib import Path
import inspect

from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest_file, read_json,
    seal, verify_seal, verified_unit, terminal_status)
from visual_peer_inspector.data.model_safe_package import membership_sha256
from .gates import CONTRACTS, verify_exposure

CACHE_SCHEMA = 'vpi-current-method-cohort-cache/v1'
FORBIDDEN_KEYS = {'label', 'labels', 'is_anomaly', 'anomaly_label', 'reference_identity_id', 'ground_truth'}


def _reject_labels(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS & set(value):
            raise ValueError('Evaluator fields are forbidden in a current-method prediction cache')
        for child in value.values():
            _reject_labels(child)
    elif isinstance(value, list):
        for child in value:
            _reject_labels(child)


def selected_inputs(payload, views, mode):
    _reject_labels(payload)
    if mode not in {'automatic', 'assisted'}:
        raise ValueError('Unknown current decision mode')
    if len(set(views)) != len(views) or len(views) not in {1, 2, 3, 5}:
        raise ValueError('Invalid selected-view contract')
    rows = {view: deepcopy(payload['score_rows'][view]) for view in views}
    observations = [deepcopy(row) for row in payload['bank_observations'] if row['view_id'] in views]
    expected = {(view, obj['candidate_id']) for view in views for obj in rows[view]['objects']}
    actual = [(row['view_id'], row['candidate_id']) for row in observations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Cached observations do not preserve selected candidate coverage')
    if mode == 'assisted' and payload.get('correspondence_provenance') not in {'dataset_reference', 'user_confirmed'}:
        raise ValueError('Assisted mode requires explicit correspondence provenance')
    if mode == 'automatic':
        for row in observations:
            row['object_id'] = None
    return rows, observations


def predict_cached_cohort(cache_root, freeze_root, output_root, decide, *, adapter_sha256,
                          mode, view_sets, exposure_root=None):
    """Run a pure decision adapter only after exact role/cache/exposure validation.

    ``decide`` has the web decision adapter signature: score_rows,
    bank_observations, views=..., mode=.... Input caches are a new explicit bridge
    contract; older 09 P1 bundles do not satisfy it automatically.
    """
    cache_root, freeze_root, output_root = map(Path, (cache_root, freeze_root, output_root))
    lock = verify_seal(cache_root)
    freeze_lock = verify_seal(freeze_root, name='scientific_lock.json')
    if 'cohort_manifest.json' not in lock['files'] or 'shareable/current_method_freeze.json' not in freeze_lock['files']:
        raise ValueError('Cached predictor seal omits its manifest or method freeze')
    manifest = read_json(cache_root / 'cohort_manifest.json')
    _reject_labels(manifest)
    contract = CONTRACTS.get(manifest.get('role'))
    if contract is None or manifest.get('schema') != CACHE_SCHEMA:
        raise ValueError('Unknown current-method cohort cache')
    if any(manifest.get(key) != getattr(contract, key) for key in ('cohort_id', 'membership_sha256', 'scene_count', 'asset_group_count')):
        raise ValueError('Cohort cache changed the original role contract')
    scenes = manifest['scenes']
    if len(scenes) != contract.scene_count or len(set(scenes)) != len(scenes):
        raise ValueError('Cohort cache has missing or duplicated scenes')
    expected_membership = contract.model_membership_sha256 or contract.membership_sha256
    if membership_sha256(scenes) != expected_membership:
        raise ValueError('Actual cached scene membership differs from the frozen model-facing role')
    if contract.role != 'development':
        verify_exposure(manifest['role'], exposure_root)
    freeze = read_json(freeze_root / 'shareable/current_method_freeze.json')
    if freeze.get('automatic_method') != 'winner_cluster_consensus' or freeze.get('assisted_method') != 'object_bank_top2' or freeze.get('selection_closed') is not True:
        raise ValueError('Current decision methods are not frozen')
    if manifest['method_freeze_sha256'] != digest_file(freeze_root / 'shareable/current_method_freeze.json'):
        raise ValueError('Cohort cache belongs to a different current-method freeze')
    if not view_sets or any(freeze['view_sets'].get(key) != list(views) for key, views in view_sets.items()):
        raise ValueError('Cohort view schedules are not frozen')
    if len(adapter_sha256) != 64 or any(c not in '0123456789abcdef' for c in adapter_sha256):
        raise ValueError('Pure decision adapter requires a source digest')
    adapter_source = inspect.getsourcefile(decide)
    if adapter_source is None or digest_file(adapter_source) != adapter_sha256:
        raise ValueError('Decision adapter source does not match its recorded digest')
    for scene in scenes:
        if f'scenes/{scene}.json' not in lock['files']:
            raise ValueError('Cohort cache seal omits a scene')
    binding = {'schema': CACHE_SCHEMA, 'cache_lock_sha256': digest_file(cache_root / 'artifact_lock.json'),
        'method_freeze_sha256': manifest['method_freeze_sha256'], 'adapter_sha256': adapter_sha256,
        'mode': mode, 'view_sets': view_sets, 'implementation_sha256': digest_file(Path(__file__)),
        'actual_model_membership_sha256': membership_sha256(scenes),
        'exposure_lock_sha256': digest_file(Path(exposure_root) / 'artifact_lock.json') if exposure_root else None}
    binding_hash = bind(output_root, binding)
    if (output_root / 'prediction_lock.json').exists():
        verify_seal(output_root, name='prediction_lock.json')
        return read_json(output_root / 'prediction_receipt.json')
    with terminal_status(output_root, 'current-method-cached-cohort'):
        for scene in scenes:
            unit = output_root / 'scenes' / scene
            if verified_unit(unit, binding_hash):
                continue
            payload = read_json(cache_root / 'scenes' / f'{scene}.json')
            if payload['scene'] != scene:
                raise ValueError('Cached predictor scene mismatch')
            results = []
            for view_set_id, views in view_sets.items():
                rows, observations = selected_inputs(payload, views, mode)
                result = decide(rows, observations, views=list(views), mode=mode)
                # Tuples used as internal lookup keys are deliberately not serialized.
                results.append({'view_set_id': view_set_id, 'views': list(views), 'mode': mode,
                                'ranking': result['ranking'], 'status': result.get('status', 'scored')})
            with atomic_unit(unit, binding_hash) as temporary:
                atomic_json(temporary / 'predictions.json', {'scene': scene, 'settings': results})
        receipt = {'status': 'prediction_complete', 'role': manifest['role'], 'scene_count': len(scenes),
                   'settings_per_scene': len(view_sets), 'mode': mode, 'anomaly_labels_opened': False,
                   'actual_model_membership_verified': True, 'asset_attribution_verified': False,
                   'neural_inference_run': False, 'metric_evaluation_run': False}
        atomic_json(output_root / 'prediction_receipt.json', receipt)
        paths = [output_root / 'run_binding.json', output_root / 'prediction_receipt.json']
        paths.extend(p for p in (output_root / 'scenes').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p))
        seal(output_root, paths, name='prediction_lock.json', metadata={'binding_sha256': binding_hash, 'scene_count': len(scenes), 'anomaly_labels_opened': False})
        return receipt
