"""Current 09 cohort adapters over explicit sealed caches; no neural inference."""
from pathlib import Path
from collections import Counter
import shutil
from uuid import uuid4
from time import perf_counter

import numpy as np
import pandas as pd

from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.web.decision import decide_cached
from vpi_object_bank.evaluation import evaluate_setting, _metric_summary, _diagnostic_summary, _atomic_csv, FAILURES, IDENTITY_METRICS
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest_file, read_json, seal,
    verify_seal, verified_unit, terminal_status)
from vpi_remaining_evaluation.cohort import CACHE_SCHEMA, _reject_labels
from .tensor_cache import current_inputs
from vpi_remaining_evaluation.workbench import resolve_input_root
from vpi_remaining_evaluation.gates import CONTRACTS, PrerequisiteBlocked
from .exposure import verify_exposure
from .contracts import METHODS, VIEW_SETS, RECEIPT_SCHEMA, verify_current_freeze, require_protocol_receipt

PREDICTION_SCHEMA = 'vpi-current-cohort-predictions/v1'
EVALUATOR_SCHEMA = 'vpi-current-cohort-evaluator/v1'
STAGE_ROLE = {'dev-predict': 'development', 'dev-evaluate': 'development', '08f1': 'calibration', '09a': 'calibration', '09b': 'calibration', '09b1': 'internal_lock', '09c': 'internal_lock', '09d': 'internal_lock'}


def _required(cfg, key):
    if not cfg.get(key):
        raise PrerequisiteBlocked('CURRENT_RESOURCE_REQUIRED', f'Configure {key}; legacy P1 bundles do not satisfy the current cache contract.')
    root = Path(cfg[key]).expanduser()
    markers = {'exposure_root': 'exposure_ledger.json', 'method_freeze_root': 'method_freeze.json',
        'calibration_receipt_root': 'protocol_lock.json', 'cache_root': 'cohort_manifest.json',
        'prediction_root': 'prediction_lock.json', 'evaluation_root': 'evaluation_lock.json'}
    # Evaluator discovery is deliberately delayed until all model-facing checks pass.
    return resolve_input_root(root, markers[key]) if key in markers else root


def _context(stage, cfg, source_root):
    stage = stage.lower()
    if stage not in STAGE_ROLE:
        raise ValueError('Unknown original-cohort stage')
    role = STAGE_ROLE[stage]
    exposure = _required(cfg, 'exposure_root')
    verify_exposure(role, exposure)
    freeze_root = _required(cfg, 'method_freeze_root')
    verify_current_freeze(freeze_root, source_root)
    freeze_hash = digest_file(freeze_root / 'method_freeze.json')
    if role == 'internal_lock':
        previous = require_protocol_receipt(_required(cfg, 'calibration_receipt_root'), '09b', freeze_hash)
        if previous['exposure_lock_sha256'] != digest_file(exposure / 'artifact_lock.json'):
            raise ValueError('Calibration protocol and current exposure audit differ')
    return role, freeze_hash, exposure


def verify_cache(cache_root, role):
    root = Path(cache_root)
    lock = verify_seal(root)
    if 'cohort_manifest.json' not in lock['files']:
        raise ValueError('Current cache seal omits the cohort manifest')
    manifest = read_json(root / 'cohort_manifest.json')
    _reject_labels(manifest)
    contract = CONTRACTS[role]
    if manifest.get('schema') != CACHE_SCHEMA or manifest.get('role') != role:
        raise ValueError('Current cache schema or role differs')
    if manifest.get('correspondence_provenance') not in {'dataset_reference', 'user_confirmed'}:
        raise ValueError('Declare one assisted correspondence provenance for the complete cohort')
    for key in ('cohort_id', 'scene_count', 'asset_group_count', 'membership_sha256'):
        if manifest.get(key) != getattr(contract, key):
            raise ValueError('Current cache changed the original role contract')
    scenes = manifest.get('scenes', [])
    if len(scenes) != contract.scene_count or len(set(scenes)) != len(scenes):
        raise ValueError('Current cache scenes are missing or duplicated')
    if membership_sha256(scenes) != (contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Actual cached scene membership differs from the original role')
    if any(not isinstance(scene, str) or Path(scene).name != scene or scene in {'.','..'} for scene in scenes):
        raise ValueError('Scene IDs must be opaque safe path tokens')
    if any(f'scenes/{scene}.json' not in lock['files'] for scene in scenes):
        raise ValueError('Current cache seal omits scene payloads')
    return manifest, lock


def readiness(stage, cfg, source_root):
    """Read-only actionable preflight; a missing resource cannot produce a ready status."""
    try:
        role, freeze_hash, _ = _context(stage, cfg, source_root)
        if stage.lower() in {'dev-predict', '08f1', '09a', '09c', '09b1'}:
            manifest, _ = verify_cache(_required(cfg, 'cache_root'), role)
            if manifest.get('method_freeze_sha256') != freeze_hash:
                raise ValueError('Cache must be explicitly rebound to the current runtime freeze')
        else:
            verify_predictions(_required(cfg, 'prediction_root'), role, freeze_hash)
            _required(cfg, 'evaluator_root')
        return {'stage': stage, 'status': 'prerequisites_present', 'role': role,
            'execution_performed': False, 'neural_inference': 'not implemented here; provide only missing frozen outputs',
            'native_cache_bridge': 'must already have produced the explicit current cache schema',
            'warning': 'Prepared adapters and preflight do not establish completed protected evaluation'}
    except (PrerequisiteBlocked, FileNotFoundError, ValueError) as error:
        return {'stage': stage, 'status': 'blocked', 'code': getattr(error, 'code', type(error).__name__),
                'detail': str(error), 'execution_performed': False}


def _observations(rows, result, views):
    scored = {(o['view_id'], o['candidate_id']): o for o in result.get('scored_observations') or []}
    return [{'view_id': view, 'candidate_id': obj['candidate_id'], 'component_scores': obj['scores'],
             'normalized_score': result['component_rank'].get((view, obj['candidate_id'])),
             **({'object_bank': scored[(view, obj['candidate_id'])]['object_bank']} if (view, obj['candidate_id']) in scored else {})}
            for view in views for obj in rows[view]['objects']]


def predict_current(stage, cfg, source_root):
    """Run both fixed modes and retain per-observation evidence before any labels."""
    started = perf_counter()
    if stage.lower() not in {'dev-predict', '09a', '09c'}:
        raise ValueError('Current prediction entry is limited to development,09A and09C')
    role, freeze_hash, exposure = _context(stage, cfg, source_root)
    cache_root, root = _required(cfg, 'cache_root'), _required(cfg, 'output_root')
    manifest, _ = verify_cache(cache_root, role)
    if manifest.get('method_freeze_sha256') != freeze_hash:
        raise ValueError('Cache belongs to a different current-method freeze')
    binding = {'schema': PREDICTION_SCHEMA, 'stage': stage.lower(), 'role': role,
        'cache_lock_sha256': digest_file(cache_root / 'artifact_lock.json'), 'method_freeze_sha256': freeze_hash,
        'exposure_lock_sha256': digest_file(exposure / 'artifact_lock.json'),
        'implementation_sha256': digest_file(__file__), 'view_sets': VIEW_SETS, 'methods': METHODS,
        'scene_count': manifest['scene_count'], 'actual_model_membership_sha256': membership_sha256(manifest['scenes'])}
    binding['assisted_provenance'] = manifest['correspondence_provenance']
    binding_hash = bind(root, binding)
    if (root / 'prediction_lock.json').exists():
        verify_seal(root, name='prediction_lock.json')
        return read_json(root / 'prediction_receipt.json')
    with terminal_status(root, 'current-original-cohort-prediction'):
        for index, scene in enumerate(sorted(manifest['scenes'])):
            unit = root / 'scenes' / scene
            if verified_unit(unit, binding_hash):
                continue
            payload = read_json(cache_root / 'scenes' / f'{scene}.json')
            if payload.get('scene') != scene:
                raise ValueError('Cached scene payload identity differs')
            if payload.get('correspondence_provenance') != manifest['correspondence_provenance']:
                raise ValueError('Dataset-reference and user-confirmed provenance cannot be pooled')
            settings = []
            for mode in ('automatic', 'assisted'):
                for name, views in VIEW_SETS.items():
                    # Reuse the completed cache hook's candidate and view isolation.
                    rows, bank = current_inputs(payload, views, mode, cache_root)
                    result = decide_cached(rows, bank, views=views, mode=mode)
                    settings.append({'view_set_id': name, 'views': views, 'view_count': len(views),
                        'mode': mode, 'route': 'automatic_predicted' if mode == 'automatic' else 'assisted_correspondence',
                        'region': 'automatic', 'method': METHODS[mode],
                        'correspondence_provenance': 'predicted' if mode == 'automatic' else payload['correspondence_provenance'],
                        'observations': _observations(rows, result, views), 'rankings': {METHODS[mode]: result['ranking']},
                        'association_tracks': result['tracks']})
            with atomic_unit(unit, binding_hash) as temporary:
                atomic_json(temporary / 'predictions.json', {'schema': PREDICTION_SCHEMA, 'scene': scene, 'role': role, 'settings': settings})
                _atomic_csv(temporary / 'per_view_scores.csv', pd.DataFrame([
                    {'mode': setting['mode'], 'view_set_id': setting['view_set_id'], 'view_id': o['view_id'],
                     'candidate_id': o['candidate_id'], 'normalized_score': o['normalized_score'],
                     'bank_score': o.get('object_bank', {}).get('score'), **o['component_scores']}
                    for setting in settings for o in setting['observations']]))
            print(f'{stage.upper()} cached scene {index + 1}/{len(manifest["scenes"])} sealed', flush=True)
        receipt = {'schema': PREDICTION_SCHEMA, 'status': 'prediction_complete', 'role': role,
            'scene_count': len(manifest['scenes']), 'settings_per_scene': 2 * len(VIEW_SETS),
            'modes': ['automatic', 'assisted'], 'anomaly_labels_opened': False, 'neural_inference_run': False,
            'invocation_elapsed_seconds': perf_counter() - started, 'runtime_scope': 'this CPU invocation, including reused scene checkpoints'}
        receipt['assisted_provenance'] = manifest['correspondence_provenance']
        atomic_json(root / 'prediction_receipt.json', receipt)
        paths = [root / 'run_binding.json', root / 'prediction_receipt.json']
        paths.extend(p for p in (root / 'scenes').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p))
        seal(root, paths, name='prediction_lock.json', metadata={'binding_sha256': binding_hash})
    return receipt


def verify_predictions(root, role, freeze_hash):
    root = Path(root)
    lock = verify_seal(root, name='prediction_lock.json')
    if not {'run_binding.json', 'prediction_receipt.json'} <= set(lock['files']):
        raise ValueError('Current prediction seal omits required records')
    binding = read_json(root / 'run_binding.json')
    receipt = read_json(root / 'prediction_receipt.json')
    if binding.get('schema') != PREDICTION_SCHEMA or binding.get('role') != role or binding.get('method_freeze_sha256') != freeze_hash:
        raise ValueError('Current prediction schema, role or method binding differs')
    if binding.get('view_sets') != VIEW_SETS or binding.get('methods') != METHODS:
        raise ValueError('Current predictions changed methods or selected-view schedules')
    if binding.get('assisted_provenance') not in {'dataset_reference', 'user_confirmed'} or receipt.get('assisted_provenance') != binding['assisted_provenance']:
        raise ValueError('Prediction correspondence provenance is missing or mixed')
    scenes = sorted(Path(name).parts[1] for name in lock['files'] if name.startswith('scenes/') and name.endswith('/predictions.json'))
    contract = CONTRACTS[role]
    if len(scenes) != contract.scene_count or membership_sha256(scenes) != (contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Sealed predictions do not cover the exact original role')
    if receipt.get('status') != 'prediction_complete' or receipt.get('scene_count') != len(scenes):
        raise ValueError('Current prediction receipt is incomplete')
    return binding, scenes


def _validate_setting(setting, expected_candidates=None):
    if setting['mode'] not in METHODS or setting['views'] != VIEW_SETS.get(setting['view_set_id']):
        raise ValueError('Evaluator prediction setting differs from the frozen modes or views')
    expected_method = METHODS[setting['mode']]
    expected_route = 'automatic_predicted' if setting['mode'] == 'automatic' else 'assisted_correspondence'
    if setting.get('region') != 'automatic' or setting.get('route') != expected_route or setting.get('method') != expected_method:
        raise ValueError('Prediction changed its region, route or declared method')
    if set(setting['rankings']) != {expected_method} or setting['view_count'] != len(setting['views']):
        raise ValueError('Evaluator received an unexpected method or view count')
    keys = [(o['view_id'], o['candidate_id']) for o in setting['observations']]
    if len(keys) != len(set(keys)) or any(view not in setting['views'] for view, _ in keys):
        raise ValueError('Duplicate candidates or unselected-view observations')
    if expected_candidates is not None and set(keys) != set(expected_candidates):
        raise ValueError('Prediction observation inventory differs from its sealed cache')
    members, ids = [], []
    for obj in setting['rankings'][expected_method]:
        ids.append(obj['id'])
        group = [(m['view_id'], m['candidate_id']) for m in obj['members']]
        if not group or len({view for view, _ in group}) != len(group) or not set(group) <= set(keys):
            raise ValueError('Invalid physical object membership')
        if obj['score'] is not None and not np.isfinite(float(obj['score'])):
            raise ValueError('Invalid ranking score')
        members.extend(group)
    if len(ids) != len(set(ids)) or len(members) != len(set(members)):
        raise ValueError('Duplicated ranking identity or observation')
    if setting['mode'] == 'assisted' and set(members) != set(keys):
        raise ValueError('Assisted retention discarded candidates')


def _finalize_protocol(root, stage, binding):
    """Finish the small receipt boundary after a sealed evaluation, including resume."""
    root = Path(root)
    verify_seal(root, name='evaluation_lock.json')
    scientific = read_json(root / 'shareable/evaluation_receipt.json')
    receipt = {'schema': RECEIPT_SCHEMA, 'stage': stage.lower(), 'status': 'protocol_complete',
        'roles': [binding['role']], 'modes': ['automatic', 'assisted'], 'no_retuning': True,
        'method_freeze_sha256': binding['method_freeze_sha256'], 'exposure_lock_sha256': binding['exposure_lock_sha256'],
        'prediction_lock_sha256': binding['prediction_lock_sha256'],
        'evaluation_lock_sha256': digest_file(root / 'evaluation_lock.json'),
        'independent_confirmation': False,
        'evaluation_claim': scientific.get('evaluation_claim', 'Fixed-method evaluation; no independent-confirmation claim'),
        'exposure_assessment': scientific.get('exposure_assessment', {}),
        'review_policy_status': 'unfitted; no calibrated readiness claim',
        'scope': 'fixed object-ranking evaluation; automatic and correspondence-assisted results remain separate'}
    if (root / 'protocol_lock.json').exists():
        existing = require_protocol_receipt(root, stage, binding['method_freeze_sha256'])
        if existing != receipt:
            raise ValueError('Protocol receipt differs from the sealed evaluation')
        return existing
    if (root / 'stage_receipt.json').exists():
        if read_json(root / 'stage_receipt.json') != receipt:
            raise ValueError('Unsealed stage receipt conflicts with the completed evaluation')
    else:
        atomic_json(root / 'stage_receipt.json', receipt)
    seal(root, [root / 'stage_receipt.json', root / 'evaluation_lock.json'], name='protocol_lock.json')
    return receipt


def evaluate_predictions(stage, cfg, source_root):
    """Evaluate the current sealed schema. Existing native P1 evaluators are rejected."""
    started = perf_counter()
    if stage.lower() not in {'dev-evaluate', '09b', '09d'}:
        raise ValueError('This evaluator handles only the original development/calibration/internal roles')
    role, freeze_hash, exposure = _context(stage, cfg, source_root)
    prediction_root, evaluator, root = (_required(cfg, key) for key in ('prediction_root', 'evaluator_root', 'output_root'))
    prediction_binding, scenes = verify_predictions(prediction_root, role, freeze_hash)
    exposure_hash = digest_file(exposure / 'artifact_lock.json')
    if prediction_binding['exposure_lock_sha256'] != exposure_hash:
        raise ValueError('Prediction and evaluation exposure audits differ')
    cache_root = _required(cfg, 'cache_root')
    cache_manifest, _ = verify_cache(cache_root, role)
    if digest_file(cache_root / 'artifact_lock.json') != prediction_binding['cache_lock_sha256'] or cache_manifest['method_freeze_sha256'] != freeze_hash:
        raise ValueError('Evaluator prediction inventory does not bind the actual current cache')
    # Verify every selected-view prediction before opening the evaluator package.
    for scene in scenes:
        cached = read_json(cache_root / 'scenes' / f'{scene}.json')
        payload = read_json(prediction_root / 'scenes' / scene / 'predictions.json')
        if payload.get('scene') != scene or payload.get('role') != role:
            raise ValueError('Prediction scene or role differs')
        expected = {(mode, name) for mode in METHODS for name in VIEW_SETS}
        actual = [(s['mode'], s['view_set_id']) for s in payload['settings']]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Current prediction settings are missing or duplicated')
        for setting in payload['settings']:
            expected_candidates = {(view, obj['candidate_id']) for view in VIEW_SETS.get(setting['view_set_id'], []) for obj in cached['score_rows'][view]['objects']}
            _validate_setting(setting, expected_candidates)
            expected_provenance = 'predicted' if setting['mode'] == 'automatic' else prediction_binding['assisted_provenance']
            if setting.get('correspondence_provenance') != expected_provenance:
                raise ValueError('Evaluation would pool incompatible correspondence provenance')
    evaluator = resolve_input_root(evaluator, 'current_evaluator_manifest.json')
    evaluator_lock = verify_seal(evaluator)
    if not {'current_evaluator_manifest.json', 'scene_index.json'} <= set(evaluator_lock['files']):
        raise PrerequisiteBlocked('NATIVE_EVALUATOR_BRIDGE_REQUIRED', 'Produce the current evaluator manifest and reviewed neutral scene-to-asset index; legacy P1 files cannot be relabelled.')
    manifest = read_json(evaluator / 'current_evaluator_manifest.json')
    if manifest.get('schema') != EVALUATOR_SCHEMA or manifest.get('role') != role or manifest.get('cache_lock_sha256') != prediction_binding['cache_lock_sha256']:
        raise ValueError('Evaluator schema, role or cached observation binding differs')
    if manifest.get('method_freeze_sha256') != freeze_hash or manifest.get('membership_sha256') != CONTRACTS[role].membership_sha256:
        raise ValueError('Evaluator method or original membership differs')
    index_hash = digest_file(evaluator / 'scene_index.json')
    audit = read_json(exposure / 'exposure_ledger.json')
    if manifest.get('asset_index_sha256') != index_hash or index_hash not in audit['reviewed_source_sha256']:
        raise PrerequisiteBlocked('ASSET_MAPPING_REVIEW_REQUIRED', 'The exposure audit must attest the exact neutral scene-to-asset index before grouped evaluation.')
    index = read_json(evaluator / 'scene_index.json')
    if set(index) != set(scenes) or len(set(index.values())) != CONTRACTS[role].asset_group_count:
        raise ValueError('Evaluator asset index changed scene or asset denominators')
    if any(f'scenes/{scene}.json' not in evaluator_lock['files'] for scene in scenes):
        raise ValueError('Evaluator seal omits scene truth')
    binding = {'schema': EVALUATOR_SCHEMA, 'stage': stage.lower(), 'role': role,
        'prediction_lock_sha256': digest_file(prediction_root / 'prediction_lock.json'),
        'evaluator_lock_sha256': digest_file(evaluator / 'artifact_lock.json'),
        'method_freeze_sha256': freeze_hash, 'exposure_lock_sha256': exposure_hash,
        'implementation_sha256': digest_file(__file__), 'asset_index_sha256': index_hash}
    binding_hash = bind(root, binding)
    if (root / 'evaluation_lock.json').exists():
        verify_seal(root, name='evaluation_lock.json')
        if role == 'development':
            return read_json(root/'shareable/evaluation_receipt.json')
        return _finalize_protocol(root, stage, binding)
    rows = []
    with terminal_status(root, 'current-original-cohort-evaluation'):
        for number, scene in enumerate(scenes):
            unit = root / 'private/scenes' / scene
            if not verified_unit(unit, binding_hash):
                truth = read_json(evaluator / 'scenes' / f'{scene}.json')
                if set(truth) != {'labels', 'correspondence', 'asset_group'} or truth['asset_group'] != index[scene]:
                    raise ValueError('Evaluator scene truth differs from its reviewed schema or asset mapping')
                payload = read_json(prediction_root / 'scenes' / scene / 'predictions.json')
                scene_rows = []
                for setting in payload['settings']:
                    records = evaluate_setting(scene, setting, truth)
                    # Correct the inherited coverage bookkeeping only in this new stage.
                    known = {truth['correspondence'][f"automatic/{o['view_id']}/{o['candidate_id']}"] for o in setting['observations']} - {None}
                    for row in records:
                        retained = {truth['correspondence'][f"automatic/{m['view_id']}/{m['candidate_id']}"]
                            for obj in setting['rankings'][row['method']] for m in obj['members']} - {None}
                        row['identity_reference_coverage'] = len(retained & known) / len(known) if known else None
                        row['role'] = role
                        row['mode'] = setting['mode']
                        row['correspondence_provenance'] = setting['correspondence_provenance']
                    scene_rows.extend(records)
                with atomic_unit(unit, binding_hash) as temporary:
                    atomic_json(temporary / 'metrics.json', scene_rows)
            rows.extend(read_json(unit / 'metrics.json'))
            print(f'{stage.upper()} scene {number + 1}/{len(scenes)} evaluated and checkpointed', flush=True)
        frame = pd.DataFrame(rows)
        _atomic_csv(root / 'private/scene_metrics.csv', frame)
        groups=['route','method','correspondence_provenance','view_count']
        _atomic_csv(root / 'shareable/metric_summary.csv', _metric_summary(frame, groups))
        _atomic_csv(root / 'shareable/view_schedule_summary.csv', _metric_summary(frame, [*groups, 'view_set_id']))
        for filename,names in [('failure_summary.csv',FAILURES),('identity_summary.csv',IDENTITY_METRICS)]:
            summary=_diagnostic_summary(frame,names)
            summary['correspondence_provenance']=summary['route'].map(lambda route:'predicted' if route=='automatic_predicted' else prediction_binding['assisted_provenance'])
            _atomic_csv(root/'shareable'/filename,summary)
        _atomic_csv(root / 'shareable/anomaly_count_summary.csv', _metric_summary(frame, [*groups, 'anomaly_count']))
        _atomic_csv(root / 'shareable/majority_summary.csv', _metric_summary(frame, [*groups, 'strict_normal_majority']))
        scientific_receipt = {'status': 'evaluation_complete', 'role': role, 'scene_count': len(scenes),
            'asset_group_count': len(set(index.values())), 'methods': METHODS, 'selection_closed': True,
            'invocation_elapsed_seconds': perf_counter() - started, 'runtime_scope': 'this CPU invocation, including reused scene checkpoints',
            'assisted_provenance':prediction_binding['assisted_provenance'],
            'exposure_assessment': verify_exposure(role, exposure),
            'independent_confirmation': False,
            'evaluation_claim': verify_exposure(role, exposure).get('claim', 'Fixed-method evaluation under the recorded asset-access policy; no independence inferred from role name'),
            'protocol_certification': 'development_only' if role == 'development' else 'fixed_protected_ranking',
            'review_policy': 'no probability calibration or readiness threshold fitted by this ranking evaluator'}
        atomic_json(root / 'shareable/evaluation_receipt.json', scientific_receipt)
        paths = [root / 'run_binding.json', *sorted((root / 'shareable').glob('*'))]
        paths.extend(p for p in (root / 'private').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p))
        seal(root, paths, name='evaluation_lock.json')
        receipt = scientific_receipt if role == 'development' else _finalize_protocol(root, stage, binding)
    return receipt


def release_calibration(cfg, source_root):
    """08F1 releases a verified current calibration cache; no raw-label conversion."""
    return _release_cache('08f1', cfg, source_root)


def release_internal(cfg, source_root):
    """09B1 additionally requires the completed current calibration receipt."""
    return _release_cache('09b1', cfg, source_root)


def _release_cache(stage, cfg, source_root):
    """Copy only an already prepared current model cache, never evaluator truth."""
    role, freeze_hash, exposure = _context(stage, cfg, source_root)
    cache_root, output_root = _required(cfg, 'cache_root'), _required(cfg, 'output_root')
    manifest, lock = verify_cache(cache_root, role)
    if manifest.get('method_freeze_sha256') != freeze_hash:
        raise ValueError('Current resource cache differs from the current method freeze')
    allowed = {'cohort_manifest.json', *(f'scenes/{scene}.json' for scene in manifest['scenes'])}
    for relative in lock['files']:
        path = cache_root / relative
        media = relative.startswith('media/') and path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}
        tensor = relative.startswith('tensors/') and path.suffix.lower() == '.npz'
        if relative not in allowed and not media and not tensor:
            raise ValueError('Current model release contains an unexpected or non-model artifact')
        if path.is_symlink() or not path.resolve().is_relative_to(cache_root.resolve()):
            raise ValueError('Current model release requires regular relative files')
    for scene in manifest['scenes']:
        payload = read_json(cache_root / 'scenes' / f'{scene}.json')
        if payload.get('scene') != scene or payload.get('correspondence_provenance') != manifest['correspondence_provenance']:
            raise ValueError('Current model release scene identity or neutral provenance differs')
        for mode in METHODS:
            for views in VIEW_SETS.values():
                current_inputs(payload, views, mode, cache_root)
    # This is a native-cache release, not a converter from old raw archive records.
    binding = {'stage': stage, 'cache_lock_sha256': digest_file(cache_root / 'artifact_lock.json'),
        'method_freeze_sha256': freeze_hash, 'exposure_lock_sha256': digest_file(exposure / 'artifact_lock.json'),
        'implementation_sha256': digest_file(__file__)}
    bind(output_root, binding)
    if (output_root / 'release_lock.json').exists():
        verify_seal(output_root, name='release_lock.json')
        return read_json(output_root / 'release_receipt.json')
    required_bytes = sum((cache_root / name).stat().st_size for name in lock['files'])
    if shutil.disk_usage(output_root).free < required_bytes * 1.1:
        raise OSError('Current resource storage preflight failed')
    destination = output_root / 'model-cache'
    if destination.exists():
        verify_seal(destination)
        if digest_file(destination / 'artifact_lock.json') != binding['cache_lock_sha256']:
            raise ValueError('Existing released model cache changed')
    else:
        temporary = output_root / 'model-cache.partial'
        if temporary.exists():
            temporary.rename(output_root / ('model-cache.interrupted-' + uuid4().hex[:8]))
        temporary.mkdir()
        for relative in [*sorted(lock['files']), 'artifact_lock.json']:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_root / relative, target)
        verify_seal(temporary)
        temporary.rename(destination)
    receipt = {'status': 'model_cache_released', 'role': role, 'scene_count': manifest['scene_count'],
        'evaluator_opened': False, 'neural_inference_run': False,
        'evaluator_resource': 'prepare and keep the current evaluator package physically separate; this stage does not release labels',
        'native_bridge': 'requires a pre-existing current cache; raw/P1 archive conversion is not implemented'}
    atomic_json(output_root / 'release_receipt.json', receipt)
    seal(output_root, [output_root/'run_binding.json',output_root/'release_receipt.json',
        *(p for p in destination.rglob('*') if p.is_file())], name='release_lock.json')
    return receipt
