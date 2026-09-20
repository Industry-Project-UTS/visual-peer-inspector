"""Typed reference conversion with geometry and outcome access kept separate.

The neutral operation reads only registered crosswalk and geometry files. The
truth operation verifies the completed prediction boundary before reading labels.
Neither operation chooses a match using an anomaly answer.
"""
from pathlib import Path
import json
import numpy as np

from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.evaluation.proposals import match_boxes
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest_file,
    read_json, seal, verified_unit, verify_seal)
from vpi_remaining_evaluation.gates import CONTRACTS
from .official import OFFICIAL
from .contracts import METHODS, VIEW_SETS
from .native_cache import native_regions, NEUTRAL_SCHEMA

RAW_SCHEMA = 'visual-peer-official-evaluator-manifest/v1'
GEOMETRY_SCHEMA = 'vpi-current-reference-geometry/v1'
MATCH_IOU = 0.5


def _contract(role):
    return {**CONTRACTS, **OFFICIAL}[role]


def _member(root, name):
    root = Path(root)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Reference member escapes its registered resource')
    return path


def _registered(cfg, names):
    """Verify only requested members; neutral conversion never opens label bytes."""
    root = Path(cfg['native_evaluator_root'])
    path = _member(root, 'evaluator_manifest.json')
    if digest_file(path) != cfg['expected_evaluator_manifest_sha256']:
        raise ValueError('Native evaluator manifest differs from reviewed source')
    manifest = read_json(path)
    contract = _contract(cfg['role'])
    expected = {'schema_version': RAW_SCHEMA, 'cohort_id': contract.cohort_id,
        'source_membership_sha256': contract.membership_sha256,
        'scene_count': contract.scene_count, 'image_count': contract.image_count}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Native evaluator role or fixed membership contract differs')
    files = {f['path']: f for f in manifest['files']}
    if len(files) != len(manifest['files']):
        raise ValueError('Duplicate native evaluator member')
    result = {}
    for name in names:
        path = _member(root, name)
        entry = files.get(name)
        if entry is None or path.stat().st_size != entry['size_bytes'] or digest_file(path) != entry['sha256']:
            raise ValueError('Native evaluator member failed its source checksum')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_scene = {row['opaque_scene_id']: row for row in rows}
        if len(by_scene) != len(rows) or len(rows) != contract.scene_count:
            raise ValueError('Reference scenes are missing or duplicated')
        for row in rows:
            if row['cohort_id'] != contract.cohort_id or row['release_id'] != contract.release_id:
                raise ValueError('Reference row belongs to a different cohort or release')
        result[name] = by_scene
    return result


def prepare_geometry(cfg):
    """Strip a native reference resource to neutral IDs, boxes and asset groups."""
    records = _registered(cfg, ['crosswalk_private.jsonl', 'geometry_private.jsonl'])
    crosswalk, geometry = (records[name] for name in ('crosswalk_private.jsonl', 'geometry_private.jsonl'))
    contract = _contract(cfg['role'])
    if set(crosswalk) != set(geometry) or membership_sha256(crosswalk) != (contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Reference opaque membership differs')
    root = Path(cfg['output_root'])
    binding = {'schema': GEOMETRY_SCHEMA, 'role': cfg['role'],
        'native_manifest_sha256': cfg['expected_evaluator_manifest_sha256'],
        'source_sha256': digest_file(__file__)}
    signature = bind(root, binding)
    if (root / 'artifact_lock.json').exists():
        verify_seal(root)
        return read_json(root / 'geometry_manifest.json')
    index = {}
    for scene in sorted(crosswalk):
        bridge = crosswalk[scene]
        asset = str(bridge['base_asset_id'])
        if not asset:
            raise ValueError('Reference asset identity is missing')
        index[scene] = asset
        mapping = bridge['object_id_map']
        if not mapping or set(mapping.values()) != set(range(len(mapping))):
            raise ValueError('Reference object index is not a bijection')
        ids = [key for key, _ in sorted(mapping.items(), key=lambda item: item[1])]
        views = {}
        for row in geometry[scene]['views']:
            view = row['view_id']
            boxes = row['boxes_xyxy']
            if view in views or len(boxes) != len(ids):
                raise ValueError('Reference boxes or views do not align with physical identities')
            if bridge['view_id_map'].get(view) != row['source_filename']:
                raise ValueError('Reference view filename mapping differs')
            values = np.asarray(boxes, dtype=float)
            if values.shape != (len(ids), 4) or not np.isfinite(values).all():
                raise ValueError('Reference boxes must be finite original-image coordinates')
            views[view] = dict(zip(ids, values.tolist()))
        if set(views) != {f'V{i:02d}' for i in range(contract.views_per_scene)}:
            raise ValueError('Reference views differ from the fixed twenty-view inventory')
        unit = root / 'scenes' / scene
        if not verified_unit(unit, signature):
            with atomic_unit(unit, signature) as temporary:
                atomic_json(temporary / 'geometry.json', {'scene': scene, 'asset_group': asset, 'views': views})
    if len(set(index.values())) != contract.asset_group_count:
        raise ValueError('Reference asset denominator differs')
    atomic_json(root / 'scene_index.json', index)
    manifest = {**binding, 'scene_count': len(index), 'asset_group_count': len(set(index.values())),
        'membership_sha256': contract.membership_sha256,
        'asset_index_sha256': digest_file(root / 'scene_index.json'), 'anomaly_labels_opened': False}
    atomic_json(root / 'geometry_manifest.json', manifest)
    seal(root, [p for p in root.rglob('*') if p.is_file()])
    return manifest


def _geometry(cfg):
    root = Path(cfg['geometry_root'])
    verify_seal(root)
    manifest = read_json(root / 'geometry_manifest.json')
    contract = _contract(cfg['role'])
    if manifest.get('schema') != GEOMETRY_SCHEMA or manifest.get('role') != cfg['role'] or manifest.get('membership_sha256') != contract.membership_sha256:
        raise ValueError('Neutral geometry contract differs')
    return root


def _correspondence(objects, reference):
    """Deterministic one-to-one geometry matching, with explicit unmatched IDs."""
    ids = sorted(key for key, box in reference.items() if box[0] < box[2] and box[1] < box[3])
    objects = sorted(objects, key=lambda item: item['candidate_id'])
    matches = match_boxes([reference[key] for key in ids], [obj['box'] for obj in objects], minimum_iou=MATCH_IOU)
    mapped = {m.proposal_index: ids[m.reference_index] for m in matches}
    return {obj['candidate_id']: mapped.get(i) for i, obj in enumerate(objects)}


def prepare_neutral(cfg):
    geometry = _geometry(cfg)
    regions, region_hash = native_regions(cfg, _contract(cfg['role']))
    root = Path(cfg['output_root'])
    signature = bind(root, {'schema': NEUTRAL_SCHEMA, 'region_lock_sha256': region_hash,
        'geometry_lock_sha256': digest_file(geometry / 'artifact_lock.json'),
        'minimum_iou': MATCH_IOU, 'source_sha256': digest_file(__file__)})
    if (root / 'artifact_lock.json').exists():
        verify_seal(root)
        return {'status': 'reused', 'anomaly_labels_opened': False}
    rows = []
    for scene in sorted({scene for scene, _ in regions}):
        truth = read_json(geometry / 'scenes' / scene / 'geometry.json')
        for key in sorted(key for key in regions if key[0] == scene):
            mapped = _correspondence(regions[key]['objects'], truth['views'][key[1]])
            rows.extend({'scene': scene, 'view_id': key[1], 'candidate_id': candidate, 'object_id': reference}
                        for candidate, reference in mapped.items())
    value = {'schema': NEUTRAL_SCHEMA, 'region_lock_sha256': region_hash,
        'provenance': 'dataset_reference', 'observations': rows}
    atomic_json(root / 'neutral_correspondence.json', value)
    seal(root, [root / 'neutral_correspondence.json'], metadata={'binding_sha256': signature})
    return {'status': 'complete', 'observation_count': len(rows),
        'unmatched_count': sum(row['object_id'] is None for row in rows), 'anomaly_labels_opened': False}


def _prediction_boundary(cfg):
    """Complete both-mode candidate validation before any outcome file access."""
    from . import original, official
    role = cfg['role']
    cache = Path(cfg['cache_root'])
    freeze_hash = digest_file(Path(cfg['method_freeze_root']) / 'method_freeze.json')
    if role in OFFICIAL:
        manifest = official.validate_cache(cache, role, freeze_hash)
        if set(cfg['prediction_roots']) != set(METHODS):
            raise ValueError('Both official modes must be sealed before native truth conversion')
        locks = {}
        for mode, path in cfg['prediction_roots'].items():
            record = official.validate_prediction(path, role, cache_root=cache)
            if record['binding']['mode'] != mode or record['binding']['method_freeze_sha256'] != freeze_hash:
                raise ValueError('Official prediction method binding differs')
            locks[mode] = digest_file(Path(path) / 'prediction_lock.json')
    else:
        manifest, _ = original.verify_cache(cache, role)
        prediction = Path(cfg['prediction_root'])
        binding, scenes = original.verify_predictions(prediction, role, freeze_hash)
        if binding['cache_lock_sha256'] != digest_file(cache / 'artifact_lock.json'):
            raise ValueError('Predictions use a different current cache')
        for scene in scenes:
            cached = read_json(cache / 'scenes' / f'{scene}.json')
            payload = read_json(prediction / 'scenes' / scene / 'predictions.json')
            actual = [(row['mode'], row['view_set_id']) for row in payload['settings']]
            if len(actual) != len(set(actual)) or set(actual) != {(mode, view) for mode in METHODS for view in VIEW_SETS}:
                raise ValueError('Both-mode prediction schedule inventory differs')
            for setting in payload['settings']:
                candidates = {(view, obj['candidate_id']) for view in VIEW_SETS[setting['view_set_id']]
                    for obj in cached['score_rows'][view]['objects']}
                original._validate_setting(setting, candidates)
        locks = {'both': digest_file(prediction / 'prediction_lock.json')}
    if manifest['method_freeze_sha256'] != freeze_hash:
        raise ValueError('Current cache freeze differs')
    return manifest, locks


def convert_evaluator(cfg):
    manifest, predictions = _prediction_boundary(cfg)
    geometry = _geometry(cfg)
    source = cfg['expected_evaluator_manifest_sha256']
    if read_json(geometry / 'geometry_manifest.json')['native_manifest_sha256'] != source:
        raise ValueError('Geometry and outcome files belong to different native sources')
    return _emit_evaluator(cfg, manifest, predictions, geometry, source,
        lambda: _registered(cfg, ['labels_private.jsonl'])['labels_private.jsonl'])


def _emit_evaluator(cfg, manifest, predictions, geometry, source, load_labels, *, converter_sources=None):
    root, cache = Path(cfg['output_root']), Path(cfg['cache_root'])
    binding = {'schema': 'vpi-current-native-evaluator-conversion/v1', 'role': cfg['role'],
        'prediction_locks': predictions, 'cache_lock_sha256': digest_file(cache / 'artifact_lock.json'),
        'geometry_lock_sha256': digest_file(geometry / 'artifact_lock.json'),
        'native_manifest_sha256': source, 'source_sha256': digest_file(__file__),
        'converter_source_sha256': converter_sources or {}}
    signature = bind(root, binding)
    if (root / 'artifact_lock.json').exists():
        verify_seal(root)
        return read_json(root / 'current_evaluator_manifest.json')
    labels = load_labels()
    index = read_json(geometry / 'scene_index.json')
    if set(labels) != set(manifest['scenes']) or set(index) != set(labels):
        raise ValueError('Native truth, geometry and candidate scene inventory differ')
    for scene in sorted(labels):
        unit = root / 'checkpoints' / scene
        if not verified_unit(unit, signature):
            reference = read_json(geometry / 'scenes' / scene / 'geometry.json')
            entries = labels[scene]['object_labels']
            truth = {item['object_id']: int(item['is_anomaly']) for item in entries}
            expected_ids = set(next(iter(reference['views'].values())))
            if len(truth) != len(entries) or set(truth) != expected_ids or any(type(item['is_anomaly']) is not bool for item in entries) or not any(truth.values()):
                raise ValueError('Native labels require unique known physical IDs and binary anomaly flags')
            if labels[scene]['anomaly_count'] != sum(truth.values()):
                raise ValueError('Native anomaly count differs from flags')
            payload = read_json(cache / 'scenes' / f'{scene}.json')
            views = payload['evidence_views']
            correspondence = {}
            for view, scores in payload['score_rows'].items():
                objects = [{'candidate_id': candidate, 'box': obj['box']} for candidate, obj in views[view]['candidates'].items()]
                if {obj['candidate_id'] for obj in objects} != {obj['candidate_id'] for obj in scores['objects']}:
                    raise ValueError('Cache geometry omits candidates required for evaluator matching')
                correspondence.update({f'automatic/{view}/{candidate}': value for candidate, value in
                    _correspondence(objects, reference['views'][view]).items()})
            with atomic_unit(unit, signature) as temporary:
                atomic_json(temporary / 'truth.json', {'labels': truth, 'correspondence': correspondence, 'asset_group': index[scene]})
        atomic_json(root / 'scenes' / f'{scene}.json', read_json(unit / 'truth.json'))
    atomic_json(root / 'scene_index.json', index)
    result = {'schema': 'vpi-current-official-evaluator/v1' if cfg['role'] in OFFICIAL else 'vpi-current-cohort-evaluator/v1',
        'role': cfg['role'], 'cache_lock_sha256': binding['cache_lock_sha256'],
        'method_freeze_sha256': manifest['method_freeze_sha256'],
        'membership_sha256': _contract(cfg['role']).membership_sha256,
        'asset_index_sha256': digest_file(root / 'scene_index.json'),
        'conversion_binding_sha256': signature, 'minimum_matching_iou': MATCH_IOU}
    atomic_json(root / 'current_evaluator_manifest.json', result)
    seal(root, [root / 'current_evaluator_manifest.json', root / 'scene_index.json', root / 'run_binding.json',
                *sorted((root / 'scenes').glob('*.json'))])
    return result
