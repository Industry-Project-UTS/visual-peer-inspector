"""Verified 08N automatic tensor reuse across the historical native-ID boundary.

The source candidate ID is a hash of the native observation ID, not an inferred
physical identity. Each reuse also verifies RGB bytes, both mask hash schemes,
crop geometry and tensor schema. No scores, supplied identities or labels enter
this operation. NPZ files are copied byte-for-byte; array indices never change.
"""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import numpy as np
from PIL import Image
from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.scoring.forward_patch import patch_policy
from visual_peer_inspector.workflows.frozen_development_pipeline import verify_prediction_lock
from visual_peer_inspector.workflows.scoring_comparison import preflight
from vpi_object_bank.inputs import _candidate_id
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest, digest_file,
    read_json, seal, terminal_status, verified_unit, verify_seal)
from .contracts import VIEW_SETS
from .native_cache import (_feature_binding, native_feature_binding, normalize_native_object)

SCHEMA = 'vpi-native-feature-remap/v1'


def _native_inventory(cfg):
    from visual_peer_inspector.data.cohort_contract import (DEVELOPMENT_FULL_CONTRACT,
        DS1_SCREEN_CONTRACT, DS2_CONFIRM_CONTRACT, DEVELOPMENT_REMAINDER_CONTRACT)
    contract = DEVELOPMENT_FULL_CONTRACT
    path = Path(cfg['full_development_lock']); combined = read_json(path)
    target_hash = digest_file(path)
    if target_hash != cfg['expected_region_lock_sha256'] or combined.get('status') != 'verified' or combined.get('cohort_id') != contract.cohort_id:
        raise ValueError('Reviewed full-development region lock differs')
    contracts = {c.cohort_id: c for c in (DS1_SCREEN_CONTRACT, DEVELOPMENT_REMAINDER_CONTRACT, DS2_CONFIRM_CONTRACT)}
    declared = {r['cohort_id']: r['lock_sha256'] for r in combined['component_locks']}
    if len(declared) != len(combined['component_locks']) or set(cfg['component_roots']) != set(contracts) or set(declared) != set(contracts):
        raise ValueError('Exact native component set required')
    rows = {}; components = {}
    expected_views = sorted({v for values in VIEW_SETS.values() for v in values})
    for name, c in contracts.items():
        root = Path(cfg['component_roots'][name]); lockpath = root / 'prediction_lock.json'
        if digest_file(lockpath) != declared[name]:
            raise ValueError('Native component lock differs')
        lock = verify_prediction_lock(lockpath)
        if lock['identity_sha256'] != combined['identity_sha256'] or lock['cohort_id'] != name:
            raise ValueError('Native components have incompatible vision identities')
        keys = set(); scenes = set()
        with (root / 'prediction_manifest.jsonl').open() as stream:
            for line in stream:
                source = json.loads(line)
                scene, view = source['opaque_scene_id'], source['view_id']
                key = (scene, view)
                if source['cohort_id'] != name or key in rows or source['object_count'] != len(source['objects']):
                    raise ValueError('Invalid native manifest membership or duplicate view')
                # Do not carry historical scores/heatmaps into the model-facing map.
                rows[key] = {'scene': scene, 'view': view, 'rgb_sha256': source['image_sha256'],
                    'objects': source['objects'], 'component': name}
                keys.add(key); scenes.add(scene)
        if len(scenes) != c.scene_count or membership_sha256(sorted(scenes)) != c.membership_sha256 or keys != {(s,v) for s in scenes for v in expected_views}:
            raise ValueError('Native component membership or selected views differ')
        components[name] = {'prediction_lock_sha256': declared[name], 'scene_count': len(scenes), 'view_count': len(keys)}
    scenes = sorted({s for s, _ in rows})
    if len(scenes) != contract.scene_count or membership_sha256(scenes) != contract.membership_sha256 or combined['scene_count'] != len(scenes) or combined['image_count'] != len(rows):
        raise ValueError('Components do not form the original full-development cohort')
    return rows, target_hash, scenes, expected_views, components


def _validate_unit(scene, view, source_row, native_row, unit, source_signature, rgb_root):
    if not verified_unit(unit, source_signature):
        raise ValueError('Selected source feature unit is missing')
    meta = read_json(unit / 'observations.json')
    if (meta['scene'], meta['route'], meta['view'], meta['rgb_sha256']) != (scene, 'automatic', view, source_row['rgb_sha256']) or native_row['rgb_sha256'] != source_row['rgb_sha256']:
        raise ValueError('RGB or scene/view lineage differs')
    path = rgb_root / source_row['rgb']
    if path.is_symlink() or not path.resolve().is_relative_to(rgb_root.resolve()) or digest_file(path) != source_row['rgb_sha256']:
        raise ValueError('Source RGB bytes or path differs')
    rgb = np.asarray(Image.open(path).convert('RGB'))
    if meta['image_shape'] != list(rgb.shape):
        raise ValueError('Source image coordinates differ')
    native = {}
    for obj in native_row['objects']:
        normalized = normalize_native_object(obj)
        source_id = _candidate_id(scene, view, 'automatic', normalized['candidate_id'])
        if source_id in native:
            raise ValueError('Native candidate hash collision or duplicate')
        native[source_id] = normalized
    source = {o['candidate_id']: o for o in source_row['objects']}
    if len(source) != len(source_row['objects']) or set(source) != set(native) or len(meta['objects']) != len(source) or {o['candidate_id'] for o in meta['objects']} != set(source):
        raise ValueError('Candidate crosswalk is incomplete or not one-to-one')
    if len({o['array_index'] for o in meta['objects']}) != len(meta['objects']):
        raise ValueError('Candidate tensor index alias')
    mapped = []; crosswalk = []
    with np.load(unit / 'features.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {f'{p}{o["array_index"]}' for o in meta['objects'] for p in ('p','g','v')}:
            raise ValueError('Unexpected or missing tensor keys')
        for item in meta['objects']:
            sid = item['candidate_id']; target = native[sid]; obj = source[sid]
            mask, oldmask = decode_binary_mask(target['mask']), decode_binary_mask(obj['mask'])
            raw_hash = sha256(oldmask.astype(np.uint8).tobytes()).hexdigest()
            if mask.shape != rgb.shape[:2] or not np.array_equal(mask, oldmask) or raw_hash != obj['mask_sha256'] or raw_hash != target['mask_sha256']:
                raise ValueError('Source and native decoded masks differ')
            crop = prepare_masked_crop(rgb, mask); valid, fallback = patch_policy(crop, 16, .70, fallback='all')
            expected = {'crop_sha256': crop.sha256, 'mask_sha256': raw_hash,
                'source_box_xyxy': list(crop.source_box_xyxy), 'padded_box_xyxy': list(crop.padded_box_xyxy),
                'grid_size': 16, 'foreground_patch_count': int(valid.sum()), 'fallback_all': fallback}
            if any(item.get(k) != value for k, value in expected.items()):
                raise ValueError('Crop transform or foreground policy differs')
            i = item['array_index']; p, g, v = arrays[f'p{i}'], arrays[f'g{i}'], arrays[f'v{i}']
            if p.shape != (256,384) or g.shape != (384,) or p.dtype != np.float32 or g.dtype != np.float32 or not np.isfinite(p).all() or not np.isfinite(g).all() or v.dtype != np.bool_ or not np.array_equal(v, valid):
                raise ValueError('Tensor dtype, coordinates or values differ from the CPU contract')
            mapped.append({**item, 'candidate_id': target['candidate_id']})
            crosswalk.append({'source_candidate_id': sid, 'native_candidate_id': target['candidate_id'], 'array_index': i,
                'normalized_mask_sha256': raw_hash, 'native_mask_sha256': target['source_mask_sha256'], 'crop_sha256': crop.sha256})
    return {**meta, 'objects': mapped}, crosswalk


def audit_remap(cfg):
    """Read actual sealed artifacts and write a reusable, label-free mapping seal."""
    source_root, features, output = (Path(cfg[k]) for k in ('source_region_root','source_feature_root','remap_root'))
    source_hash = digest_file(source_root / 'artifact_lock.json')
    if source_hash != cfg['expected_source_region_lock_sha256'] or digest_file(features / 'feature_lock.json') != cfg['expected_source_feature_lock_sha256']:
        raise ValueError('Explicit source artifact lock differs')
    preflight(source_root)
    source_binding, signature = _feature_binding(features, source_hash)
    lock = read_json(features / 'feature_lock.json')
    if lock['metadata']['binding_sha256'] != signature or lock['files']['run_binding.json'] != digest_file(features / 'run_binding.json'):
        raise ValueError('Source feature lock and binding differ')
    native, target_hash, target_scenes, target_views, components = _native_inventory(cfg)
    views = cfg.get('selected_views', target_views)
    if not views or len(views) != len(set(views)) or not set(views) <= set(target_views):
        raise ValueError('Explicit selected views are missing, duplicate or outside the frozen schedule')
    scenes = read_json(source_root / 'index.json')['scenes']
    if not set(scenes) <= set(target_scenes) or not set(scenes) <= set(source_binding['scenes']) or not set(views) <= set(source_binding['views']):
        raise ValueError('Source scenes/views are outside the declared feature/native cohort')
    binding = {'schema': SCHEMA, 'source_region_lock_sha256': source_hash,
        'source_feature_lock_sha256': digest_file(features / 'feature_lock.json'),
        'target_region_lock_sha256': target_hash, 'source_sha256': digest_file(__file__),
        'target_scenes': target_scenes, 'target_views': target_views, 'source_scenes': sorted(scenes),
        'selected_views': sorted(views), 'components': components}
    bind(output, binding)
    if (output / 'artifact_lock.json').exists():
        verify_seal(output)
        return read_json(output / 'receipt.json')
    units = []; total = 0; size = 0
    with terminal_status(output, '08N-native-feature-remap-audit'):
        for n, scene in enumerate(sorted(scenes), 1):
            records = [r for r in read_json(source_root / 'scenes' / f'{scene}.json') if r['route'] == 'automatic' and r['view'] in views]
            if len(records) != len(views) or {r['view'] for r in records} != set(views):
                raise ValueError('Selected comparison-region views differ')
            for row in sorted(records, key=lambda r:r['view']):
                view = row['view']; relative = f'views/{scene}/automatic-{view}'; unit = features / relative
                for name in ('artifact_lock.json', 'features.npz', 'observations.json'):
                    if lock['files'].get(f'{relative}/{name}') != digest_file(unit / name):
                        raise ValueError('Selected feature bytes differ from the source root seal')
                metadata, crosswalk = _validate_unit(scene, view, row, native[scene,view], unit, signature, source_root)
                units.append({'relative': relative, 'scene': scene, 'view': view,
                    'component': native[scene,view]['component'], 'features_sha256': digest_file(unit / 'features.npz'),
                    'observations_sha256': digest_file(unit / 'observations.json'),
                    'metadata': metadata, 'crosswalk': crosswalk})
                total += len(crosswalk); size += (unit / 'features.npz').stat().st_size
            if n == 1 or n % 10 == 0: print(f'Remap verified {n}/{len(scenes)} source scenes', flush=True)
        receipt = {'schema': SCHEMA, 'status':'verified', 'source_scene_count':len(scenes),
            'verified_units':len(units), 'verified_candidates':total, 'source_npz_bytes':size,
            'target_scene_denominator':len(target_scenes), 'target_unit_denominator':len(target_scenes)*len(target_views),
            'remaining_feature_units':len(target_scenes)*len(target_views)-len(units),
            'npz_reencoded':False, 'neural_inference_run':False, 'anomaly_labels_opened':False,
            'supplied_feature_units_used':0, 'selected_views':sorted(views),
            'rgb_proof':'08G RGB file bytes equal the separately sealed native image SHA; target RGB package not opened',
            'mask_proof':'historical packed digest and normalized uint8 digest verified against identical decoded masks',
            'source_feature_lock_sha256':binding['source_feature_lock_sha256'], 'target_region_lock_sha256':target_hash}
        atomic_json(output / 'mapping.json', {'schema':SCHEMA,'units':units})
        atomic_json(output / 'receipt.json', receipt)
        seal(output, [output / 'run_binding.json', output / 'mapping.json', output / 'receipt.json'])
    return receipt


def import_remapped(cfg):
    """Seed a fresh native feature directory without changing any feature bytes."""
    source, audit, target = (Path(cfg[k]) for k in ('source_feature_root','remap_root','feature_root'))
    verify_seal(audit); plan = read_json(audit / 'run_binding.json')
    if plan['source_sha256'] != digest_file(__file__) or digest_file(source / 'feature_lock.json') != plan['source_feature_lock_sha256']:
        raise ValueError('Remap implementation or source lock changed')
    binding = native_feature_binding(plan['target_region_lock_sha256'], plan['target_scenes'], plan['target_views'])
    signature = bind(target, binding); units = read_json(audit / 'mapping.json')['units']
    if (target / 'import_receipt.json').exists() and read_json(target / 'import_receipt.json')['status'] == 'import_complete':
        verify_seal(target, name='feature_shard_lock.json')
        for record in units:
            unit = target / record['relative']
            if not verified_unit(unit, signature) or digest_file(unit/'features.npz') != record['features_sha256'] or read_json(unit/'observations.json') != record['metadata']:
                raise ValueError('Completed remap import changed')
        receipt = read_json(target / 'import_receipt.json')
        if receipt['remap_lock_sha256'] != digest_file(audit / 'artifact_lock.json'):
            raise ValueError('Completed import belongs to a different audited remap')
        return receipt
    new = 0; limit = cfg.get('max_new_units')
    if limit is not None and (not isinstance(limit,int) or limit < 1): raise ValueError('max_new_units must be positive')
    with terminal_status(target, '08N-native-feature-import'):
        provenance = target / 'provenance' / '08n-remap'
        provenance.mkdir(parents=True, exist_ok=True)
        for name in ('artifact_lock.json','run_binding.json','mapping.json','receipt.json'):
            path = provenance / name
            if path.exists() and digest_file(path) != digest_file(audit / name):
                raise ValueError('Existing remap provenance differs')
            if not path.exists():
                temporary = path.with_suffix(path.suffix+'.partial')
                shutil.copyfile(audit/name,temporary);temporary.replace(path)
        verify_seal(provenance)
        for record in units:
            unit = target / record['relative']; original = source / record['relative']
            if verified_unit(unit, signature):
                if digest_file(unit / 'features.npz') != record['features_sha256'] or read_json(unit / 'observations.json') != record['metadata']:
                    raise ValueError('Existing import differs; never overwrite')
                continue
            if limit is not None and new >= limit: break
            if digest_file(original / 'features.npz') != record['features_sha256'] or digest_file(original / 'observations.json') != record['observations_sha256']:
                raise ValueError('Source bytes changed after remap audit')
            if shutil.disk_usage(target).free < (original / 'features.npz').stat().st_size + 128*1024**2:
                raise OSError('Insufficient free space for byte-preserving import')
            with atomic_unit(unit, signature) as temporary:
                shutil.copyfile(original / 'features.npz', temporary / 'features.npz')
                atomic_json(temporary / 'observations.json', record['metadata'])
                if digest_file(temporary / 'features.npz') != record['features_sha256']: raise ValueError('Tensor copy differs')
            new += 1
        completed = sum(verified_unit(target / r['relative'], signature) for r in units)
        receipt = {'status':'import_complete' if completed == len(units) else 'checkpointed', 'new_units':new,
            'imported_units':completed, 'planned_import_units':len(units), 'full_cohort_feature_lock_written':False,
            'remap_lock_sha256':digest_file(audit/'artifact_lock.json'), 'neural_inference_run':False, 'npz_reencoded':False}
        atomic_json(target / 'import_receipt.json', receipt)
        completed_records = [r for r in units if verified_unit(target / r['relative'], signature)]
        seal(target, [target / 'run_binding.json', *[p for p in provenance.iterdir() if p.is_file()],
            *[p for r in completed_records for p in (target / r['relative']).iterdir() if p.is_file()]],
            name='feature_shard_lock.json', metadata={'binding_sha256':signature, 'producer':'dinov2-small',
                'completed_units':completed, 'required_units':len(plan['target_scenes'])*len(plan['target_views']),
                'remap_lock_sha256':digest_file(audit / 'artifact_lock.json')})
    atomic_json(target / 'run_status.json', {'stage':'08N-native-feature-import','status':receipt['status']})
    return receipt
