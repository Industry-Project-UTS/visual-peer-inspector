"""Convert saved native regions and CPU DINO grids into the current cache.

Supported sources are the sealed08G comparison-region package or a native
08E/protected prediction_manifest plus its separately verified RGB package.
Old P1 scores and float16 globals are never substituted for full patch tensors.
This adapter does not read raw archives, anomaly labels, or evaluator packages.
"""
from pathlib import Path
import shutil
from hashlib import sha256
from uuid import uuid4
import numpy as np
from PIL import Image

from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.models.sam2 import decode_binary_mask, binary_mask_sha256
from visual_peer_inspector.representation.crops import CropConfiguration, prepare_masked_crop
from visual_peer_inspector.scoring.forward_patch import patch_policy
from visual_peer_inspector.scoring.peer_support import mask_shape_descriptor
from visual_peer_inspector.web.decision import component_scores
from visual_peer_inspector.web.encoder_worker import PINNED, compatible_runtime
from visual_peer_inspector.web.deployment import pipeline_config
from vpi_object_bank.extraction import model_factory, runtime_versions
from vpi_object_bank.protocol import EXTRACTION
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest, digest_file,
    read_json, seal, terminal_status, verified_unit, verify_seal)
from vpi_remaining_evaluation.cohort import CACHE_SCHEMA, _reject_labels, selected_inputs
from vpi_remaining_evaluation.gates import CONTRACTS
from .exposure import verify_exposure
from .contracts import VIEW_SETS, verify_current_freeze
from .official import OFFICIAL, verify_prerequisites, CACHE_SCHEMA as OFFICIAL_CACHE_SCHEMA

SCHEMA = 'vpi-current-native-cache-preparation/v1'
NEUTRAL_SCHEMA = 'vpi-current-neutral-correspondence/v1'
CROP_CONFIGURATION = CropConfiguration(
    input_size=EXTRACTION['input_size'],
    padding_fraction=EXTRACTION['padding_fraction'],
    background_rgb=tuple(EXTRACTION['background_rgb']),
    minimum_foreground_pixels=EXTRACTION['minimum_foreground_pixels'],
)
SMALL_MASK_REASON = 'mask_below_minimum_foreground'


def _gates(cfg, source_root):
    role = cfg['role']
    contract = {**CONTRACTS, **OFFICIAL}.get(role)
    if contract is None:
        raise ValueError('Unknown original or official cohort')
    if role in OFFICIAL:
        _, prerequisites = verify_prerequisites(cfg, source_root)
        freeze_hash = prerequisites['method_freeze_sha256']
    else:
        verify_exposure(role, cfg.get('exposure_root'))
        verify_current_freeze(cfg['method_freeze_root'], source_root)
        freeze_hash = digest_file(Path(cfg['method_freeze_root']) / 'method_freeze.json')
        if role == 'internal_lock':
            from .contracts import require_protocol_receipt
            require_protocol_receipt(cfg['calibration_receipt_root'], '09b', freeze_hash)
    return contract, freeze_hash


def native_regions(cfg, contract):
    """Whitelist geometry from actual historical formats; retain empty views."""
    fmt = cfg['region_format']
    if fmt == 'current-native-regions':
        from .native_regions import read_region_resource
        records, source_hash = read_region_resource(cfg['region_root'], cfg['rgb_root'], contract)
        if source_hash != cfg['expected_region_lock_sha256']:
            raise ValueError('Current region resource differs from its reviewed lock')
        return records, source_hash
    if fmt == 'native-development-components':
        from visual_peer_inspector.data.cohort_contract import DS1_SCREEN_CONTRACT, DS2_CONFIRM_CONTRACT, DEVELOPMENT_REMAINDER_CONTRACT
        components = {c.cohort_id: c for c in (DS1_SCREEN_CONTRACT, DEVELOPMENT_REMAINDER_CONTRACT, DS2_CONFIRM_CONTRACT)}
        specs = cfg['components']
        if cfg['role'] != 'development' or set(specs) != set(components):
            raise ValueError('Native development import requires the exact100+600+300 component resources')
        combined = read_json(cfg['full_development_lock'])
        source_hash = digest_file(cfg['full_development_lock'])
        if source_hash != cfg['expected_region_lock_sha256'] or combined.get('status') != 'verified' or combined.get('cohort_id') != contract.cohort_id or combined.get('scene_count') != contract.scene_count:
            raise ValueError('Combined native development lock differs')
        declared = {row['cohort_id']: row['lock_sha256'] for row in combined['component_locks']}
        if set(declared) != set(components):
            raise ValueError('Combined native lock omits a component')
        output = {}
        for name, component in components.items():
            subset, _ = native_regions({**cfg, **specs[name], 'region_format': 'native-prediction-manifest',
                'expected_region_lock_sha256': declared[name]}, component)
            record = read_json(Path(specs[name]['region_root']) / 'prediction_lock.json')
            if record['identity_sha256'] != combined['identity_sha256'] or set(output) & set(subset):
                raise ValueError('Native components have mixed vision identities or overlapping scenes')
            output.update(subset)
        scenes = sorted({scene for scene, _ in output})
        if len(scenes) != contract.scene_count or membership_sha256(scenes) != contract.membership_sha256:
            raise ValueError('Native components do not form the original1000-scene membership')
        return output, source_hash
    root = Path(cfg['region_root'])
    if fmt == 'comparison-regions':
        from visual_peer_inspector.workflows.scoring_comparison import preflight
        preflight(root)
        lock_path = root / 'artifact_lock.json'
        index = read_json(root / 'index.json')
        scenes = index['scenes']
        records = [row for scene in scenes for row in read_json(root / 'scenes' / f'{scene}.json') if row['route'] == 'automatic']
        rgb_root = root
    elif fmt == 'native-prediction-manifest':
        from visual_peer_inspector.workflows.frozen_development_pipeline import verify_prediction_lock
        from visual_peer_inspector.workflows.protected_cohort_inference import verify_protected_input
        lock_path = root / 'prediction_lock.json'
        lock = verify_prediction_lock(lock_path)
        if lock['cohort_id'] != contract.cohort_id:
            raise ValueError('Native prediction cohort differs')
        rgb_root = Path(cfg['rgb_root'])
        verify_protected_input(rgb_root, contract)
        records = []
        for line in (root / 'prediction_manifest.jsonl').read_text().splitlines():
            row = __import__('json').loads(line)
            _reject_labels(row)
            scene, view = row['opaque_scene_id'], row['view_id']
            if row.get('cohort_id') != contract.cohort_id or row['object_count'] != len(row['objects']):
                raise ValueError('Native record changes cohort or candidate count')
            records.append({'scene': scene, 'view': view, 'route': 'automatic',
                'rgb': f'media/{scene}/rgb/{view}.png', 'rgb_sha256': row['image_sha256'],
                'objects': [normalize_native_object(o) for o in row['objects']]})
        scenes = sorted({row['scene'] for row in records})
    else:
        raise ValueError('Unsupported region format; raw archives and unsealed SAM shards require a separate verified release')
    source_hash = digest_file(lock_path)
    if source_hash != cfg['expected_region_lock_sha256']:
        raise ValueError('Native region source differs from its explicitly reviewed lock')
    if len(scenes) != contract.scene_count or len(scenes) != len(set(scenes)) or membership_sha256(scenes) != (contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Native source does not contain the exact cohort membership')
    views = sorted({v for values in VIEW_SETS.values() for v in values})
    keys = [(row['scene'], row['view']) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != {(scene, view) for scene in scenes for view in views}:
        raise ValueError('Native source omits or duplicates a selected scene/view, including empty views')
    output = {}
    for row in records:
        scene, view = row['scene'], row['view']
        if any(Path(token).name != token or token in {'.', '..'} for token in (scene, view)):
            raise ValueError('Unsafe native scene/view token')
        path = rgb_root / row['rgb']
        if path.is_symlink() or not path.resolve().is_relative_to(rgb_root.resolve()) or digest_file(path) != row['rgb_sha256']:
            raise ValueError('Native RGB bytes or path differ from the region record')
        seen = set()
        with Image.open(path) as image:
            shape = (image.height, image.width)
        for obj in row['objects']:
            candidate = obj['candidate_id']
            if not isinstance(candidate, str) or not candidate or candidate in seen:
                raise ValueError('Native candidates are empty or duplicated')
            seen.add(candidate)
            mask = decode_binary_mask(obj['mask'])
            if mask.shape != shape:
                raise ValueError('Native mask dimensions differ from RGB')
            if sha256(mask.astype(np.uint8).tobytes()).hexdigest() != obj['mask_sha256']:
                raise ValueError('Native mask bytes differ')
            x1, y1, x2, y2 = obj['box']
            if not (0 <= x1 <= x2 <= shape[1] and 0 <= y1 <= y2 <= shape[0]):
                raise ValueError('Native box is outside RGB coordinates')
        output[scene, view] = {**row, '_rgb_path': path}
    return output, source_hash


def normalize_native_object(obj):
    """Verify the historical packed-mask digest before normalizing its schema."""
    mask = decode_binary_mask(obj['mask_rle'])
    if binary_mask_sha256(mask) != obj['mask_sha256']:
        raise ValueError('Historical native packed-mask hash differs')
    return {'candidate_id': obj['object_id'], 'box': obj['box_xyxy'], 'mask': obj['mask_rle'],
        'mask_sha256': sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
        'source_mask_sha256': obj['mask_sha256']}


def native_feature_binding(region_hash, scenes, views, *, runtime=None):
    """Mount-independent binding shared by imported and newly extracted units."""
    expected = pipeline_config()['encoder']
    return {'schema': 'vpi-current-native-features/v1', 'producer': 'dinov2-small',
        'model_input_lock_sha256': region_hash, 'extraction': EXTRACTION, 'device': 'cpu',
        'scenes': sorted(scenes), 'views': sorted(views),
        'model_files': {'model.safetensors': expected['sha256'], 'config.json': expected['config_sha256']},
        'runtime': dict(PINNED) if runtime is None else runtime,
        'source_sha256': digest_file(__file__)}


def _shard_scenes(scenes, cfg):
    count, index = cfg.get('shard_count', 1), cfg.get('shard_index', 0)
    if not isinstance(count, int) or not isinstance(index, int) or count < 1 or not 0 <= index < count:
        raise ValueError('shard_count/index must define a nonempty deterministic partition')
    selected = sorted(scenes)[index::count]
    if not selected:
        raise ValueError('Shard has no scenes')
    return selected


def _feature_collection(cfg, region_hash):
    """Use sealed read-only feature shards without staging their full tensors."""
    roots = [Path(p) for p in (cfg['feature_roots'] if 'feature_roots' in cfg else [cfg['feature_root']])]
    units, locks, binding = {}, [], None
    for root in roots:
        value, signature = _feature_binding(root, region_hash)
        if binding is not None and value != binding:
            raise ValueError('Feature shards have incompatible bindings')
        binding = value
        name = 'feature_lock.json' if (root / 'feature_lock.json').exists() else 'feature_shard_lock.json'
        lock = verify_seal(root, name=name)
        locks.append(digest_file(root / name))
        for relative in lock['files']:
            path = Path(relative)
            if len(path.parts) != 4 or path.parts[0] != 'views' or path.name != 'artifact_lock.json': continue
            scene, routeview = path.parts[1:3]
            if not routeview.startswith('automatic-'): continue
            key = scene, routeview[len('automatic-'):]
            unit = root / path.parent
            if not verified_unit(unit, signature): raise ValueError('Feature shard has missing unit')
            if key in units and digest_file(units[key] / 'artifact_lock.json') != digest_file(unit / 'artifact_lock.json'):
                raise ValueError('Feature shards conflict on an overlapping unit')
            units[key] = unit
    return binding, digest(binding), units, sorted(set(locks))


def _feature_binding(root, region_hash):
    binding = read_json(Path(root) / 'run_binding.json')
    if binding.get('schema') not in {'vpi-object-bank-features/v1', 'vpi-current-native-features/v1'}:
        raise ValueError('Unsupported native feature cache schema')
    if not binding.get('source') and not binding.get('source_sha256'):
        raise ValueError('Feature cache omits its extraction source binding')
    if binding.get('producer') != 'dinov2-small' or binding.get('device') != 'cpu' or binding.get('extraction') != EXTRACTION:
        raise ValueError('Features must use the fixed CPU FP32 DINO extraction contract')
    if binding.get('model_input_lock_sha256') != region_hash:
        raise ValueError('Feature cache belongs to different native regions')
    expected = pipeline_config()['encoder']
    for name, value in {'model.safetensors': expected['sha256'], 'config.json': expected['config_sha256']}.items():
        if binding.get('model_files', {}).get(name) != value:
            raise ValueError('Feature model/config hashes are missing or incompatible')
    if any(str(binding.get('runtime', {}).get(name, '')).split('+')[0] != version for name, version in PINNED.items()):
        raise ValueError('Feature cache runtime/precision differs from the CPU app contract')
    return binding, digest(binding)


def plan_native(cfg, source_root):
    contract, _ = _gates(cfg, source_root)
    component_missing = cfg.get('region_format') == 'native-development-components' and any(not Path(spec['region_root']).is_dir() for spec in cfg.get('components', {}).values())
    if component_missing or cfg.get('region_format') != 'native-development-components' and (not cfg.get('region_root') or not Path(cfg['region_root']).is_dir()):
        return {'schema': SCHEMA, 'role': cfg['role'], 'scene_count': contract.scene_count,
            'status': 'missing_region_inference', 'missing_la_sam': True, 'neural_inference_run': False,
            'next': 'Prepare only missing LocateAnything/SAM region outputs through a current-freeze adapter. The old P1 end-to-end09A runner is incompatible.'}
    rows, region_hash = native_regions(cfg, contract)
    if 'feature_roots' in cfg:
        _, _, units, _ = _feature_collection(cfg, region_hash)
        missing = [{'scene':s,'view':v,'candidate_count':len(row['objects'])} for (s,v),row in sorted(rows.items()) if (s,v) not in units]
        return {'schema':SCHEMA, 'role':cfg['role'], 'scene_count':contract.scene_count,
            'status':'missing_features' if missing else 'features_available', 'required_units':len(rows),
            'missing_feature_units':missing, 'missing_la_sam':False, 'neural_inference_run':False,
            'next':'extract_missing' if missing else 'assemble',
            'scope':'immutable mounted feature shards; extract into a separate writable feature_root'}
    root = Path(cfg['feature_root'])
    present = (root / 'run_binding.json').exists()
    if root.exists() and any(root.iterdir()) and not present:
        raise ValueError('Nonempty feature root has no binding')
    binding_hash = _feature_binding(root, region_hash)[1] if present else None
    missing = []
    for scene, view in sorted(rows):
        unit = root / 'views' / scene / f'automatic-{view}'
        if not present or not verified_unit(unit, binding_hash):
            missing.append({'scene': scene, 'view': view, 'candidate_count': len(rows[scene, view]['objects'])})
    size = _estimated_bytes(rows)
    return {'schema': SCHEMA, 'role': cfg['role'], 'scene_count': contract.scene_count,
        'status': 'missing_features' if missing else 'features_available',
        'projected_cache_and_checkpoint_bytes': size,
        'configured_output_limit_bytes': int(cfg.get('maximum_output_bytes', 15 * 1024**3)),
        'within_configured_output_limit': size <= int(cfg.get('maximum_output_bytes', 15 * 1024**3)),
        'required_units': len(rows), 'missing_feature_units': missing,
        'missing_la_sam': False, 'neural_inference_run': False,
        'next': 'extract_missing' if missing else 'assemble',
        'scope': 'saved complete LA/SAM regions required; missing regions stop before DINO'}


def _estimated_bytes(regions, feature_units=None):
    # A full256x384 FP32 bank upper bound is deliberately conservative. The
    # checkpoint and released copies coexist; raw grids stay in the input cache.
    tensors = sum(len(row['objects']) * 256 * 384 * 4 for row in regions.values()) if feature_units is None else sum(
        sum(o['foreground_patch_count'] * 384 * 4 for o in read_json(feature_units[key] / 'observations.json')['objects'])
        for key in regions)
    pixels = 0
    for row in regions.values():
        with Image.open(row['_rgb_path']) as image:
            pixels += image.width * image.height * 3
    return 2 * (tensors + pixels) + len(regions) * 128 * 1024


def _prepare_feature_objects(rgb, objects):
    """Split encoder-ready objects from masks rejected by the frozen crop contract.

    A sub-threshold SAM mask is an unusable candidate observation, not a fatal
    scene error.  Keep a typed rejection record so candidate loss remains
    visible in later failure accounting and visual evidence.
    """
    prepared, rejected = [], []
    for obj in objects:
        mask = decode_binary_mask(obj['mask'])
        foreground_pixels = int(mask.sum())
        if foreground_pixels < CROP_CONFIGURATION.minimum_foreground_pixels:
            rejected.append({
                'candidate_id': obj['candidate_id'],
                'reason': SMALL_MASK_REASON,
                'foreground_pixels': foreground_pixels,
                'minimum_foreground_pixels': CROP_CONFIGURATION.minimum_foreground_pixels,
                'mask_sha256': obj['mask_sha256'],
            })
            continue
        prepared.append((obj, prepare_masked_crop(
            rgb, mask, configuration=CROP_CONFIGURATION)))
    return prepared, rejected


def _validated_feature_partition(meta, candidates):
    """Validate that encoded and rejected records cover every source candidate."""
    objects = meta.get('objects')
    rejected = meta.get('rejected_objects', [])
    if not isinstance(objects, list) or not isinstance(rejected, list):
        raise ValueError('Feature candidate partition is invalid')
    encoded_ids = [item.get('candidate_id') for item in objects]
    rejected_ids = [item.get('candidate_id') for item in rejected]
    if (len(encoded_ids) != len(set(encoded_ids))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(encoded_ids) & set(rejected_ids)
            or set(encoded_ids) | set(rejected_ids) != set(candidates)):
        raise ValueError('Features must encode or explicitly reject every native candidate exactly once')
    expected_rejection_fields = {
        'candidate_id', 'reason', 'foreground_pixels',
        'minimum_foreground_pixels', 'mask_sha256',
    }
    for item in rejected:
        candidate_id = item['candidate_id']
        obj = candidates[candidate_id]
        foreground_pixels = int(decode_binary_mask(obj['mask']).sum())
        if (set(item) != expected_rejection_fields
                or item['reason'] != SMALL_MASK_REASON
                or item['foreground_pixels'] != foreground_pixels
                or item['minimum_foreground_pixels'] != CROP_CONFIGURATION.minimum_foreground_pixels
                or item['mask_sha256'] != obj['mask_sha256']
                or foreground_pixels >= CROP_CONFIGURATION.minimum_foreground_pixels):
            raise ValueError('Feature rejection record differs from the frozen crop contract')
    return objects, rejected


def _assembly_budget(regions, root, cfg, feature_units=None):
    bound = _estimated_bytes(regions, feature_units)
    limit = int(cfg.get('maximum_output_bytes', 15 * 1024**3))
    if bound > limit:
        raise OSError(f'Projected cache/checkpoint upper bound {bound} exceeds maximum_output_bytes={limit}; allocate a larger retained output volume or reviewed cohort-preserving shards')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < bound:
        raise OSError('Free disk space is below the conservative complete-cohort output budget')
    return {'conservative_output_bytes': bound, 'maximum_output_bytes': limit,
        'tensor_storage': 'compressed FP32 foreground NPZ; no full-grid JSON',
        'copies_budgeted': 2, 'full_grid_features_copied': False}


def extract_missing(cfg, source_root):
    """Extract only missing automatic-region units; no LA/SAM or scorer rerun."""
    contract, _ = _gates(cfg, source_root)
    rows, region_hash = native_regions(cfg, contract)
    root, model = Path(cfg['feature_root']), Path(cfg['model_source'])
    expected = pipeline_config()['encoder']
    if not compatible_runtime() or any(digest_file(model / filename) != expected[key] for filename, key in [('model.safetensors', 'sha256'), ('config.json', 'config_sha256')]):
        raise ValueError('Run missing-feature extraction in the separately pinned CPU DINO environment with verified local weights')
    if (root / 'run_binding.json').exists():
        binding, signature = _feature_binding(root, region_hash)
        if binding.get('schema') == 'vpi-current-native-features/v1' and binding.get('source_sha256') != digest_file(__file__):
            raise ValueError('Missing-feature resume source changed')
        if binding.get('schema') != 'vpi-current-native-features/v1':
            # Completed08N caches are reusable; modifying their incomplete run would
            # falsely attribute new extraction to the original source binding.
            if not (root / 'feature_lock.json').exists():
                raise ValueError('Preserve partial08N output; finish it with its exact original extractor or use a fresh native feature directory')
    else:
        binding = native_feature_binding(region_hash, {s for s, _ in rows}, {v for _, v in rows})
        signature = bind(root, binding)
    external = {}
    if cfg.get('feature_roots'):
        external_binding, _, external, _ = _feature_collection({'feature_roots':cfg['feature_roots']}, region_hash)
        if external_binding != binding: raise ValueError('Mounted reuse feature shards differ from extraction binding')
    if (root / 'feature_lock.json').exists():
        verify_seal(root, name='feature_lock.json')
        if plan_native(cfg, source_root)['missing_feature_units']:
            raise ValueError('Sealed feature cache is incomplete')
        return {'status': 'features_complete', 'extracted_units': 0, 'neural_inference_run': False}
    encoder, computed = None, 0
    root.mkdir(parents=True, exist_ok=True)
    selected = set(_shard_scenes({s for s, _ in rows}, cfg))
    planned = [(key, row) for key, row in sorted(rows.items()) if key[0] in selected and not verified_unit(root / 'views' / key[0] / f'automatic-{key[1]}', signature) and key not in external]
    maximum = cfg.get('max_new_units')
    if maximum is not None and (not isinstance(maximum, int) or maximum < 1): raise ValueError('max_new_units must be positive')
    planned = planned[:maximum]
    needed = sum(len(row['objects']) * (256*384*4 + 384*4 + 256) + 16384 for _, row in planned)
    limit = int(cfg.get('maximum_output_bytes', 15*1024**3))
    if needed > limit or shutil.disk_usage(root).free < needed + 128 * 1024**2:
        raise OSError('Insufficient storage for missing full-grid feature units')
    with terminal_status(root, 'current-native-DINO-features'):
        for (scene, view), row in planned:
            unit = root / 'views' / scene / f'automatic-{view}'
            if verified_unit(unit, signature):
                continue
            rgb = np.asarray(Image.open(row['_rgb_path']).convert('RGB'))
            source_objects = sorted(row['objects'], key=lambda o: o['candidate_id'])
            prepared, rejected = _prepare_feature_objects(rgb, source_objects)
            objects = [item[0] for item in prepared]
            crops = [item[1] for item in prepared]
            arrays, metadata = {}, []
            if crops and encoder is None:
                encoder = model_factory(device='cpu', model_source=model)
            for offset in range(0, len(crops), 4):
                batch = encoder.encode([crop.rgb for crop in crops[offset:offset+4]])
                if batch.global_features.shape != (len(crops[offset:offset+4]), 384) or len(batch.patch_features) != len(crops[offset:offset+4]):
                    raise ValueError('Encoder batch schema differs')
                for local, tokens in enumerate(batch.patch_features):
                    index = offset + local
                    crop, obj = crops[index], objects[index]
                    patches, global_ = np.asarray(tokens), np.asarray(batch.global_features[local])
                    if patches.shape != (256, 384) or patches.dtype != np.float32 or global_.dtype != np.float32 or not np.isfinite(patches).all() or not np.isfinite(global_).all():
                        raise ValueError('Encoder must emit finite FP32 full grids and globals')
                    valid, fallback = patch_policy(crop, 16, .70, fallback='all')
                    arrays.update({f'p{index}': patches, f'g{index}': global_, f'v{index}': valid})
                    metadata.append({'candidate_id': obj['candidate_id'], 'array_index': index,
                        'crop_sha256': crop.sha256, 'mask_sha256': obj['mask_sha256'],
                        'source_box_xyxy': list(crop.source_box_xyxy), 'padded_box_xyxy': list(crop.padded_box_xyxy),
                        'grid_size': 16, 'foreground_patch_count': int(valid.sum()), 'fallback_all': fallback})
            with atomic_unit(unit, signature) as temporary:
                np.savez_compressed(temporary / 'features.npz', **arrays)
                atomic_json(temporary / 'observations.json', {'scene': scene, 'route': 'automatic', 'view': view,
                    'rgb_sha256': row['rgb_sha256'], 'image_shape': list(rgb.shape),
                    'objects': metadata, 'rejected_objects': rejected})
            computed += 1
            if computed == 1 or computed % 10 == 0:
                print(f'CPU feature units newly completed: {computed}', flush=True)
        completed = [(s,v) for s,v in sorted(rows) if verified_unit(root / 'views' / s / f'automatic-{v}', signature)]
        complete = len(completed) == len(rows)
        files = [root / 'run_binding.json', *[p for s,v in completed for p in (root / 'views' / s / f'automatic-{v}').iterdir() if p.is_file()]]
        seal(root, files, name='feature_lock.json' if complete else 'feature_shard_lock.json',
            metadata={'binding_sha256': signature, 'producer': 'dinov2-small', 'completed_units':len(completed), 'required_units':len(rows)})
    available = set(completed) | set(external)
    result = {'status': 'features_complete' if complete else 'checkpointed', 'extracted_units': computed,
        'completed_units':len(completed), 'required_units':len(rows), 'available_selected_units':sum(key in available for key in rows if key[0] in selected),
        'required_selected_units':sum(key[0] in selected for key in rows), 'neural_inference_run': encoder is not None}
    atomic_json(root / 'run_status.json', result)
    return result


def prepare_shard(cfg, source_root):
    """One bounded CPU job: reuse/complete its feature scenes, then assemble them."""
    result = extract_missing(cfg, source_root)
    if result.get('available_selected_units', result.get('completed_units', 0)) < result.get('required_selected_units', 0):
        return {'status':'checkpointed', 'features':result, 'cache_assembled':False}
    roots = sorted(set([str(cfg['feature_root']), *map(str,cfg.get('feature_roots',[]))]))
    cache = assemble_native({**cfg,'feature_roots':roots}, source_root)
    return {'status':cache['status'], 'features':result, 'cache':cache}


def assemble_native(cfg, source_root):
    """Regenerate current scores from native grids and export a model-only cache."""
    contract, freeze_hash = _gates(cfg, source_root)
    regions, region_hash = native_regions(cfg, contract)
    feature_binding, feature_signature, feature_units, feature_locks = _feature_collection(cfg, region_hash)
    selected_scenes = _shard_scenes({s for s, _ in regions}, cfg)
    needed_regions = {key: row for key, row in regions.items() if key[0] in set(selected_scenes)}
    if not set(needed_regions) <= set(feature_units):
        raise ValueError('Selected assembly shard has missing features; extract/import its units first')
    neutral = Path(cfg['neutral_root']); neutral_lock = verify_seal(neutral)
    if set(neutral_lock['files']) != {'neutral_correspondence.json'}:
        raise ValueError('Neutral identity package must contain only its reviewed map')
    mapping = read_json(neutral / 'neutral_correspondence.json'); _reject_labels(mapping)
    if set(mapping) != {'schema', 'region_lock_sha256', 'provenance', 'observations'} or mapping['schema'] != NEUTRAL_SCHEMA or mapping['region_lock_sha256'] != region_hash or mapping['provenance'] not in {'dataset_reference', 'user_confirmed'}:
        raise ValueError('Neutral identity schema, source or provenance differs')
    identities = {}
    for item in mapping['observations']:
        if set(item) != {'scene', 'view_id', 'candidate_id', 'object_id'}:
            raise ValueError('Unexpected neutral identity fields')
        key = (item['scene'], item['view_id'], item['candidate_id'])
        if key in identities or item['object_id'] is not None and (not isinstance(item['object_id'], str) or not item['object_id']):
            raise ValueError('Duplicate or invalid neutral identity')
        identities[key] = item['object_id']
    expected = {(s, v, o['candidate_id']) for (s, v), row in regions.items() for o in row['objects']}
    if set(identities) != expected:
        raise ValueError('Neutral identity map must explicitly cover every candidate, including null identities')
    root = Path(cfg['output_root'])
    binding = {'schema': SCHEMA, 'role': cfg['role'], 'region_lock_sha256': region_hash,
        'feature_binding_sha256': feature_signature, 'neutral_lock_sha256': digest_file(neutral / 'artifact_lock.json'),
        'method_freeze_sha256': freeze_hash, 'source_sha256': digest_file(__file__)}
    signature = bind(root, {**binding, 'assembly_scenes':selected_scenes, 'assembly_feature_locks':feature_locks})
    receipt = {'status': 'cache_complete', 'scene_count': contract.scene_count, 'neural_inference_run': False,
        'anomaly_labels_opened': False, 'component_scores_regenerated': True,
        'feature_precision': 'CPU FP32', 'evaluator_adapter': 'separate typed truth conversion required after predictions'}
    if (root / 'cache_shard_lock.json').exists():
        verify_seal(root, name='cache_shard_lock.json'); verify_seal(root, name='assembly_lock.json')
        manifest = read_json(root / 'shard_manifest.json')
        if manifest['native_source_binding'] != binding or manifest['scenes'] != selected_scenes or manifest['native_feature_locks'] != feature_locks:
            raise ValueError('Completed shard provenance or partition differs')
        return read_json(root / 'preparation_receipt.json')
    if (root / 'artifact_lock.json').exists():
        verify_seal(root)
        if read_json(root / 'cohort_manifest.json').get('native_source_binding') != binding:
            raise ValueError('Completed cache provenance differs')
        if (root / 'assembly_lock.json').exists():
            verify_seal(root, name='assembly_lock.json')
        else:
            if read_json(root / 'preparation_receipt.json') != receipt:
                raise ValueError('Interrupted assembly receipt differs from sealed model cache')
            seal(root, [root / 'run_binding.json', root / 'preparation_receipt.json', root / 'artifact_lock.json'], name='assembly_lock.json')
        return read_json(root / 'preparation_receipt.json')
    scenes = selected_scenes; views = sorted({v for _, v in regions})
    budget = _assembly_budget(needed_regions, root, cfg, feature_units)
    atomic_json(root / 'storage_plan.json', budget)
    computed = 0
    with terminal_status(root, 'current-native-cache-assembly'):
        for number, scene in enumerate(scenes, 1):
            unit = root / 'checkpoints' / scene
            if not verified_unit(unit, signature):
                if cfg.get('max_new_scenes') is not None and computed >= int(cfg['max_new_scenes']):
                    break
                records, bank, evidence = [], [], {}
                with atomic_unit(unit, signature) as temporary:
                    for view in views:
                        row = regions[scene, view]; rgb = np.asarray(Image.open(row['_rgb_path']).convert('RGB'))
                        feature_unit = feature_units[scene, view]
                        if not verified_unit(feature_unit, feature_signature):
                            raise FileNotFoundError('Missing feature unit; run plan and explicit extraction first')
                        meta = read_json(feature_unit / 'observations.json')
                        if (meta['scene'], meta['view'], meta['route'], meta['rgb_sha256']) != (scene, view, 'automatic', row['rgb_sha256']):
                            raise ValueError('Feature observation provenance differs')
                        candidates = {o['candidate_id']: o for o in row['objects']}
                        encoded, rejected = _validated_feature_partition(meta, candidates)
                        if len({o['array_index'] for o in encoded}) != len(encoded):
                            raise ValueError('Feature candidates alias the same tensor index')
                        geometry = {}; seen_ids = set(); foreground = {}
                        for candidate_id in sorted(candidates):
                            identity = identities[scene, view, candidate_id]
                            if identity is not None and identity in seen_ids:
                                raise ValueError('A physical identity appears twice in one view')
                            seen_ids.add(identity)
                        for item in rejected:
                            candidate_id = item['candidate_id']; obj = candidates[candidate_id]
                            geometry[candidate_id] = {'mask': obj['mask'], 'box': obj['box'],
                                'feature_status': 'rejected', 'feature_failure': item['reason'],
                                'foreground_pixels': item['foreground_pixels']}
                        with np.load(feature_unit / 'features.npz', allow_pickle=False) as arrays:
                            expected_arrays = {f'{prefix}{o["array_index"]}' for o in encoded for prefix in ('p', 'g', 'v')}
                            if set(arrays.files) != expected_arrays:
                                raise ValueError('Feature arrays are missing or unexpected')
                            for item in encoded:
                                c = item['candidate_id']; obj = candidates[c]; index = item['array_index']
                                crop = prepare_masked_crop(rgb, decode_binary_mask(obj['mask']), configuration=CROP_CONFIGURATION)
                                patches, global_ = arrays[f'p{index}'], arrays[f'g{index}']
                                valid70, fallback = patch_policy(crop, 16, .70, fallback='all')
                                valid50, _ = patch_policy(crop, 16, .50, fallback='largest')
                                if patches.shape != (256, 384) or global_.shape != (384,) or patches.dtype != np.float32 or global_.dtype != np.float32 or not np.isfinite(patches).all() or not np.isfinite(global_).all():
                                    raise ValueError('Missing, corrupt or mixed-precision full DINO features')
                                if item['crop_sha256'] != crop.sha256 or item['mask_sha256'] != obj['mask_sha256'] or item['padded_box_xyxy'] != list(crop.padded_box_xyxy) or not np.array_equal(arrays[f'v{index}'], valid70):
                                    raise ValueError('Feature crop/mask transform differs')
                                identity = identities[scene, view, c]
                                records.append({'view_id': view, 'candidate_id': c, 'embedding': global_.tolist(),
                                    'patches': patches, 'valid50': valid50, 'valid70': valid70, 'geometry': mask_shape_descriptor(crop.foreground)})
                                foreground[f'p{index}'] = patches[valid70.ravel()]
                                bank.append({'view_id': view, 'candidate_id': c, 'object_id': identity,
                                    'patches_file': f'tensors/{scene}/{view}.npz', 'patches_key': f'p{index}'})
                                geometry[c] = {'mask': obj['mask'], 'box': obj['box'], 'valid70': valid70.tolist(),
                                    'crop_sha256': crop.sha256, 'padded_box_xyxy': list(crop.padded_box_xyxy)}
                        media = f'media/{scene}/{view}.png'
                        Image.fromarray(rgb).save(temporary / f'{view}.png')
                        np.savez_compressed(temporary / f'{view}.npz', **foreground)
                        evidence[view] = {'rgb_relative': media, 'candidates': geometry}
                    scores, _ = component_scores(records)
                    for view in views:
                        scores.setdefault(view, {'objects': []})
                    payload = {'scene': scene, 'correspondence_provenance': mapping['provenance'],
                        'score_rows': scores, 'bank_observations': bank, 'evidence_views': evidence,
                        'invalid_candidate_observations': sum(
                            item.get('feature_status') == 'rejected'
                            for view in evidence.values() for item in view['candidates'].values())}
                    atomic_json(temporary / 'scene.json', payload)
                computed += 1
            (root / 'scenes').mkdir(exist_ok=True); (root / 'media' / scene).mkdir(parents=True, exist_ok=True)
            (root / 'tensors' / scene).mkdir(parents=True, exist_ok=True)
            for source, target in [(unit / 'scene.json', root / 'scenes' / f'{scene}.json'), *[(unit / f'{v}.png', root / 'media' / scene / f'{v}.png') for v in views], *[(unit / f'{v}.npz', root / 'tensors' / scene / f'{v}.npz') for v in views]]:
                if target.exists() and digest_file(target) != digest_file(source):
                    raise ValueError('Existing assembled output changed; do not overwrite')
                if not target.exists():
                    pending = target.with_name(target.name + '.partial')
                    if pending.exists(): pending.rename(target.with_name(target.name + '.interrupted-' + uuid4().hex[:8]))
                    shutil.copyfile(source, pending)
                    pending.replace(target)
            if number == 1 or number % 25 == 0: print(f'Native cache scenes assembled: {number}/{len(scenes)}', flush=True)
        completed_scenes = [scene for scene in scenes if verified_unit(root / 'checkpoints' / scene, signature)]
        if len(completed_scenes) != len(scenes):
            partial = {'status': 'checkpointed', 'new_scenes': computed, 'sealed': False,
                'completed_shard_scenes':len(completed_scenes), 'required_shard_scenes':len(scenes),
                'final_scene_denominator':contract.scene_count}
            atomic_json(root / 'partial_receipt.json', partial)
        else:
            _seal_assembly(root, cfg, contract, freeze_hash, binding, feature_binding, mapping['provenance'], scenes, receipt, feature_locks)
    if len(completed_scenes) != len(scenes):
        atomic_json(root / 'run_status.json', partial)
        return partial
    return read_json(root / 'preparation_receipt.json')


def _seal_assembly(root, cfg, contract, freeze_hash, binding, feature_binding, provenance, scenes, receipt, feature_locks):
        manifest = {'schema': OFFICIAL_CACHE_SCHEMA if cfg['role'] in OFFICIAL else CACHE_SCHEMA,
            'role': cfg['role'], 'scenes': scenes, 'method_freeze_sha256': freeze_hash,
            'correspondence_provenance': provenance,
            **{key: getattr(contract, key) for key in ('cohort_id', 'membership_sha256', 'scene_count', 'asset_group_count')},
            'native_source_binding': binding, 'feature_runtime': feature_binding['runtime'], 'native_feature_locks':feature_locks}
        sharded = len(scenes) != contract.scene_count
        manifest_name = 'shard_manifest.json' if sharded else 'cohort_manifest.json'
        if sharded:
            manifest['schema'] = 'vpi-current-cache-shard/v1'
            manifest['full_cohort_schema'] = OFFICIAL_CACHE_SCHEMA if cfg['role'] in OFFICIAL else CACHE_SCHEMA
            receipt = {**receipt, 'status':'shard_complete', 'scene_count':len(scenes), 'final_scene_denominator':contract.scene_count}
        atomic_json(root / manifest_name, manifest); atomic_json(root / 'preparation_receipt.json', receipt)
        # Only model-facing files enter this seal; checkpoints and status stay private.
        lock_name = 'cache_shard_lock.json' if sharded else 'artifact_lock.json'
        seal(root, [root / manifest_name, *(root / 'scenes').glob('*.json'), *(root / 'media').rglob('*.png'), *(root / 'tensors').rglob('*.npz')], name=lock_name)
        seal(root, [root / 'run_binding.json', root / 'preparation_receipt.json', root / lock_name], name='assembly_lock.json')
