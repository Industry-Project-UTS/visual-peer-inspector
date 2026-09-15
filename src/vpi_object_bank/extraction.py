"""Extract only missing fixed DINO patch outputs from already prepared masks."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import os
import time

import numpy as np
from PIL import Image
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import CropConfiguration, prepare_masked_crop
from visual_peer_inspector.scoring.forward_patch import patch_policy
from visual_peer_inspector.workflows.comparison_io import safe_path
from .protocol import EXTRACTION, REVISION
from .storage import (atomic_json, atomic_unit, bind, digest_file, free_space, progress,
                      read_json, seal, source_hashes, terminal_status, verified_unit, verify_seal)

CROP = CropConfiguration()
EXTRACTION_SOURCES = [
    'src/vpi_object_bank/extraction.py', 'src/vpi_object_bank/storage.py',
    'src/vpi_object_bank/protocol.py', 'src/visual_peer_inspector/workflows/comparison_io.py',
    'src/visual_peer_inspector/scoring/peer.py',
    'src/visual_peer_inspector/models/visual_encoder.py',
    'src/visual_peer_inspector/representation/crops.py',
    'src/visual_peer_inspector/scoring/forward_patch.py',
    'src/visual_peer_inspector/models/sam2.py',
]


def runtime_versions():
    output = {}
    for package in ('torch', 'transformers', 'numpy', 'pillow', 'safetensors'):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = 'unavailable'
    return output


def model_factory(*, device, model_source, expected_revision=REVISION):
    from visual_peer_inspector.models.visual_encoder import FrozenVisualEncoder, specification_by_key
    if expected_revision != REVISION:
        raise ValueError('Only the established immutable DINOv2-small revision is permitted')
    source = Path(model_source).resolve()
    if not (source / 'config.json').is_file():
        raise FileNotFoundError('A downloaded and verified DINO snapshot is required')
    import torch
    torch.manual_seed(32040)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    if device not in ('cpu', 'mps', 'cuda:0'):
        raise ValueError('Unsupported device')
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    print({'event': 'encoder_load_start', 'device': device, 'revision': REVISION}, flush=True)
    encoder = FrozenVisualEncoder(specification_by_key('dinov2_small'), revision=REVISION,
                                  device=device, model_source=source)
    print({'event': 'encoder_ready', 'device': device}, flush=True)
    return encoder


def snapshot_files(model_source):
    source = Path(model_source).resolve()
    files = sorted(p for p in source.iterdir() if p.is_file() and
                   (p.suffix in ('.safetensors', '.bin') or p.name in ('config.json', 'model.safetensors.index.json')))
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('Model snapshot has no weights')
    return {p.name: digest_file(p) for p in files}


def extract_features(model_root, prepared_root, output_root, source_root, *, device,
                     model_source=None, encoder_factory=None, producer='dinov2-small',
                     max_units=None):
    model_root, prepared_root, output_root = map(Path, (model_root, prepared_root, output_root))
    verify_seal(prepared_root)
    manifest = read_json(prepared_root / 'input_manifest.json')
    verify_seal(model_root)
    # The trusted preparer records this hash; do not substitute similarly named inputs.
    expected = manifest.get('model_input_lock_sha256')
    if not expected or digest_file(model_root / 'artifact_lock.json') != expected:
        raise ValueError('Model input and neutral registry are from different packages')
    scene_ids = manifest['scenes']
    views = sorted({v for values in manifest['view_sets'].values() for v in values})
    weights = snapshot_files(model_source) if model_source is not None else {'producer': producer}
    if producer == 'dinov2-small' and model_source is not None and weights.get('model.safetensors') != 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1':
        raise ValueError('Model bytes differ from the verified immutable DINO snapshot')
    binding = {'schema': 'vpi-object-bank-features/v1', 'producer': producer,
               'prepared_lock_sha256': digest_file(prepared_root / 'artifact_lock.json'),
               'model_input_lock_sha256': digest_file(model_root / 'artifact_lock.json'),
               'scenes': scene_ids, 'views': views, 'extraction': EXTRACTION,
               'device': device, 'model_files': weights, 'runtime': runtime_versions(),
               'source': source_hashes(source_root, EXTRACTION_SOURCES)}
    binding_hash = bind(output_root, binding)
    if (output_root / 'feature_lock.json').exists():
        lock = verify_seal(output_root, name='feature_lock.json')
        if lock['metadata']['binding_sha256'] != binding_hash:
            raise ValueError('Feature lock belongs to another extraction binding')
        for scene in scene_ids:
            for route in ('supplied', 'automatic'):
                for view in views:
                    if not verified_unit(output_root / 'views' / scene / f'{route}-{view}', binding_hash):
                        raise ValueError('Sealed feature unit is missing')
        return {'completed_units': lock['metadata']['completed_units'], 'total_units': lock['metadata']['completed_units'],
                'extracted_units': 0, 'elapsed_seconds': 0.0, 'sealed': True}
    expected_units = [(scene, route, view) for scene in scene_ids
                      for route in ('supplied', 'automatic') for view in views]
    completed = 0
    missing = []
    for scene, route, view in expected_units:
        unit = output_root / 'views' / scene / f'{route}-{view}'
        if verified_unit(unit, binding_hash):
            completed += 1
        else:
            missing.append((scene, route, view))
    # Budget is conservative for float32 256x384 tokens; no source outputs are deleted.
    free_space(output_root, max(128 * 1024**2, len(missing) * 3 * 1024**2))
    start = time.monotonic()
    encoder = None
    extracted = 0
    with terminal_status(output_root, '08N-patch-extraction'):
        progress(output_root, completed=completed, total=len(expected_units), extracted=0,
                 reused=completed, elapsed_seconds=0)
        current_scene = None
        for scene, route, view in missing:
            if max_units is not None and extracted >= max_units:
                break
            if scene != current_scene:
                model_rows = {(r['route'], r['view']): r for r in read_json(model_root / 'scenes' / f'{scene}.json')}
                current_scene = scene
            row = model_rows[(route, view)]
            image = np.asarray(Image.open(safe_path(model_root, row['rgb'])).convert('RGB'))
            objects = sorted(row['objects'], key=lambda obj: obj['candidate_id'])
            crops = [prepare_masked_crop(image, decode_binary_mask(obj['mask']), configuration=CROP) for obj in objects]
            if objects and encoder is None:
                encoder = encoder_factory() if encoder_factory else model_factory(device=device, model_source=model_source)
            unit_start = time.monotonic()
            arrays, metadata = {}, []
            for offset in range(0, len(crops), EXTRACTION['batch_size']):
                batch = encoder.encode([crop.rgb for crop in crops[offset:offset+EXTRACTION['batch_size']]])
                batch_count = len(crops[offset:offset+EXTRACTION['batch_size']])
                if len(batch.patch_features) != batch_count or batch.global_features.shape != (batch_count, batch.feature_dimension):
                    raise ValueError('Encoder batch length or global shape does not match input objects')
                if not batch.patch_grid_size or any(np.asarray(x).shape != (batch.patch_grid_size**2, batch.feature_dimension) for x in batch.patch_features):
                    raise ValueError('Encoder patch tensor shape does not match its declared schema')
                if producer == 'dinov2-small' and (batch.feature_dimension != 384 or batch.patch_grid_size != 16):
                    raise ValueError('DINO patch schema does not match fixed feature contract')
                for local, tokens in enumerate(batch.patch_features):
                    index = offset + local
                    crop, obj = crops[index], objects[index]
                    valid, fallback = patch_policy(crop, batch.patch_grid_size, .70, fallback='all')
                    patches = np.asarray(tokens, dtype=np.float32)
                    global_feature = np.asarray(batch.global_features[local], dtype=np.float32)
                    if not np.isfinite(patches).all() or not np.isfinite(global_feature).all():
                        raise ValueError('Nonfinite neural output')
                    arrays[f'p{index}'] = patches
                    arrays[f'v{index}'] = valid
                    arrays[f'g{index}'] = global_feature
                    metadata.append({'candidate_id': obj['candidate_id'], 'array_index': index,
                                     'crop_sha256': crop.sha256, 'mask_sha256': obj['mask_sha256'],
                                     'source_box_xyxy': list(crop.source_box_xyxy),
                                     'padded_box_xyxy': list(crop.padded_box_xyxy),
                                     'grid_size': batch.patch_grid_size,
                                     'foreground_patch_count': int(valid.sum()), 'fallback_all': fallback})
            unit = output_root / 'views' / scene / f'{route}-{view}'
            with atomic_unit(unit, binding_hash) as temporary:
                with (temporary / 'features.npz').open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                atomic_json(temporary / 'observations.json', {
                    'scene': scene, 'route': route, 'view': view, 'rgb_sha256': row['rgb_sha256'],
                    'image_shape': list(image.shape), 'objects': metadata,
                    'elapsed_seconds': time.monotonic() - unit_start,
                })
            completed += 1
            extracted += 1
            if extracted == 1 or extracted % 10 == 0 or completed == len(expected_units):
                progress(output_root, completed=completed, total=len(expected_units), extracted=extracted,
                         reused=completed-extracted, elapsed_seconds=time.monotonic()-start)
        if completed == len(expected_units):
            paths = [output_root / 'run_binding.json', output_root / 'progress.json']
            paths += [p for p in (output_root / 'views').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p)]
            seal(output_root, paths, name='feature_lock.json', metadata={
                'scenes': scene_ids, 'views': views, 'completed_units': completed,
                'producer': producer, 'binding_sha256': binding_hash})
        else:
            atomic_json(output_root / 'partial_receipt.json', {'status': 'pilot_complete',
                        'completed_units': completed, 'total_units': len(expected_units),
                        'continuation': 'rerun with identical binding and without max_units'})
    if completed != len(expected_units):
        atomic_json(output_root / 'run_status.json', {'stage': '08N-patch-extraction', 'status': 'pilot_complete',
                    'completed_units': completed, 'total_units': len(expected_units)})
    return {'completed_units': completed, 'total_units': len(expected_units),
            'extracted_units': extracted, 'elapsed_seconds': time.monotonic() - start,
            'sealed': (output_root / 'feature_lock.json').is_file()}


def load_observations(feature_root, scene, route, views, identity_map):
    """Load only the chosen images; a cache of extra views grants no extra evidence."""
    feature_root = Path(feature_root)
    rows, geometry = [], {}
    if len(views) != len(set(views)):
        raise ValueError('Duplicate selected views')
    binding = read_json(feature_root / 'run_binding.json')
    from .storage import digest
    binding_sha256 = digest(binding)
    if scene not in binding['scenes'] or not set(views) <= set(binding['views']):
        raise ValueError('Requested scene or views are outside feature binding')
    for view in views:
        unit = feature_root / 'views' / scene / f'{route}-{view}'
        if not verified_unit(unit, binding_sha256):
            raise FileNotFoundError('Selected feature checkpoint is missing')
        metadata = read_json(unit / 'observations.json')
        if (metadata['scene'], metadata['route'], metadata['view']) != (scene, route, view):
            raise ValueError('Feature checkpoint coordinates differ from requested scene, route or view')
        with np.load(unit / 'features.npz', allow_pickle=False) as arrays:
            for obj in metadata['objects']:
                idx, candidate = obj['array_index'], obj['candidate_id']
                valid = arrays[f'v{idx}'].astype(bool).reshape(-1)
                rows.append({'view_id': view, 'candidate_id': candidate,
                             'object_id': identity_map[(view, candidate)],
                             'patches': arrays[f'p{idx}'][valid].copy()})
                geometry[(view, candidate)] = obj
    return rows, geometry
