"""Exercise the upload app with byte-bound, recorded development model outputs.

This is a local QA tool, not a production backend or a new accuracy experiment.
No anomaly labels are opened. Neutral dataset IDs are used only for the explicitly
labelled assisted check. Unknown images/crops cannot fall back to model inference.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.web.core import Box, InspectionRequest, Session, load_views, validate_result
from visual_peer_inspector.web.pipeline import UploadPipeline
from visual_peer_inspector.web.presentation import discard_exports, export_report
from vpi_object_bank.storage import atomic_json, digest_file, read_json


VIEW_SETS = {2: ('V00', 'V10'), 3: ('V00', 'V07', 'V14'),
             5: ('V00', 'V04', 'V08', 'V12', 'V16')}


def pixel_hash(image):
    array = np.ascontiguousarray(image, dtype=np.uint8)
    return sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def checked_path(root, relative, lock):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or digest_file(path) != lock['files'].get(relative):
        raise ValueError(f'Recorded artifact checksum mismatch: {relative}')
    return path


class RecordedStages:
    """Replay only known RGBs, boxes, masks and exact crop pixels."""

    def __init__(self, config, scene, views):
        self.calls = dict(propose=0, segment=0, encode=0, encoded_crops=0)
        self.images, self.crops, self.paths = {}, {}, {}
        self.scene = scene
        model_root, feature_root = Path(config['model_root']), Path(config['feature_root'])
        model_lock = read_json(model_root / 'artifact_lock.json')
        feature_lock = read_json(feature_root / 'feature_lock.json')
        binding = read_json(checked_path(feature_root, 'run_binding.json', feature_lock))
        if binding['model_input_lock_sha256'] != digest_file(model_root / 'artifact_lock.json'):
            raise ValueError('Recorded features and RGB/mask inputs have different provenance')
        rows = read_json(checked_path(model_root, f'scenes/{scene}.json', model_lock))
        for row in rows:
            if row['route'] != 'automatic' or row['view'] not in views:
                continue
            view = row['view']
            path = checked_path(model_root, row['rgb'], model_lock)
            image = np.asarray(Image.open(path).convert('RGB'))
            self.paths[view] = path
            prefix = f'views/{scene}/automatic-{view}'
            metadata = read_json(checked_path(feature_root, prefix + '/observations.json', feature_lock))
            objects = {o['candidate_id']: o for o in row['objects']}
            lookup = {o['candidate_id']: o for o in metadata['objects']}
            prepared = []
            with np.load(checked_path(feature_root, prefix + '/features.npz', feature_lock), allow_pickle=False) as arrays:
                for candidate, obj in sorted(objects.items()):
                    mask = decode_binary_mask(obj['mask'])
                    crop = prepare_masked_crop(image, mask)
                    meta = lookup[candidate]
                    if crop.sha256 != meta['crop_sha256']:
                        raise ValueError('Recorded crop differs from current preprocessing')
                    index = meta['array_index']
                    tensors = (arrays[f'g{index}'].copy(), arrays[f'p{index}'].copy())
                    key = pixel_hash(crop.rgb)
                    if key in self.crops and any(not np.array_equal(a, b) for a, b in zip(self.crops[key], tensors)):
                        raise ValueError('Identical crop pixels have conflicting cached features')
                    self.crops[key] = tensors
                    prepared.append({'candidate_id': candidate, 'box': tuple(obj['box']), 'mask': mask})
            self.images[pixel_hash(image)] = {'view': view, 'objects': prepared}
        if set(self.paths) != set(views):
            raise ValueError('A requested recorded view is unavailable')

    def row(self, image):
        try:
            return self.images[pixel_hash(np.asarray(image.convert('RGB')))]
        except KeyError as exc:
            raise ValueError('Recorded QA accepts only its exact development images') from exc

    def propose(self, image, view_id):
        self.calls['propose'] += 1
        return [o['box'] for o in self.row(image)['objects']], []

    def segment(self, image, boxes):
        self.calls['segment'] += 1
        objects = {o['box']: o for o in self.row(image)['objects']}
        if any(tuple(box) not in objects for box in boxes):
            raise ValueError('An edited box requires new SAM inference; recorded QA cannot supply it')
        masks = np.asarray([objects[tuple(box)]['mask'] for box in boxes])
        # Segmentation quality was not retained in this cache. It is unused by
        # both selected scorers and is explicitly identified as a QA placeholder.
        return masks, np.zeros(len(boxes), dtype=np.float32)

    def encode(self, crops):
        self.calls['encode'] += 1
        self.calls['encoded_crops'] += len(crops)
        try:
            tensors = [self.crops[pixel_hash(crop)] for crop in crops]
        except KeyError as exc:
            raise ValueError('Current upload preprocessing produced an unknown recorded crop') from exc
        return EncodedBatch(np.stack([t[0] for t in tensors]), tuple(t[1] for t in tensors), 16, 384)

    def release(self):
        pass


class RecordedBackend:
    """Explicit operator-only QA adapter. Never selects dataset identities."""

    def __init__(self, config, scene, views, cache_root):
        self.models = RecordedStages(config, scene, views)
        self.pipeline = UploadPipeline(self.models, cache_root=cache_root)

    def inspect(self, request, progress):
        result = self.pipeline.inspect(request, progress)
        result = replace(result, provenance={**result.provenance,
            'input_source': 'recorded_development', 'recorded_scene': self.models.scene,
            'model_execution': 'replayed saved outputs; no fresh neural inference',
            'sam_quality': 'unavailable in recording; unused zero placeholder',
            'recorded_stage_calls': dict(self.models.calls)})
        return result

    def clear_session(self, session_id, cache_token=None):
        self.pipeline.clear_session(session_id, cache_token)


def persist_export(root, views, result):
    root.mkdir(parents=True, exist_ok=True)
    session = Session(views=views, result=result)
    try:
        paths = export_report(session, 'Recorded development QA; not a fresh accuracy experiment')
        for path in paths:
            shutil.copy2(path, root / Path(path).name)
    finally:
        discard_exports(session)


def run(config, scene, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(config['prepared_root'])
    prepared = read_json(checked_path(prepared_root, f'scenes/{scene}.json', read_json(prepared_root / 'artifact_lock.json')))
    neutral = {(o['view'], o['candidate_id']): o['neutral_id'] for o in prepared['observations'] if o['route'] == 'automatic'}
    outcomes = []
    for count, schedule in VIEW_SETS.items():
        started = time.monotonic()
        backend = RecordedBackend(config, scene, schedule, output / 'session-cache')
        inputs = output / f'{count}-views' / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        for view in schedule:
            shutil.copy2(backend.models.paths[view], inputs / f'{view}.png')
        request = InspectionRequest(load_views([str(inputs / f'{v}.png') for v in schedule]))
        automatic = backend.inspect(request, lambda message: print(f'{count} views: {message}', flush=True))
        validate_result(automatic, request)
        persist_export(output / f'{count}-views' / 'automatic', request.views, automatic)
        # Supply neutral correspondence only after the automatic decision exists.
        boxes = []
        for observation in automatic.observations:
            view = next(v for v in request.views if v.view_id == observation.box.view_id)
            row = backend.models.row(view.image)
            source = next(o for o in row['objects'] if o['box'] == observation.box.xyxy)
            identity = neutral[(row['view'], source['candidate_id'])] or ''
            boxes.append(replace(observation.box, identity_id=identity))
        before = dict(backend.models.calls)
        assisted_request = replace(request, corrections=tuple(boxes), mode='assisted', cache_token=automatic.cache_token)
        assisted = backend.inspect(assisted_request, lambda message: None)
        if backend.models.calls != before:
            raise AssertionError('Identity-only reranking repeated a recorded neural stage')
        assisted = replace(assisted,
            observations=tuple(replace(o, identity_source='dataset_reference' if o.identity_source == 'user_confirmed' else o.identity_source)
                               for o in assisted.observations),
            provenance={**assisted.provenance, 'identity_source': 'dataset_reference',
                        'assistance': 'supplied neutral IDs; not a human-confirmation study'})
        validate_result(assisted, assisted_request)
        persist_export(output / f'{count}-views' / 'dataset-assisted', request.views, assisted)
        for result in (automatic, assisted):
            assert len(result.observations) == len(automatic.observations)
            for observation in result.observations:
                image = next(v.image for v in request.views if v.view_id == observation.box.view_id)
                assert observation.mask.size == image.size
                assert observation.heatmap is None or observation.heatmap.size == image.size
        outcomes.append({'views': list(schedule), 'candidate_count': len(automatic.observations),
            'automatic_ranking_count': len(automatic.ranking), 'assisted_ranking_count': len(assisted.ranking),
            'automatic_selection': automatic.selected_identity_id, 'assisted_selection': assisted.selected_identity_id,
            'initial_recorded_stage_calls': before, 'identity_rerank_extra_stage_calls': 0,
            'assisted_cache': assisted.provenance['cache'], 'elapsed_seconds': time.monotonic()-started})
        backend.clear_session(request.session_id, assisted.cache_token)
    receipt = {'schema': 'vpi-web-recorded-qa/v1', 'status': 'passed', 'scene': scene,
               'selection_rule': 'fixed scene supplied before outcomes; no anomaly labels opened',
               'scope': 'recorded stage replay and current upload scoring, cache, evidence and export code',
               'fresh_gpu_inference_verified': False, 'checks': outcomes}
    atomic_json(output / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scene', default='S000021')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.scene, args.output)
