"""Check four real CPU crops through the isolated app encoder against 08N tensors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.web.model_runtime import GPUModels
from vpi_object_bank.storage import atomic_json, digest_file, read_json
from verify_web_cached_replay import RecordedStages, pixel_hash


def run(config, python, output):
    scene, view = 'S000021', 'V00'
    recorded = RecordedStages(config, scene, (view,))
    source = next(r for r in read_json(Path(config['model_root']) / f'scenes/{scene}.json')
                  if r['view'] == view and r['route'] == 'automatic')
    image = np.asarray(Image.open(recorded.paths[view]).convert('RGB'))
    crops = [prepare_masked_crop(image, decode_binary_mask(o['mask'])).rgb
             for o in sorted(source['objects'], key=lambda o:o['candidate_id'])[:4]]
    models = GPUModels(allow_downloads=False)
    # This smoke checks the encoder process boundary, using the already verified
    # local snapshot directly. Download/cache discovery is a separate deployment check.
    models._snapshot = lambda config_record: Path(config['model_source'])
    previous = os.environ.get('VPI_DINO_PYTHON')
    os.environ['VPI_DINO_PYTHON'] = str(python)
    started = time.monotonic()
    try:
        batch = models.encode(crops)
    finally:
        if previous is None:
            os.environ.pop('VPI_DINO_PYTHON', None)
        else:
            os.environ['VPI_DINO_PYTHON'] = previous
    expected = [recorded.crops[pixel_hash(crop)] for crop in crops]
    global_delta = float(np.max(np.abs(batch.global_features - np.stack([v[0] for v in expected]))))
    patch_delta = float(np.max(np.abs(np.stack(batch.patch_features) - np.stack([v[1] for v in expected]))))
    assert global_delta <= 1e-6 and patch_delta <= 1e-6, (global_delta, patch_delta)
    receipt = {'schema':'vpi-web-real-cpu-encoder-smoke/v1', 'status':'passed',
        'scene':scene,'view':view,'crops':len(crops),'device':'cpu','dtype':'float32',
        'global_max_abs_delta':global_delta,'patch_max_abs_delta':patch_delta,
        'elapsed_seconds':time.monotonic()-started,
        'weights_sha256':digest_file(Path(config['model_source'])/'model.safetensors'),
        'model_config_sha256':digest_file(Path(config['model_source'])/'config.json'),
        'scope':'real isolated CPU DINO encoding; supplied local snapshot; no detector/SAM or GPU execution',
        'fresh_gpu_inference_verified':False}
    atomic_json(Path(output), receipt)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dino-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.dino_python,args.output)
