"""Download pinned assets once; validate SAM and CPU DINO without user data."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/model-cache'))
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
from visual_peer_inspector.web.deployment import pipeline_config
from visual_peer_inspector.web.model_runtime import GPUModels

started = time.monotonic()
config = pipeline_config()
models = GPUModels(allow_downloads=True)
receipts = {}
for stage, key in [('locate', 'locateanything'), ('encoder', 'encoder')]:
    print(f'Downloading pinned {stage} assets', flush=True)
    source = Path(models._snapshot(config[key]))
    if key == 'encoder':
        for name, expected in [('model.safetensors', config[key]['sha256']), ('config.json', config[key]['config_sha256'])]:
            with (source / name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, f'{name} integrity'
    receipts[key] = {'revision': config[key]['revision'], 'source': str(source)}
print('Loading pinned SAM checkpoint on GPU', flush=True)
models._load('sam')
models.release()
receipts['sam2'] = {'revision': config['sam2']['revision'], 'sha256': config['sam2']['sha256'], 'gpu_loaded': True}
print('Loading LocateAnything on GPU', flush=True)
models._load('locate')
models.release()
receipts['locateanything']['gpu_loaded'] = True
print('Checking exact CPU DINO worker with synthetic crops', flush=True)
import numpy as np
batch = models.encode([np.zeros((224, 224, 3), dtype=np.uint8), np.full((224, 224, 3), 128, dtype=np.uint8)])
assert batch.global_features.shape == (2, 384)
assert np.isfinite(batch.global_features).all()
receipts['encoder']['synthetic_encode_passed'] = True
receipts['elapsed_seconds'] = round(time.monotonic() - started, 2)
output = ROOT / 'logs/model-preflight.json'
temporary = output.with_suffix('.tmp')
temporary.write_text(json.dumps(receipts, indent=2) + '\n')
temporary.replace(output)
print('model_preflight_passed', flush=True)
