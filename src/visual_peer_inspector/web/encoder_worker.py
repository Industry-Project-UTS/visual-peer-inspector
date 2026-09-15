"""Pinned CPU DINO exchange, isolated from LocateAnything's older runtime."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np

PINNED = {'torch':'2.13.0','transformers':'5.14.1','numpy':'2.5.3','pillow':'12.3.0'}


def compatible_runtime():
    try:
        return all(importlib.metadata.version(name).split('+')[0] == version for name,version in PINNED.items())
    except importlib.metadata.PackageNotFoundError:
        return False


def encode_cpu(crops, model_source):
    if not compatible_runtime():
        raise RuntimeError('dino_runtime')
    import torch
    from visual_peer_inspector.models.visual_encoder import FrozenVisualEncoder, specification_by_key
    from .deployment import pipeline_config
    from .session_cache import file_hash
    config=pipeline_config()['encoder']
    source=Path(model_source)
    if file_hash(source/'model.safetensors') != config['sha256']:
        raise RuntimeError('checkpoint_integrity')
    if file_hash(source/'config.json') != config['config_sha256']:
        raise RuntimeError('checkpoint_integrity')
    torch.manual_seed(32040)
    torch.set_num_threads(min(4,__import__('os').cpu_count() or 1))
    model=FrozenVisualEncoder(specification_by_key(config['key']),revision=config['revision'],device='cpu',model_source=source)
    return model.encode(crops)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--model-source',required=True)
    args=parser.parse_args()
    with np.load(args.job/'input.npz',allow_pickle=False) as data:
        crops=data['crops']
    if crops.ndim!=4 or crops.shape[1:]!=(224,224,3) or crops.dtype!=np.uint8 or len(crops)>4:
        raise ValueError('Invalid DINO crop exchange')
    batch=encode_cpu(list(crops),args.model_source)
    with (args.job/'output.npz').open('wb') as stream:
        np.savez_compressed(stream,global_features=batch.global_features,patches=np.stack(batch.patch_features))
    (args.job/'receipt.json').write_text(json.dumps({'runtime':PINNED,'feature_dimension':batch.feature_dimension,'grid':batch.patch_grid_size}))


if __name__=='__main__':
    main()
