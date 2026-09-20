#!/usr/bin/env python3
"""Download pinned model bytes once, without importing or running a vision model."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from current_notebook_runtime import atomic_json, digest, verify_model_snapshot


def specs(source_root):
    pipeline=json.loads((Path(source_root)/'src/visual_peer_inspector/web/pipeline_config.json').read_text())
    return {'locateanything':pipeline['locateanything'],'sam2':pipeline['sam2'],'dino':pipeline['encoder']}


def _download(spec,destination):
    from huggingface_hub import snapshot_download
    patterns=['*.json','*.safetensors','*.safetensors.index.json','*.model','*.py','*.txt','*.tiktoken','README*','LICENSE*']
    if 'filename' in spec:patterns=[spec['filename'],'README*','LICENSE*']
    snapshot_download(repo_id=spec['model_id'],revision=spec['revision'],local_dir=destination,
                      allow_patterns=patterns,max_workers=2)


def prepare(source_root,output_root,only='all',timeout_seconds=1800):
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    selected=specs(source_root)
    selected={name:spec for name,spec in selected.items() if only=='all' or name==only or only=='regions' and name in {'locateanything','sam2'}}
    if not selected:raise ValueError('Unknown model asset selection')
    config_hash=digest(Path(source_root)/'src/visual_peer_inspector/web/pipeline_config.json')
    binding={'schema':'vpi-current-model-assets/v1','pipeline_config_sha256':config_hash,'models':{name:{k:spec[k] for k in ('model_id','revision')} for name,spec in selected.items()}}
    bind_path=root/'asset_binding.json'
    if bind_path.exists() and json.loads(bind_path.read_text())!=binding:raise ValueError('Existing assets belong to another pinned pipeline')
    atomic_json(bind_path,binding)
    for name,spec in selected.items():
        target=root/name
        if (target/'model_snapshot.json').exists():
            verify_model_snapshot(target,spec['model_id'],spec['revision']);continue
        temporary=root/(name+'.download')
        temporary.mkdir(exist_ok=True)
        job=root/(name+'.download-job.json');atomic_json(job,{'spec':spec,'destination':str(temporary)})
        # The child owns transient download state. Error receipts omit URL/token text.
        with (root/(name+'.download.log')).open('wb') as log:
            child=subprocess.Popen([sys.executable,__file__,'--download-job',str(job)],stdout=log,stderr=log)
            started=time.monotonic()
            while child.poll() is None:
                if time.monotonic()-started>timeout_seconds:
                    child.kill();child.wait();atomic_json(root/(name+'-failure.json'),{'status':'timeout','model_loaded':False});raise TimeoutError('Pinned model asset download timed out; retained partial bytes can resume')
                print(f'Preparing pinned {name} model bytes: {int(time.monotonic()-started)} seconds',flush=True)
                time.sleep(10)
        if child.returncode:
            atomic_json(root/(name+'-failure.json'),{'status':'download_failed','returncode':child.returncode,'model_loaded':False})
            raise RuntimeError('Pinned model download failed; inspect the private local log')
        # Hub local-dir bookkeeping contains no model input; keep it outside the sealed snapshot.
        if (temporary/'.cache').exists():
            bookkeeping=root/(name+'.download-metadata')
            if bookkeeping.exists():shutil.rmtree(bookkeeping)
            (temporary/'.cache').rename(bookkeeping)
        files={p.relative_to(temporary).as_posix():digest(p) for p in temporary.rglob('*') if p.is_file()}
        if name=='dino':
            if files.get('model.safetensors')!=spec['sha256'] or files.get('config.json')!=spec['config_sha256']:
                raise ValueError('DINO weights/config differ from the selected pipeline')
        if name=='sam2':
            path=temporary/spec['filename']
            if files.get(spec['filename'])!=spec['sha256'] or path.stat().st_size!=spec['size_bytes']:
                raise ValueError('SAM checkpoint differs from the selected pipeline')
        atomic_json(temporary/'model_snapshot.json',{'schema':'vpi-immutable-model-snapshot/v1',
            'model_id':spec['model_id'],'revision':spec['revision'],'files':files})
        verify_model_snapshot(temporary,spec['model_id'],spec['revision'])
        if target.exists():raise ValueError('Existing incomplete model target cannot be overwritten')
        temporary.rename(target)
    receipt={'schema':binding['schema'],'status':'pinned_bytes_prepared','pipeline_config_sha256':config_hash,
        'models':{name:verify_model_snapshot(root/name,spec['model_id'],spec['revision']) for name,spec in selected.items()},
        'model_inference_executed':False,'contains_dataset_images':False}
    atomic_json(root/'model_assets_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--only',choices=['all','regions','locateanything','sam2','dino'],default='all')
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--download-job',type=Path)
    args=parser.parse_args()
    if args.download_job:
        job=json.loads(args.download_job.read_text());_download(job['spec'],job['destination']);return
    if args.source_root is None or args.output_root is None:parser.error('source-root and output-root are required')
    print(json.dumps(prepare(args.source_root,args.output_root,args.only,args.timeout_seconds),indent=2))


if __name__=='__main__':main()
