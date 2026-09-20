#!/usr/bin/env python3
"""Read-only setup inventory. No dataset file, credential or model is opened."""
import argparse,importlib.metadata,json,platform,shutil,subprocess,sys
from pathlib import Path


def probe(profile_paths):
    names=set();profiles={}
    for path in profile_paths:
        profile=json.loads(Path(path).read_text());profiles[Path(path).name]=profile
        names.update(profile['runtime_contract']['packages'])
    packages={}
    for name in sorted(names):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    image={}
    for name in ('git_commit','build_date'):
        path=Path('/etc')/name
        image[name]=path.read_text().strip() if path.is_file() else None
    gpu={'available':False}
    if shutil.which('nvidia-smi'):
        result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
        gpu={'available':result.returncode==0,'devices':result.stdout.strip().splitlines() if result.returncode==0 else []}
    comparisons={}
    for name,profile in profiles.items():
        contract=profile['runtime_contract']
        comparisons[name]={'python_matches':platform.python_version()==contract['python'],
            'package_mismatches':{p:{'expected':v,'actual':packages[p]} for p,v in contract['packages'].items() if packages[p]!=v}}
    return {'schema':'vpi-kaggle-setup-probe/v1','status':'metadata_probe_complete','system':platform.system(),'machine':platform.machine(),
            'python':platform.python_version(),'packages':packages,'image':image,'gpu':gpu,'profiles':comparisons,
            'dataset_or_model_access':False,'scientific_run':False,'inference_run':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',action='append',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=probe(args.profile);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
