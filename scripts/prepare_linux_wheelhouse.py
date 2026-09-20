#!/usr/bin/env python3
"""Download exact PyPI CPU wheels for CPython3.12 Linux; never install or run them."""
import argparse,email,json,re,time,urllib.request,zipfile
from pathlib import Path
from current_notebook_runtime import atomic_json,digest


def compatible(filename):
    if not filename.endswith('.whl'):return False
    if filename.endswith('-none-any.whl'):return True
    if '-cp312-' not in filename or 'x86_64' not in filename:return False
    versions=[tuple(map(int,m)) for m in re.findall(r'manylinux_(\d+)_(\d+)_x86_64',filename)]
    return bool(versions and min(versions)<=(2,28)) or 'manylinux2014_x86_64' in filename


def download(url,target,expected=None):
    if not url.startswith(('https://pypi.org/','https://files.pythonhosted.org/')):
        raise ValueError('Wheel preparation accepts official PyPI metadata and files only')
    target.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'VPI-runtime-preparation'}),timeout=45) as response:
                with target.with_suffix(target.suffix+'.partial').open('wb') as output:
                    size=0
                    while block:=response.read(1024*1024):
                        size+=len(block)
                        if size>100*1024*1024:raise ValueError('One CPU wheel exceeds the100MiB preparation bound')
                        output.write(block)
            temporary=target.with_suffix(target.suffix+'.partial')
            if expected and digest(temporary)!=expected:raise ValueError('Downloaded wheel hash differs from PyPI')
            temporary.replace(target);return
        except (OSError,TimeoutError):
            if attempt:raise
            time.sleep(1)


def prepare(profile_path,output_root):
    profile=json.loads(Path(profile_path).read_text());contract=profile['runtime_contract']
    root=Path(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve existing wheelhouse; choose a fresh directory')
    root.mkdir(parents=True,exist_ok=True);files={};records=[]
    for name,version in contract['packages'].items():
        meta=root/'metadata'/(name+'.json');download(f'https://pypi.org/pypi/{name}/{version}/json',meta)
        candidates=[r for r in json.loads(meta.read_text())['urls'] if not r.get('yanked') and compatible(r['filename'])]
        if not candidates:raise ValueError(f'No pinned Linux CPython3.12 wheel for {name}=={version}')
        row=sorted(candidates,key=lambda r:(r['size'],r['filename']))[0];target=root/row['filename']
        download(row['url'],target,row['digests']['sha256'])
        with zipfile.ZipFile(target) as wheel:
            if wheel.testzip():raise ValueError('CPU wheel CRC mismatch')
            metadata_names=[n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
            if len(metadata_names)!=1:raise ValueError('Wheel metadata is ambiguous')
            metadata=email.message_from_bytes(wheel.read(metadata_names[0]))
            if metadata['Version']!=version:raise ValueError('Wheel version differs')
        files[target.name]=digest(target);records.append({'package':name,'version':version,'filename':target.name,'size_bytes':row['size'],'pypi_metadata_sha256':digest(meta)})
        print(f'Prepared Linux wheel{len(files)}/{len(contract["packages"])}: {name}=={version}',flush=True)
    manifest={'schema':'vpi-current-offline-wheels/v1','runtime_contract':contract,'files':files}
    atomic_json(root/'wheelhouse_manifest.json',manifest)
    receipt={'status':'linux_wheel_bytes_verified','target':'CPython3.12 manylinuxx86_64 glibc<=2.28 wheels',
             'profile_sha256':digest(profile_path),'wheel_count':len(files),'total_bytes':sum(r['size_bytes'] for r in records),
             'installation_or_execution_performed':False,'kaggle_runtime_verified':False,'wheels':records}
    atomic_json(root/'preparation_receipt.json',receipt);return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);parser.add_argument('--output-root',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.profile,args.output_root),indent=2))
