#!/usr/bin/env python3
"""Setup-only verified private dependency target and actual-kernel runtime profile.

No model, dataset or scientific method is imported. This must precede the new
method freeze; its profile cannot change a runtime already sealed by that freeze.
"""
import argparse,json,platform,subprocess,sys
from pathlib import Path
from current_notebook_runtime import atomic_json,digest,canonical,install_offline,check_runtime,writable_output
import hashlib


def _isolated_failure_receipt(error):
    receipt_root = (Path('/kaggle/working/runtime-receipts') if Path('/kaggle/working').is_dir()
                    else Path.cwd()/'.local/runtime-receipts')
    atomic_json(receipt_root/'runtime-environment-isolated-failure.json', {
        'status': 'failed', 'error_type': type(error).__name__,
        'scientific_run_started': False, 'isolated_child_process': True})


def prepare_isolated(profile_path,wheelhouse,target,output_root):
    """Prepare dependencies in a fresh interpreter, never in the notebook kernel."""
    path=Path(profile_path);base=json.loads(path.read_text());contract=base.get('runtime_contract',base)
    timeout=1020 if 'torch' in contract['packages'] else 420
    command=[sys.executable,str(Path(__file__).resolve()),'--profile',str(path),
             '--wheelhouse',str(Path(wheelhouse)),'--target',str(Path(target)),
             '--output-root',str(Path(output_root))]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        if result.returncode:
            raise ChildProcessError('Isolated runtime setup child failed')
        root=Path(output_root)
        probe=json.loads((root/'setup_probe.json').read_text())
        profile=json.loads((root/'runtime_profile.json').read_text())
        lock=json.loads((root/'runtime_profile_lock.json').read_text())
        expected={name:digest(root/name) for name in ('runtime_profile.json','setup_probe.json')}
        if (probe.get('status')!='package_metadata_verified'
                or probe.get('scientific_imports_started') is not False
                or probe.get('model_or_dataset_access') is not False
                or profile.get('schema')!='vpi-observed-runtime-profile/v1'
                or profile.get('setup_probe_sha256')!=expected['setup_probe.json']
                or lock.get('schema')!='vpi-observed-runtime-lock/v1'
                or lock.get('files')!=expected):
            raise ValueError('Isolated runtime setup output failed verification')
    except BaseException as error:
        _isolated_failure_receipt(error)
        raise RuntimeError('Isolated runtime setup failed; retained a safe failure receipt') from None
    return {'status':'runtime_profile_prepared','python':probe['python'],
            'profile_sha256':expected['runtime_profile.json'],'private_target_prepared':True,
            'models_loaded':False,'datasets_opened':False,'scientific_methods_executed':False,
            'isolated_child_process':True}


def prepare(profile_path,wheelhouse,target,output_root):
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('This setup requires Linux x86_64 with CPython3.12')
    path=Path(profile_path);base=json.loads(path.read_text());contract=base.get('runtime_contract',base)
    root=writable_output(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve prior setup receipt; choose a fresh output directory')
    install_offline(contract,wheelhouse,target,timeout=900 if 'torch' in contract['packages'] else 300)
    sys.path.insert(0,str(Path(target).resolve()))
    runtime_contract={**contract,'python':platform.python_version()}
    actual=check_runtime(runtime_contract)
    probe={'schema':'vpi-runtime-setup-observation/v1','status':'package_metadata_verified',
           'python':platform.python_version(),'packages':actual['packages'],'system':platform.system(),'machine':platform.machine(),
           'scientific_imports_started':False,'model_or_dataset_access':False,'wheelhouse_manifest_sha256':digest(Path(wheelhouse)/'wheelhouse_manifest.json')}
    atomic_json(root/'setup_probe.json',probe)
    profile={'schema':'vpi-observed-runtime-profile/v1','base_profile_sha256':digest(path),
             'runtime_contract':runtime_contract,'setup_probe_sha256':digest(root/'setup_probe.json'),
             'python_patch_adjusted':contract['python']!=platform.python_version(),
             'scope':'Observed setup metadata; development numerical compatibility still required before a new method freeze'}
    atomic_json(root/'runtime_profile.json',profile)
    atomic_json(root/'runtime_profile_lock.json',{'schema':'vpi-observed-runtime-lock/v1',
                'files':{name:digest(root/name) for name in ('runtime_profile.json','setup_probe.json')}})
    return {'status':'runtime_profile_prepared','python':platform.python_version(),'profile_sha256':digest(root/'runtime_profile.json'),
            'private_target_prepared':True,'models_loaded':False,'datasets_opened':False,'scientific_methods_executed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);parser.add_argument('--wheelhouse',required=True)
    parser.add_argument('--target',required=True);parser.add_argument('--output-root',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.profile,args.wheelhouse,args.target,args.output_root),indent=2))
