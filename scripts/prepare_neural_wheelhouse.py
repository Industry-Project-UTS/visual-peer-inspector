#!/usr/bin/env python3
"""Linux setup-only exact neural wheel resolver/builder; no models or datasets.

Run this in a separate setup session. It creates verified offline artifacts and
never installs into the notebook kernel. Native pure-Python source wheels may
be built by pip in its isolated build environment.
"""
import argparse,base64,csv,email,hashlib,io,json,os,platform,re,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
from current_notebook_runtime import atomic_json,digest,writable_output


def root_dist_info_member(wheel,filename):
    """Return the wheel's root metadata member, ignoring vendored metadata."""
    names=[]
    for name in wheel.namelist():
        parts=PurePosixPath(name).parts
        if len(parts)==2 and parts[0].endswith('.dist-info') and parts[1]==filename:
            names.append(name)
    if len(names)!=1:raise ValueError('Ambiguous root wheel metadata')
    return names[0]


def _normalise_distribution(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _record_digest(payload):
    value=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=')
    return 'sha256='+value.decode('ascii')


def add_vcs_provenance(wheel_path, *, distribution, version, repository, revision):
    """Canonicalise one source wheel and embed verified PEP 610 provenance."""
    path=Path(wheel_path)
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Source package revision must be one exact Git commit')
    if not repository.startswith('https://github.com/') or repository.endswith('/'):
        raise ValueError('Source package repository must be a canonical HTTPS GitHub URL')
    with zipfile.ZipFile(path) as source:
        if source.testzip():raise ValueError('Source wheel CRC differs')
        metadata_name=root_dist_info_member(source,'METADATA')
        record_name=root_dist_info_member(source,'RECORD')
        metadata=email.message_from_bytes(source.read(metadata_name))
        if (_normalise_distribution(metadata['Name'])!=_normalise_distribution(distribution)
                or metadata['Version']!=version):
            raise ValueError('Built source wheel name or version differs from the pinned profile')
        dist_info=PurePosixPath(metadata_name).parts[0]
        direct_name=f'{dist_info}/direct_url.json'
        direct=json.dumps({'url':repository+'.git','vcs_info':{'commit_id':revision,
            'requested_revision':revision,'vcs':'git'}},sort_keys=True,separators=(',',':')).encode()+b'\n'
        members={name:source.read(name) for name in source.namelist()
                 if not name.endswith('/') and name not in {direct_name,record_name}}
        record_rows=list(csv.reader(io.StringIO(source.read(record_name).decode())))
        record_rows=[row for row in record_rows if row and row[0] not in {direct_name,record_name}]
        record_rows.append([direct_name,_record_digest(direct),str(len(direct))])
        record_rows.append([record_name,'',''])
        output=io.StringIO(newline='');csv.writer(output,lineterminator='\n').writerows(sorted(record_rows))
        members[direct_name]=direct;members[record_name]=output.getvalue().encode()
    descriptor,temporary_name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    os.close(descriptor);temporary=Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as target:
            for name in sorted(members):
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.create_system=3;info.external_attr=0o100644 << 16
                target.writestr(info,members[name])
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return verify_vcs_wheel(path,distribution=distribution,version=version,
        repository=repository,revision=revision)


def verify_vcs_wheel(wheel_path, *, distribution, version, repository, revision):
    """Verify source wheel identity, PEP 610 commit and its RECORD binding."""
    path=Path(wheel_path)
    with zipfile.ZipFile(path) as wheel:
        if wheel.testzip():raise ValueError('Source wheel CRC differs')
        metadata_name=root_dist_info_member(wheel,'METADATA')
        record_name=root_dist_info_member(wheel,'RECORD')
        metadata=email.message_from_bytes(wheel.read(metadata_name))
        dist_info=PurePosixPath(metadata_name).parts[0]
        direct_name=f'{dist_info}/direct_url.json'
        direct_bytes=wheel.read(direct_name);direct=json.loads(direct_bytes)
        rows={row[0]:row[1:] for row in csv.reader(io.StringIO(wheel.read(record_name).decode())) if row}
    vcs=direct.get('vcs_info',{})
    if (_normalise_distribution(metadata['Name'])!=_normalise_distribution(distribution)
            or metadata['Version']!=version or direct.get('url')!=repository+'.git'
            or vcs!={'commit_id':revision,'requested_revision':revision,'vcs':'git'}):
        raise ValueError('Source wheel PEP 610 provenance differs from the pinned profile')
    if rows.get(direct_name)!=[_record_digest(direct_bytes),str(len(direct_bytes))]:
        raise ValueError('Source wheel RECORD does not bind its PEP 610 provenance')
    return {'distribution':metadata['Name'],'version':metadata['Version'],'repository':repository,
        'revision':revision,'wheel':path.name,'wheel_sha256':digest(path)}


def prepare(profile_path,output_root):
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Dependency wheel preparation requires Linux x86_64 and CPython 3.12; no cross-platform source builds')
    profile=json.loads(Path(profile_path).read_text());contract=profile['runtime_contract'];root=writable_output(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve earlier setup output; choose a fresh wheelhouse')
    root.mkdir(parents=True,exist_ok=True)
    source_packages=profile.get('source_packages',{})
    if not isinstance(source_packages,dict):raise ValueError('Source package profile must be a mapping')
    source_names={_normalise_distribution(name) for name in source_packages}
    if any(_normalise_distribution(name) not in {_normalise_distribution(k) for k in contract['packages']} for name in source_packages):
        raise ValueError('Every source package must be present in the runtime contract')
    required_gib = 12 if 'torch' in contract['packages'] else 2
    if shutil.disk_usage(root).free < required_gib * 1024**3:
        raise OSError(f'Reserve at least {required_gib} GiB for dependency wheel build/download and verification')
    requirement=root/'exact-input-requirements.txt'
    requirement.write_text('\n'.join(f'{name}=={version}' for name,version in contract['packages'].items()
        if _normalise_distribution(name) not in source_names)+'\n')
    locked=Path(profile_path).parent/profile['requirements_lock']
    if digest(locked)!=profile['requirements_lock_sha256']:
        raise ValueError('Neural dependency lock differs from its registered profile')
    command=[sys.executable,'-m','pip','wheel','--disable-pip-version-check','--no-input',
             '--wheel-dir',str(root),'--index-url','https://pypi.org/simple','--extra-index-url',profile['torch_index'],
             '--require-hashes','-r',str(locked)]
    started=time.monotonic();atomic_json(root/'setup_receipt.json',{'status':'running','profile_sha256':digest(profile_path),'models_loaded':False,'datasets_opened':False})
    try:
        # Nothing is installed into the kernel; the installer uses only official PyPI/PyTorch indexes.
        source_receipts={}
        with (root/'dependency-build.log').open('wb') as log:
            subprocess.run(command,check=True,timeout=1800,stdout=log,stderr=subprocess.STDOUT,
                           env={**{k:v for k,v in os.environ.items() if not k.startswith(('PIP_','UV_'))},'PIP_CONFIG_FILE':os.devnull,'PIP_DISABLE_PIP_VERSION_CHECK':'1'})
            for name,spec in sorted(source_packages.items()):
                if set(spec)!={'version','repository','revision','build_environment'} or spec['version']!=contract['packages'][name]:
                    raise ValueError('Source package profile differs from the runtime contract')
                environment={**{k:v for k,v in os.environ.items() if not k.startswith(('PIP_','UV_'))},
                    'PIP_CONFIG_FILE':os.devnull,'PIP_DISABLE_PIP_VERSION_CHECK':'1',**spec['build_environment']}
                before=set(root.glob('*.whl'))
                source_command=[sys.executable,'-m','pip','wheel','--disable-pip-version-check','--no-input',
                    '--no-deps','--no-build-isolation','--wheel-dir',str(root),
                    f"git+{spec['repository']}.git@{spec['revision']}"]
                subprocess.run(source_command,check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT,env=environment)
                created=set(root.glob('*.whl'))-before
                matches=[]
                for candidate in created:
                    with zipfile.ZipFile(candidate) as wheel:
                        meta=email.message_from_bytes(wheel.read(root_dist_info_member(wheel,'METADATA')))
                    if _normalise_distribution(meta['Name'])==_normalise_distribution(name):matches.append(candidate)
                if len(matches)!=1:raise ValueError('Pinned source build did not create exactly one expected wheel')
                source_receipts[_normalise_distribution(name)]=add_vcs_provenance(matches[0],distribution=name,
                    version=spec['version'],repository=spec['repository'],revision=spec['revision'])
        files={};resolved={};binary_tags={}
        for path in sorted(root.glob('*.whl')):
            with zipfile.ZipFile(path) as wheel:
                if wheel.testzip():raise ValueError('Neural wheel CRC differs')
                meta=email.message_from_bytes(wheel.read(root_dist_info_member(wheel,'METADATA')));name=meta['Name'].lower().replace('_','-')
                if name in resolved:raise ValueError('Multiple versions of one dependency in wheelhouse')
                resolved[name]=meta['Version']
                binary_tags[path.name]=[line for line in wheel.read(root_dist_info_member(wheel,'WHEEL')).decode().splitlines() if line.startswith('Tag:')]
            files[path.name]=digest(path)
        for name,version in contract['packages'].items():
            if resolved.get(name.lower().replace('_','-'))!=version:raise ValueError('Resolved neural dependencies changed an exact package pin')
        for name,receipt in source_receipts.items():
            if files.get(receipt['wheel'])!=receipt['wheel_sha256']:
                raise ValueError('Source package receipt differs from sealed wheel bytes')
        manifest={'schema':'vpi-current-offline-wheels/v1','runtime_contract':contract,'resolved_packages':resolved,'files':files,
                  'source_packages':source_receipts,
                  'wheel_tags':binary_tags,'profile_sha256':digest(profile_path),'target_python_abi':'cp312'}
        atomic_json(root/'wheelhouse_manifest.json',manifest)
        receipt={'status':'resolved_wheelhouse_prepared','elapsed_seconds':time.monotonic()-started,'observed_python':platform.python_version(),
                 'profile_python':contract['python'],'python_patch_matches':platform.python_version()==contract['python'],
                 'wheel_count':len(files),'total_bytes':sum(p.stat().st_size for p in root.glob('*.whl')),
                 'models_loaded':False,'datasets_opened':False,'kernel_packages_changed':False,'target_imports_verified':False}
        atomic_json(root/'setup_receipt.json',receipt);return receipt
    except BaseException as error:
        atomic_json(root/'setup_receipt.json',{'status':'failed','error_type':type(error).__name__,'elapsed_seconds':time.monotonic()-started,
                    'kernel_packages_changed':False,'scientific_run_started':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);parser.add_argument('--output-root',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.profile,args.output_root),indent=2))
