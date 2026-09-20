"""Setup-only profiles and source/metadata integrity; no dependency installation."""
from pathlib import Path
import csv,hashlib,importlib.metadata,importlib.util,io,json,subprocess,sys,zipfile
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from current_notebook_runtime import atomic_json,digest,observed_contract,verify_model_snapshot
from current_notebook_cells import bootstrap_cell


def test_registered_neural_locks_bind_exact_packages():
    for name in ('linux-regions-cuda126','linux-dino-fp32-cpu'):
        path=ROOT/'requirements/profiles'/(name+'.json');profile=json.loads(path.read_text())
        lock=path.parent/profile['requirements_lock']
        assert digest(lock)==profile['requirements_lock_sha256']
        assert 'torch=='+profile['runtime_contract']['packages']['torch'] in lock.read_text()
    source=bootstrap_cell('example.ipynb','requirements/profiles/linux-regions-cuda126.json').source
    compile(source,'profile-bootstrap','exec')
    assert "RUNTIME_PROFILE = 'requirements/profiles/linux-regions-cuda126.json'" in source


def _fixture_sam_wheel(path):
    dist='SAM_2-1.0.dist-info';records=[]
    members={
        'sam2/__init__.py':b'',
        f'{dist}/METADATA':b'Metadata-Version: 2.1\nName: SAM-2\nVersion: 1.0\n',
        f'{dist}/WHEEL':b'Wheel-Version: 1.0\nGenerator: vpi-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    for name,payload in members.items():
        digest=hashlib.sha256(payload).digest();encoded=__import__('base64').urlsafe_b64encode(digest).rstrip(b'=').decode()
        records.append([name,'sha256='+encoded,str(len(payload))])
    records.append([f'{dist}/RECORD','',''])
    output=io.StringIO(newline='');csv.writer(output,lineterminator='\n').writerows(records)
    members[f'{dist}/RECORD']=output.getvalue().encode()
    with zipfile.ZipFile(path,'w') as wheel:
        for name,payload in members.items():wheel.writestr(name,payload)


def test_region_profile_pins_exact_sam2_source_commit():
    profile=json.loads((ROOT/'requirements/profiles/linux-regions-cuda126.json').read_text())
    source=profile['source_packages']['SAM-2']
    assert profile['runtime_contract']['packages']['SAM-2']=='1.0'
    assert source=={'version':'1.0','repository':'https://github.com/facebookresearch/sam2',
        'revision':'2b90b9f5ceec907a1c18123530e92e794ad901a4',
        'build_environment':{'SAM2_BUILD_CUDA':'0'}}
    builder=(ROOT/'scripts/prepare_neural_wheelhouse.py').read_text()
    assert "'--no-deps','--no-build-isolation'" in builder


def test_sam2_source_wheel_is_canonical_and_retains_pep610_after_offline_install(tmp_path):
    if importlib.util.find_spec('pip') is None:
        pytest.skip('offline installation check requires pip in the test interpreter')
    from prepare_neural_wheelhouse import add_vcs_provenance,verify_vcs_wheel
    revision='2b90b9f5ceec907a1c18123530e92e794ad901a4';repository='https://github.com/facebookresearch/sam2'
    first=tmp_path/'first'/'sam_2-1.0-py3-none-any.whl';second=tmp_path/'second'/'sam_2-1.0-py3-none-any.whl'
    first.parent.mkdir();second.parent.mkdir();_fixture_sam_wheel(first);_fixture_sam_wheel(second)
    receipt=add_vcs_provenance(first,distribution='SAM-2',version='1.0',repository=repository,revision=revision)
    add_vcs_provenance(second,distribution='SAM-2',version='1.0',repository=repository,revision=revision)
    assert digest(first)==digest(second)==receipt['wheel_sha256']
    assert verify_vcs_wheel(first,distribution='SAM-2',version='1.0',repository=repository,revision=revision)==receipt
    target=tmp_path/'target'
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input',
        '--no-index','--no-deps','--find-links',str(first.parent),'--target',str(target),'SAM-2==1.0'],check=True,capture_output=True,text=True,timeout=60)
    distribution=next(d for d in importlib.metadata.distributions(path=[str(target)])
        if d.metadata['Name']=='SAM-2')
    direct=json.loads(distribution.read_text('direct_url.json'))
    assert direct=={'url':repository+'.git','vcs_info':{'commit_id':revision,
        'requested_revision':revision,'vcs':'git'}}


def test_observed_runtime_allows_python_patch_but_not_package_changes(tmp_path):
    base={'python':'3.12.14','packages':{'numpy':'2.5.3'},'device':'cpu'}
    base_path=tmp_path/'base.json';atomic_json(base_path,base)
    observed={**base,'python':'3.12.12'}
    def write(packages):
        atomic_json(tmp_path/'setup_probe.json',{'status':'package_metadata_verified','python':'3.12.12','packages':packages,'system':'Linux','machine':'x86_64','model_or_dataset_access':False})
        atomic_json(tmp_path/'runtime_profile.json',{'schema':'vpi-observed-runtime-profile/v1','base_profile_sha256':digest(base_path),'setup_probe_sha256':digest(tmp_path/'setup_probe.json'),'runtime_contract':{**observed,'packages':packages}})
        atomic_json(tmp_path/'runtime_profile_lock.json',{'schema':'vpi-observed-runtime-lock/v1','files':{n:digest(tmp_path/n) for n in ('setup_probe.json','runtime_profile.json')}})
    write(base['packages']);assert observed_contract(tmp_path,base_path,base)['python']=='3.12.12'
    write({'numpy':'9.0'})
    with pytest.raises(ValueError,match='package pins'):
        observed_contract(tmp_path,base_path,base)


def test_model_snapshot_verifier_never_loads_weights(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'synthetic-not-a-real-model')
    record={'schema':'vpi-immutable-model-snapshot/v1','model_id':'fixture/model','revision':'a'*40,'files':{'model.safetensors':digest(tmp_path/'model.safetensors')}}
    atomic_json(tmp_path/'model_snapshot.json',record)
    assert verify_model_snapshot(tmp_path,'fixture/model','a'*40)['model_loaded'] is False
    (tmp_path/'injected.py').write_text('raise RuntimeError("must never import")')
    with pytest.raises(ValueError,match='extra'):
        verify_model_snapshot(tmp_path,'fixture/model','a'*40)


def test_setup_notebook_is_clean_and_has_no_scientific_model_call():
    notebook=json.loads((ROOT/'notebooks/input-preparation/current-kaggle-runtime-setup.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert not cell['outputs'] and cell['execution_count'] is None
            compile(''.join(cell['source']),'runtime-setup','exec')
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'prepare_isolated' in source and 'run_model_asset_preparation' in source
    assert '.encode(' not in source and 'decide_cached(' not in source


def test_runtime_setup_uses_a_fresh_child_after_parent_scientific_imports(tmp_path,monkeypatch):
    import prepare_runtime_environment as setup
    profile_path=tmp_path/'profile.json';atomic_json(profile_path,{'python':'3.12.14','packages':{'fixture':'1.0'},'device':'cpu'})
    output=tmp_path/'observed';target=tmp_path/'target';wheelhouse=tmp_path/'wheels';seen={}
    def run(command,**kwargs):
        seen.update(command=command,kwargs=kwargs)
        probe={'schema':'vpi-runtime-setup-observation/v1','status':'package_metadata_verified','python':'3.12.13',
            'packages':{'fixture':'1.0'},'system':'Linux','machine':'x86_64','scientific_imports_started':False,
            'model_or_dataset_access':False,'wheelhouse_manifest_sha256':'a'*64}
        atomic_json(output/'setup_probe.json',probe)
        atomic_json(output/'runtime_profile.json',{'schema':'vpi-observed-runtime-profile/v1',
            'setup_probe_sha256':digest(output/'setup_probe.json')})
        atomic_json(output/'runtime_profile_lock.json',{'schema':'vpi-observed-runtime-lock/v1','files':{
            name:digest(output/name) for name in ('runtime_profile.json','setup_probe.json')}})
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(setup.subprocess,'run',run)
    monkeypatch.setitem(sys.modules,'numpy',object())
    receipt=setup.prepare_isolated(profile_path,wheelhouse,target,output)
    assert receipt['isolated_child_process'] and receipt['private_target_prepared']
    assert seen['kwargs']=={'capture_output':True,'text':True,'timeout':420}
    assert seen['command'][0]==sys.executable and '--output-root' in seen['command']


def test_runtime_setup_child_failure_is_sanitized_and_retained(tmp_path,monkeypatch):
    import prepare_runtime_environment as setup
    profile_path=tmp_path/'profile.json';atomic_json(profile_path,{'python':'3.12.14','packages':{'fixture':'1.0'},'device':'cpu'})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup.subprocess,'run',lambda *args,**kwargs:type('Result',(),{
        'returncode':1,'stdout':'private output','stderr':'credential-like value'})())
    with pytest.raises(RuntimeError,match='safe failure receipt') as error:
        setup.prepare_isolated(profile_path,tmp_path/'wheels',tmp_path/'target',tmp_path/'observed')
    assert 'private' not in str(error.value) and 'credential' not in str(error.value)
    receipt=json.loads((tmp_path/'.local/runtime-receipts/runtime-environment-isolated-failure.json').read_text())
    assert receipt=={'error_type':'ChildProcessError','isolated_child_process':True,
        'scientific_run_started':False,'status':'failed'}


def test_neural_wheel_metadata_ignores_vendored_dist_info(tmp_path):
    from prepare_neural_wheelhouse import root_dist_info_member
    path=tmp_path/'fixture.whl'
    with zipfile.ZipFile(path,'w') as wheel:
        wheel.writestr('fixture-1.0.dist-info/METADATA','Name: fixture\nVersion: 1.0\n')
        wheel.writestr('fixture-1.0.dist-info/WHEEL','Wheel-Version: 1.0\nTag: py3-none-any\n')
        wheel.writestr('fixture/_vendor/helper-2.0.dist-info/METADATA','Name: helper\nVersion: 2.0\n')
    with zipfile.ZipFile(path) as wheel:
        assert root_dist_info_member(wheel,'METADATA')=='fixture-1.0.dist-info/METADATA'
        assert root_dist_info_member(wheel,'WHEEL')=='fixture-1.0.dist-info/WHEEL'


def test_tracking_profile_is_separate_and_pinned():
    path = ROOT/'requirements/profiles/linux-tracking-cp312.json'
    profile = json.loads(path.read_text())
    assert profile['runtime_contract']['packages']['clearml'] == '2.1.10'
    assert digest(path.parent/profile['requirements_lock']) == profile['requirements_lock_sha256']
    assert 'clearml' not in json.loads((ROOT/'requirements/current-evaluation-runtime.json').read_text())['packages']


def test_tracking_child_inherits_private_target_and_missing_setup_is_blocked(tmp_path, monkeypatch):
    import current_notebook_runtime as runtime
    base = {'python': '3.12.14', 'packages': {'clearml': '2.1.10'}, 'device': 'cpu'}
    relative = 'requirements/profiles/linux-tracking-cp312.json'
    atomic_json(tmp_path/relative, {'runtime_contract': base})
    wheels = tmp_path/'wheels'
    atomic_json(wheels/'wheelhouse_manifest.json', {'runtime_contract': base})
    target = tmp_path/'tracking-target'
    env = {'VPI_TRACKING_PROFILE_ROOT': str(tmp_path/'observed'),
        'VPI_TRACKING_WHEELHOUSE': str(wheels), 'VPI_TRACKING_TARGET': str(target),
        'PYTHONPATH': '/existing-source'}
    monkeypatch.setattr(runtime, 'verify_source', lambda root: {'files': {relative: digest(tmp_path/relative)}})
    monkeypatch.setattr(runtime, 'observed_contract', lambda *args: base)
    monkeypatch.setattr(runtime, 'check_runtime', lambda contract: {'python': contract['python']})
    with pytest.raises(ValueError, match='Prepare the separate tracking target'):
        runtime.configure_tracking_runtime(tmp_path, environ=env)
    assert not target.exists()
    atomic_json(target/'dependency_receipt.json', {'runtime_contract': base,
        'wheelhouse_manifest_sha256': digest(wheels/'wheelhouse_manifest.json')})
    receipt = runtime.configure_tracking_runtime(tmp_path, environ=env)
    assert receipt['pinned_tracking_runtime'] and not receipt['service_contacted']
    assert env['PYTHONPATH'].split(__import__('os').pathsep) == [str(target), str(tmp_path/'scripts'), str(tmp_path/'src'), '/existing-source']


def test_kaggle_dependency_targets_are_ephemeral_and_separate_from_science():
    from current_notebook_runtime import runtime_target
    assert runtime_target('/tmp/vpi-runtime-fixture', kaggle=True).name == 'vpi-runtime-fixture'
    for path in ('/kaggle/working/sitepackages', '/kaggle/input/existing-env', '/tmp/scientific-output', '/tmp/vpi-runtime-'):
        with pytest.raises(ValueError, match='dependency targets'):
            runtime_target(path, kaggle=True)


def test_all_model_asset_download_budget_covers_each_model(tmp_path, monkeypatch):
    import current_notebook_runtime as runtime
    observed = {}
    class Child:
        pid = 123456
        def wait(self, timeout=None):
            observed['timeout'] = timeout
            return 0
    def popen(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return Child()
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    receipt = runtime.run_model_asset_preparation(tmp_path, tmp_path/'models', 'all', '/tmp/vpi-runtime-test')
    assert observed['timeout'] == 3*(1800+120)
    assert observed['kwargs']['start_new_session'] is True
    assert receipt['model_count'] == 3 and receipt['model_inference_executed'] is False


def test_asset_download_timeout_stops_grandchildren_and_records_failure(tmp_path, monkeypatch):
    import current_notebook_runtime as runtime
    calls = []
    class Child:
        pid = 123456
        def wait(self, timeout=None):
            if timeout is not None:
                raise runtime.subprocess.TimeoutExpired('fixture-download', timeout)
            calls.append('reaped')
        def poll(self):
            return None
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *args, **kwargs: Child())
    monkeypatch.setattr(runtime.os, 'killpg', lambda pid, signal: calls.append(('killed-group', pid)))
    with pytest.raises(runtime.subprocess.TimeoutExpired):
        runtime.run_model_asset_preparation(tmp_path, tmp_path/'models', 'dino', '/tmp/vpi-runtime-test')
    assert calls == [('killed-group', 123456), 'reaped']
    receipt = json.loads((tmp_path/'models/download-wrapper-receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['error_type'] == 'TimeoutExpired'
