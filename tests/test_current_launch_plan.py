"""Validate the prepared queue/configuration only; never submit or run a stage."""
from pathlib import Path
import sys
import json
import copy
import hashlib
import pytest
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_current_launch_plan import plan
from current_launch import validate_plan,build_kernels,resolve_resource,resolve_job,configure_runtime,complete_job,publish_completed_job
from build_continuation_queue import _externalize


def test_exact_original_roles_methods_schedules_and_declared_dependencies():
    value=plan();receipt=validate_plan(value)
    assert receipt['submitted'] is False
    assert value['original_roles']=={'development':1000,'calibration':600,'internal_lock':400}
    assert receipt['jobs']==152
    assert next(job for job in value['jobs'] if job['id']=='setup-tracking')['config']['profile']=='tracking'
    assert value['methods']==['winner_cluster_consensus','object_bank_top2']
    assert value['runtime_readiness']['sam2_code']=={
        'resource':'wheels-regions','distribution':'SAM-2',
        'repository':'https://github.com/facebookresearch/sam2',
        'revision':'2b90b9f5ceec907a1c18123530e92e794ad901a4',
        'preflight_marker':'sam2_preflight_verified'}
    assert value['core_views']==[2,3,5] and value['schedule_count']==19
    shards=[job for job in value['jobs'] if job['config']['action']=='prepare_shard']
    assert len(shards)==53
    region_shards=[job for job in value['jobs'] if job['config']['action']=='extract']
    assert len(region_shards)==43
    assert all(job['config']['maximum_new_views']==2000 and job['config']['shard_count']>=1 for job in region_shards)
    assert len([job for job in value['jobs'] if job['config']['action']=='region_merge'])==4
    for job in shards: assert job['config']['max_new_scenes']==100 and job['runtime']=='dino-cpu'
    modified=copy.deepcopy(value)
    job=next(j for j in modified['jobs'] if j['id']=='calibration-rgb')
    job['depends_on'].remove('freeze')
    with pytest.raises(ValueError,match='declared job prerequisite'):validate_plan(modified)


def test_real_released_cache_suffixes_and_historical_component_paths():
    value=plan();resources=value['resources']
    assert resources['08f1']['suffixes']==['08f1/model-cache']
    assert resources['09b1']['suffixes']==['09b1/model-cache']
    assert resources['10r-seen']['suffixes']==['10r-seen/model-input']
    cfg=next(j['config'] for j in value['jobs'] if j['id']=='development-cache-00')
    assert all(row['region_root']['path'].endswith('/model-safe') for row in cfg['components'].values())
    setup=next(j['config'] for j in value['jobs'] if j['id']=='setup-dino')
    assert setup['download_models'] and setup['models_only']=='all'
    assert setup['model_asset_root']=='/kaggle/working/setup-dino/model-assets'


def test_model_stage_rejects_explicit_raw_outcome_attachment():
    value=plan();job=next(j for j in value['jobs'] if j['id']=='development-cache-00')
    job['config']['unnecessary_raw']={'$resource':'raw'}
    with pytest.raises(ValueError,match='outcome resource'):validate_plan(value)


def test_resource_flat_nested_ambiguous_and_explicit_root(tmp_path):
    spec={'kind':'dataset','ref':'owner/fixture-data','marker':'artifact_lock.json','suffixes':['data']}
    root=tmp_path/'datasets/owner/fixture-data/versions/3/data';root.mkdir(parents=True);(root/'artifact_lock.json').write_text('{}')
    assert resolve_resource(spec,mount=tmp_path,name='fixture')==root
    other=tmp_path/'fixture-data/data';other.mkdir(parents=True);(other/'artifact_lock.json').write_text('{}')
    with pytest.raises(ValueError,match='found 2'):resolve_resource(spec,mount=tmp_path,name='fixture')
    assert resolve_resource(spec,mount=tmp_path,overrides={'fixture':str(root)},name='fixture')==root


def test_live_resource_hash_and_path_safety_in_resolved_config(tmp_path,monkeypatch):
    monkeypatch.setenv('VPI_NEXT_CONFIG','')
    root=tmp_path/'resource';root.mkdir();(root/'artifact_lock.json').write_text('{"fixture":1}')
    value={'resources':{'input':{'ref':'owner/input','kind':'dataset','marker':'artifact_lock.json','suffixes':['']}},
        'jobs':[{'id':'demo','config':{'source_root':{'$resource':'input'},'expected':{'$resource':'input','path':'artifact_lock.json','sha256':True},'output_root':{'$output':'demo'}}}]}
    result=resolve_job(value,'demo',output_base=tmp_path/'work',overrides={'input':str(root)})
    config=json.loads((tmp_path/'work/launch-config/demo.json').read_text())
    assert config['expected']==hashlib.sha256((root/'artifact_lock.json').read_bytes()).hexdigest()
    assert result['experiment_submitted'] is False
    value['jobs'][0]['config']['source_root']['path']='../escape'
    with pytest.raises(ValueError,match='relative artifact'):resolve_job(value,'demo',output_base=tmp_path/'work',overrides={'input':str(root)})


def test_runtime_uses_relative_profile_and_exact_attached_setup_outputs(tmp_path,monkeypatch):
    for name in ('VPI_RUNTIME_PROFILE_ROOT','VPI_WHEELHOUSE','VPI_PACKAGE_TARGET','VPI_INSTALL_DEPENDENCIES','VPI_BASE_RUNTIME_PROFILE'):
        monkeypatch.setenv(name,'')
    value=plan();overrides={}
    for name,marker in [('runtime-dino','runtime_profile_lock.json'),('wheels-dino','wheelhouse_manifest.json')]:
        path=tmp_path/name;path.mkdir();(path/marker).write_text('{}');overrides[name]=str(path)
    override=tmp_path/'overrides.json';override.write_text(json.dumps(overrides));monkeypatch.setenv('VPI_RESOURCE_OVERRIDES',str(override))
    result=configure_runtime(value,'development-cache-00',ROOT,mount=tmp_path)
    import os
    assert result['profile']=='dino'
    assert os.environ['VPI_BASE_RUNTIME_PROFILE']=='requirements/profiles/linux-dino-fp32-cpu.json'
    assert os.environ['VPI_RUNTIME_PROFILE_ROOT']==overrides['runtime-dino']


def test_all_generated_notebooks_compile_and_setup_config_reaches_actual_reader(tmp_path):
    value=plan();receipt=build_kernels(value,ROOT,tmp_path/'launch')
    assert not receipt['submitted'] and receipt['jobs']==len(value['jobs'])
    for job in value['jobs']:
        folder=tmp_path/'launch'/job['id'];path=folder/Path(job['notebook']).name
        assert path.stat().st_size<1024**2, 'Generated notebook must remain below 1MiB'
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                compile(cell.source,path.name,'exec')
                assert cell.execution_count is None and not cell.outputs
        if job.get('setup_job'):
            text='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
            assert "os.environ['VPI_NEXT_CONFIG']" in text and 'VPI_RUNTIME_SETUP_CONFIG' not in text
        else:
            cells=[c.source for c in notebook.cells if c.cell_type=='code']
            bootstrap=next(c for c in cells if 'BOOTSTRAP = _runtime.bootstrap' in c)
            assert bootstrap.index('configure_runtime(')<bootstrap.index('BOOTSTRAP = _runtime.bootstrap')
        text='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
        assert 'JOB_COMPLETION = complete_job(' in notebook.cells[-1].source
        assert text.count('publish_completed_job(')==int(job['config'].get('publish_tracking',False))
        if job['config']['action']=='prepare_shard':
            assert job['compact_checkpoints']
            assert 'begin_cache_job(' in text
            assert 'if not _LAUNCH_SCIENCE_REUSED:' in text
            assert 'retain_completed_cache(' in notebook.cells[-2].source
            assert text.index('begin_cache_job(')<text.index('if not _LAUNCH_SCIENCE_REUSED:')<text.index('retain_completed_cache(')<text.index('JOB_COMPLETION = complete_job(')
        if job['config'].get('publish_tracking'):
            assert 'setup-tracking' in job['depends_on']
        metadata=json.loads((folder/'kernel-metadata.json').read_text())
        assert metadata['is_private'] and metadata['enable_internet']==bool(job.get('setup_job'))
        if metadata['enable_gpu']: assert metadata['machine_shape']=='NvidiaTeslaT4'
        else: assert 'machine_shape' not in metadata
        assert 'esreekarreddy/vpi-current-evaluation-source' in metadata['dataset_sources']


def test_cpu_setup_uses_default_bare_contract_not_a_wrapper_profile(tmp_path,monkeypatch):
    import os
    for name in ('VPI_RUNTIME_PROFILE_ROOT','VPI_WHEELHOUSE','VPI_PACKAGE_TARGET','VPI_INSTALL_DEPENDENCIES','VPI_BASE_RUNTIME_PROFILE'):
        monkeypatch.setenv(name,'stale')
    value=plan();overrides={}
    for name,marker in [('runtime-cpu','runtime_profile_lock.json'),('wheels-cpu','wheelhouse_manifest.json')]:
        path=tmp_path/name;path.mkdir();(path/marker).write_text('{}');overrides[name]=str(path)
    path=tmp_path/'overrides.json';path.write_text(json.dumps(overrides));monkeypatch.setenv('VPI_RESOURCE_OVERRIDES',str(path))
    configure_runtime(value,'freeze',ROOT,mount=tmp_path)
    assert 'VPI_BASE_RUNTIME_PROFILE' not in os.environ
    assert os.environ['VPI_RUNTIME_PROFILE_ROOT']==overrides['runtime-cpu']


def test_completion_waits_for_every_output_marker_then_hashes_them(tmp_path):
    value=plan();job='setup-dino'
    produced={name:spec for name,spec in value['resources'].items() if spec.get('producer_job')==job}
    with pytest.raises(ValueError,match='checkpointed or incomplete'):
        complete_job(value,job,'a'*64,output_base=tmp_path)
    assert not (tmp_path/'job_completion.json').exists()
    for name,spec in produced.items():
        marker=tmp_path/spec['suffixes'][0]/spec['marker'];marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps({'synthetic_resource':name}))
    receipt=complete_job(value,job,'a'*64,output_base=tmp_path)
    assert receipt['job_id']==job and receipt['source_snapshot_sha256']=='a'*64
    assert set(receipt['outputs'])==set(produced) and receipt['full_download_verified'] is False
    assert all(len(value)==64 for value in receipt['outputs'].values())


def test_failed_tracking_setup_keeps_scientific_completion(tmp_path,monkeypatch):
    value=plan();job='09b';science=tmp_path/job;report=tmp_path/(job+'-report')
    for root,marker in [(science,'evaluation_lock.json'),(report,'report_lock.json')]:
        root.mkdir();(root/marker).write_text('{}')
    path=tmp_path/'launch-config'/f'{job}.json';path.parent.mkdir();path.write_text(json.dumps({'output_root':str(science),'report_root':str(report)}))
    import subprocess
    calls=[]
    def fail(command,**kwargs):calls.append(command);raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(subprocess,'run',fail)
    receipt=publish_completed_job(value,job,ROOT,output_base=tmp_path)
    assert receipt['status']=='pending' and receipt['scientific_results_preserved']
    assert len(calls)==1 and 'prepare_tracking_runtime.py' in calls[0][2]
    completed=complete_job(value,job,'a'*64,output_base=tmp_path,tracking=receipt)
    assert completed['status']=='outputs_complete' and completed['tracking']['status']=='pending'


def test_installed_runtime_targets_are_ephemeral():
    for job in plan()['jobs']:
        if job.get('setup_job'):
            assert job['config']['target'].startswith('/tmp/vpi-runtime-')
            assert job['config']['wheelhouse'].startswith('/kaggle/working/')


def test_continuation_externalizes_verified_outputs_but_reruns_region_setup():
    value = plan()
    completed = {'setup-cpu', 'freeze'}
    evidence = {}
    for job_id in completed:
        outputs = {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name, spec in value['resources'].items()
            if spec.get('producer_job') == job_id
        }
        ref = {value['resources'][name]['ref'] for name in outputs}
        assert len(ref) == 1
        evidence[job_id] = {
            'plan_sha256': 'a' * 64,
            'source_snapshot_sha256': 'b' * 64,
            'completion_sha256': 'c' * 64,
            'outputs': outputs,
            'kernel_ref': ref.pop(),
        }
    continuation = _externalize(value, completed, evidence)
    assert 'setup-cpu' not in {job['id'] for job in continuation['jobs']}
    assert 'freeze' not in {job['id'] for job in continuation['jobs']}
    assert 'setup-regions' in {job['id'] for job in continuation['jobs']}
    assert validate_plan(continuation)['status'] == 'valid'
    runtime = continuation['resources']['runtime-cpu']
    assert 'producer_job' not in runtime
    assert runtime['external_source_job'] == 'setup-cpu'
    assert runtime['required_sha256']['runtime_profile_lock.json'] == evidence['setup-cpu']['outputs']['runtime-cpu']


def test_external_completion_contract_rejects_unbound_or_overlapping_evidence():
    value = plan()
    completed = {'setup-cpu'}
    outputs = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name, spec in value['resources'].items()
        if spec.get('producer_job') == 'setup-cpu'
    }
    evidence = {'setup-cpu': {
        'plan_sha256': 'a' * 64,
        'source_snapshot_sha256': 'b' * 64,
        'completion_sha256': 'c' * 64,
        'outputs': outputs,
        'kernel_ref': 'esreekarreddy/vpi-current-setup-cpu',
    }}
    continuation = _externalize(value, completed, evidence)
    resource = next(iter(outputs))
    continuation['resources'][resource]['required_sha256'][
        continuation['resources'][resource]['marker']
    ] = '0' * 64
    with pytest.raises(ValueError, match='does not match completion evidence'):
        validate_plan(continuation)
