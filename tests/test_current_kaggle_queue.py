"""Queue policy and recovery tests; all external commands are intercepted."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import hashlib
import shutil
from types import SimpleNamespace
import nbformat
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import current_kaggle_queue as queue
from current_launch import build_kernels,complete_job
from current_notebook_runtime import atomic_json,canonical


@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    monkeypatch.setattr(queue.subprocess,'run',lambda *a,**k:pytest.fail('No real subprocess/Kaggle calls in queue tests'))


@pytest.fixture
def fixture(tmp_path):
    source=tmp_path/'source';(source/'scripts').mkdir(parents=True);(source/'requirements').mkdir()
    (source/'scripts/current_notebook_runtime.py').write_text('# fixture bootstrap\n')
    (source/'requirements/current-evaluation-runtime.json').write_text('{}')
    notebook=source/'notebooks/fixture.ipynb';notebook.parent.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('value = 1')]),notebook)
    inventory={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
    snapshot=hashlib.sha256(canonical(inventory)).hexdigest()
    atomic_json(source/'prepared_source_manifest.json',{'schema':'vpi-current-notebook-source/v1','source_state':'working_directory_snapshot','files':inventory,'snapshot_sha256':snapshot})
    plan={'schema':'fixture','resources':{'source':{'kind':'dataset','ref':'esreekarreddy/source','marker':'prepared_source_manifest.json','suffixes':['']},
        'first-output':{'kind':'kernel','ref':'esreekarreddy/vpi-current-first','producer_job':'first','marker':'artifact_lock.json','suffixes':['first']},
        'second-output':{'kind':'kernel','ref':'esreekarreddy/vpi-current-second','producer_job':'second','marker':'artifact_lock.json','suffixes':['second']}},
        'jobs':[{'id':'first','purpose':'Fixture setup','notebook':'notebooks/fixture.ipynb','setup_job':True,'depends_on':[],'config':{'output_root':{'$output':'first'}}},
                {'id':'second','purpose':'Fixture dependent setup','notebook':'notebooks/fixture.ipynb','setup_job':True,'depends_on':['first'],
                 'config':{'input':{'$resource':'first-output'},'output_root':{'$output':'second'}}}]}
    root=tmp_path/'queue';build_kernels(plan,source,root)
    return root,source/'prepared_source_manifest.json',plan,snapshot


def fake_kaggle(monkeypatch,fixture,tmp_path,*,mutation=None,push_error=False,status='complete'):
    root,manifest,plan,snapshot=fixture;calls=[]
    def command(args,timeout=120):
        calls.append((args,timeout));action=args[2]
        if action=='push':
            state=json.loads((root/'queue-state.json').read_text())
            assert any(value['status']=='submission_requested' for value in state['jobs'].values())
            if push_error:raise subprocess.TimeoutExpired(args,timeout)
            return 'Kernel version1 successfully pushed'
        if action=='status':return f'{args[3]} has status "{status}"'
        assert action=='output' and '--file-pattern' in args and args[args.index('--file-pattern')+1]=='(^|/)job_completion[.]json$'
        job=next(name for name in ('first','second') if args[3].endswith('-'+name))
        output=tmp_path/'remote'/job;(output/job).mkdir(parents=True,exist_ok=True)
        atomic_json(output/job/'artifact_lock.json',{'fixture':job})
        receipt=complete_job(plan,job,snapshot,output_base=output)
        if mutation:mutation(receipt)
        target=Path(args[args.index('-p')+1])/'retained/job_completion.json';atomic_json(target,receipt)
        return ''
    monkeypatch.setattr(queue,'_command',command)
    return calls


def test_preview_default_is_read_only_without_hidden_network(fixture,monkeypatch,capsys):
    root,_,_,_=fixture;before=sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))
    monkeypatch.setattr(sys,'argv',['queue','--queue-root',str(root)])
    queue.main();result=json.loads(capsys.readouterr().out)
    assert result['status']=='local_preview_only' and result['next_ready']==['first']
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))==before


def test_explicit_submit_pins_source_and_dependency_receipts(fixture,tmp_path,monkeypatch):
    root,manifest,_,snapshot=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    with pytest.raises(ValueError,match='source-manifest'):queue.submit_next(root)
    assert not calls and not (root/'queue-state.json').exists()
    with pytest.raises(ValueError,match='ready'):queue.submit_next(root,'second',manifest)
    queue.submit_next(root,'first',manifest)
    state=json.loads((root/'queue-state.json').read_text());assert state['source_snapshot_sha256']==snapshot
    with pytest.raises(ValueError,match='ready'):queue.submit_next(root,'second')
    result=queue.sync(root);assert result['next_ready']==['second']
    assert not json.loads((root/'queue-state.json').read_text())['jobs']['first']['full_scientific_download_verified']
    queue.submit_next(root,'second')
    assert [args[2] for args,_ in calls]==['push','status','output','push']
    assert calls[0][1]==180 and calls[1][1]==60 and calls[2][1]==180


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(status='checkpointed'),lambda r:r.update(schema='legacy'),
    lambda r:r.update(source_snapshot_sha256='0'*64),lambda r:r.update(plan_sha256='0'*64),
    lambda r:r.update(full_download_verified=True),lambda r:r.update(outputs={'wrong':'a'*64}),
    lambda r:r.update(outputs={'first-output':'not-a-sha'}),lambda r:r.pop('source_snapshot_sha256')])
def test_forged_or_incomplete_receipt_cannot_unlock_job(fixture,tmp_path,monkeypatch,mutation):
    root,manifest,_,_=fixture;fake_kaggle(monkeypatch,fixture,tmp_path,mutation=mutation)
    queue.submit_next(root,source_manifest=manifest)
    with pytest.raises(ValueError):queue.sync(root)
    state=json.loads((root/'queue-state.json').read_text())
    assert state['jobs']['first']['status']=='receipt_rejected'
    assert queue.preview(root)['next_ready']==[]


def test_uncertain_push_persists_intent_and_requires_sync(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path,push_error=True)
    with pytest.raises(subprocess.TimeoutExpired):queue.submit_next(root,source_manifest=manifest)
    assert json.loads((root/'queue-state.json').read_text())['jobs']['first']['status']=='submission_uncertain'
    with pytest.raises(ValueError,match='ready'):queue.submit_next(root,'first')
    assert len(calls)==1
    calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    assert queue.sync(root)['next_ready']==['second']
    assert [args[2] for args,_ in calls]==['status','output']


def test_retained_receipt_corruption_and_forged_state_fail_closed(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;fake_kaggle(monkeypatch,fixture,tmp_path)
    queue.submit_next(root,source_manifest=manifest);queue.sync(root)
    path=root/'queue-state.json';state=json.loads(path.read_text());record=state['jobs']['first']
    receipt=root/record['completion_receipt'];receipt.write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):queue.preview(root)
    record['completion_receipt']='../outside.json';atomic_json(path,state)
    with pytest.raises(ValueError,match='inside'):queue.preview(root)


def test_changed_prepared_metadata_or_notebook_rejected_before_push(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    queue.submit_next(root,source_manifest=manifest);queue.sync(root)
    notebook=root/'second/fixture.ipynb';value=json.loads(notebook.read_text());value['cells'].append({'cell_type':'markdown','metadata':{},'source':'unreviewed change'});atomic_json(notebook,value)
    before=len(calls)
    with pytest.raises(ValueError,match='files changed'):queue.submit_next(root,'second')
    assert len(calls)==before


def test_status_substring_is_not_completion_and_unknown_state_is_preserved(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path,status='incomplete')
    queue.submit_next(root,source_manifest=manifest)
    assert queue.sync(root)['next_ready']==[]
    state=json.loads((root/'queue-state.json').read_text());assert state['jobs']['first']['status']=='submitted'
    assert [args[2] for args,_ in calls]==['push','status']


@pytest.mark.parametrize(('message','expected'),[
    ('owner/job has status "KernelWorkerStatus.ERROR"','failed'),
    ('KernelWorkerStatus.RUNNING','running'),
    ('{"status":"KernelWorkerStatus.COMPLETE"}','complete'),
    ('owner/job has status "KernelRunStatus.QUEUED"','queued')])
def test_remote_status_accepts_current_kaggle_worker_enums(message,expected):
    assert queue._remote_status(message)==expected


def test_missing_receipt_can_be_synced_again_without_resubmission(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    queue.submit_next(root,source_manifest=manifest)
    original=queue._command
    monkeypatch.setattr(queue,'_command',lambda args,timeout=120:'' if args[2]=='output' else original(args,timeout))
    queue.sync(root)
    assert json.loads((root/'queue-state.json').read_text())['jobs']['first']['status']=='completed_without_required_receipt'
    monkeypatch.setattr(queue,'_command',original)
    assert queue.sync(root)['next_ready']==['second']
    assert sum(args[2]=='push' for args,_ in calls)==1


def test_cli_subprocess_uses_bounded_timeout_and_sanitized_failure(monkeypatch):
    seen=[]
    def run(args,**kwargs):seen.append((args,kwargs));return SimpleNamespace(returncode=1,stdout='',stderr='secret credential-like text')
    monkeypatch.setattr(queue.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='require reconciliation') as error:queue._command(['kaggle','kernels','status','owner/job'],timeout=60)
    assert 'secret' not in str(error.value)
    assert seen[0][1]['timeout']==60 and seen[0][1]['capture_output'] and seen[0][1]['text']


def test_concurrent_controller_cannot_duplicate_submission(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    with queue._queue_lock(root):
        with pytest.raises(RuntimeError,match='Another queue controller'):queue.submit_next(root,source_manifest=manifest)
    assert not calls and not (root/'queue-state.json').exists()


def test_changed_packaged_source_stops_before_external_call(fixture,tmp_path,monkeypatch):
    root,manifest,_,_=fixture;calls=fake_kaggle(monkeypatch,fixture,tmp_path)
    (manifest.parent/'scripts/current_notebook_runtime.py').write_text('# altered source')
    with pytest.raises(ValueError,match='source file'):queue.submit_next(root,source_manifest=manifest)
    assert not calls and not (root/'queue-state.json').exists()


def test_verified_external_completion_unlocks_fresh_continuation(fixture, tmp_path):
    _, manifest, original, _ = fixture
    plan = copy.deepcopy(original)
    first = plan['resources']['first-output']
    first.pop('producer_job')
    first.update(
        availability='existing_kaggle', external_source_job='first',
        required_sha256={'artifact_lock.json': 'a' * 64},
    )
    plan['jobs'] = [job for job in plan['jobs'] if job['id'] != 'first']
    plan['external_completed_jobs'] = {
        'first': {
            'plan_sha256': 'b' * 64,
            'source_snapshot_sha256': 'c' * 64,
            'completion_sha256': 'd' * 64,
            'outputs': {'first-output': 'a' * 64},
            'kernel_ref': 'esreekarreddy/vpi-current-first',
        }
    }
    root = tmp_path / 'continuation'
    build_kernels(plan, manifest.parent, root)
    assert queue.preview(root)['next_ready'] == ['second']

    forged = json.loads((root / 'launch-plan.json').read_text())
    forged['external_completed_jobs']['first']['completion_sha256'] = 'not-a-sha'
    atomic_json(root / 'launch-plan.json', forged)
    with pytest.raises(ValueError, match='evidence hash'):
        queue.preview(root)
