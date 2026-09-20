#!/usr/bin/env python3
"""Preview locally or explicitly advance one prepared Kaggle job.

The first submission requires the verified packaged-source manifest. A small
completion receipt verifies queue/source identity and marker hashes; it is not
proof of a complete scientific download. Uncertain pushes are reconciled by sync.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import uuid
from current_notebook_runtime import atomic_json, verify_source, MANIFEST
from current_launch import validate_plan, referenced_resources, external_completed_jobs

SCHEMA='vpi-current-kaggle-queue/v1'
COMPLETE='kaggle_complete_with_receipt'
ACTIVE={'submitted','submission_requested','submission_uncertain','running','queued',
        'completed_without_required_receipt','receipt_rejected'}
STATUSES=ACTIVE|{COMPLETE,'failed'}


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _member(root,relative):
    value=Path(relative)
    if value.is_absolute() or '..' in value.parts:raise ValueError('Queue member must remain inside its root')
    path=Path(root)/value
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Unsafe queue member')
    return path


def _validate_receipt(receipt,plan,job_id,source_hash):
    if receipt.get('schema')!='vpi-current-job-completion/v1' or receipt.get('status')!='outputs_complete':
        raise ValueError('Completion receipt is not a completed current job')
    if receipt.get('job_id')!=job_id or receipt.get('plan_sha256')!=_hash(plan):
        raise ValueError('Kaggle completion receipt belongs to another prepared queue')
    if not _sha(source_hash) or receipt.get('source_snapshot_sha256')!=source_hash:
        raise ValueError('Completion receipt source snapshot differs')
    if receipt.get('full_download_verified') is not False:
        raise ValueError('A marker receipt cannot certify complete scientific downloads')
    expected={name for name,spec in plan['resources'].items() if spec.get('producer_job')==job_id}
    outputs=receipt.get('outputs')
    if not expected or not isinstance(outputs,dict) or set(outputs)!=expected or any(not _sha(value) for value in outputs.values()):
        raise ValueError('Completion receipt output inventory or hashes differ')


def _job_files(root,plan,job):
    folder=_member(root,job['id']);metadata_path=_member(folder,'kernel-metadata.json')
    metadata=json.loads(metadata_path.read_text());expected_file=Path(job['notebook']).name
    if metadata.get('code_file')!=expected_file or metadata.get('is_private') is not True:
        raise ValueError('Job metadata changes notebook or private publication boundary')
    producers={spec['ref'] for spec in plan['resources'].values() if spec.get('producer_job')==job['id']}
    if producers!={metadata.get('id')}:
        raise ValueError('Kernel reference differs from the declared output producer')
    names=referenced_resources(job['config'])|set(job.get('attach_resources',[]))|{'source'}
    if not job.get('setup_job'):
        key={'cpu':'cpu','dino-cpu':'dino','dino-gpu':'dino-gpu','regions-gpu':'regions'}[job.get('runtime','cpu')]
        names|={'runtime-'+key,'wheels-'+key}
    resources=[plan['resources'][name] for name in names]
    for key,kind in [('dataset_sources','dataset'),('kernel_sources','kernel')]:
        if metadata.get(key)!=sorted({r['ref'] for r in resources if r['kind']==kind}):
            raise ValueError('Metadata attaches resources outside the reviewed launch plan')
    if metadata.get('competition_sources')!=[] or metadata.get('enable_gpu')!=bool(job.get('runtime') in {'dino-gpu','regions-gpu'}) or metadata.get('enable_internet')!=bool(job.get('setup_job')):
        raise ValueError('Metadata changes the reviewed runtime or network setting')
    notebook=_member(folder,expected_file);nb=json.loads(notebook.read_text())
    if any(c.get('outputs') or c.get('execution_count') is not None for c in nb.get('cells',[]) if c.get('cell_type')=='code'):
        raise ValueError('Submit only the clean prepared notebook')
    return {'kernel_ref':metadata['id'],'notebook_sha256':_file_hash(notebook),
            'metadata_sha256':_file_hash(metadata_path)}


def open_queue(root):
    root=Path(root);plan=json.loads((root/'launch-plan.json').read_text());validate_plan(plan)
    path=root/'queue-state.json'
    state=json.loads(path.read_text()) if path.exists() else {'schema':SCHEMA,'plan_sha256':_hash(plan),'jobs':{}}
    if state.get('schema')!=SCHEMA or state.get('plan_sha256')!=_hash(plan):
        raise ValueError('Queue plan/state changed; retain it and prepare a new queue')
    jobs={job['id']:job for job in plan['jobs']}
    external=set(external_completed_jobs(plan,jobs))
    if not isinstance(state.get('jobs'),dict) or not set(state['jobs'])<=set(jobs):raise ValueError('Queue state names unknown jobs')
    if set(state['jobs']) & external:raise ValueError('Queue state overlaps external completed jobs')
    if state['jobs'] and (not _sha(state.get('source_snapshot_sha256')) or set(state.get('prepared_jobs',{}))!=set(jobs)):
        raise ValueError('Queue state omits source or prepared-job pins')
    for name,record in state['jobs'].items():
        if record.get('status') not in STATUSES:raise ValueError('Unknown queue state status')
        expected=state['prepared_jobs'][name]
        if any(record.get(key)!=value for key,value in expected.items()) or _job_files(root,plan,jobs[name])!=expected:
            raise ValueError('Submitted job bytes or metadata changed')
        if record['status']==COMPLETE:
            receipt_path=_member(root,record.get('completion_receipt',''))
            if not receipt_path.is_file() or _file_hash(receipt_path)!=record.get('completion_sha256'):
                raise ValueError('Retained completion receipt changed or is missing')
            _validate_receipt(json.loads(receipt_path.read_text()),plan,name,state['source_snapshot_sha256'])
            if record.get('full_scientific_download_verified') is not False:
                raise ValueError('Queue completion cannot certify full scientific downloads')
    complete=external|{name for name,record in state['jobs'].items() if record['status']==COMPLETE}
    if any(not set(jobs[name].get('depends_on',[]))<=complete for name in state['jobs']):
        raise ValueError('Queue state bypasses unfinished prerequisites')
    return root,plan,state,path


def next_jobs(plan,state):
    jobs={job['id']:job for job in plan['jobs']}
    complete=set(external_completed_jobs(plan,jobs))|{
        name for name,value in state['jobs'].items() if value['status']==COMPLETE}
    return [job for job in plan['jobs'] if job['id'] not in state['jobs'] and set(job.get('depends_on',[]))<=complete]


@contextmanager
def _queue_lock(root):
    # OS releases the lock after a crash; a second controller never races a push.
    with (Path(root)/'.queue-state.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another queue controller is active') from error
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def _command(args,timeout=120):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError('Kaggle command failed; queue will require reconciliation')
    return result.stdout


def _pin_source(root,plan,state,source_manifest):
    if source_manifest is not None:
        manifest=Path(source_manifest)
        if manifest.name!=MANIFEST or manifest.is_symlink():raise ValueError('Supply the packaged prepared_source_manifest.json')
        source=verify_source(manifest.parent)
        if state.get('source_snapshot_sha256') not in {None,source['snapshot_sha256']}:
            raise ValueError('Queue source snapshot changed')
        state['source_snapshot_sha256']=source['snapshot_sha256']
    if not _sha(state.get('source_snapshot_sha256')):
        raise ValueError('First submit requires --source-manifest from the verified packaged source')
    prepared={job['id']:_job_files(root,plan,job) for job in plan['jobs']}
    if state.get('prepared_jobs') not in (None,prepared):raise ValueError('Prepared queue files changed after pinning')
    state['prepared_jobs']=prepared


def submit_next(root,job_id=None,source_manifest=None):
    with _queue_lock(root):
        root,plan,state,path=open_queue(root)
        ready=next_jobs(plan,state)
        if job_id is not None:ready=[job for job in ready if job['id']==job_id]
        if not ready:raise ValueError('No selected job is ready; sync active jobs or inspect prerequisites')
        _pin_source(root,plan,state,source_manifest)
        job=ready[0];record={'status':'submission_requested',**state['prepared_jobs'][job['id']]}
        state['jobs'][job['id']]=record;atomic_json(path,state)
        try:_command(['kaggle','kernels','push','-p',str(root/job['id'])],timeout=180)
        except BaseException:
            record['status']='submission_uncertain';atomic_json(path,state);raise
        record['status']='submitted';atomic_json(path,state)
        return {'job':job['id'],'status':'submitted','kernel_ref':record['kernel_ref'],'dataset_experiment':not job.get('setup_job',False)}


def _remote_status(message):
    aliases={'complete':'complete','running':'running','queued':'queued','error':'failed','failed':'failed','canceled':'failed','cancelled':'failed'}
    try:value=json.loads(message).get('status')
    except (ValueError,AttributeError):value=None
    if isinstance(value,str):
        candidate=re.sub(r'^kernel(?:run|worker)status[.]','',value.lower())
        if candidate in aliases:return aliases[candidate]
    text=message.strip().lower()
    if text in aliases:return aliases[text]
    exact=re.fullmatch(r'kernel(?:run|worker)status[.](complete|running|queued|error|failed|canceled|cancelled)',text)
    if exact:return aliases[exact.group(1)]
    found=re.findall(r'\bstatus\b["\']?\s*[:=]?\s*["\']?(?:kernel(?:run|worker)status[.])?(complete|running|queued|error|failed|canceled|cancelled)\b',text)
    if len(found)!=1:raise ValueError('Unrecognized Kaggle status; retain current state')
    return aliases[found[0]]


def sync(root):
    with _queue_lock(root):
        root,plan,state,path=open_queue(root);changes=[]
        for job_id,record in state['jobs'].items():
            if record['status'] not in ACTIVE:continue
            try:status=_remote_status(_command(['kaggle','kernels','status',record['kernel_ref']],timeout=60))
            except Exception as error:
                record['last_sync_error_type']=type(error).__name__;atomic_json(path,state)
                changes.append({'job':job_id,'status':record['status'],'sync_error_type':type(error).__name__});continue
            if status!='complete':
                record['status']=status;record.pop('last_sync_error_type',None);atomic_json(path,state)
                changes.append({'job':job_id,'status':status});continue
            destination=root/'completion-downloads'/job_id/uuid.uuid4().hex;destination.mkdir(parents=True)
            try:
                _command(['kaggle','kernels','output',record['kernel_ref'],'-p',str(destination),
                    '--file-pattern','(^|/)job_completion[.]json$','--quiet'],timeout=180)
                files=list(destination.rglob('job_completion.json'))
                if len(files)!=1:
                    record['status']='completed_without_required_receipt';atomic_json(path,state)
                    changes.append({'job':job_id,'status':record['status']});continue
                if files[0].is_symlink() or not files[0].resolve().is_relative_to(destination.resolve()) or files[0].stat().st_size>1024**2:
                    raise ValueError('Invalid downloaded completion receipt path/size')
                _validate_receipt(json.loads(files[0].read_text()),plan,job_id,state['source_snapshot_sha256'])
            except (ValueError,KeyError,TypeError):
                record['status']='receipt_rejected';atomic_json(path,state);raise
            except Exception as error:
                record['last_sync_error_type']=type(error).__name__;atomic_json(path,state)
                changes.append({'job':job_id,'status':record['status'],'sync_error_type':type(error).__name__});continue
            record.update(status=COMPLETE,completion_receipt=str(files[0].relative_to(root)),
                completion_sha256=_file_hash(files[0]),full_scientific_download_verified=False)
            record.pop('last_sync_error_type',None);atomic_json(path,state)
            changes.append({'job':job_id,'status':record['status']})
        return {'changes':changes,'next_ready':[job['id'] for job in next_jobs(plan,state)],
            'full_output_downloads':'Small receipts do not verify scientific downloads; verify every registered artifact before reporting that.'}


def preview(root):
    _,plan,state,_=open_queue(root)
    return {'status':'local_preview_only','job_count':len(plan['jobs']),'next_ready':[job['id'] for job in next_jobs(plan,state)],
        'submitted_by_this_command':False,'tracked_jobs':state['jobs']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue-root',type=Path,required=True)
    parser.add_argument('--action',choices=['preview','submit-next','sync'],default='preview');parser.add_argument('--job')
    parser.add_argument('--source-manifest',type=Path,help='Required for first submission; verifies and pins the packaged source snapshot')
    args=parser.parse_args()
    if args.action=='submit-next':result=submit_next(args.queue_root,args.job,args.source_manifest)
    elif args.action=='sync':result=sync(args.queue_root)
    else:result=preview(args.queue_root)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
