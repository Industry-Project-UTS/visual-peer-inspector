"""Resolve a prepared Kaggle job from its declared artifact dependency graph.

This module never uploads or submits jobs. Upstream output checksums are read
from retained resources when the next job is configured, not invented in advance.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
from current_notebook_runtime import bounded_roots, atomic_json


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def external_completed_jobs(plan, jobs=None):
    """Validate immutable predecessor evidence carried into a continuation plan."""
    jobs = jobs or {row['id']: row for row in plan['jobs']}
    external = plan.get('external_completed_jobs', {})
    if not isinstance(external, dict) or any(
        not isinstance(name, str) or not name or name in jobs for name in external
    ):
        raise ValueError('External completed jobs must be a disjoint named mapping')
    required = {'plan_sha256', 'source_snapshot_sha256', 'completion_sha256',
                'outputs', 'kernel_ref'}
    claimed_outputs = set()
    for name, record in external.items():
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError('External completion evidence fields differ')
        if any(not _sha256(record[key]) for key in (
            'plan_sha256', 'source_snapshot_sha256', 'completion_sha256'
        )):
            raise ValueError('External completion evidence hash differs')
        kernel_ref = record['kernel_ref']
        if (not isinstance(kernel_ref, str) or kernel_ref.count('/') != 1
                or any(not part for part in kernel_ref.split('/'))):
            raise ValueError('External completion kernel reference differs')
        outputs = record['outputs']
        if (not isinstance(outputs, dict) or not outputs
                or any(not isinstance(resource, str) or not _sha256(digest)
                       for resource, digest in outputs.items())):
            raise ValueError('External completion output inventory differs')
        if claimed_outputs & set(outputs):
            raise ValueError('External completion outputs overlap')
        claimed_outputs.update(outputs)
        for resource, digest in outputs.items():
            spec = plan.get('resources', {}).get(resource)
            if (not isinstance(spec, dict) or spec.get('producer_job') is not None
                    or spec.get('external_source_job') != name
                    or spec.get('ref') != kernel_ref
                    or spec.get('required_sha256', {}).get(spec.get('marker')) != digest):
                raise ValueError('External resource does not match completion evidence')
    for resource, spec in plan.get('resources', {}).items():
        source = spec.get('external_source_job') if isinstance(spec, dict) else None
        if source is not None and (
            source not in external or resource not in external[source]['outputs']
        ):
            raise ValueError('External resource has no matching completion evidence')
    return external


def _safe(path):
    value=Path(path)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Launch member must be a relative artifact path')
    return value


def _overrides(value=None):
    if value is not None: return value
    path=os.getenv('VPI_RESOURCE_OVERRIDES')
    result=json.loads(Path(path).read_text()) if path else {}
    if not isinstance(result,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in result.items()):
        raise ValueError('Resource overrides must map declared names to mounted directory paths')
    return result


def resolve_resource(spec, *, mount=Path('/kaggle/input'), overrides=None, name=None):
    overrides=_overrides(overrides)
    if name in overrides:
        roots=[Path(overrides[name])]
    else:
        # Restrict discovery to the explicitly attached slug; never scan raw media.
        slug=spec['ref'].split('/')[-1]
        def normalized(value):return value.lower().replace('_','-')
        roots=[p for p in bounded_roots(mount) if normalized(slug) in {normalized(part) for part in p.parts}]
    found=[]
    if name in overrides and (roots[0]/_safe(spec['marker'])).is_file():
        found.append(roots[0].resolve())
    for base in roots:
        for suffix in spec.get('suffixes',['']):
            root=base/_safe(suffix)
            if (root/_safe(spec['marker'])).is_file():found.append(root.resolve())
    found=sorted(set(found))
    if len(found)!=1:
        raise ValueError(f'Expected one mounted resource for {name}; found {len(found)}. Use explicit resource overrides.')
    root=found[0]
    for relative,expected in spec.get('required_sha256',{}).items():
        path=root/_safe(relative)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Launch resource checksum differs: {name}/{relative}')
    if name == 'unseen-rgb' and (root.parent/'rgb_transport.json').is_file():
        from current_rgb_transport import restore_rgb_transport
        root = restore_rgb_transport(root)
    return root


def referenced_resources(value):
    if isinstance(value,dict):
        if '$resource' in value:return {value['$resource']}
        return set().union(*(referenced_resources(v) for v in value.values())) if value else set()
    if isinstance(value,list):return set().union(*(referenced_resources(v) for v in value)) if value else set()
    return set()


def configure_runtime(plan,job_id,source_root,*,mount=Path('/kaggle/input')):
    job=next(row for row in plan['jobs'] if row['id']==job_id)
    profile=job.get('runtime','cpu')
    aliases={'cpu':'cpu','dino-cpu':'dino','dino-gpu':'dino-gpu','regions-gpu':'regions'}
    key=aliases[profile]
    observed=resolve_resource(plan['resources']['runtime-'+key],mount=mount,name='runtime-'+key)
    wheels=resolve_resource(plan['resources']['wheels-'+key],mount=mount,name='wheels-'+key)
    profile_path={'cpu':None,'dino':'requirements/profiles/linux-dino-fp32-cpu.json',
                  'dino-gpu':'requirements/profiles/linux-dino-fp32-cuda126.json',
                  'regions':'requirements/profiles/linux-regions-cuda126.json'}[key]
    os.environ.update(VPI_RUNTIME_PROFILE_ROOT=str(observed),VPI_WHEELHOUSE=str(wheels),
        VPI_PACKAGE_TARGET='/tmp/vpi-runtime-'+job_id,VPI_INSTALL_DEPENDENCIES='1')
    if profile_path: os.environ['VPI_BASE_RUNTIME_PROFILE']=profile_path
    else: os.environ.pop('VPI_BASE_RUNTIME_PROFILE',None)
    if job['config'].get('publish_tracking'):
        tracking_profile=resolve_resource(plan['resources']['runtime-tracking'],mount=mount,name='runtime-tracking')
        tracking_wheels=resolve_resource(plan['resources']['wheels-tracking'],mount=mount,name='wheels-tracking')
        os.environ.update(VPI_TRACKING_PROFILE_ROOT=str(tracking_profile),VPI_TRACKING_WHEELHOUSE=str(tracking_wheels),
            VPI_TRACKING_TARGET='/tmp/vpi-runtime-tracking-'+job_id)
    return {'profile':key,'observed_profile':str(observed),'wheelhouse':str(wheels)}


def resolve_job(plan, job_id, *, mount=Path('/kaggle/input'), output_base=Path('/kaggle/working'), overrides=None):
    jobs={row['id']:row for row in plan['jobs']}
    if len(jobs)!=len(plan['jobs']) or job_id not in jobs:
        raise ValueError('Unknown or duplicate launch job')
    job=jobs[job_id]
    names=referenced_resources(job['config'])
    roots={name:resolve_resource(plan['resources'][name],mount=mount,overrides=overrides,name=name) for name in names}
    def resolve(value):
        if isinstance(value,dict):
            if '$resource' in value:
                path=roots[value['$resource']]/_safe(value.get('path',''))
                if not path.resolve().is_relative_to(roots[value['$resource']]):raise ValueError('Launch path escapes resource')
                if value.get('sha256'):
                    if not path.is_file():raise FileNotFoundError(path)
                    return hashlib.sha256(path.read_bytes()).hexdigest()
                if not path.exists():raise FileNotFoundError(path)
                return str(path)
            if '$output' in value:return str(output_base/_safe(value['$output']))
            return {k:resolve(v) for k,v in value.items()}
        if isinstance(value,list):return [resolve(v) for v in value]
        return value
    cfg=resolve(copy.deepcopy(job['config']))
    receipt={'schema':'vpi-current-launch-binding/v1','job_id':job_id,
        'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'resources':{name:{'ref':plan['resources'][name]['ref'],'resolved_root':str(root)} for name,root in roots.items()},
        'experiment_submitted':False}
    target=output_base/'launch-config'/f'{job_id}.json'
    atomic_json(target,cfg);atomic_json(target.with_suffix('.receipt.json'),receipt)
    os.environ['VPI_NEXT_CONFIG']=str(target)
    return receipt


def validate_plan(plan):
    jobs={row['id']:row for row in plan['jobs']}
    if len(jobs)!=len(plan['jobs']):raise ValueError('Duplicate launch job')
    external=external_completed_jobs(plan,jobs)
    seen=set(external)
    for job in plan['jobs']:
        if not set(job.get('depends_on',[]))<=seen:raise ValueError('Jobs must be in dependency order')
        refs=referenced_resources(job['config']) | set(job.get('attach_resources',[]))
        if not refs<=set(plan['resources']):raise ValueError('Job references an undeclared resource')
        for name in refs:
            producer=plan['resources'][name].get('producer_job')
            if producer and producer not in seen:raise ValueError('Job consumes an output before its producer')
            if producer and producer not in job.get('depends_on',[]):raise ValueError('Resource producer is not a declared job prerequisite')
        if job.get('input_boundary')=='model_only':
            protocol_refs=set().union(*(referenced_resources(job['config'].get(key)) for key in ('calibration_receipt_root','decision_root')))
            if any(plan['resources'][name].get('contains_outcomes',False) and name not in protocol_refs for name in refs):
                raise ValueError('Model-facing job attaches an outcome resource')
        if not job.get('setup_job'):
            runtime={'cpu':'cpu','dino-cpu':'dino','dino-gpu':'dino-gpu','regions-gpu':'regions'}.get(job.get('runtime','cpu'))
            if runtime is None or 'setup-'+runtime not in job.get('depends_on',[]):
                raise ValueError('Required runtime setup is not a declared prerequisite')
        seen.add(job['id'])
    return {'status':'valid','jobs':len(jobs),'resources':len(plan['resources']),'submitted':False}


def build_kernels(plan, source_root, output_root):
    """Write ready-to-push private metadata and notebooks; perform no submission."""
    import nbformat
    validate_plan(plan)
    source_root, output_root=Path(source_root),Path(output_root)
    if output_root.exists():raise FileExistsError('Choose a fresh launch directory')
    output_root.mkdir(parents=True)
    atomic_json(output_root/'launch-plan.json',plan)
    for job in plan['jobs']:
        if 'notebook' not in job:continue
        nb=nbformat.read(source_root/job['notebook'],as_version=4)
        nb.cells.insert(0,nbformat.v4.new_markdown_cell(f"# Kaggle job: {job['id']}\n\n{job['purpose']}\n\nInputs are resolved from the declared resource graph. No raw dataset re-upload is required."))
        if job.get('setup_job'):
            nb.cells.insert(1,nbformat.v4.new_code_cell("import os,json\nfrom pathlib import Path\n_SETUP_CONFIG = "+repr(job['config'])+"\n_SETUP_PATH = Path('/kaggle/working/setup-config.json')\n_SETUP_PATH.write_text(json.dumps(_SETUP_CONFIG))\nos.environ['VPI_NEXT_CONFIG'] = str(_SETUP_PATH)"))
        else:
            positions=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'BOOTSTRAP = _runtime.bootstrap' in c.source]
            if len(positions)!=1:raise ValueError('Notebook must have one verified current bootstrap')
            bootstrap=nb.cells[positions[0]]
            before="    _launch_helper = CODE_ROOT / 'scripts/current_launch.py'\n    if hashlib.sha256(_launch_helper.read_bytes()).hexdigest() != _manifest['files']['scripts/current_launch.py']:\n        raise ValueError('Launch helper checksum differs')\n    sys.path.insert(0,str(CODE_ROOT/'scripts'))\n    from current_launch import configure_runtime\n    configure_runtime("+repr(plan)+", "+repr(job['id'])+", CODE_ROOT)\n"
            bootstrap.source=bootstrap.source.replace('    BOOTSTRAP = _runtime.bootstrap',before+'    BOOTSTRAP = _runtime.bootstrap')
            cell=nbformat.v4.new_code_cell("from current_launch import resolve_job\nLAUNCH_PLAN = "+repr(plan)+"\nLAUNCH_BINDING = resolve_job(LAUNCH_PLAN, "+repr(job['id'])+")\ndisplay(LAUNCH_BINDING)")
            nb.cells.insert(positions[0]+1,cell)
            if job.get('compact_checkpoints'):
                import textwrap
                # Bootstrap and exact resource/config binding always run. Only
                # scientific cells can be skipped after completed seals verify.
                for scientific in nb.cells[positions[0]+2:]:
                    if scientific.cell_type=='code':
                        scientific.source='if not _LAUNCH_SCIENCE_REUSED:\n'+textwrap.indent(scientific.source,'    ')
                nb.cells.insert(positions[0]+2,nbformat.v4.new_code_cell(
                    "from post_completion_retention import begin_cache_job\n_LAUNCH_SCIENCE_REUSED = begin_cache_job(LAUNCH_PLAN, "+repr(job['id'])+", CODE_PROVENANCE['snapshot_sha256'])\nprint({'verified_completed_science_reused': _LAUNCH_SCIENCE_REUSED})"))
        if job['config'].get('publish_tracking'):
            publication=nbformat.v4.new_code_cell("from current_launch import publish_completed_job\nLAUNCH_TRACKING = publish_completed_job(LAUNCH_PLAN, "+repr(job['id'])+", CODE_ROOT)\ndisplay(LAUNCH_TRACKING)")
            existing=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'publish_next_stage.py' in c.source]
            if len(existing)>1:raise ValueError('Duplicate publication entrypoints')
            if existing:nb.cells[existing[0]]=publication
            else:nb.cells.append(publication)
        if job.get('compact_checkpoints'):
            nb.cells.append(nbformat.v4.new_code_cell(
                "from post_completion_retention import retain_completed_cache\nLAUNCH_RETENTION = retain_completed_cache(LAUNCH_PLAN, "+repr(job['id'])+", CODE_PROVENANCE['snapshot_sha256'])\ndisplay(LAUNCH_RETENTION)"))
        snapshot="SOURCE['snapshot_sha256']" if job.get('setup_job') else "CODE_PROVENANCE['snapshot_sha256']"
        nb.cells.append(nbformat.v4.new_code_cell("from current_launch import complete_job\nJOB_COMPLETION = complete_job("+repr(plan)+", "+repr(job['id'])+", "+snapshot+", tracking=globals().get('LAUNCH_TRACKING'))\nprint(JOB_COMPLETION)"))
        target=output_root/job['id'];target.mkdir();filename=Path(job['notebook']).name
        nbformat.write(nb,target/filename)
        names=referenced_resources(job['config'])|set(job.get('attach_resources',[]))|{'source'}
        if not job.get('setup_job'):
            key={'cpu':'cpu','dino-cpu':'dino','dino-gpu':'dino-gpu','regions-gpu':'regions'}[job.get('runtime','cpu')]
            names|={'runtime-'+key,'wheels-'+key}
        resources=[plan['resources'][name] for name in names]
        metadata={'id':'esreekarreddy/vpi-current-'+job['id'].replace('_','-'), 'title':'VPI Current '+job['id'].replace('-',' ').title(),
            'code_file':filename,'language':'python','kernel_type':'notebook','is_private':True,
            'enable_gpu':job.get('runtime') in {'dino-gpu','regions-gpu'},'enable_internet':job.get('setup_job',False),
            **({'machine_shape':'NvidiaTeslaT4'} if job.get('runtime') in {'dino-gpu','regions-gpu'} else {}),
            'dataset_sources':sorted({r['ref'] for r in resources if r['kind']=='dataset'}),
            'kernel_sources':sorted({r['ref'] for r in resources if r['kind']=='kernel'}),'competition_sources':[]}
        atomic_json(target/'kernel-metadata.json',metadata)
    return {'status':'prepared','jobs':len(plan['jobs']),'submitted':False}


def publish_completed_job(plan,job_id,source_root,*,output_base=Path('/kaggle/working')):
    """Publication runs after science/report seals and never controls completion."""
    import subprocess
    import sys
    job=next(row for row in plan['jobs'] if row['id']==job_id)
    if job_id not in {'development-evaluate','09b','09d','10b'} or not job['config'].get('publish_tracking'):
        raise ValueError('This job has no registered aggregate publication')
    config=json.loads((Path(output_base)/'launch-config'/f'{job_id}.json').read_text())
    source_root=Path(source_root);receipt=Path(output_base)/'tracking'/f'{job_id}.json'
    stage='dev-evaluate' if job_id=='development-evaluate' else job_id
    # Carry the already verified scientific dependency target into child processes.
    environment={**os.environ,'PYTHONPATH':os.pathsep.join(filter(None,[os.environ.get('VPI_PACKAGE_TARGET'),str(source_root/'src'),os.environ.get('PYTHONPATH')]))}
    try:
        if not (Path(config['output_root'])/'evaluation_lock.json').is_file() or not (Path(config['report_root'])/'report_lock.json').is_file():
            raise ValueError('Publication requires sealed scientific and report outputs')
        subprocess.run([sys.executable,'-B',str(source_root/'scripts/prepare_tracking_runtime.py'),'--source-root',str(source_root)],
            timeout=330,check=True,env=environment)
        subprocess.run([sys.executable,'-B',str(source_root/'scripts/publish_next_stage.py'),'--stage',stage,
            '--evaluation-root',config['output_root'],'--report-root',config['report_root'],'--receipt',str(receipt)],
            timeout=200,check=True,env=environment)
    except Exception as error:
        record=json.loads(receipt.read_text()) if receipt.is_file() else {}
        record.update(status='pending',error_type=type(error).__name__,scientific_results_preserved=True)
        atomic_json(receipt,record)
    if not receipt.is_file():atomic_json(receipt,{'status':'pending','error_type':'MissingPublicationReceipt','scientific_results_preserved':True})
    record=json.loads(receipt.read_text())
    return {key:record[key] for key in ('status','task_id','task_url','error_type','scientific_results_preserved') if key in record}


def complete_job(plan,job_id,source_snapshot_sha256,*,output_base=Path('/kaggle/working'),tracking=None):
    """Write the small queue receipt only after every declared output marker exists.

This verifies completion markers, not an external download; artifact consumers
perform the complete scientific seal checks before using the retained resource.
"""
    output_base=Path(output_base)
    if len(source_snapshot_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_snapshot_sha256):
        raise ValueError('Completion requires the verified source snapshot digest')
    expected={name:spec for name,spec in plan['resources'].items() if spec.get('producer_job')==job_id}
    if not expected:raise ValueError('Queue job has no declared completion outputs')
    outputs={}
    for name,spec in expected.items():
        if len(spec['suffixes'])!=1:raise ValueError('Produced resource must have one declared output root')
        marker=output_base/_safe(spec['suffixes'][0])/_safe(spec['marker'])
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('Job is checkpointed or incomplete; a declared completion marker is missing')
        outputs[name]=hashlib.sha256(marker.read_bytes()).hexdigest()
    receipt={'schema':'vpi-current-job-completion/v1','status':'outputs_complete','job_id':job_id,
        'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'source_snapshot_sha256':source_snapshot_sha256,'outputs':outputs,'full_download_verified':False}
    if tracking is not None:receipt['tracking']=tracking
    job=next(row for row in plan['jobs'] if row['id']==job_id)
    if job.get('compact_checkpoints'):
        from post_completion_retention import verify_retention_receipt
        receipt['checkpoint_retention']=verify_retention_receipt(plan,job_id,source_snapshot_sha256,output_base=output_base)
    atomic_json(output_base/'job_completion.json',receipt)
    return receipt
