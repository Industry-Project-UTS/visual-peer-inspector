"""Notebook invocation receipts and explicit recovery, outside immutable science."""
from pathlib import Path
import time
import uuid
from current_notebook_runtime import atomic_json, writable_output
from current_stage_recovery import export_state


def run_stage(stage, cfg, operation, provenance):
    fallback = Path('/kaggle/working/runtime-receipts') if Path('/kaggle').is_dir() else Path.cwd()/'.local/runtime-receipts'
    try:
        output = writable_output(cfg['output_root'])
        receipts = writable_output(cfg.get('runtime_receipt_root', output.parent / 'runtime-receipts'))
        if output == receipts or output in receipts.parents:
            raise ValueError('Runtime receipts must stay outside the scientific output directory')
    except BaseException as error:
        atomic_json(fallback/(stage+'-configuration-failure.json'),
                    {'stage':stage,'status':'configuration_failed','error_type':type(error).__name__,
                     'scientific_run_started':False})
        raise
    started = time.monotonic()
    identifier = stage + '-' + uuid.uuid4().hex[:12]
    record = {'stage': stage, 'source_provenance': provenance, 'status': 'running',
              'runtime_scope': 'Notebook invocation for '+str(provenance.get('runtime_profile','registered CPU profile'))+'; neural timing requires a model-stage receipt'}
    atomic_json(receipts / (identifier + '.json'), record)
    try:
        result = operation()
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        _recovery(stage,cfg,output,provenance,receipts,identifier,record)
        atomic_json(receipts / (identifier + '.json'), record)
        raise
    record.update(status='completed', elapsed_seconds=time.monotonic()-started)
    _recovery(stage,cfg,output,provenance,receipts,identifier,record)
    atomic_json(receipts / (identifier + '.json'), record)
    return result


def _recovery(stage,cfg,output,provenance,receipts,identifier,record):
    directory = cfg.get('recovery_export_dir')
    if not directory:
        record['recovery'] = {'status':'not_exported','retention':'Configure recovery_export_dir and retain archive plus sidecar'}
        return
    try:
        # Do not silently duplicate a large feature cache or consume the session disk.
        maximum = int(cfg.get('recovery_max_bytes', 512*1024*1024))
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size > maximum:
            raise ValueError('Recovery exceeds the configured export size bound')
        directory = writable_output(directory); directory.mkdir(parents=True, exist_ok=True)
        record['recovery'] = export_state(output,directory/(identifier+'.zip.bin'),directory/(identifier+'.json'),
                                          stage=stage,provenance=provenance)
    except BaseException as error:
        record['recovery'] = {'status':'pending','error_type':type(error).__name__,
                              'scientific_files_preserved':True,'cross_session_recovery_guaranteed':False}
