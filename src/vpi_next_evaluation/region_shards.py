"""Merge fixed LA/SAM scene partitions without copying RGB or treating gaps as empty."""
from pathlib import Path
import shutil
from uuid import uuid4
from vpi_object_bank.storage import atomic_json,bind,digest,digest_file,read_json,seal,terminal_status,verify_seal
from .native_regions import (SCHEMA, preparation_gates, fixed_vision, _rgb_records,
    _read_records, read_region_resource)
from vpi_remaining_evaluation.workbench import resolve_input_root

SHARD_SCHEMA='vpi-current-native-region-shard/v1'


def read_region_shard(root, expected, contract):
    """Validate exact partition membership and all normalized geometry."""
    root=Path(root);lock=verify_seal(root,name='region_shard_lock.json')
    manifest=read_json(root/'region_shard_manifest.json');binding=read_json(root/'run_binding.json')
    receipt=read_json(root/'region_receipt.json');partition=binding.get('partition',{})
    count,index=partition.get('count'),partition.get('index')
    if isinstance(count,bool) or isinstance(index,bool) or not isinstance(count,int) or not isinstance(index,int) or count<2 or not 0<=index<count:
        raise ValueError('Invalid region shard partition binding')
    scenes=sorted({s for s,v in expected})[index::count]
    if not scenes or partition.get('scenes')!=scenes or manifest.get('partition')!=partition or manifest.get('scenes')!=scenes:
        raise ValueError('Region shard does not match its declared partition')
    if manifest.get('schema')!=SHARD_SCHEMA or manifest.get('role')!=binding.get('role') or manifest.get('cohort_id')!=contract.cohort_id or binding.get('cohort_id')!=contract.cohort_id or manifest.get('scene_count')!=contract.scene_count or manifest.get('image_count')!=len(expected):
        raise ValueError('Region shard changes the complete cohort denominator')
    if manifest.get('model_input_sha256')!=binding.get('model_input_sha256') or not manifest.get('model_input_sha256') or manifest.get('vision')!=fixed_vision() or binding.get('vision')!=fixed_vision() or manifest.get('mask_hash_scheme')!='sha256-row-major-uint8':
        raise ValueError('Region shard source/model/hash contract differs')
    paths=manifest['records'];keys={key for key in expected if key[0] in set(scenes)}
    if len(paths)!=len(set(paths)) or set(paths)!={f'regions/{s}/{v}.json' for s,v in keys}:
        raise ValueError('Region shard omits or duplicates selected views')
    required={'run_binding.json','region_shard_manifest.json','region_receipt.json',*paths}
    if 'model_runtime.json' in lock['files']:
        runtime=verify_seal(root,name='model_runtime_lock.json')
        if set(runtime['files'])!={'model_runtime.json'}:raise ValueError('Region runtime lock inventory differs')
        required|={'model_runtime.json','model_runtime_lock.json'}
    elif not binding.get('reuse_lock_sha256'):
        raise ValueError('Inferred region shard omits verified model runtime provenance')
    if set(lock['files'])!=required or receipt.get('status')!='shard_complete' or receipt.get('binding_sha256')!=digest(binding) or lock.get('metadata',{}).get('binding_sha256')!=digest(binding):
        raise ValueError('Region shard seal or completion receipt differs')
    if receipt.get('completed_views')!=len(keys) or receipt.get('image_count')!=len(expected) or receipt.get('scene_count')!=contract.scene_count or receipt.get('remaining_views')!=0:
        raise ValueError('Region shard completion counts differ')
    rows=_read_records(root,paths,expected)
    if set(rows)!=keys:raise ValueError('Region shard candidate record inventory differs')
    return rows,binding,manifest


def merge_region_shards(cfg,source_root):
    """Publish a full native-region resource only after every partition verifies."""
    contract,gates=preparation_gates(cfg,source_root)
    rgb_root=resolve_input_root(Path(cfg['rgb_root']),'package_record.json')
    expected,audit=_rgb_records(rgb_root,contract)
    common=None;partitions=set();partition_count=None;paths={};locks=[];runtime_source=None;runtime_sha=None
    roots=list(map(Path,cfg['region_shards']))
    if not roots:raise ValueError('No region shards supplied')
    for root in roots:
        rows,binding,manifest=read_region_shard(root,expected,contract)
        if binding.get('gates')!=gates or binding.get('model_input_sha256')!=digest(audit):
            raise ValueError('Region shards differ from current gates or RGB package')
        partition=binding['partition'];base={k:v for k,v in binding.items() if k!='partition'}
        if common is not None and base!=common:raise ValueError('Region shards have different scientific inputs/configuration')
        common=base
        if partition_count is not None and partition['count']!=partition_count:raise ValueError('Region shard partition counts differ')
        partition_count=partition['count']
        if partition['index'] in partitions or set(rows)&set(paths):raise ValueError('Region shards overlap')
        partitions.add(partition['index'])
        for key in rows:paths[key]=root/f'regions/{key[0]}/{key[1]}.json'
        locks.append(digest_file(root/'region_shard_lock.json'))
        if (root/'model_runtime.json').exists():
            value=digest_file(root/'model_runtime.json')
            if runtime_sha is not None and value!=runtime_sha:raise ValueError('Region shards use different model/runtime bytes')
            runtime_sha=value;runtime_source=root
    if partitions!=set(range(partition_count)) or set(paths)!=set(expected):
        raise ValueError('Region shard union is incomplete; unprocessed views cannot become empty results')
    output=Path(cfg['output_root'])
    binding={**common,'region_shard_locks':sorted(locks),'merge_source_sha256':digest_file(__file__)}
    signature=bind(output,binding)
    if (output/'artifact_lock.json').exists():
        read_region_resource(output,rgb_root,contract)
        return read_json(output/'region_receipt.json')
    required=sum(path.stat().st_size for path in paths.values())
    if required>int(cfg.get('maximum_output_bytes',18*1024**3)) or shutil.disk_usage(output).free<required+128*1024**2:
        raise OSError('Region merge exceeds output or available storage budget')
    with terminal_status(output,'native-region-shard-merge'):
        released=[]
        for (scene,view),source in sorted(paths.items()):
            target=output/f'regions/{scene}/{view}.json';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if digest_file(target)!=digest_file(source):raise ValueError('Existing merged region changed; do not overwrite')
            else:
                temporary=target.with_name(target.name+'.partial')
                if temporary.exists():temporary.rename(target.with_name(target.name+'.interrupted-'+uuid4().hex[:8]))
                shutil.copyfile(source,temporary)
                if digest_file(source)!=digest_file(temporary):raise ValueError('Region copy changed bytes')
                temporary.replace(target)
            released.append(target)
        manifest={'schema':SCHEMA,'role':cfg['role'],'cohort_id':contract.cohort_id,'scene_count':contract.scene_count,
            'image_count':len(expected),'scenes':sorted({s for s,v in expected}),'records':[p.relative_to(output).as_posix() for p in released],
            'model_input_sha256':digest(audit),'vision':fixed_vision(),'mask_hash_scheme':'sha256-row-major-uint8'}
        receipt={'schema':SCHEMA,'status':'complete','scene_count':contract.scene_count,'image_count':len(expected),
            'completed_views':len(paths),'remaining_views':0,'merged_shards':len(roots),'binding_sha256':signature,
            'no_scoring_or_dino':True,'neural_inference_run':False,'rgb_files_copied':0}
        atomic_json(output/'region_manifest.json',manifest);atomic_json(output/'region_receipt.json',receipt)
        files=[output/'run_binding.json',output/'region_manifest.json',output/'region_receipt.json',*released]
        if runtime_source is not None:
            for name in ('model_runtime.json','model_runtime_lock.json'):
                target=output/name
                if target.exists() and digest_file(target)!=digest_file(runtime_source/name):raise ValueError('Merged model runtime changed')
                if not target.exists():shutil.copyfile(runtime_source/name,target)
                files.append(target)
        seal(output,files,metadata={'schema':SCHEMA,'binding_sha256':signature})
        read_region_resource(output,rgb_root,contract)
    return receipt
