"""Privileged source indexing and model-only RGB release for fixed cohort roles.

The private index retains source paths; the released model-input directory has
only the canonical opaque manifest, checksums, package record and RGB images.
No scene2D/3D metadata, object masks or anomaly-label file is opened here.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from hashlib import sha256
import json
import os
import sqlite3
import time

from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest, digest_file,
    free_space, read_json, seal, terminal_status, verified_unit, verify_seal)
from vpi_remaining_evaluation.workbench import resolve_input_root
from visual_peer_inspector.data.model_safe_package import (PackageAudit, _validate_rgb_bytes,
    canonical_json_bytes, membership_sha256, verify_checksum_manifest, verify_model_input_package,
    EXPECTED_SELECTION_CHECKSUM_MANIFEST_SHA256, EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256,
    EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256)
from visual_peer_inspector.data.toysad_selection import validate_model_facing_media_record
from visual_peer_inspector.workflows.primary_resource_release import (verify_release_source,
    _role_rows, _required_by_scene, _safe_neutral, _atomic_bytes)
from visual_peer_inspector.workflows.official_release import (IndexedArchive,
    official_opaque_scene_id, verify_official_membership)
from .native_regions import preparation_gates
from .official import OFFICIAL
from vpi_remaining_evaluation.gates import CONTRACTS

SCHEMA = 'vpi-current-private-rgb-index/v1'


def _safe_member(value):
    path = PurePosixPath(value)
    if not isinstance(value, str) or path.is_absolute() or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('Unsafe archive member')
    return value


def _source(cfg):
    root = resolve_input_root(Path(cfg['raw_root']), 'package-checksums.sha256')
    identity = verify_release_source(root)
    expected = cfg.get('expected_source_checksums_sha256', EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256)
    if identity['package_checksums_sha256'] != expected:
        raise ValueError('Raw source differs from the reviewed mounted dataset version')
    return root, identity


def _admin(root, marker, expected, required):
    root = resolve_input_root(Path(root), marker)
    if digest_file(root/'checksums.sha256') != expected:
        raise ValueError('Original selection/admin resource differs from frozen input')
    return root, verify_checksum_manifest(root, manifest_name='checksums.sha256', required_names=required)


def prepare_rgb_index(cfg, source_root):
    """Administrative projection only; official master answers never enter output."""
    contract = {**CONTRACTS, **OFFICIAL}.get(cfg['role'])
    if contract is None: raise ValueError('Unknown original or official role')
    # Metadata preparation establishes the exposure index itself, so it must
    # precede the exposure gate. No RGB/model work is performed here.
    gates = {'role': cfg['role'], 'cohort_id': contract.cohort_id}
    raw, identity = _source(cfg)
    entries = []; administrative = {}
    if cfg['role'] in OFFICIAL:
        master = read_json(raw/'toysad8k_scene_master.json')
        rows = verify_official_membership(master, contract)
        for number, row in enumerate(rows, 1):
            scene = official_opaque_scene_id(contract.role, number)
            neutral = {'schema_version': 1, 'release_id': contract.release_id,
                'opaque_scene_id': scene, 'object_count': int(row['object_count']),
                'object_ids': [f'O{i:02d}' for i in range(int(row['object_count']))],
                'view_ids': [f'V{i:02d}' for i in range(20)], 'media_root': f'media/{scene}'}
            entries.append({'neutral': neutral, 'source_scene_id': str(row['scene_id']),
                'source_base_asset_id': str(row['base_asset_id']),
                'rgb_members': {f'V{i:02d}': _safe_member(f"{row['archive_members']['rgb_directory']}/{i:04d}.png") for i in range(20)}})
    else:
        selection, administrative['selection'] = _admin(cfg['selection_root'], 'experiment_roles.jsonl',
            EXPECTED_SELECTION_CHECKSUM_MANIFEST_SHA256, ['experiment_roles.jsonl','neutral_corpus_manifest.jsonl'])
        admin, administrative['administrative'] = _admin(cfg['administrative_root'], 'restricted_source_map.jsonl',
            EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256, ['restricted_source_map.jsonl','required_source_members.jsonl'])
        neutral, trusted, _ = _role_rows(selection_root=selection, administrative_root=admin, contract=contract)
        members = _required_by_scene(admin, {row['opaque_scene_id'] for row in neutral})
        for row in neutral:
            scene = row['opaque_scene_id']; source = trusted[scene]
            rgb = [item for item in members[scene] if item['component']=='rgb']
            if len(rgb) != 20 or len({item['view_id'] for item in rgb}) != 20:
                raise ValueError('Original RGB source rows omit or duplicate views')
            entries.append({'neutral': _safe_neutral(row, contract), 'source_scene_id': str(source['source_scene_id']),
                'source_base_asset_id': str(source['source_base_asset_id']),
                'trusted': {k: str(source[k]) for k in ('source_scene_id','source_base_asset_id','opaque_asset_group_id')},
                'rgb_members': {item['view_id']: _safe_member(item['source_member']) for item in rgb}})
    scenes = [row['neutral']['opaque_scene_id'] for row in entries]
    if len(scenes)!=contract.scene_count or membership_sha256(scenes)!=(contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('RGB index changes original model membership')
    total = 0
    with sqlite3.connect(f'file:{raw / "archive_index.sqlite3"}?mode=ro', uri=True) as db:
        all_members = []
        for row in entries:
            validate_model_facing_media_record(row['neutral'], expected_release_id=contract.release_id)
            if set(row['rgb_members']) != set(row['neutral']['view_ids']): raise ValueError('RGB view inventory differs')
            for member in row['rgb_members'].values():
                all_members.append(member)
                values = db.execute('SELECT uncompressed_size,is_directory,is_symlink,unsafe_reason FROM members WHERE name=?',(member,)).fetchall()
                if len(values)!=1 or values[0][1] or values[0][2] or values[0][3]: raise ValueError('Unsafe or duplicate indexed RGB member')
                total += int(values[0][0])
        if len(all_members)!=len(set(all_members)): raise ValueError('RGB archive member reused across cohort images')
    record = {'schema': SCHEMA, 'role': cfg['role'], 'cohort_id': contract.cohort_id,
        'source_identity': identity, 'scenes': entries, 'estimated_rgb_bytes': total}
    binding = {'schema': SCHEMA,'gates': gates,'source_identity': identity,'administrative': administrative,
        'index_sha256': digest(record),'implementation_sha256': digest_file(Path(source_root)/'src/vpi_next_evaluation/native_rgb.py')}
    root = Path(cfg['output_root']); bind(root,binding)
    if (root/'artifact_lock.json').exists():
        verify_seal(root)
        if read_json(root/'rgb_index.json')!=record: raise ValueError('Private RGB index changed')
    else:
        atomic_json(root/'rgb_index.json',record)
        seal(root,[root/'rgb_index.json',root/'run_binding.json'])
    return {'status':'private_index_complete','scene_count':len(entries),'image_count':len(entries)*20,
        'estimated_rgb_bytes':total,'index_lock_sha256':digest_file(root/'artifact_lock.json'),
        'contains_source_paths':True,'labels_opened':False,'rgb_copied':False}


def read_rgb_index(cfg):
    """Verify private metadata provenance without requiring a future exposure gate."""
    contract={**CONTRACTS,**OFFICIAL}.get(cfg['role'])
    if contract is None: raise ValueError('Unknown original or official role')
    index_root=resolve_input_root(Path(cfg['rgb_index_root']),'rgb_index.json')
    lock=verify_seal(index_root)
    if set(lock['files'])!={'rgb_index.json','run_binding.json'}: raise ValueError('Unexpected private index files')
    record=read_json(index_root/'rgb_index.json'); index_binding=read_json(index_root/'run_binding.json')
    if record.get('schema')!=SCHEMA or record.get('role')!=cfg['role'] or record.get('cohort_id')!=contract.cohort_id:
        raise ValueError('Private index belongs to another cohort')
    if index_binding.get('index_sha256')!=digest(record) or index_binding.get('gates')!={'role':cfg['role'],'cohort_id':contract.cohort_id}:
        raise ValueError('Index input gates changed')
    scenes=[]
    for row in record['scenes']:
        allowed={'neutral','source_scene_id','source_base_asset_id','rgb_members'} | ({'trusted'} if cfg['role'] not in OFFICIAL else set())
        if set(row)!=allowed: raise ValueError('Unexpected private index scene fields')
        validate_model_facing_media_record(row['neutral'],expected_release_id=contract.release_id)
        scenes.append(row['neutral']['opaque_scene_id'])
        if set(row['rgb_members'])!=set(row['neutral']['view_ids']): raise ValueError('Index omits views')
        for member in row['rgb_members'].values(): _safe_member(member)
    if len(scenes)!=contract.scene_count or len(scenes)!=len(set(scenes)) or membership_sha256(scenes)!=(contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Private index membership differs')
    raw,identity=_source(cfg)
    if identity!=record['source_identity']: raise ValueError('Private index raw source changed')
    return contract,record,raw,digest_file(index_root/'artifact_lock.json')


def _verified_index(cfg,source_root):
    contract,gates=preparation_gates(cfg,source_root)
    verified,record,raw,index_hash=read_rgb_index(cfg)
    if verified!=contract: raise ValueError('Prepared and released RGB cohort contracts differ')
    return contract,record,raw,index_hash,gates


def release_rgb(cfg,source_root):
    started=time.monotonic()
    contract,index,raw,index_hash,gates=_verified_index(cfg,source_root)
    root=Path(cfg['output_root']); model=root/'model-input'
    maximum_output=cfg.get('maximum_output_bytes',18*1024**3)
    if isinstance(maximum_output,bool) or not isinstance(maximum_output,int) or maximum_output<1:
        raise ValueError('maximum_output_bytes must be a positive integer')
    # The RGB release retains both atomic checkpoints and canonical media. Add
    # a conservative manifest/receipt allowance; check before reading RGB bytes.
    estimated_output=int(index['estimated_rgb_bytes'])*2+contract.scene_count*20*16*1024+1024**2
    actual_output=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    if max(estimated_output,actual_output)>maximum_output:
        raise ValueError('RGB release exceeds maximum_output_bytes before extraction')
    binding={'schema':SCHEMA,'rgb_index_lock_sha256':index_hash,'gates':gates,
        'implementation_sha256':digest_file(Path(source_root)/'src/vpi_next_evaluation/native_rgb.py')}
    signature=bind(root,binding)
    if (root/'release_lock.json').exists():
        verify_seal(root,name='release_lock.json'); verify_model_input_package(model,contract=contract)
        return read_json(root/'release_receipt.json')
    maximum=cfg.get('maximum_new_scenes',25)
    if isinstance(maximum,bool) or not isinstance(maximum,int) or maximum<1: raise ValueError('maximum_new_scenes must be positive')
    free_space(root,estimated_output)
    processed=0
    with terminal_status(root,'native-rgb'):
        with IndexedArchive(raw/'toysAD8K.zip',raw/'archive_index.sqlite3') as archive:
            for row in index['scenes']:
                scene=row['neutral']['opaque_scene_id']; unit=root/'checkpoints'/scene
                if verified_unit(unit,signature): continue
                if processed>=maximum: break
                with atomic_unit(unit,signature) as temp:
                    media=[]
                    for view,member in sorted(row['rgb_members'].items()):
                        content=archive.read(member); mode=_validate_rgb_bytes(content,(256,256))
                        relative=f'media/{scene}/rgb/{view}.png'; _atomic_bytes(temp/relative,content)
                        media.append({'path':relative,'sha256':sha256(content).hexdigest(),'size_bytes':len(content),'media_mode':mode})
                    atomic_json(temp/'scene.json',{'neutral':row['neutral'],'media':media})
                processed+=1
                print(json.dumps({'event':'rgb_scene_checkpoint','completed_this_call':processed,'limit':maximum}),flush=True)
        complete=[row['neutral']['opaque_scene_id'] for row in index['scenes'] if verified_unit(root/'checkpoints'/row['neutral']['opaque_scene_id'],signature)]
        receipt={'status':'rgb_complete' if len(complete)==contract.scene_count else 'checkpointed',
            'scene_count':contract.scene_count,'completed_scenes':len(complete),'remaining_scenes':contract.scene_count-len(complete),
            'labels_opened':False,'elapsed_seconds_this_call':time.monotonic()-started,'rgb_index_lock_sha256':index_hash}
        if len(complete)==contract.scene_count:
            states=[read_json(root/'checkpoints'/scene/'scene.json') for scene in complete]
            media=[item for state in states for item in state['media']]
            for scene,state in zip(complete,states):
                for item in state['media']: _atomic_bytes(model/item['path'],(root/'checkpoints'/scene/item['path']).read_bytes())
            _atomic_bytes(model/'neutral_corpus_manifest.jsonl',b''.join(canonical_json_bytes(state['neutral'])+b'\n' for state in states))
            paths=['neutral_corpus_manifest.jsonl',*[item['path'] for item in media]]
            checks=''.join(f'{digest_file(model/path)}  {path}\n' for path in sorted(paths)).encode()
            _atomic_bytes(model/'content_checksums.sha256',checks)
            modes=Counter(item['media_mode'] for item in media)
            audit=PackageAudit(cohort_id=contract.cohort_id,release_id=contract.release_id,scene_count=contract.scene_count,
                view_count=20,image_count=len(media),width=256,height=256,
                media_mode=next(iter(modes)) if len(modes)==1 else 'mixed',total_media_bytes=sum(item['size_bytes'] for item in media),
                manifest_sha256=digest_file(model/'neutral_corpus_manifest.jsonl'),membership_sha256=membership_sha256(complete),
                content_checksums_sha256=sha256(checks).hexdigest(),verified_image_count=len(media),
                duplicate_archive_member_count=0,duplicate_package_path_count=0,decoded_image_failure_count=0,all_checks_passed=True)
            _atomic_bytes(model/'package_record.json',canonical_json_bytes(asdict(audit))+b'\n')
            verify_model_input_package(model,contract=contract)
            receipt['model_package_record_sha256']=digest_file(model/'package_record.json')
            atomic_json(root/'release_receipt.json',receipt)
            seal(root,[root/'run_binding.json',root/'release_receipt.json',*[p for p in model.rglob('*') if p.is_file()]],name='release_lock.json')
        else: atomic_json(root/'checkpoint_receipt.json',receipt)
    return receipt
