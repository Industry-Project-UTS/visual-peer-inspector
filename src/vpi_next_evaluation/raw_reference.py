"""Prepare geometry and later truth directly from the existing indexed dataset."""
from pathlib import Path
import json

from visual_peer_inspector.data.model_safe_package import membership_sha256, read_jsonl
from visual_peer_inspector.data.toysad_selection import deterministic_object_permutation
from visual_peer_inspector.workflows.primary_resource_release import verify_release_source, _required_by_scene, _mask_box
from visual_peer_inspector.workflows.official_release import IndexedArchive, verify_official_membership
from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, verified_unit, verify_seal
from .native_evaluator import _contract, GEOMETRY_SCHEMA, _prediction_boundary, _geometry, _emit_evaluator
from .official import OFFICIAL


def _index(cfg):
    from .native_rgb import read_rgb_index
    contract,index,raw,index_hash=read_rgb_index(cfg)
    if contract != _contract(cfg['role']): raise ValueError('Raw index converter cohort differs')
    cfg['raw_root']=raw
    rows={row['neutral']['opaque_scene_id']:row for row in index['scenes']}
    return rows,index_hash,index['source_identity']


def _administrative(cfg, scenes):
    """Verify the administrative mapping without opening its separate label vault."""
    from .native_rgb import _admin, EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256
    root,_=_admin(cfg['administrative_root'],'restricted_source_map.jsonl',
        EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256,['restricted_source_map.jsonl','required_source_members.jsonl'])
    checks={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'checksums.sha256').read_text().splitlines() if line}
    names=('restricted_source_map.jsonl','required_source_members.jsonl')
    for name in names:
        if checks.get(name)!=digest_file(root/name):raise ValueError('Administrative mapping checksum differs')
    selected={row['opaque_scene_id']:row for row in read_jsonl(root/names[0]) if row['opaque_scene_id'] in scenes}
    if set(selected)!=set(scenes):raise ValueError('Administrative mapping omits selected scenes')
    return selected,_required_by_scene(root,set(scenes)),{name:digest_file(root/name) for name in names}


def prepare_raw_geometry(cfg):
    """Extract neutral per-view reference boxes; no anomaly outcome is exported."""
    cfg=dict(cfg)
    rows,index_hash,identity=_index(cfg)
    contract=_contract(cfg['role']);raw=Path(cfg['raw_root']);root=Path(cfg['output_root'])
    admin,members,admin_hashes=({}, {}, {})
    master={}
    if cfg['role'] in OFFICIAL:
        # This mixed raw metadata is read only by this trusted resource preparer.
        # Only archive member addresses enter the geometry operation.
        master={row['scene_id']:row for row in verify_official_membership(read_json(raw/'toysad8k_scene_master.json'),contract)}
    else:
        admin,members,admin_hashes=_administrative(cfg,rows)
    binding={'schema':GEOMETRY_SCHEMA,'role':cfg['role'],'native_manifest_sha256':index_hash,
        'raw_source_identity':identity,'administrative_sha256':admin_hashes,'source_sha256':digest_file(__file__)}
    signature=bind(root,binding)
    if (root/'artifact_lock.json').exists():verify_seal(root);return read_json(root/'geometry_manifest.json')
    index={scene:row['source_base_asset_id'] for scene,row in rows.items()}
    with IndexedArchive(raw/'toysAD8K.zip',raw/'archive_index.sqlite3') as archive:
        for number,(scene,row) in enumerate(sorted(rows.items()),1):
            unit=root/'scenes'/scene
            if not verified_unit(unit,signature):
                ids=row['neutral']['object_ids'];views={view:{} for view in row['neutral']['view_ids']}
                if cfg['role'] in OFFICIAL:
                    source=master[row['source_scene_id']]
                    metadata=json.loads(archive.read(source['archive_members']['scene2d_metadata']))
                    for i,view in enumerate(row['neutral']['view_ids']):
                        boxes=metadata['bbox'][f'{i:04d}.png']
                        if len(boxes)!=len(ids):raise ValueError('Official reference geometry object count differs')
                        views[view]=dict(zip(ids,boxes))
                else:
                    trusted=admin[scene]
                    if trusted['source_scene_id']!=row['source_scene_id'] or trusted['source_base_asset_id']!=row['source_base_asset_id']:
                        raise ValueError('Administrative scene/asset differs from RGB source index')
                    for member in members[scene]:
                        if member['component']!='object_mask':continue
                        view,obj=member['view_id'],member['opaque_object_id']
                        if view not in views or obj not in ids or obj in views[view]:raise ValueError('Reference mask key is unknown or duplicated')
                        views[view][obj]=_mask_box(archive.read(member['source_member']))
                if any(set(values)!=set(ids) for values in views.values()):raise ValueError('Reference masks are incomplete')
                with atomic_unit(unit,signature) as temporary:
                    atomic_json(temporary/'geometry.json',{'scene':scene,'asset_group':index[scene],'views':views})
            if number==1 or number%25==0:print(f'Neutral reference geometry: {number}/{len(rows)} scenes',flush=True)
    if len(set(index.values()))!=contract.asset_group_count:raise ValueError('Neutral asset denominator differs')
    atomic_json(root/'scene_index.json',index)
    manifest={**binding,'scene_count':len(rows),'asset_group_count':contract.asset_group_count,
        'membership_sha256':contract.membership_sha256,'asset_index_sha256':digest_file(root/'scene_index.json'),
        'anomaly_outcomes_used':False,'anomaly_labels_exported':False,'trusted_preparer_only':True}
    atomic_json(root/'geometry_manifest.json',manifest)
    seal(root,[p for p in root.rglob('*') if p.is_file()])
    return manifest


def convert_raw_evaluator(cfg):
    """Open normalized anomaly flags only after both predictions verify."""
    cfg=dict(cfg)
    manifest,predictions=_prediction_boundary(cfg)
    geometry=_geometry(cfg)
    rows,index_hash,identity=_index(cfg)
    if read_json(geometry/'geometry_manifest.json')['native_manifest_sha256']!=index_hash:
        raise ValueError('Geometry does not bind this raw RGB source index')
    admin,_,hashes=_administrative(cfg,rows) if cfg['role'] not in OFFICIAL else ({},{},{})
    if read_json(geometry/'geometry_manifest.json')['administrative_sha256']!=hashes:
        raise ValueError('Geometry and labels use different administrative mappings')
    permutations={}
    for scene,row in rows.items():
        ids=row['neutral']['object_ids']
        permutation=list(range(len(ids)))
        if cfg['role'] not in OFFICIAL:
            permutation=list(deterministic_object_permutation(row['source_scene_id'],len(ids)))
            if admin[scene]['source_scene_id']!=row['source_scene_id'] or admin[scene]['source_base_asset_id']!=row['source_base_asset_id']:
                raise ValueError('Original source mapping differs from the RGB index')
            if admin[scene]['object_permutation_source_indices']!=permutation or admin[scene]['runtime_object_ids']!=ids:
                raise ValueError('Original runtime object permutation differs')
        permutations[scene]=permutation
    def labels():
        master={row['scene_id']:row for row in read_json(Path(cfg['raw_root'])/'toysad8k_scene_master.json')['scenes']}
        result={}
        for scene,row in rows.items():
            source=master[row['source_scene_id']];ids=row['neutral']['object_ids']
            flags=source['normalized_anomaly_flags']
            if len(flags)!=len(ids) or any(value not in (0,1,False,True) for value in flags):raise ValueError('Raw normalized anomaly flags differ from the object inventory')
            permutation=permutations[scene]
            entries=[{'object_id':obj,'is_anomaly':bool(flags[permutation[i]])} for i,obj in enumerate(ids)]
            result[scene]={'object_labels':entries,'anomaly_count':sum(item['is_anomaly'] for item in entries)}
        return result
    return _emit_evaluator(cfg,manifest,predictions,geometry,index_hash,labels,
        converter_sources={'raw_reference.py':digest_file(__file__)})
