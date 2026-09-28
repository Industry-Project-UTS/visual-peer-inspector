#!/usr/bin/env python3
"""Build neutral official object boxes from the dataset's per-object masks.

The scene2d metadata box list describes only a subset of objects. It cannot
establish physical-object correspondence. Missing mask members become explicit
zero-area boxes, which the existing matcher leaves unmatched.
"""
from pathlib import Path

from visual_peer_inspector.workflows.official_release import IndexedArchive, verify_official_membership
from visual_peer_inspector.workflows.primary_resource_release import _mask_box
from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, verified_unit, verify_seal
from vpi_next_evaluation.native_evaluator import GEOMETRY_SCHEMA, _contract
from vpi_next_evaluation.official import OFFICIAL
from vpi_next_evaluation.raw_reference import _index, prepare_raw_geometry


def mask_member(root, object_index, view_index):
    if object_index < 0 or view_index < 0:
        raise ValueError('Object and view indices must be nonnegative')
    name = 'object' if object_index == 0 else f'object.{object_index:03d}'
    return f'{root}/{name}/{name}_{view_index:04d}.png'


def _mask_box_or_missing(archive, member):
    rows = archive.database.execute('SELECT name FROM members WHERE name = ?', (member,)).fetchall()
    if not rows:
        return [0, 0, 0, 0], True, False
    if len(rows) != 1:
        raise ValueError('Duplicate indexed official object mask')
    box = _mask_box(archive.read(member))
    if len(box) != 4 or any(not isinstance(value, int) or value < 0 or value > 256 for value in box):
        raise ValueError('Official object mask has invalid box coordinates')
    empty = box == [0, 0, 0, 0]
    if not empty and (box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError('Official object mask has invalid box area')
    return box, False, empty


def prepare_official_mask_geometry(cfg):
    if cfg['role'] not in OFFICIAL:
        return prepare_raw_geometry(cfg)
    cfg = dict(cfg)
    rows, index_hash, identity = _index(cfg)
    contract = _contract(cfg['role'])
    raw, root = Path(cfg['raw_root']), Path(cfg['output_root'])
    master = {row['scene_id']: row for row in verify_official_membership(
        read_json(raw / 'toysad8k_scene_master.json'), contract)}
    binding = {'schema': GEOMETRY_SCHEMA, 'role': cfg['role'],
        'native_manifest_sha256': index_hash, 'raw_source_identity': identity,
        'administrative_sha256': {}, 'source_sha256': digest_file(__file__),
        'reference_box_source': 'per_object_segmentation_masks'}
    signature = bind(root, binding)
    if (root / 'artifact_lock.json').exists():
        verify_seal(root)
        return read_json(root / 'geometry_manifest.json')
    asset_index = {scene: row['source_base_asset_id'] for scene, row in rows.items()}
    missing_total = empty_total = 0
    with IndexedArchive(raw / 'toysAD8K.zip', raw / 'archive_index.sqlite3') as archive:
        for number, (scene, row) in enumerate(sorted(rows.items()), 1):
            unit = root / 'scenes' / scene
            if not verified_unit(unit, signature):
                source = master[row['source_scene_id']]
                ids = row['neutral']['object_ids']
                if (len(ids) != source['object_count'] or
                        ids != [f'O{i:02d}' for i in range(len(ids))] or
                        [item['object_index'] for item in source['objects']] != list(range(len(ids)))):
                    raise ValueError('Official object inventory differs from mask ordering')
                views, missing, empty = {}, 0, 0
                prefix = source['archive_members']['object_segmentation_directory']
                for view_index, view in enumerate(row['neutral']['view_ids']):
                    if view != f'V{view_index:02d}':
                        raise ValueError('Official view inventory is not ordered')
                    boxes = {}
                    for object_index, object_id in enumerate(ids):
                        member = mask_member(prefix, object_index, view_index)
                        box, absent, zero = _mask_box_or_missing(archive, member)
                        boxes[object_id] = box
                        missing += int(absent)
                        empty += int(zero)
                    views[view] = boxes
                with atomic_unit(unit, signature) as temporary:
                    atomic_json(temporary / 'geometry.json', {'scene': scene,
                        'asset_group': asset_index[scene], 'views': views,
                        'missing_mask_count': missing, 'empty_mask_count': empty})
            saved = read_json(unit / 'geometry.json')
            missing_total += saved['missing_mask_count']
            empty_total += saved['empty_mask_count']
            if number == 1 or number % 25 == 0:
                print(f'Neutral official mask geometry: {number}/{len(rows)} scenes', flush=True)
    if len(set(asset_index.values())) != contract.asset_group_count:
        raise ValueError('Neutral asset denominator differs')
    atomic_json(root / 'scene_index.json', asset_index)
    manifest = {**binding, 'scene_count': len(rows),
        'asset_group_count': contract.asset_group_count,
        'membership_sha256': contract.membership_sha256,
        'asset_index_sha256': digest_file(root / 'scene_index.json'),
        'missing_mask_count': missing_total, 'empty_mask_count': empty_total,
        'anomaly_outcomes_used': False, 'anomaly_labels_exported': False,
        'trusted_preparer_only': True}
    atomic_json(root / 'geometry_manifest.json', manifest)
    seal(root, [path for path in root.rglob('*') if path.is_file()])
    return manifest
