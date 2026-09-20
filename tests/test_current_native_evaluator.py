"""Exercise the native JSONL bridge and the real current prediction boundary."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from test_next_original_evaluation import resources, ROOT
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal
from vpi_next_evaluation import native_evaluator as bridge, original
from visual_peer_inspector.models.sam2 import encode_binary_mask

sys.path.insert(0, str(ROOT / 'scripts'))
from current_truth_conversion import _scored_geometry, convert_raw_evaluator_with_typed_rejections


@pytest.fixture
def native(resources, tmp_path):
    cfg = {**resources, 'role':'calibration'}
    cfg.pop('scenes')
    cache = Path(cfg['cache_root'])
    contract = bridge._contract('calibration')
    index = read_json(Path(cfg['evaluator_root'])/'scene_index.json')
    boxes = [[0,0,10,10],[15,0,25,10],[30,0,40,10]]
    rows = {name:[] for name in ('crosswalk_private.jsonl','geometry_private.jsonl','labels_private.jsonl')}
    for scene, asset in index.items():
        base = {'opaque_scene_id':scene,'cohort_id':contract.cohort_id,'release_id':contract.release_id}
        rows['crosswalk_private.jsonl'].append({**base,'base_asset_id':asset,
            'object_id_map':{f'R{i}':i for i in range(3)}, 'view_id_map':{f'V{i:02d}':f'V{i:02d}.png' for i in range(20)}})
        rows['geometry_private.jsonl'].append({**base,'views':[{'view_id':f'V{i:02d}','source_filename':f'V{i:02d}.png','boxes_xyxy':boxes} for i in range(20)]})
        rows['labels_private.jsonl'].append({**base,'anomaly_count':1,'object_labels':[{'object_id':f'R{i}','is_anomaly':i==2} for i in range(3)]})
        payload=read_json(cache/'scenes'/f'{scene}.json')
        payload['evidence_views']={view:{'candidates':{f'C{i}':{'box':box} for i,box in enumerate(boxes)}} for view in payload['score_rows']}
        atomic_json(cache/'scenes'/f'{scene}.json',payload)
    seal(cache,[p for p in cache.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    raw=tmp_path/'native';raw.mkdir()
    for name, values in rows.items():
        (raw/name).write_text(''.join(json.dumps(row)+'\n' for row in values))
    atomic_json(raw/'evaluator_manifest.json',{'schema_version':bridge.RAW_SCHEMA,
        'cohort_id':contract.cohort_id,'source_membership_sha256':contract.membership_sha256,
        'scene_count':contract.scene_count,'image_count':contract.image_count,
        'files':[{'path':name,'size_bytes':(raw/name).stat().st_size,'sha256':digest_file(raw/name)} for name in rows]})
    cfg.update(native_evaluator_root=str(raw),expected_evaluator_manifest_sha256=digest_file(raw/'evaluator_manifest.json'),
        geometry_root=str(tmp_path/'geometry'))
    return cfg


def test_geometry_does_not_open_or_require_label_bytes(native):
    labels=Path(native['native_evaluator_root'])/'labels_private.jsonl'
    labels.unlink()
    result=bridge.prepare_geometry({**native,'output_root':native['geometry_root']})
    assert result['anomaly_labels_opened'] is False
    geometry=Path(native['geometry_root'])
    assert (geometry/'scene_index.json').read_bytes()==(Path(native['evaluator_root'])/'scene_index.json').read_bytes()
    assert not any('label' in p.name for p in geometry.rglob('*'))
    assert bridge.prepare_geometry({**native,'output_root':native['geometry_root']})==result


def test_actual_current_prediction_then_native_conversion(native,tmp_path):
    bridge.prepare_geometry({**native,'output_root':native['geometry_root']})
    original.predict_current('09a',native,ROOT)
    cfg={**native,'prediction_root':native['output_root'],'output_root':str(tmp_path/'converted')}
    manifest=bridge.convert_evaluator(cfg)
    assert manifest['schema']==original.EVALUATOR_SCHEMA
    result=Path(cfg['output_root']);verify_seal(result)
    for scene in read_json(result/'scene_index.json'):
        truth=read_json(result/'scenes'/f'{scene}.json')
        assert truth['labels']=={'R0':0,'R1':0,'R2':1}
        assert set(truth['correspondence'].values())=={'R0','R1','R2'}
    assert bridge.convert_evaluator(cfg)==manifest
    evaluated=original.evaluate_predictions('09b',{**native,'prediction_root':native['output_root'],
        'evaluator_root':str(result),'output_root':str(tmp_path/'evaluated')},ROOT)
    assert evaluated['status']=='protocol_complete'


def test_truth_conversion_accepts_only_explicit_tiny_mask_rejections(native,monkeypatch):
    cache=Path(native['cache_root']);scene=sorted(read_json(cache/'cohort_manifest.json')['scenes'])[0]
    payload=read_json(cache/'scenes'/f'{scene}.json')
    view=sorted(payload['evidence_views'])[0]
    tiny=np.zeros((10,10),dtype=bool);tiny[0,0:3]=True
    payload['evidence_views'][view]['candidates']['C-rejected']={
        'mask':encode_binary_mask(tiny),'box':[50,50,52,52],
        'feature_status':'rejected','feature_failure':'mask_below_minimum_foreground',
        'foreground_pixels':3,
    }
    payload['invalid_candidate_observations']=1
    for other in read_json(cache/'cohort_manifest.json')['scenes']:
        path=cache/'scenes'/f'{other}.json'
        row=payload if other==scene else read_json(path)
        row.setdefault('invalid_candidate_observations',0)
        atomic_json(path,row)
    filtered=_scored_geometry(payload)
    assert 'C-rejected' in payload['evidence_views'][view]['candidates']
    assert 'C-rejected' not in filtered['evidence_views'][view]['candidates']
    from vpi_next_evaluation import raw_reference
    original_read=bridge.read_json
    seen={}
    def convert(cfg):
        seen['payload']=bridge.read_json(cache/'scenes'/f'{scene}.json')
        return {'status':'converted'}
    monkeypatch.setattr(raw_reference,'convert_raw_evaluator',convert)
    assert convert_raw_evaluator_with_typed_rejections(native)=={'status':'converted'}
    assert 'C-rejected' not in seen['payload']['evidence_views'][view]['candidates']
    assert bridge.read_json is original_read


def test_truth_conversion_rejects_untyped_unscored_geometry(native):
    cache=Path(native['cache_root']);scenes=read_json(cache/'cohort_manifest.json')['scenes']
    payload=read_json(cache/'scenes'/f'{scenes[0]}.json');view=sorted(payload['evidence_views'])[0]
    payload['evidence_views'][view]['candidates']['C-untyped']={'box':[50,50,52,52]}
    payload['invalid_candidate_observations']=1
    with pytest.raises(ValueError,match='complete typed rejection'):
        _scored_geometry(payload)


def test_truth_is_not_opened_before_complete_predictions(native,tmp_path,monkeypatch):
    opened=[]
    monkeypatch.setattr(bridge,'_registered',lambda *args: opened.append(args))
    with pytest.raises((ValueError,FileNotFoundError)):
        bridge.convert_evaluator({**native,'prediction_root':str(tmp_path/'missing'),'output_root':str(tmp_path/'out')})
    assert opened==[]


def test_geometry_rejects_resealed_bad_object_mapping(native):
    raw=Path(native['native_evaluator_root'])
    path=raw/'crosswalk_private.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['object_id_map']['R2']=0
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest=read_json(raw/'evaluator_manifest.json')
    for entry in manifest['files']:
        if entry['path']==path.name:entry.update(sha256=digest_file(path),size_bytes=path.stat().st_size)
    atomic_json(raw/'evaluator_manifest.json',manifest)
    native['expected_evaluator_manifest_sha256']=digest_file(raw/'evaluator_manifest.json')
    with pytest.raises(ValueError,match='bijection'):
        bridge.prepare_geometry({**native,'output_root':native['geometry_root']})


def test_neutral_matching_is_one_to_one_permutation_invariant_and_unmatched():
    reference={'B':[20,0,30,10],'A':[0,0,10,10]}
    objects=[{'candidate_id':'c','box':[0,0,10,10]}, {'candidate_id':'d','box':[0,0,10,10]},
             {'candidate_id':'z','box':[80,80,90,90]}]
    first=bridge._correspondence(objects,reference)
    assert first==bridge._correspondence(list(reversed(objects)),reference)
    assert list(first.values()).count('A')==1 and first['z'] is None
