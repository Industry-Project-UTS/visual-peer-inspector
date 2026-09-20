"""Tiny indexed raw references; no original-role dataset or protected labels."""
from pathlib import Path
from dataclasses import replace
import json
import pytest
from test_current_native_rgb import fixture as rgb_fixture, ROOT
from vpi_next_evaluation import raw_reference as raw, native_rgb as rgb, native_evaluator as evaluator
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal
from visual_peer_inspector.data.toysad_selection import deterministic_object_permutation


@pytest.fixture
def fixture(rgb_fixture,tmp_path,monkeypatch):
    cfg,contract,paths=rgb_fixture
    monkeypatch.setattr(raw,'_contract',lambda role:contract)
    monkeypatch.setattr(evaluator,'_contract',lambda role:contract)
    admin=paths['administration'];path=admin/'restricted_source_map.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row['object_permutation_source_indices']=list(deterministic_object_permutation(row['source_scene_id'],3))
        row['runtime_object_ids']=['O00','O01','O02']
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (admin/'checksums.sha256').write_text(''.join(f'{digest_file(admin/name)}  {name}\n' for name in ['restricted_source_map.jsonl','required_source_members.jsonl']))
    monkeypatch.setattr(rgb,'EXPECTED_ADMINISTRATIVE_CHECKSUM_MANIFEST_SHA256',digest_file(admin/'checksums.sha256'))
    master=read_json(paths['source']/'toysad8k_scene_master.json')
    for row in master['scenes']:row['normalized_anomaly_flags']=[1,0,0]
    atomic_json(paths['source']/'toysad8k_scene_master.json',master)
    (paths['source']/'package-checksums.sha256').write_text(''.join(f'{digest_file(paths["source"]/name)}  {name}\n' for name in ['archive_index.sqlite3','toysAD8K.zip','toysad8k_scene_master.json']))
    cfg['expected_source_checksums_sha256']=digest_file(paths['source']/'package-checksums.sha256')
    rgb.prepare_rgb_index(cfg,ROOT)
    result={**cfg,'rgb_index_root':cfg['output_root'],'geometry_root':tmp_path/'geometry','output_root':tmp_path/'geometry'}
    return result,contract,paths


def test_original_geometry_has_opaque_ids_and_no_outcomes(fixture,monkeypatch):
    cfg,contract,paths=fixture
    original=Path.read_text
    def guard(path,*args,**kwargs):
        if path.name in {'toysad8k_scene_master.json','calibration_labels.jsonl'}:raise AssertionError('Original geometry must not open answers')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guard)
    result=raw.prepare_raw_geometry(cfg)
    assert not result['anomaly_labels_exported'] and result['scene_count']==2
    root=cfg['geometry_root'];verify_seal(root)
    index=read_json(root/'scene_index.json');assert set(index.values())=={'asset-0','asset-1'}
    for scene in index:
        geometry=read_json(root/'scenes'/scene/'geometry.json')
        assert set(geometry)=={'scene','asset_group','views'}
        assert set(geometry['views']['V00'])=={'O00','O01','O02'}
        assert geometry['views']['V00']['O00']==[20,30,60,80]
    assert raw.prepare_raw_geometry(cfg)==result


def _label_adapter(monkeypatch,cfg,contract):
    # Prediction validation is exercised without mocks in test_current_native_evaluator.
    # Here isolate the independently trusted source-permutation adapter.
    monkeypatch.setattr(raw,'_prediction_boundary',lambda config:({'scenes':['S001','S002']},{'both':'f'*64}))
    captured={}
    def emit(config,manifest,predictions,geometry,source,load_labels,**kwargs):
        captured.update(load_labels());captured['sources']=kwargs['converter_sources'];return captured
    monkeypatch.setattr(raw,'_emit_evaluator',emit)
    return captured


def test_original_label_permutation_is_exact_and_bound_to_source(fixture,monkeypatch):
    cfg,contract,_=fixture;raw.prepare_raw_geometry(cfg);captured=_label_adapter(monkeypatch,cfg,contract)
    result=raw.convert_raw_evaluator(cfg)
    for i,scene in enumerate(['S001','S002']):
        permutation=list(deterministic_object_permutation(f'source-{i}',3))
        assert result[scene]['object_labels']==[{'object_id':f'O{j:02d}','is_anomaly':permutation[j]==0} for j in range(3)]
    assert result['sources']=={'raw_reference.py':digest_file(raw.__file__)}


def test_missing_predictions_stops_before_any_raw_index_or_label_access(fixture,tmp_path,monkeypatch):
    cfg,_,_=fixture;called=[]
    monkeypatch.setattr(raw,'_index',lambda config:called.append('index'))
    with pytest.raises((KeyError,FileNotFoundError,ValueError)):
        raw.convert_raw_evaluator({**cfg,'cache_root':tmp_path/'missing','method_freeze_root':tmp_path/'missing','prediction_root':tmp_path/'missing'})
    assert called==[]


def test_corrupt_administrative_input_rejected_before_master_labels(fixture,monkeypatch):
    cfg,contract,paths=fixture;raw.prepare_raw_geometry(cfg);_label_adapter(monkeypatch,cfg,contract)
    (paths['administration']/'restricted_source_map.jsonl').write_text('{}\n')
    original=raw.read_json
    def guard(path):
        if Path(path).name=='toysad8k_scene_master.json':raise AssertionError('Labels opened before invalid admin rejected')
        return original(path)
    monkeypatch.setattr(raw,'read_json',guard)
    with pytest.raises(ValueError,match='checksum'):raw.convert_raw_evaluator(cfg)


def test_corrupt_index_rejected_before_master_labels(fixture,monkeypatch):
    cfg,contract,_=fixture;raw.prepare_raw_geometry(cfg);_label_adapter(monkeypatch,cfg,contract)
    (cfg['rgb_index_root']/'rgb_index.json').write_text('{}')
    with pytest.raises(ValueError):raw.convert_raw_evaluator(cfg)


def test_wrong_original_permutation_is_rejected_before_master_labels(fixture,monkeypatch):
    cfg,contract,_=fixture;raw.prepare_raw_geometry(cfg);_label_adapter(monkeypatch,cfg,contract)
    original=raw._administrative
    def changed(config,scenes):
        admin,members,hashes=original(config,scenes)
        admin['S001']['object_permutation_source_indices']=[-1,0,1]
        return admin,members,hashes
    monkeypatch.setattr(raw,'_administrative',changed)
    read=raw.read_json
    def guard(path):
        if Path(path).name=='toysad8k_scene_master.json':raise AssertionError('Labels opened before permutation rejection')
        return read(path)
    monkeypatch.setattr(raw,'read_json',guard)
    with pytest.raises(ValueError,match='permutation'):raw.convert_raw_evaluator(cfg)


def test_index_must_match_source_model_role_membership(fixture,monkeypatch):
    cfg,contract,_=fixture;index=read_json(cfg['rgb_index_root']/'rgb_index.json')
    index['role']='unseen';atomic_json(cfg['rgb_index_root']/'rgb_index.json',index)
    lock=read_json(cfg['rgb_index_root']/'artifact_lock.json');seal(cfg['rgb_index_root'],[cfg['rgb_index_root']/name for name in lock['files']])
    with pytest.raises(ValueError,match='another cohort'):raw.prepare_raw_geometry(cfg)


def test_official_source_and_model_memberships_and_label_order(tmp_path,monkeypatch):
    from test_official_release import _fixture_source,_fixture_contract
    source,master=_fixture_source(tmp_path);contract=_fixture_contract()
    atomic_json(source/'toysad8k_scene_master.json',master)
    (source/'package-checksums.sha256').write_text(''.join(f'{digest_file(source/name)}  {name}\n' for name in ['archive_index.sqlite3','toysAD8K.zip','toysad8k_scene_master.json']))
    monkeypatch.setattr(rgb,'OFFICIAL',{'seen':contract});monkeypatch.setattr(raw,'OFFICIAL',{'seen':contract})
    monkeypatch.setattr(raw,'_contract',lambda role:contract);monkeypatch.setattr(evaluator,'_contract',lambda role:contract)
    cfg={'role':'seen','raw_root':source,'expected_source_checksums_sha256':digest_file(source/'package-checksums.sha256'),
        'output_root':tmp_path/'index'}
    rgb.prepare_rgb_index(cfg,ROOT);record=read_json(cfg['output_root']/'rgb_index.json')
    assert record['scenes'][0]['neutral']['opaque_scene_id']=='S1000001'
    assert 'normalized_anomaly_flags' not in json.dumps(record)
    cfg.update(rgb_index_root=cfg['output_root'],geometry_root=tmp_path/'geometry',output_root=tmp_path/'geometry')
    manifest=raw.prepare_raw_geometry(cfg)
    assert manifest['membership_sha256']==contract.membership_sha256
    assert read_json(cfg['geometry_root']/'scenes/S1000001/geometry.json')['views']['V00']['O00']==[10,20,60,80]
    _label_adapter(monkeypatch,cfg,contract)
    truth=raw.convert_raw_evaluator(cfg)
    assert truth['S1000001']['object_labels']==[{'object_id':'O00','is_anomaly':True},{'object_id':'O01','is_anomaly':False},{'object_id':'O02','is_anomaly':False}]
    monkeypatch.setattr(rgb,'OFFICIAL',{'seen':replace(contract,model_membership_sha256='0'*64)})
    with pytest.raises(ValueError,match='membership'):rgb.read_rgb_index(cfg)
