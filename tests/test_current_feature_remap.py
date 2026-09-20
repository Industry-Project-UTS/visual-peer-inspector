"""Hash/geometry reuse and shard conservation tests; no inference or truth."""
from pathlib import Path
import numpy as np
import pytest
from test_current_native_cache import fixture
from vpi_next_evaluation import feature_remap as remap, native_cache as native, cache_shards
from vpi_object_bank.storage import atomic_json, digest, digest_file, read_json, seal, verify_seal
from vpi_object_bank.inputs import _candidate_id
from visual_peer_inspector.models.sam2 import binary_mask_sha256


def prepare_remap_fixture(fixture, tmp_path, monkeypatch):
    cfg, contract, scenes, views = fixture
    region, features = cfg['region_root'], cfg['feature_root']; rows = {}
    signature = digest(read_json(features/'run_binding.json'))
    for scene in scenes:
        source_rows = read_json(region/'scenes'/f'{scene}.json')
        for row in source_rows:
            if row['route'] != 'automatic': continue
            unit = features/'views'/scene/f'automatic-{row["view"]}'
            meta = read_json(unit/'observations.json'); crosswalk = {}
            native_objects = []
            for obj in row['objects']:
                old = obj['candidate_id']; hashed = _candidate_id(scene,row['view'],'automatic',old)
                crosswalk[old] = hashed
                native_objects.append({'object_id':old,'box_xyxy':obj['box'],'mask_rle':obj['mask'],
                    'mask_sha256':binary_mask_sha256(native.decode_binary_mask(obj['mask']))})
                obj['candidate_id'] = hashed
            for item in meta['objects']: item['candidate_id'] = crosswalk[item['candidate_id']]
            atomic_json(unit/'observations.json',meta)
            seal(unit,[unit/'observations.json',unit/'features.npz'],metadata={'binding_sha256':signature})
            rows[scene,row['view']] = {'objects':native_objects,'rgb_sha256':row['rgb_sha256'],'component':'fixture'}
        atomic_json(region/'scenes'/f'{scene}.json',source_rows)
    seal(region,[p for p in region.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    value = read_json(features/'run_binding.json');value['model_input_lock_sha256']=digest_file(region/'artifact_lock.json')
    atomic_json(features/'run_binding.json',value);signature=digest(value)
    for unit in features.glob('views/*/*'):
        seal(unit,[unit/'observations.json',unit/'features.npz'],metadata={'binding_sha256':signature})
    seal(features,[p for p in features.rglob('*') if p.is_file() and p.name!='feature_lock.json'],name='feature_lock.json',metadata={'binding_sha256':signature})
    monkeypatch.setattr(remap,'_native_inventory',lambda cfg:(rows,'d'*64,scenes,views,{}))
    config={'source_region_root':region,'source_feature_root':features,'expected_source_region_lock_sha256':digest_file(region/'artifact_lock.json'),
        'expected_source_feature_lock_sha256':digest_file(features/'feature_lock.json'),'remap_root':tmp_path/'audit',
        'feature_root':tmp_path/'import','selected_views':views[:1]}
    return config,rows,scenes,views


def test_exact_bytes_indices_and_selected_scope_survive_import(fixture,tmp_path,monkeypatch):
    cfg,rows,scenes,views = prepare_remap_fixture(fixture,tmp_path,monkeypatch)
    receipt=remap.audit_remap(cfg)
    assert receipt['verified_units']==2 and receipt['verified_candidates']==6
    assert receipt['remaining_feature_units']==4 and not receipt['neural_inference_run']
    result=remap.import_remapped({**cfg,'max_new_units':1})
    assert result['status']=='checkpointed'
    assert not (cfg['feature_root']/'feature_lock.json').exists()
    first=next(cfg['feature_root'].glob('views/*/*/features.npz'));before=(digest_file(first),first.stat().st_mtime_ns)
    assert remap.import_remapped(cfg)['status']=='import_complete'
    assert (digest_file(first),first.stat().st_mtime_ns)==before
    lock=verify_seal(cfg['feature_root'],name='feature_shard_lock.json')
    assert lock['metadata']['required_units']==6
    for unit in cfg['feature_root'].glob('views/*/*'):
        relative=unit.relative_to(cfg['feature_root'])
        assert digest_file(unit/'features.npz')==digest_file(cfg['source_feature_root']/relative/'features.npz')
        assert {r['candidate_id'] for r in read_json(unit/'observations.json')['objects']}=={'C0','C1','C2'}
        assert unit.name=='automatic-'+views[0]


def test_remap_rejects_wrong_native_mask_even_when_candidate_hash_matches(fixture,tmp_path,monkeypatch):
    cfg,rows,scenes,views=prepare_remap_fixture(fixture,tmp_path,monkeypatch)
    rows[scenes[0],views[0]]['objects'][0]['mask_sha256']='0'*64
    with pytest.raises(ValueError,match='packed-mask'):remap.audit_remap(cfg)
    assert not (cfg['remap_root']/'artifact_lock.json').exists()


def test_remap_rejects_changed_source_after_audit(fixture,tmp_path,monkeypatch):
    cfg,_,_,_=prepare_remap_fixture(fixture,tmp_path,monkeypatch);remap.audit_remap(cfg)
    relative = read_json(cfg['remap_root']/'mapping.json')['units'][0]['relative']
    (cfg['source_feature_root']/relative/'features.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):remap.import_remapped(cfg)


def test_native_shards_merge_only_complete_nonoverlapping_cohort(fixture,tmp_path,monkeypatch):
    cfg,contract,scenes,views=fixture
    roots=[]
    for index in range(2):
        root=tmp_path/f'cache-shard-{index}';roots.append(root)
        options={**cfg,'output_root':root,'shard_count':2,'shard_index':index}
        assert native.assemble_native(options,'.')['status']=='shard_complete'
        before={str(p.relative_to(root)):(digest_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        native.assemble_native(options,'.')
        assert before=={str(p.relative_to(root)):(digest_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        assert not (root/'artifact_lock.json').exists()
    monkeypatch.setattr(cache_shards,'_gates',lambda cfg,root:(contract,'f'*64))
    options={'role':'development','cache_shards':roots[:1],'output_root':tmp_path/'missing'}
    with pytest.raises(ValueError,match='complete fixed cohort'):cache_shards.merge_cache_shards(options,'.')
    with pytest.raises(ValueError,match='overlap'):cache_shards.merge_cache_shards({**options,'cache_shards':[roots[0],roots[0]]},'.')
    output=tmp_path/'merged'; options={**options,'cache_shards':roots,'output_root':output}
    assert cache_shards.merge_cache_shards(options,'.')['scene_count']==2
    verify_seal(output)
    assert read_json(output/'cohort_manifest.json')['scenes']==scenes
    before=digest_file(output/'artifact_lock.json');cache_shards.merge_cache_shards(options,'.')
    assert digest_file(output/'artifact_lock.json')==before
