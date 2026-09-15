"""Two-mode cache tests use generated pixels; optional parity reads no labels."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.web.core import InspectionRequest, Box, load_views, validate_result
from visual_peer_inspector.web.pipeline import UploadPipeline
from visual_peer_inspector.web.decision import decide_cached
from visual_peer_inspector.web.session_cache import InspectionError
from visual_peer_inspector.web.transport import write_request, read_request, write_result, read_result


class CountingModels:
    boxes = [(15,30,65,210),(90,30,140,210),(170,30,220,210)]
    def __init__(self, forbidden=False):
        self.calls = {'propose':0,'segment':0,'encode':0}
        self.forbidden=forbidden
    def count(self,key,n):
        if self.forbidden:
            pytest.fail('Cached-only inspection invoked a model: '+key)
        self.calls[key]+=n
    def propose(self,image,view_id):
        self.count('propose',1)
        return self.boxes,[]
    def segment(self,image,boxes):
        self.count('segment',len(boxes))
        masks=np.zeros((len(boxes),256,256),bool)
        for m,(x1,y1,x2,y2) in zip(masks,boxes):
            m[y1:y2,x1:x2]=True
        return masks,np.ones(len(boxes))*.95
    def encode(self,crops):
        self.count('encode',len(crops))
        vectors=[]
        for crop in crops:
            seed=int.from_bytes(hashlib.sha256(crop.tobytes()).digest()[:4],'little')
            vector=np.random.default_rng(seed).normal(size=384).astype(np.float32)
            vectors.append(vector/np.linalg.norm(vector))
        return EncodedBatch(np.asarray(vectors),tuple(np.tile(v,(256,1)) for v in vectors),16,384)
    def release(self):
        pass


@pytest.fixture
def inspection(tmp_path):
    files=[]
    for i in range(2):
        image=np.full((256,256,3),100+i*15,np.uint8)
        for j,(x1,y1,x2,y2) in enumerate(CountingModels.boxes):
            image[y1:y2,x1:x2]=[30+j*70,20+j*60,50+j*30]
        path=tmp_path/f'image{i}.png';Image.fromarray(image).save(path);files.append(str(path))
    return InspectionRequest(load_views(files))


def confirmed(inspection,result,unknown=False):
    boxes=tuple(replace(o.box,identity_id=('' if unknown and i==0 else 'I'+o.box.object_id)) for i,o in enumerate(result.observations))
    return replace(inspection,mode='assisted',corrections=boxes,cache_token=result.cache_token)


def test_identity_only_confirmation_and_mode_switch_call_no_models_across_pipeline_instances(inspection,tmp_path):
    root=tmp_path/'cache'; models=CountingModels()
    first=UploadPipeline(models,cache_root=root).inspect(inspection,lambda _:None)
    assert models.calls=={'propose':2,'segment':6,'encode':6}
    assert len(first.observations)==6
    second=UploadPipeline(CountingModels(forbidden=True),cache_root=root).inspect(confirmed(inspection,first),lambda _:None)
    assert second.mode=='assisted' and second.cache_token==first.cache_token
    assert second.provenance['cache']=={'reused_candidates':6,'encoded_candidates':0}
    back=UploadPipeline(CountingModels(forbidden=True),cache_root=root).inspect(replace(inspection,cache_token=first.cache_token),lambda _:None)
    assert back.ranking==first.ranking
    assert back.selected_identity_id==first.selected_identity_id


def test_box_edit_invalidates_only_one_candidate(inspection,tmp_path):
    models=CountingModels();pipeline=UploadPipeline(models,cache_root=tmp_path/'cache')
    first=pipeline.inspect(inspection,lambda _:None); corrected=confirmed(inspection,first)
    boxes=list(corrected.corrections); b=boxes[0];boxes[0]=replace(b,xyxy=(b.xyxy[0]+2,*b.xyxy[1:]))
    result=pipeline.inspect(replace(corrected,corrections=tuple(boxes)),lambda _:None)
    assert models.calls=={'propose':2,'segment':7,'encode':7}
    assert result.provenance['cache']=={'reused_candidates':5,'encoded_candidates':1}


def test_image_change_reuses_unchanged_view(inspection,tmp_path):
    models=CountingModels();pipeline=UploadPipeline(models,cache_root=tmp_path/'cache')
    first=pipeline.inspect(inspection,lambda _:None)
    changed=inspection.views[0].image.copy();changed.putpixel((0,0),(1,2,3))
    views=(replace(inspection.views[0],image=changed),inspection.views[1])
    result=pipeline.inspect(replace(inspection,views=views,cache_token=first.cache_token),lambda _:None)
    assert models.calls=={'propose':3,'segment':9,'encode':9}
    assert result.provenance['cache']=={'reused_candidates':3,'encoded_candidates':3}


def test_unknown_confirmed_identity_is_retained_unscored(inspection,tmp_path):
    pipeline=UploadPipeline(CountingModels(),cache_root=tmp_path/'cache')
    first=pipeline.inspect(inspection,lambda _:None)
    result=pipeline.inspect(confirmed(inspection,first,unknown=True),lambda _:None)
    unknown=[o for o in result.observations if o.identity_status=='unknown']
    assert len(unknown)==1 and unknown[0].score is None
    assert len(result.observations)==6


def test_fresh_assisted_mode_requires_confirmation_and_rich_transport_roundtrip(inspection,tmp_path):
    inspection=replace(inspection,mode='assisted')
    result=UploadPipeline(CountingModels(),cache_root=tmp_path/'cache').inspect(inspection,lambda _:None)
    assert not result.ranking and result.selected_identity_id is None
    assert all(o.identity_status=='unknown' for o in result.observations)
    assert result.score_metadata['suggested_identities']
    assert all(s['identity_id'] for s in result.score_metadata['suggested_identities'])
    assert all(o.identity_source=='unknown' for o in result.observations)
    job=tmp_path/'job';job.mkdir();write_request(job,inspection)
    restored=read_request(job);assert restored.session_id==inspection.session_id and restored.mode=='assisted'
    write_result(job,result);loaded=read_result(job,inspection)
    assert loaded.score_metadata==result.score_metadata and loaded.cache_token==result.cache_token
    assert loaded.observations[0].component_scores==result.observations[0].component_scores


@pytest.mark.parametrize('failure',['corrupt','token','binding','expired'])
def test_bad_cache_fails_without_any_neural_fallback(inspection,tmp_path,failure,monkeypatch):
    root=tmp_path/'cache';pipeline=UploadPipeline(CountingModels(),cache_root=root)
    first=pipeline.inspect(inspection,lambda _:None)
    replay=confirmed(inspection,first)
    if failure=='corrupt':
        next(root.rglob('*.npz')).write_bytes(b'corrupt')
    elif failure=='token':
        replay=replace(replay,cache_token='wrong')
    elif failure=='binding':
        monkeypatch.setattr('visual_peer_inspector.web.pipeline.cache_binding',lambda:'changed')
    else:
        path=pipeline.cache.path(inspection.session_id);state=pipeline.cache._read(path);state['expires_at']=0;pipeline.cache.save(path,state)
    with pytest.raises(InspectionError):
        UploadPipeline(CountingModels(forbidden=True),cache_root=root).inspect(replay,lambda _:None)


def test_reset_deletes_session_cache_only(inspection,tmp_path):
    pipeline=UploadPipeline(CountingModels(),cache_root=tmp_path/'cache')
    first=pipeline.inspect(inspection,lambda _:None)
    other=replace(inspection,session_id='another_session')
    second=pipeline.inspect(other,lambda _:None)
    pipeline.clear_session(inspection.session_id,first.cache_token)
    assert not pipeline.cache.path(inspection.session_id).exists()
    assert pipeline.cache.path(other.session_id).exists()
    assert second.cache_token!=first.cache_token


def test_real_subprocesses_reuse_cache_and_expiry_removes_owner_token(inspection):
    import sys
    from visual_peer_inspector.web.backend import SubprocessBackend
    from visual_peer_inspector.web.deployment import WorkerSettings
    from visual_peer_inspector.web.session_cache import SessionCache
    backend=SubprocessBackend(WorkerSettings(sys.executable,timeout_seconds=20),
        worker_module='tests.web_two_mode_worker_stub')
    first=backend.inspect(inspection,lambda _:None)
    assert first.provenance['test_model_calls']=={'propose':2,'segment':6,'encode':6}
    second=backend.inspect(confirmed(inspection,first),lambda _:None)
    assert second.provenance['test_model_calls']=={'propose':0,'segment':0,'encode':0}
    cache=SessionCache(backend._cache_directory.name)
    path=cache.path(inspection.session_id)
    state=cache._read(path);state['expires_at']=0;cache.save(path,state)
    backend.cleanup_expired_sessions()
    assert not path.exists() and inspection.session_id not in backend._sessions
    with pytest.raises(InspectionError,match='unavailable'):
        backend.inspect(confirmed(inspection,first),lambda _:None)


def test_runtime_mismatch_fails_before_model_snapshot(inspection,monkeypatch):
    from visual_peer_inspector.web.model_runtime import GPUModels, RuntimeUnavailable
    monkeypatch.delenv('VPI_DINO_PYTHON',raising=False)
    monkeypatch.setattr('visual_peer_inspector.web.encoder_worker.compatible_runtime',lambda:False)
    models=GPUModels()
    monkeypatch.setattr(models,'_snapshot',lambda _:pytest.fail('Mismatched runtime attempted a model lookup'))
    with pytest.raises(RuntimeUnavailable,match='dino_runtime'):
        models.encode([np.zeros((224,224,3),np.uint8)])


def test_isolated_encoder_exchange_is_offline_cpu_and_bounded(tmp_path,monkeypatch):
    import sys
    from visual_peer_inspector.web.model_runtime import GPUModels
    from visual_peer_inspector.web.encoder_worker import PINNED
    models=GPUModels()
    monkeypatch.setenv('VPI_DINO_PYTHON',sys.executable)
    monkeypatch.setattr(models,'_snapshot',lambda _:str(tmp_path))
    crop=np.full((224,224,3),17,np.uint8)
    def run(command,**kwargs):
        job=Path(command[command.index('--job')+1])
        assert kwargs['timeout']==180 and kwargs['check'] is True
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        assert kwargs['env']['HF_HUB_OFFLINE']=='1'
        with np.load(job/'input.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['crops'],crop[None])
        np.savez_compressed(job/'output.npz',global_features=np.ones((1,384),np.float32),patches=np.ones((1,256,384),np.float32))
        (job/'receipt.json').write_text(json.dumps({'runtime':PINNED,'feature_dimension':384,'grid':16}))
    monkeypatch.setattr('subprocess.run',run)
    batch=models.encode([crop])
    assert batch.global_features.shape==(1,384) and batch.patch_features[0].shape==(256,384)


def test_encoder_configuration_hash_is_checked_before_model_construction(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from visual_peer_inspector.web.encoder_worker import encode_cpu
    from visual_peer_inspector.web.session_cache import file_hash
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'fixed weights')
    config=tmp_path/'config.json';config.write_bytes(b'changed config')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    monkeypatch.setattr('visual_peer_inspector.web.encoder_worker.compatible_runtime',lambda:True)
    monkeypatch.setattr('visual_peer_inspector.web.deployment.pipeline_config',lambda:{'encoder':{
        'sha256':file_hash(weights),'config_sha256':hashlib.sha256(b'original config').hexdigest()}})
    monkeypatch.setattr('visual_peer_inspector.models.visual_encoder.FrozenVisualEncoder',lambda *a,**k:pytest.fail('Changed configuration reached model construction'))
    with pytest.raises(RuntimeError,match='checkpoint_integrity'):
        encode_cpu([],tmp_path)


def test_actual_sealed_decision_parity_without_labels():
    value=os.environ.get('VPI_WEB_PARITY_CONFIG')
    if not value:
        pytest.skip('Set VPI_WEB_PARITY_CONFIG for frozen cached scientific parity')
    from vpi_object_bank.extraction import load_observations
    cfg=json.loads(Path(value).read_text())
    manifest=json.loads((Path(cfg['prepared_root'])/'input_manifest.json').read_text())
    scene=manifest['scenes'][0]
    raw=json.loads((Path(cfg['scoring_root'])/'scenes'/f'{scene}.json').read_text())
    previous=json.loads((Path(cfg['previous_prediction_root'])/'scenes'/scene/'predictions.json').read_text())
    prepared=json.loads((Path(cfg['prepared_root'])/'scenes'/f'{scene}.json').read_text())
    neutral={(r['view'],r['candidate_id']):r['neutral_id'] for r in prepared['observations'] if r['route']=='automatic'}
    for schedule in ['two-primary','three-primary','five-primary']:
        views=manifest['view_sets'][schedule]
        rows={r['view']:r for r in raw if r['route']=='automatic' and r['view'] in views}
        obs,_=load_observations(cfg['feature_root'],scene,'automatic',views,neutral)
        for mode,oldroute,method in [('automatic','automatic_predicted','winner_cluster_consensus'),('assisted','automatic_reference','object_bank_top2')]:
            actual=decide_cached(rows,obs,views=views,mode=mode)['ranking']
            expected=next(s for s in previous['settings'] if s['route']==oldroute and s['view_set_id']==schedule)['rankings'][method]
            def clean(items):
                return [{k:o[k] for k in ('id','score','members','support','fusion_observations')} for o in sorted(items,key=lambda o:o['id'])]
            assert clean(actual)==clean(expected)
