from pathlib import Path
import pytest
from vpi_object_bank.prediction import fuse_observations
from vpi_object_bank import reporting
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal


def row(view, candidate, identity, score):
    return {'view_id': view, 'candidate_id': candidate, 'object_id': identity,
            'within_view': {'score': score}}


def test_rank_fusion_retains_ties_and_is_order_invariant():
    observations = [row('V0','a','A',3), row('V0','b','B',2), row('V0','c','C',1),
                    row('V1','d','A',2), row('V1','e','B',3), row('V1','f','C',1)]
    result, ranks = fuse_observations(observations, method='within_view', fusion='mean')
    values = {r['id']: r['score'] for r in result}
    assert values == {'A':.75,'B':.75,'C':0}
    reverse, reversed_ranks = fuse_observations(observations[::-1], method='within_view', fusion='mean')
    assert {r['id']:r['score'] for r in reverse} == values
    assert reversed_ranks == ranks


def test_missing_scores_and_unknown_observations_remain_explicit():
    observations = [row('V0','a','A',1), row('V0','b','B',2), row('V0','c',None,3),
                    row('V1','d','A',None), row('V1','e',None,None)]
    result, ranks = fuse_observations(observations, method='within_view', fusion='mean')
    assert len(result) == 4
    assert next(o for o in result if o['id']=='A')['support'] == 1
    assert next(o for o in result if o['id']=='unmatched-V1-e')['score'] is None
    assert not any(k.startswith('V1/') for k in ranks)


def test_top_two_uses_only_available_strongest_observations():
    observations = []
    for view, a, b, c in [('V0',3,2,1),('V1',3,2,1),('V2',1,3,2)]:
        observations += [row(view,'a','A',a),row(view,'b','B',b),row(view,'c','C',c)]
    mean,_=fuse_observations(observations,method='within_view',fusion='mean')
    top2,_=fuse_observations(observations,method='within_view',fusion='top2')
    assert next(o for o in mean if o['id']=='A')['score'] == pytest.approx(2/3)
    assert next(o for o in top2 if o['id']=='A')['score'] == 1
    assert next(o for o in top2 if o['id']=='A')['fusion_observations'] == 2


def test_equal_scores_share_rank_without_candidate_id_tie_break():
    observations=[row('V0',str(i),str(i),2.0) for i in range(4)]
    result,_=fuse_observations(observations,method='within_view',fusion='mean')
    assert {o['score'] for o in result} == {.5}


def test_report_failure_preserves_successful_scientific_seals(tmp_path,monkeypatch):
    cfg=report_input_fixture(tmp_path)
    before={key:digest_file(Path(cfg[key])/lock) for key,lock in [('evaluation_root','evaluation_lock.json'),('prediction_root','prediction_lock.json')]}
    def broken(*args):
        raise RuntimeError('injected chart failure')
    monkeypatch.setattr(reporting,'aggregate_figures',broken)
    with pytest.raises(RuntimeError,match='chart failure'):
        reporting.build_report(cfg,Path(__file__).resolve().parents[1])
    assert read_json(Path(cfg['report_root'])/'run_status.json')['status']=='failed'
    assert not (Path(cfg['report_root'])/'report_lock.json').exists()
    for key,lock in [('evaluation_root','evaluation_lock.json'),('prediction_root','prediction_lock.json')]:
        assert digest_file(Path(cfg[key])/lock)==before[key]


def report_input_fixture(tmp_path):
    cfg={key:str(tmp_path/key) for key in ('evaluation_root','prediction_root','feature_root','prepared_root','model_root','report_root')}
    def package(key,filename,payload,lock):
        root=Path(cfg[key]);root.mkdir(exist_ok=True)
        atomic_json(root/filename,payload)
        seal(root,[root/filename],name=lock)
        return digest_file(root/lock)
    model=package('model_root','index.json',{'schema':1},'artifact_lock.json')
    prepared=package('prepared_root','input_manifest.json',{'model_input_lock_sha256':model},'artifact_lock.json')
    features=package('feature_root','run_binding.json',{'model_input_lock_sha256':model,'prepared_lock_sha256':prepared},'feature_lock.json')
    predictions=package('prediction_root','run_binding.json',{'feature_lock_sha256':features,'prepared_lock_sha256':prepared},'prediction_lock.json')
    package('evaluation_root','run_binding.json',{'prediction_lock_sha256':predictions,'prepared_lock_sha256':prepared},'evaluation_lock.json')
    return cfg


def test_report_rejects_swapped_even_self_consistently_sealed_images(tmp_path):
    cfg=report_input_fixture(tmp_path)
    assert len(reporting.verified_report_inputs(cfg))==5
    model=Path(cfg['model_root'])
    atomic_json(model/'index.json',{'schema':1,'swapped':True})
    seal(model,[model/'index.json'])
    with pytest.raises(ValueError,match='provenance'):
        reporting.verified_report_inputs(cfg)
