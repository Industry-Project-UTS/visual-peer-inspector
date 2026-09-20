"""Fixture-only phase/resume and model-package validation; no neural execution."""
from dataclasses import replace, asdict
from pathlib import Path
from hashlib import sha256
import shutil
import json
import numpy as np
from PIL import Image
import pytest
from vpi_next_evaluation import native_regions as region
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal
from visual_peer_inspector.data.model_safe_package import PackageAudit, membership_sha256, canonical_json_bytes
from visual_peer_inspector.models.sam2 import ContractSamWorker

ROOT = Path(__file__).resolve().parents[1]


def package(tmp_path):
    root=tmp_path/'rgb'; scene='S000001'; views=[f'V{i:02d}' for i in range(20)]
    neutral={'schema_version':1,'release_id':region.CONTRACTS['development'].release_id,'opaque_scene_id':scene,
        'object_count':3,'object_ids':['O00','O01','O02'],'view_ids':views,'media_root':f'media/{scene}'}
    media=[]
    for view in views:
        path=root/f'media/{scene}/rgb/{view}.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.full((256,256,3),50,dtype=np.uint8)).save(path);media.append(path)
    (root/'neutral_corpus_manifest.jsonl').write_bytes(canonical_json_bytes(neutral)+b'\n')
    files=[root/'neutral_corpus_manifest.jsonl',*media]
    checks=''.join(f'{digest_file(p)}  {p.relative_to(root)}\n' for p in sorted(files)).encode()
    (root/'content_checksums.sha256').write_bytes(checks)
    contract=replace(region.CONTRACTS['development'],scene_count=1,image_count=20,asset_group_count=1,membership_sha256=membership_sha256([scene]))
    audit=PackageAudit(contract.cohort_id,contract.release_id,1,20,20,256,256,'RGB',sum(p.stat().st_size for p in media),
        digest_file(root/'neutral_corpus_manifest.jsonl'),contract.membership_sha256,sha256(checks).hexdigest(),20,0,0,0,True)
    atomic_json(root/'package_record.json',asdict(audit))
    return root,contract


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    rgb,contract=package(tmp_path)
    monkeypatch.setattr(region,'preparation_gates',lambda cfg,source:(contract,{'fixture_only':True}))
    cfg={'role':'development','rgb_root':rgb,'output_root':tmp_path/'out','minimum_free_bytes':1,'maximum_new_views':20}
    calls={'proposals':0,'regions':0}; failure={'la':None,'sam':False}
    class LA:
        def run_job(self,job,attempt):
            calls['proposals']+=1; error=failure['la']
            return {'image_sha256':job.image_sha256,'configuration_sha256':region._configs()[0].sha256,
                'status':'error' if error else 'ok','error_type':error,'boxes_xyxy':[[10,10,40,40]],'runtime_seconds':.1}
    class SAM(ContractSamWorker):
        def process(self,job,attempt):
            calls['regions']+=1
            if failure['sam']: raise PermissionError('fixture')
            return super().process(job,attempt)
    def factory(phase,cfg,source): return LA() if phase=='proposals' else SAM(region._configs()[1])
    def run(config=cfg): return region.extract_regions(config,ROOT,worker_factory=factory,model_verifier=lambda *args:{'fixture':'no weights loaded'})
    return cfg,contract,calls,failure,run


def test_plan_has_no_neural_constructor_and_complete_twenty_views(fixture):
    cfg,_,calls,_,_=fixture
    plan=region.plan_regions(cfg,ROOT)
    assert plan['image_count']==20 and plan['missing_region_views']==20 and not plan['neural_execution_performed']
    assert calls=={'proposals':0,'regions':0}


def test_real_package_separate_phases_and_exact_resume(fixture):
    cfg,contract,calls,_,run=fixture
    first=run({**cfg,'maximum_new_views':3});assert first['remaining_views']==17
    receipt=run();assert receipt['status']=='complete' and calls=={'proposals':20,'regions':20}
    before=digest_file(cfg['output_root']/'artifact_lock.json')
    assert run()==receipt and digest_file(cfg['output_root']/'artifact_lock.json')==before
    records,_=region.read_region_resource(cfg['output_root'],cfg['rgb_root'],contract)
    obj=next(iter(records.values()))['objects'][0]
    assert obj['mask_sha256']!=obj['source_mask_sha256']


def test_sam_interruption_reuses_all_proposals(fixture):
    cfg,_,calls,failure,run=fixture
    failure['sam']=True
    with pytest.raises(PermissionError):run({**cfg,'maximum_new_views':2})
    assert calls['proposals']==2 and not (cfg['output_root']/'artifact_lock.json').exists()
    failure['sam']=False;run({**cfg,'maximum_new_views':2})
    assert calls['proposals']==2 and region.plan_regions(cfg,ROOT)['completed_region_views']==2


def test_known_empty_proposals_are_terminal_and_preserved(fixture):
    cfg,contract,calls,failure,run=fixture;failure['la']='NoValidBoxes'
    receipt=run();assert receipt['status']=='complete' and calls=={'proposals':40,'regions':0}
    records,_=region.read_region_resource(cfg['output_root'],cfg['rgb_root'],contract)
    assert all(not row['objects'] and row['terminal_failure']=='NoValidBoxes' for row in records.values())


def test_infrastructure_errors_never_become_empty_candidates(fixture):
    cfg,_,calls,failure,run=fixture;failure['la']='OutOfMemoryError'
    with pytest.raises(RuntimeError,match='Fatal'):run()
    assert calls['proposals']==1 and region.plan_regions(cfg,ROOT)['completed_proposal_views']==0
    assert read_json(cfg['output_root']/'run_status.json')['status']=='failed'


def test_changed_config_and_corrupt_checkpoint_stop(fixture):
    cfg,_,_,_,run=fixture;run({**cfg,'maximum_new_views':1})
    with pytest.raises(ValueError,match='different|changed'):run({**cfg,'attempt_timeout_seconds':50})
    path=cfg['output_root']/'checkpoints/regions/S000001/V00/region.json';path.write_text('{}')
    with pytest.raises(ValueError):run()


def test_mount_path_relocation_keeps_region_identity(fixture,tmp_path):
    cfg,_,_,_,run=fixture;run()
    moved=tmp_path/'new-mount';shutil.copytree(cfg['rgb_root'],moved)
    assert run({**cfg,'rgb_root':moved})['status']=='complete'


def test_model_input_extra_label_like_file_stops_before_workers(fixture):
    cfg,_,calls,_,run=fixture;(cfg['rgb_root']/'labels.json').write_text('{}')
    with pytest.raises(ValueError,match='allowlist'):run()
    assert calls['proposals']==0


def test_region_mask_corruption_is_rejected_even_after_reseal(fixture):
    cfg,contract,_,_,run=fixture;run();root=cfg['output_root'];lock=read_json(root/'artifact_lock.json')
    path=root/'regions/S000001/V00.json';row=read_json(path);row['objects'][0]['mask_sha256']='0'*64;atomic_json(path,row)
    seal(root,[root/p for p in lock['files']],metadata=lock['metadata'])
    with pytest.raises(ValueError,match='normalized mask'):region.read_region_resource(root,cfg['rgb_root'],contract)
