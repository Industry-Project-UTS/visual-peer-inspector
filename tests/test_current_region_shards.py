"""Bounded LA/SAM region partitions with fixture workers, never real models."""
from dataclasses import replace,asdict
from pathlib import Path
from hashlib import sha256
import shutil
import numpy as np
from PIL import Image
import pytest
from test_current_native_regions import package
from visual_peer_inspector.data.model_safe_package import membership_sha256,canonical_json_bytes,PackageAudit
from visual_peer_inspector.models.sam2 import ContractSamWorker
from vpi_object_bank.storage import atomic_json,digest_file,read_json,seal,verify_seal
from vpi_next_evaluation import native_regions as region,region_shards as shards,native_cache

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def two_scenes(tmp_path,monkeypatch):
    root,contract=package(tmp_path)
    first=__import__('json').loads((root/'neutral_corpus_manifest.jsonl').read_text());second={**first,'opaque_scene_id':'S000002','media_root':'media/S000002'}
    for view in first['view_ids']:
        target=root/'media/S000002/rgb'/f'{view}.png';target.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.full((256,256,3),70,dtype=np.uint8)).save(target)
    (root/'neutral_corpus_manifest.jsonl').write_bytes(canonical_json_bytes(first)+b'\n'+canonical_json_bytes(second)+b'\n')
    media=sorted((root/'media').glob('*/rgb/*.png'));files=[root/'neutral_corpus_manifest.jsonl',*media]
    checks=''.join(f'{digest_file(p)}  {p.relative_to(root)}\n' for p in sorted(files)).encode();(root/'content_checksums.sha256').write_bytes(checks)
    contract=replace(contract,scene_count=2,image_count=40,membership_sha256=membership_sha256(['S000001','S000002']))
    audit=PackageAudit(contract.cohort_id,contract.release_id,2,20,40,256,256,'RGB',sum(p.stat().st_size for p in media),
        digest_file(root/'neutral_corpus_manifest.jsonl'),contract.membership_sha256,sha256(checks).hexdigest(),40,0,0,0,True)
    atomic_json(root/'package_record.json',asdict(audit))
    gates={'fixture_only':True}
    monkeypatch.setattr(region,'preparation_gates',lambda cfg,source:(contract,gates))
    monkeypatch.setattr(shards,'preparation_gates',lambda cfg,source:(contract,gates))
    cfg={'role':'development','rgb_root':root,'output_root':tmp_path/'out','minimum_free_bytes':1,
        'maximum_new_views':20,'shard_count':2,'shard_index':0}
    seen=[];fail={'sam':False}
    class LA:
        def run_job(self,job,attempt):
            seen.append(('la',job.opaque_scene_id,job.view_id))
            return {'image_sha256':job.image_sha256,'configuration_sha256':region._configs()[0].sha256,
                'status':'ok','error_type':None,'boxes_xyxy':[[10,10,40,40]],'runtime_seconds':.1}
    class SAM(ContractSamWorker):
        def process(self,job,attempt):
            seen.append(('sam',job.opaque_scene_id,job.view_id))
            if fail['sam']:raise PermissionError('fixture stop')
            return super().process(job,attempt)
    def factory(phase,cfg,source):return LA() if phase=='proposals' else SAM(region._configs()[1])
    def run(options):return region.extract_regions(options,ROOT,worker_factory=factory,model_verifier=lambda *args:{'fixture':'no models'})
    return cfg,contract,seen,fail,run


def test_partition_limits_exact_resume_and_complete_merge(two_scenes,tmp_path):
    cfg,contract,seen,_,run=two_scenes;roots=[]
    for index in range(2):
        root=tmp_path/f'shard-{index}';roots.append(root);opts={**cfg,'shard_index':index,'output_root':root}
        first=run({**opts,'maximum_new_views':2})
        assert first['status']=='checkpointed' and first['image_count']==40 and first['remaining_views']==18
        assert read_json(root/'run_status.json')['status']=='checkpointed'
        result=run(opts)
        assert result['status']=='shard_complete' and result['completed_views']==20 and result['image_count']==40
        assert not (root/'artifact_lock.json').exists()
        before={p.relative_to(root).as_posix():(digest_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        assert run(opts)==result
        assert before=={p.relative_to(root).as_posix():(digest_file(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        with pytest.raises(ValueError,match='different'):run({**opts,'shard_index':1-index})
    assert len(seen)==80 and sum(scene=='S000001' for _,scene,_ in seen)==40
    output=tmp_path/'merge';opts={**cfg,'output_root':output,'region_shards':roots}
    result=shards.merge_region_shards(opts,ROOT)
    assert result['completed_views']==40 and result['rgb_files_copied']==0
    rows,_=region.read_region_resource(output,cfg['rgb_root'],contract)
    assert len(rows)==40
    assert not list(output.rglob('*.png'))
    before={p.relative_to(output).as_posix():(digest_file(p),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    shards.merge_region_shards(opts,ROOT)
    assert before=={p.relative_to(output).as_posix():(digest_file(p),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}


def test_shard_sam_resume_and_missing_or_overlapping_union(two_scenes,tmp_path):
    cfg,_,seen,fail,run=two_scenes
    fail['sam']=True
    with pytest.raises(PermissionError):run({**cfg,'maximum_new_views':2})
    assert sum(phase=='la' for phase,_,_ in seen)==2
    fail['sam']=False;run({**cfg,'maximum_new_views':2})
    assert sum(phase=='la' for phase,_,_ in seen)==2
    run(cfg)
    with pytest.raises(ValueError,match='incomplete'):shards.merge_region_shards({**cfg,'output_root':tmp_path/'missing','region_shards':[cfg['output_root']]},ROOT)
    with pytest.raises(ValueError,match='overlap'):shards.merge_region_shards({**cfg,'output_root':tmp_path/'dup','region_shards':[cfg['output_root'],cfg['output_root']]},ROOT)


def test_shard_corruption_and_resealed_wrong_row_are_rejected(two_scenes):
    cfg,contract,_,_,run=two_scenes;run(cfg);root=cfg['output_root']
    path=root/'regions/S000001/V00.json';row=read_json(path);row['view']='V01';atomic_json(path,row)
    lock=read_json(root/'region_shard_lock.json');seal(root,[root/p for p in lock['files']],name='region_shard_lock.json',metadata=lock['metadata'])
    expected,_=region._rgb_records(cfg['rgb_root'],contract)
    with pytest.raises(ValueError,match='image/route'):shards.read_region_shard(root,expected,contract)


def test_reuse_projects_only_geometry_fields(two_scenes,tmp_path,monkeypatch):
    cfg,contract,_,_,run=two_scenes;run(cfg)
    expected,_=region._rgb_records(cfg['rgb_root'],contract)
    row=read_json(cfg['output_root']/'regions/S000001/V00.json')
    reused={}
    for key,source in expected.items():
        reused[key]={**row,**source,'embedding':[999],'scores':{'anomaly':999},'objects':[{**row['objects'][0],'candidate_id':key[1]+'-P01','scores':{'P1':999}}]}
    monkeypatch.setattr(native_cache,'native_regions',lambda cfg,contract:(reused,'a'*64))
    def forbidden(*args,**kwargs):pytest.fail('Reuse must not load vision models')
    options={**cfg,'reuse':{'region_format':'fixture'},'output_root':tmp_path/'reuse'}
    result=region.extract_regions(options,ROOT,worker_factory=forbidden,model_verifier=forbidden)
    assert result['status']=='shard_complete'
    record=read_json(options['output_root']/'regions/S000001/V00.json')
    assert 'scores' not in record and 'embedding' not in record and 'scores' not in record['objects'][0]
