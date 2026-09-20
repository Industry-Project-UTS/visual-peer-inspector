"""Synthetic current09 adapters; no protected dataset or external service access."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import nbformat
import numpy as np
import pandas as pd
import pytest

from visual_peer_inspector.data.model_safe_package import membership_sha256
from vpi_object_bank.storage import atomic_json,digest_file,read_json,seal,verify_seal
from vpi_remaining_evaluation.gates import CONTRACTS,PrerequisiteBlocked
from vpi_remaining_evaluation.cohort import CACHE_SCHEMA
from vpi_next_evaluation.contracts import METHODS,VIEW_SETS,prepare_method_freeze,verify_current_freeze,require_protocol_receipt
from vpi_next_evaluation import original,reporting

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def resources(tmp_path,monkeypatch):
    scenes=['SYNTHETIC-001','SYNTHETIC-002','SYNTHETIC-003']
    c=replace(CONTRACTS['calibration'],cohort_id='synthetic-calibration',scene_count=3,image_count=60,
              asset_group_count=2,membership_sha256=membership_sha256(scenes))
    monkeypatch.setitem(CONTRACTS,'calibration',c)
    reference,freeze,cache,evaluator,exposure=(tmp_path/name for name in ['reference','freeze','cache','evaluator','exposure'])
    atomic_json(reference/'tables/current_method_freeze.json',{'automatic_method':METHODS['automatic'],'assisted_method':METHODS['assisted']})
    seal(reference,[reference/'tables/current_method_freeze.json'],name='publication_lock.json')
    prepare_method_freeze(reference,freeze,ROOT)
    freeze_hash=digest_file(freeze/'method_freeze.json')
    views=sorted({v for selected in VIEW_SETS.values() for v in selected})
    vectors=[[1.,.01],[1.,-.01],[.01,1.]]
    for scene in scenes:
        payload={'scene':scene,'correspondence_provenance':'dataset_reference',
            'score_rows':{view:{'objects':[{'candidate_id':f'C{i}','embedding':vector,
                'scores':{name:float(i) for name in ('forward_02_cov70','P1_current','P2_consensus')}} for i,vector in enumerate(vectors)]} for view in views},
            'bank_observations':[{'view_id':view,'candidate_id':f'C{i}','object_id':f'R{i}','patches':[vector]} for view in views for i,vector in enumerate(vectors)]}
        atomic_json(cache/'scenes'/f'{scene}.json',payload)
    manifest={'schema':CACHE_SCHEMA,'role':'calibration','scenes':scenes,'method_freeze_sha256':freeze_hash,
        'correspondence_provenance':'dataset_reference',**{key:getattr(c,key) for key in ('cohort_id','scene_count','asset_group_count','membership_sha256')}}
    atomic_json(cache/'cohort_manifest.json',manifest);seal(cache,[p for p in cache.rglob('*') if p.is_file()])
    index={scene:f'ASSET-{i%2}' for i,scene in enumerate(scenes)};atomic_json(evaluator/'scene_index.json',index)
    index_hash=digest_file(evaluator/'scene_index.json')
    records=[]
    for role,contract in CONTRACTS.items():
        quotient,remainder=divmod(contract.scene_count,contract.asset_group_count)
        records += [{'asset_group':f'{role}-A{i}','role':role,'scene_count':quotient+(i<remainder),'resolved':True,
            'prior_access':[{'kind':'rgb','source_sha256':index_hash}]} for i in range(contract.asset_group_count)]
    atomic_json(exposure/'asset_records.json',records)
    atomic_json(exposure/'exposure_ledger.json',{'schema':'vpi-asset-exposure-audit/v1','status':'resolved',
        'reviewed_source_sha256':[index_hash],'roles':{role:{'membership_sha256':contract.membership_sha256} for role,contract in CONTRACTS.items()}})
    seal(exposure,[exposure/'asset_records.json',exposure/'exposure_ledger.json'])
    for scene in scenes:
        atomic_json(evaluator/'scenes'/f'{scene}.json',{'labels':{'R0':0,'R1':0,'R2':1},'asset_group':index[scene],
            'correspondence':{f'automatic/{view}/C{i}':f'R{i}' for view in views for i in range(3)}})
    atomic_json(evaluator/'current_evaluator_manifest.json',{'schema':original.EVALUATOR_SCHEMA,'role':'calibration',
        'cache_lock_sha256':digest_file(cache/'artifact_lock.json'),'method_freeze_sha256':freeze_hash,
        'membership_sha256':c.membership_sha256,'asset_index_sha256':index_hash})
    seal(evaluator,[p for p in evaluator.rglob('*') if p.is_file()])
    return {'cache_root':str(cache),'evaluator_root':str(evaluator),'exposure_root':str(exposure),
            'method_freeze_root':str(freeze),'output_root':str(tmp_path/'predictions'),'scenes':scenes}


def test_current_freeze_and_missing_exposure_cannot_be_bypassed(resources):
    assert verify_current_freeze(resources['method_freeze_root'],ROOT)['methods']==METHODS
    cfg={**resources,'exposure_root':None,'ready':True,'allow_protected':True}
    assert original.readiness('09a',cfg,ROOT)['status']=='blocked'
    with pytest.raises(PrerequisiteBlocked):original.predict_current('09a',cfg,ROOT)


def test_actual_membership_and_assistance_provenance_are_checked(resources):
    root=Path(resources['cache_root']);manifest=read_json(root/'cohort_manifest.json')
    manifest['scenes'][0]='FORGED';atomic_json(root/'cohort_manifest.json',manifest)
    seal(root,[p for p in root.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    with pytest.raises(ValueError,match='Actual cached scene membership'):original.verify_cache(root,'calibration')


def test_synthetic_predict_evaluate_report_and_exact_resume(resources,tmp_path):
    cfg=dict(resources);cfg.pop('scenes')
    pred_receipt=original.predict_current('09a',cfg,ROOT)
    assert pred_receipt['scene_count']==3 and pred_receipt['anomaly_labels_opened'] is False
    assert pred_receipt['settings_per_scene']==38
    prediction=Path(cfg['output_root'])
    before=digest_file(prediction/'prediction_lock.json')
    assert original.predict_current('09a',cfg,ROOT)==pred_receipt
    assert digest_file(prediction/'prediction_lock.json')==before
    evaluation=tmp_path/'evaluation'
    evaluation_cfg={**cfg,'prediction_root':str(prediction),'output_root':str(evaluation)}
    receipt=original.evaluate_predictions('09b',evaluation_cfg,ROOT)
    assert receipt['status']=='protocol_complete' and receipt['review_policy_status'].startswith('unfitted')
    verify_seal(evaluation,name='evaluation_lock.json')
    assert require_protocol_receipt(evaluation,'09b',digest_file(Path(cfg['method_freeze_root'])/'method_freeze.json'))==receipt
    frame=pd.read_csv(evaluation/'private/scene_metrics.csv')
    assert len(frame)==3*38 and set(frame.scene)==set(resources['scenes'])
    assert set(frame[frame['mode']=='assisted'].correspondence_provenance)=={'dataset_reference'}
    summary=pd.read_csv(evaluation/'shareable/metric_summary.csv')
    assert set(summary.scene_count)=={3} and set(summary.asset_group_count)=={2}
    assert set(summary.correspondence_provenance)=={'predicted','dataset_reference'}
    assert original.evaluate_predictions('09b',evaluation_cfg,ROOT)==receipt
    report=reporting.render_current_report({**evaluation_cfg,'evaluation_root':str(evaluation),'report_root':str(tmp_path/'report')})
    assert report['gallery_count']>=6 and report['missing_media_examples']==report['gallery_count']
    first=tmp_path/'report/private/galleries/example-01'
    assert (first/'object_per_view_scores.csv').is_file() and (first/'combined_ranking.json').is_file()


@pytest.mark.parametrize('mutation',['region','route','drop_candidate'])
def test_resealed_prediction_tampering_rejected_before_evaluator(resources,tmp_path,monkeypatch,mutation):
    cfg=dict(resources);cfg.pop('scenes');original.predict_current('09a',cfg,ROOT)
    root=Path(cfg['output_root']);path=root/'scenes'/resources['scenes'][0]/'predictions.json'
    payload=read_json(path);setting=payload['settings'][0]
    if mutation=='region':setting['region']='supplied'
    elif mutation=='route':setting['route']='supplied_reference'
    else:setting['observations']=setting['observations'][1:]
    atomic_json(path,payload)
    old=read_json(root/'prediction_lock.json');seal(root,[root/name for name in old['files']],name='prediction_lock.json')
    real=original.read_json
    def guarded(path):
        assert not Path(path).is_relative_to(Path(resources['evaluator_root']))
        return real(path)
    monkeypatch.setattr(original,'read_json',guarded)
    with pytest.raises(ValueError):original.evaluate_predictions('09b',{**cfg,'prediction_root':str(root),'output_root':str(tmp_path/'evaluation')},ROOT)


def test_protocol_finalization_recovers_without_reopening_scene_labels(resources,tmp_path,monkeypatch):
    cfg=dict(resources);cfg.pop('scenes');original.predict_current('09a',cfg,ROOT)
    root=tmp_path/'evaluation';cfg={**cfg,'prediction_root':cfg['output_root'],'output_root':str(root)}
    receipt=original.evaluate_predictions('09b',cfg,ROOT)
    before={name:digest_file(root/name) for name in read_json(root/'evaluation_lock.json')['files']}
    (root/'protocol_lock.json').unlink();(root/'stage_receipt.json').unlink()
    real=original.read_json
    def guarded(path):
        assert not Path(path).is_relative_to(Path(cfg['evaluator_root'])/'scenes'), 'resume must not reopen scene labels'
        return real(path)
    monkeypatch.setattr(original,'read_json',guarded)
    assert original.evaluate_predictions('09b',cfg,ROOT)==receipt
    assert before=={name:digest_file(root/name) for name in before}


def test_unbacked_protocol_receipt_cannot_authorize_downstream(tmp_path):
    atomic_json(tmp_path/'stage_receipt.json',{'status':'protocol_complete'})
    seal(tmp_path,[tmp_path/'stage_receipt.json'],name='protocol_lock.json')
    with pytest.raises(ValueError,match='actual evaluation seal'):require_protocol_receipt(tmp_path,'09b','a'*64)


def test_09e_generation_cannot_reinstall_guards_in_current09(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    import build_remaining_evaluation_workbench as builder
    monkeypatch.setattr(builder,'DIRECTORY',tmp_path)
    untouched=tmp_path/'09a-current.ipynb';untouched.write_text('new current09 source')
    monkeypatch.setattr(builder,'protect_legacy',lambda:(_ for _ in ()).throw(AssertionError('must not touch future09')))
    builder.build()
    assert untouched.read_text()=='new current09 source'


def test_freeze_covers_transitive_candidate_and_exposure_adapters():
    from vpi_next_evaluation.contracts import runtime_source_files
    files=runtime_source_files(ROOT)
    assert 'src/vpi_remaining_evaluation/cohort.py' in files
    assert 'src/vpi_remaining_evaluation/gates.py' in files
    for module in ('reporting.py','official_reporting.py','diagnostic_figures.py'):
        assert 'src/vpi_next_evaluation/'+module not in files


def test_gallery_heatmaps_are_per_view_and_missing_geometry_is_explicit(tmp_path,monkeypatch):
    from PIL import Image
    from matplotlib.axes import Axes
    from visual_peer_inspector.models.sam2 import encode_binary_mask
    from visual_peer_inspector.representation.crops import prepare_masked_crop
    cache=tmp_path/'cache';cache.mkdir()
    rgb=np.full((20,20,3),140,dtype=np.uint8);mask=np.ones((20,20),dtype=bool)
    crop=prepare_masked_crop(rgb,mask)
    payload={'evidence_views':{}}
    for view in ['V00','V10']:
        Image.fromarray(rgb).save(cache/f'{view}.png')
        geometry={'mask':encode_binary_mask(mask),'box':[0,0,20,20]}
        if view=='V00':geometry.update(valid70=np.ones((16,16),dtype=bool).tolist(),crop_sha256=crop.sha256,padded_box_xyxy=list(crop.padded_box_xyxy))
        payload['evidence_views'][view]={'rgb_relative':f'{view}.png','candidates':{'C0':geometry}}
    seal(cache,list(cache.glob('*.png')));lock=verify_seal(cache)
    setting={'views':['V00','V10'],'mode':'assisted','method':'object_bank_top2','correspondence_provenance':'dataset_reference',
        'rankings':{'object_bank_top2':[{'id':'R0','score':.75,'support':2,'fusion_observations':2,
            'members':[{'view_id':v,'candidate_id':'C0'} for v in ['V00','V10']]}]},
        'observations':[{'view_id':v,'candidate_id':'C0','component_scores':{},'normalized_score':score,
            'object_bank':{'score':score,'patch_scores':[.2]*256}} for v,score in [('V00',.8),('V10',.7)]]}
    titles=[];real_title=Axes.set_title
    def capture(self,label,*args,**kwargs):titles.append(label);return real_title(self,label,*args,**kwargs)
    monkeypatch.setattr(Axes,'set_title',capture)
    output=tmp_path/'panel.png'
    receipt=reporting.scene_panel(payload,setting,cache,lock,output)
    assert receipt['projected_heatmap_count']==1 and receipt['selected_object_located_views']==2
    assert titles.count('Selected object; coarse patch evidence')==1
    assert titles.count('Selected object; heatmap unavailable')==1
    table=pd.read_csv(tmp_path/'object_per_view_scores.csv')
    assert table.fusion_weight_equivalent.tolist()==[.5,.5]
    payload['evidence_views']['V10']['candidates']={}
    receipt=reporting.scene_panel(payload,setting,cache,lock,output)
    assert receipt['selected_object_located_views']==1 and receipt['unlocated_views']==['V10']
    payload['evidence_views']['V10']['candidates']={'C0':{'mask':encode_binary_mask(np.ones((2,2),dtype=bool)),'box':[0,0,20,20]}}
    with pytest.raises(ValueError,match='Mask shape'):reporting.scene_panel(payload,setting,cache,lock,output)


def test_corrupted_predictions_stop_before_any_evaluator_read(resources,tmp_path,monkeypatch):
    cfg=dict(resources);cfg.pop('scenes');original.predict_current('09a',cfg,ROOT)
    prediction=Path(cfg['output_root']);target=prediction/'scenes'/resources['scenes'][0]/'predictions.json'
    target.write_text('{}')
    real=original.read_json
    def guarded(path):
        assert not Path(path).is_relative_to(Path(resources['evaluator_root'])), 'must not open evaluator on corrupted predictions'
        return real(path)
    monkeypatch.setattr(original,'read_json',guarded)
    with pytest.raises(ValueError):original.evaluate_predictions('09b',{**cfg,'prediction_root':str(prediction),'output_root':str(tmp_path/'evaluation')},ROOT)


def test_all_current09_notebooks_use_real_adapter_calls_and_clean_cells():
    for stage,call in [('09a','predict_current'),('09b','evaluate_predictions'),('09b1','release_internal'),('09c','predict_current'),('09d','evaluate_predictions')]:
        path=next((ROOT/'notebooks/protected-evaluation').glob(stage+'-*.ipynb'))
        notebook=nbformat.read(path,as_version=4)
        assert any(call+'(' in c.source for c in notebook.cells)
        assert not any('require_stage_adapter(' in c.source for c in notebook.cells)
        for cell in notebook.cells:
            if cell.cell_type=='code':compile(cell.source,str(path),'exec');assert cell.execution_count is None and not cell.outputs
        metadata=read_json(path.parent/f'kernel-metadata-{stage}.json')
        assert metadata['enable_gpu'] is False and not metadata['dataset_sources']


def test_nested_input_roots_resolve_and_ambiguity_is_rejected(resources,tmp_path):
    import shutil
    nested=tmp_path/'mount'/'auto-extracted'/'package'
    shutil.copytree(resources['cache_root'],nested)
    assert original._required({'cache_root':str(tmp_path/'mount')},'cache_root')==nested
    assert original._required({'cache_root':str(nested)},'cache_root')==nested
    shutil.copytree(resources['cache_root'],tmp_path/'mount'/'second')
    with pytest.raises(ValueError,match='found 2'):
        original._required({'cache_root':str(tmp_path/'mount')},'cache_root')
    assert original._required({'evaluator_root':str(tmp_path/'not-mounted')},'evaluator_root')==tmp_path/'not-mounted'


def test_evaluator_discovery_waits_for_prediction_validation(resources,tmp_path,monkeypatch):
    cfg=dict(resources);cfg.pop('scenes');original.predict_current('09a',cfg,ROOT)
    prediction=Path(cfg['output_root'])
    (prediction/'scenes'/resources['scenes'][0]/'predictions.json').write_text('{}')
    real=original.resolve_input_root
    def guard(path,marker):
        assert marker!='current_evaluator_manifest.json', 'cannot discover evaluator before valid sealed predictions'
        return real(path,marker)
    monkeypatch.setattr(original,'resolve_input_root',guard)
    with pytest.raises(ValueError):
        original.evaluate_predictions('09b',{**cfg,'prediction_root':str(prediction),'output_root':str(tmp_path/'evaluation')},ROOT)


def test_current_calibration_release_copies_only_verified_model_cache(resources,tmp_path):
    cfg={**resources,'output_root':str(tmp_path/'release')}
    receipt=original.release_calibration(cfg,ROOT)
    assert receipt['role']=='calibration' and receipt['scene_count']==3
    assert receipt['evaluator_opened'] is False and receipt['neural_inference_run'] is False
    output=Path(cfg['output_root']);verify_seal(output,name='release_lock.json')
    assert digest_file(output/'model-cache/artifact_lock.json')==digest_file(Path(cfg['cache_root'])/'artifact_lock.json')
    assert not any('evaluator' in p.name for p in output.rglob('*'))


def test_primary_resource_builder_preserves_completed08e1_and_current_entries():
    import subprocess
    old=next((ROOT/'notebooks/development-freeze').glob('08e1-*.ipynb'))
    before=old.read_bytes()
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/build_primary_resource_notebooks.py')],check=True,cwd=ROOT)
    assert old.read_bytes()==before
    for stage,call in [('08f1','release_calibration'),('09b1','release_internal')]:
        path=next((ROOT/'notebooks/protected-evaluation').glob(stage+'-*.ipynb'))
        source='\n'.join(c.source for c in nbformat.read(path,as_version=4).cells)
        assert 'lambda:'+call+'(cfg,ROOT)' in source
        assert 'run_primary_resource_release.py' not in source


def test_current_release_rejects_sealed_evaluator_artifacts(resources,tmp_path):
    root=Path(resources['cache_root'])
    atomic_json(root/'evaluator.json',{'labels':{'R0':1}})
    seal(root,[p for p in root.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    with pytest.raises(ValueError,match='unexpected or non-model'):
        original.release_calibration({**resources,'output_root':str(tmp_path/'release')},ROOT)
    assert not (tmp_path/'release').exists()


def test_current_release_rejects_embedded_answer_fields(resources,tmp_path):
    root=Path(resources['cache_root']);path=root/'scenes'/f"{resources['scenes'][0]}.json"
    payload=read_json(path);payload['labels']={'R0':1};atomic_json(path,payload)
    seal(root,[p for p in root.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    with pytest.raises(ValueError,match='label|answer|Evaluator'):
        original.release_calibration({**resources,'output_root':str(tmp_path/'release')},ROOT)


def test_report_only_change_does_not_invalidate_scientific_freeze(resources,tmp_path):
    import shutil
    from vpi_next_evaluation.contracts import runtime_source_files
    copied=tmp_path/'source'
    for relative in runtime_source_files(ROOT):
        target=copied/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    reference=tmp_path/'reference-copy'
    atomic_json(reference/'tables/current_method_freeze.json',{'automatic_method':METHODS['automatic'],'assisted_method':METHODS['assisted']})
    seal(reference,[reference/'tables/current_method_freeze.json'],name='publication_lock.json')
    freeze=tmp_path/'new-freeze';prepare_method_freeze(reference,freeze,copied)
    report=copied/'src/vpi_next_evaluation/reporting.py';report.write_text('changed report display')
    official_report=copied/'src/vpi_next_evaluation/official_reporting.py';official_report.write_text('changed official report display')
    assert verify_current_freeze(freeze,copied)['methods']==METHODS
    scientific=copied/'src/vpi_next_evaluation/original.py';scientific.write_text(scientific.read_text()+'\n# changed scientific behavior')
    with pytest.raises(ValueError,match='runtime differs'):
        verify_current_freeze(freeze,copied)


def test_current_publication_collects_only_verified_aggregates(resources,tmp_path,monkeypatch):
    import importlib.util
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('current_publisher',ROOT/'scripts/publish_next_stage.py')
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    cfg=dict(resources);cfg.pop('scenes');original.predict_current('09a',cfg,ROOT)
    cfg.update(prediction_root=cfg['output_root'],output_root=str(tmp_path/'evaluation'),report_root=str(tmp_path/'report'))
    original.evaluate_predictions('09b',cfg,ROOT)
    reporting.render_current_report({**cfg,'evaluation_root':cfg['output_root']})
    files,manifest=publisher.verify_publication('09b',cfg['output_root'],cfg['report_root'])
    assert len(files)==16 and all(name.startswith(('evaluation/','report/')) for name in files)
    assert len(manifest['locks'])==2
    with pytest.raises(ValueError,match='stage differs'):
        publisher.verify_publication('10b',cfg['output_root'],cfg['report_root'])
    Path(cfg['output_root'],'shareable','private-gallery.csv').write_text('scene,score\nPRIVATE,1\n')
    with pytest.raises(ValueError,match='unapproved'):
        publisher.verify_publication('09b',cfg['output_root'],cfg['report_root'])


def test_current09_compact_tensor_prediction_and_model_release(resources,tmp_path):
    root=Path(resources['cache_root'])
    for scene in resources['scenes']:
        path=root/'scenes'/f'{scene}.json';payload=read_json(path)
        observations=[]
        for view in sorted(payload['score_rows']):
            previous=[o for o in payload['bank_observations'] if o['view_id']==view]
            relative=f'tensors/{scene}/{view}.npz';target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            matrices={}
            for number,row in enumerate(previous):
                key=f'p{number}';matrix=np.zeros((1,384),dtype=np.float32);matrix[0,:2]=row['patches'][0]
                matrices[key]=matrix
                observations.append({k:row[k] for k in ('view_id','candidate_id','object_id')}|{'patches_file':relative,'patches_key':key})
            np.savez_compressed(target,**matrices)
        payload['bank_observations']=observations;atomic_json(path,payload)
    seal(root,[p for p in root.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    cfg={**resources,'output_root':str(tmp_path/'compact-predictions')}
    assert original.predict_current('09a',cfg,ROOT)['settings_per_scene']==38
    release=tmp_path/'compact-release'
    assert original.release_calibration({**cfg,'output_root':str(release)},ROOT)['scene_count']==3
    assert len(list((release/'model-cache/tensors').rglob('*.npz')))==60
    verify_seal(release,name='release_lock.json')


def test_fixed_development_bridge_has_no_protected_protocol_certificate(resources,tmp_path,monkeypatch):
    contract=replace(CONTRACTS['calibration'],role='development',cohort_id='synthetic-development',pilot_stage='synthetic')
    monkeypatch.setitem(CONTRACTS,'development',contract)
    cache=Path(resources['cache_root']);manifest=read_json(cache/'cohort_manifest.json')
    manifest.update(role='development',cohort_id=contract.cohort_id);atomic_json(cache/'cohort_manifest.json',manifest)
    seal(cache,[p for p in cache.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    evaluator=Path(resources['evaluator_root']);truth_manifest=read_json(evaluator/'current_evaluator_manifest.json')
    truth_manifest.update(role='development',cache_lock_sha256=digest_file(cache/'artifact_lock.json'))
    atomic_json(evaluator/'current_evaluator_manifest.json',truth_manifest)
    seal(evaluator,[p for p in evaluator.rglob('*') if p.is_file() and p.name!='artifact_lock.json'])
    index_hash=digest_file(evaluator/'scene_index.json');rows=[]
    for role,current in CONTRACTS.items():
        quotient,remainder=divmod(current.scene_count,current.asset_group_count)
        rows += [{'asset_group':f'{role}-A{i}','role':role,'scene_count':quotient+(i<remainder),'resolved':True,
            'prior_access':[{'kind':'rgb','source_sha256':index_hash}]} for i in range(current.asset_group_count)]
    exposure=Path(resources['exposure_root']);atomic_json(exposure/'asset_records.json',rows)
    atomic_json(exposure/'exposure_ledger.json',{'schema':'vpi-asset-exposure-audit/v1','status':'resolved',
        'reviewed_source_sha256':[index_hash],'roles':{role:{'membership_sha256':c.membership_sha256} for role,c in CONTRACTS.items()}})
    seal(exposure,[exposure/'asset_records.json',exposure/'exposure_ledger.json'])
    cfg={**resources,'output_root':str(tmp_path/'dev-predictions')}
    assert original.predict_current('dev-predict',cfg,ROOT)['scene_count']==3
    evaluation=tmp_path/'dev-evaluation';cfg.update(prediction_root=cfg['output_root'],output_root=str(evaluation))
    receipt=original.evaluate_predictions('dev-evaluate',cfg,ROOT)
    assert receipt['role']=='development' and receipt['protocol_certification']=='development_only'
    assert receipt['independent_confirmation'] is False
    assert receipt['evaluation_claim'] and receipt['protocol_certification']=='development_only'
    assert not (evaluation/'protocol_lock.json').exists()
    before=digest_file(evaluation/'evaluation_lock.json')
    assert original.evaluate_predictions('dev-evaluate',cfg,ROOT)==receipt
    assert digest_file(evaluation/'evaluation_lock.json')==before
    with pytest.raises((ValueError,FileNotFoundError)):
        require_protocol_receipt(evaluation,'09b',digest_file(Path(cfg['method_freeze_root'])/'method_freeze.json'))
