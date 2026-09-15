"""Synthetic role/exposure checks; existing source notebooks remain guarded."""
from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import nbformat
import pytest

from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal
from vpi_remaining_evaluation.cohort import CACHE_SCHEMA, predict_cached_cohort, selected_inputs
from vpi_remaining_evaluation.gates import CONTRACTS, PrerequisiteBlocked, require_stage_adapter, verify_exposure
from vpi_remaining_evaluation.workbench import resource_recipe
from visual_peer_inspector.data.model_safe_package import membership_sha256

ROOT=Path(__file__).resolve().parents[1]


def exposure(root):
    source='a'*64
    records=[]
    for role,c in CONTRACTS.items():
        base,extra=divmod(c.scene_count,c.asset_group_count)
        records += [{'asset_group':f'{role}-A{i:03d}','role':role,'scene_count':base+(i<extra),
                     'resolved':True,'prior_access':[{'kind':'rgb','source_sha256':source}]} for i in range(c.asset_group_count)]
    atomic_json(root/'asset_records.json',records)
    atomic_json(root/'exposure_ledger.json',{'schema':'vpi-asset-exposure-audit/v1','status':'resolved',
        'reviewed_source_sha256':[source],'roles':{role:{'membership_sha256':c.membership_sha256} for role,c in CONTRACTS.items()}})
    seal(root,[root/'asset_records.json',root/'exposure_ledger.json'])
    return root


def payload(scene='S0'):
    return {'scene':scene,'correspondence_provenance':'dataset_reference',
        'score_rows':{v:{'objects':[{'candidate_id':f'C{i}','embedding':[1.,0.],
            'scores':{'forward_02_cov70':.1,'P1_current':.2,'P2_consensus':.3}} for i in range(3)]} for v in ['V00','V10','V19']},
        'bank_observations':[{'view_id':v,'candidate_id':f'C{i}','object_id':f'N{i}','patches':[[1.,0.]]} for v in ['V00','V10','V19'] for i in range(3)]}


def pure_decision(rows,observations,*,views,mode):
    assert set(rows)==set(views)=={'V00','V10'}
    assert {o['view_id'] for o in observations}==set(views)
    assert all(o['object_id'] is None for o in observations) if mode=='automatic' else all(o['object_id'] for o in observations)
    return {'ranking':[],'status':'insufficient_evidence'}


def test_missing_empty_and_unresolved_exposure_cannot_pass(tmp_path):
    with pytest.raises(PrerequisiteBlocked,match='EXPOSURE_LEDGER_MISSING'):
        verify_exposure('calibration',None)
    root=exposure(tmp_path/'audit')
    atomic_json(root/'asset_records.json',[])
    seal(root,[root/'asset_records.json',root/'exposure_ledger.json'])
    with pytest.raises(ValueError,match='empty schema'):
        verify_exposure('calibration',root)
    root=exposure(root)
    rows=read_json(root/'asset_records.json');rows[0]['resolved']=False
    atomic_json(root/'asset_records.json',rows);seal(root,[root/'asset_records.json',root/'exposure_ledger.json'])
    with pytest.raises(PrerequisiteBlocked,match='EXPOSURE_UNRESOLVED'):
        verify_exposure('internal_lock',root)


def test_prior_protected_label_access_blocks_confirmation_without_reshuffling(tmp_path):
    root=exposure(tmp_path/'audit')
    rows=read_json(root/'asset_records.json')
    next(r for r in rows if r['role']=='internal_lock')['prior_access'].append({'kind':'evaluation_results','source_sha256':'a'*64})
    atomic_json(root/'asset_records.json',rows);seal(root,[root/'asset_records.json',root/'exposure_ledger.json'])
    with pytest.raises(PrerequisiteBlocked,match='PROTECTED_ASSETS_ALREADY_EXPOSED'):
        verify_exposure('internal_lock',root)


def test_legacy_adapter_remains_blocked_after_valid_exposure(tmp_path):
    root=exposure(tmp_path/'audit')
    for stage in ['09a','09b','09b1','09c','09d']:
        with pytest.raises(PrerequisiteBlocked,match='CURRENT_METHOD_ADAPTER_REQUIRED'):
            require_stage_adapter(stage,root)


def test_selected_view_isolation_and_automatic_identity_removal():
    original=payload()
    rows,observations=selected_inputs(original,['V00','V10'],'automatic')
    assert set(rows)=={'V00','V10'} and len(observations)==6
    assert all(o['object_id'] is None for o in observations)
    assert original['bank_observations'][0]['object_id']=='N0'
    original['labels']={'O0':1}
    with pytest.raises(ValueError,match='Evaluator fields'):
        selected_inputs(original,['V00','V10'],'assisted')


@pytest.mark.parametrize('role,mode',[('development','automatic'),('calibration','assisted'),('internal_lock','automatic')])
def test_synthetic_1000_600_400_sizes_predict_without_labels_and_resume(tmp_path,role,mode,monkeypatch):
    original=CONTRACTS[role]
    scenes=[f'SYNTHETIC-{i:06d}' for i in range(original.scene_count)]
    c=replace(original,cohort_id='synthetic-'+role,membership_sha256=membership_sha256(scenes))
    monkeypatch.setitem(CONTRACTS,role,c)
    cache,freeze,output=(tmp_path/name for name in ['cache','freeze','output'])
    atomic_json(freeze/'shareable/current_method_freeze.json',{'automatic_method':'winner_cluster_consensus',
        'assisted_method':'object_bank_top2','selection_closed':True,'view_sets':{'two-primary':['V00','V10']}})
    seal(freeze,[freeze/'shareable/current_method_freeze.json'],name='scientific_lock.json')
    for scene in scenes: atomic_json(cache/'scenes'/f'{scene}.json',payload(scene))
    atomic_json(cache/'cohort_manifest.json',{'schema':CACHE_SCHEMA,'role':role,'scenes':scenes,
        **{key:getattr(c,key) for key in ['cohort_id','membership_sha256','scene_count','asset_group_count']},
        'method_freeze_sha256':digest_file(freeze/'shareable/current_method_freeze.json')})
    seal(cache,[p for p in cache.rglob('*') if p.is_file()])
    audit=exposure(tmp_path/'audit')
    args={'adapter_sha256':digest_file(Path(__file__)),'mode':mode,'view_sets':{'two-primary':['V00','V10']},'exposure_root':audit}
    receipt=predict_cached_cohort(cache,freeze,output,pure_decision,**args)
    assert receipt['scene_count']==c.scene_count and receipt['anomaly_labels_opened'] is False
    assert receipt['metric_evaluation_run'] is False
    assert receipt['actual_model_membership_verified'] is True
    assert receipt['asset_attribution_verified'] is False
    verify_seal(output,name='prediction_lock.json')
    before={str(p):(digest_file(p),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}
    assert predict_cached_cohort(cache,freeze,output,pure_decision,**args)==receipt
    assert before=={str(p):(digest_file(p),p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}


def test_original_hash_claim_cannot_authorize_arbitrary_scene_ids(tmp_path):
    c=CONTRACTS['development'];cache,freeze=tmp_path/'cache',tmp_path/'freeze'
    atomic_json(freeze/'shareable/current_method_freeze.json',{})
    seal(freeze,[freeze/'shareable/current_method_freeze.json'],name='scientific_lock.json')
    atomic_json(cache/'cohort_manifest.json',{'schema':CACHE_SCHEMA,'role':'development',
        'scenes':[f'SYNTHETIC-{i}' for i in range(c.scene_count)],
        **{key:getattr(c,key) for key in ['cohort_id','membership_sha256','scene_count','asset_group_count']}})
    seal(cache,[cache/'cohort_manifest.json'])
    with pytest.raises(ValueError,match='Actual cached scene membership'):
        predict_cached_cohort(cache,freeze,tmp_path/'output',pure_decision,
            adapter_sha256=digest_file(__file__),mode='automatic',view_sets={'two-primary':['V00','V10']})


def test_invalid_inherited_coverage_is_preserved_but_not_usable():
    import pandas as pd
    from vpi_remaining_evaluation.workbench import _decorate
    frame=pd.DataFrame([{'route':'automatic_predicted','method':'winner_cluster_consensus',
        'metric':'identity_reference_coverage','mean':1.0},
        {'route':'automatic_predicted','method':'winner_cluster_consensus',
        'metric':'identity_purity','mean':0.6}])
    result=_decorate(frame,'08O')
    assert pd.isna(result.iloc[0]['mean']) and result.iloc[0]['original_reported_mean']==1.0
    assert 'invalid inherited' in result.iloc[0]['metric_validity']
    assert result.iloc[1]['mean']==0.6


def test_legacy_guard_helper_is_idempotent_without_editing_repository_notebooks():
    import importlib.util
    spec=importlib.util.spec_from_file_location('remaining_builder', ROOT/'scripts/build_remaining_evaluation_workbench.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for stage in ['09a','09b','09b1','09c','09d']:
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Legacy stage'),
            nbformat.v4.new_code_cell("subprocess.run(['legacy'])")])
        notebook=builder.apply_current_guard(builder.apply_current_guard(notebook,stage),stage)
        guards=[c for c in notebook.cells if 'vpi-current-method-guard' in c.metadata.get('tags',[])]
        assert len(guards)==2
        assert f"require_stage_adapter('{stage}')" in guards[1].source
        assert notebook.cells.index(guards[1]) < next(i for i,c in enumerate(notebook.cells) if 'subprocess.run(' in c.source)
        for c in notebook.cells:
            if c.cell_type=='code':
                compile(c.source,'notebook','exec');assert c.execution_count is None and not c.outputs


def test_completed_09e_has_clean_executable_cells():
    notebook=nbformat.read(ROOT/'notebooks/protected-evaluation/09e-remaining-cohort-readiness-and-evidence.ipynb',as_version=4)
    text='\n'.join(c.source for c in notebook.cells)
    assert 'prepare_workbench' in text and 'render_workbench' in text and 'require_stage_adapter' in text
    assert 'new_protected_scenes_evaluated' in text
    for c in notebook.cells:
        if c.cell_type=='code': compile(c.source,'workbench','exec')


def test_builder_only_writes_completed_09e(tmp_path, monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('remaining_builder_only', ROOT/'scripts/build_remaining_evaluation_workbench.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    original=tmp_path/'09a-preserved.ipynb'
    original.write_bytes(b'unchanged legacy notebook')
    monkeypatch.setattr(builder,'DIRECTORY',tmp_path)
    paths=builder.build()
    assert [p.name for p in paths]==['09e-remaining-cohort-readiness-and-evidence.ipynb']
    assert original.read_bytes()==b'unchanged legacy notebook'
    built=nbformat.read(paths[0],as_version=4)
    completed=nbformat.read(ROOT/'notebooks/protected-evaluation'/paths[0].name,as_version=4)
    assert [(c.cell_type,c.source) for c in built.cells]==[(c.cell_type,c.source) for c in completed.cells]


def test_resource_recipe_preserves_roles_and_freezes_protected_method_selection():
    recipe=resource_recipe()
    assert recipe['original_roles']=={'development':1000,'calibration':600,'internal_lock':400}
    assert len(recipe['view_sets'])==19
    assert 'not implemented' in recipe['adapter_status']
    assert recipe['current_methods']=={'automatic':'winner_cluster_consensus','assisted':'object_bank_top2'}


def test_flat_nested_and_ambiguous_extracted_inputs(tmp_path):
    from vpi_remaining_evaluation.workbench import resolve_input_root
    nested=tmp_path/'mount'/'expanded'/'package';nested.mkdir(parents=True)
    (nested/'prediction_lock.json').write_text('{}')
    assert resolve_input_root(nested,'prediction_lock.json')==nested
    assert resolve_input_root(tmp_path/'mount','prediction_lock.json')==nested
    other=tmp_path/'mount'/'second';other.mkdir();(other/'prediction_lock.json').write_text('{}')
    with pytest.raises(ValueError,match='found 2'):
        resolve_input_root(tmp_path/'mount','prediction_lock.json')


def test_report_rejects_different_inputs_before_selecting_galleries(tmp_path,monkeypatch):
    from vpi_remaining_evaluation import reporting
    root=tmp_path/'scientific'
    atomic_json(root/'run_binding.json',{'input_locks':{'prediction':'original'}})
    seal(root,[root/'run_binding.json'],name='scientific_lock.json')
    monkeypatch.setattr(reporting,'resolved_config',lambda cfg:cfg)
    monkeypatch.setattr(reporting,'verify_inputs',lambda cfg:{'prediction':'different'})
    monkeypatch.setattr(reporting,'gallery_plan',lambda cfg:(_ for _ in ()).throw(AssertionError('must not select examples')))
    with pytest.raises(ValueError,match='differs from sealed scientific'):
        reporting.render_workbench({},root,tmp_path/'report',ROOT)
