"""Generated neutral fixtures exercise current official compatibility and recovery."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from visual_peer_inspector.data.model_safe_package import membership_sha256
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal
from vpi_next_evaluation import official
from vpi_next_evaluation.contracts import VIEW_SETS

def fake_decision(rows, observations, *, views, mode):
    assert set(rows) == set(views)
    assert all(row['view_id'] in views for row in observations)
    if mode == 'automatic':
        assert all(row['object_id'] is None for row in observations)
    members=[{'view_id':row['view_id'],'candidate_id':row['candidate_id']} for row in observations]
    rank={(m['view_id'],m['candidate_id']):1.0 for m in members}
    if mode=='assisted':
        return {'ranking':[{'id':'A','score':1.0,'members':members,'support':len(views),'fusion_observations':min(2,len(views))}],
                'identities':{key:'A' for key in rank},'component_rank':rank,
                'scored_observations':[dict(m,object_bank={'score':1.0,'patch_scores':[1.0]}) for m in members]}
    return {'ranking': [], 'identities': {}, 'component_rank': rank, 'scored_observations': None}

@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import visual_peer_inspector.web.decision as decision
    monkeypatch.setattr(decision, 'decide_cached', fake_decision)
    contracts = {}
    scenes = {}
    for role, original in official.OFFICIAL.items():
        scenes[role] = [f'{role}-scene-1', f'{role}-scene-2']
        contracts[role] = replace(original, scene_count=2, asset_group_count=1, image_count=40,
                                  model_membership_sha256=membership_sha256(scenes[role]))
    monkeypatch.setattr(official, 'OFFICIAL', contracts)
    frozen = {'view_sets': VIEW_SETS, 'source_sha256': {'src/visual_peer_inspector/web/decision.py': digest_file(__file__)}}
    prerequisites = {'method_freeze_sha256': 'a'*64, 'exposure_lock_sha256': 'b'*64,
                     'internal_lock_protocol_sha256': 'c'*64, 'official_exposure_lock_sha256': 'd'*64,
                     'official_exposure_status': 'access_audited', 'untouched_claim': False}
    monkeypatch.setattr(official, 'verify_prerequisites', lambda config, source_root=None: (frozen, prerequisites))
    for role, contract in contracts.items():
        cache = tmp_path / (role+'-cache')
        manifest = {'schema': official.CACHE_SCHEMA, 'role': role, 'cohort_id': contract.cohort_id,
                    'scene_count': 2, 'asset_group_count': 1, 'membership_sha256': contract.membership_sha256,
                    'method_freeze_sha256': 'a'*64, 'scenes': scenes[role], 'correspondence_provenance': 'dataset_reference'}
        atomic_json(cache/'cohort_manifest.json',manifest)
        views=sorted({view for selected in VIEW_SETS.values() for view in selected})
        for scene in scenes[role]:
            payload={'scene':scene,'correspondence_provenance':'dataset_reference',
                     'score_rows':{view:{'objects':[{'candidate_id':'C1','embedding':[1,0],'scores':{}}]} for view in views},
                     'bank_observations':[{'view_id':view,'candidate_id':'C1','object_id':'A','patches':[[1,0]]} for view in views]}
            atomic_json(cache/'scenes'/f'{scene}.json',payload)
        seal(cache,[path for path in cache.rglob('*') if path.is_file()])
    return tmp_path, frozen, prerequisites

def config(root, role, mode='automatic'):
    return {'role':role,'mode':mode,'cache_root':root/(role+'-cache'),'output_root':root/(role+'-'+mode)}

def test_all_frozen_schedules_selected_only_and_verified_resume(fixture):
    root, _, _ = fixture
    cfg=config(root,'seen')
    result=official.predict_official(cfg, root)
    assert result['scene_count']==2 and result['settings_per_scene']==19
    assert result['anomaly_labels_opened'] is False
    path=next((cfg['output_root']/'scenes').glob('*/predictions.json'))
    before=path.read_bytes()
    assert {len(row['views']) for row in read_json(path)['settings']}=={1,2,3,5}
    assert official.predict_official(cfg,root)==result
    assert path.read_bytes()==before

def test_corrupt_checkpoint_is_rejected_without_recomputation(fixture):
    root,_,_=fixture
    cfg=config(root,'seen')
    official.predict_official(cfg,root)
    path=next((cfg['output_root']/'scenes').glob('*/predictions.json'))
    path.write_text('{}')
    with pytest.raises(ValueError):
        official.predict_official(cfg,root)

def test_changed_mode_cannot_reuse_same_output(fixture):
    root,_,_=fixture
    cfg=config(root,'seen')
    official.predict_official(cfg,root)
    with pytest.raises(ValueError,match='different inputs'):
        official.predict_official({**cfg,'mode':'assisted'},root)

def test_prediction_seal_rejects_legacy_schema_and_wrong_role(fixture):
    root,_,_=fixture
    cfg=config(root,'seen')
    official.predict_official(cfg,root)
    with pytest.raises(ValueError):
        official.validate_prediction(cfg['output_root'],'unseen')
    binding=read_json(cfg['output_root']/'run_binding.json')
    binding['schema']='visual-peer-10a-official-inference/v1'
    atomic_json(cfg['output_root']/'run_binding.json',binding)
    with pytest.raises(ValueError):
        official.validate_prediction(cfg['output_root'],'seen')

def test_finalizer_preserves_official_roles_and_rejects_mixed_modes(fixture):
    root,_,_=fixture
    for role in ('seen','unseen'):
        official.predict_official(config(root,role),root)
    cfg={'seen_prediction_root':root/'seen-automatic','unseen_prediction_root':root/'unseen-automatic','output_root':root/'summary',
         'cache_roots':{role:root/(role+'-cache') for role in official.OFFICIAL}}
    summary=official.finalize_official(cfg,root)
    assert [row['role'] for row in summary['roles']]==['seen','unseen']
    assert summary['accuracy_evaluated'] is False
    assert official.finalize_official(cfg,root)==summary
    official.predict_official(config(root,'unseen','assisted'),root)
    with pytest.raises(ValueError,match='do not match'):
        official.finalize_official({**cfg,'unseen_prediction_root':root/'unseen-assisted','output_root':root/'bad-summary'},root)

def test_official_exposure_is_distinct_and_outcome_access_blocks(fixture):
    root,_,_=fixture
    with pytest.raises(official.PrerequisiteBlocked,match='OFFICIAL_EXPOSURE_MISSING'):
        official.verify_official_exposure(None)
    audit=root/'official-audit'
    ledger={'schema':official.EXPOSURE_SCHEMA,'status':'resolved','reviewed_source_sha256':['e'*64],
            'roles':{role:{'membership_sha256':contract.membership_sha256} for role,contract in official.OFFICIAL.items()}}
    rows=[{'role':role,'asset_group':'assetA','scene_count':2,'resolved':True,'prior_access':[]} for role in official.OFFICIAL]
    atomic_json(audit/'official_exposure_ledger.json',ledger)
    atomic_json(audit/'asset_records.json',rows)
    seal(audit,[audit/'official_exposure_ledger.json',audit/'asset_records.json'])
    assert official.verify_official_exposure(audit)['untouched_claim'] is False
    rows[0]['prior_access']=[{'kind':'evaluation_results','source_sha256':'e'*64}]
    atomic_json(audit/'asset_records.json',rows)
    seal(audit,[audit/'official_exposure_ledger.json',audit/'asset_records.json'])
    with pytest.raises(official.PrerequisiteBlocked,match='OFFICIAL_OUTCOMES_ALREADY_EXPOSED'):
        official.verify_official_exposure(audit)

def test_release_copies_only_a_complete_validated_neutral_cache(fixture):
    root,_,_=fixture
    cfg=config(root,'seen');cfg['output_root']=root/'release'
    receipt=official.release_verified_cache(cfg,root)
    assert receipt['source_archive_opened'] is False
    assert receipt['evaluator_resources_copied'] is False
    assert (root/'release/model-input/artifact_lock.json').read_bytes()==(root/'seen-cache/artifact_lock.json').read_bytes()
    assert official.release_verified_cache(cfg,root)==receipt

def test_interruption_keeps_completed_scene_and_resume_finishes(fixture, monkeypatch):
    import visual_peer_inspector.web.decision as decision
    root, frozen, _=fixture
    cfg=config(root,'seen')
    calls=[]
    def interrupted(rows, observations, *, views, mode):
        calls.append(views)
        if len(calls)==20:
            raise RuntimeError('simulated interruption')
        return fake_decision(rows,observations,views=views,mode=mode)
    monkeypatch.setattr(decision,'decide_cached',interrupted)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        official.predict_official(cfg,root)
    completed=list((cfg['output_root']/'scenes').glob('*/predictions.json'))
    assert len(completed)==1
    original=completed[0].read_bytes()
    monkeypatch.setattr(decision,'decide_cached',fake_decision)
    assert official.predict_official(cfg,root)['status']=='prediction_complete'
    assert completed[0].read_bytes()==original

def test_report_only_accepts_complete_sealed_metrics_and_rejects_changed_denominator(fixture, monkeypatch):
    import pandas as pd
    from vpi_next_evaluation import official_reporting as reporting
    root, frozen, prerequisites=fixture
    monkeypatch.setattr(reporting,'OFFICIAL',official.OFFICIAL)
    monkeypatch.setattr(reporting,'verify_prerequisites',lambda config, source_root=None:(frozen,prerequisites))
    for role in ('seen','unseen'):
        official.predict_official(config(root,role),root)
    summary_config={'seen_prediction_root':root/'seen-automatic','unseen_prediction_root':root/'unseen-automatic','output_root':root/'summary',
                    'cache_roots':{role:root/(role+'-cache') for role in official.OFFICIAL}}
    official.finalize_official(summary_config,root)
    evaluation=root/'evaluation'
    receipt={'schema':'vpi-current-official-evaluation/v1','status':'evaluation_complete',
             'no_retuning':True,'uncertainty_unit':'asset_group','scene_denominator_preserved':True,
             'modes':['automatic'],'prediction_summary_lock_sha256':{'automatic':digest_file(root/'summary/summary_lock.json')},**prerequisites}
    atomic_json(evaluation/'evaluation_receipt.json',receipt)
    rows=[{'role':role,'mode':'automatic','correspondence_provenance':'predicted','view_set_id':schedule,
           'view_count':len(views),'metric':metric,'mean':.5,'ci_low':.2,'ci_high':.8,'scene_count':2,'asset_group_count':1}
          for role in ('seen','unseen') for schedule,views in VIEW_SETS.items() for metric in official.REQUIRED_METRICS]
    path=evaluation/'shareable/view_schedule_summary.csv'
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    seal(evaluation,[evaluation/'evaluation_receipt.json',path],name='evaluation_lock.json')
    cfg={'evaluation_root':evaluation,'prediction_summary_roots':{'automatic':root/'summary'}}
    table,_=reporting.read_official_aggregates(cfg)
    assert len(table)==2*19*4
    rows[0]['scene_count']=1
    pd.DataFrame(rows).to_csv(path,index=False)
    seal(evaluation,[evaluation/'evaluation_receipt.json',path],name='evaluation_lock.json')
    with pytest.raises(ValueError,match='denominators'):
        reporting.read_official_aggregates(cfg)


@pytest.fixture
def evaluation_fixture(fixture, monkeypatch):
    from vpi_next_evaluation import official_evaluation as evaluation
    from vpi_next_evaluation import official_reporting as reporting
    root,frozen,prerequisites=fixture
    # The fixture replaces the prerequisite gate; expose its synthetic original-role assessment explicitly.
    monkeypatch.setattr(evaluation, 'verify_exposure', lambda role, path: {'role': role, 'status': 'synthetic_prerequisite_verified', 'independent_confirmation': False})
    for module in (evaluation,reporting):
        monkeypatch.setattr(module,'OFFICIAL',official.OFFICIAL)
        monkeypatch.setattr(module,'verify_prerequisites',lambda cfg, source_root=None:(frozen,prerequisites))
    predictions={mode:{role:root/(role+'-'+mode) for role in official.OFFICIAL} for mode in ('automatic','assisted')}
    summaries={mode:root/('summary-'+mode) for mode in predictions}
    caches={role:root/(role+'-cache') for role in official.OFFICIAL}
    for mode in predictions:
        for role in official.OFFICIAL:
            official.predict_official(config(root,role,mode),root)
        official.finalize_official({'seen_prediction_root':predictions[mode]['seen'],
            'unseen_prediction_root':predictions[mode]['unseen'],'cache_roots':caches,'output_root':summaries[mode]},root)
    evaluators={role:root/('truth-'+role) for role in official.OFFICIAL}
    index_hashes=[]
    for role,contract in official.OFFICIAL.items():
        manifest=read_json(caches[role]/'cohort_manifest.json')
        evaluator=evaluators[role]
        index={scene:'assetA' for scene in manifest['scenes']}
        atomic_json(evaluator/'scene_index.json',index)
        index_hashes.append(digest_file(evaluator/'scene_index.json'))
        atomic_json(evaluator/'current_evaluator_manifest.json',{
            'schema':evaluation.SCHEMA,'role':role,'method_freeze_sha256':prerequisites['method_freeze_sha256'],
            'membership_sha256':contract.membership_sha256,'asset_index_sha256':index_hashes[-1],
            'cache_lock_sha256':digest_file(caches[role]/'artifact_lock.json')})
        for scene in manifest['scenes']:
            cached=read_json(caches[role]/'scenes'/f'{scene}.json')
            atomic_json(evaluator/'scenes'/f'{scene}.json',{'labels':{'R1':1,'R2':0,'R3':0},'asset_group':'assetA',
                'correspondence':{f"automatic/{row['view_id']}/{row['candidate_id']}":'R1' for row in cached['bank_observations']}})
        seal(evaluator,[p for p in evaluator.rglob('*') if p.is_file()])
    audit=root/'audit'
    atomic_json(audit/'official_exposure_ledger.json',{'reviewed_source_sha256':index_hashes})
    cfg={'prediction_roots':predictions,'prediction_summary_roots':summaries,'cache_roots':caches,
         'evaluator_roots':evaluators,'official_exposure_root':audit,'exposure_root':root/'synthetic-original-audit','output_root':root/'typed-evaluation'}
    return root,cfg


def test_typed_official_evaluation_keeps_scene_denominators_and_can_resume(evaluation_fixture):
    from vpi_next_evaluation.official_evaluation import evaluate_official
    from vpi_next_evaluation.official_reporting import read_official_aggregates
    root,cfg=evaluation_fixture
    receipt=evaluate_official(cfg,root)
    assert receipt['modes']==['automatic','assisted']
    assert receipt['uncertainty_unit']=='asset_group'
    assert receipt['original_role_exposure']['independent_confirmation'] is False
    table,_=read_official_aggregates({**cfg,'evaluation_root':cfg['output_root']},root)
    assert len(table)==2*2*19*4
    assert set(table.scene_count)=={2} and set(table.asset_group_count)=={1}
    primary=table[(table.view_set_id=='three-primary')&(table.metric=='top1_tie_expected')]
    assert set(primary[primary['mode']=='automatic']['mean'])=={0.0}
    assert set(primary[primary['mode']=='assisted']['mean'])=={1.0}
    assert evaluate_official(cfg,root)==receipt


def test_corrupt_prediction_stops_before_any_evaluator_access(evaluation_fixture,monkeypatch):
    from vpi_next_evaluation import official_evaluation as evaluation
    root,cfg=evaluation_fixture
    prediction=next(cfg['prediction_roots']['automatic']['seen'].glob('scenes/*/predictions.json'))
    prediction.write_text('{}')
    cfg['evaluator_roots']={role:root/'must-not-be-opened' for role in official.OFFICIAL}
    with pytest.raises(ValueError):
        evaluation.evaluate_official(cfg,root)
    assert not cfg['output_root'].exists()


def test_finalizer_rejects_resealed_unselected_members_and_changed_inventory(fixture):
    root,_,_=fixture
    cfg=config(root,'seen','assisted')
    official.predict_official(cfg,root)
    path=next(cfg['output_root'].glob('scenes/*/predictions.json'))
    payload=read_json(path)
    payload['settings'][0]['ranking'][0]['members'][0]['view_id']='unselected'
    atomic_json(path,payload)
    lock=read_json(cfg['output_root']/'prediction_lock.json')
    seal(cfg['output_root'],[cfg['output_root']/name for name in lock['files']],name='prediction_lock.json')
    with pytest.raises(ValueError):
        official.validate_prediction(cfg['output_root'],'seen',cache_root=cfg['cache_root'])


def test_report_cannot_relabel_supplied_correspondence_as_user_confirmation(evaluation_fixture):
    import pandas as pd
    from vpi_next_evaluation.official_evaluation import evaluate_official
    from vpi_next_evaluation.official_reporting import read_official_aggregates
    root,cfg=evaluation_fixture
    evaluate_official(cfg,root)
    output=cfg['output_root']
    receipt=read_json(output/'evaluation_receipt.json')
    receipt['assisted_correspondence_provenance']='user_confirmed'
    atomic_json(output/'evaluation_receipt.json',receipt)
    path=output/'shareable/view_schedule_summary.csv'
    table=pd.read_csv(path)
    table.loc[table['mode']=='assisted','correspondence_provenance']='user_confirmed'
    table.to_csv(path,index=False)
    lock=read_json(output/'evaluation_lock.json')
    seal(output,[output/name for name in lock['files']],name='evaluation_lock.json')
    with pytest.raises(ValueError,match='relabels'):
        read_official_aggregates({**cfg,'evaluation_root':output},root)


def test_report_builds_figures_and_counts_missing_media_without_changing_science(evaluation_fixture,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from vpi_next_evaluation.official_evaluation import evaluate_official
    from vpi_next_evaluation.official_reporting import render_official_report
    root,cfg=evaluation_fixture
    evaluate_official(cfg,root)
    before=digest_file(cfg['output_root']/'evaluation_lock.json')
    report=render_official_report({**cfg,'evaluation_root':cfg['output_root'],'report_root':root/'report'},root)
    assert report['gallery_count']==12 and report['missing_media_examples']==12
    assert len(report['aggregate_figures'])==9
    from publish_next_stage import verify_publication
    artifacts=verify_publication('10b',cfg['output_root'],root/'report')
    assert artifacts
    assert read_json(cfg['output_root']/'run_binding.json')['stage']=='10b'
    assert (cfg['output_root']/'shareable/evaluation_receipt.json').is_file()
    assert digest_file(cfg['output_root']/'evaluation_lock.json')==before
