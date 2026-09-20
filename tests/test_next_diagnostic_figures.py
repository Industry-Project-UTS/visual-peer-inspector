import json
import pandas as pd
import pytest
from vpi_next_evaluation.diagnostic_figures import _matrix, render_diagnostic_figures
from vpi_object_bank.storage import seal


def test_exposure_qualification_survives_report_rendering(tmp_path):
    root=tmp_path/'evaluation';share=root/'shareable';share.mkdir(parents=True)
    (share/'evaluation_receipt.json').write_text(json.dumps({
        'evaluation_claim':'Exposure-qualified fixed-method evaluation; not independent confirmation.'}))
    pd.DataFrame([{'route':'automatic','method':'winner_cluster_consensus',
                  'correspondence_provenance':'predicted','view_count':3,
                  'metric':'candidate_coverage','mean':1.0}]).to_csv(share/'failure_summary.csv',index=False)
    pd.DataFrame([{'route':'automatic','method':'winner_cluster_consensus',
                  'correspondence_provenance':'predicted','view_count':3,
                  'metric':'identity_precision','mean':0.5}]).to_csv(share/'identity_summary.csv',index=False)
    pd.DataFrame([{'route':'automatic','method':'winner_cluster_consensus',
                  'correspondence_provenance':'predicted','view_count':3,
                  'view_set_id':'three-primary','metric':'top1_tie_expected','mean':0.5}]).to_csv(share/'view_schedule_summary.csv',index=False)
    seal(root,list(share.iterdir()),name='evaluation_lock.json')
    result=render_diagnostic_figures(root,tmp_path/'report')
    assert 'Exposure-qualified' in result['evaluation_caption']
    assert 'not independent confirmation' in result['evaluation_caption']
    assert result['scientific_computation_repeated'] is False


def test_provenance_is_never_pooled_and_duplicates_fail():
    frame = pd.DataFrame([
        {'route':'assisted','correspondence_provenance':p,'view_count':3,'metric':'precision','mean':v}
        for p,v in [('dataset_reference',.7),('user_confirmed',.6)]])
    assert len(_matrix(frame,column='metric',include_views=True))==2
    with pytest.raises(ValueError,match='Duplicate aggregate'):
        _matrix(pd.concat([frame,frame]),column='metric',include_views=True)


def test_sealed_aggregate_figures_keep_undefined_values(tmp_path):
    root=tmp_path/'evaluation';share=root/'shareable';share.mkdir(parents=True)
    base={'role':'fixture','route':'automatic','method':'winner_cluster_consensus',
          'correspondence_provenance':'predicted','view_count':1}
    for name,metric,value in [('failure_summary.csv','candidate_coverage',.5),
                              ('identity_summary.csv','identity_precision',None)]:
        pd.DataFrame([{**base,'metric':metric,'mean':value}]).to_csv(share/name,index=False)
    pd.DataFrame([{**base,'view_set_id':'one-primary','metric':'top1_tie_expected','mean':.4}]).to_csv(share/'view_schedule_summary.csv',index=False)
    seal(root,list(share.glob('*.csv')),name='evaluation_lock.json')
    report=tmp_path/'report'
    result=render_diagnostic_figures(root,report)
    assert len(result['figures'])==3
    identity=next(row for row in result['figures'] if row['file']=='identity_quality.png')
    assert identity['undefined_cells']==4
    assert result['scientific_computation_repeated'] is False
    assert all((report/'shareable/figures'/row['file']).stat().st_size>100 for row in result['figures'])
