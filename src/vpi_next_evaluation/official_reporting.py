"""Read-only aggregate report compatibility, without implementing outcome access."""
from pathlib import Path
import numpy as np
import pandas as pd
from vpi_object_bank.storage import digest_file, read_json, verify_seal
from .official import OFFICIAL, REQUIRED_METRICS, verify_prerequisites

COLUMNS = ('role', 'mode', 'correspondence_provenance', 'view_set_id', 'view_count',
           'metric', 'mean', 'ci_low', 'ci_high', 'scene_count', 'asset_group_count')

def read_official_aggregates(config, source_root=None):
    freeze, prerequisites = verify_prerequisites(config, source_root)
    root = Path(config['evaluation_root'])
    lock = verify_seal(root, name='evaluation_lock.json')
    required = {'evaluation_receipt.json', 'shareable/view_schedule_summary.csv'}
    if not required <= set(lock['files']):
        raise ValueError('Official evaluation seal omits the aggregate result or receipt')
    receipt = read_json(root / 'evaluation_receipt.json')
    if receipt.get('schema') != 'vpi-current-official-evaluation/v1' or receipt.get('status') != 'evaluation_complete':
        raise ValueError('Legacy official reports do not establish current-method evaluation')
    if any(receipt.get(key) != value for key, value in prerequisites.items()) or receipt.get('no_retuning') is not True:
        raise ValueError('Official evaluation differs from current prerequisites or changed methods')
    if receipt.get('uncertainty_unit') != 'asset_group' or receipt.get('scene_denominator_preserved') is not True:
        raise ValueError('Official report must preserve scene denominators and grouped intervals')
    modes = receipt.get('modes')
    if not modes or set(modes) - {'automatic', 'assisted'}:
        raise ValueError('Official report has invalid mode provenance')
    for mode in modes:
        summary_root = Path(config['prediction_summary_roots'][mode])
        summary_lock = verify_seal(summary_root, name='summary_lock.json')
        if not {'shareable/official_prediction_summary.json','run_binding.json'}<=set(summary_lock['files']):
            raise ValueError('Official prediction summary is not sealed')
        summary_binding=read_json(summary_root/'run_binding.json')
        if any(summary_binding.get(key)!=value for key,value in prerequisites.items()):
            raise ValueError('Official summary has different freeze or exposure prerequisites')
        summary = read_json(summary_root / 'shareable/official_prediction_summary.json')
        if summary.get('mode') != mode or summary.get('view_sets') != freeze['view_sets']:
            raise ValueError('Official evaluation summary changed modes or schedules')
        expected_provenance='predicted' if mode=='automatic' else receipt.get('assisted_correspondence_provenance')
        if summary.get('correspondence_provenance')!=expected_provenance:
            raise ValueError('Official report relabels prediction correspondence provenance')
        if receipt.get('prediction_summary_lock_sha256', {}).get(mode) != digest_file(summary_root / 'summary_lock.json'):
            raise ValueError('Evaluation is not bound to the sealed official prediction pair')
    table = pd.read_csv(root / 'shareable/view_schedule_summary.csv', keep_default_na=False)
    if set(table.columns) != set(COLUMNS) or table.empty:
        raise ValueError('Official aggregate schema must be exact and contain no scene identifiers')
    if table.duplicated(['role', 'mode', 'correspondence_provenance', 'view_set_id', 'metric']).any():
        raise ValueError('Duplicated official metric rows')
    expected = {(role, mode, schedule, metric) for role in OFFICIAL for mode in modes
                for schedule in freeze['view_sets'] for metric in REQUIRED_METRICS}
    actual = set(zip(table.role, table['mode'], table.view_set_id, table.metric))
    if actual != expected or len(table) != len(expected):
        raise ValueError('Official results omit modes, roles, schedules or required metrics')
    for row in table.itertuples(index=False):
        contract = OFFICIAL[row.role]
        if (row.scene_count, row.asset_group_count, row.view_count) != (contract.scene_count, contract.asset_group_count, len(freeze['view_sets'][row.view_set_id])):
            raise ValueError('Official metric denominators or selected view counts changed')
        provenance = 'predicted' if row.mode == 'automatic' else receipt.get('assisted_correspondence_provenance')
        if row.correspondence_provenance != provenance or provenance not in {'predicted', 'dataset_reference', 'user_confirmed'}:
            raise ValueError('Official mode and assistance provenance are inconsistent')
        values = np.array([row.mean, row.ci_low, row.ci_high], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or row.ci_low > row.ci_high:
            raise ValueError('Official metrics require finite proportions and valid intervals')
    return table, receipt


def render_official_report(config, source_root=None):
    """Repair aggregate figures and fixed private examples after the science seal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vpi_object_bank.storage import atomic_json, atomic_unit, bind, seal, verified_unit
    from .official_evaluation import canonical_setting
    from .reporting import scene_panel
    from .diagnostic_figures import render_diagnostic_figures
    table,receipt=read_official_aggregates(config,source_root)
    evaluation=Path(config['evaluation_root']);root=Path(config['report_root'])
    sources={'evaluation_lock_sha256':digest_file(evaluation/'evaluation_lock.json'),'source_sha256':digest_file(__file__),
             'shared_renderer_sha256':digest_file(Path(__file__).with_name('reporting.py')),
             'diagnostic_renderer_sha256':digest_file(Path(__file__).with_name('diagnostic_figures.py'))}
    cached={}
    for role in OFFICIAL:
        cache=Path(config['cache_roots'][role]);lock=verify_seal(cache)
        cached[role]=(cache,lock)
        for mode in receipt['modes']:
            prediction=Path(config['prediction_roots'][mode][role])
            verify_seal(prediction,name='prediction_lock.json')
            if read_json(prediction/'run_binding.json')['cache_lock_sha256']!=digest_file(cache/'artifact_lock.json'):
                raise ValueError('Official report mixes cache and prediction evidence')
            expected=read_json(Path(config['prediction_summary_roots'][mode])/'shareable/official_prediction_summary.json')['prediction_locks'][role]
            if digest_file(prediction/'prediction_lock.json')!=expected:
                raise ValueError('Official report prediction differs from its evaluated summary')
            sources[f'{mode}/{role}']=expected
    binding_hash=bind(root,sources)
    if (root/'report_lock.json').exists():
        verify_seal(root,name='report_lock.json');return read_json(root/'report_receipt.json')
    figures=root/'shareable/figures';figures.mkdir(parents=True,exist_ok=True)
    for metric in REQUIRED_METRICS:
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,constrained_layout=True)
        for ax,role in zip(axes,OFFICIAL):
            subset=table[(table.role==role)&(table.metric==metric)&table.view_set_id.str.endswith('primary')]
            for mode,rows in subset.groupby('mode'):
                rows=rows.sort_values('view_count')
                ax.plot(rows.view_count,100*rows['mean'],marker='o',label=mode+' · '+str(rows.correspondence_provenance.iloc[0]))
                ax.fill_between(rows.view_count,100*rows.ci_low,100*rows.ci_high,alpha=.12)
            ax.set(title=role+' official split',xlabel='Views (1 diagnostic)',xticks=[1,2,3,5],ylim=(0,100));ax.legend()
        fig.suptitle('Fixed methods; exposure-qualified evaluation, no independent-confirmation claim',fontsize=10)
        axes[0].set_ylabel(metric+' (%)');fig.savefig(figures/(metric+'.png'),dpi=120);plt.close(fig)
    diagnostics=render_diagnostic_figures(evaluation,root)
    plans=[]
    for role in OFFICIAL:
        cache,lock=cached[role]
        first=sorted(read_json(cache/'cohort_manifest.json')['scenes'])[0]
        for mode in receipt['modes']:
            payload=read_json(cache/'scenes'/f'{first}.json')
            prediction=read_json(Path(config['prediction_roots'][mode][role])/'scenes'/first/'predictions.json')
            for schedule in ('two-primary','three-primary','five-primary'):
                unit=root/'private/galleries'/role/mode/schedule
                if not verified_unit(unit,binding_hash):
                    setting=canonical_setting(next(s for s in prediction['settings'] if s['view_set_id']==schedule))
                    with atomic_unit(unit,binding_hash) as temporary:
                        result=scene_panel(payload,setting,cache,lock,temporary/'objects_masks_selection.png')
                        atomic_json(temporary/'receipt.json',{'role':role,'mode':mode,'scene':first,
                            'view_set_id':schedule,'selection':'fixed first scene; outcome-independent',**result})
                plans.append(read_json(unit/'receipt.json'))
    atomic_json(root/'private/gallery_manifest.json',plans)
    result={'status':'report_complete','aggregate_figures':[metric+'.png' for metric in REQUIRED_METRICS]+[item['file'] for item in diagnostics['figures']],
            'gallery_count':len(plans),'missing_media_examples':sum(bool(plan['missing_media_views']) for plan in plans),
            'scientific_results_unchanged':True,'tracking_status':'pending bounded aggregate publication',
            'private_evidence':'images, IDs and per-object scores remain private'}
    atomic_json(root/'report_receipt.json',result)
    seal(root,[p for p in root.rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p)],name='report_lock.json')
    return result
