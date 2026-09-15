"""Rich aggregate and private scene evidence, repairable after the scientific seal."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vpi_object_bank.reporting import gallery
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest_file, read_json,
                                    seal, source_hashes, verified_unit, verify_seal)
from .workbench import runtime_versions, resolved_config, verify_inputs


def gallery_plan(cfg):
    manifest = read_json(Path(cfg['bank']['prepared_root']) / 'input_manifest.json')
    first = manifest['scenes'][0]
    choices = []
    for schedule in ('one-primary', 'two-primary', 'three-primary', 'five-primary'):
        for route, method in [('automatic_predicted', 'winner_cluster_consensus'), ('automatic_reference', 'object_bank_top2')]:
            choices.append({'scene': first, 'route': route, 'view_set_id': schedule, 'method': method,
                            'reason': 'fixed first development scene; outcome-independent example'})
    metrics = pd.read_csv(Path(cfg['bank']['evaluation_root']) / 'private/scene_metrics.csv')
    subset = metrics[(metrics.route == 'automatic_predicted') & (metrics.method == 'winner_cluster_consensus')
                     & (metrics.view_set_id == 'three-primary')].sort_values('scene')
    strata = [('ranking failure', subset.top1_tie_expected == 0),
              ('tied ranking', subset.top_tie_count > 1),
              ('selected object missing from some views', subset.winner_support < subset.view_count),
              ('unknown candidate correspondence', subset.unknown_candidate_count > 0)]
    for reason, flag in strata:
        examples = subset[flag]
        if len(examples):
            choices.append({'scene': str(examples.iloc[0].scene), 'route': 'automatic_predicted',
                'view_set_id': 'three-primary', 'method': 'winner_cluster_consensus',
                'reason': f'post-hoc development illustration: {reason}; not an evaluation subset'})
    # Keep each requested stratum in the manifest even when one scene illustrates several.
    return choices


def score_panel(cfg, choice, output):
    scene, route, schedule, method = (choice[k] for k in ('scene', 'route', 'view_set_id', 'method'))
    payload = read_json(Path(cfg['bank']['prediction_root']) / 'scenes' / scene / 'predictions.json')
    setting = next(row for row in payload['settings'] if row['route'] == route and row['view_set_id'] == schedule)
    objects = sorted(setting['rankings'][method], key=lambda row: str(row['id']))
    views = setting['views']
    scorer = 'cached_forward' if method == 'winner_cluster_consensus' else method.rsplit('_', 1)[0]
    observed = {(row['view_id'], row['candidate_id']): row for row in setting['observations']}
    ranks = setting['per_view_ranks'][scorer]
    rows, matrix = [], np.full((len(objects), len(views)), np.nan)
    for index, obj in enumerate(objects):
        members = {m['view_id']: m['candidate_id'] for m in obj['members']}
        for j, view in enumerate(views):
            candidate = members.get(view)
            observation = observed.get((view, candidate), {})
            rank = ranks.get(f'{view}/{candidate}') if candidate else None
            matrix[index, j] = np.nan if rank is None else rank
            row = {'object': f'O{index+1}', 'view': view, 'candidate': candidate,
                   'view_rank': rank, 'combined_score': obj['score'], 'support': obj['support']}
            for component in ('cached_forward', 'within_view', 'object_bank', 'view_mean_control'):
                row[component] = observation.get(component, {}).get('score')
            rows.append(row)
    from vpi_object_bank.evaluation import _atomic_csv
    _atomic_csv(output / 'object_scores.csv', pd.DataFrame(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, .5 * len(objects))), constrained_layout=True)
    labels = [f'O{i+1}' for i in range(len(objects))]
    if objects:
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#dddddd')
        plot = axes[0].imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect='auto')
        axes[0].set(xticks=range(len(views)), xticklabels=views, yticks=range(len(objects)), yticklabels=labels)
        for i in range(len(objects)):
            for j in range(len(views)):
                axes[0].text(j, i, '—' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if matrix[i,j] < .6 else 'black')
        fig.colorbar(plot, ax=axes[0], label='Within-view percentile rank')
        values = [np.nan if obj['score'] is None else obj['score'] for obj in objects]
        axes[1].barh(labels, values, color='#246581')
        for i, obj in enumerate(objects):
            axes[1].text(0, i, f"  support {obj['support']}/{len(views)}", va='center', fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set(xlim=(0,1.05), xlabel='Final combined object score')
    else:
        for ax in axes:
            ax.text(.5, .5, 'No candidate objects', ha='center', va='center', transform=ax.transAxes)
    evidence_note = 'Forward ranks shown; automatic winner selection used forward/P1/P2 consensus' if method == 'winner_cluster_consensus' else 'Object-bank ranks and strongest-two fusion'
    axes[0].set_title('Every retained object across selected views\nGrey cells: unavailable or unmatched')
    axes[1].set_title('One combined physical-object ranking')
    fig.suptitle(evidence_note)
    fig.savefig(output / 'score_matrix.png', dpi=130)
    plt.close(fig)
    return {'objects': len(objects), 'views': len(views), 'score_explanation': evidence_note}


def aggregate_figures(scientific_root, output):
    table = pd.read_csv(scientific_root / 'shareable/metric_summary.csv')
    primary = pd.read_csv(scientific_root / 'shareable/view_schedule_summary.csv')
    titles = {'supplied_reference': 'Supplied masks + supplied IDs', 'automatic_reference': 'Predicted masks + supplied IDs',
              'automatic_predicted': 'Predicted masks + predicted IDs'}
    files = []
    for metric in ('top1_tie_expected', 'ap_tie_expected', 'recall_at_2_tie_expected', 'reciprocal_rank_tie_expected'):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
        for ax, (route, title) in zip(axes, titles.items(), strict=True):
            selected = table[(table.source_stage == '08O') & (table.route == route) & (table.metric == metric)]
            for method, rows in selected.groupby('method'):
                rows = rows.sort_values('view_count')
                ax.plot(rows.view_count, 100*rows['mean'], marker='o', linewidth=1.2, label=method)
                ax.fill_between(rows.view_count, 100*rows.ci_low, 100*rows.ci_high, alpha=.035)
            ax.set(title=title, xlabel='Selected views (1 is diagnostic)', xticks=[1,2,3,5], ylim=(0,100))
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc='lower right')
        axes[0].set_ylabel(metric.replace('_tie_expected','').replace('_',' ') + ' (%)')
        fig.suptitle('All registered 08O techniques: scene-averaged schedules; asset-group 95% intervals\nExisting development100 only; no protected evaluation')
        filename = f'{metric}.png'
        fig.savefig(output / filename, dpi=130); plt.close(fig); files.append(filename)
    fig, axes = plt.subplots(1, 4, figsize=(23, 8), constrained_layout=True)
    panels = [('08O', route, title) for route, title in titles.items()] + [('08P', 'automatic_predicted', 'Mapping comparison; assisted row explicitly named')]
    for ax, (stage, route, title) in zip(axes, panels, strict=True):
        rows = primary[(primary.source_stage == stage) & (primary.route == route) & (primary.metric == 'top1_tie_expected') & primary.view_set_id.str.endswith('primary')]
        pivot = rows.pivot(index='method', columns='view_count', values='mean').reindex(columns=[1,2,3,5])
        image = ax.imshow(pivot*100, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        ax.set(xticks=range(4), xticklabels=['1 diag','2','3','5'], yticks=range(len(pivot)), yticklabels=pivot.index, title=title)
        ax.tick_params(axis='y', labelsize=7)
        for i in range(len(pivot)):
            for j in range(4):
                ax.text(j,i,f'{100*pivot.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8)
    fig.colorbar(image, ax=axes, shrink=.7, label='Tie-aware top-1 (%)')
    fig.savefig(output/'technique_primary_matrix.png', dpi=130); plt.close(fig); files.append('technique_primary_matrix.png')
    fig, ax = plt.subplots(figsize=(13,6), constrained_layout=True)
    chosen = [('08O','automatic_reference','object_bank_top2','Assisted bank top2'),
              ('08P','automatic_predicted','winner_cluster_consensus','Automatic winner consensus'),
              ('08P','automatic_predicted','global_partial_assignment','Joint mapping diagnostic')]
    for stage, route, method, label in chosen:
        rows=primary[(primary.source_stage==stage)&(primary.route==route)&(primary.method==method)&(primary.view_count==3)&(primary.metric=='top1_tie_expected')]
        ax.plot(rows.view_set_id,rows['mean']*100,marker='o',label=label)
    ax.set(ylabel='Tie-aware top-1 (%)',title='Every predefined three-view schedule; same100 development scenes',ylim=(0,100))
    ax.tick_params(axis='x',rotation=15);ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'schedule_sensitivity.png',dpi=130);plt.close(fig);files.append('schedule_sensitivity.png')
    failures=pd.read_csv(scientific_root/'shareable/failure_summary.csv')
    components=['detection_loss','pruning_loss','association_loss','score_unavailability_loss','ranking_loss']
    fig,ax=plt.subplots(figsize=(13,6),constrained_layout=True)
    bottom=np.zeros(8);names=[]
    for route,method,label in [('automatic_predicted','winner_cluster_consensus','Automatic'),('automatic_reference','object_bank_top2','Assisted')]:
        names.extend(f'{label} {v} views' for v in [1,2,3,5])
    for component in components:
        values=[]
        for route,method,_ in [('automatic_predicted','winner_cluster_consensus','Automatic'),('automatic_reference','object_bank_top2','Assisted')]:
            rows=failures[(failures.source_stage=='08O')&(failures.route==route)&(failures.method==method)&(failures.metric==component)].set_index('view_count')
            values.extend(100*float(rows.loc[v,'mean']) for v in [1,2,3,5])
        ax.bar(names,values,bottom=bottom,label=component);bottom+=values
    ax.set(ylabel='Share of scene outcomes (%)',ylim=(0,100),title='Failure stages preserve every scene; unfilled share is tie-aware success')
    ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8)
    fig.savefig(output/'failure_decomposition.png',dpi=130);plt.close(fig);files.append('failure_decomposition.png')
    return files


def render_workbench(cfg, scientific_root, report_root, source_root):
    cfg = resolved_config(cfg)
    scientific_root, root = Path(scientific_root), Path(report_root)
    verify_seal(scientific_root, name='scientific_lock.json')
    if verify_inputs(cfg) != read_json(scientific_root / 'run_binding.json')['input_locks']:
        raise ValueError('Report configuration differs from sealed scientific inputs')
    choices = gallery_plan(cfg)
    binding={'schema':'vpi-remaining-report/v1','scientific_lock_sha256':digest_file(scientific_root/'scientific_lock.json'),
             'gallery_plan':choices,'runtime_versions':runtime_versions(),
             'source':source_hashes(source_root,['src/vpi_remaining_evaluation/reporting.py','src/vpi_object_bank/reporting.py',
                                                'src/visual_peer_inspector/representation/crops.py'])}
    binding_hash=bind(root,binding)
    if (root/'report_lock.json').exists():
        verify_seal(root,name='report_lock.json');return read_json(root/'report_receipt.json')
    figures=root/'shareable/figures';figures.mkdir(parents=True,exist_ok=True)
    files=aggregate_figures(scientific_root,figures)
    receipts=[]
    for index,choice in enumerate(choices):
        unit=root/'private/galleries'/f'example-{index+1:02d}'
        if not verified_unit(unit,binding_hash):
            with atomic_unit(unit,binding_hash) as temporary:
                picture=gallery(cfg['bank'],choice['scene'],choice['route'],choice['view_set_id'],choice['method'],temporary/'views_masks_selection.png')
                matrix=score_panel(cfg,choice,temporary)
                atomic_json(temporary/'gallery_receipt.json',{**choice,**picture,**matrix})
        receipts.append(read_json(unit/'gallery_receipt.json'))
        print(f'Gallery {index+1}/{len(choices)} saved',flush=True)
    atomic_json(root/'private/gallery_manifest.json',receipts)
    receipt={'status':'complete','aggregate_figures':files,'gallery_count':len(receipts),'gallery_view_counts':[1,2,3,5],
             'privacy':'all images/scene IDs/object scores remain under private; aggregate figures are shareable',
             'selection':'fixed first scenes plus disclosed post-hoc development failure/tie/missing-view illustrations',
             'box_semantics':'selected object region, not a validated defect boundary',
             'scientific_results_preserved':True,'tracking_status':'not_attempted; publish aggregates after report seal'}
    atomic_json(root/'report_receipt.json',receipt)
    seal(root,[p for p in root.rglob('*') if p.is_file() and p.name!='report_lock.json' and '.partial' not in str(p) and '.interrupted-' not in str(p)],name='report_lock.json')
    return receipt
