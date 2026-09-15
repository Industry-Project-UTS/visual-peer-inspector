"""Plot the two selected routes from sealed aggregate evidence, without rescoring."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from vpi_object_bank.storage import atomic_json, digest_file, verify_seal


def plot(cpu_root, reference_root, output):
    verify_seal(cpu_root, name='aggregate_lock.json')
    verify_seal(reference_root, name='publication_lock.json')
    cpu = pd.read_csv(cpu_root / 'metric_summary.csv')
    reference = pd.read_csv(reference_root / 'tables/view_schedule_summary.csv')
    a = cpu[(cpu.method == 'cpu_app_winner_consensus') & (cpu.metric == 'top1_tie_expected')].copy()
    b = reference[(reference.source_stage == '08O') & (reference.route == 'automatic_reference') &
                  (reference.method == 'object_bank_top2') & (reference.metric == 'top1_tie_expected') &
                  reference.view_count.isin([2,3,5]) & reference.view_set_id.str.endswith('-primary')].copy()
    for frame in (a,b):
        assert set(frame.view_count) == {2,3,5} and len(frame) == 3
        assert (frame.scene_count == 100).all() and (frame.asset_group_count == 27).all()
    a['route_label'] = 'Automatic · current CPU decision'
    b['route_label'] = 'Assisted diagnostic · supplied object IDs'
    rows = pd.concat([a,b],ignore_index=True)[['route_label','view_count','mean','ci_low','ci_high','scene_count','asset_group_count']]
    output.mkdir(parents=True,exist_ok=True)
    rows.to_csv(output / 'selected_primary_top1.csv',index=False)
    fig, ax = plt.subplots(figsize=(9,5.4), layout='constrained')
    for frame, colour in [(a,'#4f46a5'),(b,'#0f766e')]:
        frame=frame.sort_values('view_count')
        y=frame['mean'].to_numpy()*100
        errors=[y-frame.ci_low.to_numpy()*100,frame.ci_high.to_numpy()*100-y]
        ax.errorbar(frame.view_count,y,yerr=errors,marker='o',capsize=5,lw=2,color=colour,label=frame.route_label.iloc[0])
        for x,value in zip(frame.view_count,y):
            ax.annotate(f'{value:.1f}%',(x,value),xytext=(9,6),textcoords='offset points',color=colour,weight='bold')
    ax.axhline(60,color='#777',ls='--',lw=1,label='Project target (60%)')
    ax.set(xticks=[2,3,5],xlabel='Uploaded views in the fixed primary schedule',ylabel='Tie-aware top-1 (%)',ylim=(0,90),xlim=(1.7,5.6))
    ax.grid(axis='y',alpha=.18)
    ax.set_title('Multi-view object ranking on the existing development cohort\n100 scenes · 27 asset groups · 95% asset-group intervals',loc='left',pad=15)
    ax.legend(loc='lower left',frameon=False)
    fig.supxlabel('Predicted regions in both routes. Cached development evidence; no independent test or human-confirmation study.',fontsize=9)
    fig.savefig(output / 'selected_primary_top1.png',dpi=180)
    fig.savefig(output / 'selected_primary_top1.svg')
    plt.close(fig)
    atomic_json(output / 'selected_primary_top1.receipt.json',{
        'source_cpu_lock_sha256':digest_file(cpu_root/'aggregate_lock.json'),
        'source_reference_lock_sha256':digest_file(reference_root/'publication_lock.json'),
        'script_sha256':digest_file(Path(__file__)), 'scene_count':100,'asset_group_count':27,
        'selection':'two already selected methods; predefined primary 2/3/5 view schedules',
        'scope':'aggregate visualization only; no labels opened or scores recomputed'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-root',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plot(args.cpu_root,args.reference_root,args.output)
