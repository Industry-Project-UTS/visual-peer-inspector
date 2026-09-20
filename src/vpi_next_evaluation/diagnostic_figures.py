"""Render additional report figures from sealed aggregates without recomputing science."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vpi_object_bank.storage import atomic_json, digest_file, verify_seal, read_json


SERIES = ('role', 'mode', 'route', 'method', 'correspondence_provenance')
PANELS = (
    ('failure_summary.csv', 'failure_breakdown',
     ('detection_loss', 'pruning_loss', 'association_loss', 'score_unavailability_loss', 'ranking_loss'), True),
    ('failure_summary.csv', 'candidate_coverage',
     ('candidate_coverage', 'candidate_anomaly_recall', 'rankable'), True),
    ('identity_summary.csv', 'identity_quality',
     ('identity_precision', 'identity_recall', 'identity_purity', 'identity_reference_coverage'), True),
    ('identity_summary.csv', 'identity_fragmentation', ('identity_fragmentation',), False),
)


def exposure_caption(evaluation_root):
    root=Path(evaluation_root)
    for relative in ('shareable/evaluation_receipt.json','evaluation_receipt.json'):
        path=root/relative
        if path.is_file():
            receipt=read_json(path)
            if 'exposure-qualified' in str(receipt).lower() or receipt.get('exposure_assessment',{}).get('status')=='exposure_qualified':
                return 'Exposure-qualified fixed-method evaluation · not independent confirmation'
    return 'Fixed-method scene-level evaluation · use the recorded cohort and access policy'


def _labels(frame, include_views):
    columns = [name for name in SERIES if name in frame]
    if include_views:
        columns.append('view_count')
    if not columns:
        raise ValueError('Aggregate series identity is missing')
    return frame[columns].fillna('unspecified').astype(str).agg(' | '.join, axis=1)


def _matrix(frame, *, column, include_views):
    frame = frame.copy()
    frame['series'] = _labels(frame, include_views)
    if frame.duplicated(['series', column]).any():
        raise ValueError('Duplicate aggregate rows cannot be silently averaged for a report')
    return frame.pivot(index='series', columns=column, values='mean').sort_index()


def _draw(matrix, destination, title, percentage):
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if np.isinf(values).any() or (finite.size and ((finite < -1e-8).any() or (percentage and (finite > 1+1e-8).any()))):
        raise ValueError('Aggregate rates are outside their valid range')
    display = values * (100 if percentage else 1)
    fig, ax = plt.subplots(figsize=(max(9, .7*len(matrix.columns)), max(3.5, .48*len(matrix)+1.8)), constrained_layout=True)
    palette = plt.get_cmap('Blues').with_extremes(bad='#eeeeee')
    image = ax.imshow(np.ma.masked_invalid(display), aspect='auto', cmap=palette,
                      vmin=0, vmax=100 if percentage else max(1, float(np.max(finite))) if finite.size else 1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title(title, fontsize=11)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = display[i,j]
            ax.text(j,i,'—' if np.isnan(value) else f'{value:.1f}',ha='center',va='center',fontsize=7,
                    color='white' if percentage and value>65 else '#222222')
    fig.colorbar(image, ax=ax, label='Percent' if percentage else 'Mean extra tracks per reference object')
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return {'rows': len(matrix), 'columns': len(matrix.columns), 'undefined_cells': int(np.isnan(values).sum())}


def render_diagnostic_figures(evaluation_root, report_root):
    evaluation_root, report_root = Path(evaluation_root), Path(report_root)
    lock = verify_seal(evaluation_root, name='evaluation_lock.json')
    output = report_root/'shareable/figures'
    output.mkdir(parents=True, exist_ok=True)
    sources, artifacts, unavailable = {}, [], []
    caption=exposure_caption(evaluation_root)

    def read(name):
        relative = 'shareable/'+name
        if relative not in lock['files']:
            raise ValueError(f'Required aggregate is not scientifically sealed: {relative}')
        path = evaluation_root/relative
        sources[relative] = digest_file(path)
        table = pd.read_csv(path, keep_default_na=False)
        table['mean'] = pd.to_numeric(table['mean'].mask(table['mean'].eq(''), np.nan), errors='raise')
        return table

    for table, name, metrics, percent in PANELS:
        frame = read(table)
        frame = frame[frame.metric.isin(metrics)]
        frame['mean'] = pd.to_numeric(frame['mean'], errors='raise')
        if frame.empty:
            unavailable.append(name)
            continue
        matrix = _matrix(frame, column='metric', include_views=True).reindex(columns=list(metrics))
        details = _draw(matrix, output/(name+'.png'), name.replace('_',' ').capitalize()+' · scene means across fixed schedules\n'+caption, percent)
        artifacts.append({'file': name+'.png', **details})

    frame = read('view_schedule_summary.csv')
    frame = frame[frame.metric=='top1_tie_expected'].copy()
    frame['mean'] = pd.to_numeric(frame['mean'], errors='raise')
    if not frame.empty:
        matrix = _matrix(frame, column='view_set_id', include_views=False)
        details = _draw(matrix, output/'schedule_sensitivity.png',
                        'Tie-aware top-1 for every predefined schedule · uncertainty intervals remain in the source table\n'+caption, True)
        artifacts.append({'file': 'schedule_sensitivity.png', **details})
    receipt = {'status': 'complete', 'figures': artifacts, 'unavailable': unavailable,
               'evaluation_lock_sha256': digest_file(evaluation_root/'evaluation_lock.json'),
               'renderer_sha256': digest_file(__file__), 'sources': sources,
               'undefined_values': 'shown as a dash, never replaced by zero',
               'scientific_computation_repeated': False, 'evaluation_caption':caption}
    atomic_json(report_root/'diagnostic_figures_receipt.json', receipt)
    return receipt
