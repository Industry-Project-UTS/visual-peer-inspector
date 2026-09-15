"""Repairable aggregate plots and original-coordinate mapping evidence."""
from copy import deepcopy
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from vpi_object_bank.reporting import gallery, verified_report_inputs
from vpi_object_bank.storage import atomic_json, bind, digest_file, read_json, seal, source_hashes, terminal_status, verify_seal

LABELS = {
    'existing_all_candidates': 'Existing map + bank scorer',
    'global_partial_assignment': 'Joint map + bank scorer',
    'winner_cluster_consensus': 'Previous winner consensus',
    'correspondence_assisted_reference': 'Supplied identities + bank scorer (assisted)',
}


def build_report(cfg, source_root):
    root = Path(cfg['report_root'])
    hashes = verified_report_inputs(cfg)
    binding = {'schema': 'vpi-mapping-report/v1', 'input_locks': hashes,
               'source': source_hashes(source_root, ['src/vpi_object_mapping/reporting.py',
                   'src/vpi_object_bank/reporting.py', 'src/vpi_object_bank/storage.py',
                   'src/visual_peer_inspector/representation/crops.py',
                   'src/visual_peer_inspector/models/sam2.py',
                   'src/visual_peer_inspector/workflows/comparison_io.py'])}
    bind(root, binding)
    if (root / 'report_lock.json').exists():
        verify_seal(root, name='report_lock.json')
        return read_json(root / 'report_receipt.json')
    with terminal_status(root, '08P-report'):
        figures = root / 'shareable/figures'
        figures.mkdir(parents=True, exist_ok=True)
        metrics = pd.read_csv(Path(cfg['evaluation_root']) / 'shareable/metric_summary.csv')
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for method, label in LABELS.items():
            subset = metrics[(metrics.method == method) & (metrics.metric == 'top1_tie_expected')].sort_values('view_count')
            ax.plot(subset.view_count, subset['mean'] * 100, marker='o', label=label)
        ax.set(xlabel='Uploaded views', ylabel='Tie-aware scene top-1 (%)', xticks=[1, 2, 3, 5], ylim=(0, 100),
               title='100 development scenes / 27 assets\nEach scene averaged across predefined schedules')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=.2)
        fig.savefig(figures / 'view_count_accuracy.png', dpi=150)
        plt.close(fig)
        paired = pd.read_csv(Path(cfg['evaluation_root']) / 'shareable/paired_asset_intervals.csv')
        subset = paired[(paired.view_set_id == 'three-primary') & (paired.metric == 'top1_tie_expected')]
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for i, row in enumerate(subset.itertuples()):
            ax.errorbar(row.delta*100, i, xerr=[[max(0, row.delta-row.ci_low)*100],
                                             [max(0, row.ci_high-row.delta)*100]], fmt='o', capsize=5)
        ax.set(yticks=range(len(subset)), yticklabels=[LABELS[x] for x in subset.baseline],
               xlabel='Joint-map improvement in percentage points (95% asset bootstrap)',
               title='Primary three-view schedule: paired improvement')
        ax.axvline(0, color='black', linewidth=1)
        fig.savefig(figures / 'paired_improvement.png', dpi=150)
        plt.close(fig)
        scene_metrics = pd.read_csv(Path(cfg['evaluation_root']) / 'private/scene_metrics.csv')
        current = scene_metrics[(scene_metrics.view_set_id == 'three-primary') &
                                (scene_metrics.method == 'global_partial_assignment')].sort_values(['top1_tie_expected', 'scene'])
        manifest = read_json(Path(cfg['prepared_root']) / 'input_manifest.json')
        choices = {manifest['scenes'][0]: 'fixed first scene; outcome independent'}
        choices[str(current.iloc[0].scene)] = 'post-hoc low-accuracy illustration'
        choices[str(current.iloc[-1].scene)] = 'post-hoc high-accuracy illustration'
        receipts = []
        for scene, reason in choices.items():
            source = Path(cfg['prediction_root']) / 'scenes' / scene
            payload = read_json(source / 'predictions.json')
            setting = deepcopy(next(s for s in payload['settings'] if s['view_set_id'] == 'three-primary'))
            setting['rankings'] = {'object_bank_top2': setting['rankings']['global_partial_assignment']}
            # A presentation alias lets the previously coordinate-tested renderer
            # use the unchanged bank scorer's patch fields. Scientific files stay sealed.
            rendering = root / 'private/rendering-inputs'
            unit = rendering / 'scenes' / scene
            unit.mkdir(parents=True, exist_ok=True)
            atomic_json(unit / 'predictions.json', {**payload, 'settings': [setting]})
            shutil.copyfile(source / 'patch_evidence.npz', unit / 'patch_evidence.npz')
            path = root / 'private' / f'{scene}-joint-mapping.png'
            receipt = gallery({**cfg, 'prediction_root': str(rendering)}, scene,
                              'automatic_predicted', 'three-primary', 'object_bank_top2', path)
            receipt.update(selection_reason=reason, mapping_method='global_partial_assignment',
                           mapping_status=setting['mapping']['status'], uncertainty=setting['uncertainty'])
            receipts.append(receipt)
        atomic_json(root / 'private/gallery_manifest.json', receipts)
        # Aggregate solver diagnostics are independent of evaluator labels.
        statuses, alternatives, seconds = {}, {}, []
        for scene in manifest['scenes']:
            payload = read_json(Path(cfg['prediction_root']) / 'scenes' / scene / 'predictions.json')
            for setting in payload['settings']:
                status = setting['mapping']['status']
                statuses[status] = statuses.get(status, 0) + 1
                n = len(setting['mapping']['hypotheses'])
                alternatives[str(n)] = alternatives.get(str(n), 0) + 1
                seconds.append(setting['elapsed_seconds'])
        receipt = {'status': 'completed', 'gallery_count': len(receipts), 'solver_status_counts': statuses,
                   'available_hypothesis_counts': alternatives, 'setting_count': len(seconds),
                   'matching_and_scoring_seconds_sum': sum(seconds),
                   'maximum_setting_seconds': max(seconds),
                   'figures': ['view_count_accuracy.png', 'paired_improvement.png'],
                   'gallery_boxes': 'saved mask-bounding boxes; not validated defect boundaries',
                   'uncertainty': 'up to three appearance maps, not calibrated probabilities'}
        atomic_json(root / 'report_receipt.json', receipt)
        seal(root, [p for p in root.rglob('*') if p.is_file() and p.name not in ('report_lock.json', 'run_status.json')], name='report_lock.json')
    return receipt
