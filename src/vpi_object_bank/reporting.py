"""Repairable figures and private multi-view evidence from sealed predictions."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask
from visual_peer_inspector.representation.crops import prepare_masked_crop, project_crop_heatmap
from .storage import atomic_json, bind, digest_file, read_json, seal, source_hashes, terminal_status, verify_seal


def gallery(cfg, scene, route, view_set_id, method, output):
    model_root, prediction_root, feature_root = (Path(cfg[k]) for k in ('model_root', 'prediction_root', 'feature_root'))
    payload = read_json(prediction_root / 'scenes' / scene / 'predictions.json')
    setting = next(s for s in payload['settings'] if s['route'] == route and s['view_set_id'] == view_set_id)
    observations = {(o['view_id'], o['candidate_id']): o for o in setting['observations']}
    objects = setting['rankings'][method]
    ranked = sorted([o for o in objects if o['score'] is not None], key=lambda o: (-o['score'], o['id']))
    top = ranked[0]['score'] if ranked else None
    tied = [o for o in ranked if np.isclose(o['score'], top, atol=1e-12, rtol=0)]
    selected = tied[0] if len(tied) == 1 else None
    name_map = {obj['id']: f'O{index+1}' for index, obj in enumerate(sorted(objects, key=lambda obj: obj['id']))}
    member_map = {(m['view_id'], m['candidate_id']): obj for obj in objects for m in obj['members']}
    model_rows = {(r['route'], r['view']): r for r in read_json(model_root / 'scenes' / f'{scene}.json')}
    views = setting['views']
    fig = plt.figure(figsize=(5 * len(views), 16), constrained_layout=True)
    gs = fig.add_gridspec(5, len(views), height_ratios=[1, 1, 1, 1, .75])
    evidence = np.load(prediction_root / 'scenes' / scene / 'patch_evidence.npz', allow_pickle=False)
    scorer = method.rsplit('_', 1)[0] if method != 'winner_cluster_consensus' else 'cached_forward'
    rank_lines, unlocated = [], []
    palette = plt.get_cmap('tab10')
    for col, view in enumerate(views):
        model_row = model_rows[(setting['region'], view)]
        rgb = np.asarray(Image.open(model_root / model_row['rgb']).convert('RGB'))
        source_objects = {o['candidate_id']: o for o in model_row['objects']}
        axes = [fig.add_subplot(gs[row, col]) for row in range(4)]
        for ax in axes:
            ax.set_axis_off()
        axes[0].imshow(rgb)
        axes[0].set_title(f'{view}: uploaded image')
        overlay = rgb.astype(float).copy()
        black = np.zeros_like(rgb)
        selected_candidate = next((m['candidate_id'] for m in selected['members'] if m['view_id'] == view), None) if selected else None
        for candidate, obj in source_objects.items():
            mask = decode_binary_mask(obj['mask'])
            black[mask] = rgb[mask]
            mapped = member_map.get((view, candidate))
            if mapped:
                idx = int(name_map[mapped['id']][1:]) - 1
                colour = np.asarray(palette(idx % 10)[:3]) * 255
                overlay[mask] = .65 * overlay[mask] + .35 * colour
        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title('Object IDs, boxes and masks')
        axes[2].imshow(black)
        axes[2].set_title('Masked objects on black')
        for candidate, obj in source_objects.items():
            mapped = member_map.get((view, candidate))
            if not mapped:
                continue
            label = name_map[mapped['id']]
            x1, y1, x2, y2 = obj['box']
            chosen = candidate == selected_candidate
            colour = '#e00000' if chosen else palette((int(label[1:])-1) % 10)
            axes[1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=3 if chosen else 1.4, edgecolor=colour))
            axes[1].text(x1, max(0, y1-3), label + (' SELECTED' if chosen else ''), color='white', fontsize=10,
                         bbox={'facecolor': colour, 'alpha': .9, 'pad': 2, 'edgecolor': 'none'})
        axes[3].imshow(rgb)
        if selected_candidate:
            obj = source_objects[selected_candidate]
            mask = decode_binary_mask(obj['mask'])
            observed = observations.get((view, selected_candidate))
            info = observed.get(scorer, {}) if observed else {}
            key = info.get('patch_array_key')
            if key:
                unit = feature_root / 'views' / scene / f"{setting['region']}-{view}"
                metadata = read_json(unit / 'observations.json')
                stored = next(o for o in metadata['objects'] if o['candidate_id'] == selected_candidate)
                with np.load(unit / 'features.npz', allow_pickle=False) as features:
                    valid = features[f"v{stored['array_index']}"].astype(bool)
                heatmap = np.full(valid.shape, np.nan, dtype=np.float32)
                heatmap[valid] = evidence[key]
                crop = prepare_masked_crop(rgb, mask)
                if list(crop.padded_box_xyxy) != stored['padded_box_xyxy'] or crop.sha256 != stored['crop_sha256']:
                    raise ValueError('Gallery crop no longer matches extracted patch coordinates')
                projected = project_crop_heatmap(heatmap, crop, mask)
                axes[3].imshow(np.ma.masked_invalid(projected), cmap='magma', alpha=.65)
            x1, y1, x2, y2 = obj['box']
            axes[3].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=3))
            axes[3].set_title('Selected object; coarse patch evidence')
        else:
            reason = 'Tied ranking: no unique selection' if len(tied) > 1 else 'Selected object not located in this view'
            axes[3].set_title(reason)
            unlocated.append(view)
    evidence.close()
    table_ax = fig.add_subplot(gs[4, :])
    table_ax.set_axis_off()
    header = ['Object', *views, 'Combined rank score', 'Status']
    table_rows = []
    for obj in ranked + [o for o in objects if o['score'] is None]:
        values = []
        for view in views:
            candidate = next((m['candidate_id'] for m in obj['members'] if m['view_id'] == view), None)
            obs = observations.get((view, candidate)) if candidate else None
            score = obs.get(scorer, {}).get('score') if obs else None
            values.append('—' if score is None else f'{score:.4f}')
        table_rows.append([name_map[obj['id']], *values,
                           'unavailable' if obj['score'] is None else f"{obj['score']:.4f}",
                           obj.get('identity_status', 'assigned')])
    if table_rows:
        table = table_ax.table(cellText=table_rows, colLabels=header, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
    assistance = 'Supplied correspondence' if route.endswith('_reference') else 'Predicted correspondence'
    headline = f"Selected {name_map[selected['id']]}" if selected else 'No unique selection'
    fig.suptitle(f'{assistance} | {len(views)} views | {method}\n{headline}; per-view raw scores and one combined rank score', fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=115)
    plt.close(fig)
    return {'scene': scene, 'route': route, 'view_set_id': view_set_id, 'method': method,
            'selected_id': selected['id'] if selected else None, 'top_tie_count': len(tied),
            'unlocated_views': unlocated, 'file': output.name}



def aggregate_figures(evaluation_root, report_root):
    """Only aggregate values enter these shareable charts."""
    table = pd.read_csv(evaluation_root / 'shareable/view_schedule_summary.csv')
    pairs = pd.read_csv(evaluation_root / 'shareable/paired_asset_intervals.csv')
    figures = report_root / 'shareable/figures'
    figures.mkdir(parents=True, exist_ok=True)
    methods = ['within_view_mean', 'object_bank_mean', 'object_bank_top2', 'view_mean_control_mean', 'winner_cluster_consensus']
    titles = {'supplied_reference': 'Supplied masks + supplied IDs',
              'automatic_reference': 'Predicted masks + supplied IDs',
              'automatic_predicted': 'Predicted masks + predicted IDs'}
    labels = {'within_view_mean': 'Forward mean', 'object_bank_mean': 'Patch bank mean',
              'object_bank_top2': 'Patch bank strongest two', 'view_mean_control_mean': 'Peer-view mean control',
              'winner_cluster_consensus': 'Previous winner consensus'}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True, constrained_layout=True)
    primary = table[(table['metric'] == 'top1_tie_expected') & table['view_set_id'].str.endswith('primary')]
    for ax, (route, title) in zip(axes, titles.items(), strict=True):
        for method in methods:
            rows = primary[(primary['route'] == route) & (primary['method'] == method)].sort_values('view_count')
            if rows.empty:
                continue
            ax.plot(rows['view_count'], 100*rows['mean'], marker='o', label=labels[method])
            ax.fill_between(rows['view_count'], 100*rows['ci_low'], 100*rows['ci_high'], alpha=.08)
        ax.set(title=title, xlabel='Uploaded views', ylim=(0,100), xticks=[1,2,3,5])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Tie-aware scene top-1 (%)')
    handles, names = axes[-1].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Primary schedules: 100 development scenes; shaded asset-group 95% intervals')
    fig.savefig(figures/'view_count_accuracy.png', dpi=150)
    plt.close(fig)
    candidates = pairs[(pairs['metric'] == 'top1_tie_expected') & (pairs['view_set_id'] == 'three-primary') & pairs['method'].isin(['object_bank_mean','object_bank_top2','view_mean_control_mean'])]
    fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)
    names, values, lower, upper = [], [], [], []
    for _, row in candidates.iterrows():
        names.append(titles[row['route']] + ' / ' + labels[row['method']])
        values.append(row['delta']*100)
        lower.append(max(0, (row['delta']-row['ci_low'])*100))
        upper.append(max(0, (row['ci_high']-row['delta'])*100))
    ax.errorbar(values, np.arange(len(values)), xerr=[lower,upper], fmt='o', capsize=4)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(np.arange(len(values)), names, fontsize=9)
    ax.set_xlabel('Top-1 change versus regenerated forward mean (percentage points)')
    ax.set_title('Three uploaded views: paired asset-group 95% intervals')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(figures/'paired_improvement.png', dpi=150)
    plt.close(fig)
    return ['view_count_accuracy.png', 'paired_improvement.png']



def verified_report_inputs(cfg):
    roots = {key: Path(cfg[key]) for key in ('evaluation_root','prediction_root','feature_root','prepared_root','model_root')}
    locks = {'evaluation_root':'evaluation_lock.json', 'prediction_root':'prediction_lock.json',
             'feature_root':'feature_lock.json', 'prepared_root':'artifact_lock.json', 'model_root':'artifact_lock.json'}
    hashes = {}
    for key, name in locks.items():
        verified = verify_seal(roots[key], name=name)
        required = 'input_manifest.json' if key == 'prepared_root' else 'run_binding.json'
        if key != 'model_root' and required not in verified['files']:
            raise ValueError('Report input seal omits its binding or manifest')
        hashes[key] = digest_file(roots[key]/name)
    evaluation = read_json(roots['evaluation_root']/'run_binding.json')
    prediction = read_json(roots['prediction_root']/'run_binding.json')
    features = read_json(roots['feature_root']/'run_binding.json')
    prepared = read_json(roots['prepared_root']/'input_manifest.json')
    comparisons = [
        (evaluation['prediction_lock_sha256'], hashes['prediction_root']),
        (evaluation['prepared_lock_sha256'], hashes['prepared_root']),
        (prediction['feature_lock_sha256'], hashes['feature_root']),
        (prediction['prepared_lock_sha256'], hashes['prepared_root']),
        (features['prepared_lock_sha256'], hashes['prepared_root']),
        (features['model_input_lock_sha256'], hashes['model_root']),
        (prepared['model_input_lock_sha256'], hashes['model_root']),
    ]
    if any(left != right for left,right in comparisons):
        raise ValueError('Report inputs do not share the sealed scientific provenance')
    return hashes


def build_report(cfg, source_root):
    evaluation_root, prediction_root, report_root = (Path(cfg[k]) for k in ('evaluation_root', 'prediction_root', 'report_root'))
    input_hashes = verified_report_inputs(cfg)
    binding = {'schema': 'vpi-object-bank-report/v1', 'input_locks': input_hashes,
               'evaluation_lock_sha256': digest_file(evaluation_root / 'evaluation_lock.json'),
               'prediction_lock_sha256': digest_file(prediction_root / 'prediction_lock.json'),
               'source': source_hashes(source_root, ['src/vpi_object_bank/reporting.py', 'src/vpi_object_bank/storage.py',
                   'src/visual_peer_inspector/representation/crops.py', 'src/visual_peer_inspector/models/sam2.py',
                   'src/visual_peer_inspector/workflows/comparison_io.py'])}
    bind(report_root, binding)
    if (report_root / 'report_lock.json').exists():
        verify_seal(report_root, name='report_lock.json')
        return read_json(report_root / 'report_receipt.json')
    with terminal_status(report_root, '08O-report'):
        figure_names = aggregate_figures(evaluation_root, report_root)
        manifest = read_json(Path(cfg['prepared_root']) / 'input_manifest.json')
        # Fixed first scene is selected without outcomes. Additional illustrations are explicitly post-hoc.
        choices = [(manifest['scenes'][0], 'fixed first development scene')]
        metrics_path = evaluation_root / 'private' / 'scene_metrics.csv'
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            subset = metrics[(metrics['route'] == 'automatic_reference') & (metrics['view_set_id'] == 'three-primary')]
            bank = subset[subset['method'] == 'object_bank_mean'].set_index('scene')['top1_tie_expected']
            anchor = subset[subset['method'] == 'within_view_mean'].set_index('scene')['top1_tie_expected']
            delta = (bank-anchor).sort_values(kind='stable')
            if len(delta):
                choices.extend([(str(delta.index[0]), 'largest development loss, post-hoc illustration'),
                                (str(delta.index[-1]), 'largest development gain, post-hoc illustration')])
        receipts = []
        for scene, reason in dict(choices).items():
            for route in ('supplied_reference', 'automatic_reference', 'automatic_predicted'):
                for method in ('within_view_mean', 'object_bank_mean'):
                    path = report_root / 'private' / f'{scene}-{route}-{method}.png'
                    receipt = gallery(cfg, scene, route, 'three-primary', method, path)
                    receipt['selection_reason'] = reason
                    receipts.append(receipt)
        atomic_json(report_root / 'private/gallery_manifest.json', receipts)
        receipt = {'status': 'completed', 'gallery_count': len(receipts), 'aggregate_figures': figure_names,
                   'source': 'sealed development predictions', 'heatmap_scope': 'coarse patch evidence, not validated defect masks'}
        atomic_json(report_root / 'report_receipt.json', receipt)
        paths = [p for p in report_root.rglob('*') if p.is_file() and p.name not in ('report_lock.json', 'run_status.json')]
        seal(report_root, paths, name='report_lock.json', metadata=receipt)
    return receipt
