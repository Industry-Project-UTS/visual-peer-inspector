"""Produce physical-object rankings before the evaluator is available."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np
from scipy.stats import rankdata
from visual_peer_inspector.workflows.learned_peer_features import _identity_tracks, _winner_tracks
from .extraction import load_observations
from .protocol import PROTOCOL
from .scoring import score_scene
from .storage import (atomic_json, atomic_unit, bind, digest_file, progress, read_json,
                      seal, source_hashes, terminal_status, verified_unit, verify_seal)

PREDICTION_SOURCES = [
    'src/vpi_object_bank/prediction.py', 'src/vpi_object_bank/scoring.py',
    'src/visual_peer_inspector/workflows/comparison_io.py',
    'src/vpi_object_bank/protocol.py', 'src/vpi_object_bank/storage.py',
    'src/vpi_object_bank/extraction.py',
    'src/visual_peer_inspector/workflows/learned_peer_features.py',
    'src/visual_peer_inspector/representation/cross_view_identity.py',
]


def fuse_observations(observations, *, method, fusion):
    if fusion not in ('mean', 'top2'):
        raise ValueError('Unknown fusion')
    rank_scores = {}
    by_view = defaultdict(list)
    for row in observations:
        by_view[row['view_id']].append(row)
    for view, rows in by_view.items():
        available = [row for row in rows if row[method]['score'] is not None]
        if len(available) < 3:
            continue
        values = np.asarray([row[method]['score'] for row in available])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite scored observation')
        ranks = (rankdata(values, method='average') - 1) / (len(values) - 1)
        rank_scores.update({(view, row['candidate_id']): float(value) for row, value in zip(available, ranks, strict=True)})
    groups = defaultdict(list)
    for row in observations:
        identity = row['object_id'] or f"unmatched-{row['view_id']}-{row['candidate_id']}"
        groups[identity].append(row)
    objects = []
    for identity, members in sorted(groups.items()):
        usable = [rank_scores[(m['view_id'], m['candidate_id'])] for m in members
                  if (m['view_id'], m['candidate_id']) in rank_scores]
        selected = sorted(usable, reverse=True)[:2] if fusion == 'top2' else usable
        objects.append({'id': identity, 'score': float(np.mean(selected)) if selected else None,
                        'support': len(usable), 'fusion_observations': len(selected),
                        'identity_status': 'unknown' if members[0]['object_id'] is None else 'assigned',
                        'members': [{'view_id': m['view_id'], 'candidate_id': m['candidate_id']} for m in members]})
    return objects, {f'{v}/{c}': value for (v, c), value in rank_scores.items()}


def predict_scene(scene, prepared_root, feature_root, scoring_root):
    prepared = read_json(Path(prepared_root) / 'scenes' / f'{scene}.json')
    correspondence = {(r['route'], r['view'], r['candidate_id']): r['neutral_id'] for r in prepared['observations']}
    scoring = {(r['route'], r['view']): r for r in read_json(Path(scoring_root) / 'scenes' / f'{scene}.json')}
    settings, patch_arrays = [], {}
    # Loading all stored tensors is an I/O convenience; each scorer receives only its selected views.
    loaded = {}
    all_views = sorted({view for views in PROTOCOL['view_sets'].values() for view in views})
    for region in ('supplied', 'automatic'):
        neutral = {(v, o['candidate_id']): correspondence[(region, v, o['candidate_id'])]
                   for v in all_views for o in scoring[(region, v)]['objects']}
        loaded[region], _ = load_observations(feature_root, scene, region, all_views, neutral)
    for route in PROTOCOL['routes']:
        region = 'supplied' if route == 'supplied_reference' else 'automatic'
        rows = {view: scoring[(region, view)] for view in all_views}
        for view_set_id, views in PROTOCOL['view_sets'].items():
            started = time.monotonic()
            observations = [dict(row) for row in loaded[region] if row['view_id'] in views]
            if route == 'automatic_predicted':
                tracks = _identity_tracks(rows, tuple(views))
                predicted = {(m['view_id'], m['candidate_id']): t['identity_id'] for t in tracks for m in t['members']}
                for row in observations:
                    row['object_id'] = predicted[(row['view_id'], row['candidate_id'])]
            result = score_scene(observations, selected_views=views, top_fraction=.02)
            source_scores = {(v, obj['candidate_id']): obj for v in views for obj in rows[v]['objects']}
            raw_rankings = {}
            per_view_ranks = {}
            for row_index, row in enumerate(result['observations']):
                old = source_scores[(row['view_id'], row['candidate_id'])]['scores']['forward_02_cov70']
                row['cached_forward'] = {'score': old, 'status': 'scored' if old is not None else 'insufficient_peers'}
                for method in ('within_view', 'object_bank', 'view_mean_control'):
                    values = row[method].pop('patch_scores', [])
                    key = f's{len(settings)}o{row_index}m{method}'
                    if values:
                        patch_arrays[key] = np.asarray(values, dtype=np.float32)
                        row[method]['patch_array_key'] = key
            for method in PROTOCOL['scorers']:
                for fusion in PROTOCOL['fusion']:
                    name = f'{method}_{fusion}'
                    raw_rankings[name], per_view_ranks[method] = fuse_observations(result['observations'], method=method, fusion=fusion)
            if route == 'automatic_predicted':
                winner_tracks = _winner_tracks(rows, tuple(views), selector='consensus')
                raw_rankings['winner_cluster_consensus'] = [
                    {'id': t['identity_id'], 'score': t['cluster_score'], 'members': t['members'],
                     'support': len(t['members']), 'fusion_observations': len(t['members']),
                     'identity_status': t['status']} for t in winner_tracks]
            settings.append({'route': route, 'region': region, 'view_set_id': view_set_id,
                             'views': list(views), 'view_count': len(views),
                             'observations': result['observations'], 'counts': result['counts'],
                             'rankings': raw_rankings, 'per_view_ranks': per_view_ranks,
                             'elapsed_seconds': time.monotonic() - started})
    return {'schema': PROTOCOL['schema'], 'scene': scene, 'settings': settings}, patch_arrays


def predict(prepared_root, feature_root, scoring_root, output_root, source_root, *, max_scenes=None):
    prepared_root, feature_root, scoring_root, output_root = map(Path, (prepared_root, feature_root, scoring_root, output_root))
    verify_seal(prepared_root)
    feature_lock = verify_seal(feature_root, name='feature_lock.json')
    if feature_lock['metadata']['producer'] != 'dinov2-small':
        raise ValueError('Scientific predictions require real DINO features, not a contract encoder')
    verify_seal(scoring_root, name='prediction_lock.json')
    manifest = read_json(prepared_root / 'input_manifest.json')
    feature_binding = read_json(feature_root / 'run_binding.json')
    if feature_binding['prepared_lock_sha256'] != digest_file(prepared_root / 'artifact_lock.json'):
        raise ValueError('Feature cache belongs to a different registry/input manifest')
    if manifest.get('scoring_prediction_lock_sha256') != digest_file(scoring_root / 'prediction_lock.json'):
        raise ValueError('Cached scoring input does not match prepared inputs')
    binding = {'schema': PROTOCOL['schema'], 'protocol': PROTOCOL,
               'prepared_lock_sha256': digest_file(prepared_root / 'artifact_lock.json'),
               'feature_lock_sha256': digest_file(feature_root / 'feature_lock.json'),
               'scoring_lock_sha256': digest_file(scoring_root / 'prediction_lock.json'),
               'source': source_hashes(source_root, PREDICTION_SOURCES)}
    binding_hash = bind(output_root, binding)
    scenes = manifest['scenes']
    if (output_root / 'prediction_lock.json').exists():
        verify_seal(output_root, name='prediction_lock.json')
        for scene in scenes:
            if not verified_unit(output_root / 'scenes' / scene, binding_hash):
                raise ValueError('Sealed prediction scene is missing')
        return {'computed': 0, 'reused': len(scenes), 'scenes': len(scenes), 'elapsed_seconds': 0.0, 'sealed': True}
    start, computed, reused = time.monotonic(), 0, 0
    with terminal_status(output_root, '08N-model-facing-prediction'):
        for index, scene in enumerate(scenes):
            unit = output_root / 'scenes' / scene
            if verified_unit(unit, binding_hash):
                reused += 1
                continue
            if max_scenes is not None and computed >= max_scenes:
                break
            payload, arrays = predict_scene(scene, prepared_root, feature_root, scoring_root)
            with atomic_unit(unit, binding_hash) as temporary:
                atomic_json(temporary / 'predictions.json', payload)
                with (temporary / 'patch_evidence.npz').open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
            computed += 1
            progress(output_root, completed=computed+reused, total=len(scenes), computed=computed,
                     reused=reused, elapsed_seconds=time.monotonic()-start)
        if computed+reused == len(scenes):
            paths = [output_root / 'run_binding.json']
            paths += [p for p in (output_root / 'scenes').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p)]
            seal(output_root, paths, name='prediction_lock.json', metadata={'scenes': scenes,
                 'settings_per_scene': len(PROTOCOL['view_sets'])*len(PROTOCOL['routes']),
                 'binding_sha256': binding_hash})
    if computed+reused != len(scenes):
        atomic_json(output_root / 'run_status.json', {'stage': '08N-model-facing-prediction',
                    'status': 'pilot_complete', 'completed_scenes': computed+reused, 'total_scenes': len(scenes)})
    return {'computed': computed, 'reused': reused, 'scenes': len(scenes),
            'elapsed_seconds': time.monotonic()-start, 'sealed': (output_root/'prediction_lock.json').is_file()}
