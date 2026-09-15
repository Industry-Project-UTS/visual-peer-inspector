"""Run one frozen global mapping comparison; predictions precede labels."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import time

import numpy as np
import scipy

from vpi_object_bank.extraction import load_observations
from vpi_object_bank.prediction import fuse_observations
from vpi_object_bank.scoring import score_scene
from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest_file, progress,
    read_json, seal, source_hashes, terminal_status, verified_unit, verify_seal)
from .matching import match_scene
from .protocol import PROTOCOL

SOURCES = ['src/vpi_object_mapping/' + name for name in
           ('prediction.py', 'matching.py', 'protocol.py')]
SOURCES += ['src/vpi_object_bank/' + name for name in
            ('extraction.py', 'scoring.py', 'prediction.py', 'protocol.py', 'storage.py')]
SOURCES += ['src/visual_peer_inspector/workflows/comparison_io.py']


def rank_map(observations, tracks, views):
    expected = [(o['view_id'], o['candidate_id']) for o in observations]
    members = [(m['view_id'], m['candidate_id']) for t in tracks for m in t['members']]
    track_ids = [t['identity_id'] for t in tracks]
    if (len(expected) != len(set(expected)) or len(members) != len(set(members))
            or set(members) != set(expected) or len(track_ids) != len(set(track_ids))
            or any(not t['members'] or len(t['members']) != len({m['view_id'] for m in t['members']}) for t in tracks)):
        raise ValueError('The map must cover each observation once with unique tracks and at most one member per view')
    identities = {(m['view_id'], m['candidate_id']): t['identity_id']
                  for t in tracks for m in t['members']}
    if len(identities) != len(observations):
        raise ValueError('The map must retain every observation exactly once')
    rows = [{**o, 'object_id': identities[(o['view_id'], o['candidate_id'])]}
            for o in observations]
    result = score_scene(rows, selected_views=views, top_fraction=.02)
    ranking, ranks = fuse_observations(result['observations'], method='object_bank', fusion='top2')
    statuses = {t['identity_id']: t['status'] for t in tracks}
    for obj in ranking:
        obj['mapping_status'] = statuses[obj['id']]
    return result, ranking, ranks


def selected_members(ranking):
    usable = [o for o in ranking if o['score'] is not None]
    if not usable:
        return None
    top = max(o['score'] for o in usable)
    winners = [o for o in usable if np.isclose(o['score'], top, atol=1e-12, rtol=0)]
    return {(m['view_id'], m['candidate_id']) for m in winners[0]['members']} if len(winners) == 1 else None


def uncertainty(primary, alternative_rankings):
    selected = selected_members(primary)
    if selected is None:
        return {'status': 'no_unique_primary_selection', 'selected_members': [],
                'alternative_count': len(alternative_rankings), 'selection_overlap': []}
    overlaps = []
    for ranking in alternative_rankings:
        other = selected_members(ranking)
        overlaps.append(None if other is None else len(selected & other) / len(selected | other))
    return {'status': 'hypothesis_sensitivity_only',
            'selected_members': [dict(view_id=v, candidate_id=c) for v, c in sorted(selected)],
            'alternative_count': len(alternative_rankings), 'selection_overlap': overlaps,
            'all_available_maps_same_selected_members': all(x == 1 for x in overlaps) if overlaps else None,
            'interpretation': 'overlap among at most three maps is not a calibrated confidence'}


def predict_scene(scene, cfg):
    previous = read_json(Path(cfg['previous_prediction_root']) / 'scenes' / scene / 'predictions.json')
    old = {(s['route'], s['view_set_id']): s for s in previous['settings']}
    raw = read_json(Path(cfg['scoring_root']) / 'scenes' / f'{scene}.json')
    automatic = {r['view']: r for r in raw if r['route'] == 'automatic'}
    settings, arrays = [], {}
    for view_set_id, views in PROTOCOL['view_sets'].items():
        started = time.monotonic()
        # Neither neutral reference IDs nor unused view tensors enter the matcher.
        matching_input = [{'view_id': v, 'candidate_id': o['candidate_id'], 'embedding': o['embedding']}
                          for v in views for o in automatic[v]['objects']]
        mcfg = PROTOCOL['matching']
        mapping = match_scene(matching_input, selected_views=views,
                              time_limit_seconds=mcfg['time_limit_seconds'],
                              max_hypotheses=mcfg['max_hypotheses'], max_tracks=mcfg['max_tracks'])
        no_identity = {(o['view_id'], o['candidate_id']): None for o in matching_input}
        observations, _ = load_observations(cfg['feature_root'], scene, 'automatic', views, no_identity)
        result, ranked, per_view = rank_map(observations, mapping['primary']['tracks'], views)
        for i, obs in enumerate(result['observations']):
            for method in ('within_view', 'object_bank', 'view_mean_control'):
                patches = obs[method].pop('patch_scores', [])
                if patches and method == 'object_bank':
                    key = f's{len(settings)}o{i}'
                    arrays[key] = np.asarray(patches, dtype=np.float32)
                    obs[method]['patch_array_key'] = key
        alternate_rankings = []
        for hypothesis in mapping['hypotheses'][1:]:
            _, ranking, _ = rank_map(observations, hypothesis['tracks'], views)
            hypothesis['ranking'] = ranking
            alternate_rankings.append(ranking)
        baseline = old[('automatic_predicted', view_set_id)]
        assisted = old[('automatic_reference', view_set_id)]
        settings.append({
            'route': 'automatic_predicted', 'region': 'automatic', 'view_set_id': view_set_id,
            'views': list(views), 'view_count': len(views), 'observations': result['observations'],
            'counts': result['counts'], 'per_view_ranks': {'object_bank': per_view},
            'rankings': {
                'existing_all_candidates': deepcopy(baseline['rankings']['object_bank_top2']),
                'global_partial_assignment': ranked,
                'winner_cluster_consensus': deepcopy(baseline['rankings']['winner_cluster_consensus']),
                'correspondence_assisted_reference': deepcopy(assisted['rankings']['object_bank_top2']),
            },
            'mapping': mapping, 'uncertainty': uncertainty(ranked, alternate_rankings),
            'elapsed_seconds': time.monotonic() - started,
        })
    return {'schema': PROTOCOL['schema'], 'scene': scene, 'settings': settings}, arrays


def preflight(cfg, source_root):
    roots = {k: Path(cfg[k]) for k in ('prepared_root', 'feature_root', 'scoring_root', 'previous_prediction_root')}
    names = {'prepared_root': 'artifact_lock.json', 'feature_root': 'feature_lock.json',
             'scoring_root': 'prediction_lock.json', 'previous_prediction_root': 'prediction_lock.json'}
    for key, root in roots.items():
        verify_seal(root, name=names[key])
    manifest = read_json(roots['prepared_root'] / 'input_manifest.json')
    if manifest['view_sets'] != PROTOCOL['view_sets'] or len(manifest['scenes']) != 100:
        raise ValueError('The frozen cohort/schedules changed')
    old_binding = read_json(roots['previous_prediction_root'] / 'run_binding.json')
    expected = {'prepared_lock_sha256': digest_file(roots['prepared_root'] / names['prepared_root']),
                'feature_lock_sha256': digest_file(roots['feature_root'] / names['feature_root']),
                'scoring_lock_sha256': digest_file(roots['scoring_root'] / names['scoring_root'])}
    for key, value in expected.items():
        if old_binding[key] != value:
            raise ValueError('Old results and current cached resources differ')
    # Prevent a new mapping notebook from silently changing the selected scorer.
    hashes = source_hashes(source_root, SOURCES)
    for key, value in old_binding['source'].items():
        if key in hashes and hashes[key] != value:
            raise ValueError('Frozen scorer or feature code changed')
    binding = {'schema': PROTOCOL['schema'], 'protocol': PROTOCOL, **expected,
               'previous_prediction_lock_sha256': digest_file(roots['previous_prediction_root'] / names['previous_prediction_root']),
               'source': hashes, 'runtime': {'numpy': np.__version__, 'scipy': scipy.__version__}}
    return manifest['scenes'], binding


def predict(cfg, source_root, *, max_scenes=None):
    scenes, binding = preflight(cfg, source_root)
    root = Path(cfg['prediction_root'])
    bound = bind(root, binding)
    if (root / 'prediction_lock.json').exists():
        verify_seal(root, name='prediction_lock.json')
        if not all(verified_unit(root / 'scenes' / s, bound) for s in scenes):
            raise ValueError('Completed prediction is missing a scene')
        return {'computed': 0, 'reused': len(scenes), 'sealed': True}
    start, computed, reused = time.monotonic(), 0, 0
    with terminal_status(root, '08P-mapping-predictions'):
        for scene in scenes:
            unit = root / 'scenes' / scene
            if verified_unit(unit, bound):
                reused += 1
                continue
            if max_scenes is not None and computed >= max_scenes:
                break
            payload, arrays = predict_scene(scene, cfg)
            with atomic_unit(unit, bound) as temporary:
                atomic_json(temporary / 'predictions.json', payload)
                with (temporary / 'patch_evidence.npz').open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
            computed += 1
            progress(root, completed=computed+reused, total=len(scenes), elapsed_seconds=time.monotonic()-start)
        completed = computed + reused == len(scenes)
        if completed:
            files = [root / 'run_binding.json']
            files += [p for s in scenes for p in (root / 'scenes' / s).iterdir() if p.is_file()]
            seal(root, files, name='prediction_lock.json', metadata={'binding_sha256': bound, 'scenes': scenes})
    if not completed:
        atomic_json(root / 'run_status.json', {'status': 'pilot_complete', 'completed_scenes': computed+reused})
    return {'computed': computed, 'reused': reused, 'sealed': completed, 'elapsed_seconds': time.monotonic()-start}
