"""Versioned adapters to the preserved, selected scientific decision functions."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from vpi_object_bank.scoring import score_scene
from vpi_object_bank.prediction import fuse_observations
from visual_peer_inspector.workflows.learned_peer_features import _winner_tracks, _score_columns, _identity_tracks
from visual_peer_inspector.scoring.peer import p1_scores
from visual_peer_inspector.scoring.forward_patch import forward_patch_scores
from visual_peer_inspector.scoring.peer_support import mutual_missing_scores, geometry_peer_scores, rank_consensus

COMPONENTS = ('forward_02_cov70', 'P1_current', 'P2_consensus')


def component_scores(candidates):
    """Reproduce08G forward/P1/P2 from the same full patch grids and geometry."""
    by_view = defaultdict(list)
    for candidate in candidates:
        by_view[candidate['view_id']].append(candidate)
    rows, maps = {}, {}
    for view, values in by_view.items():
        values = sorted(values, key=lambda c: c['candidate_id'])
        usable = [c for c in values if c['embedding'] is not None and len(c['patches'])]
        scores = {c['candidate_id']: {name: None for name in (*COMPONENTS, 'mutual_missing_05', 'geometry_peer')} for c in values}
        if len(usable) >= 3:
            patches = [c['patches'] for c in usable]
            p1 = p1_scores(patches, [c['valid50'] for c in usable])
            forward = forward_patch_scores(patches, [c['valid70'] for c in usable], top_fraction=.02)
            mutual = mutual_missing_scores(patches, [c['valid50'] for c in usable], tail_fraction=.05)
            geometry = geometry_peer_scores(np.stack([c['geometry'] for c in usable]))
            p2 = rank_consensus(forward.scores, mutual.scores, geometry.scores)
            for index, c in enumerate(usable):
                scores[c['candidate_id']] = dict(zip((*COMPONENTS, 'mutual_missing_05', 'geometry_peer'),
                    map(float, (forward.scores[index], p1.scores[index], p2[index], mutual.scores[index], geometry.scores[index])), strict=True))
                maps[(view, c['candidate_id'])] = forward.heatmaps[index]
        rows[view] = {'objects': [{'candidate_id': c['candidate_id'], 'embedding': c['embedding'],
                                   'scores': scores[c['candidate_id']]} for c in values]}
    return rows, maps


def decide_cached(score_rows, bank_observations, *, views, mode):
    """Apply only selected-view inputs. Assisted IDs identify peers, never answers."""
    if mode not in ('automatic', 'assisted'):
        raise ValueError('Unknown decision mode')
    if set(score_rows) != set(views) or any(o['view_id'] not in views for o in bank_observations):
        raise ValueError('Decision inputs must contain exactly the selected views')
    component_rank = {}
    for view in views:
        objects = score_rows[view]['objects']
        ranks = [_score_columns([o['scores'].get(name) for o in objects])[3] for name in COMPONENTS]
        for index, obj in enumerate(objects):
            values = [r[index] for r in ranks if r[index] is not None]
            component_rank[(view, obj['candidate_id'])] = None if not values else float(np.mean(values))
    suggestions = []
    valid_rows = {v: {'objects': [o for o in score_rows[v]['objects'] if o['embedding'] is not None]} for v in views}
    for track in _identity_tracks(valid_rows, tuple(views)):
        suggestions.extend({**m, 'identity_id': track['identity_id'], 'status': track['status']} for m in track['members'])
    if mode == 'automatic':
        tracks = _winner_tracks(score_rows, tuple(views), selector='consensus', distance_threshold=.20)
        ranking = [{'id': t['identity_id'], 'score': t['cluster_score'], 'members': t['members'],
                    'support': len(t['members']), 'fusion_observations': len(t['members']),
                    'identity_status': t['status']} for t in tracks]
        identities = {(m['view_id'], m['candidate_id']): t['identity_id'] for t in tracks for m in t['members']}
        observations = None
    else:
        result = score_scene(bank_observations, selected_views=views, top_fraction=.02)
        observations = result['observations']
        ranking, normalized = fuse_observations(observations, method='object_bank', fusion='top2')
        component_rank = {(o['view_id'], o['candidate_id']): normalized.get(f"{o['view_id']}/{o['candidate_id']}") for o in observations}
        identities = {(m['view_id'], m['candidate_id']): t['id'] for t in ranking for m in t['members']}
        tracks = [{'identity_id': t['id'], 'members': t['members'], 'status': t['identity_status']} for t in ranking]
    return {'ranking': ranking, 'tracks': tracks, 'identities': identities,
            'component_rank': component_rank, 'scored_observations': observations, 'suggested_identities': suggestions}
