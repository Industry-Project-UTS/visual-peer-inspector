"""Evaluate sealed object rankings without exposing evaluator identities to scorers."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from visual_peer_inspector.workflows.learned_peer_evaluation import _rank_metrics, _track_references
from .protocol import PROTOCOL
from .storage import (atomic_json, atomic_unit, bind, digest, digest_file, progress, read_json,
                      seal, source_hashes, terminal_status, verified_unit, verify_seal)

METRICS = tuple(PROTOCOL['metrics'])
SHAREABLE_FILES = (
    'metric_summary.csv', 'view_schedule_summary.csv', 'paired_asset_intervals.csv',
    'anomaly_count_summary.csv', 'majority_summary.csv', 'failure_summary.csv',
    'identity_summary.csv', 'decision.json', 'evaluation_receipt.json',
)
EVALUATION_SOURCES = (
    'src/vpi_object_bank/evaluation.py', 'src/vpi_object_bank/protocol.py',
    'src/vpi_object_bank/storage.py',
    'src/visual_peer_inspector/workflows/learned_peer_evaluation.py',
    'src/visual_peer_inspector/workflows/comparison_io.py',
)
FAILURES = (
    'candidate_coverage', 'candidate_anomaly_recall', 'detection_loss', 'pruning_loss',
    'association_loss', 'score_unavailability_loss', 'rankable', 'ranking_loss',
    'candidate_count', 'retained_candidate_count', 'unknown_candidate_count',
    'unscored_object_count', 'scored_object_count', 'all_objects_unscored',
)
IDENTITY_METRICS = (
    'identity_precision', 'identity_recall', 'identity_purity',
    'identity_fragmentation', 'identity_reference_coverage',
    'predicted_cross_view_pairs', 'correct_cross_view_pairs', 'reference_cross_view_pairs',
)
KNOWN_MODEL_LOCK = '9baa53454d2482caf8240e7a8b591a6e9e9646dd160c5b3abdc8bd17265ebda1'


def _key(member):
    return str(member['view_id']), str(member['candidate_id'])


def _methods(route):
    result = {f'{method}_{fusion}' for method in PROTOCOL['scorers'] for fusion in PROTOCOL['fusion']}
    if route == 'automatic_predicted':
        result.add('winner_cluster_consensus')
    return result


def validate_predictions(payload, scene, scoring_rows):
    """Check selected-view isolation and candidate conservation before labels open."""
    if payload.get('schema') != PROTOCOL['schema'] or payload.get('scene') != scene:
        raise ValueError('Prediction scene or schema mismatch')
    expected_settings = {(r, v) for r in PROTOCOL['routes'] for v in PROTOCOL['view_sets']}
    actual_settings = [(s['route'], s['view_set_id']) for s in payload['settings']]
    if len(actual_settings) != len(set(actual_settings)) or set(actual_settings) != expected_settings:
        raise ValueError('Prediction settings are missing, duplicated or unexpected')
    source = {(r['route'], r['view']): r for r in scoring_rows}
    for setting in payload['settings']:
        route = setting['route']
        region = 'supplied' if route == 'supplied_reference' else 'automatic'
        views = PROTOCOL['view_sets'][setting['view_set_id']]
        if setting['region'] != region or setting['views'] != list(views) or setting['view_count'] != len(views):
            raise ValueError('Selected-view or region contract mismatch')
        expected = {(v, o['candidate_id']) for v in views for o in source[(region, v)]['objects']}
        keys = [_key(row) for row in setting['observations']]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('Observation candidates are missing, duplicated or outside selected views')
        if set(setting['rankings']) != _methods(route):
            raise ValueError('Unexpected ranking method schema')
        for method, objects in setting['rankings'].items():
            ids, members = [], []
            for obj in objects:
                ids.append(obj['id'])
                these = [_key(m) for m in obj['members']]
                if not these or len({v for v, _ in these}) != len(these):
                    raise ValueError('Empty track or same-view duplicate in a physical-object track')
                if not set(these) <= expected:
                    raise ValueError('Track uses unknown or unselected observations')
                if obj['score'] is not None and not np.isfinite(float(obj['score'])):
                    raise ValueError('Nonfinite object score')
                members.extend(these)
            if len(ids) != len(set(ids)) or len(members) != len(set(members)):
                raise ValueError('Duplicate object ID or observation assignment')
            if method != 'winner_cluster_consensus' and set(members) != expected:
                raise ValueError('Retained-candidate method dropped observations')


def _identity_metrics(objects, correspondence, candidates):
    known = {key: correspondence[key] for key in candidates if correspondence[key] is not None}
    predicted_pairs = correct_pairs = 0
    tracks_per_reference = defaultdict(set)
    majority_support = 0
    for obj in objects:
        members = [_key(m) for m in obj['members']]
        votes = Counter(correspondence[key] for key in members if correspondence[key] is not None)
        majority_support += max(votes.values(), default=0)
        for ref in votes:
            tracks_per_reference[ref].add(obj['id'])
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if left[0] != right[0]:
                    predicted_pairs += 1
                    correct_pairs += left in known and right in known and known[left] == known[right]
    by_reference = defaultdict(list)
    for key, reference in known.items():
        by_reference[reference].append(key)
    reference_pairs = sum(sum(a[0] != b[0] for i, a in enumerate(items) for b in items[i + 1:])
                          for items in by_reference.values())
    member_count = sum(len(obj['members']) for obj in objects)
    return {
        # A zero-link setting has undefined precision/recall; retain it explicitly.
        'identity_precision': correct_pairs / predicted_pairs if predicted_pairs else None,
        'identity_recall': correct_pairs / reference_pairs if reference_pairs else None,
        'identity_purity': majority_support / member_count if member_count else None,
        'identity_fragmentation': float(np.mean([max(0, len(tracks_per_reference[r]) - 1) for r in by_reference])) if by_reference else None,
        'identity_reference_coverage': len(tracks_per_reference) / len(by_reference) if by_reference else None,
        'predicted_cross_view_pairs': predicted_pairs, 'correct_cross_view_pairs': correct_pairs,
        'reference_cross_view_pairs': reference_pairs,
    }


def evaluate_setting(scene, setting, truth):
    """Canonical tie-aware metrics plus an exhaustive loss decomposition."""
    labels = truth['labels']
    if not labels or any(v not in (0, 1) for v in labels.values()) or not any(labels.values()):
        raise ValueError('Evaluator requires binary labels and at least one anomaly')
    positives = {reference for reference, value in labels.items() if value == 1}
    observations = setting['observations']
    candidates = {_key(row) for row in observations}
    correspondence = {}
    for key in candidates:
        source_key = f"{setting['region']}/{key[0]}/{key[1]}"
        if source_key not in truth['correspondence']:
            raise ValueError('Missing evaluator correspondence differs from explicit unknown')
        reference = truth['correspondence'][source_key]
        if reference is not None and reference not in labels:
            raise ValueError('Evaluator correspondence names an absent reference')
        correspondence[key] = reference
    candidate_anomalies = {r for r in correspondence.values() if r in positives}
    rows = []
    for method, objects in sorted(setting['rankings'].items()):
        # Include unscored tracks in the one-to-one assignment; filtering them first
        # can change which competing track receives a reference identity.
        objects = sorted(objects, key=lambda obj: str(obj['id']))
        tracks = [{'identity_id': obj['id'], 'members': obj['members']} for obj in objects]
        references = _track_references(tracks, truth, setting['region'])
        items = [{'id': obj['id'], 'score': float(obj['score']), 'support': obj['support'],
                  'reference': references[obj['id']][0]}
                 for obj in objects if obj['score'] is not None]
        retained = {_key(m) for obj in objects for m in obj['members']}
        retained_anomalies = {correspondence[key] for key in retained if correspondence[key] in positives}
        associated_anomalies = {r for r, _ in references.values() if r in positives}
        ranked_anomalies = {item['reference'] for item in items if item['reference'] in positives}
        metrics = _rank_metrics(items, labels)
        rankable = bool(ranked_anomalies)
        row = {
            'scene': scene, 'asset_group': str(truth['asset_group']), 'route': setting['route'],
            'region': setting['region'], 'view_set_id': setting['view_set_id'],
            'view_count': setting['view_count'], 'method': method,
            'anomaly_count': len(positives), 'normal_count': len(labels) - len(positives),
            'strict_normal_majority': int(len(labels) - len(positives) > len(positives)),
            **metrics,
            'candidate_coverage': int(bool(candidate_anomalies)),
            'candidate_anomaly_recall': len(candidate_anomalies) / len(positives),
            'detection_loss': int(not candidate_anomalies),
            'pruning_loss': int(bool(candidate_anomalies) and not retained_anomalies),
            'association_loss': int(bool(retained_anomalies) and not associated_anomalies),
            'score_unavailability_loss': int(bool(associated_anomalies) and not rankable),
            'rankable': int(rankable),
            'ranking_loss': float(rankable) * (1.0 - metrics['top1_tie_expected']),
            'candidate_count': len(candidates), 'retained_candidate_count': len(retained),
            'unknown_candidate_count': sum(reference is None for reference in correspondence.values()),
            'unscored_object_count': len(objects) - len(items), 'scored_object_count': len(items),
            'all_objects_unscored': int(not items),
            **_identity_metrics(objects, correspondence, candidates),
        }
        loss = sum(row[name] for name in ('detection_loss', 'pruning_loss', 'association_loss', 'score_unavailability_loss', 'ranking_loss'))
        if not np.isclose(loss + metrics['top1_tie_expected'], 1.0):
            raise AssertionError('Failure decomposition does not preserve the scene denominator')
        rows.append(row)
    return rows


@lru_cache(maxsize=32)
def _bootstrap_weights(assets, samples, seed):
    groups = sorted(set(assets))
    membership = np.asarray([[a == group for a in assets] for group in groups], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(samples, len(groups)))
    counts = np.zeros((samples, len(groups)), dtype=float)
    for index in range(len(groups)):
        counts[:, index] = (draws == index).sum(axis=1)
    return counts @ membership


def _intervals(frame, metrics):
    frame = frame.sort_values('scene')
    if frame['scene'].duplicated().any():
        raise ValueError('Bootstrap rows must be one row per scene')
    values = frame[list(metrics)].to_numpy(dtype=float)
    weights = _bootstrap_weights(tuple(frame['asset_group'].astype(str)),
                                 PROTOCOL['bootstrap']['samples'], PROTOCOL['bootstrap']['seed'])
    samples = (weights @ values) / weights.sum(axis=1, keepdims=True)
    return {metric: (float(values[:, i].mean()), float(np.quantile(samples[:, i], .025)),
                     float(np.quantile(samples[:, i], .975))) for i, metric in enumerate(metrics)}


def _scene_means(frame, group_columns, metrics):
    return frame.groupby([*group_columns, 'scene', 'asset_group'], sort=True, dropna=False)[list(metrics)].mean().reset_index()


def _metric_summary(frame, group_columns):
    averaged = _scene_means(frame, group_columns, METRICS)
    result = []
    for key, group in averaged.groupby(group_columns, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_columns, key, strict=True))
        for metric, (mean, low, high) in _intervals(group, METRICS).items():
            result.append({**base, 'metric': metric, 'mean': mean, 'ci_low': low, 'ci_high': high,
                           'scene_count': len(group), 'asset_group_count': group['asset_group'].nunique()})
    return pd.DataFrame(result)


def _paired_summary(frame):
    output = []
    for exact in (True, False):
        columns = ['route', 'method', 'view_count'] + (['view_set_id'] if exact else [])
        averaged = _scene_means(frame, columns, METRICS)
        baseline = averaged[averaged['method'] == 'within_view_mean'].drop(columns='method')
        join = ['route', 'view_count', 'scene', 'asset_group'] + (['view_set_id'] if exact else [])
        for key, group in averaged.groupby(columns, sort=True):
            base = dict(zip(columns, key, strict=True))
            if base['method'] == 'within_view_mean':
                continue
            paired = group.merge(baseline, on=join, suffixes=('', '_baseline'), validate='one_to_one')
            if len(paired) != len(group):
                raise ValueError('Paired comparison has different scene denominators')
            for metric in METRICS:
                paired[metric] -= paired[f'{metric}_baseline']
            for metric, (delta, low, high) in _intervals(paired, METRICS).items():
                output.append({**base, 'view_set_id': base.get('view_set_id', 'all-schedules'),
                               'baseline': 'within_view_mean', 'metric': metric, 'delta': delta,
                               'ci_low': low, 'ci_high': high, 'scene_count': len(paired),
                               'asset_group_count': paired['asset_group'].nunique()})
    return pd.DataFrame(output)


def _diagnostic_summary(frame, names):
    columns = ['route', 'method', 'view_count']
    averaged = _scene_means(frame, columns, names)
    result = []
    for key, group in averaged.groupby(columns, sort=True):
        for metric in names:
            valid = group[metric].dropna()
            result.append({**dict(zip(columns, key, strict=True)), 'view_set_id': 'all-schedules',
                           'metric': metric, 'mean': float(valid.mean()) if len(valid) else None,
                           'scene_count': len(group), 'defined_scene_count': len(valid),
                           'asset_group_count': group['asset_group'].nunique()})
    return pd.DataFrame(result)


def choose_method(paired):
    """Apply the declared scoring gate; assisted development evidence only."""
    results = []
    for method in ('object_bank_mean', 'object_bank_top2'):
        subset = paired[(paired['route'] == 'supplied_reference') & (paired['method'] == method)]
        def value(schedule, metric, field):
            rows = subset[(subset['view_set_id'] == schedule) & (subset['metric'] == metric) & (subset['view_count'] == 3)]
            if len(rows) != 1:
                raise ValueError('Decision gate is missing a declared comparison')
            return float(rows.iloc[0][field])
        lower = value('three-primary', 'top1_tie_expected', 'ci_low')
        ap = value('three-primary', 'ap_tie_expected', 'delta')
        recall = value('three-primary', 'recall_at_2_tie_expected', 'delta')
        across = value('all-schedules', 'top1_tie_expected', 'delta')
        passed = lower > 0 and ap >= 0 and recall >= 0 and across > 0
        results.append({'method': method, 'passed': passed,
                        'primary_top1_delta': value('three-primary', 'top1_tie_expected', 'delta'),
                        'primary_top1_ci_low': lower, 'primary_ap_delta': ap,
                        'primary_recall_at_2_delta': recall, 'three_view_schedule_mean_top1_delta': across})
    eligible = [r for r in results if r['passed']]
    # Predeclared simplicity tie-break: mean precedes top2; no diagnostic control can win.
    selected = max(eligible, key=lambda r: (r['primary_top1_delta'], r['method'] == 'object_bank_mean'))['method'] if eligible else 'within_view_mean'
    return {'schema': PROTOCOL['schema'], 'mapping_allowed': bool(eligible), 'selected_method': selected,
            'baseline_method': 'within_view_mean', 'eligible_candidates': results,
            'decision': 'continue_bounded_mapping' if eligible else 'stop_new_scoring_and_matching',
            'interpretation': 'development selection only; supplied correspondence; no independent confirmation',
            'selection_basis': PROTOCOL['selection'],
            'diagnostic_controls_eligible': False}


def _atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.partial')
    with temporary.open('w') as stream:
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _preflight(prediction_root, prepared_root, scoring_root, evaluator_root):
    prediction_lock = verify_seal(prediction_root, name='prediction_lock.json')
    prepared_lock = verify_seal(prepared_root)
    scoring_lock = verify_seal(scoring_root, name='prediction_lock.json')
    verify_seal(scoring_root, name='feature_lock.json')
    if 'input_manifest.json' not in prepared_lock['files'] or 'run_binding.json' not in prediction_lock['files'] or 'run_binding.json' not in scoring_lock['files']:
        raise ValueError('Scientific seal omits a manifest or run binding')
    manifest = read_json(prepared_root / 'input_manifest.json')
    scenes = manifest['scenes']
    if len(scenes) != 100 or len(set(scenes)) != 100 or manifest['scene_count'] != 100:
        raise ValueError('This frozen evaluation requires the complete 100-scene cohort')
    if manifest['view_sets'] != PROTOCOL['view_sets']:
        raise ValueError('Prepared schedules differ from the frozen protocol')
    if manifest.get('anomaly_labels_opened') is not False or prepared_lock['metadata'].get('anomaly_labels_opened') is not False:
        raise ValueError('Prepared inputs do not establish a label-free boundary')
    scoring_hash = digest_file(scoring_root / 'prediction_lock.json')
    if manifest['scoring_prediction_lock_sha256'] != scoring_hash or manifest['scoring_feature_lock_sha256'] != digest_file(scoring_root / 'feature_lock.json'):
        raise ValueError('Prepared scoring seals do not match')
    scoring_binding = read_json(scoring_root / 'run_binding.json')
    if scoring_binding['input_lock_sha256'] != manifest['model_input_lock_sha256'] or scoring_binding['producer'] != 'dinov2-small':
        raise ValueError('Scoring/model binding or feature producer mismatch')
    binding = read_json(prediction_root / 'run_binding.json')
    if (prediction_lock['metadata'].get('binding_sha256') != digest(binding)
            or binding['protocol'] != PROTOCOL or binding['prepared_lock_sha256'] != digest_file(prepared_root / 'artifact_lock.json')
            or binding['scoring_lock_sha256'] != scoring_hash or prediction_lock['metadata']['scenes'] != scenes):
        raise ValueError('Prediction binding or scene membership mismatch')
    # Parse every model-facing setting before any evaluator scene is opened.
    for scene in scenes:
        relative = f'scenes/{scene}/predictions.json'
        if relative not in prediction_lock['files'] or f'scenes/{scene}.json' not in scoring_lock['files']:
            raise ValueError('Prediction or scoring scene is not part of the scientific seal')
        validate_predictions(read_json(prediction_root / relative), scene,
                             read_json(scoring_root / 'scenes' / f'{scene}.json'))
    # Hashing sealed evaluator bytes is permitted; parsing the scene labels follows binding checks.
    evaluator_lock = verify_seal(evaluator_root)
    index = read_json(evaluator_root / 'index.json')
    if len(index['scenes']) != 100 or set(index['scenes']) != set(scenes) or index['model_lock_sha256'] != manifest['model_input_lock_sha256']:
        raise ValueError('Evaluator belongs to different model inputs or scenes')
    if 'index.json' not in evaluator_lock['files'] or any(f'scenes/{s}.json' not in evaluator_lock['files'] for s in scenes):
        raise ValueError('Evaluator seal omits required scientific inputs')
    return manifest


def _historical_checks(frame, manifest):
    if manifest['model_input_lock_sha256'] != KNOWN_MODEL_LOCK:
        return []
    checks = []
    for route, method, schedule, expected in (
        ('automatic_predicted', 'winner_cluster_consensus', 'three-primary', 0.37666666666666665),
        ('supplied_reference', 'cached_forward_mean', 'three-primary', 0.5116666666666667),
        ('supplied_reference', 'cached_forward_mean', 'five-primary', 0.61),
    ):
        selected = frame[(frame['route'] == route) & (frame['method'] == method) & (frame['view_set_id'] == schedule)]
        observed = float(selected['top1_tie_expected'].mean())
        passed = len(selected) == 100 and np.isclose(observed, expected, atol=1e-10, rtol=0)
        checks.append({'route': route, 'method': method, 'view_set_id': schedule,
                       'metric': 'top1_tie_expected', 'expected': expected, 'observed': observed, 'passed': bool(passed)})
        if not passed:
            raise ValueError('Cached historical baseline failed exact reproduction')
    return checks


def evaluate(prediction_root, prepared_root, scoring_root, evaluator_root, output_root, source_root):
    roots = tuple(map(Path, (prediction_root, prepared_root, scoring_root, evaluator_root, output_root)))
    prediction_root, prepared_root, scoring_root, evaluator_root, output_root = roots
    manifest = _preflight(prediction_root, prepared_root, scoring_root, evaluator_root)
    binding = {'schema': PROTOCOL['schema'], 'protocol': PROTOCOL,
               'prediction_lock_sha256': digest_file(prediction_root / 'prediction_lock.json'),
               'prepared_lock_sha256': digest_file(prepared_root / 'artifact_lock.json'),
               'scoring_lock_sha256': digest_file(scoring_root / 'prediction_lock.json'),
               'evaluator_lock_sha256': digest_file(evaluator_root / 'artifact_lock.json'),
               'source': source_hashes(source_root, EVALUATION_SOURCES)}
    binding_hash = bind(output_root, binding)
    if (output_root / 'evaluation_lock.json').exists():
        completed = verify_seal(output_root, name='evaluation_lock.json')
        if completed['metadata'].get('binding_sha256') != binding_hash:
            raise ValueError('Completed evaluation seal has a different binding')
        return read_json(output_root / 'shareable/evaluation_receipt.json')
    started = time.monotonic()
    with terminal_status(output_root, '08O-object-bank-evaluation'):
        all_rows = []
        for index, scene in enumerate(manifest['scenes']):
            unit = output_root / 'private/scenes' / scene
            if not verified_unit(unit, binding_hash):
                truth = read_json(evaluator_root / 'scenes' / f'{scene}.json')
                if truth['scene'] != scene:
                    raise ValueError('Evaluator scene mismatch')
                payload = read_json(prediction_root / 'scenes' / scene / 'predictions.json')
                rows = [row for setting in payload['settings'] for row in evaluate_setting(scene, setting, truth)]
                with atomic_unit(unit, binding_hash) as temporary:
                    atomic_json(temporary / 'metrics.json', {'scene': scene, 'rows': rows})
            all_rows.extend(read_json(unit / 'metrics.json')['rows'])
            if (index + 1) % 10 == 0 or index == 0:
                progress(output_root, completed=index + 1, total=100, elapsed_seconds=time.monotonic() - started)
        frame = pd.DataFrame(all_rows)
        _atomic_csv(output_root / 'private/scene_metrics.csv', frame)
        historical = _historical_checks(frame, manifest)
        shareable = output_root / 'shareable'
        _atomic_csv(shareable / 'metric_summary.csv', _metric_summary(frame, ['route', 'method', 'view_count']))
        _atomic_csv(shareable / 'view_schedule_summary.csv', _metric_summary(frame, ['route', 'method', 'view_count', 'view_set_id']))
        paired = _paired_summary(frame)
        _atomic_csv(shareable / 'paired_asset_intervals.csv', paired)
        _atomic_csv(shareable / 'anomaly_count_summary.csv', _metric_summary(frame, ['route', 'method', 'view_count', 'anomaly_count']))
        _atomic_csv(shareable / 'majority_summary.csv', _metric_summary(frame, ['route', 'method', 'view_count', 'strict_normal_majority']))
        _atomic_csv(shareable / 'failure_summary.csv', _diagnostic_summary(frame, FAILURES))
        _atomic_csv(shareable / 'identity_summary.csv', _diagnostic_summary(frame, IDENTITY_METRICS))
        decision = choose_method(paired)
        atomic_json(shareable / 'decision.json', decision)
        receipt = {'schema': PROTOCOL['schema'], 'scene_count': 100,
                   'asset_group_count': int(frame['asset_group'].nunique()),
                   'view_schedule_count': len(PROTOCOL['view_sets']), 'metric_rows': len(frame),
                   'bootstrap': PROTOCOL['bootstrap'], 'historical_baseline_checks': historical,
                   'mapping_allowed': decision['mapping_allowed'], 'selected_method': decision['selected_method'],
                   'scientific_status': 'complete', 'scope': 'existing development cohort; no independent confirmation',
                   'aggregation': 'one row per scene per method; schedules averaged within scene before asset bootstrap',
                   'identity_undefined': 'no cross-view pairs yields undefined precision/recall; defined_scene_count is explicit',
                   'elapsed_seconds': time.monotonic() - started, 'shareable_files': list(SHAREABLE_FILES)}
        atomic_json(shareable / 'evaluation_receipt.json', receipt)
        paths = [output_root / 'run_binding.json', output_root / 'private/scene_metrics.csv']
        paths.extend(shareable / name for name in SHAREABLE_FILES)
        paths.extend(p for p in (output_root / 'private/scenes').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p))
        seal(output_root, paths, name='evaluation_lock.json', metadata={'binding_sha256': binding_hash,
             'scene_count': 100, 'labels_opened_after_prediction_seal': True, 'scientific_status': 'complete'})
        verify_seal(output_root, name='evaluation_lock.json')
    return receipt
