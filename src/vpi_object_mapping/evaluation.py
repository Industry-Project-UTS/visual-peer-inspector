"""Sealed evaluation of one bounded physical-object association comparison."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from vpi_object_bank import evaluation as bank
from vpi_object_bank.storage import (
    atomic_json, atomic_unit, bind, digest, digest_file, progress, read_json,
    seal, source_hashes, terminal_status, verified_unit, verify_seal,
)
from .protocol import PROTOCOL

METHODS = (
    'existing_all_candidates', 'global_partial_assignment',
    'winner_cluster_consensus', 'correspondence_assisted_reference',
)
BASELINE_SOURCES = {
    'existing_all_candidates': ('automatic_predicted', 'object_bank_top2'),
    'winner_cluster_consensus': ('automatic_predicted', 'winner_cluster_consensus'),
    'correspondence_assisted_reference': ('automatic_reference', 'object_bank_top2'),
}
METRICS = bank.METRICS
IDENTITY_METRICS = (*bank.IDENTITY_METRICS, 'identity_f1', 'object_track_count',
                    'singleton_track_count', 'singleton_track_fraction',
                    'singleton_observation_fraction')
PAIRED_METRICS = (*METRICS, 'identity_f1', 'identity_fragmentation',
                  'identity_precision', 'identity_recall', 'singleton_observation_fraction')
SHAREABLE_FILES = bank.SHAREABLE_FILES
EVALUATION_SOURCES = (
    'src/vpi_object_mapping/evaluation.py', 'src/vpi_object_mapping/protocol.py',
    'src/vpi_object_bank/evaluation.py', 'src/vpi_object_bank/protocol.py',
    'src/vpi_object_bank/storage.py',
    'src/visual_peer_inspector/workflows/learned_peer_evaluation.py',
    'src/visual_peer_inspector/workflows/comparison_io.py',
)


def _normalized_objects(objects):
    """Serialization order must not conceal a changed baseline score or member."""
    return [{**obj, 'members': sorted(obj['members'], key=bank._key)}
            for obj in sorted(objects, key=lambda row: str(row['id']))]


def validate_predictions(payload, scene, previous_payload, scoring_rows):
    if payload.get('schema') != PROTOCOL['schema'] or payload.get('scene') != scene:
        raise ValueError('Mapping prediction schema or scene mismatch')
    settings = payload['settings']
    schedules = [s['view_set_id'] for s in settings]
    if len(schedules) != len(set(schedules)) or set(schedules) != set(PROTOCOL['view_sets']):
        raise ValueError('Mapping schedules are missing, duplicated or unexpected')
    previous = {(s['route'], s['view_set_id']): s for s in previous_payload['settings']}
    source = {(row['route'], row['view']): row for row in scoring_rows}
    for setting in settings:
        schedule = setting['view_set_id']
        views = PROTOCOL['view_sets'][schedule]
        if (setting['route'] != 'automatic_predicted' or setting['region'] != 'automatic'
                or setting['views'] != list(views) or setting['view_count'] != len(views)):
            raise ValueError('Mapping route, region or selected views mismatch')
        expected = {(v, obj['candidate_id']) for v in views for obj in source[('automatic', v)]['objects']}
        observed = [bank._key(row) for row in setting['observations']]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError('Mapping candidates are missing, duplicated or outside selected views')
        if set(setting['rankings']) != set(METHODS):
            raise ValueError('Mapping ranking method schema mismatch')
        for method, objects in setting['rankings'].items():
            ids, members = [], []
            for obj in objects:
                ids.append(obj['id'])
                keys = [bank._key(member) for member in obj['members']]
                if not keys or len(keys) != len({v for v, _ in keys}):
                    raise ValueError('Empty mapping track or same-view duplicate')
                if not set(keys) <= expected:
                    raise ValueError('Mapping track uses an unknown or unselected observation')
                score = obj['score']
                if score is not None and not np.isfinite(float(score)):
                    raise ValueError('Mapping object score is nonfinite')
                if not isinstance(obj['support'], int) or not 0 <= obj['support'] <= len(keys):
                    raise ValueError('Mapping object support exceeds its observations')
                members.extend(keys)
            if len(ids) != len(set(ids)) or len(members) != len(set(members)):
                raise ValueError('Duplicate mapping identity or observation assignment')
            if method != 'winner_cluster_consensus' and set(members) != expected:
                raise ValueError('All-candidate mapping method dropped observations')
            if method in BASELINE_SOURCES:
                old_route, old_method = BASELINE_SOURCES[method]
                original = previous[(old_route, schedule)]['rankings'][old_method]
                if _normalized_objects(objects) != _normalized_objects(original):
                    raise ValueError('A frozen previous-stage baseline changed')


def evaluate_setting(scene, setting, truth):
    """Retain the established reference assignment, ranking and loss definitions."""
    rows = bank.evaluate_setting(scene, setting, truth)
    for row in rows:
        objects = setting['rankings'][row['method']]
        denominator = row['predicted_cross_view_pairs'] + row['reference_cross_view_pairs']
        singletons = sum(len(obj['members']) == 1 for obj in objects)
        row.update({
            'identity_f1': 2 * row['correct_cross_view_pairs'] / denominator if denominator else None,
            'object_track_count': len(objects), 'singleton_track_count': singletons,
            'singleton_track_fraction': singletons / len(objects) if objects else None,
            'singleton_observation_fraction': singletons / row['retained_candidate_count'] if row['retained_candidate_count'] else None,
            'identity_assistance': int(row['method'] == 'correspondence_assisted_reference'),
        })
    return rows


def paired_summary(frame):
    """Paired scene deltas, with schedules averaged before asset resampling."""
    result = []
    for exact in (True, False):
        columns = ['route', 'method', 'view_count'] + (['view_set_id'] if exact else [])
        averaged = bank._scene_means(frame, columns, PAIRED_METRICS)
        selected = averaged[averaged['method'] == 'global_partial_assignment']
        join = ['route', 'view_count', 'scene', 'asset_group'] + (['view_set_id'] if exact else [])
        for baseline_name in ('existing_all_candidates', 'winner_cluster_consensus'):
            baseline = averaged[averaged['method'] == baseline_name].drop(columns='method')
            for key, group in selected.groupby(columns, sort=True):
                base = dict(zip(columns, key, strict=True))
                paired = group.merge(baseline, on=join, suffixes=('', '_baseline'), validate='one_to_one')
                if len(paired) != len(group):
                    raise ValueError('Mapping paired comparison has different scene denominators')
                for metric in PAIRED_METRICS:
                    complete = paired.dropna(subset=[metric, f'{metric}_baseline']).copy()
                    complete[metric] -= complete[f'{metric}_baseline']
                    if len(complete):
                        delta, low, high = bank._intervals(complete, [metric])[metric]
                    else:
                        delta = low = high = None
                    result.append({**base, 'view_set_id': base.get('view_set_id', 'all-schedules'),
                                   'baseline': baseline_name, 'metric': metric,
                                   'delta': delta, 'ci_low': low, 'ci_high': high,
                                   'scene_count': len(paired), 'defined_scene_count': len(complete),
                                   'asset_group_count': paired['asset_group'].nunique()})
    return pd.DataFrame(result)


def choose_method(paired):
    """Apply the frozen mapping gate once; assisted results cannot select a matcher."""
    candidate = paired[(paired['route'] == 'automatic_predicted')
                       & (paired['method'] == 'global_partial_assignment')
                       & (paired['view_count'] == 3)]
    def comparison(baseline, schedule, metric):
        selected = candidate[(candidate['baseline'] == baseline)
                             & (candidate['view_set_id'] == schedule) & (candidate['metric'] == metric)]
        if len(selected) != 1:
            raise ValueError('Mapping decision is missing a declared paired comparison')
        row = selected.iloc[0]
        return {key: (None if pd.isna(row[key]) else float(row[key]))
                for key in ('delta', 'ci_low', 'ci_high', 'scene_count', 'defined_scene_count')}
    top1 = comparison('winner_cluster_consensus', 'three-primary', 'top1_tie_expected')
    ap = comparison('winner_cluster_consensus', 'three-primary', 'ap_tie_expected')
    recall = comparison('winner_cluster_consensus', 'three-primary', 'recall_at_2_tie_expected')
    across = comparison('winner_cluster_consensus', 'all-schedules', 'top1_tie_expected')
    f1 = comparison('existing_all_candidates', 'three-primary', 'identity_f1')
    fragmentation = comparison('existing_all_candidates', 'three-primary', 'identity_fragmentation')
    all_defined = all(row['scene_count'] == 100 and row['defined_scene_count'] == 100
                      for row in (top1, ap, recall, across, f1, fragmentation))
    gates = {
        'complete_primary_comparisons': all_defined,
        'top1_paired_lower_positive': top1['ci_low'] is not None and top1['ci_low'] > 0,
        'ap_no_regression': ap['delta'] is not None and ap['delta'] >= 0,
        'recall_at_2_no_regression': recall['delta'] is not None and recall['delta'] >= 0,
        'identity_f1_improved': f1['delta'] is not None and f1['delta'] > 0,
        'fragmentation_not_increased': fragmentation['delta'] is not None and fragmentation['delta'] <= 0,
        'three_view_schedule_mean_improved': across['delta'] is not None and across['delta'] > 0,
    }
    passed = all(gates.values())
    return {
        'schema': PROTOCOL['schema'], 'global_mapping_accepted': passed,
        'selected_method': 'global_partial_assignment' if passed else 'winner_cluster_consensus',
        'automatic_method': 'global_partial_assignment' if passed else 'winner_cluster_consensus',
        'assisted_fallback_method': 'correspondence_assisted_reference',
        'assisted_fallback_label': 'predicted masks with supplied neutral correspondence; dataset-assisted development mode',
        'decision': 'freeze_global_mapping' if passed else 'retain_winner_and_assisted_fallback_stop_search',
        'gate_checks': gates,
        'comparisons': {'primary_top1_vs_winner': top1, 'primary_ap_vs_winner': ap,
                        'primary_recall_at_2_vs_winner': recall, 'three_view_mean_top1_vs_winner': across,
                        'primary_identity_f1_vs_existing': f1, 'primary_fragmentation_vs_existing': fragmentation},
        'assisted_method_eligible': False, 'further_search_allowed': False,
        'interpretation': 'existing development cohort; no independent confirmation or 60-percent claim',
    }


def _preflight(prediction_root, previous_prediction_root, prepared_root, scoring_root, evaluator_root):
    if PROTOCOL['view_sets'] != bank.PROTOCOL['view_sets'] or PROTOCOL['bootstrap'] != bank.PROTOCOL['bootstrap']:
        raise ValueError('Mapping schedules or bootstrap differ from the reusable evaluation contract')
    lock = verify_seal(prediction_root, name='prediction_lock.json')
    if 'run_binding.json' not in lock['files']:
        raise ValueError('Mapping seal omits its run binding')
    binding = read_json(prediction_root / 'run_binding.json')
    expected = {
        'previous_prediction_lock_sha256': digest_file(previous_prediction_root / 'prediction_lock.json'),
        'prepared_lock_sha256': digest_file(prepared_root / 'artifact_lock.json'),
        'scoring_lock_sha256': digest_file(scoring_root / 'prediction_lock.json'),
    }
    if (binding.get('schema') != PROTOCOL['schema'] or binding.get('protocol') != PROTOCOL
            or lock['metadata'].get('binding_sha256') != digest(binding)
            or any(binding.get(name) != value for name, value in expected.items())):
        raise ValueError('Mapping predictions have incompatible sealed input bindings')
    # This helper verifies the old prediction, input/scoring packages and evaluator
    # index without parsing any evaluator scene labels.
    manifest = bank._preflight(previous_prediction_root, prepared_root, scoring_root, evaluator_root)
    previous_binding = read_json(previous_prediction_root / 'run_binding.json')
    if binding.get('feature_lock_sha256') != previous_binding.get('feature_lock_sha256') or not binding.get('feature_lock_sha256'):
        raise ValueError('Mapping features differ from the frozen object-bank features')
    scenes = manifest['scenes']
    if lock['metadata'].get('scenes') != scenes:
        raise ValueError('Mapping scene membership differs from the complete cohort')
    for scene in scenes:
        relative = f'scenes/{scene}/predictions.json'
        if relative not in lock['files']:
            raise ValueError('Mapping scene is absent from its scientific seal')
        validate_predictions(read_json(prediction_root / relative), scene,
                             read_json(previous_prediction_root / relative),
                             read_json(scoring_root / 'scenes' / f'{scene}.json'))
    return manifest


def evaluate(prediction_root, previous_prediction_root, prepared_root, scoring_root,
             evaluator_root, output_root, source_root):
    prediction_root, previous_prediction_root, prepared_root, scoring_root, evaluator_root, output_root = map(
        Path, (prediction_root, previous_prediction_root, prepared_root, scoring_root, evaluator_root, output_root))
    manifest = _preflight(prediction_root, previous_prediction_root, prepared_root, scoring_root, evaluator_root)
    binding = {
        'schema': PROTOCOL['schema'], 'protocol': PROTOCOL,
        'prediction_lock_sha256': digest_file(prediction_root / 'prediction_lock.json'),
        'previous_prediction_lock_sha256': digest_file(previous_prediction_root / 'prediction_lock.json'),
        'prepared_lock_sha256': digest_file(prepared_root / 'artifact_lock.json'),
        'scoring_lock_sha256': digest_file(scoring_root / 'prediction_lock.json'),
        'evaluator_lock_sha256': digest_file(evaluator_root / 'artifact_lock.json'),
        'source': source_hashes(source_root, EVALUATION_SOURCES),
    }
    binding_hash = bind(output_root, binding)
    if (output_root / 'evaluation_lock.json').exists():
        lock = verify_seal(output_root, name='evaluation_lock.json')
        if lock['metadata'].get('binding_sha256') != binding_hash:
            raise ValueError('Completed mapping evaluation has a different binding')
        return read_json(output_root / 'shareable/evaluation_receipt.json')
    started = time.monotonic()
    with terminal_status(output_root, 'physical-object-mapping-evaluation'):
        all_rows = []
        for index, scene in enumerate(manifest['scenes']):
            unit = output_root / 'private/scenes' / scene
            if not verified_unit(unit, binding_hash):
                truth = read_json(evaluator_root / 'scenes' / f'{scene}.json')
                if truth['scene'] != scene:
                    raise ValueError('Mapping evaluator scene mismatch')
                payload = read_json(prediction_root / 'scenes' / scene / 'predictions.json')
                rows = [row for setting in payload['settings'] for row in evaluate_setting(scene, setting, truth)]
                with atomic_unit(unit, binding_hash) as temporary:
                    atomic_json(temporary / 'metrics.json', {'scene': scene, 'rows': rows})
            all_rows.extend(read_json(unit / 'metrics.json')['rows'])
            if index == 0 or (index + 1) % 10 == 0:
                progress(output_root, completed=index + 1, total=100, elapsed_seconds=time.monotonic() - started)
        frame = pd.DataFrame(all_rows)
        if frame['scene'].nunique() != 100 or frame['asset_group'].nunique() != 27:
            raise ValueError('Mapping evaluation must retain 100 scenes and all 27 asset groups')
        counts = frame.groupby(['route', 'view_set_id', 'method'])['scene'].nunique()
        if not (counts == 100).all() or len(frame) != 100 * len(PROTOCOL['view_sets']) * len(METHODS):
            raise ValueError('Mapping metrics changed a method or schedule denominator')
        bank._atomic_csv(output_root / 'private/scene_metrics.csv', frame)
        shareable = output_root / 'shareable'
        bank._atomic_csv(shareable / 'metric_summary.csv', bank._metric_summary(frame, ['route', 'method', 'view_count']))
        bank._atomic_csv(shareable / 'view_schedule_summary.csv', bank._metric_summary(frame, ['route', 'method', 'view_count', 'view_set_id']))
        paired = paired_summary(frame)
        bank._atomic_csv(shareable / 'paired_asset_intervals.csv', paired)
        bank._atomic_csv(shareable / 'anomaly_count_summary.csv', bank._metric_summary(frame, ['route', 'method', 'view_count', 'anomaly_count']))
        bank._atomic_csv(shareable / 'majority_summary.csv', bank._metric_summary(frame, ['route', 'method', 'view_count', 'strict_normal_majority']))
        bank._atomic_csv(shareable / 'failure_summary.csv', bank._diagnostic_summary(frame, bank.FAILURES))
        bank._atomic_csv(shareable / 'identity_summary.csv', bank._diagnostic_summary(frame, IDENTITY_METRICS))
        decision = choose_method(paired)
        atomic_json(shareable / 'decision.json', decision)
        cohort = frame.drop_duplicates('scene')
        receipt = {
            'schema': PROTOCOL['schema'], 'scientific_status': 'complete',
            'scene_count': 100, 'asset_group_count': 27, 'view_schedule_count': len(PROTOCOL['view_sets']),
            'metric_rows': len(frame), 'bootstrap': PROTOCOL['bootstrap'],
            'selected_method': decision['selected_method'], 'global_mapping_accepted': decision['global_mapping_accepted'],
            'assisted_fallback_method': decision['assisted_fallback_method'],
            'scope': 'existing development cohort; no independent confirmation',
            'aggregation': 'one scene per method; average schedules within scene before asset bootstrap',
            'identity_f1_definition': '2*correct_cross_view_pairs/(predicted_cross_view_pairs+reference_cross_view_pairs); undefined if both zero',
            'undefined_identity_handling': 'explicit defined_scene_count; the primary gate requires all 100 scenes',
            'assisted_method_label': decision['assisted_fallback_label'],
            'cohort_anomaly_objects': int(cohort['anomaly_count'].sum()),
            'cohort_normal_objects': int(cohort['normal_count'].sum()),
            'strict_normal_majority_scenes': int(cohort['strict_normal_majority'].sum()),
            'elapsed_seconds': time.monotonic() - started, 'shareable_files': list(SHAREABLE_FILES),
        }
        atomic_json(shareable / 'evaluation_receipt.json', receipt)
        paths = [output_root / 'run_binding.json', output_root / 'private/scene_metrics.csv']
        paths.extend(shareable / name for name in SHAREABLE_FILES)
        paths.extend(path for path in (output_root / 'private/scenes').rglob('*')
                     if path.is_file() and '.partial' not in str(path) and '.interrupted-' not in str(path))
        seal(output_root, paths, name='evaluation_lock.json', metadata={
            'binding_sha256': binding_hash, 'scene_count': 100, 'asset_group_count': 27,
            'labels_opened_after_prediction_seal': True, 'scientific_status': 'complete'})
        verify_seal(output_root, name='evaluation_lock.json')
    return receipt
