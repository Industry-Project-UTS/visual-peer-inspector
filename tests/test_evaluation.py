"""Evaluator tests use synthetic identities; private source fixtures are unnecessary."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from vpi_object_bank import evaluation as ev
from vpi_object_bank.storage import atomic_json, digest, digest_file, read_json, seal, verify_seal


def setting(objects=None, *, views=('V00', 'V07', 'V14'), method='object_bank_mean'):
    if objects is None:
        objects = [
            {'id': name, 'score': score, 'support': len(views),
             'members': [{'view_id': v, 'candidate_id': f'C{index}'} for v in views]}
            for index, (name, score) in enumerate((('N-a', .9), ('N-b', .4), ('N-c', .2)))
        ]
    observations = [{'view_id': view, 'candidate_id': f'C{i}', 'object_id': f'N-{i}'}
                    for view in views for i in range(3)]
    return {'route': 'supplied_reference', 'region': 'supplied', 'view_set_id': 'three-primary',
            'views': list(views), 'view_count': len(views), 'observations': observations,
            'rankings': {method: objects}}


def truth(views=('V00', 'V07', 'V14')):
    return {'scene': 'S1', 'asset_group': 'A1', 'labels': {'O0': 1, 'O1': 0, 'O2': 0},
            'correspondence': {f'{region}/{v}/C{i}': f'O{i}' for region in ('supplied', 'automatic') for v in views for i in range(3)}}


def test_tie_metrics_are_exact_and_candidate_order_invariant():
    s = setting()
    for obj in s['rankings']['object_bank_mean']:
        obj['score'] = .5
    a = ev.evaluate_setting('S1', s, truth())[0]
    s['rankings']['object_bank_mean'].reverse()
    s['observations'].reverse()
    b = ev.evaluate_setting('S1', s, truth())[0]
    assert a == b
    assert a['top1_tie_expected'] == pytest.approx(1 / 3)
    assert a['recall_at_2_tie_expected'] == pytest.approx(2 / 3)
    assert a['ap_tie_expected'] == pytest.approx((1 + .5 + 1 / 3) / 3)
    assert a['ranking_loss'] == pytest.approx(2 / 3)


def test_missing_candidates_and_empty_rankings_preserve_positive_denominator():
    s = setting(objects=[])
    s['observations'] = []
    t = truth()
    t['labels']['O1'] = 1
    row = ev.evaluate_setting('S1', s, t)[0]
    assert row['positive_count'] == 2
    assert row['detection_loss'] == 1 and row['all_objects_unscored'] == 1
    assert row['top1_tie_expected'] == row['recall_at_2_tie_expected'] == 0
    assert row['identity_precision'] is None


def test_pruning_association_score_and_ranking_losses_are_distinct():
    s = setting()
    base_objects = deepcopy(s['rankings']['object_bank_mean'])
    # Candidate exists but the method only retains normals.
    s['rankings']['object_bank_mean'] = base_objects[1:]
    assert ev.evaluate_setting('S1', s, truth())[0]['pruning_loss'] == 1
    # A 1:1 mixed identity is not an anomaly-majority track.
    s['rankings']['object_bank_mean'] = [{'id': 'mixed', 'score': 1., 'support': 2,
        'members': [{'view_id': 'V00', 'candidate_id': 'C0'}, {'view_id': 'V07', 'candidate_id': 'C1'}]}]
    assert ev.evaluate_setting('S1', s, truth())[0]['association_loss'] == 1
    # Correct identity still exists but its score is unavailable.
    base_objects[0]['score'] = None
    s['rankings']['object_bank_mean'] = base_objects
    row = ev.evaluate_setting('S1', s, truth())[0]
    assert row['score_unavailability_loss'] == 1 and row['unscored_object_count'] == 1
    base_objects[0]['score'] = .1
    row = ev.evaluate_setting('S1', s, truth())[0]
    assert row['rankable'] == 1 and row['ranking_loss'] == 1


def test_unscored_competing_track_remains_in_one_to_one_reference_assignment():
    s = setting()
    s['rankings']['object_bank_mean'] = [
        {'id': 'N-complete', 'score': None, 'support': 0, 'members': [
            {'view_id': 'V00', 'candidate_id': 'C0'}, {'view_id': 'V07', 'candidate_id': 'C0'}]},
        {'id': 'N-fragment', 'score': 1., 'support': 1, 'members': [{'view_id': 'V14', 'candidate_id': 'C0'}]},
    ]
    row = ev.evaluate_setting('S1', s, truth())[0]
    assert row['top1_tie_expected'] == 0 and row['score_unavailability_loss'] == 1
    assert row['identity_fragmentation'] == pytest.approx(1 / 3)
    assert row['identity_recall'] == pytest.approx(1 / 9)


def test_unknown_is_not_a_normal_reference_and_missing_is_rejected():
    s, t = setting(), truth()
    t['correspondence']['supplied/V00/C0'] = None
    row = ev.evaluate_setting('S1', s, t)[0]
    assert row['unknown_candidate_count'] == 1
    assert row['top1_tie_expected'] == 1  # two of three known anomaly observations
    assert row['identity_precision'] == pytest.approx(7 / 9)
    del t['correspondence']['supplied/V00/C0']
    with pytest.raises(ValueError, match='explicit unknown'):
        ev.evaluate_setting('S1', s, t)


def test_scene_bootstrap_matches_cluster_resampling_and_averages_schedules_first():
    frame = pd.DataFrame([
        {'scene': 'S0', 'asset_group': 'A', 'route': 'r', 'method': 'm', 'view_count': 3,
         'view_set_id': schedule, **{m: value for m in ev.METRICS}}
        for schedule, value in [('p', 1.), ('q', 0.)]]
        + [{'scene': 'S1', 'asset_group': 'B', 'route': 'r', 'method': 'm', 'view_count': 3,
            'view_set_id': schedule, **{m: 0. for m in ev.METRICS}} for schedule in ('p', 'q')])
    summary = ev._metric_summary(frame, ['route', 'method', 'view_count'])
    assert set(summary['scene_count']) == {2}
    assert set(summary['mean']) == {.25}
    assert set(summary['ci_low']) == {0.}
    assert set(summary['ci_high']) == {.5}
    duplicated_asset = pd.DataFrame({'scene': ['s0', 's1', 's2'], 'asset_group': ['a', 'a', 'b'], 'x': [0., 1., 1.]})
    mean, low, high = ev._intervals(duplicated_asset, ['x'])['x']
    assert mean == pytest.approx(2 / 3) and low == .5 and high == 1.


def test_gate_excludes_controls_and_requires_each_declared_condition():
    rows = []
    for method in ('object_bank_mean', 'object_bank_top2', 'view_mean_control_mean'):
        for schedule in ('three-primary', 'all-schedules'):
            for metric in ev.METRICS:
                rows.append({'route': 'supplied_reference', 'method': method, 'view_count': 3,
                             'view_set_id': schedule, 'metric': metric, 'delta': .1, 'ci_low': .01})
    paired = pd.DataFrame(rows)
    assert ev.choose_method(paired)['selected_method'] == 'object_bank_mean'
    paired.loc[paired.method == 'object_bank_mean', 'ci_low'] = -.01
    assert ev.choose_method(paired)['selected_method'] == 'object_bank_top2'
    paired.loc[(paired.method == 'object_bank_top2') & (paired.metric == 'ap_tie_expected'), 'delta'] = -.01
    decision = ev.choose_method(paired)
    assert decision['mapping_allowed'] is False and decision['selected_method'] == 'within_view_mean'
    assert decision['diagnostic_controls_eligible'] is False


@pytest.fixture
def sealed_inputs(tmp_path, monkeypatch):
    protocol = deepcopy(ev.PROTOCOL)
    protocol['view_sets'] = {'three-primary': ['V00', 'V07', 'V14'], 'three-rotated-04': ['V04', 'V11', 'V18']}
    protocol['routes'] = ['supplied_reference']
    protocol['scorers'] = ['within_view', 'object_bank', 'view_mean_control', 'cached_forward']
    monkeypatch.setattr(ev, 'PROTOCOL', protocol)
    roots = {name: tmp_path / name for name in ('prepared', 'scoring', 'evaluator', 'prediction', 'evaluation')}
    scenes = [f'S{i:06d}' for i in range(100)]
    all_views = [v for views in protocol['view_sets'].values() for v in views]
    for i, scene in enumerate(scenes):
        score_rows = [{'route': 'supplied', 'view': v, 'objects': [{'candidate_id': f'C{c}'} for c in range(3)]} for v in all_views]
        atomic_json(roots['scoring'] / 'scenes' / f'{scene}.json', score_rows)
        t = truth(all_views)
        t['scene'], t['asset_group'] = scene, f'A{i // 4:03d}'
        atomic_json(roots['evaluator'] / 'scenes' / f'{scene}.json', t)
        settings = []
        for schedule, views in protocol['view_sets'].items():
            s = setting(views=views)
            objects = s['rankings']['object_bank_mean']
            s['view_set_id'] = schedule
            s['rankings'] = {name: deepcopy(objects) for name in ev._methods('supplied_reference')}
            settings.append(s)
        atomic_json(roots['prediction'] / 'scenes' / scene / 'predictions.json', {'schema': protocol['schema'], 'scene': scene, 'settings': settings})
    atomic_json(roots['scoring'] / 'run_binding.json', {'input_lock_sha256': 'model-digest', 'producer': 'dinov2-small'})
    seal(roots['scoring'], list((roots['scoring'] / 'scenes').glob('*.json')), name='feature_lock.json')
    seal(roots['scoring'], [p for p in roots['scoring'].rglob('*') if p.is_file()], name='prediction_lock.json')
    atomic_json(roots['prepared'] / 'input_manifest.json', {
        'scenes': scenes, 'scene_count': 100, 'view_sets': protocol['view_sets'], 'anomaly_labels_opened': False,
        'scoring_prediction_lock_sha256': digest_file(roots['scoring'] / 'prediction_lock.json'),
        'scoring_feature_lock_sha256': digest_file(roots['scoring'] / 'feature_lock.json'),
        'model_input_lock_sha256': 'model-digest'})
    seal(roots['prepared'], [roots['prepared'] / 'input_manifest.json'], metadata={'anomaly_labels_opened': False})
    atomic_json(roots['evaluator'] / 'index.json', {'scenes': scenes, 'model_lock_sha256': 'model-digest'})
    seal(roots['evaluator'], [p for p in roots['evaluator'].rglob('*') if p.is_file()])
    binding = {'schema': protocol['schema'], 'protocol': protocol,
        'prepared_lock_sha256': digest_file(roots['prepared'] / 'artifact_lock.json'),
        'scoring_lock_sha256': digest_file(roots['scoring'] / 'prediction_lock.json')}
    atomic_json(roots['prediction'] / 'run_binding.json', binding)
    seal(roots['prediction'], [p for p in roots['prediction'].rglob('*') if p.is_file()], name='prediction_lock.json', metadata={'scenes': scenes, 'binding_sha256': digest(binding)})
    return roots


def run(roots):
    return ev.evaluate(roots['prediction'], roots['prepared'], roots['scoring'], roots['evaluator'],
                       roots['evaluation'], Path(__file__).resolve().parents[1])


def test_complete_run_seals_aggregates_and_exact_resume_does_not_write(sealed_inputs):
    receipt = run(sealed_inputs)
    assert receipt['scene_count'] == 100 and receipt['mapping_allowed'] is False
    output = sealed_inputs['evaluation']
    lock = verify_seal(output, name='evaluation_lock.json')
    assert set(f'shareable/{name}' for name in ev.SHAREABLE_FILES) <= set(lock['files'])
    before = {str(p): (p.stat().st_mtime_ns, digest_file(p)) for p in output.rglob('*') if p.is_file()}
    assert run(sealed_inputs) == receipt
    after = {str(p): (p.stat().st_mtime_ns, digest_file(p)) for p in output.rglob('*') if p.is_file()}
    assert before == after
    summary = pd.read_csv(output / 'shareable/metric_summary.csv')
    assert set(summary['scene_count']) == {100} and set(summary['mean']) == {1.}
    for path in (output / 'shareable').iterdir():
        text = path.read_text()
        assert 'S000000' not in text and 'A000' not in text and str(sealed_inputs['evaluator']) not in text


def test_corrupt_model_seal_stops_before_evaluator_labels_open(sealed_inputs, monkeypatch):
    p = sealed_inputs['prediction'] / 'scenes/S000000/predictions.json'
    p.write_text(p.read_text() + ' ')
    opened = []
    original = ev.read_json
    def spy(path):
        if Path(path).is_relative_to(sealed_inputs['evaluator'] / 'scenes'):
            opened.append(path)
        return original(path)
    monkeypatch.setattr(ev, 'read_json', spy)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        run(sealed_inputs)
    assert opened == [] and not sealed_inputs['evaluation'].exists()


def test_invalid_selected_view_stops_before_labels_open(sealed_inputs, monkeypatch):
    root = sealed_inputs['prediction']
    p = root / 'scenes/S000000/predictions.json'
    payload = read_json(p)
    payload['settings'][0]['observations'][0]['view_id'] = 'V19'
    atomic_json(p, payload)
    lock = read_json(root / 'prediction_lock.json')
    seal(root, [root / path for path in lock['files']], name='prediction_lock.json', metadata=lock['metadata'])
    opened = []
    original = ev.read_json
    monkeypatch.setattr(ev, 'read_json', lambda path: (opened.append(path) or original(path)) if Path(path).is_relative_to(sealed_inputs['evaluator'] / 'scenes') else original(path))
    with pytest.raises(ValueError, match='outside selected views'):
        run(sealed_inputs)
    assert opened == []


def test_same_view_track_duplicates_are_rejected():
    s = setting()
    objects = s['rankings']['object_bank_mean']
    objects[0]['members'][1]['view_id'] = 'V00'
    protocol = deepcopy(ev.PROTOCOL)
    # Use a complete payload so validation reaches the duplicate constraint.
    payload = {'schema': ev.PROTOCOL['schema'], 'scene': 'S1', 'settings': []}
    score_rows = []
    for route in ev.PROTOCOL['routes']:
        region = 'supplied' if route == 'supplied_reference' else 'automatic'
        for schedule, views in ev.PROTOCOL['view_sets'].items():
            row = setting(views=views)
            row.update(route=route, region=region, view_set_id=schedule)
            row['rankings'] = {m: deepcopy(row['rankings']['object_bank_mean']) for m in ev._methods(route)}
            payload['settings'].append(row)
            score_rows.extend({'route': region, 'view': v, 'objects': [{'candidate_id': f'C{i}'} for i in range(3)]} for v in views)
    target = next(r for r in payload['settings'] if len(r['views']) == 3)
    next(iter(target['rankings'].values()))[0]['members'][1]['view_id'] = target['views'][0]
    with pytest.raises(ValueError, match='same-view duplicate'):
        ev.validate_predictions(payload, 'S1', score_rows)


def test_report_failure_reuses_atomic_scientific_scene_checkpoints(sealed_inputs, monkeypatch):
    original = ev._metric_summary
    monkeypatch.setattr(ev, '_metric_summary', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('synthetic report failure')))
    with pytest.raises(RuntimeError, match='synthetic report'):
        run(sealed_inputs)
    root = sealed_inputs['evaluation']
    assert len(list((root / 'private/scenes').glob('*/metrics.json'))) == 100
    assert read_json(root / 'run_status.json')['status'] == 'failed'
    assert not (root / 'evaluation_lock.json').exists()
    monkeypatch.setattr(ev, '_metric_summary', original)
    monkeypatch.setattr(ev, 'evaluate_setting', lambda *args: (_ for _ in ()).throw(AssertionError('completed scenes must be reused')))
    assert run(sealed_inputs)['scientific_status'] == 'complete'


def test_corrupt_scene_checkpoint_stops_instead_of_recomputing(sealed_inputs, monkeypatch):
    monkeypatch.setattr(ev, '_metric_summary', lambda *args: (_ for _ in ()).throw(RuntimeError('report failure')))
    with pytest.raises(RuntimeError):
        run(sealed_inputs)
    path = sealed_inputs['evaluation'] / 'private/scenes/S000000/metrics.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        run(sealed_inputs)


def test_changed_configuration_rejects_resume(sealed_inputs):
    run(sealed_inputs)
    path = sealed_inputs['evaluation'] / 'run_binding.json'
    binding = read_json(path)
    binding['protocol']['bootstrap']['seed'] += 1
    atomic_json(path, binding)
    with pytest.raises(ValueError, match='different inputs'):
        run(sealed_inputs)


def test_unmocked_preflight_checks_binding_without_parsing_evaluator_labels(sealed_inputs, monkeypatch):
    original = ev.read_json
    def forbid_label_parse(path):
        if Path(path).is_relative_to(sealed_inputs['evaluator'] / 'scenes'):
            raise AssertionError('Preflight must not parse evaluator scene labels')
        return original(path)
    monkeypatch.setattr(ev, 'read_json', forbid_label_parse)
    manifest = ev._preflight(sealed_inputs['prediction'], sealed_inputs['prepared'],
                             sealed_inputs['scoring'], sealed_inputs['evaluator'])
    assert manifest['scene_count'] == 100
    binding = read_json(sealed_inputs['prediction'] / 'run_binding.json')
    lock = read_json(sealed_inputs['prediction'] / 'prediction_lock.json')
    assert lock['metadata']['binding_sha256'] == ev.digest(binding)


def test_unmocked_preflight_rejects_stale_binding_digest(sealed_inputs):
    path = sealed_inputs['prediction'] / 'prediction_lock.json'
    lock = read_json(path)
    lock['metadata']['binding_sha256'] = '0' * 64
    atomic_json(path, lock)
    with pytest.raises(ValueError, match='Prediction binding'):
        ev._preflight(sealed_inputs['prediction'], sealed_inputs['prepared'],
                      sealed_inputs['scoring'], sealed_inputs['evaluator'])
