"""Synthetic evaluation contracts for the fixed physical-object mapping stage."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vpi_object_bank import evaluation as bank
from vpi_object_bank.storage import atomic_json, digest, digest_file, read_json, seal, verify_seal
from vpi_object_mapping import evaluation as ev


def objects(views, *, singleton=False):
    if singleton:
        return [{'id': f'N{i}-{v}', 'score': .9 - i * .3, 'support': 1,
                 'members': [{'view_id': v, 'candidate_id': f'C{i}'}]}
                for i in range(3) for v in views]
    return [{'id': f'N{i}', 'score': .9 - i * .3, 'support': len(views),
             'members': [{'view_id': v, 'candidate_id': f'C{i}'} for v in views]}
            for i in range(3)]


def setting(views=('V00', 'V07', 'V14')):
    return {'route': 'automatic_predicted', 'region': 'automatic', 'view_set_id': 'three-primary',
            'views': list(views), 'view_count': len(views),
            'observations': [{'view_id': v, 'candidate_id': f'C{i}', 'object_id': f'N{i}'} for v in views for i in range(3)],
            'rankings': {method: objects(views) for method in ev.METHODS}}


def truth(views=('V00', 'V07', 'V14')):
    return {'scene': 'S0', 'asset_group': 'A0', 'labels': {'O0': 1, 'O1': 0, 'O2': 0},
            'correspondence': {f'{region}/{v}/C{i}': f'O{i}' for region in ('supplied', 'automatic') for v in views for i in range(3)}}


def previous_settings(schedules):
    settings = []
    for route in bank.PROTOCOL['routes']:
        region = 'supplied' if route == 'supplied_reference' else 'automatic'
        for schedule, views in schedules.items():
            current = setting(views)
            current.update(route=route, region=region, view_set_id=schedule)
            current['rankings'] = {method: objects(views) for method in bank._methods(route)}
            settings.append(current)
    return settings


def payloads():
    previous = {'scene': 'S0', 'schema': bank.PROTOCOL['schema'], 'settings': previous_settings(ev.PROTOCOL['view_sets'])}
    current = {'scene': 'S0', 'schema': ev.PROTOCOL['schema'], 'settings': []}
    scores = []
    for schedule, views in ev.PROTOCOL['view_sets'].items():
        current_setting = setting(views)
        current_setting['view_set_id'] = schedule
        current['settings'].append(current_setting)
        scores.extend({'route': region, 'view': v, 'objects': [{'candidate_id': f'C{i}'} for i in range(3)]}
                      for region in ('supplied', 'automatic') for v in views)
    return current, previous, scores


def test_frozen_protocol_retains_all_nineteen_schedules_and_four_methods():
    assert len(ev.PROTOCOL['view_sets']) == 19
    assert set(ev.PROTOCOL['methods']) == set(ev.METHODS)
    assert set(map(len, ev.PROTOCOL['view_sets'].values())) == {1, 2, 3, 5}


def test_complete_map_has_perfect_identity_f1_and_no_singletons():
    row = next(r for r in ev.evaluate_setting('S0', setting(), truth()) if r['method'] == 'global_partial_assignment')
    assert row['identity_f1'] == row['identity_precision'] == row['identity_recall'] == 1
    assert row['identity_fragmentation'] == row['singleton_track_count'] == 0
    assert row['top1_tie_expected'] == 1


def test_all_singletons_have_zero_f1_when_reference_links_exist():
    current = setting()
    current['rankings']['global_partial_assignment'] = objects(current['views'], singleton=True)
    row = next(r for r in ev.evaluate_setting('S0', current, truth()) if r['method'] == 'global_partial_assignment')
    assert row['identity_f1'] == 0 and row['identity_precision'] is None
    assert row['identity_fragmentation'] == 2
    assert row['singleton_track_fraction'] == row['singleton_observation_fraction'] == 1
    assert row['candidate_count'] == row['retained_candidate_count'] == 9


def test_single_view_identity_f1_is_explicitly_undefined():
    current = setting(('V00',))
    row = ev.evaluate_setting('S0', current, truth(('V00',)))[0]
    assert row['identity_f1'] is None and row['singleton_track_fraction'] == 1


def test_unknowns_null_scores_and_ties_keep_canonical_failure_accounting():
    current, reference = setting(), truth()
    reference['correspondence']['automatic/V00/C0'] = None
    for obj in current['rankings']['global_partial_assignment']:
        obj['score'] = .5
    rows = ev.evaluate_setting('S0', current, reference)
    row = next(r for r in rows if r['method'] == 'global_partial_assignment')
    assert row['unknown_candidate_count'] == 1 and row['top1_tie_expected'] == pytest.approx(1 / 3)
    assert row['ranking_loss'] == pytest.approx(2 / 3)
    current['rankings']['global_partial_assignment'].reverse()
    current['observations'].reverse()
    assert ev.evaluate_setting('S0', current, reference) == rows
    next(o for o in current['rankings']['global_partial_assignment'] if o['id'] == 'N0')['score'] = None
    row = next(r for r in ev.evaluate_setting('S0', current, reference) if r['method'] == 'global_partial_assignment')
    assert row['score_unavailability_loss'] == 1 and row['unscored_object_count'] == 1


def test_validator_conserves_every_candidate_and_freezes_baselines():
    current, previous, scores = payloads()
    ev.validate_predictions(current, 'S0', previous, scores)
    target = next(s for s in current['settings'] if s['view_set_id'] == 'three-primary')
    target['rankings']['global_partial_assignment'].pop()
    with pytest.raises(ValueError, match='dropped observations'):
        ev.validate_predictions(current, 'S0', previous, scores)
    target['rankings']['global_partial_assignment'] = objects(target['views'])
    target['rankings']['winner_cluster_consensus'][0]['score'] = .123
    with pytest.raises(ValueError, match='frozen previous-stage baseline changed'):
        ev.validate_predictions(current, 'S0', previous, scores)


def test_validator_rejects_same_view_duplicates_and_view_leakage():
    current, previous, scores = payloads()
    target = next(s for s in current['settings'] if s['view_set_id'] == 'three-primary')
    target['rankings']['global_partial_assignment'][0]['members'][1]['view_id'] = 'V00'
    with pytest.raises(ValueError, match='same-view duplicate'):
        ev.validate_predictions(current, 'S0', previous, scores)
    target['rankings']['global_partial_assignment'] = objects(target['views'])
    target['observations'][0]['view_id'] = 'V19'
    with pytest.raises(ValueError, match='outside selected views'):
        ev.validate_predictions(current, 'S0', previous, scores)


def gate_frame():
    rows = []
    for baseline in ('existing_all_candidates', 'winner_cluster_consensus'):
        for schedule in ('three-primary', 'all-schedules'):
            for metric in ev.PAIRED_METRICS:
                rows.append({'route': 'automatic_predicted', 'method': 'global_partial_assignment',
                             'view_count': 3, 'view_set_id': schedule, 'baseline': baseline, 'metric': metric,
                             'delta': -.1 if metric == 'identity_fragmentation' else .1,
                             'ci_low': .01, 'ci_high': .2, 'scene_count': 100, 'defined_scene_count': 100})
    return pd.DataFrame(rows)


def test_mapping_gate_requires_all_predeclared_conditions_and_cannot_select_assistance():
    paired = gate_frame()
    assert ev.choose_method(paired)['global_mapping_accepted'] is True
    for metric, baseline, field, value in (
        ('top1_tie_expected', 'winner_cluster_consensus', 'ci_low', 0.),
        ('ap_tie_expected', 'winner_cluster_consensus', 'delta', -.01),
        ('recall_at_2_tie_expected', 'winner_cluster_consensus', 'delta', -.01),
        ('identity_f1', 'existing_all_candidates', 'delta', 0.),
        ('identity_fragmentation', 'existing_all_candidates', 'delta', .01),
        ('identity_f1', 'existing_all_candidates', 'defined_scene_count', 99),
    ):
        candidate = paired.copy()
        candidate.loc[(candidate.metric == metric) & (candidate.baseline == baseline), field] = value
        decision = ev.choose_method(candidate)
        assert decision['selected_method'] == 'winner_cluster_consensus'
        assert decision['assisted_method_eligible'] is False
        assert decision['further_search_allowed'] is False
    candidate = paired.copy()
    candidate.loc[(candidate.metric == 'top1_tie_expected') & (candidate.view_set_id == 'all-schedules'), 'delta'] = 0
    assert ev.choose_method(candidate)['global_mapping_accepted'] is False


def test_paired_comparison_averages_schedules_within_scenes_and_keeps_undefined_counts():
    rows = []
    for scene in ('S0', 'S1'):
        for schedule in ('three-primary', 'three-other'):
            for method in ev.METHODS:
                value = float(scene == 'S0' and schedule == 'three-primary' and method == 'global_partial_assignment')
                rows.append({'scene': scene, 'asset_group': scene, 'route': 'automatic_predicted',
                             'method': method, 'view_count': 3, 'view_set_id': schedule,
                             **{metric: value for metric in ev.PAIRED_METRICS}})
    frame = pd.DataFrame(rows)
    frame['identity_precision'] = None
    result = ev.paired_summary(frame)
    averaged = result[(result.view_set_id == 'all-schedules') & (result.metric == 'top1_tie_expected')]
    assert set(averaged.scene_count) == {2} and set(averaged['delta']) == {.25}
    assert set(averaged.ci_low) == {0.} and set(averaged.ci_high) == {.5}
    undefined = result[result.metric == 'identity_precision']
    assert set(undefined.defined_scene_count) == {0} and undefined['delta'].isna().all()


@pytest.fixture
def sealed_inputs(tmp_path, monkeypatch):
    # Keep the real preflight and 100-scene denominator; two schedules make the
    # synthetic end-to-end fixture small. Production protocol equality is checked separately.
    schedules = {'three-primary': ['V00', 'V07', 'V14'], 'three-rotated-04': ['V04', 'V11', 'V18']}
    bank_protocol = deepcopy(bank.PROTOCOL)
    bank_protocol['view_sets'] = schedules
    mapping_protocol = deepcopy(ev.PROTOCOL)
    mapping_protocol['view_sets'] = schedules
    monkeypatch.setattr(bank, 'PROTOCOL', bank_protocol)
    monkeypatch.setattr(ev, 'PROTOCOL', mapping_protocol)
    roots = {name: tmp_path / name for name in ('prepared', 'scoring', 'evaluator', 'previous', 'prediction', 'evaluation')}
    scenes = [f'S{i:06d}' for i in range(100)]
    all_views = [v for views in schedules.values() for v in views]
    for index, scene in enumerate(scenes):
        current, previous, scores = payloads()
        current['scene'] = previous['scene'] = scene
        # Avoid duplicate route/view rows introduced by helper schedules.
        scores = list({(s['route'], s['view']): s for s in scores}.values())
        atomic_json(roots['scoring'] / 'scenes' / f'{scene}.json', scores)
        reference = truth(all_views)
        reference.update(scene=scene, asset_group=f'A{index % 27:02d}')
        atomic_json(roots['evaluator'] / 'scenes' / f'{scene}.json', reference)
        atomic_json(roots['previous'] / 'scenes' / scene / 'predictions.json', previous)
        atomic_json(roots['prediction'] / 'scenes' / scene / 'predictions.json', current)
    atomic_json(roots['scoring'] / 'run_binding.json', {'input_lock_sha256': 'model-digest', 'producer': 'dinov2-small'})
    seal(roots['scoring'], list((roots['scoring'] / 'scenes').glob('*.json')), name='feature_lock.json')
    seal(roots['scoring'], [p for p in roots['scoring'].rglob('*') if p.is_file()], name='prediction_lock.json')
    atomic_json(roots['prepared'] / 'input_manifest.json', {
        'scenes': scenes, 'scene_count': 100, 'view_sets': schedules, 'anomaly_labels_opened': False,
        'scoring_prediction_lock_sha256': digest_file(roots['scoring'] / 'prediction_lock.json'),
        'scoring_feature_lock_sha256': digest_file(roots['scoring'] / 'feature_lock.json'),
        'model_input_lock_sha256': 'model-digest'})
    seal(roots['prepared'], [roots['prepared'] / 'input_manifest.json'], metadata={'anomaly_labels_opened': False})
    atomic_json(roots['evaluator'] / 'index.json', {'scenes': scenes, 'model_lock_sha256': 'model-digest'})
    seal(roots['evaluator'], [p for p in roots['evaluator'].rglob('*') if p.is_file()])
    previous_binding = {'schema': bank_protocol['schema'], 'protocol': bank_protocol,
        'prepared_lock_sha256': digest_file(roots['prepared'] / 'artifact_lock.json'),
        'scoring_lock_sha256': digest_file(roots['scoring'] / 'prediction_lock.json'), 'feature_lock_sha256': 'features-digest'}
    atomic_json(roots['previous'] / 'run_binding.json', previous_binding)
    seal(roots['previous'], [p for p in roots['previous'].rglob('*') if p.is_file()], name='prediction_lock.json',
         metadata={'scenes': scenes, 'binding_sha256': digest(previous_binding)})
    mapping_binding = {'schema': mapping_protocol['schema'], 'protocol': mapping_protocol,
        'prepared_lock_sha256': digest_file(roots['prepared'] / 'artifact_lock.json'),
        'scoring_lock_sha256': digest_file(roots['scoring'] / 'prediction_lock.json'), 'feature_lock_sha256': 'features-digest',
        'previous_prediction_lock_sha256': digest_file(roots['previous'] / 'prediction_lock.json')}
    atomic_json(roots['prediction'] / 'run_binding.json', mapping_binding)
    seal(roots['prediction'], [p for p in roots['prediction'].rglob('*') if p.is_file()], name='prediction_lock.json',
         metadata={'scenes': scenes, 'binding_sha256': digest(mapping_binding)})
    return roots


def preflight(roots):
    return ev._preflight(roots['prediction'], roots['previous'], roots['prepared'], roots['scoring'], roots['evaluator'])


def run(roots):
    return ev.evaluate(roots['prediction'], roots['previous'], roots['prepared'], roots['scoring'],
                       roots['evaluator'], roots['evaluation'], Path(__file__).resolve().parents[1])


def reseal(root):
    lock = read_json(root / 'prediction_lock.json')
    lock['metadata']['binding_sha256'] = digest(read_json(root / 'run_binding.json'))
    seal(root, [root / path for path in lock['files']], name='prediction_lock.json', metadata=lock['metadata'])


def test_unmocked_preflight_verifies_old_and_new_seals_without_parsing_truth(sealed_inputs, monkeypatch):
    original = ev.read_json
    def guarded(path):
        if Path(path).is_relative_to(sealed_inputs['evaluator'] / 'scenes'):
            raise AssertionError('Preflight must not parse anomaly labels')
        return original(path)
    monkeypatch.setattr(ev, 'read_json', guarded)
    monkeypatch.setattr(bank, 'read_json', guarded)
    assert preflight(sealed_inputs)['scene_count'] == 100


def test_unmocked_preflight_rejects_cross_binding_and_changed_baseline(sealed_inputs):
    root = sealed_inputs['prediction']
    path = root / 'run_binding.json'
    binding = read_json(path)
    binding['feature_lock_sha256'] = 'wrong-features'
    atomic_json(path, binding)
    reseal(root)
    with pytest.raises(ValueError, match='features differ'):
        preflight(sealed_inputs)
    binding['feature_lock_sha256'] = 'features-digest'
    atomic_json(path, binding)
    scene_path = root / 'scenes/S000000/predictions.json'
    payload = read_json(scene_path)
    payload['settings'][0]['rankings']['existing_all_candidates'][0]['score'] = .01
    atomic_json(scene_path, payload)
    reseal(root)
    with pytest.raises(ValueError, match='baseline changed'):
        preflight(sealed_inputs)


def test_complete_evaluation_keeps_denominators_private_boundaries_and_exact_resume(sealed_inputs):
    receipt = run(sealed_inputs)
    assert receipt['scene_count'] == 100 and receipt['asset_group_count'] == 27
    assert receipt['selected_method'] == 'winner_cluster_consensus'
    assert receipt['cohort_anomaly_objects'] == 100 and receipt['cohort_normal_objects'] == 200
    root = sealed_inputs['evaluation']
    lock = verify_seal(root, name='evaluation_lock.json')
    assert set(f'shareable/{name}' for name in ev.SHAREABLE_FILES) <= set(lock['files'])
    before = {str(path): (path.stat().st_mtime_ns, digest_file(path)) for path in root.rglob('*') if path.is_file()}
    assert run(sealed_inputs) == receipt
    after = {str(path): (path.stat().st_mtime_ns, digest_file(path)) for path in root.rglob('*') if path.is_file()}
    assert before == after
    frame = pd.read_csv(root / 'shareable/metric_summary.csv')
    assert set(frame['scene_count']) == {100} and set(frame['mean']) == {1.}
    for path in (root / 'shareable').iterdir():
        text = path.read_text()
        assert 'S000000' not in text and 'A00' not in text and str(sealed_inputs['evaluator']) not in text


def test_report_failure_preserves_scientific_scene_checkpoints(sealed_inputs, monkeypatch):
    original = bank._metric_summary
    monkeypatch.setattr(bank, '_metric_summary', lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic report failure')))
    with pytest.raises(RuntimeError, match='synthetic report'):
        run(sealed_inputs)
    root = sealed_inputs['evaluation']
    assert len(list((root / 'private/scenes').glob('*/metrics.json'))) == 100
    assert read_json(root / 'run_status.json')['status'] == 'failed'
    assert not (root / 'evaluation_lock.json').exists()
    monkeypatch.setattr(bank, '_metric_summary', original)
    monkeypatch.setattr(ev, 'evaluate_setting', lambda *args: (_ for _ in ()).throw(AssertionError('completed scene must be reused')))
    assert run(sealed_inputs)['scientific_status'] == 'complete'


def test_corrupted_checkpoint_stops_instead_of_recomputing(sealed_inputs, monkeypatch):
    monkeypatch.setattr(bank, '_metric_summary', lambda *args: (_ for _ in ()).throw(RuntimeError('report failure')))
    with pytest.raises(RuntimeError):
        run(sealed_inputs)
    path = sealed_inputs['evaluation'] / 'private/scenes/S000000/metrics.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        run(sealed_inputs)


def test_changed_evaluation_binding_cannot_resume(sealed_inputs):
    run(sealed_inputs)
    path = sealed_inputs['evaluation'] / 'run_binding.json'
    binding = read_json(path)
    binding['protocol']['matching']['maximum_cosine_distance'] = .9
    atomic_json(path, binding)
    with pytest.raises(ValueError, match='different inputs'):
        run(sealed_inputs)
