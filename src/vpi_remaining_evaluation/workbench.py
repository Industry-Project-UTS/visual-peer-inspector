"""Seal existing development evidence and concrete remaining-cohort resource recipes."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import pandas as pd

from vpi_object_bank import evaluation as bank_evaluation
from vpi_object_bank.reporting import verified_report_inputs
from vpi_object_bank.storage import atomic_json, bind, digest_file, read_json, seal, source_hashes, verify_seal
from .gates import readiness

SCHEMA = 'vpi-remaining-evaluation-workbench/v1'
TABLES = ('metric_summary.csv', 'view_schedule_summary.csv', 'paired_asset_intervals.csv',
          'anomaly_count_summary.csv', 'majority_summary.csv', 'failure_summary.csv', 'identity_summary.csv')
SOURCES = ('src/vpi_remaining_evaluation/workbench.py', 'src/vpi_remaining_evaluation/gates.py',
           'src/vpi_remaining_evaluation/cohort.py', 'src/vpi_object_bank/storage.py', 'src/vpi_object_bank/reporting.py')


def runtime_versions():
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'pandas', 'matplotlib', 'Pillow'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'unavailable'
    return versions


def _decorate(frame, stage):
    frame = frame.copy()
    frame.insert(0, 'source_stage', stage)
    frame['correspondence'] = frame.apply(lambda r: 'supplied' if str(r['route']).endswith('_reference')
                                         or r['method'] == 'correspondence_assisted_reference' else 'predicted', axis=1)
    frame['regions'] = frame['route'].map(lambda r: 'supplied masks' if r == 'supplied_reference' else 'predicted masks')
    frame['report_status'] = frame.apply(lambda r: 'frozen automatic baseline' if r['method'] == 'winner_cluster_consensus'
        else 'frozen assisted method' if r['correspondence'] == 'supplied' and r['method'] in {'object_bank_top2', 'correspondence_assisted_reference'}
        else 'development comparison; not selected', axis=1)
    # Keep the old sealed result inspectable while excluding a known bookkeeping
    # defect from usable comparisons. The inherited defaultdict lookup inflates it.
    frame['metric_validity'] = 'usable as reported'
    frame['original_reported_mean'] = frame['mean'] if 'mean' in frame else None
    invalid = frame['metric'].eq('identity_reference_coverage') if 'metric' in frame else pd.Series(False, index=frame.index)
    frame.loc[invalid, 'metric_validity'] = 'invalid inherited coverage accounting; not evidence of full identity coverage'
    if 'mean' in frame:
        frame.loc[invalid, 'mean'] = float('nan')
    return frame


def resolve_input_root(path, marker):
    root = Path(path).expanduser()
    if (root / marker).is_file():
        return root
    matches = set()
    # Mounted flat or automatically extracted nested packages, without scanning media.
    for depth in range(1, 5):
        matches.update(p.parent for p in root.glob('*/' * depth + marker) if p.is_file())
    if len(matches) != 1:
        raise ValueError(f'Expected one package with {marker}; found {len(matches)}')
    return next(iter(matches))


def resolved_config(cfg):
    result = {**cfg, 'bank': dict(cfg['bank']), 'mapping': dict(cfg['mapping'])}
    markers = {'model_root': 'artifact_lock.json', 'prepared_root': 'artifact_lock.json',
               'feature_root': 'feature_lock.json', 'prediction_root': 'prediction_lock.json',
               'evaluation_root': 'evaluation_lock.json', 'scoring_root': 'prediction_lock.json'}
    for stage in ('bank', 'mapping'):
        for key in result[stage]:
            if key in markers:
                result[stage][key] = str(resolve_input_root(result[stage][key], markers[key]))
    return result


def verify_inputs(cfg):
    hashes = verified_report_inputs(cfg['bank'])
    mapping = cfg['mapping']
    for key, name in (('prediction_root', 'prediction_lock.json'), ('evaluation_root', 'evaluation_lock.json')):
        verified = verify_seal(Path(mapping[key]), name=name)
        if 'run_binding.json' not in verified['files']:
            raise ValueError('Mapping evidence seal omits its binding')
    prediction = read_json(Path(mapping['prediction_root']) / 'run_binding.json')
    evaluation = read_json(Path(mapping['evaluation_root']) / 'run_binding.json')
    mapping_hash = digest_file(Path(mapping['prediction_root']) / 'prediction_lock.json')
    comparisons = [(prediction['previous_prediction_lock_sha256'], hashes['prediction_root']),
                   (prediction['prepared_lock_sha256'], hashes['prepared_root']),
                   (prediction['feature_lock_sha256'], hashes['feature_root']),
                   (evaluation['prediction_lock_sha256'], mapping_hash),
                   (evaluation['previous_prediction_lock_sha256'], hashes['prediction_root'])]
    if any(left != right for left, right in comparisons):
        raise ValueError('Remaining workbench mixes incompatible scientific stages')
    bank_decision = read_json(Path(cfg['bank']['evaluation_root']) / 'shareable/decision.json')
    mapping_decision = read_json(Path(mapping['evaluation_root']) / 'shareable/decision.json')
    if bank_decision['selected_method'] != 'object_bank_top2' or mapping_decision['selected_method'] != 'winner_cluster_consensus':
        raise ValueError('Workbench requires the exact frozen two-mode decision')
    return {**{f'bank_{k}': v for k, v in hashes.items()},
            'mapping_prediction_root': mapping_hash,
            'mapping_evaluation_root': digest_file(Path(mapping['evaluation_root']) / 'evaluation_lock.json')}


def resource_recipe():
    return {'schema': SCHEMA, 'original_roles': {'development': 1000, 'calibration': 600, 'internal_lock': 400},
        'current_methods': {'automatic': 'winner_cluster_consensus', 'assisted': 'object_bank_top2'},
        'assisted_provenance': ['dataset_reference', 'user_confirmed'],
        'view_sets': bank_evaluation.PROTOCOL['view_sets'],
        'input_scene_contract': {
            'required': ['opaque scene ID', 'exact original role and membership hash', 'selected RGB checksums',
                         'all candidate IDs', 'original-coordinate boxes', 'SAM RLE masks and checksums',
                         'DINOv2-small global embeddings', 'forward/P1/P2 scores for winner consensus',
                         'foreground patch arrays for object bank', 'padded crop box, crop checksum and valid patch grid'],
            'assisted_only': ['opaque neutral correspondence IDs and explicit provenance; null for unknown'],
            'forbidden_model_inputs': ['anomaly labels', 'normal/anomaly filenames', 'evaluator verdicts', 'protected score-selected schedules'],
            'missing_neural_outputs': 'inventory missing items first; reuse existing boxes/masks/global features; extract only missing frozen DINO patches'},
        'stage_bridges': {
            '09A': '600 calibration scenes: current-method label-free prediction adapter, exact shards and final prediction seal',
            '09B': 'verify current prediction seal; evaluate only frozen methods; any review policy changes only within the declared calibration protocol',
            '09B1': '400 internal-lock resources: require resolved exposure and frozen current method/review policy; keep identities and labels separate',
            '09C': '400 internal-lock scenes: unchanged current-method prediction adapter; no evaluator mounted',
            '09D': 'verify all fixed prediction/method/policy/membership seals; one-shot evaluation, no method selection',
            '09E': 'reusable compatibility/readiness and rich report workbench; currently verified on existing development100'},
        'adapter_status': 'generic current-method cached prediction hook implemented; actual role cache/resource bridges and new protected evaluator adapters not implemented; legacy P1 contracts remain guarded',
        'cached_prediction_hook': {'module': 'vpi_remaining_evaluation.cohort.predict_cached_cohort', 'adapter': 'visual_peer_inspector.web.decision.decide_cached', 'schema': 'vpi-current-method-cohort-cache/v1', 'required_manifest': ['role', 'cohort_id', 'membership_sha256', 'scene_count', 'asset_group_count', 'scenes', 'method_freeze_sha256'], 'per_scene': ['scene', 'score_rows by selected view', 'bank_observations with foreground patches', 'correspondence_provenance'], 'membership_validation': 'recompute canonical sorted opaque-scene membership hash from actual scene IDs and compare with original model-facing role hash', 'asset_attribution': 'native scene-to-asset resource bridge is still unimplemented; declared asset count alone does not verify attribution'},
        'exposure_package': {'schema': 'vpi-asset-exposure-audit/v1',
            'files': ['artifact_lock.json', 'exposure_ledger.json', 'asset_records.json'],
            'ledger_fields': ['schema', 'status=resolved', 'reviewed_source_sha256 (nonempty)', 'roles with exact membership_sha256'],
            'asset_record_fields': ['asset_group', 'role', 'scene_count', 'resolved', 'prior_access: kind + source_sha256'],
            'event_kinds': ['rgb', 'predictions', 'reference_identities', 'anomaly_labels', 'evaluation_results'],
            'coverage': 'all original 273 development, 163 calibration and 109 internal-lock assets; no empty or reshuffled protected subset'},
        'report_contract': ['all registered compatible techniques stay in development tables',
            'primary schedules and scene-averaged schedules reported separately', 'tie-aware top1/AP/Recall2/MRR with asset-group intervals',
            'candidate/association/score/ranking failures retain scene denominators', 'selected-object boxes are not validated defect boundaries',
            'raw galleries and scene score matrices stay private; aggregate tables and charts may be selected for report later'],
        'known_report_issues': ['Inherited identity_reference_coverage is invalid because defaultdict lookups add absent references before its numerator is computed. Original means remain in original_reported_mean for audit, while usable means are blank. Ranking metrics and failure counts are unchanged.', 'Current app cache revalidation is a separate post-hoc runtime check; it is not merged into the historical 08O/08P conditions.'],
        'protected_selection': 'frozen automatic winner consensus and assisted top2; protected outcomes cannot promote alternatives'}


def prepare_workbench(cfg, output_root, source_root):
    root = Path(output_root)
    cfg = resolved_config(cfg)
    hashes = verify_inputs(cfg)
    exposure_root = cfg.get('exposure_root')
    exposure_hash = None if exposure_root is None else digest_file(Path(exposure_root) / 'artifact_lock.json')
    binding = {'schema': SCHEMA, 'input_locks': hashes, 'runtime_versions': runtime_versions(),
               'exposure_lock_sha256': exposure_hash, 'source': source_hashes(source_root, SOURCES)}
    bind(root, binding)
    if (root / 'scientific_lock.json').exists():
        verify_seal(root, name='scientific_lock.json')
        return read_json(root / 'shareable/workbench_receipt.json')
    tables = {name: [] for name in TABLES}
    for stage, part in [('08O', cfg['bank']), ('08P', cfg['mapping'])]:
        evaluation = Path(part['evaluation_root'])
        lock = read_json(evaluation / 'evaluation_lock.json')
        for name in TABLES:
            if f'shareable/{name}' not in lock['files']:
                raise ValueError('Scientific source does not seal a required report table')
            frame = pd.read_csv(evaluation / 'shareable' / name)
            if not set(frame['view_count']) <= {1, 2, 3, 5} or not set(frame['route']) <= {'supplied_reference', 'automatic_reference', 'automatic_predicted'}:
                raise ValueError('Unexpected report conditions')
            tables[name].append(_decorate(frame, stage))
    for name, frames in tables.items():
        bank_evaluation._atomic_csv(root / 'shareable' / name, pd.concat(frames, ignore_index=True))
    registry = pd.concat(tables['metric_summary.csv'], ignore_index=True)
    registry = registry[['source_stage', 'route', 'method', 'view_count', 'correspondence', 'regions', 'report_status']].drop_duplicates()
    registry['development_evidence'] = 'verified 100 scenes / 27 assets'
    registry['full_role_compatibility'] = 'not yet verified'
    bank_evaluation._atomic_csv(root / 'shareable/technique_feature_matrix.csv', registry)
    roles = readiness(exposure_root)
    for row in roles:
        row['current_scoring_evaluated_scenes'] = 100 if row['role'] == 'development' else 0
        row['scenes_without_current_scoring_verification'] = row['scene_count'] - row['current_scoring_evaluated_scenes']
    bank_evaluation._atomic_csv(root / 'shareable/cohort_readiness.csv', pd.DataFrame(roles))
    recipe = resource_recipe()
    atomic_json(root / 'shareable/resource_recipe.json', recipe)
    freeze = {'schema': SCHEMA, 'automatic_method': 'winner_cluster_consensus', 'assisted_method': 'object_bank_top2',
              'method_source_locks': hashes, 'view_sets': recipe['view_sets'],
              'selection_closed': True, 'new_protected_results_may_select_methods': False}
    atomic_json(root / 'shareable/current_method_freeze.json', freeze)
    receipt = {'schema': SCHEMA, 'status': 'development_evidence_prepared', 'evaluated_scenes': 100, 'asset_groups': 27,
               'original_role_scene_count': 2000, 'new_protected_evaluations': 0,
               'protected_execution_ready': False, 'neural_inference_run': False,
               'registered_condition_rows': len(registry), 'table_count': len(TABLES),
               'current_methods': {'automatic': 'winner_cluster_consensus', 'assisted': 'object_bank_top2'},
               'protected_blockers': ['resolved asset-exposure ledger', 'current-method cohort prediction/evaluation adapters'],
               'scope': 'original roles retained; actual completed evidence is the existing development100 only'}
    atomic_json(root / 'shareable/workbench_receipt.json', receipt)
    seal(root, [root / 'run_binding.json', *sorted((root / 'shareable').glob('*'))], name='scientific_lock.json',
         metadata={'schema': SCHEMA, 'scientific_status': 'development_evidence_prepared', 'protected_execution_ready': False})
    verify_seal(root, name='scientific_lock.json')
    return receipt
