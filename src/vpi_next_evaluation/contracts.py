"""Current runtime freeze and protocol receipts, separate from executed 09E source."""
from pathlib import Path

from vpi_object_bank.protocol import PROTOCOL
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal
from vpi_remaining_evaluation.workbench import runtime_versions

SCHEMA = 'vpi-current-two-mode-freeze/v1'
RECEIPT_SCHEMA = 'vpi-current-stage-receipt/v1'
METHODS = {'automatic': 'winner_cluster_consensus', 'assisted': 'object_bank_top2'}
VIEW_SETS = {name: list(views) for name, views in PROTOCOL['view_sets'].items() if len(views) in (1, 2, 3, 5)}
METRICS = tuple(PROTOCOL['metrics'])


def current_method_contract():
    return {'schema': SCHEMA, 'methods': dict(METHODS), 'view_sets': dict(VIEW_SETS),
        'view_roles': {'1': 'diagnostic', '2': 'core', '3': 'core', '5': 'core'},
        'region_route': 'predicted LocateAnything and SAM regions',
        'assisted_provenance': ['dataset_reference', 'user_confirmed'],
        'automatic': {'selector': 'forward/P1/P2 percentile consensus', 'association_distance': 0.20},
        'assisted': {'scorer': 'foreground patch bank', 'target_identity_exclusion': 'all selected views',
                     'tail_fraction': 0.02, 'normalization': 'within-view percentile', 'fusion': 'top2'},
        'metrics': list(METRICS), 'bootstrap': dict(PROTOCOL['bootstrap']),
        'no_retuning': True, 'selection_closed': True,
        'claim': 'relative object ranking; no calibrated probability or independent confirmation implied'}


def runtime_source_files(source_root):
    root = Path(source_root)
    directories = ('src/visual_peer_inspector', 'src/vpi_object_bank', 'src/vpi_remaining_evaluation', 'src/vpi_next_evaluation')
    files = {p.relative_to(root).as_posix() for directory in directories for p in (root / directory).rglob('*.py')}
    files -= {'src/vpi_next_evaluation/reporting.py', 'src/vpi_next_evaluation/official_reporting.py',
        'src/vpi_next_evaluation/diagnostic_figures.py'}
    files |= {'src/visual_peer_inspector/web/decision.py', 'src/visual_peer_inspector/web/pipeline_config.json',
        'src/visual_peer_inspector/workflows/learned_peer_features.py',
        'src/visual_peer_inspector/workflows/learned_peer_evaluation.py',
        'src/visual_peer_inspector/workflows/comparison_io.py', 'src/vpi_next_evaluation/contracts.py'}
    return {name: digest_file(root / name) for name in sorted(files)}


def prepare_method_freeze(reference_root, output_root, source_root):
    """Bind the already selected methods to the exact current CPU decision runtime."""
    reference_root, root = Path(reference_root), Path(output_root)
    lock = verify_seal(reference_root, name='publication_lock.json')
    relative = 'tables/current_method_freeze.json'
    if relative not in lock['files']:
        raise ValueError('Expected the completed 09E publication bundle')
    old = read_json(reference_root / relative)
    if old.get('automatic_method') != METHODS['automatic'] or old.get('assisted_method') != METHODS['assisted']:
        raise ValueError('09E reference does not contain the selected methods')
    record = {**current_method_contract(), 'source_sha256': runtime_source_files(source_root),
        'reference_publication_sha256': digest_file(reference_root / 'publication_lock.json'),
        'runtime_versions': runtime_versions()}
    if (root / 'artifact_lock.json').exists():
        existing = verify_current_freeze(root, source_root)
        if existing != record:
            raise ValueError('Existing freeze differs; preserve it and select a fresh output directory')
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError('A nonempty freeze directory cannot be overwritten')
    atomic_json(root / 'method_freeze.json', record)
    seal(root, [root / 'method_freeze.json'])
    return record


def verify_current_freeze(freeze_root, source_root=None):
    root = Path(freeze_root)
    lock = verify_seal(root)
    if set(lock['files']) != {'method_freeze.json'}:
        raise ValueError('Current method freeze must contain its exact sealed record')
    record = read_json(root / 'method_freeze.json')
    for key, expected in current_method_contract().items():
        if record.get(key) != expected:
            raise ValueError('Current method freeze changed methods, schedules or evaluation rules')
    if not record.get('source_sha256') or not record.get('reference_publication_sha256'):
        raise ValueError('Current method freeze omits its source or development decision binding')
    if source_root is not None and record['source_sha256'] != runtime_source_files(source_root):
        raise ValueError('Current decision runtime differs from the frozen source tree')
    if record.get('runtime_versions') != runtime_versions():
        raise ValueError('Current scientific package versions differ from the freeze')
    return record


def require_protocol_receipt(receipt_root, stage, freeze_sha256):
    root = Path(receipt_root)
    lock = verify_seal(root, name='protocol_lock.json')
    if not {'stage_receipt.json', 'evaluation_lock.json'} <= set(lock['files']):
        raise ValueError('Protocol lock omits the stage receipt or actual evaluation seal')
    receipt = read_json(root / 'stage_receipt.json')
    if receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('stage') != stage.lower() or receipt.get('status') != 'protocol_complete':
        raise ValueError('Expected a completed current-method stage receipt')
    if receipt.get('method_freeze_sha256') != freeze_sha256 or receipt.get('no_retuning') is not True:
        raise ValueError('Stage receipt belongs to a different method or retuned run')
    if not receipt.get('exposure_lock_sha256') or not receipt.get('prediction_lock_sha256') or not receipt.get('evaluation_lock_sha256'):
        raise ValueError('Stage receipt omits its exposure, prediction or evaluation binding')
    if receipt.get('modes') != ['automatic', 'assisted'] or not receipt.get('roles'):
        raise ValueError('Stage receipt omits separately evaluated current modes and role')
    expected_role = {'09b':'calibration','09d':'internal_lock'}.get(stage.lower())
    if expected_role is None or receipt['roles'] != [expected_role]:
        raise ValueError('Protocol stage does not certify the required original role')
    evaluation = verify_seal(root, name='evaluation_lock.json')
    if not {'run_binding.json', 'shareable/evaluation_receipt.json'} <= set(evaluation['files']):
        raise ValueError('Evaluation seal omits completed scientific provenance')
    if receipt['evaluation_lock_sha256'] != digest_file(root / 'evaluation_lock.json'):
        raise ValueError('Protocol receipt names a different evaluation seal')
    binding = read_json(root / 'run_binding.json')
    for key in ('method_freeze_sha256','exposure_lock_sha256','prediction_lock_sha256'):
        if binding.get(key) != receipt[key]:
            raise ValueError('Protocol receipt differs from the scientific input binding')
    completed = read_json(root / 'shareable/evaluation_receipt.json')
    if binding.get('stage') != stage.lower() or binding.get('role') != expected_role or completed.get('role') != expected_role:
        raise ValueError('Completed evaluation does not match the protocol stage and role')
    if completed.get('status') != 'evaluation_complete' or completed.get('methods') != METHODS or completed.get('selection_closed') is not True:
        raise ValueError('Protocol receipt is not backed by completed fixed-method evaluation')
    return receipt
