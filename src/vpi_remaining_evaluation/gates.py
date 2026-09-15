"""Explicit protected-role prerequisites; this module never reads anomaly labels."""
from pathlib import Path
import os

from visual_peer_inspector.data.cohort_contract import (
    DEVELOPMENT_FULL_CONTRACT, CALIBRATION_CONTRACT, INTERNAL_LOCK_CONTRACT,
)
from vpi_object_bank.storage import read_json, verify_seal

CONTRACTS = {'development': DEVELOPMENT_FULL_CONTRACT, 'calibration': CALIBRATION_CONTRACT,
             'internal_lock': INTERNAL_LOCK_CONTRACT}
STAGE_ROLES = {'09a': 'calibration', '09b': 'calibration', '09b1': 'internal_lock',
               '09c': 'internal_lock', '09d': 'internal_lock'}
LEDGER_SCHEMA = 'vpi-asset-exposure-audit/v1'


class PrerequisiteBlocked(RuntimeError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f'{code}: {detail}')


def verify_exposure(role, exposure_root):
    if role not in CONTRACTS:
        raise ValueError('Unknown original role')
    if exposure_root is None:
        raise PrerequisiteBlocked('EXPOSURE_LEDGER_MISSING',
            'Set VPI_EXPOSURE_ROOT to a sealed audit containing exposure_ledger.json and asset_records.json; '
            'reconcile earlier asset access before opening protected resources.')
    root = Path(exposure_root)
    required = {'exposure_ledger.json', 'asset_records.json'}
    if not (root / 'artifact_lock.json').is_file() or any(not (root / name).is_file() for name in required):
        raise PrerequisiteBlocked('EXPOSURE_LEDGER_INCOMPLETE', 'The exposure audit needs its seal, ledger and asset records.')
    lock = verify_seal(root)
    if not required <= set(lock['files']):
        raise ValueError('Exposure seal omits required audit records')
    ledger = read_json(root / 'exposure_ledger.json')
    if ledger.get('schema') != LEDGER_SCHEMA or ledger.get('status') != 'resolved':
        raise PrerequisiteBlocked('EXPOSURE_UNRESOLVED', 'The original asset-access audit is not resolved.')
    sources = ledger.get('reviewed_source_sha256', [])
    if not sources or any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s) for s in sources):
        raise ValueError('Exposure audit requires nonempty reviewed source digests')
    rows = read_json(root / 'asset_records.json')
    if not rows or any(set(row) != {'asset_group', 'role', 'scene_count', 'resolved', 'prior_access'} for row in rows):
        raise ValueError('Exposure asset records have an invalid or empty schema')
    keys = [(r['role'], r['asset_group']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate asset records in exposure audit')
    # All original roles must remain explicit; an empty protected subset cannot pass.
    for original_role, contract in CONTRACTS.items():
        cohort = [r for r in rows if r['role'] == original_role]
        if len(cohort) != contract.asset_group_count or sum(r['scene_count'] for r in cohort) != contract.scene_count:
            raise ValueError('Exposure audit changed original role cardinalities')
        recorded = ledger.get('roles', {}).get(original_role, {})
        if recorded.get('membership_sha256') != contract.membership_sha256:
            raise ValueError('Exposure ledger changed original role membership')
        if any(r['resolved'] is not True for r in cohort):
            raise PrerequisiteBlocked('EXPOSURE_UNRESOLVED', 'At least one original asset has unresolved prior access.')
        for row in cohort:
            for event in row['prior_access']:
                if set(event) != {'kind', 'source_sha256'} or event['kind'] not in {
                        'rgb', 'predictions', 'reference_identities', 'anomaly_labels', 'evaluation_results'}:
                    raise ValueError('Exposure event schema is invalid')
                if event['source_sha256'] not in sources:
                    raise ValueError('Exposure event does not link a reviewed source')
    selected = [r for r in rows if r['role'] == role]
    if role != 'development' and any(event['kind'] in {'anomaly_labels', 'evaluation_results'}
                                    for row in selected for event in row['prior_access']):
        raise PrerequisiteBlocked('PROTECTED_ASSETS_ALREADY_EXPOSED',
            'Prior outcome access prevents an untouched-role claim; reconcile the confirmation protocol without reshuffling roles.')
    return {'role': role, 'status': 'exposure_audited', 'scene_count': CONTRACTS[role].scene_count,
            'asset_group_count': CONTRACTS[role].asset_group_count}


def require_stage_adapter(stage, exposure_root=None):
    """Called before every legacy 09 stage can reach GPU or evaluator setup."""
    stage = stage.lower()
    if stage not in STAGE_ROLES:
        raise ValueError('Unknown protected notebook stage')
    verify_exposure(STAGE_ROLES[stage], exposure_root or os.getenv('VPI_EXPOSURE_ROOT'))
    raise PrerequisiteBlocked('CURRENT_METHOD_ADAPTER_REQUIRED',
        f'{stage.upper()} retains the 08F/P1 artifact contract. Its body cannot run winner-consensus or '
        'the assisted object-bank-top2 method. Use 09E resource_recipe.json to implement and verify the '
        'current prediction/evaluation bridge before replacing this guard. No protected labels were opened.')


def readiness(exposure_root=None):
    rows = []
    for role, contract in CONTRACTS.items():
        try:
            status = verify_exposure(role, exposure_root)
            exposure = status['status']
            detail = 'Current-method protected adapters still require verification.'
        except PrerequisiteBlocked as error:
            exposure, detail = error.code, str(error)
        rows.append({'role': role, 'scene_count': contract.scene_count, 'asset_group_count': contract.asset_group_count,
                     'membership_sha256': contract.membership_sha256, 'exposure_status': exposure,
                     'current_method_adapter': 'not_implemented_for_full_role', 'execution_ready': False,
                     'action': detail})
    return rows
