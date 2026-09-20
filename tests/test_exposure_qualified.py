"""Metadata-only checks of explicit exposure qualification, never outcome access."""
import json
from pathlib import Path
import pytest
from vpi_next_evaluation.exposure import verify_exposure
from vpi_object_bank.storage import read_json,atomic_json,seal
ROOT=Path(__file__).resolve().parents[1]


def test_actual_original_administrative_audit_is_qualified_not_untouched():
    audit=ROOT/'.local/exposure-qualified-original2000'
    if not audit.exists():pytest.skip('Private administrative audit is deliberately outside repository source')
    expected={'development':(1000,273,0),'calibration':(600,163,163),'internal_lock':(400,109,109)}
    for role,(scenes,assets,unknown) in expected.items():
        receipt=verify_exposure(role,audit)
        assert (receipt['scene_count'],receipt['asset_group_count'],receipt['unknown_history_asset_groups'])==(scenes,assets,unknown)
        assert receipt['independent_confirmation'] is False and receipt['untouched_assets_claim'] is False


def test_qualified_policy_rejects_fake_independent_claim(tmp_path):
    atomic_json(tmp_path/'exposure_ledger.json',{'schema':'vpi-asset-exposure-qualified-audit/v2','status':'reviewed_with_uncertainty','evaluation_policy':'exposure_qualified_fixed_methods','no_retuning':True,'independent_confirmation':True,'untouched_assets_claim':False})
    atomic_json(tmp_path/'asset_records.json',[]);seal(tmp_path,[tmp_path/'exposure_ledger.json',tmp_path/'asset_records.json'])
    with pytest.raises(ValueError,match='independent'):
        verify_exposure('calibration',tmp_path)


def _official_geometry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import vpi_next_evaluation.exposure as exposure
    from visual_peer_inspector.data.model_safe_package import membership_sha256
    from vpi_object_bank.storage import digest_file
    indexes = {'seen': {'S001': 'asset-a', 'S002': 'asset-a'},
               'unseen': {'U001': 'asset-b', 'U002': 'asset-c'}}
    contracts = {role: SimpleNamespace(scene_count=len(index), asset_group_count=len(set(index.values())),
        model_membership_sha256=membership_sha256(index), membership_sha256=role+'-fixed-source-membership')
        for role, index in indexes.items()}
    monkeypatch.setattr(exposure, '_official_contracts', lambda: contracts)
    roots = {}
    for role, index in indexes.items():
        root = tmp_path / role
        atomic_json(root / 'scene_index.json', index)
        atomic_json(root / 'geometry_manifest.json', {'schema': 'vpi-current-reference-geometry/v1',
            'role': role, 'scene_count': contracts[role].scene_count,
            'asset_group_count': contracts[role].asset_group_count,
            'membership_sha256': contracts[role].membership_sha256, 'anomaly_labels_opened': False,
            'asset_index_sha256': digest_file(root / 'scene_index.json')})
        seal(root, [root/'scene_index.json', root/'geometry_manifest.json'])
        roots[role] = root
    return {'index_roots': roots, 'output_root': tmp_path/'audit',
        'policy': 'exposure_qualified_fixed_methods'}


def test_official_qualified_builder_retains_unknown_history_and_resumes(tmp_path, monkeypatch):
    from vpi_next_evaluation.exposure import prepare_official_exposure
    from vpi_next_evaluation.official import verify_official_exposure
    cfg = _official_geometry(tmp_path, monkeypatch)
    receipt = prepare_official_exposure(cfg)
    assert receipt == prepare_official_exposure(cfg) == verify_official_exposure(cfg['output_root'])
    assert receipt['official_roles']['seen']['unknown_history_asset_groups'] == 1
    assert receipt['official_roles']['unseen']['unknown_history_asset_groups'] == 2
    assert receipt['independent_confirmation'] is False and receipt['untouched_claim'] is False
    assert all(row['prior_access'] == [] and row['potentially_exposed']
               for row in read_json(cfg['output_root']/'asset_records.json'))


def test_official_qualified_builder_blocks_wrong_membership_before_output(tmp_path, monkeypatch):
    from vpi_next_evaluation.exposure import prepare_official_exposure
    from vpi_object_bank.storage import digest_file
    cfg = _official_geometry(tmp_path, monkeypatch)
    source = cfg['index_roots']['seen']
    atomic_json(source/'scene_index.json', {'replaced': 'asset-a', 'S002': 'asset-a'})
    manifest = read_json(source/'geometry_manifest.json')
    manifest['asset_index_sha256'] = digest_file(source/'scene_index.json')
    atomic_json(source/'geometry_manifest.json', manifest)
    seal(source, [source/'scene_index.json', source/'geometry_manifest.json'])
    with pytest.raises(ValueError, match='membership changed'):
        prepare_official_exposure(cfg)
    assert not cfg['output_root'].exists()


def test_official_qualified_policy_cannot_forge_no_access(tmp_path, monkeypatch):
    from vpi_next_evaluation.exposure import prepare_official_exposure, verify_qualified_official_exposure
    cfg = _official_geometry(tmp_path, monkeypatch)
    prepare_official_exposure(cfg)
    root = cfg['output_root']
    rows = read_json(root/'asset_records.json')
    rows[0]['potentially_exposed'] = False
    atomic_json(root/'asset_records.json', rows)
    seal(root, [root/'asset_records.json', root/'official_exposure_ledger.json',
        *(root/'indexes').glob('*/scene_index.json')])
    with pytest.raises(ValueError, match='no-access'):
        verify_qualified_official_exposure(root)
