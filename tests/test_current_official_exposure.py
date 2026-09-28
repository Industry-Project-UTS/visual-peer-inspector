"""The official audit accepts the geometry producer's sealed label flags."""
from types import SimpleNamespace

import pytest


def _inputs(tmp_path, monkeypatch):
    import current_official_exposure as adapter
    import vpi_next_evaluation.exposure as exposure
    from visual_peer_inspector.data.model_safe_package import membership_sha256
    from vpi_object_bank.storage import atomic_json, digest_file, seal

    indexes = {'seen': {'S1': 'asset-a'}, 'unseen': {'U1': 'asset-b'}}
    contracts = {role: SimpleNamespace(
        scene_count=1, asset_group_count=1,
        model_membership_sha256=membership_sha256(index),
        membership_sha256=f'{role}-source-membership')
        for role, index in indexes.items()}
    monkeypatch.setattr(adapter, '_official_contracts', lambda: contracts)
    monkeypatch.setattr(exposure, '_official_contracts', lambda: contracts)
    roots = {}
    for role, index in indexes.items():
        root = tmp_path / role
        atomic_json(root / 'scene_index.json', index)
        atomic_json(root / 'geometry_manifest.json', {
            'schema': 'vpi-current-reference-geometry/v1', 'role': role,
            'scene_count': 1, 'asset_group_count': 1,
            'membership_sha256': contracts[role].membership_sha256,
            'asset_index_sha256': digest_file(root / 'scene_index.json'),
            'anomaly_outcomes_used': False, 'anomaly_labels_exported': False,
            'trusted_preparer_only': True,
            'reference_box_source': 'per_object_segmentation_masks',
        })
        seal(root, [root / 'scene_index.json', root / 'geometry_manifest.json'])
        roots[role] = root
    return {'index_roots': roots, 'output_root': tmp_path / 'audit',
            'policy': 'exposure_qualified_fixed_methods'}


def test_current_official_exposure_accepts_actual_geometry_schema(tmp_path, monkeypatch):
    from current_official_exposure import prepare_official_exposure_from_geometry
    from vpi_object_bank.storage import read_json
    cfg = _inputs(tmp_path, monkeypatch)
    receipt = prepare_official_exposure_from_geometry(cfg)
    assert receipt == prepare_official_exposure_from_geometry(cfg)
    assert receipt['official_roles']['seen']['scene_count'] == 1
    assert receipt['official_roles']['unseen']['unknown_history_asset_groups'] == 1
    assert read_json(cfg['output_root'] / 'official_exposure_ledger.json')['no_retuning'] is True


@pytest.mark.parametrize('field', ['anomaly_outcomes_used', 'anomaly_labels_exported'])
def test_current_official_exposure_rejects_label_access(tmp_path, monkeypatch, field):
    from current_official_exposure import prepare_official_exposure_from_geometry
    from vpi_object_bank.storage import atomic_json, read_json, seal
    cfg = _inputs(tmp_path, monkeypatch)
    root = cfg['index_roots']['seen']
    manifest = read_json(root / 'geometry_manifest.json')
    manifest[field] = True
    atomic_json(root / 'geometry_manifest.json', manifest)
    seal(root, [root / 'scene_index.json', root / 'geometry_manifest.json'])
    with pytest.raises(ValueError, match='neutral geometry'):
        prepare_official_exposure_from_geometry(cfg)
    assert not cfg['output_root'].exists()
