"""The official neutral geometry includes every physical object, even if masks are absent."""
import json
from types import SimpleNamespace

import pytest


def test_official_geometry_uses_object_masks_and_marks_missing(tmp_path, monkeypatch):
    import current_official_geometry as geometry

    source = {'scene_id': 'source-1', 'object_count': 3,
        'objects': [{'object_index': i} for i in range(3)],
        'archive_members': {'object_segmentation_directory': 'scene/segmentations'}}
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'toysad8k_scene_master.json').write_text(json.dumps({'scenes': [source]}))
    row = {'source_scene_id': 'source-1', 'source_base_asset_id': 'asset-1',
        'neutral': {'object_ids': ['O00', 'O01', 'O02'],
                    'view_ids': ['V00', 'V01']}}
    monkeypatch.setattr(geometry, '_index', lambda cfg: ({'S1': row}, 'a' * 64, {}))
    monkeypatch.setattr(geometry, '_contract', lambda role: SimpleNamespace(
        asset_group_count=1, membership_sha256='b' * 64))
    monkeypatch.setattr(geometry, 'verify_official_membership',
        lambda master, contract: master['scenes'])
    monkeypatch.setattr(geometry, 'OFFICIAL', {'seen': object()})

    missing = geometry.mask_member('scene/segmentations', 2, 1)

    class Archive:
        def __init__(self, *args):
            self.database = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            self.name = params[0]
            return self

        def fetchall(self):
            return [] if self.name == missing else [(self.name,)]

        def read(self, name):
            return name.encode()

    monkeypatch.setattr(geometry, 'IndexedArchive', Archive)
    monkeypatch.setattr(geometry, '_mask_box',
        lambda content: [1, 2, 10, 20])
    cfg = {'role': 'seen', 'raw_root': str(raw), 'output_root': str(tmp_path / 'output')}
    manifest = geometry.prepare_official_mask_geometry(cfg)
    data = json.loads((tmp_path / 'output/scenes/S1/geometry.json').read_text())
    assert manifest['scene_count'] == 1
    assert manifest['missing_mask_count'] == 1
    assert len(data['views']) == 2
    assert all(set(view) == {'O00', 'O01', 'O02'} for view in data['views'].values())
    assert data['views']['V01']['O02'] == [0, 0, 0, 0]
    assert data['views']['V00']['O02'] == [1, 2, 10, 20]
    assert geometry.prepare_official_mask_geometry(cfg) == manifest


def test_official_mask_member_naming_and_invalid_index():
    from current_official_geometry import mask_member

    assert mask_member('scene/segmentations', 0, 0).endswith('/object/object_0000.png')
    assert mask_member('scene/segmentations', 3, 19).endswith('/object.003/object.003_0019.png')
    with pytest.raises(ValueError):
        mask_member('scene/segmentations', -1, 0)
