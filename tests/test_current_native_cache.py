"""Native-format fixture conversion; no datasets, evaluator truth or neural model."""
from dataclasses import replace
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image
import pytest

from vpi_next_evaluation import native_cache as native
from vpi_next_evaluation.tensor_cache import current_inputs
from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, verify_seal
from vpi_object_bank.protocol import EXTRACTION
from visual_peer_inspector.data.model_safe_package import membership_sha256
from visual_peer_inspector.models.sam2 import encode_binary_mask, binary_mask_sha256
from visual_peer_inspector.representation.crops import prepare_masked_crop
from visual_peer_inspector.scoring.forward_patch import patch_policy
from visual_peer_inspector.workflows.scoring_comparison import PROTOCOL


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    scenes = ['S000001', 'S000002']; views = ['V00', 'V07', 'V14']
    contract = replace(native.CONTRACTS['development'], scene_count=2, image_count=6,
        views_per_scene=3, asset_group_count=1, membership_sha256=membership_sha256(scenes))
    monkeypatch.setattr(native, '_gates', lambda cfg, source_root: (contract, 'f' * 64))
    monkeypatch.setattr(native, 'VIEW_SETS', {'three-primary': views})
    region = tmp_path / 'regions'; region.mkdir()
    for scene in scenes:
        rows = []
        for view in views:
            rgb = np.full((64, 64, 3), 90, dtype=np.uint8)
            path = region / 'media' / scene / f'{view}.png'; path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            objects = []
            for index in range(3):
                mask = np.zeros((64, 64), dtype=bool); mask[8:24, 4+index*19:16+index*19] = True
                objects.append({'candidate_id': f'C{index}', 'box': [4+index*19, 8, 16+index*19, 24],
                    'mask': encode_binary_mask(mask), 'mask_sha256': sha256(mask.astype('uint8').tobytes()).hexdigest()})
            for route in ('supplied', 'automatic'):
                rows.append({'scene': scene, 'view': view, 'route': route, 'rgb': path.relative_to(region).as_posix(),
                    'rgb_sha256': digest_file(path), 'objects': objects})
        atomic_json(region / 'scenes' / f'{scene}.json', rows)
    atomic_json(region / 'index.json', {'schema': 'fixture', 'study': PROTOCOL['study'],
        'scenes': scenes, 'smoke_scenes': scenes[:1], 'views': views, 'routes': ['supplied', 'automatic']})
    seal(region, [p for p in region.rglob('*') if p.is_file()])
    region_hash = digest_file(region / 'artifact_lock.json')
    features = tmp_path / 'features'; encoder = native.pipeline_config()['encoder']
    binding = {'schema': 'vpi-object-bank-features/v1', 'producer': 'dinov2-small', 'device': 'cpu',
        'extraction': EXTRACTION, 'model_input_lock_sha256': region_hash, 'scenes': scenes, 'views': views,
        'model_files': {'model.safetensors': encoder['sha256'], 'config.json': encoder['config_sha256']},
        'runtime': native.PINNED, 'source': {'fixture.py': 'a'*64}}
    signature = bind(features, binding)
    for scene in scenes:
        for row in read_json(region / 'scenes' / f'{scene}.json'):
            if row['route'] != 'automatic': continue
            rgb = np.asarray(Image.open(region / row['rgb']).convert('RGB'))
            arrays, metadata = {}, []
            for index, obj in enumerate(row['objects']):
                crop = prepare_masked_crop(rgb, native.decode_binary_mask(obj['mask']))
                valid, fallback = patch_policy(crop, 16, .7, fallback='all')
                vector = np.zeros(384, dtype=np.float32); vector[index] = 1.
                arrays.update({f'p{index}': np.tile(vector, (256, 1)), f'g{index}': vector, f'v{index}': valid})
                metadata.append({'candidate_id': obj['candidate_id'], 'array_index': index,
                    'crop_sha256': crop.sha256, 'mask_sha256': obj['mask_sha256'],
                    'source_box_xyxy': list(crop.source_box_xyxy), 'grid_size':16,
                    'foreground_patch_count':int(valid.sum()), 'fallback_all':fallback,
                    'padded_box_xyxy': list(crop.padded_box_xyxy)})
            with atomic_unit(features / 'views' / scene / f'automatic-{row["view"]}', signature) as unit:
                np.savez_compressed(unit / 'features.npz', **arrays)
                atomic_json(unit / 'observations.json', {'scene': scene, 'view': row['view'], 'route': 'automatic',
                    'rgb_sha256': row['rgb_sha256'], 'image_shape':list(rgb.shape), 'objects': metadata})
    seal(features, [p for p in features.rglob('*') if p.is_file()], name='feature_lock.json')
    neutral = tmp_path / 'neutral'
    atomic_json(neutral / 'neutral_correspondence.json', {'schema': native.NEUTRAL_SCHEMA,
        'region_lock_sha256': region_hash, 'provenance': 'dataset_reference',
        'observations': [{'scene': scene, 'view_id': view, 'candidate_id': f'C{i}', 'object_id': f'N{i}'}
            for scene in scenes for view in views for i in range(3)]})
    seal(neutral, [neutral / 'neutral_correspondence.json'])
    cfg = {'role': 'development', 'region_format': 'comparison-regions', 'region_root': region,
        'expected_region_lock_sha256': region_hash, 'feature_root': features, 'neutral_root': neutral,
        'output_root': tmp_path / 'cache'}
    return cfg, contract, scenes, views


def test_native_conversion_is_compact_complete_and_exact_resume(fixture):
    cfg, contract, scenes, views = fixture
    assert native.plan_native(cfg, '.')['missing_feature_units'] == []
    receipt = native.assemble_native(cfg, '.')
    assert receipt['scene_count'] == 2 and receipt['neural_inference_run'] is False
    root = cfg['output_root']; lock = verify_seal(root)
    assert len([name for name in lock['files'] if name.endswith('.npz')]) == 6
    before = digest_file(root / 'artifact_lock.json')
    payload = read_json(root / 'scenes' / f'{scenes[0]}.json')
    assert all('patches' not in row for row in payload['bank_observations'])
    rows, bank = current_inputs(payload, views[:1], 'automatic', root)
    assert set(rows) == {views[0]} and len(bank) == 3 and all(row['object_id'] is None for row in bank)
    assert all(row['patches'].dtype == np.float32 for row in bank)
    assert native.assemble_native(cfg, '.') == receipt
    assert digest_file(root / 'artifact_lock.json') == before


def test_incremental_assembly_keeps_completed_scene_and_full_denominator(fixture):
    cfg, _, scenes, _ = fixture
    assert native.assemble_native({**cfg, 'max_new_scenes': 1}, '.')['status'] == 'checkpointed'
    root = cfg['output_root']; before = digest_file(root / 'checkpoints' / scenes[0] / 'scene.json')
    assert not (root / 'artifact_lock.json').exists()
    assert native.assemble_native(cfg, '.')['scene_count'] == 2
    assert digest_file(root / 'checkpoints' / scenes[0] / 'scene.json') == before


def test_tensor_selected_view_isolation_and_corruption(fixture):
    cfg, _, scenes, views = fixture; native.assemble_native(cfg, '.')
    root = cfg['output_root']; payload = read_json(root / 'scenes' / f'{scenes[0]}.json')
    bad = root / 'tensors' / scenes[0] / f'{views[-1]}.npz'; bad.write_bytes(b'corrupt')
    assert len(current_inputs(payload, views[:1], 'assisted', root)[1]) == 3
    with pytest.raises(ValueError, match='corrupt'):
        current_inputs(payload, views, 'assisted', root)


@pytest.mark.parametrize('mutation', ['precision', 'source', 'region'])
def test_missing_or_incompatible_feature_binding_rejected(fixture, mutation):
    cfg, _, _, _ = fixture; path = cfg['feature_root'] / 'run_binding.json'; record = read_json(path)
    if mutation == 'precision': record['device'] = 'cuda:0'
    elif mutation == 'source': record.pop('source')
    else: record['model_input_lock_sha256'] = 'a'*64
    atomic_json(path, record)
    with pytest.raises(ValueError): native.plan_native(cfg, '.')


def test_explicit_missing_regions_and_output_budget(fixture):
    cfg, _, _, _ = fixture
    assert native.plan_native({**cfg, 'region_root': None}, '.')['missing_la_sam'] is True
    with pytest.raises(OSError, match='upper bound'):
        native.assemble_native({**cfg, 'maximum_output_bytes': 1}, '.')
    assert not (cfg['output_root'] / 'checkpoints').exists()


def test_neutral_labels_and_same_view_conflicts_rejected(fixture):
    cfg, _, _, _ = fixture; root = cfg['neutral_root']; path = root / 'neutral_correspondence.json'
    value = read_json(path); value['observations'][1]['object_id'] = value['observations'][0]['object_id']
    atomic_json(path, value); seal(root, [path])
    with pytest.raises(ValueError, match='twice'):
        native.assemble_native(cfg, '.')


def test_actual_native_prediction_manifest_import_ignores_p1_globals(fixture, tmp_path, monkeypatch):
    from visual_peer_inspector.workflows.frozen_development_pipeline import FrozenPipelineIdentity, lock_predictions
    from visual_peer_inspector.workflows import protected_cohort_inference
    cfg, contract, scenes, views = fixture
    root, rgb_root = tmp_path / 'native', tmp_path / 'rgb'; root.mkdir()
    records = []
    for scene in scenes:
        for row in read_json(cfg['region_root'] / 'scenes' / f'{scene}.json'):
            if row['route'] != 'automatic': continue
            target = rgb_root / 'media' / scene / 'rgb' / f'{row["view"]}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((cfg['region_root'] / row['rgb']).read_bytes())
            records.append({'cohort_id': contract.cohort_id, 'opaque_scene_id': scene, 'view_id': row['view'],
                'image_sha256': row['rgb_sha256'], 'object_count': len(row['objects']),
                'objects': [{'object_id': obj['candidate_id'], 'box_xyxy': obj['box'], 'mask_rle': obj['mask'],
                    'mask_sha256': binary_mask_sha256(native.decode_binary_mask(obj['mask'])), 'p1_score': 999, 'p1_heatmap': [[999]]} for obj in row['objects']]})
    (root / 'prediction_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    identity = FrozenPipelineIdentity(*['a'*64 for _ in range(7)])
    lock_predictions(prediction_root=root, identity=identity, cohort_id=contract.cohort_id,
        cohort_membership_sha256=contract.membership_sha256, scene_count=2, image_count=6)
    calls = []
    monkeypatch.setattr(protected_cohort_inference, 'verify_protected_input', lambda root, c: calls.append((root, c)))
    cfg = {**cfg, 'region_format': 'native-prediction-manifest', 'region_root': root, 'rgb_root': rgb_root,
        'expected_region_lock_sha256': digest_file(root / 'prediction_lock.json'), 'feature_root': tmp_path / 'missing-features'}
    rows, _ = native.native_regions(cfg, contract)
    assert calls and len(rows) == 6
    assert all(set(o) == {'candidate_id', 'box', 'mask', 'mask_sha256', 'source_mask_sha256'} for row in rows.values() for o in row['objects'])
    assert len(native.plan_native(cfg, '.')['missing_feature_units']) == 6


def test_missing_feature_extraction_uses_cpu_contract_and_resumes_without_encoder(fixture, tmp_path, monkeypatch):
    cfg, _, _, _ = fixture
    model = tmp_path / 'synthetic-model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'fixture-only'); (model / 'config.json').write_text('{}')
    encoder = {'sha256': digest_file(model / 'model.safetensors'), 'config_sha256': digest_file(model / 'config.json')}
    monkeypatch.setattr(native, 'pipeline_config', lambda: {'encoder': encoder})
    monkeypatch.setattr(native, 'compatible_runtime', lambda: True)
    monkeypatch.setattr(native, 'runtime_versions', lambda: dict(native.PINNED))
    calls = []
    class Encoder:
        def encode(self, crops):
            calls.append(len(crops))
            vectors = np.zeros((len(crops), 384), dtype=np.float32); vectors[:, 0] = 1
            return SimpleNamespace(global_features=vectors, patch_features=[np.tile(vector, (256, 1)) for vector in vectors])
    monkeypatch.setattr(native, 'model_factory', lambda **kwargs: Encoder())
    cfg = {**cfg, 'feature_root': tmp_path / 'new-features', 'model_source': model}
    assert native.extract_missing(cfg, '.')['extracted_units'] == 6
    before = len(calls)
    assert native.extract_missing(cfg, '.')['extracted_units'] == 0
    assert len(calls) == before
    first = next(cfg['feature_root'].glob('views/*/*/features.npz')); first.write_bytes(b'corrupt')
    with pytest.raises(ValueError): native.extract_missing(cfg, '.')


def test_tiny_mask_is_retained_as_typed_rejection_without_aborting_scene(fixture, tmp_path, monkeypatch):
    cfg, _, scenes, views = fixture
    region = cfg['region_root']; scene_path = region / 'scenes' / f'{scenes[0]}.json'
    rows = read_json(scene_path)
    automatic = next(row for row in rows if row['route'] == 'automatic' and row['view'] == views[0])
    rejected_id = automatic['objects'][0]['candidate_id']
    tiny = np.zeros((64, 64), dtype=bool); tiny[8, 8:11] = True
    automatic['objects'][0].update(
        box=[8, 8, 11, 9],
        mask=encode_binary_mask(tiny),
        mask_sha256=sha256(tiny.astype('uint8').tobytes()).hexdigest(),
    )
    atomic_json(scene_path, rows)
    (region / 'artifact_lock.json').unlink()
    seal(region, [p for p in region.rglob('*') if p.is_file()])
    region_hash = digest_file(region / 'artifact_lock.json')

    neutral = cfg['neutral_root']; mapping = read_json(neutral / 'neutral_correspondence.json')
    mapping['region_lock_sha256'] = region_hash
    atomic_json(neutral / 'neutral_correspondence.json', mapping)
    (neutral / 'artifact_lock.json').unlink()
    seal(neutral, [neutral / 'neutral_correspondence.json'])

    model = tmp_path / 'synthetic-model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'fixture-only'); (model / 'config.json').write_text('{}')
    encoder = {'sha256': digest_file(model / 'model.safetensors'), 'config_sha256': digest_file(model / 'config.json')}
    monkeypatch.setattr(native, 'pipeline_config', lambda: {'encoder': encoder})
    monkeypatch.setattr(native, 'compatible_runtime', lambda: True)
    class Encoder:
        def encode(self, crops):
            vectors = np.zeros((len(crops), 384), dtype=np.float32); vectors[:, 0] = 1
            return SimpleNamespace(global_features=vectors,
                patch_features=[np.tile(vector, (256, 1)) for vector in vectors])
    monkeypatch.setattr(native, 'model_factory', lambda **kwargs: Encoder())
    cfg = {**cfg, 'expected_region_lock_sha256': region_hash,
        'feature_root': tmp_path / 'tiny-mask-features', 'model_source': model}

    result = native.extract_missing(cfg, '.')
    assert result['status'] == 'features_complete'
    meta = read_json(cfg['feature_root'] / 'views' / scenes[0] / f'automatic-{views[0]}' / 'observations.json')
    assert {item['candidate_id'] for item in meta['objects']} == {'C1', 'C2'}
    assert meta['rejected_objects'] == [{
        'candidate_id': rejected_id,
        'reason': native.SMALL_MASK_REASON,
        'foreground_pixels': 3,
        'minimum_foreground_pixels': 4,
        'mask_sha256': automatic['objects'][0]['mask_sha256'],
    }]

    receipt = native.assemble_native(cfg, '.')
    assert receipt['status'] == 'cache_complete'
    payload = read_json(cfg['output_root'] / 'scenes' / f'{scenes[0]}.json')
    assert payload['invalid_candidate_observations'] == 1
    evidence = payload['evidence_views'][views[0]]['candidates'][rejected_id]
    assert evidence['feature_status'] == 'rejected'
    assert evidence['feature_failure'] == native.SMALL_MASK_REASON
    assert len(payload['bank_observations']) == len(views) * 3 - 1
    assert rejected_id not in {row['candidate_id'] for row in payload['score_rows'][views[0]]['objects']}


def test_feature_shard_resume_and_mounted_reuse_run_no_duplicate_encoder(fixture,tmp_path,monkeypatch):
    cfg,_,scenes,views=fixture
    model=tmp_path/'mock-model';model.mkdir()
    (model/'model.safetensors').write_bytes(b'fixture');(model/'config.json').write_text('{}')
    monkeypatch.setattr(native,'pipeline_config',lambda:{'encoder':{'sha256':digest_file(model/'model.safetensors'),'config_sha256':digest_file(model/'config.json')}})
    monkeypatch.setattr(native,'compatible_runtime',lambda:True)
    calls=[]
    class Encoder:
        def encode(self,crops):
            calls.append(len(crops));g=np.zeros((len(crops),384),dtype=np.float32);g[:,0]=1
            return SimpleNamespace(global_features=g,patch_features=[np.tile(v,(256,1)) for v in g])
    monkeypatch.setattr(native,'model_factory',lambda **kw:Encoder())
    seed=tmp_path/'seed-features';options={**cfg,'feature_root':seed,'model_source':model,'shard_count':2,'shard_index':0}
    first=native.extract_missing({**options,'max_new_units':1},'.')
    assert first['status']=='checkpointed' and first['extracted_units']==1
    before=digest_file(seed/'views'/scenes[0]/f'automatic-{views[0]}'/'features.npz')
    result=native.extract_missing(options,'.')
    assert result['extracted_units']==2 and result['available_selected_units']==3 and result['required_units']==6
    assert not (seed/'feature_lock.json').exists()
    assert digest_file(seed/'views'/scenes[0]/f'automatic-{views[0]}'/'features.npz')==before
    monkeypatch.setattr(native,'model_factory',lambda **kw:pytest.fail('Mounted reuse must not load DINO'))
    combined=native.prepare_shard({**options,'feature_root':tmp_path/'empty-local-features','feature_roots':[seed],
        'output_root':tmp_path/'reuse-cache'},'.')
    assert combined['status']=='shard_complete'
    assert combined['features']['extracted_units']==0 and combined['features']['neural_inference_run'] is False
