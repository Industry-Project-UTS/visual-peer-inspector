"""Merge completed native cache shards into one exact-cohort model input seal."""
from pathlib import Path
import shutil
from uuid import uuid4
from .native_cache import _gates
from vpi_object_bank.storage import atomic_json, bind, digest_file, read_json, seal, terminal_status, verify_seal
from visual_peer_inspector.data.model_safe_package import membership_sha256


def merge_cache_shards(cfg, source_root):
    contract, freeze_hash = _gates(cfg, source_root)
    manifests = []; files = {}; native_binding = None; all_scenes = []; feature_locks = set(); locks = []
    for folder in cfg['cache_shards']:
        root = Path(folder); lock = verify_seal(root, name='cache_shard_lock.json')
        verify_seal(root, name='assembly_lock.json')
        manifest = read_json(root / 'shard_manifest.json')
        if manifest.get('schema') != 'vpi-current-cache-shard/v1' or manifest['role'] != cfg['role'] or manifest['cohort_id'] != contract.cohort_id or manifest['scene_count'] != contract.scene_count or manifest['membership_sha256'] != contract.membership_sha256 or manifest['method_freeze_sha256'] != freeze_hash:
            raise ValueError('Cache shard cohort or freeze differs')
        if native_binding is not None and manifest['native_source_binding'] != native_binding:
            raise ValueError('Cache shards were produced from incompatible inputs')
        native_binding = manifest['native_source_binding']
        expected = {'shard_manifest.json'}
        for scene in manifest['scenes']:
            if scene in all_scenes: raise ValueError('Cache shards overlap; duplicate scenes cannot inflate denominator')
            all_scenes.append(scene)
            payload = read_json(root / 'scenes' / f'{scene}.json')
            if payload['scene'] != scene: raise ValueError('Shard scene payload differs')
            expected.add(f'scenes/{scene}.json')
            for view, evidence in payload['evidence_views'].items():
                expected.add(evidence['rgb_relative'])
            expected.update(row['patches_file'] for row in payload['bank_observations'])
            # Empty views still have a complete empty tensor file.
            expected.update(f'tensors/{scene}/{view}.npz' for view in payload['evidence_views'])
        if set(lock['files']) != expected:
            raise ValueError('Cache shard omits declared media/tensors or contains unexpected files')
        for relative in sorted(expected - {'shard_manifest.json'}):
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe shard path')
            if relative in files: raise ValueError('Overlapping shard file')
            files[relative] = path
        manifests.append(manifest); locks.append(digest_file(root / 'cache_shard_lock.json'))
        feature_locks.update(manifest['native_feature_locks'])
    scenes = sorted(all_scenes)
    if len(scenes) != contract.scene_count or membership_sha256(scenes) != (contract.model_membership_sha256 or contract.membership_sha256):
        raise ValueError('Shard union does not contain the complete fixed cohort')
    output = Path(cfg['output_root'])
    binding = {'schema':'vpi-current-native-cache-merge/v1', 'cache_shard_locks':sorted(locks),
        'native_source_binding':native_binding, 'source_sha256':digest_file(__file__)}
    bind(output, binding)
    if (output / 'artifact_lock.json').exists():
        verify_seal(output); verify_seal(output, name='assembly_lock.json')
        return read_json(output / 'preparation_receipt.json')
    total = sum(path.stat().st_size for path in files.values())
    if total > int(cfg.get('maximum_output_bytes',15*1024**3)) or shutil.disk_usage(output).free < sum(path.stat().st_size for relative,path in files.items() if not (output/relative).exists()) + 128*1024**2:
        raise OSError('Complete cache merge exceeds configured or available storage')
    with terminal_status(output, 'current-native-cache-merge'):
        for relative, path in files.items():
            target = output / relative
            if target.exists():
                if digest_file(target) != digest_file(path): raise ValueError('Existing merged bytes differ; do not overwrite')
                continue
            target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_name(target.name + '.partial')
            if temporary.exists(): temporary.rename(target.with_name(target.name+'.interrupted-'+uuid4().hex[:8]))
            shutil.copyfile(path,temporary)
            if digest_file(temporary) != digest_file(path): raise ValueError('Shard copy changed bytes')
            temporary.replace(target)
        manifest = {**manifests[0], 'schema':manifests[0]['full_cohort_schema'], 'scenes':scenes,
            'native_feature_locks':sorted(feature_locks)}
        manifest.pop('full_cohort_schema')
        receipt = {'status':'cache_complete', 'scene_count':contract.scene_count, 'merged_shards':len(manifests),
            'neural_inference_run':False, 'anomaly_labels_opened':False, 'cache_bytes':total}
        atomic_json(output/'cohort_manifest.json',manifest); atomic_json(output/'preparation_receipt.json',receipt)
        seal(output, [output/'cohort_manifest.json',*[output/relative for relative in files]])
        seal(output, [output/'artifact_lock.json',output/'run_binding.json',output/'preparation_receipt.json'],name='assembly_lock.json')
    return receipt
