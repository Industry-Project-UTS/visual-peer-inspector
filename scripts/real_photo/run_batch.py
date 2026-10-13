"""Offline Kaggle orchestration around the unchanged frozen web pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time

WHEELS = {
    'regions': 'b95309ec18a1c4de59cdb09fa1d7817668db2d7a1432fe8254f86a378da0893f',
    'dino': '908cd70d0da8788f6cfd2e752d7dc1ddef11c5d81349653eaa08546b3e014207',
}
ASSETS = 'f407fdc4c45291526d82bf39fa0f5a841a6dbd301a6d6febace9d3eaa2f79698'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def unique(paths, description):
    values = list(paths)
    if len(values) != 1:
        raise ValueError(f'Expected exactly one {description}; found {len(values)}')
    return values[0]

def verify_payload(root):
    manifest = json.loads((root / 'batch-manifest.json').read_text())
    if manifest['schema'] != 'vpi-real-photo-batch/v1' or manifest['mode'] != 'automatic':
        raise ValueError('Unexpected batch contract')
    if (manifest['scene_count'], manifest['image_count'], manifest['case_count']) != (29, 231, 58):
        raise ValueError('The full 29-scene cohort is required')
    if (len(manifest['scenes']), len(manifest['image_files']), len(manifest['cases'])) != (29, 231, 58):
        raise ValueError('Declared cardinalities differ from the actual inventories')
    if manifest['source_commit'] != '5b78942e29fe7186458c6a13306eeba2657fc3e6':
        raise ValueError('Source commit differs from the frozen run')
    if sha(root / 'run_batch.py') != manifest['runner_sha256']:
        raise ValueError('Runner changed after preparation')
    for relative, expected in manifest['image_files'].items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('RGB input changed or escaped its root')
    actual_rgb = {p.relative_to(root).as_posix() for p in (root / 'rgb').rglob('*') if p.is_file()}
    if actual_rgb != set(manifest['image_files']):
        raise ValueError('Extra or missing RGB inputs')
    actual_source = {p.relative_to(root / 'source').as_posix() for p in (root / 'source').rglob('*')
                     if p.is_file()}
    if actual_source != set(manifest['source_files']):
        raise ValueError('Extra or missing frozen source files')
    for relative, expected in manifest['source_files'].items():
        path = root / 'source' / relative
        if path.is_symlink() or not path.resolve().is_relative_to((root / 'source').resolve()) or sha(path) != expected:
            raise ValueError('Frozen source integrity failure')
    cases = manifest['cases']
    if len({c['case_id'] for c in cases}) != 58 or any(not 2 <= len(c['views']) <= 5 for c in cases):
        raise ValueError('Duplicate cases or incompatible view counts')
    if sorted(v for c in cases for v in c['views']) != sorted(manifest['image_files']):
        raise ValueError('Cases must cover every view exactly once')
    if {s['scene_id'] for s in manifest['scenes']} != {f'S{i:03d}' for i in range(1, 30)}:
        raise ValueError('Scene membership changed')
    for index, scene in enumerate(manifest['scenes'], 1):
        expected_id = f'S{index:03d}'
        if scene['scene_id'] != expected_id or len(scene['views']) != (7 if index == 9 else 8):
            raise ValueError('Scene order or view inventory changed')
        for group, views in [('A', scene['views'][:5]), ('B', scene['views'][5:])]:
            expected_case = {'case_id': f'{expected_id}-{group}', 'scene_id': expected_id,
                             'views': views, 'shard': 0 if index <= 15 else 1}
            if expected_case not in cases:
                raise ValueError('Scene grouping or shard assignment changed')
    return manifest

def verify_asset_compatibility(config, receipt, config_hash):
    # This exact earlier preparation differs only in the review-policy key;
    # it contains identical immutable weights, model settings and decision settings.
    if config_hash != '498beffae36101388024a95ba9ff9e3503fde66fc7a7903b6ca0c0113760e842':
        raise ValueError('Current pipeline configuration changed')
    if receipt.get('pipeline_config_sha256') != '250bb7ae2a883e016947458b2603d8ca975ac31d88db718dcb33bc3f2e10a3fe':
        raise ValueError('Unreviewed asset-preparation configuration')
    selected = {'locateanything': config['locateanything'], 'sam2': config['sam2'], 'dino': config['encoder']}
    if set(receipt.get('models', {})) != set(selected):
        raise ValueError('Model asset receipt inventory differs')
    for name, spec in selected.items():
        model = receipt['models'][name]
        if model['model_id'] != spec['model_id'] or model['revision'] != spec['revision']:
            raise ValueError('Prepared model ID or immutable revision changed')
    return {'prepared_config_sha256': receipt['pipeline_config_sha256'],
            'current_config_sha256': config_hash,
            'review_policy_change': {'prepared': 'provisional-review-all-v1', 'current': 'calibration-selective-v1'},
            'current_review_policy_retained': True, 'model_revisions_identical': True}

def prepare_runtime(root, output):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Requires Linux x86_64 CPython 3.12')
    if shutil.disk_usage('/tmp').free < 20 * 1024**3:
        raise RuntimeError('Insufficient temporary storage for isolated runtimes')
    sys.path.insert(0, str(root / 'source/scripts'))
    from current_notebook_runtime import install_offline, verify_model_snapshot
    wheel_paths = list(Path('/kaggle/input').rglob('wheelhouse_manifest.json'))
    runtime = {}
    for name, expected in WHEELS.items():
        marker = unique((p for p in wheel_paths if sha(p) == expected), f'{name} wheelhouse')
        contract = json.loads(marker.read_text())['runtime_contract']
        target = Path('/tmp') / f'vpi-runtime-realphoto-{name}'
        install_offline(contract, marker.parent, target, timeout=1800)
        runtime[name] = {'target': str(target), 'contract': contract,
                         'observed_contract': {**contract, 'python': platform.python_version()},
                         'python_patch_changed': contract['python'] != platform.python_version(),
                         'wheelhouse_sha256': expected}
        print(f'Installed verified {name} runtime', flush=True)
    marker = unique((p for p in Path('/kaggle/input').rglob('model_assets_receipt.json')
                     if sha(p) == ASSETS), 'pinned model assets')
    assets_root = marker.parent
    config = json.loads((root / 'source/src/visual_peer_inspector/web/pipeline_config.json').read_text())
    asset_receipt = json.loads(marker.read_text())
    runtime['asset_compatibility'] = verify_asset_compatibility(
        config, asset_receipt, sha(root / 'source/src/visual_peer_inspector/web/pipeline_config.json'))
    for name, spec in [('locateanything', config['locateanything']), ('sam2', config['sam2']), ('dino', config['encoder'])]:
        verified = verify_model_snapshot(assets_root / name, spec['model_id'], spec['revision'])
        if verified != asset_receipt['models'][name]:
            raise ValueError('Model snapshot differs from the sealed preparation receipt')
    runtime['assets_root'] = str(assets_root)
    runtime['assets_receipt_sha256'] = ASSETS
    # Retain the app's native hf_hub_download lookup, seeded from verified offline bytes.
    cache = Path('/tmp/vpi-runtime-realphoto-hf')
    sam = config['sam2']
    checkpoint = cache / ('models--' + sam['model_id'].replace('/', '--')) / 'snapshots' / sam['revision'] / sam['filename']
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(assets_root / 'sam2' / sam['filename'])
    runtime['hf_cache'] = str(cache)
    # A clean -S interpreter avoids the notebook image's site packages and startup hooks.
    dino_entry = Path('/tmp/vpi-runtime-realphoto-dino-entry.py')
    dino_entry.write_text('import sys, runpy, traceback, os\nfrom pathlib import Path\n' +
        'sys.path[:0] = ' + repr([runtime['dino']['target'], str(root / 'source/src')]) + '\n' +
        "assert sys.argv[1] == '-m'\nmodule = sys.argv[2]\nsys.argv = [module] + sys.argv[3:]\n" +
        "try:\n    runpy.run_module(module, run_name='__main__', alter_sys=True)\n" +
        "except Exception:\n    Path(" + repr(str(output / 'logs')) + ") .joinpath('dino-worker-' + str(os.getpid()) + '.log').write_text(traceback.format_exc())\n    raise\n")
    dino_python = Path('/tmp/vpi-runtime-realphoto-dino-python')
    import shlex
    dino_python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -S -B ' + shlex.quote(str(dino_entry)) + ' "$@"\n')
    dino_python.chmod(0o700)
    runtime['dino_python'] = str(dino_python)
    dump(output / 'runtime.json', runtime)

def activate(root, output):
    runtime = json.loads((output / 'runtime.json').read_text())
    sys.path[:0] = [runtime['regions']['target'], str(root / 'source/src'), str(root / 'source/scripts')]
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      HF_HUB_CACHE=runtime['hf_cache'], VPI_DINO_PYTHON=runtime['dino_python'],
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    return runtime

def models_for(runtime):
    from visual_peer_inspector.web.model_runtime import GPUModels
    class OfflineModels(GPUModels):
        def _snapshot(self, config):
            name = 'dino' if config['model_id'] == self.config['encoder']['model_id'] else 'locateanything'
            return str(Path(runtime['assets_root']) / name)
    return OfflineModels(allow_downloads=False)

def preflight(root, output):
    runtime = activate(root, output)
    from current_notebook_runtime import check_runtime
    import traceback
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    checks = {}
    def check(name, operation):
        print('Preflight: ' + name, flush=True)
        try:
            checks[name] = {'status': 'passed', 'details': operation()}
        except Exception as error:
            checks[name] = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)[:1500]}
            traceback.print_exc()
        dump(output / 'preflight.json', {'status': 'checking', 'checks': checks,
             'synthetic_probes_excluded_from_cohort': True})
    check('regions_runtime', lambda: check_runtime(runtime['regions']['observed_contract']))
    def gpu():
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
            raise RuntimeError('A CUDA GPU with at least 14 GiB is required')
        return {'name': torch.cuda.get_device_name(0), 'memory_bytes': torch.cuda.get_device_properties(0).total_memory}
    check('cuda', gpu)
    def inputs():
        from visual_peer_inspector.web.core import load_views
        manifest = json.loads((root / 'batch-manifest.json').read_text())
        for case in manifest['cases']:
            assert len(load_views([root / p for p in case['views']])) == len(case['views'])
        return {'views_validated': 231, 'inspections': 58}
    check('all_rgb_inputs', inputs)
    def policy():
        from visual_peer_inspector.web.deployment import pipeline_config, pipeline_fingerprint, load_review_policy
        from visual_peer_inspector.web.pipeline import UploadPipeline
        rule, version = load_review_policy(None)
        UploadPipeline(models_for(runtime), cache_root=output / 'synthetic-preflight/policy-cache')
        assert version == pipeline_config()['default_review_policy']
        return {'version': version, 'policy_sha256': rule.source_sha256, 'pipeline_fingerprint': pipeline_fingerprint()}
    check('frozen_policy_and_source', policy)
    synthetic = Image.new('RGB', (256, 256), (238, 241, 245))
    draw = ImageDraw.Draw(synthetic)
    draw.rectangle((25, 50, 75, 190), fill=(150, 70, 30))
    draw.rectangle((95, 50, 145, 190), fill=(150, 70, 30))
    draw.rectangle((165, 50, 215, 190), fill=(30, 90, 150))
    def sam():
        from visual_peer_inspector.models.sam2 import verify_installed_sam2_revision
        from visual_peer_inspector.web.deployment import pipeline_config
        provenance = verify_installed_sam2_revision(pipeline_config()['sam2']['code_revision'])
        models = models_for(runtime)
        try:
            masks, scores = models.segment(synthetic, [(25, 50, 75, 190)])
            assert masks.shape == (1, 256, 256) and scores.shape == (1,) and np.isfinite(scores).all()
            return {'source': provenance, 'mask_shape': list(masks.shape)}
        finally:
            models.release()
    check('sam_source_checkpoint_config_and_forward', sam)
    def dino():
        models = models_for(runtime)
        try:
            probe = models.encode([np.asarray(synthetic.resize((224, 224)))] * 40)
            assert probe.global_features.shape == (40, 384) and np.isfinite(probe.global_features).all()
            assert len(probe.patch_features) == 40 and all(p.shape == (256, 384) and np.isfinite(p).all() for p in probe.patch_features)
            return {'crops': 40, 'feature_dimension': 384, 'patch_grid': 16}
        finally:
            models.release()
    check('isolated_dino_maximum_batch', dino)
    def locate():
        models = models_for(runtime)
        try:
            boxes, notes = models.propose(synthetic, 'synthetic_preflight')
            assert len(boxes) <= 8
            return {'proposal_count': len(boxes), 'synthetic_generation_completed': True}
        finally:
            models.release()
    check('locateanything_load_and_generation', locate)
    def exchange():
        return synthetic_pipeline_exchange(output)
    check('scoring_cache_masks_transport_and_archive', exchange)
    passed = all(c['status'] == 'passed' for c in checks.values())
    dump(output / 'preflight.json', {'status': 'passed' if passed else 'failed', 'checks': checks,
         'inference_mode': 'automatic', 'human_labels_loaded': False,
         'synthetic_probes_excluded_from_cohort': True,
         'python_patch_changed': runtime['regions']['python_patch_changed']})
    if not passed:
        raise RuntimeError('Startup checks failed: ' + ', '.join(n for n, c in checks.items() if c['status'] != 'passed'))
    print('All eight independent startup checks passed; real-photo inference may start', flush=True)

def synthetic_pipeline_exchange(output):
    """Contract-only adapters exercise scoring and persistence, never report accuracy."""
    import tempfile, zipfile
    import numpy as np
    from PIL import Image
    from visual_peer_inspector.models.visual_encoder import EncodedBatch
    from visual_peer_inspector.web.core import InspectionRequest, load_views, validate_result
    from visual_peer_inspector.web.pipeline import UploadPipeline
    from visual_peer_inspector.web.transport import write_request, read_request, write_result, read_result
    class ContractModels:
        def propose(self, image, view_id):
            return [(20, 50, 75, 190), (92, 50, 147, 190), (164, 50, 219, 190)], []
        def segment(self, image, boxes):
            masks = np.zeros((len(boxes), 256, 256), dtype=bool)
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                masks[i, y1:y2, x1:x2] = True
            return masks, np.full(len(boxes), .95)
        def encode(self, crops):
            vectors = np.zeros((len(crops), 384), dtype=np.float32)
            for i in range(len(crops)):
                vectors[i, i % 3] = 1
            return EncodedBatch(vectors, tuple(np.tile(v, (256, 1)) for v in vectors), 16, 384)
        def release(self):
            pass
    with tempfile.TemporaryDirectory(prefix='vpi-preflight-exchange-') as directory:
        p = Path(directory)
        for i, color in enumerate([(150, 160, 170), (155, 165, 175)]):
            Image.new('RGB', (512, 320), color).save(p / f'input-{i}.png')
        request = InspectionRequest(load_views([p / 'input-0.png', p / 'input-1.png']), mode='automatic')
        result = UploadPipeline(ContractModels(), cache_root=p / 'cache').inspect(request, lambda _: None)
        validate_result(result, request)
        assert len(result.observations) == 6 and all(o.mask.size == (512, 320) for o in result.observations)
        write_request(p, request)
        restored = read_request(p)
        assert tuple(v.digest for v in restored.views) == tuple(v.digest for v in request.views)
        write_result(p, result)
        assert read_result(p, request).ranking == result.ranking
        with zipfile.ZipFile(p / 'export.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(p / 'result.json', 'result.json')
        with zipfile.ZipFile(p / 'export.zip') as archive:
            assert archive.testzip() is None and json.loads(archive.read('result.json'))['mode'] == 'automatic'
    return {'synthetic_views': 2, 'synthetic_observations': 6, 'scoring_and_export_verified': True}

def seal_case(case_root, manifest_sha):
    files = {p.relative_to(case_root).as_posix(): sha(p) for p in sorted(case_root.rglob('*'))
             if p.is_file() and p.name != 'case-lock.json'}
    dump(case_root / 'case-lock.json', {'manifest_sha256': manifest_sha, 'files': files})

def run_case(root, output, case_id):
    runtime = activate(root, output)
    manifest = json.loads((root / 'batch-manifest.json').read_text())
    case = unique((c for c in manifest['cases'] if c['case_id'] == case_id), 'case')
    destination = output / 'cases' / case_id
    destination.mkdir(parents=True)
    started = time.monotonic()
    from visual_peer_inspector.web.core import InspectionRequest, load_views, validate_result
    from visual_peer_inspector.web.pipeline import UploadPipeline
    from visual_peer_inspector.web.session_cache import SessionCache
    from visual_peer_inspector.web.transport import write_result
    views = load_views([root / p for p in case['views']])
    request = InspectionRequest(views=tuple(views), mode='automatic')
    dump(destination / 'inputs.json', {'case': case, 'view_map': [
        {'path': p, 'view_id': v.view_id, 'digest': v.digest, 'size': v.image.size}
        for p, v in zip(case['views'], views, strict=True)]})
    models = models_for(runtime)
    pipeline = UploadPipeline(models, cache_root=destination / 'candidate-cache')
    # Durable private research checkpoints; this changes retention only, not inference.
    pipeline.cache = SessionCache(destination / 'candidate-cache', ttl_seconds=7 * 86400)
    try:
        def progress(value):
            print(json.dumps({'case': case_id, 'stage': str(value), 'detail': value.detail,
                              'elapsed_seconds': value.elapsed_seconds}), flush=True)
        result = pipeline.inspect(request, progress)
        validate_result(result, request)
        result.provenance['real_photo_batch'] = {
            'source_commit': manifest['source_commit'], 'manifest_sha256': sha(root / 'batch-manifest.json'),
            'case_id': case_id, 'view_rule': manifest['view_rule'], 'labels_loaded': False,
            'compute': 'Kaggle CUDA 12.6 regions / isolated CPU FP32 DINO', 'checkpoint_retention_seconds': 7 * 86400}
        write_result(destination, result)
        receipt = {'status': 'complete', 'case_id': case_id, 'view_count': len(views),
                   'review_state': result.review_state, 'selected_identity_id': result.selected_identity_id,
                   'observations': len(result.observations), 'elapsed_seconds': round(time.monotonic() - started, 3)}
    except Exception as error:
        receipt = {'status': 'failed', 'case_id': case_id, 'view_count': len(views),
                   'error_type': type(error).__name__, 'error': str(error)[:1500],
                   'elapsed_seconds': round(time.monotonic() - started, 3)}
        import traceback
        traceback.print_exc()
    finally:
        models.release()
    dump(destination / 'case-receipt.json', receipt)
    seal_case(destination, sha(root / 'batch-manifest.json'))
    print(json.dumps(receipt), flush=True)

def wait_child(command, log_path, timeout):
    with log_path.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        started = last_heartbeat = time.monotonic()
        while child.poll() is None:
            if time.monotonic() - started > timeout:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 'timeout'
            if time.monotonic() - last_heartbeat >= 30:
                print(f'{log_path.stem}: {int(time.monotonic()-started)} seconds', flush=True)
                last_heartbeat = time.monotonic()
            time.sleep(2)
        return child.returncode

def orchestrate(root, output, shard):
    manifest = verify_payload(root)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'run-binding.json', {'manifest_sha256': sha(root / 'batch-manifest.json'),
                                       'shard': shard, 'source_commit': manifest['source_commit']})
    logs = output / 'logs'
    logs.mkdir()
    script = root / 'run_batch.py'
    base = [sys.executable, '-S', '-B', str(script), '--root', str(root), '--output', str(output)]
    for action, timeout in [('setup', 4200), ('preflight', 1800)]:
        status = wait_child(base + ['--action', action], logs / (action + '.log'), timeout)
        if status != 0:
            dump(output / 'run-status.json', {'status': 'infrastructure_failed', 'stage': action, 'returncode': status})
            print((logs / (action + '.log')).read_text()[-12000:], flush=True)
            raise RuntimeError(f'{action} failed; science has not started')
    receipts = []
    for case in (c for c in manifest['cases'] if c['shard'] == shard):
        case_id = case['case_id']
        print(f'Starting {case_id}, {len(case["views"])} views', flush=True)
        status = wait_child(base + ['--action', 'case', '--case-id', case_id], logs / (case_id + '.log'), 1800)
        case_root = output / 'cases' / case_id
        receipt_path = case_root / 'case-receipt.json'
        if status != 0 or not receipt_path.exists():
            dump(receipt_path, {'status': 'failed', 'case_id': case_id, 'view_count': len(case['views']),
                                'error_type': 'ChildProcessFailure', 'returncode': status})
            seal_case(case_root, sha(root / 'batch-manifest.json'))
        receipt = json.loads(receipt_path.read_text())
        receipts.append(receipt)
        dump(output / 'progress.json', {'cases_attempted': len(receipts), 'receipts': receipts})
        print(json.dumps(receipt), flush=True)
        def infrastructure_failure(r):
            return r['status'] == 'failed' and (r.get('error_type') in {'ChildProcessFailure', 'OutOfMemoryError'} or
                r.get('error_type') == 'RuntimeUnavailable' and r.get('error') in
                {'cuda_unavailable', 'models_not_cached', 'checkpoint_integrity', 'dino_runtime'})
        if len(receipts) >= 2 and all(infrastructure_failure(r) for r in receipts[-2:]):
            dump(output / 'run-status.json', {'status': 'infrastructure_failed', 'stage': 'cases', 'receipts': receipts})
            print((logs / (case_id + '.log')).read_text()[-12000:], flush=True)
            raise RuntimeError('Repeated runtime failure; preserve checkpoints and repair before continuing')
    expected = [c for c in manifest['cases'] if c['shard'] == shard]
    summary = {'schema': 'vpi-real-photo-run/v1', 'status': 'complete', 'shard': shard,
               'manifest_sha256': sha(root / 'batch-manifest.json'), 'cases_expected': len(expected),
               'cases_attempted': len(receipts), 'cases_complete': sum(r['status'] == 'complete' for r in receipts),
               'cases_failed': sum(r['status'] != 'complete' for r in receipts),
               'views_attempted': sum(len(c['views']) for c in expected), 'receipts': receipts,
               'accuracy_evaluated': False, 'assisted_executed': False}
    dump(output / 'run-summary.json', summary)
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    dump(output / 'run-lock.json', {'manifest_sha256': sha(root / 'batch-manifest.json'), 'files': files})
    print(json.dumps(summary, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=['run', 'setup', 'preflight', 'case'], default='run')
    parser.add_argument('--shard', type=int, choices=[0, 1])
    parser.add_argument('--case-id')
    args = parser.parse_args()
    if args.action == 'run':
        if args.shard is None:
            parser.error('--shard is required')
        orchestrate(args.root, args.output, args.shard)
    elif args.action == 'setup':
        prepare_runtime(args.root, args.output)
    elif args.action == 'preflight':
        preflight(args.root, args.output)
    else:
        run_case(args.root, args.output, args.case_id)

if __name__ == '__main__':
    main()
