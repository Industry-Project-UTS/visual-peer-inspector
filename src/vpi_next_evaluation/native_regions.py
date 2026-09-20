"""Current, label-free LocateAnything/SAM preparation; no DINO or scoring calls.

Proposal and mask checkpoints are separate. Final regions normalize mask hashes
as SHA256 of row-major uint8 pixels, retaining the worker hash for provenance.
"""
from pathlib import Path
from hashlib import sha256
import gc
import json
import math
import time
import os
import numpy as np
from PIL import Image

from vpi_object_bank.storage import (atomic_json, atomic_unit, bind, digest, digest_file,
    free_space, read_json, seal, terminal_status, verified_unit, verify_seal)
from vpi_remaining_evaluation.gates import CONTRACTS
from .exposure import verify_exposure
from vpi_remaining_evaluation.workbench import resolve_input_root
from visual_peer_inspector.data.model_safe_package import verify_model_input_package, membership_sha256
from visual_peer_inspector.models.locateanything import (LocateAnythingConfiguration,
    LocateAnythingWorker, ProposalJob, bounded_locateanything_execution_policy, generation_wall_time_limit)
from visual_peer_inspector.models.sam2 import (Sam2Configuration, Sam2ImageWorker,
    SamImageJob, SamPrompt, decode_binary_mask, binary_mask_sha256)
from visual_peer_inspector.web.deployment import pipeline_config
from visual_peer_inspector.workflows.proposal_refinement_ds1 import ProposalPolicy, apply_proposal_policy
from .contracts import current_method_contract, runtime_source_files, require_protocol_receipt
from .official import OFFICIAL, verify_official_exposure

SCHEMA = 'vpi-current-native-regions/v1'
RETRYABLE = {'NoValidBoxes', 'PossibleTruncation', 'TimeoutError'}


def preparation_gates(cfg, source_root):
    """Verify scientific source/selection on a separately pinned neural runtime.

The CPU decision runtime is checked when decisions run; it is not substituted
with the GPU preparation runtime. All other method/exposure gates remain exact.
"""
    role = cfg['role']; contract = {**CONTRACTS, **OFFICIAL}.get(role)
    if contract is None:
        raise ValueError('Unknown original or official role')
    freeze_root = resolve_input_root(Path(cfg.get('method_freeze_root') or cfg['freeze_root']), 'method_freeze.json')
    lock = verify_seal(freeze_root)
    if set(lock['files']) != {'method_freeze.json'}:
        raise ValueError('Freeze inventory differs')
    freeze = read_json(freeze_root / 'method_freeze.json')
    if any(freeze.get(k) != v for k, v in current_method_contract().items()):
        raise ValueError('Frozen method or evaluation protocol differs')
    if freeze.get('source_sha256') != runtime_source_files(source_root) or not freeze.get('reference_publication_sha256') or not freeze.get('runtime_versions'):
        raise ValueError('Preparation source differs from frozen current method')
    result = {'method_freeze_sha256': digest_file(freeze_root / 'method_freeze.json')}
    exposure_root = cfg.get('exposure_root')
    if exposure_root:
        exposure_root = resolve_input_root(Path(exposure_root), 'exposure_ledger.json')
    verify_exposure('internal_lock' if role in OFFICIAL else role, exposure_root)
    if exposure_root:
        result['exposure_lock_sha256'] = digest_file(Path(exposure_root) / 'artifact_lock.json')
    if role == 'internal_lock' or role in OFFICIAL:
        stage = '09d' if role in OFFICIAL else '09b'
        key = 'decision_root' if role in OFFICIAL else 'calibration_receipt_root'
        receipt_root = resolve_input_root(Path(cfg[key]), 'protocol_lock.json')
        receipt = require_protocol_receipt(receipt_root, stage, result['method_freeze_sha256'])
        if receipt['exposure_lock_sha256'] != result['exposure_lock_sha256']:
            raise ValueError('Preparation and prerequisite exposure audits differ')
        result['prerequisite_protocol_sha256'] = digest_file(receipt_root / 'protocol_lock.json')
    if role in OFFICIAL:
        official_root = resolve_input_root(Path(cfg['official_exposure_root']), 'official_exposure_ledger.json')
        result.update(verify_official_exposure(official_root))
    return contract, result


def fixed_vision():
    config = pipeline_config()
    return {key: config[key] for key in ('locateanything', 'proposal_policy', 'sam2')}


def _configs():
    vision = fixed_vision(); la = dict(vision['locateanything'])
    la.pop('maximum_attempts'); la['model_revision'] = la.pop('revision')
    return LocateAnythingConfiguration(**la), Sam2Configuration(), ProposalPolicy(**vision['proposal_policy'])


def _rgb_records(root, contract):
    audit = verify_model_input_package(root, contract=contract)
    checks = dict((line.split('  ', 1)[1], line.split('  ', 1)[0])
        for line in (root / 'content_checksums.sha256').read_text().splitlines() if line)
    output = {}
    for line in (root / 'neutral_corpus_manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        for view in row['view_ids']:
            scene = row['opaque_scene_id']; relative = f'media/{scene}/rgb/{view}.png'
            output[scene, view] = {'scene': scene, 'view': view, 'rgb': relative,
                'rgb_sha256': checks[relative], '_rgb_path': str(root / relative)}
    return output, audit.to_dict()


def _preflight(cfg, source_root):
    contract, gates = preparation_gates(cfg, source_root)
    rgb_root = resolve_input_root(Path(cfg['rgb_root']), 'package_record.json')
    records, audit = _rgb_records(rgb_root, contract)
    count, index = cfg.get('shard_count', 1), cfg.get('shard_index', 0)
    if isinstance(count, bool) or isinstance(index, bool) or not isinstance(count, int) or not isinstance(index, int) or count < 1 or not 0 <= index < count:
        raise ValueError('Invalid deterministic region shard partition')
    scenes = sorted({s for s,v in records}); selected = scenes[index::count]
    if not selected: raise ValueError('Region shard is empty')
    maximum = cfg.get('maximum_new_views', 100)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('maximum_new_views must be a positive integer')
    timeout = cfg.get('attempt_timeout_seconds', 180)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise ValueError('Attempt timeout must be within 1–600 seconds')
    binding = {'schema': SCHEMA, 'role': cfg['role'], 'cohort_id': contract.cohort_id,
        'model_input_sha256': digest(audit), 'vision': fixed_vision(), 'gates': gates,
        'attempt_timeout_seconds': timeout, 'terminal_failure_policy': 'retain_explicit_after_two_attempts',
        'partition': {'count':count, 'index':index, 'scenes':selected},
        'mask_hash_scheme': 'sha256-row-major-uint8', 'source_sha256': {
            'native_regions.py': digest_file(Path(source_root) / 'src/vpi_next_evaluation/native_regions.py')}}
    if cfg.get('reuse'):
        from .native_cache import native_regions
        reused, source_hash = native_regions({**cfg['reuse'], 'role': cfg['role']}, contract)
        if set(reused) != set(records) or any(reused[k]['rgb_sha256'] != records[k]['rgb_sha256'] for k in records):
            raise ValueError('Reused regions and current RGB differ')
        binding['reuse_lock_sha256'] = source_hash
    else:
        reused = {}
    return contract, records, reused, binding


def plan_regions(cfg, source_root):
    contract, records, reused, binding = _preflight(cfg, source_root)
    root = Path(cfg['output_root']); binding_hash = digest(binding)
    if (root / 'run_binding.json').exists() and read_json(root / 'run_binding.json') != binding:
        raise ValueError('Region output binding changed')
    selected = {key:row for key,row in records.items() if key[0] in set(binding['partition']['scenes'])}
    complete = sum(verified_unit(root / 'checkpoints/regions' / s / v, binding_hash) for s, v in selected)
    proposals = sum(verified_unit(root / 'checkpoints/proposals' / s / v, binding_hash) for s, v in selected)
    return {'schema': SCHEMA, 'status': 'plan_only', 'scene_count': contract.scene_count,
        'image_count': len(records), 'completed_region_views': complete, 'completed_proposal_views': proposals,
        'shard_scene_count':len(binding['partition']['scenes']), 'shard_image_count':len(selected),
        'reusable_region_views': sum(key in reused for key in selected), 'missing_region_views': len(selected)-complete,
        'maximum_new_views': cfg.get('maximum_new_views', 100), 'requires_gpu': complete < len(selected) and not reused,
        'binding_sha256': binding_hash, 'neural_execution_performed': False}


def _models(cfg, source_root):
    """Validate local model bytes before any model constructor or GPU operation."""
    import sys
    scripts = str(Path(source_root) / 'scripts')
    if scripts not in sys.path: sys.path.insert(0, scripts)
    from current_notebook_runtime import verify_model_snapshot, check_runtime, observed_contract
    profile_path = Path(source_root) / 'requirements/profiles/linux-regions-cuda126.json'
    profile = read_json(profile_path)
    contract = profile['runtime_contract']
    if os.getenv('VPI_RUNTIME_PROFILE_ROOT'):
        contract = observed_contract(os.environ['VPI_RUNTIME_PROFILE_ROOT'], profile_path, contract)
    runtime = check_runtime(contract)
    vision = fixed_vision()
    receipt = verify_model_snapshot(Path(cfg['locate_model_root']),
        expected_model_id=vision['locateanything']['model_id'], expected_revision=vision['locateanything']['revision'])
    checkpoint = Path(cfg['sam_checkpoint'])
    if checkpoint.is_symlink() or checkpoint.stat().st_size != vision['sam2']['size_bytes'] or digest_file(checkpoint) != vision['sam2']['sha256']:
        raise ValueError('SAM checkpoint differs from pinned bytes')
    return {'locate_snapshot': receipt, 'sam_checkpoint_sha256': digest_file(checkpoint),
        'runtime': runtime, 'runtime_profile_sha256': digest_file(profile_path)}


def _factory(phase, cfg, source_root):
    la, sam, _ = _configs()
    if phase == 'proposals':
        import torch
        policy = bounded_locateanything_execution_policy(la, maximum_attempts_per_job=2,
            terminal_failure_disposition='carry_as_zero_proposal', attempt_timeout_seconds=cfg.get('attempt_timeout_seconds', 180))
        return LocateAnythingWorker(la, device='cuda', dtype=torch.float16,
            model_source=str(cfg['locate_model_root']), execution_policy=policy)
    return Sam2ImageWorker(sam, Path(cfg['sam_checkpoint']))


def _close(worker):
    close = getattr(worker, 'close', None)
    if close: close()
    del worker
    gc.collect()
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()


def _proposal(worker, row, contract, binding):
    job = ProposalJob(cohort_id=contract.cohort_id, input_manifest_sha256=binding['model_input_sha256'],
        opaque_scene_id=row['scene'], view_id=row['view'], image_path=Path(row['_rgb_path']), image_sha256=row['rgb_sha256'])
    attempts = []; boxes = []; error = None
    la, _, policy = _configs()
    for attempt in range(1, 3):
        result = worker.run_job(job, attempt=attempt)
        if result.get('image_sha256') != row['rgb_sha256'] or result.get('configuration_sha256') != la.sha256:
            raise ValueError('Proposal worker returned incompatible image or configuration')
        error = result.get('error_type')
        attempts.append({'attempt': attempt, 'status': result['status'], 'error_type': error,
            'runtime_seconds': result.get('runtime_seconds')})
        if result['status'] == 'ok':
            boxes, _, _ = apply_proposal_policy(result['boxes_xyxy'], policy)
            error = None
            break
        if error not in RETRYABLE:
            raise RuntimeError(f'Fatal LocateAnything worker failure: {error}')
    return {'boxes': boxes, 'attempts': attempts, 'status': 'ok' if boxes else 'zero_proposals',
        'terminal_failure': error or ('filtered_empty' if not boxes else None)}


def _region(worker, row, proposal, contract, binding):
    base = {k: v for k, v in row.items() if not k.startswith('_')}
    base.update(route='automatic', objects=[], status=proposal['status'], terminal_failure=proposal['terminal_failure'])
    if not proposal['boxes']: return base
    prompts = tuple(SamPrompt('automatic_box', f'P{i:02d}', box) for i, box in enumerate(proposal['boxes'], 1))
    job = SamImageJob(opaque_scene_id=row['scene'], view_id=row['view'], image_path=Path(row['_rgb_path']),
        image_sha256=row['rgb_sha256'], prompts=prompts, prompt_package_sha256=digest(binding), cohort_id=contract.cohort_id)
    with generation_wall_time_limit(binding['attempt_timeout_seconds']) as supported:
        if not supported: raise RuntimeError('A bounded main-thread SAM timeout is required')
        result = worker.process(job, 1)
    if result.get('image_sha256') != row['rgb_sha256'] or result.get('configuration_sha256') != _configs()[1].sha256 or result.get('prompt_package_sha256') != digest(binding):
        raise ValueError('SAM worker returned another image or configuration')
    objects = result['routes']['automatic_box']
    if len(objects) != len(prompts): raise ValueError('SAM omitted prompted candidates')
    for item, prompt in zip(objects, prompts):
        if item['prompt_key'] != prompt.prompt_key or list(item['box_xyxy']) != list(prompt.box_xyxy):
            raise ValueError('SAM prompt order or geometry changed')
        mask = decode_binary_mask(item['mask_rle'])
        if mask.shape != (256, 256) or binary_mask_sha256(mask) != item['mask_sha256']:
            raise ValueError('SAM mask shape/hash differs')
        if int(mask.sum()) != item['foreground_pixels']:
            raise ValueError('SAM mask foreground count differs')
        if not mask.any(): continue
        base['objects'].append({'candidate_id': f"{row['view']}-{prompt.prompt_key}", 'box': list(prompt.box_xyxy),
            'mask': item['mask_rle'], 'mask_sha256': sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            'source_mask_sha256': item['mask_sha256'], 'predicted_quality_score': item['predicted_quality_score'],
            'foreground_pixels': int(mask.sum())})
    base['status'] = 'ok' if base['objects'] else 'empty_masks'
    base['terminal_failure'] = None if base['objects'] else 'empty_masks'
    base['empty_mask_count'] = len(prompts)-len(base['objects'])
    return base


def extract_regions(cfg, source_root, *, worker_factory=None, model_verifier=None):
    """Run at most the configured new views; injection is for fixture workers only."""
    started = time.monotonic()
    contract, records, reused, binding = _preflight(cfg, source_root)
    root = Path(cfg['output_root']); signature = bind(root, binding)
    selected_scenes = set(binding['partition']['scenes'])
    selected = {key:row for key,row in records.items() if key[0] in selected_scenes}
    sharded = len(selected) != len(records)
    if (root / 'region_shard_lock.json').exists():
        from .region_shards import read_region_shard
        read_region_shard(root, records, contract)
        return read_json(root / 'region_receipt.json')
    if (root / 'artifact_lock.json').exists():
        read_region_resource(root, Path(cfg['rgb_root']), contract)
        return read_json(root / 'region_receipt.json')
    maximum = cfg.get('maximum_new_views', 100)
    required = int(cfg.get('minimum_free_bytes', 1024**3))
    if required < 1: raise ValueError('minimum_free_bytes must be positive')
    free_space(root, required)
    keys = [key for key in sorted(selected) if not verified_unit(root / 'checkpoints/regions' / key[0] / key[1], signature)][:maximum]
    models = None
    factory = worker_factory or _factory
    with terminal_status(root, 'native-regions'):
        if not reused and keys:
            models = (model_verifier or _models)(cfg, source_root)
            runtime_binding = {'models': models, 'precision': 'float16', 'device': 'cuda'}
            runtime_path = root / 'model_runtime.json'
            if runtime_path.exists():
                runtime_lock = verify_seal(root, name='model_runtime_lock.json')
                if set(runtime_lock['files']) != {'model_runtime.json'} or read_json(runtime_path) != runtime_binding:
                    raise ValueError('Model resource bytes changed during resume')
            else:
                atomic_json(runtime_path, runtime_binding)
                seal(root, [runtime_path], name='model_runtime_lock.json')
            worker = None
            try:
                for key in keys:
                    unit = root / 'checkpoints/proposals' / key[0] / key[1]
                    if verified_unit(unit, signature): continue
                    if worker is None: worker = factory('proposals', cfg, source_root)
                    with atomic_unit(unit, signature) as temp:
                        atomic_json(temp / 'proposal.json', _proposal(worker, records[key], contract, binding))
                    print(json.dumps({'event': 'proposal_checkpoint', 'completed': len(list((root/'checkpoints/proposals').glob('*/*/artifact_lock.json')))}), flush=True)
            finally:
                if worker is not None: _close(worker)
                worker = None
        worker = None
        try:
            for i, key in enumerate(keys, 1):
                unit = root / 'checkpoints/regions' / key[0] / key[1]
                if key in reused:
                    original = reused[key]
                    result = {k:original[k] for k in ('scene','view','route','rgb','rgb_sha256','status','terminal_failure','empty_mask_count') if k in original}
                    result['objects'] = [{k:obj[k] for k in ('candidate_id','box','mask','mask_sha256','source_mask_sha256','predicted_quality_score','foreground_pixels') if k in obj} for obj in original['objects']]
                    result.setdefault('status', 'ok' if result['objects'] else 'empty_reused')
                    result.setdefault('terminal_failure', None if result['objects'] else 'empty_reused')
                else:
                    proposal = read_json(root / 'checkpoints/proposals' / key[0] / key[1] / 'proposal.json')
                    if proposal['boxes'] and worker is None: worker = factory('regions', cfg, source_root)
                    result = _region(worker, records[key], proposal, contract, binding)
                with atomic_unit(unit, signature) as temp:
                    atomic_json(temp / 'region.json', result)
                print(json.dumps({'event': 'region_checkpoint', 'completed_this_call': i, 'planned_this_call': len(keys)}), flush=True)
        finally:
            if worker is not None: _close(worker)
        complete = [key for key in sorted(selected) if verified_unit(root / 'checkpoints/regions' / key[0] / key[1], signature)]
        done = len(complete) == len(selected)
        receipt = {'schema': SCHEMA, 'status': ('shard_complete' if sharded else 'complete') if done else 'checkpointed',
            'scene_count': contract.scene_count, 'image_count': len(records), 'completed_views': len(complete),
            'shard_scene_count':len(selected_scenes),'shard_image_count':len(selected),
            'remaining_views': len(selected)-len(complete), 'full_cohort_views_not_in_this_shard':len(records)-len(selected), 'elapsed_seconds_this_call': time.monotonic()-started,
            'no_scoring_or_dino': True, 'binding_sha256': signature}
        if done:
            paths = []
            for scene, view in complete:
                path = root / 'regions' / scene / f'{view}.json'
                atomic_json(path, read_json(root / 'checkpoints/regions' / scene / view / 'region.json'))
                paths.append(path)
            manifest = {'schema': SCHEMA, 'role': cfg['role'], 'cohort_id': contract.cohort_id,
                'scene_count': contract.scene_count, 'image_count': len(records), 'scenes': sorted(selected_scenes),
                'records': [p.relative_to(root).as_posix() for p in paths], 'model_input_sha256': binding['model_input_sha256'],
                'vision': fixed_vision(), 'mask_hash_scheme': binding['mask_hash_scheme']}
            manifest_name = 'region_shard_manifest.json' if sharded else 'region_manifest.json'
            if sharded:
                manifest['schema'] = 'vpi-current-native-region-shard/v1'
                manifest['partition'] = binding['partition']
            atomic_json(root / manifest_name, manifest)
            atomic_json(root / 'region_receipt.json', receipt)
            sealed = [root/'run_binding.json', root/manifest_name, root/'region_receipt.json', *paths]
            if (root/'model_runtime.json').exists(): sealed.extend([root/'model_runtime.json', root/'model_runtime_lock.json'])
            seal(root, sealed, name='region_shard_lock.json' if sharded else 'artifact_lock.json', metadata={'schema': SCHEMA, 'binding_sha256': signature})
            if sharded:
                from .region_shards import read_region_shard
                read_region_shard(root, records, contract)
            else: read_region_resource(root, Path(cfg['rgb_root']), contract)
        else: atomic_json(root / 'checkpoint_receipt.json', receipt)
    atomic_json(root / 'run_status.json', {'stage':'native-regions', 'status':receipt['status']})
    return receipt


def read_region_resource(region_root, rgb_root, contract):
    root = resolve_input_root(Path(region_root), 'region_manifest.json')
    rgb_root = resolve_input_root(Path(rgb_root), 'package_record.json')
    lock = verify_seal(root)
    required = {'run_binding.json','region_manifest.json','region_receipt.json'}
    if not required <= set(lock['files']): raise ValueError('Region seal omits provenance')
    manifest = read_json(root / 'region_manifest.json'); binding = read_json(root/'run_binding.json')
    expected, audit = _rgb_records(rgb_root, contract)
    if manifest.get('schema') != SCHEMA or manifest.get('cohort_id') != contract.cohort_id or manifest.get('scene_count') != contract.scene_count:
        raise ValueError('Native region cohort/schema differs')
    if manifest.get('scenes') != sorted({s for s,v in expected}) or manifest.get('image_count') != len(expected):
        raise ValueError('Native region manifest membership or image denominator differs')
    if manifest.get('model_input_sha256') != digest(audit) or binding.get('model_input_sha256') != digest(audit):
        raise ValueError('Native region model input differs')
    if manifest.get('vision') != fixed_vision() or binding.get('vision') != fixed_vision():
        raise ValueError('Native region frozen vision differs')
    if manifest.get('mask_hash_scheme') != 'sha256-row-major-uint8': raise ValueError('Unknown mask hash scheme')
    expected_files = required | set(manifest['records'])
    if 'model_runtime.json' in lock['files']:
        expected_files |= {'model_runtime.json','model_runtime_lock.json'}
        verify_seal(root, name='model_runtime_lock.json')
    if set(lock['files']) != expected_files: raise ValueError('Region resource has unexpected sealed files')
    receipt = read_json(root/'region_receipt.json')
    if receipt.get('status') != 'complete' or receipt.get('binding_sha256') != digest(binding) or lock.get('metadata',{}).get('binding_sha256') != digest(binding):
        raise ValueError('Region receipt or lock binding differs')
    paths = manifest['records']
    if len(paths) != len(set(paths)) or set(paths) != {f'regions/{s}/{v}.json' for s,v in expected}:
        raise ValueError('Native regions omit or duplicate contracted views')
    if not set(paths) <= set(lock['files']): raise ValueError('Unsealed region record')
    return _read_records(root, paths, expected), digest_file(root / 'artifact_lock.json')


def _read_records(root, paths, expected):
    """Validate normalized rows identically for final resources and shards."""
    output = {}
    for relative in paths:
        row = read_json(root / relative); key = row['scene'], row['view']
        allowed_row = {'scene','view','route','rgb','rgb_sha256','objects','status','terminal_failure','empty_mask_count'}
        if set(row)-allowed_row: raise ValueError('Unexpected region record fields')
        if key in output or relative != f'regions/{key[0]}/{key[1]}.json' or key not in expected or row.get('route') != 'automatic' or row['rgb'] != expected[key]['rgb'] or row['rgb_sha256'] != expected[key]['rgb_sha256']:
            raise ValueError('Native region image/route differs')
        seen = set()
        for obj in row['objects']:
            allowed_object = {'candidate_id','box','mask','mask_sha256','source_mask_sha256','predicted_quality_score','foreground_pixels'}
            if set(obj)-allowed_object: raise ValueError('Unexpected model-facing object fields')
            candidate = obj['candidate_id']; box = obj['box']; mask = decode_binary_mask(obj['mask'])
            if not candidate or candidate in seen or '/' in candidate or '..' in candidate: raise ValueError('Invalid candidate identity')
            seen.add(candidate)
            if len(box)!=4 or not 0<=box[0]<box[2]<=256 or not 0<=box[1]<box[3]<=256: raise ValueError('Invalid region box')
            if mask.shape != (256,256) or not mask.any() or sha256(mask.astype(np.uint8).tobytes()).hexdigest()!=obj['mask_sha256']:
                raise ValueError('Invalid normalized mask')
            if 'source_mask_sha256' in obj and binary_mask_sha256(mask) != obj['source_mask_sha256']:
                raise ValueError('Invalid historical/source mask hash')
            if 'foreground_pixels' in obj and int(mask.sum()) != obj['foreground_pixels']:
                raise ValueError('Region foreground count differs')
        row['_rgb_path'] = expected[key]['_rgb_path']; output[key] = row
    return output
