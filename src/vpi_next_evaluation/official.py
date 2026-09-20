"""Current-method official prediction contracts, kept separate from original roles.

This module reads exposure records and sealed model-facing caches, never outcomes.
Official source release, cache extraction and outcome evaluation are separate adapters.
"""
from __future__ import annotations

import inspect
import math
import shutil
from uuid import uuid4
from pathlib import Path

from visual_peer_inspector.data.cohort_contract import OFFICIAL_SEEN_TEST_CONTRACT, OFFICIAL_UNSEEN_TEST_CONTRACT
from visual_peer_inspector.data.model_safe_package import membership_sha256
from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, terminal_status, verified_unit, verify_seal
from vpi_remaining_evaluation.cohort import _reject_labels
from .tensor_cache import current_inputs
from vpi_remaining_evaluation.workbench import resolve_input_root
from vpi_remaining_evaluation.gates import PrerequisiteBlocked
from .exposure import verify_exposure

OFFICIAL = {"seen": OFFICIAL_SEEN_TEST_CONTRACT, "unseen": OFFICIAL_UNSEEN_TEST_CONTRACT}
CACHE_SCHEMA = "vpi-current-official-cache/v1"
PREDICTION_SCHEMA = "vpi-current-official-predictions/v1"
EXPOSURE_SCHEMA = "vpi-official-exposure-audit/v1"
REQUIRED_METRICS = ("top1_tie_expected", "recall_at_2_tie_expected", "ap_tie_expected", "reciprocal_rank_tie_expected")


def verify_official_exposure(root):
    if root is None:
        raise PrerequisiteBlocked("OFFICIAL_EXPOSURE_MISSING", "Official seen/unseen need their own membership and asset-access audit; the original 2,000-scene ledger is not a substitute.")
    root = Path(root)
    lock = verify_seal(root)
    required = {"official_exposure_ledger.json", "asset_records.json"}
    if not required <= set(lock["files"]):
        raise ValueError("Official exposure seal omits audit records")
    ledger = read_json(root / "official_exposure_ledger.json")
    rows = read_json(root / "asset_records.json")
    from .exposure import OFFICIAL_SCHEMA, verify_qualified_official_exposure
    if ledger.get('schema') == OFFICIAL_SCHEMA:
        return verify_qualified_official_exposure(root)
    if ledger.get("schema") != EXPOSURE_SCHEMA or ledger.get("status") != "resolved":
        raise PrerequisiteBlocked("OFFICIAL_EXPOSURE_UNRESOLVED", "Reconcile prior official-role asset access without reshuffling or replacing the original roles.")
    sources = ledger.get("reviewed_source_sha256", [])
    if not sources or any(not isinstance(s, str) or len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in sources):
        raise ValueError("Official exposure needs reviewed source digests")
    if any(set(row) != {"role", "asset_group", "scene_count", "resolved", "prior_access"} for row in rows):
        raise ValueError("Invalid official exposure asset schema")
    keys = [(row["role"], row["asset_group"]) for row in rows]
    if len(keys) != len(set(keys)) or any(role not in OFFICIAL for role, _ in keys):
        raise ValueError("Duplicate or unknown official asset role")
    for role, contract in OFFICIAL.items():
        subset = [row for row in rows if row["role"] == role]
        if len(subset) != contract.asset_group_count or sum(row["scene_count"] for row in subset) != contract.scene_count:
            raise ValueError("Official exposure changed the fixed split cardinalities")
        if ledger.get("roles", {}).get(role, {}).get("membership_sha256") != contract.membership_sha256:
            raise ValueError("Official exposure changed source membership")
        for row in subset:
            if row["resolved"] is not True:
                raise PrerequisiteBlocked("OFFICIAL_EXPOSURE_UNRESOLVED", "An official asset-access record remains unresolved.")
            for event in row["prior_access"]:
                if set(event) != {"kind", "source_sha256"} or event["source_sha256"] not in sources or event["kind"] not in {"rgb", "predictions", "reference_identities", "anomaly_labels", "evaluation_results"}:
                    raise ValueError("Official exposure event is not linked to reviewed evidence")
                if event["kind"] in {"anomaly_labels", "evaluation_results"}:
                    raise PrerequisiteBlocked("OFFICIAL_OUTCOMES_ALREADY_EXPOSED", "Prior outcome access prevents treating this official run as unused confirmation; revise its declared claim before execution.")
    return {"official_exposure_status": "access_audited", "official_exposure_lock_sha256": digest_file(root / "artifact_lock.json"), "untouched_claim": False}


def resolve_official_inputs(config):
    """Resolve only model/prerequisite roots now; evaluator discovery stays delayed."""
    markers={'freeze_root':'method_freeze.json','decision_root':'protocol_lock.json',
        'exposure_root':'exposure_ledger.json','official_exposure_root':'official_exposure_ledger.json',
        'cache_root':'cohort_manifest.json','evaluation_root':'evaluation_lock.json',
        'seen_prediction_root':'prediction_lock.json','unseen_prediction_root':'prediction_lock.json'}
    for name,marker in markers.items():
        if config.get(name):
            config[name]=resolve_input_root(Path(config[name]),marker)
    for name,marker in [('cache_roots','cohort_manifest.json'),('prediction_summary_roots','summary_lock.json')]:
        if config.get(name):
            config[name]={key:resolve_input_root(Path(value),marker) for key,value in config[name].items()}
    if config.get('prediction_roots'):
        config['prediction_roots']={mode:{role:resolve_input_root(Path(value),'prediction_lock.json') for role,value in roles.items()} for mode,roles in config['prediction_roots'].items()}
    return config


def verify_prerequisites(config, source_root=None):
    from .contracts import verify_current_freeze, require_protocol_receipt
    resolve_official_inputs(config)
    freeze_root = Path(config["freeze_root"])
    freeze = verify_current_freeze(freeze_root, source_root)
    freeze_hash = digest_file(freeze_root / "method_freeze.json")
    verify_exposure("internal_lock", config.get("exposure_root"))
    exposure_hash = digest_file(Path(config["exposure_root"]) / "artifact_lock.json")
    receipt = require_protocol_receipt(config["decision_root"], "09d", freeze_hash)
    if receipt.get('roles') != ['internal_lock']:
        raise ValueError('Official stages require the internal-lock09D protocol receipt')
    if receipt.get("exposure_lock_sha256") != exposure_hash:
        raise ValueError("09D and the official stage use different exposure audits")
    official_audit = verify_official_exposure(config.get("official_exposure_root"))
    return freeze, {"method_freeze_sha256": freeze_hash, "exposure_lock_sha256": exposure_hash,
                    "internal_lock_protocol_sha256": digest_file(Path(config["decision_root"]) / "protocol_lock.json"),
                    **official_audit}


def validate_cache(cache_root, role, freeze_hash):
    root = Path(cache_root)
    if role not in OFFICIAL:
        raise ValueError("Official role must be seen or unseen")
    lock = verify_seal(root)
    if "cohort_manifest.json" not in lock["files"]:
        raise ValueError("Official cache manifest is not sealed")
    manifest = read_json(root / "cohort_manifest.json")
    _reject_labels(manifest)
    contract = OFFICIAL[role]
    expected = {"schema": CACHE_SCHEMA, "role": role, "cohort_id": contract.cohort_id,
                "membership_sha256": contract.membership_sha256, "scene_count": contract.scene_count,
                "asset_group_count": contract.asset_group_count, "method_freeze_sha256": freeze_hash}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Official cache is incompatible with the current method or fixed official split")
    scenes = manifest.get("scenes", [])
    if any(not isinstance(scene,str) or Path(scene).name!=scene or scene in {'.','..'} for scene in scenes):
        raise ValueError('Official scene IDs must be safe opaque path tokens')
    if len(scenes) != contract.scene_count or len(scenes) != len(set(scenes)) or membership_sha256(scenes) != contract.model_membership_sha256:
        raise ValueError("Official opaque scene membership changed or is incomplete")
    if any(f"scenes/{scene}.json" not in lock["files"] for scene in scenes):
        raise ValueError("Official cache seal omits scene evidence")
    return manifest


def release_verified_cache(config, source_root=None):
    """Release an already typed neutral cache; never convert unknown source data."""
    freeze, prerequisites=verify_prerequisites(config,source_root)
    source=Path(config['cache_root']);root=Path(config['output_root']);role=config['role']
    manifest=validate_cache(source,role,prerequisites['method_freeze_sha256'])
    lock=verify_seal(source)
    allowed={'cohort_manifest.json',*(f'scenes/{scene}.json' for scene in manifest['scenes'])}
    for relative in lock['files']:
        path=source/relative
        media=relative.startswith('media/') and path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}
        tensor=relative.startswith('tensors/') and path.suffix.lower()=='.npz'
        if relative not in allowed and not media and not tensor:
            raise ValueError('Official model release contains an unexpected or non-model artifact')
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError('Official model resource must contain regular relative files')
    for scene in manifest['scenes']:
        payload=read_json(source/'scenes'/f'{scene}.json')
        if payload.get('scene')!=scene or payload.get('correspondence_provenance')!=manifest.get('correspondence_provenance'):
            raise ValueError('Official model scene identity or neutral provenance differs')
        for mode in ('automatic','assisted'):
            for selected in freeze['view_sets'].values():
                current_inputs(payload,selected,mode,source)
    binding={'schema':'vpi-current-official-release/v1','role':role,**prerequisites,
             'source_cache_lock_sha256':digest_file(source/'artifact_lock.json'),'source_sha256':digest_file(__file__)}
    bind(root,binding)
    if (root/'release_lock.json').exists():
        verify_seal(root,name='release_lock.json')
        return read_json(root/'release_receipt.json')
    required=sum((source/name).stat().st_size for name in lock['files'])+(source/'artifact_lock.json').stat().st_size
    if shutil.disk_usage(root).free < required*1.1:
        raise OSError('Official resource storage preflight failed')
    destination=root/'model-input'
    if destination.exists():
        verify_seal(destination)
        if digest_file(destination/'artifact_lock.json')!=binding['source_cache_lock_sha256']:
            raise ValueError('Existing released model resource changed')
    else:
        temporary=root/'model-input.partial'
        if temporary.exists():
            temporary.rename(root/('model-input.interrupted-'+uuid4().hex[:8]))
        temporary.mkdir()
        for relative in [*lock['files'],'artifact_lock.json']:
            target=temporary/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/relative,target)
        verify_seal(temporary)
        temporary.rename(destination)
    receipt={'schema':binding['schema'],'status':'verified_current_cache_released','role':role,
             'scene_count':manifest['scene_count'],'asset_group_count':manifest['asset_group_count'],
             'source_archive_opened':False,'evaluator_resources_copied':False,'model_lock_sha256':binding['source_cache_lock_sha256'],
             'native_source_bridge':'not performed; only a complete prevalidated current cache was released',
             'transport_recovery':'local atomic copy; export and verify files before a Kaggle session ends'}
    atomic_json(root/'release_receipt.json',receipt)
    seal(root,[root/'run_binding.json',root/'release_receipt.json',*(p for p in destination.rglob('*') if p.is_file())],name='release_lock.json')
    return receipt


def predict_official(config, source_root):
    """CPU decision over a complete compatible cache; no RGB/model/evaluator access."""
    from visual_peer_inspector.web.decision import decide_cached
    role, mode = config["role"], config.get("mode", "automatic")
    freeze, prerequisites = verify_prerequisites(config, source_root)
    manifest = validate_cache(config["cache_root"], role, prerequisites["method_freeze_sha256"])
    if mode not in {"automatic", "assisted"}:
        raise ValueError("Unknown prediction mode")
    provenance = manifest.get("correspondence_provenance", "predicted") if mode == "assisted" else "predicted"
    if mode == "assisted" and provenance not in {"dataset_reference", "user_confirmed"}:
        raise ValueError("Assisted predictions require neutral IDs with explicit provenance")
    views = freeze["view_sets"]
    adapter_hash = digest_file(inspect.getsourcefile(decide_cached))
    if freeze["source_sha256"].get("src/visual_peer_inspector/web/decision.py") != adapter_hash:
        raise ValueError("Decision adapter is not part of the current source freeze")
    root, cache = Path(config["output_root"]), Path(config["cache_root"])
    binding = {"schema": PREDICTION_SCHEMA, "role": role, "mode": mode, "correspondence_provenance": provenance,
               "cache_lock_sha256": digest_file(cache / "artifact_lock.json"), "view_sets": views,
               "adapter_sha256": adapter_hash, "implementation_sha256": digest_file(__file__),
               "membership_sha256": manifest["membership_sha256"],
               "model_membership_sha256": membership_sha256(manifest["scenes"]), **prerequisites}
    binding_hash = bind(root, binding)
    if (root / "prediction_lock.json").exists():
        validate_prediction(root, role, binding, cache_root=cache)
        return read_json(root / "prediction_receipt.json")
    with terminal_status(root, f"official-{role}-{mode}"):
        for index, scene in enumerate(manifest["scenes"], 1):
            unit = root / "scenes" / scene
            if verified_unit(unit, binding_hash):
                continue
            payload = read_json(cache / "scenes" / f"{scene}.json")
            if payload.get("scene") != scene or (mode == "assisted" and payload.get("correspondence_provenance") != provenance):
                raise ValueError("Cached scene or correspondence provenance differs from its manifest")
            settings = []
            for schedule, selected in views.items():
                rows, observations = current_inputs(payload, selected, mode,cache)
                result = decide_cached(rows, observations, views=list(selected), mode=mode)
                score_rows = [{"view_id": view, "candidate_id": obj["candidate_id"],
                               "identity_id": result["identities"].get((view, obj["candidate_id"])),
                               "component_scores": obj["scores"],
                               "normalized_rank": result["component_rank"].get((view, obj["candidate_id"]))}
                              for view in selected for obj in rows[view]["objects"]]
                settings.append({"view_set_id": schedule, "views": selected, "mode": mode,
                                 "correspondence_provenance": provenance, "ranking": result["ranking"], "tracks": result.get("tracks", []),
                                 "observation_scores": score_rows,
                                 "assisted_scores": result["scored_observations"]})
            with atomic_unit(unit, binding_hash) as temporary:
                atomic_json(temporary / "predictions.json", {"scene": scene, "settings": settings})
            if index % 25 == 0 or index == len(manifest["scenes"]):
                print(f"Official {role}: {index}/{len(manifest['scenes'])} scenes checkpointed", flush=True)
        receipt = {"schema": PREDICTION_SCHEMA, "status": "prediction_complete", "role": role, "mode": mode,
                   "scene_count": len(manifest["scenes"]), "asset_group_count": manifest["asset_group_count"],
                   "settings_per_scene": len(views), "correspondence_provenance": provenance,
                   "anomaly_labels_opened": False, "neural_inference_run": False, "metric_evaluation_run": False,
                   "asset_attribution_verified": False, "visual_report_status": "requires compatible sealed per-view images, boxes and masks"}
        atomic_json(root / "prediction_receipt.json", receipt)
        files = [root / "run_binding.json", root / "prediction_receipt.json"]
        files.extend(p for p in (root / "scenes").rglob("*") if p.is_file() and ".partial" not in str(p) and ".interrupted-" not in str(p))
        seal(root, files, name="prediction_lock.json", metadata={"binding_sha256": binding_hash})
        validate_prediction(root, role, binding, cache_root=cache)
        return receipt


def validate_prediction(root, role, expected_binding=None, cache_root=None):
    from .official_evaluation import canonical_setting
    from .original import _validate_setting
    root = Path(root)
    lock = verify_seal(root, name="prediction_lock.json")
    if not {"run_binding.json", "prediction_receipt.json"} <= set(lock["files"]):
        raise ValueError("Official prediction seal omits its binding or receipt")
    binding, receipt = read_json(root / "run_binding.json"), read_json(root / "prediction_receipt.json")
    contract = OFFICIAL[role]
    if binding.get("schema") != PREDICTION_SCHEMA or binding.get("role") != role or receipt.get("status") != "prediction_complete":
        raise ValueError("Legacy P1 or wrong-role prediction is not a current official result")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("Official prediction changed its frozen inputs")
    if binding.get('mode') not in {'automatic', 'assisted'} or receipt.get('mode') != binding['mode'] or receipt.get('correspondence_provenance') != binding.get('correspondence_provenance'):
        raise ValueError('Official prediction mode or correspondence provenance changed')
    if any(receipt.get(key) != value for key, value in {"schema": PREDICTION_SCHEMA, "role": role,
           "scene_count": contract.scene_count, "asset_group_count": contract.asset_group_count,
           "anomaly_labels_opened": False, "metric_evaluation_run": False}.items()):
        raise ValueError("Official prediction receipt changed its complete role or evaluation boundary")
    paths = sorted(name for name in lock["files"] if name.startswith("scenes/") and name.endswith("/predictions.json"))
    scenes = [Path(name).parent.name for name in paths]
    if len(paths) != contract.scene_count or membership_sha256(scenes) != contract.model_membership_sha256:
        raise ValueError("Official prediction scene coverage is incomplete or changed")
    if cache_root is not None:
        validate_cache(cache_root,role,binding['method_freeze_sha256'])
        if digest_file(Path(cache_root)/'artifact_lock.json')!=binding['cache_lock_sha256']:
            raise ValueError('Official prediction inventory belongs to a different cache')
    for name in paths:
        payload = read_json(root / name)
        if payload["scene"] != Path(name).parent.name:
            raise ValueError("Official prediction scene identity is inconsistent")
        settings = payload["settings"]
        actual = {row["view_set_id"]: row["views"] for row in settings}
        if len(actual) != len(settings) or actual != binding["view_sets"] or any(row["mode"] != binding["mode"] for row in settings):
            raise ValueError("Official prediction has missing or changed view schedules")
        cached=read_json(Path(cache_root)/'scenes'/f"{payload['scene']}.json") if cache_root is not None else None
        for setting in settings:
            if setting.get('correspondence_provenance')!=binding['correspondence_provenance']:
                raise ValueError('Official setting changes correspondence provenance')
            canonical=canonical_setting(setting)
            _validate_setting(canonical)
            keys={(row['view_id'],row['candidate_id']) for row in setting['observation_scores']}
            if cached is not None:
                rows,_=current_inputs(cached,setting['views'],binding['mode'],cache_root)
                expected={(view,obj['candidate_id']) for view,values in rows.items() for obj in values['objects']}
                if keys!=expected:
                    raise ValueError('Official prediction changed the bound candidate inventory')
            for observation in setting['observation_scores']:
                values=[observation['normalized_rank'],*observation['component_scores'].values()]
                if any(value is not None and not math.isfinite(float(value)) for value in values):
                    raise ValueError('Official prediction contains nonfinite observation scores')
            if binding['mode']=='assisted':
                scored=setting.get('assisted_scores') or []
                scored_keys=[(row['view_id'],row['candidate_id']) for row in scored]
                if len(scored_keys)!=len(set(scored_keys)) or set(scored_keys)!=keys:
                    raise ValueError('Assisted score coverage differs from retained candidates')
                for observation in scored:
                    bank=observation['object_bank']
                    if any(value is not None and not math.isfinite(float(value)) for value in [bank['score'],*bank['patch_scores']]):
                        raise ValueError('Assisted bank scores are nonfinite')
    return {"binding": binding, "receipt": receipt, "prediction_lock_sha256": digest_file(root / "prediction_lock.json"),
            "candidate_inventory_verified": cache_root is not None}


def finalize_official(config, source_root=None):
    """Verify both official roles; never pool scenes or read labels to make a result."""
    freeze, prerequisites = verify_prerequisites(config, source_root)
    records = {role: validate_prediction(config[f"{role}_prediction_root"], role, cache_root=config['cache_roots'][role]) for role in OFFICIAL}
    for record in records.values():
        if any(record["binding"].get(key) != value for key, value in prerequisites.items()):
            raise ValueError("Official prediction is not linked to the current09D and exposure gates")
        if record["binding"]["view_sets"] != freeze["view_sets"]:
            raise ValueError("Official view schedules differ from the current freeze")
        if record['binding']['adapter_sha256'] != freeze['source_sha256'].get('src/visual_peer_inspector/web/decision.py'):
            raise ValueError('Official prediction used a different decision adapter')
    shared = ("method_freeze_sha256", "mode", "correspondence_provenance", "view_sets", "adapter_sha256")
    if any(records["seen"]["binding"][key] != records["unseen"]["binding"][key] for key in shared):
        raise ValueError("Seen and unseen methods, modes, correspondence or schedules do not match")
    root = Path(config["output_root"])
    binding = {"schema": "vpi-current-official-summary/v1", **prerequisites,
               "prediction_locks": {role: record["prediction_lock_sha256"] for role, record in records.items()},
               "implementation_sha256": digest_file(__file__)}
    bind(root, binding)
    if (root / "summary_lock.json").exists():
        verify_seal(root, name="summary_lock.json")
        return read_json(root / "shareable/official_prediction_summary.json")
    summary = {"schema": binding["schema"], "status": "predictions_verified", "accuracy_evaluated": False,
               "official_roles_separate_from_original_2000": True,
               "mode": records["seen"]["binding"]["mode"],
               "correspondence_provenance": records["seen"]["binding"]["correspondence_provenance"],
               "view_sets": freeze["view_sets"], "prediction_locks": binding["prediction_locks"],
               "roles": [{"role": role, "scene_count": OFFICIAL[role].scene_count,
                          "asset_group_count": OFFICIAL[role].asset_group_count} for role in OFFICIAL],
               "next": "10B requires a separately validated official evaluator adapter; no pooled primary metric."}
    atomic_json(root / "shareable/official_prediction_summary.json", summary)
    seal(root, [root / "run_binding.json", root / "shareable/official_prediction_summary.json"], name="summary_lock.json")
    return summary
