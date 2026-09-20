"""Official current-schema evaluation; prediction verification precedes truth access."""
from pathlib import Path
from time import perf_counter
import pandas as pd
from vpi_object_bank.storage import atomic_json, atomic_unit, bind, digest_file, read_json, seal, verify_seal, verified_unit, terminal_status
from vpi_object_bank.evaluation import evaluate_setting, _metric_summary, _diagnostic_summary, _atomic_csv, FAILURES, IDENTITY_METRICS
from .contracts import METHODS
from .exposure import verify_exposure
from .official import OFFICIAL, verify_prerequisites, validate_prediction, resolve_input_root
from .original import _validate_setting

SCHEMA = 'vpi-current-official-evaluator/v1'

def canonical_setting(setting):
    mode=setting['mode']
    bank={(row['view_id'],row['candidate_id']):row for row in setting.get('assisted_scores') or []}
    observations=[]
    for row in setting['observation_scores']:
        key=(row['view_id'],row['candidate_id'])
        item={'view_id':key[0],'candidate_id':key[1],'component_scores':row['component_scores'],
              'normalized_score':row['normalized_rank']}
        if key in bank:
            item['object_bank']=bank[key]['object_bank']
        observations.append(item)
    return {'mode':mode,'route':'automatic_predicted' if mode=='automatic' else 'assisted_correspondence',
            'region':'automatic','method':METHODS[mode],'view_set_id':setting['view_set_id'],
            'views':setting['views'],'view_count':len(setting['views']),
            'correspondence_provenance':setting['correspondence_provenance'],
            'observations':observations,'rankings':{METHODS[mode]:setting['ranking']}, 'tracks':setting.get('tracks',[])}


def evaluate_official(config, source_root=None):
    started=perf_counter()
    freeze, prerequisites=verify_prerequisites(config,source_root)
    roots=config['prediction_roots']
    if set(roots) != {'automatic','assisted'} or any(set(roles)!=set(OFFICIAL) for roles in roots.values()):
        raise ValueError('Official evaluation requires separate seen/unseen predictions for both fixed modes')
    records,scenes={},{}
    # Validate every prediction and physical-object setting before evaluator access.
    for mode in ('automatic','assisted'):
        for role in OFFICIAL:
            root=Path(roots[mode][role])
            record=validate_prediction(root,role,cache_root=config['cache_roots'][role])
            binding=record['binding']
            if binding['mode'] != mode or any(binding.get(k)!=v for k,v in prerequisites.items()) or binding['view_sets']!=freeze['view_sets']:
                raise ValueError('Official predictions and current prerequisites differ')
            if binding['adapter_sha256']!=freeze['source_sha256'].get('src/visual_peer_inspector/web/decision.py'):
                raise ValueError('Official predictions used an unregistered adapter')
            records[(mode,role)]=record
            names=sorted(path.parent.name for path in root.glob('scenes/*/predictions.json'))
            if role in scenes and scenes[role]!=names:
                raise ValueError('Automatic and assisted official scenes differ')
            scenes[role]=names
            for scene in names:
                for setting in read_json(root/'scenes'/scene/'predictions.json')['settings']:
                    if setting.get('correspondence_provenance')!=binding['correspondence_provenance']:
                        raise ValueError('Official prediction contains mixed assistance provenance')
                    _validate_setting(canonical_setting(setting))
    provenance={records[('assisted',role)]['binding']['correspondence_provenance'] for role in OFFICIAL}
    if len(provenance)!=1 or not provenance <= {'dataset_reference','user_confirmed'}:
        raise ValueError('Report dataset-reference and user-confirmed assistance as separate experiments')
    summary_locks={}
    for mode in ('automatic','assisted'):
        summary_root=Path(config['prediction_summary_roots'][mode])
        verify_seal(summary_root,name='summary_lock.json')
        summary=read_json(summary_root/'shareable/official_prediction_summary.json')
        if summary['mode']!=mode or summary['view_sets']!=freeze['view_sets'] or summary['prediction_locks']!={role:records[(mode,role)]['prediction_lock_sha256'] for role in OFFICIAL}:
            raise ValueError('Official finalizer summary is incompatible')
        summary_locks[mode]=digest_file(summary_root/'summary_lock.json')
    audit=read_json(Path(config['official_exposure_root'])/'official_exposure_ledger.json')
    evaluators,indexes,evaluator_locks={},{},{}
    for role,contract in OFFICIAL.items():
        evaluator=resolve_input_root(Path(config['evaluator_roots'][role]),'current_evaluator_manifest.json')
        lock=verify_seal(evaluator)
        if not {'current_evaluator_manifest.json','scene_index.json'}<=set(lock['files']):
            raise ValueError('Native official evaluator bridge must supply its current manifest and asset index')
        manifest=read_json(evaluator/'current_evaluator_manifest.json')
        index_hash=digest_file(evaluator/'scene_index.json')
        expected={'schema':SCHEMA,'role':role,'method_freeze_sha256':prerequisites['method_freeze_sha256'],
                  'membership_sha256':contract.membership_sha256,'asset_index_sha256':index_hash}
        if any(manifest.get(k)!=v for k,v in expected.items()) or index_hash not in audit['reviewed_source_sha256']:
            raise ValueError('Official evaluator membership, freeze or reviewed asset-index binding differs')
        for mode in ('automatic','assisted'):
            if manifest.get('cache_lock_sha256')!=records[(mode,role)]['binding']['cache_lock_sha256']:
                raise ValueError('Evaluator correspondence is not bound to the exact predicted candidate cache')
        index=read_json(evaluator/'scene_index.json')
        if set(index)!=set(scenes[role]) or len(set(index.values()))!=contract.asset_group_count:
            raise ValueError('Official evaluator changes scene or asset denominators')
        if any(f'scenes/{scene}.json' not in lock['files'] for scene in scenes[role]):
            raise ValueError('Official evaluator seal omits scene truth')
        evaluators[role],indexes[role]=evaluator,index
        evaluator_locks[role]=digest_file(evaluator/'artifact_lock.json')
    root=Path(config['output_root'])
    binding={'schema':SCHEMA,'stage':'10b',**prerequisites,'prediction_locks':{f'{m}/{r}':v['prediction_lock_sha256'] for (m,r),v in records.items()},
             'prediction_summary_lock_sha256':summary_locks,'evaluator_locks':evaluator_locks,
             'source_sha256':digest_file(__file__),'metric_source_sha256':digest_file(__import__('inspect').getsourcefile(evaluate_setting))}
    binding_hash=bind(root,binding)
    if (root/'evaluation_lock.json').exists():
        verify_seal(root,name='evaluation_lock.json')
        return read_json(root/'evaluation_receipt.json')
    rows=[]
    with terminal_status(root,'official-current-evaluation'):
        for role in OFFICIAL:
            for number,scene in enumerate(scenes[role],1):
                unit=root/'private/scenes'/role/scene
                if not verified_unit(unit,binding_hash):
                    truth=read_json(evaluators[role]/'scenes'/f'{scene}.json')
                    if set(truth)!={'labels','correspondence','asset_group'} or truth['asset_group']!=indexes[role][scene]:
                        raise ValueError('Official truth schema or reviewed asset identity differs')
                    scene_rows=[]
                    for mode in ('automatic','assisted'):
                        payload=read_json(Path(roots[mode][role])/'scenes'/scene/'predictions.json')
                        for raw in payload['settings']:
                            setting=canonical_setting(raw)
                            evaluated=evaluate_setting(scene,setting,truth)
                            known={truth['correspondence'][f"automatic/{o['view_id']}/{o['candidate_id']}"] for o in setting['observations']}-{None}
                            retained={truth['correspondence'][f"automatic/{m['view_id']}/{m['candidate_id']}"] for obj in setting['rankings'][METHODS[mode]] for m in obj['members']}-{None}
                            for row in evaluated:
                                row.update(role=role,mode=mode,correspondence_provenance=setting['correspondence_provenance'],
                                           identity_reference_coverage=len(retained & known)/len(known) if known else None)
                            scene_rows.extend(evaluated)
                    with atomic_unit(unit,binding_hash) as temporary:
                        atomic_json(temporary/'metrics.json',scene_rows)
                rows.extend(read_json(unit/'metrics.json'))
                if number%25==0 or number==len(scenes[role]):
                    print(f'Official {role}: {number}/{len(scenes[role])} scenes evaluated and checkpointed',flush=True)
        frame=pd.DataFrame(rows)
        _atomic_csv(root/'private/scene_metrics.csv',frame)
        grouping=['role','mode','correspondence_provenance','view_count']
        _atomic_csv(root/'shareable/metric_summary.csv',_metric_summary(frame,grouping))
        _atomic_csv(root/'shareable/view_schedule_summary.csv',_metric_summary(frame,[*grouping,'view_set_id']))
        for filename,metrics in [('failure_summary.csv',FAILURES),('identity_summary.csv',IDENTITY_METRICS)]:
            tables=[]
            for (role,mode,provenance_name),subset in frame.groupby(['role','mode','correspondence_provenance']):
                table=_diagnostic_summary(subset,metrics);table['role']=role;table['mode']=mode
                table['correspondence_provenance']=provenance_name;tables.append(table)
            _atomic_csv(root/'shareable'/filename,pd.concat(tables,ignore_index=True))
        _atomic_csv(root/'shareable/anomaly_count_summary.csv',_metric_summary(frame,[*grouping,'anomaly_count']))
        _atomic_csv(root/'shareable/majority_summary.csv',_metric_summary(frame,[*grouping,'strict_normal_majority']))
        receipt={'schema':'vpi-current-official-evaluation/v1','status':'evaluation_complete',**prerequisites,
                 'invocation_elapsed_seconds':perf_counter()-started,'runtime_scope':'this CPU invocation, including reused checkpoints; no neural latency claim',
                 'modes':['automatic','assisted'],'no_retuning':True,'uncertainty_unit':'asset_group',
                 'scene_denominator_preserved':True,'assisted_correspondence_provenance':next(iter(provenance)),
                 'prediction_summary_lock_sha256':summary_locks,'independent_confirmation_claim':False,
                 'original_role_exposure':verify_exposure('internal_lock', config['exposure_root']),
                 'roles':{role:{'scene_count':c.scene_count,'asset_group_count':c.asset_group_count} for role,c in OFFICIAL.items()}}
        atomic_json(root/'evaluation_receipt.json',receipt)
        atomic_json(root/'shareable/evaluation_receipt.json',receipt)
        paths=[root/'run_binding.json',root/'evaluation_receipt.json',*sorted((root/'shareable').glob('*'))]
        paths.extend(p for p in (root/'private').rglob('*') if p.is_file() and '.partial' not in str(p) and '.interrupted-' not in str(p))
        seal(root,paths,name='evaluation_lock.json')
    return receipt
