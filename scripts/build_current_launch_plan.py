#!/usr/bin/env python3
"""Prepare the bounded queue using existing Kaggle resources and typed outputs."""
from pathlib import Path
import argparse
import json
import math
from current_launch import build_kernels,validate_plan

OWNER='esreekarreddy/'
PREP='notebooks/input-preparation/current-kaggle-preparation.ipynb'
NATIVE='notebooks/input-preparation/current-native-regions-to-object-cache.ipynb'
REFERENCE='notebooks/input-preparation/current-native-reference-conversion.ipynb'
COUNTS={'development':1000,'calibration':600,'internal_lock':400,'seen':715,'unseen':2437}


def resource(name,path='',sha256=False):return {'$resource':name,**({'path':path} if path else {}),**({'sha256':True} if sha256 else {})}
def output(path):return {'$output':path}


def plan():
    resources={};jobs=[]
    def data(name,slug,marker,suffixes=('',),**extra):
        resources[name]={'kind':'dataset','ref':OWNER+slug,'marker':marker,'suffixes':list(suffixes),**extra}
    def produced(name,job,marker,suffix,**extra):
        resources[name]={'kind':'kernel','ref':OWNER+'vpi-current-'+job.replace('_','-'),'producer_job':job,
            'marker':marker,'suffixes':[suffix],**extra}
    def add(name,action,config,*,nb=PREP,deps=(),runtime='cpu',boundary='trusted_preparation',purpose=None):
        cfg={'action':action,**config}
        cfg.setdefault('output_root',output(name))
        jobs.append({'id':name,'purpose':purpose or action.replace('_',' '),'notebook':nb,'runtime':runtime,
            'input_boundary':boundary,'depends_on':list(deps),'config':cfg})
    data('source','vpi-current-evaluation-source','prepared_source_manifest.json',('source',''),availability='prepare_upload')
    data('support','vpi-current-evaluation-support','support_manifest.json',availability='prepare_upload')
    data('raw','toysad8k-full-private-eda','package-checksums.sha256',('','toysad8k-full-private-eda'),availability='existing_kaggle',contains_outcomes=True)
    data('selection','vpi-toysad-selection-private','experiment_roles.jsonl',('toysad8k-train-2000-v2',),availability='existing_kaggle')
    data('administrative','vpi-toysad-selection-private','restricted_source_map.jsonl',('toysad8k-train-2000-v2-administrative-vault',),availability='existing_kaggle')
    for key,slug,suffixes in [('screen','vpi-toysad-ds1-screen100-model-input',('model-input','')),
        ('remainder','vpi-toysad-development-remainder-model-input',('development-remainder/model-input','model-input','')),
        ('confirm','vpi-toysad-ds2-confirmation-rgb',('model-input',''))]:
        data('rgb-'+key,slug,'package_record.json',suffixes,availability='existing_kaggle')
    resources['development-regions']={'kind':'kernel','ref':OWNER+'toysad-8k-frozen-development-pipeline',
        'marker':'full-development-lock.json','suffixes':['toysad8k_frozen_development_pipeline'],
        'required_sha256':{'full-development-lock.json':'9d8505d2bccdcf9d6e59ebf39c52676278517ff6564c391aa88cb4fb113c050c'},'availability':'existing_kaggle'}
    data('reuse100','vpi-current-development100-feature-reuse','feature_shard_lock.json',availability='prepared_local_upload')
    for profile in ('cpu','dino','regions','tracking'):
        name='setup-'+profile
        cfg={'profile':profile,'action':'prepare','build_wheelhouse':True,
            'wheelhouse':f'/kaggle/working/{name}/wheelhouse','target':f'/tmp/vpi-runtime-{name}',
            'output_root':f'/kaggle/working/{name}','download_models':profile=='dino',
            **({'model_asset_root':f'/kaggle/working/{name}/model-assets','models_only':'all'} if profile=='dino' else {})}
        jobs.append({'id':name,'purpose':'Setup only: resolve/install pinned Linux dependencies and record actual runtime; no dataset experiment',
            'notebook':'notebooks/input-preparation/current-kaggle-runtime-setup.ipynb','runtime':'cpu',
            'setup_job':True,'depends_on':[],'config':cfg,'attach_resources':[]})
        produced('runtime-'+profile,name,'runtime_profile_lock.json',name+'/runtime-profile')
        produced('wheels-'+profile,name,'wheelhouse_manifest.json',name+'/wheelhouse')
    produced('models','setup-dino','model_assets_receipt.json','setup-dino/model-assets')
    add('freeze','freeze',{'reference_root':resource('support','method-reference')},deps=['setup-cpu'])
    produced('freeze','freeze','method_freeze.json','freeze')
    # Original exposure indexes are already derived and reviewed locally. Official
    # indexes are derived from the canonical raw role assignments below.
    for role in COUNTS:
        cfg={'role':role,'raw_root':resource('raw')}
        if role not in {'seen','unseen'}:cfg.update(selection_root=resource('selection'),administrative_root=resource('administrative'))
        add(role+'-index','rgb_index',cfg,deps=['setup-cpu'])
        produced(role+'-index',role+'-index','rgb_index.json',role+'-index')
        add(role+'-geometry','raw_geometry',{**cfg,'rgb_index_root':resource(role+'-index')},deps=[role+'-index'])
        produced(role+'-geometry',role+'-geometry','geometry_manifest.json',role+'-geometry')
    add('official-exposure','official_exposure',{'index_roots':{role:resource(role+'-geometry') for role in ('seen','unseen')},
        'policy':'exposure_qualified_fixed_methods'},deps=['seen-geometry','unseen-geometry'])
    produced('official-exposure','official-exposure','official_exposure_ledger.json','official-exposure')
    last_evaluation=None
    for role,count in COUNTS.items():
        common={'role':role,'method_freeze_root':resource('freeze'),'exposure_root':resource('support','original-exposure')}
        if role=='internal_lock':common['calibration_receipt_root']=resource('09b')
        if role in {'seen','unseen'}:
            common.update(freeze_root=resource('freeze'),decision_root=resource('09d'),official_exposure_root=resource('official-exposure'))
        deps=['freeze',role+'-geometry']+([last_evaluation] if last_evaluation else [])
        if role=='development':
            components={}
            for suffix,key in [('screen-100','screen'),('remainder-600','remainder'),('confirm-300','confirm')]:
                cohort='toysad8k-development-'+suffix+'-v1'
                components[cohort]={'region_root':resource('development-regions',cohort+'/model-safe'),'rgb_root':resource('rgb-'+key)}
            regions={'region_format':'native-development-components','components':components,
                'full_development_lock':resource('development-regions','full-development-lock.json'),
                'expected_region_lock_sha256':resource('development-regions','full-development-lock.json',True)}
        else:
            add(role+'-rgb','rgb_release',{**common,'raw_root':resource('raw'),'rgb_index_root':resource(role+'-index'),
                'maximum_new_scenes':count},deps=deps)
            produced(role+'-rgb',role+'-rgb','package_record.json',role+'-rgb/model-input')
            region_jobs=[]
            for i in range(math.ceil(count/100)):
                name=f'{role}-regions-{i:02d}'
                add(name,'extract',{**common,'rgb_root':resource(role+'-rgb'),
                    'locate_model_root':resource('models','locateanything'),
                    'sam_checkpoint':resource('models','sam2/sam2.1_hiera_large.pt'),
                    'shard_count':math.ceil(count/100),'shard_index':i,
                    'maximum_new_views':2000,'attempt_timeout_seconds':600},
                    nb='notebooks/input-preparation/current-native-region-inference.ipynb',
                    deps=[role+'-rgb','setup-regions','setup-dino'],runtime='regions-gpu',boundary='model_only',
                    purpose='Fixed LocateAnything/SAM regions for at most100 scenes; checkpoint each phase')
                produced(name,name,'region_shard_lock.json',name);region_jobs.append(name)
            add(role+'-regions','region_merge',{**common,'rgb_root':resource(role+'-rgb'),
                'region_shards':[resource(name) for name in region_jobs]},deps=region_jobs,boundary='model_only')
            produced(role+'-regions',role+'-regions','artifact_lock.json',role+'-regions')
            regions={'region_format':'current-native-regions','region_root':resource(role+'-regions'),'rgb_root':resource(role+'-rgb'),
                'expected_region_lock_sha256':resource(role+'-regions','artifact_lock.json',True)}
            deps=[role+'-regions']
        add(role+'-neutral','neutral',{**common,**regions,'geometry_root':resource(role+'-geometry')},deps=deps)
        produced(role+'-neutral',role+'-neutral','neutral_correspondence.json',role+'-neutral')
        shards=[]
        for i in range(math.ceil(count/100)):
            name=f'{role}-cache-{i:02d}'
            cfg={**common,**regions,'neutral_root':resource(role+'-neutral'),'model_source':resource('models','dino'),
                'feature_root':output(name+'-features'),'shard_count':math.ceil(count/100),'shard_index':i,
                'max_new_units':2000,'max_new_scenes':100,'maximum_output_bytes':15*1024**3}
            if role=='development':cfg['feature_roots']=[resource('reuse100')]
            add(name,'prepare_shard',cfg,nb=NATIVE,deps=[role+'-neutral','setup-dino'],runtime='dino-cpu',boundary='model_only',
                purpose='Reuse compatible features and extract only missing fixed DINO crops; assemble at most100 scenes')
            jobs[-1]['compact_checkpoints']=True
            produced(name,name,'cache_shard_lock.json',name);shards.append(name)
        add(role+'-cache','cache_merge',{**common,'cache_shards':[resource(name) for name in shards],
            'maximum_output_bytes':18*1024**3},deps=shards,boundary='model_only')
        produced(role+'-cache',role+'-cache','cohort_manifest.json',role+'-cache')
        if role in {'development','calibration','internal_lock'}:
            pred={'development':'development-predict','calibration':'09a','internal_lock':'09c'}[role]
            evaluation={'development':'development-evaluate','calibration':'09b','internal_lock':'09d'}[role]
            cache=role+'-cache';pred_deps=[cache]
            if role!='development':
                release='08f1' if role=='calibration' else '09b1'
                filename='08f1-toysad8k-calibration-resource-release.ipynb' if role=='calibration' else '09b1-toysad8k-internal-lock-resource-release.ipynb'
                add(release,'release',{**common,'cache_root':resource(cache)},nb='notebooks/protected-evaluation/'+filename,deps=[cache],boundary='model_only')
                produced(release,release,'cohort_manifest.json',release+'/model-cache');cache=release;pred_deps=[release]
            pred_nb=NATIVE if role=='development' else 'notebooks/protected-evaluation/'+('09a-toysad8k-protected-calibration-inference.ipynb' if role=='calibration' else '09c-toysad8k-protected-internal-lock-inference.ipynb')
            add(pred,'dev-predict' if role=='development' else 'predict',{**common,'cache_root':resource(cache)},nb=pred_nb,deps=pred_deps,boundary='model_only')
            produced(pred,pred,'prediction_lock.json',pred)
            add(role+'-truth','raw_truth',{**common,'raw_root':resource('raw'),'administrative_root':resource('administrative'),
                'rgb_index_root':resource(role+'-index'),'geometry_root':resource(role+'-geometry'),
                'cache_root':resource(cache),'prediction_root':resource(pred)},deps=[pred])
            produced(role+'-truth',role+'-truth','current_evaluator_manifest.json',role+'-truth',contains_outcomes=True)
            eval_nb=NATIVE if role=='development' else 'notebooks/protected-evaluation/'+('09b-toysad8k-calibration-review-policy.ipynb' if role=='calibration' else '09d-toysad8k-internal-lock-one-shot-evaluation.ipynb')
            add(evaluation,'dev-evaluate' if role=='development' else 'evaluate',{**common,'cache_root':resource(cache),'prediction_root':resource(pred),
                'evaluator_root':resource(role+'-truth'),'report_root':output(evaluation+'-report')},nb=eval_nb,deps=[role+'-truth'])
            produced(evaluation,evaluation,'evaluation_lock.json',evaluation,contains_outcomes=True)
            last_evaluation=evaluation
        else:
            release='10r-'+role
            add(release,'release',{**common,'cache_root':resource(role+'-cache')},nb='notebooks/official-evaluation/10r-toysad8k-official-release.ipynb',deps=[role+'-cache'],boundary='model_only')
            produced(release,release,'cohort_manifest.json',release+'/model-input')
            predictions={}
            for mode in ('automatic','assisted'):
                name=('10a' if role=='seen' else '10a1')+'-'+mode
                filename='10a-toysad8k-official-seen-frozen-inference.ipynb' if role=='seen' else '10a1-toysad8k-official-unseen-frozen-inference.ipynb'
                add(name,'predict',{**common,'mode':mode,'cache_root':resource(release)},nb='notebooks/official-evaluation/'+filename,deps=[release],boundary='model_only')
                produced(name,name,'prediction_lock.json',name);predictions[mode]=resource(name)
            add(role+'-truth','raw_truth',{**common,'raw_root':resource('raw'),'rgb_index_root':resource(role+'-index'),
                'geometry_root':resource(role+'-geometry'),'cache_root':resource(release),'prediction_roots':predictions},deps=[('10a' if role=='seen' else '10a1')+'-'+mode for mode in ('automatic','assisted')])
            produced(role+'-truth',role+'-truth','current_evaluator_manifest.json',role+'-truth',contains_outcomes=True)
    official_common={'freeze_root':resource('freeze'),'exposure_root':resource('support','original-exposure'),
        'decision_root':resource('09d'),'official_exposure_root':resource('official-exposure')}
    roots={mode:{role:resource(('10a' if role=='seen' else '10a1')+'-'+mode) for role in ('seen','unseen')} for mode in ('automatic','assisted')}
    summaries={}
    for mode in ('automatic','assisted'):
        name='10a-finalize-'+mode
        add(name,'finalize',{**official_common,'mode':mode,'prediction_roots':roots[mode],
            'cache_roots':{role:resource('10r-'+role) for role in ('seen','unseen')}},
            nb='notebooks/official-evaluation/10a-finalize-toysad8k-official-prediction-summary.ipynb',deps=['10a-'+mode,'10a1-'+mode],boundary='model_only')
        produced(name,name,'summary_lock.json',name);summaries[mode]=resource(name)
    add('10b','evaluate',{**official_common,'prediction_roots':roots,'prediction_summary_roots':summaries,
        'cache_roots':{role:resource('10r-'+role) for role in ('seen','unseen')},
        'evaluator_roots':{role:resource(role+'-truth') for role in ('seen','unseen')},'report_root':output('10b-report')},
        nb='notebooks/official-evaluation/10b-toysad8k-official-seen-unseen-evaluation.ipynb',
        deps=['10a-finalize-automatic','10a-finalize-assisted','seen-truth','unseen-truth'])
    produced('10b','10b','evaluation_lock.json','10b',contains_outcomes=True)
    for job in jobs:
        if job['id'] in {'development-evaluate','09b','09d','10b'}:
            job['config']['publish_tracking']=True
            job['attach_resources']=['runtime-tracking','wheels-tracking']
    # Resource dependencies and runtime setup are executable queue prerequisites.
    from current_launch import referenced_resources
    for job in jobs:
        required={resources[name]['producer_job'] for name in (referenced_resources(job['config']) | set(job.get('attach_resources',[]))) if resources[name].get('producer_job')}
        if not job.get('setup_job'):
            required.add('setup-'+{'cpu':'cpu','dino-cpu':'dino','regions-gpu':'regions'}[job['runtime']])
        job['depends_on']=sorted(set(job.get('depends_on',[])) | required)
    return {'schema':'vpi-current-kaggle-launch-plan/v1','models_fixed':True,'raw_dataset_reupload_required':False,
        'runtime_readiness':{'sam2_code':{'resource':'wheels-regions','distribution':'SAM-2',
            'repository':'https://github.com/facebookresearch/sam2',
            'revision':'2b90b9f5ceec907a1c18123530e92e794ad901a4',
            'preflight_marker':'sam2_preflight_verified'}},
        'original_roles':{r:COUNTS[r] for r in ('development','calibration','internal_lock')},
        'official_roles':{r:COUNTS[r] for r in ('seen','unseen')},'methods':['winner_cluster_consensus','object_bank_top2'],
        'core_views':[2,3,5],'diagnostic_views':[1],'schedule_count':19,'resources':resources,'jobs':jobs,
        'scientific_runs_started':False,'queue_note':'Repeated cache jobs are100-scene resource shards of fixed methods, not new scoring experiments.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();value=plan();print(json.dumps(validate_plan(value)));print(json.dumps(build_kernels(value,args.source_root,args.output_root)))


if __name__=='__main__':main()
