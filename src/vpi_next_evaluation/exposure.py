"""Exposure-qualified evaluation preserves original roles and uncertain access history.

This new policy never changes the historical strict gate or certifies untouched
assets. It permits measuring fixed methods on declared potentially exposed data.
"""
from collections import Counter
from pathlib import Path
from vpi_object_bank.storage import atomic_json,digest_file,read_json,seal,verify_seal
from vpi_remaining_evaluation.gates import CONTRACTS, verify_exposure as strict_exposure
from visual_peer_inspector.data.model_safe_package import membership_sha256

SCHEMA='vpi-asset-exposure-qualified-audit/v2'
CLAIM='Exposure-qualified fixed-method evaluation; prior access is incomplete and assets are potentially exposed. This is not independent confirmation.'


def verify_exposure(role,exposure_root):
    if exposure_root is None:
        return strict_exposure(role,exposure_root)
    root=Path(exposure_root)
    ledger=read_json(root/'exposure_ledger.json')
    if ledger.get('schema')!=SCHEMA:
        return strict_exposure(role,exposure_root)
    lock=verify_seal(root)
    if not {'exposure_ledger.json','asset_records.json'}<=set(lock['files']):
        raise ValueError('Exposure seal omits reviewed policy and asset inventory')
    if ledger.get('status')!='reviewed_with_uncertainty' or ledger.get('evaluation_policy')!='exposure_qualified_fixed_methods':
        raise ValueError('Qualified exposure policy must retain its explicit uncertainty')
    if ledger.get('independent_confirmation') is not False or ledger.get('no_retuning') is not True or ledger.get('untouched_assets_claim') is not False:
        raise ValueError('Qualified evaluation cannot claim independent or untouched performance')
    rows=read_json(root/'asset_records.json')
    sources=ledger.get('reviewed_source_sha256',[])
    if not sources or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in sources):
        raise ValueError('Exposure policy requires real reviewed source hashes')
    if len({(r['role'],r['asset_group']) for r in rows})!=len(rows):
        raise ValueError('Exposure asset inventory duplicates a role/asset')
    for name,contract in CONTRACTS.items():
        recorded=ledger['roles'][name]
        relative=f'indexes/{name}/scene_index.json'
        if relative not in lock['files'] or recorded['scene_index_sha256']!=digest_file(root/relative):
            raise ValueError('Exposure neutral scene index is not exactly sealed')
        index=read_json(root/relative)
        expected_membership=contract.model_membership_sha256 or contract.membership_sha256
        if len(index)!=contract.scene_count or membership_sha256(index)!=expected_membership or len(set(index.values()))!=contract.asset_group_count:
            raise ValueError('Qualified exposure changes original scene or asset membership')
        if recorded['membership_sha256']!=contract.membership_sha256 or recorded['scene_index_sha256'] not in sources:
            raise ValueError('Exposure scene index is not bound to reviewed original membership')
        subset=[r for r in rows if r['role']==name]
        if {r['asset_group']:r['scene_count'] for r in subset}!=dict(Counter(index.values())):
            raise ValueError('Exposure asset records do not match the neutral index')
        for row in subset:
            if row['history_status'] not in {'documented_development_reuse','unknown_prior_exploration'} or row['potentially_exposed'] is not True:
                raise ValueError('Unknown access must remain potentially exposed')
            for event in row['prior_access']:
                if set(event)!={'kind','source_sha256'} or event['source_sha256'] not in sources:
                    raise ValueError('Exposure event omits real source attribution')
                if event['kind'] not in {'rgb','predictions','reference_identities','evaluation_results','anomaly_labels'}:
                    raise ValueError('Unknown exposure event kind')
    selected=[r for r in rows if r['role']==role]
    if role not in CONTRACTS or not selected:
        raise ValueError('Unknown original role')
    return {'role':role,'status':'exposure_qualified','scene_count':CONTRACTS[role].scene_count,
            'asset_group_count':CONTRACTS[role].asset_group_count,'independent_confirmation':False,
            'untouched_assets_claim':False,'potentially_exposed_asset_groups':len(selected),
            'unknown_history_asset_groups':sum(r['history_status']=='unknown_prior_exploration' for r in selected),
            'claim':CLAIM,'exposure_lock_sha256':digest_file(root/'artifact_lock.json')}


def prepare_audit(cfg):
    """Project only neutral membership fields from the existing administrative map."""
    import json
    source=Path(cfg['administrative_source'])
    if digest_file(source)!=cfg['administrative_source_sha256']:
        raise ValueError('Administrative membership source changed')
    output=Path(cfg['output_root'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve an existing audit; choose a fresh output root')
    evidence=[]
    for item in cfg['reviewed_evidence']:
        path=Path(item['path'])
        if digest_file(path)!=item['sha256']:
            raise ValueError('Reviewed exposure evidence changed')
        evidence.append({'description':item['description'],'sha256':item['sha256']})
    if not evidence or not cfg.get('development_reuse_source_sha256') in {e['sha256'] for e in evidence}:
        raise ValueError('Development reuse requires a reviewed evidence source')
    indexes={role:{} for role in CONTRACTS}
    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            # No object names, anomaly answers, outcome scores or source order enter the audit.
            role,scene,asset=row['role'],row['opaque_scene_id'],row['source_base_asset_id']
            if role not in indexes or scene in indexes[role]:
                raise ValueError('Administrative source changes or duplicates original roles')
            indexes[role][scene]=asset
    records=[];roles={};source_hashes=[cfg['administrative_source_sha256'],*(e['sha256'] for e in evidence)]
    for role,index in indexes.items():
        contract=CONTRACTS[role]
        if len(index)!=contract.scene_count or membership_sha256(index)!=(contract.model_membership_sha256 or contract.membership_sha256) or len(set(index.values()))!=contract.asset_group_count:
            raise ValueError('Administrative map is not the exact original-role scene/asset membership')
        path=output/'indexes'/role/'scene_index.json';atomic_json(path,index);index_hash=digest_file(path);source_hashes.append(index_hash)
        roles[role]={'scene_count':len(index),'asset_group_count':len(set(index.values())),
                     'membership_sha256':contract.membership_sha256,'scene_index_sha256':index_hash}
        for asset,count in sorted(Counter(index.values()).items()):
            events=[{'kind':kind,'source_sha256':cfg['development_reuse_source_sha256']} for kind in ('rgb','predictions','reference_identities','evaluation_results')] if role=='development' else []
            records.append({'role':role,'asset_group':asset,'scene_count':count,'potentially_exposed':True,
                            'history_status':'documented_development_reuse' if role=='development' else 'unknown_prior_exploration',
                            'prior_access':events})
    ledger={'schema':SCHEMA,'status':'reviewed_with_uncertainty','evaluation_policy':'exposure_qualified_fixed_methods',
            'no_retuning':True,'independent_confirmation':False,'untouched_assets_claim':False,'roles':roles,
            'reviewed_source_sha256':sorted(set(source_hashes)),'reviewed_evidence':evidence,'claim':CLAIM,
            'unresolved_history':'Exact asset access in earlier Vaibhav exploratory runs is unavailable; absence of records is not evidence of no access.',
            'administrative_fields_used':['role','opaque_scene_id','source_base_asset_id']}
    atomic_json(output/'asset_records.json',records);atomic_json(output/'exposure_ledger.json',ledger)
    seal(output,[output/'asset_records.json',output/'exposure_ledger.json',*(output/'indexes').glob('*/scene_index.json')])
    return {'status':'exposure_qualified_audit_prepared','roles':{role:verify_exposure(role,output) for role in CONTRACTS},
            'new_anomaly_labels_opened':False,'previous_access_resolved':False}


OFFICIAL_SCHEMA = 'vpi-official-exposure-qualified-audit/v2'


def _official_contracts():
    from visual_peer_inspector.data.cohort_contract import OFFICIAL_SEEN_TEST_CONTRACT, OFFICIAL_UNSEEN_TEST_CONTRACT
    return {'seen': OFFICIAL_SEEN_TEST_CONTRACT, 'unseen': OFFICIAL_UNSEEN_TEST_CONTRACT}


def verify_qualified_official_exposure(root):
    """Validate a neutral official inventory without certifying missing history."""
    root = Path(root)
    lock = verify_seal(root)
    ledger = read_json(root / 'official_exposure_ledger.json')
    rows = read_json(root / 'asset_records.json')
    if not {'official_exposure_ledger.json', 'asset_records.json'} <= set(lock['files']):
        raise ValueError('Official exposure seal omits its policy and asset inventory')
    expected = {'schema': OFFICIAL_SCHEMA, 'status': 'reviewed_with_uncertainty',
        'evaluation_policy': 'exposure_qualified_fixed_methods', 'no_retuning': True,
        'independent_confirmation': False, 'untouched_assets_claim': False}
    if any(ledger.get(key) != value for key, value in expected.items()):
        raise ValueError('Official qualified evaluation cannot claim resolved or independent access')
    sources = ledger.get('reviewed_source_sha256', [])
    if not sources or any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in sources):
        raise ValueError('Official exposure requires verified neutral source digests')
    contracts = _official_contracts()
    keys = [(row['role'], row['asset_group']) for row in rows]
    if len(keys) != len(set(keys)) or any(role not in contracts for role, _ in keys):
        raise ValueError('Official exposure duplicates or changes an asset role')
    summary = {}
    for role, contract in contracts.items():
        relative = f'indexes/{role}/scene_index.json'
        recorded = ledger.get('roles', {}).get(role, {})
        if relative not in lock['files'] or recorded.get('scene_index_sha256') != digest_file(root / relative):
            raise ValueError('Official exposure omits its exact sealed scene index')
        index = read_json(root / relative)
        if (len(index) != contract.scene_count or membership_sha256(index) != contract.model_membership_sha256
                or len(set(index.values())) != contract.asset_group_count
                or recorded.get('membership_sha256') != contract.membership_sha256
                or recorded['scene_index_sha256'] not in sources):
            raise ValueError('Official exposure changes fixed scene or asset membership')
        subset = [row for row in rows if row['role'] == role]
        if {row['asset_group']: row['scene_count'] for row in subset} != dict(Counter(index.values())):
            raise ValueError('Official exposure asset counts differ from its neutral index')
        for row in subset:
            if (row.get('potentially_exposed') is not True
                    or row.get('history_status') != 'unknown_prior_exploration'
                    or row.get('prior_access') != []):
                raise ValueError('Unknown official history cannot be converted into documented no-access')
        summary[role] = {'scene_count': len(index), 'asset_group_count': len(subset),
            'unknown_history_asset_groups': len(subset), 'potentially_exposed_asset_groups': len(subset)}
    return {'official_exposure_status': 'exposure_qualified',
        'official_exposure_lock_sha256': digest_file(root / 'artifact_lock.json'),
        'untouched_claim': False, 'independent_confirmation': False,
        'evaluation_claim': CLAIM, 'official_roles': summary}


def prepare_official_exposure(cfg):
    """Seal exact official neutral indexes while retaining unknown prior access."""
    if cfg.get('policy') != 'exposure_qualified_fixed_methods':
        raise ValueError('Explicit exposure-qualified official policy is required')
    contracts = _official_contracts()
    if set(cfg.get('index_roots', {})) != set(contracts):
        raise ValueError('Both fixed official role indexes must be supplied separately')
    output = Path(cfg['output_root'])
    sources, indexes, bindings = [], {}, {}
    for role, contract in contracts.items():
        source = Path(cfg['index_roots'][role])
        lock = verify_seal(source)
        if not {'geometry_manifest.json', 'scene_index.json'} <= set(lock['files']):
            raise ValueError('Official neutral index lacks a sealed native geometry manifest')
        manifest = read_json(source / 'geometry_manifest.json')
        expected = {'schema': 'vpi-current-reference-geometry/v1', 'role': role,
            'scene_count': contract.scene_count, 'asset_group_count': contract.asset_group_count,
            'membership_sha256': contract.membership_sha256, 'anomaly_labels_opened': False,
            'asset_index_sha256': digest_file(source / 'scene_index.json')}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('Official neutral geometry metadata differs from fixed membership')
        index = read_json(source / 'scene_index.json')
        if (not isinstance(index, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in index.items())
                or len(index) != contract.scene_count or len(set(index.values())) != contract.asset_group_count
                or membership_sha256(index) != contract.model_membership_sha256):
            raise ValueError('Official neutral scene or asset membership changed')
        indexes[role] = index
        bindings[role] = {'geometry_lock_sha256': digest_file(source / 'artifact_lock.json'),
            'geometry_manifest_sha256': digest_file(source / 'geometry_manifest.json'),
            'scene_index_sha256': digest_file(source / 'scene_index.json')}
        sources.extend(bindings[role].values())
    if (output / 'artifact_lock.json').exists():
        receipt = verify_qualified_official_exposure(output)
        if read_json(output / 'official_exposure_ledger.json').get('neutral_source_bindings') != bindings:
            raise ValueError('Existing official audit binds different neutral inputs')
        return receipt
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve partial audit evidence; choose a fresh official output root')
    records, roles = [], {}
    for role, index in indexes.items():
        target = output / 'indexes' / role / 'scene_index.json'
        atomic_json(target, index)
        index_hash = digest_file(target)
        sources.append(index_hash)
        roles[role] = {'scene_count': len(index), 'asset_group_count': len(set(index.values())),
            'membership_sha256': contracts[role].membership_sha256, 'scene_index_sha256': index_hash}
        records.extend({'role': role, 'asset_group': asset, 'scene_count': count,
            'history_status': 'unknown_prior_exploration', 'potentially_exposed': True,
            'prior_access': []} for asset, count in sorted(Counter(index.values()).items()))
    ledger = {'schema': OFFICIAL_SCHEMA, 'status': 'reviewed_with_uncertainty',
        'evaluation_policy': 'exposure_qualified_fixed_methods', 'no_retuning': True,
        'independent_confirmation': False, 'untouched_assets_claim': False, 'claim': CLAIM,
        'roles': roles, 'reviewed_source_sha256': sorted(set(sources)),
        'neutral_source_bindings': bindings,
        'unresolved_history': 'Neutral membership is verified. Complete prior asset-access records are unavailable; empty event lists mean unknown history, not no access.'}
    atomic_json(output / 'asset_records.json', records)
    atomic_json(output / 'official_exposure_ledger.json', ledger)
    seal(output, [output / 'asset_records.json', output / 'official_exposure_ledger.json',
        *(output / 'indexes').glob('*/scene_index.json')])
    return verify_qualified_official_exposure(output)
