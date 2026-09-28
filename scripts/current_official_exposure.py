#!/usr/bin/env python3
"""Prepare the frozen official exposure policy from sealed neutral geometry.

The geometry producer records two explicit label-access flags. The original
policy gate asks for a different flag name, so this transport adapter checks
the producer's sealed fields and keeps the policy and output schema unchanged.
"""
from collections import Counter
from pathlib import Path

from visual_peer_inspector.data.model_safe_package import membership_sha256
from vpi_object_bank.storage import atomic_json, digest_file, read_json, seal, verify_seal
from vpi_next_evaluation.exposure import (
    CLAIM, OFFICIAL_SCHEMA, _official_contracts, verify_qualified_official_exposure,
)


def prepare_official_exposure_from_geometry(cfg):
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
            raise ValueError('Official neutral index lacks sealed geometry')
        manifest = read_json(source / 'geometry_manifest.json')
        expected = {
            'schema': 'vpi-current-reference-geometry/v1', 'role': role,
            'scene_count': contract.scene_count,
            'asset_group_count': contract.asset_group_count,
            'membership_sha256': contract.membership_sha256,
            'asset_index_sha256': digest_file(source / 'scene_index.json'),
            'anomaly_outcomes_used': False,
            'anomaly_labels_exported': False,
            'trusted_preparer_only': True,
            'reference_box_source': 'per_object_segmentation_masks',
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError('Official neutral geometry differs from fixed membership')
        index = read_json(source / 'scene_index.json')
        if (not isinstance(index, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                       for k, v in index.items())
                or len(index) != contract.scene_count
                or len(set(index.values())) != contract.asset_group_count
                or membership_sha256(index) != contract.model_membership_sha256):
            raise ValueError('Official neutral scene or asset membership changed')
        indexes[role] = index
        bindings[role] = {
            'geometry_lock_sha256': digest_file(source / 'artifact_lock.json'),
            'geometry_manifest_sha256': digest_file(source / 'geometry_manifest.json'),
            'scene_index_sha256': digest_file(source / 'scene_index.json'),
        }
        sources.extend(bindings[role].values())
    if (output / 'artifact_lock.json').exists():
        receipt = verify_qualified_official_exposure(output)
        if read_json(output / 'official_exposure_ledger.json').get('neutral_source_bindings') != bindings:
            raise ValueError('Existing official audit binds different neutral inputs')
        return receipt
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve partial audit evidence; choose a fresh output root')
    records, roles = [], {}
    for role, index in indexes.items():
        target = output / 'indexes' / role / 'scene_index.json'
        atomic_json(target, index)
        index_hash = digest_file(target)
        sources.append(index_hash)
        roles[role] = {
            'scene_count': len(index),
            'asset_group_count': len(set(index.values())),
            'membership_sha256': contracts[role].membership_sha256,
            'scene_index_sha256': index_hash,
        }
        records.extend({
            'role': role, 'asset_group': asset, 'scene_count': count,
            'history_status': 'unknown_prior_exploration',
            'potentially_exposed': True, 'prior_access': [],
        } for asset, count in sorted(Counter(index.values()).items()))
    ledger = {
        'schema': OFFICIAL_SCHEMA, 'status': 'reviewed_with_uncertainty',
        'evaluation_policy': 'exposure_qualified_fixed_methods',
        'no_retuning': True, 'independent_confirmation': False,
        'untouched_assets_claim': False, 'claim': CLAIM,
        'roles': roles, 'reviewed_source_sha256': sorted(set(sources)),
        'neutral_source_bindings': bindings,
        'unresolved_history': (
            'Neutral membership is verified. Complete prior asset-access records are '
            'unavailable; empty event lists mean unknown history, not no access.'
        ),
    }
    atomic_json(output / 'asset_records.json', records)
    atomic_json(output / 'official_exposure_ledger.json', ledger)
    seal(output, [output / 'asset_records.json', output / 'official_exposure_ledger.json',
                  *(output / 'indexes').glob('*/scene_index.json')])
    return verify_qualified_official_exposure(output)
