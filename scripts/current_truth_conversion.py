#!/usr/bin/env python3
"""Run truth conversion with strict support for typed cache rejections.

The frozen cache retains a candidate whose SAM mask is too small to encode, but
does not fabricate DINO features or scores for it.  The original truth bridge
predated that explicit rejection state and required evidence and score IDs to
be identical.  This adapter filters only fully validated tiny-mask rejections
from the temporary geometry view used for neutral matching.  It never changes
the sealed cache, predictions, method freeze, anomaly labels, or denominators.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

from visual_peer_inspector.models.sam2 import decode_binary_mask
from vpi_next_evaluation.native_cache import CROP_CONFIGURATION, SMALL_MASK_REASON


def _typed_rejection(candidate_id, geometry):
    expected = {
        'mask', 'box', 'feature_status', 'feature_failure', 'foreground_pixels',
    }
    if not isinstance(candidate_id, str) or not candidate_id or set(geometry) != expected:
        raise ValueError('Unscored cache candidate is not a complete typed rejection')
    box = geometry['box']
    if (not isinstance(box, list) or len(box) != 4
            or any(type(value) not in (int, float) or not isfinite(value) for value in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError('Typed rejection has invalid candidate geometry')
    foreground = int(decode_binary_mask(geometry['mask']).sum())
    if (geometry['feature_status'] != 'rejected'
            or geometry['feature_failure'] != SMALL_MASK_REASON
            or type(geometry['foreground_pixels']) is not int
            or geometry['foreground_pixels'] != foreground
            or foreground >= CROP_CONFIGURATION.minimum_foreground_pixels):
        raise ValueError('Typed rejection differs from the frozen crop contract')


def _scored_geometry(payload):
    """Return a copy whose matching geometry contains only scored candidates."""
    if not isinstance(payload, dict):
        raise ValueError('Cache scene payload is not an object')
    scores = payload.get('score_rows')
    evidence = payload.get('evidence_views')
    if not isinstance(scores, dict) or not isinstance(evidence, dict) or set(scores) != set(evidence):
        raise ValueError('Cache score and evidence view inventories differ')
    filtered = deepcopy(payload)
    rejected_count = 0
    for view in sorted(scores):
        objects = scores[view].get('objects')
        candidates = evidence[view].get('candidates')
        if not isinstance(objects, list) or not isinstance(candidates, dict):
            raise ValueError('Cache score rows or evidence candidates are invalid')
        scored = [item.get('candidate_id') for item in objects]
        if (any(not isinstance(candidate, str) or not candidate for candidate in scored)
                or len(scored) != len(set(scored))
                or not set(scored) <= set(candidates)):
            raise ValueError('Scored cache candidates differ from retained geometry')
        extras = set(candidates) - set(scored)
        for candidate in sorted(extras):
            _typed_rejection(candidate, candidates[candidate])
        rejected_count += len(extras)
        filtered['evidence_views'][view]['candidates'] = {
            candidate: candidates[candidate] for candidate in scored
        }
    if payload.get('invalid_candidate_observations') != rejected_count:
        raise ValueError('Cache invalid-candidate count differs from typed rejections')
    return filtered


def convert_raw_evaluator_with_typed_rejections(cfg):
    """Convert labels after predictions seal, filtering only typed rejections."""
    from vpi_next_evaluation import native_evaluator, raw_reference

    if raw_reference._emit_evaluator is not native_evaluator._emit_evaluator:
        raise RuntimeError('Raw evaluator bridge was already replaced')
    cache_scenes = (Path(cfg['cache_root']).resolve(strict=True) / 'scenes').resolve()
    original_read = native_evaluator.read_json

    def read_for_matching(path):
        value = original_read(path)
        resolved = Path(path).resolve()
        if (resolved.parent == cache_scenes and resolved.suffix == '.json'):
            return _scored_geometry(value)
        return value

    native_evaluator.read_json = read_for_matching
    try:
        return raw_reference.convert_raw_evaluator(cfg)
    finally:
        native_evaluator.read_json = original_read

