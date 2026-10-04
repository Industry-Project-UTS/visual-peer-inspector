"""Two-mode upload orchestration using frozen decisions and session-owned tensors.

Model access is injected, so geometry, scoring, fusion and review can be tested
without GPU libraries or checkpoint downloads. No dataset manifests are read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from visual_peer_inspector.representation.crops import (
    CropConfiguration,
    prepare_masked_crop,
    project_crop_heatmap,
)
from visual_peer_inspector.representation.cross_view_identity import (
    CandidateObservation,
    IdentityConfiguration,
    ViewObservation,
    match_cross_view_identities,
)
from visual_peer_inspector.scoring.peer import deterministic_ranking, p1_scores

from .core import (
    MAX_VIEWS,
    MIN_VIEWS,
    Box,
    InspectionError,
    InspectionRequest,
    InspectionResult,
    Observation,
    RankedObject,
    validate_boxes,
    validate_result,
)
from .deployment import load_review_policy, pipeline_config, pipeline_fingerprint
from .heatmaps import HEATMAP_STYLE, heatmap_overlay


class ModelStages(Protocol):
    def propose(self, image: Image.Image, view_id: str) -> tuple[list, list[str]]: ...
    def segment(
        self, image: Image.Image, boxes: list
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def encode(self, crops: list): ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class ImageTransform:
    original_size: tuple[int, int]
    content_size: tuple[int, int]
    offset: tuple[int, int]

    @classmethod
    def prepare(cls, image: Image.Image):
        size = pipeline_config()["input_size"]
        scale = size / max(image.size)
        content = tuple(max(1, round(v * scale)) for v in image.size)
        offset = ((size - content[0]) // 2, (size - content[1]) // 2)
        canvas = Image.new("RGB", (size, size), (238, 241, 245))
        canvas.paste(image.resize(content, Image.Resampling.BICUBIC), offset)
        return canvas, cls(image.size, content, offset)

    def to_model_box(self, box):
        sx, sy = (self.content_size[i] / self.original_size[i] for i in (0, 1))
        left, top = self.offset
        return (
            math.floor(box[0] * sx) + left,
            math.floor(box[1] * sy) + top,
            math.ceil(box[2] * sx) + left,
            math.ceil(box[3] * sy) + top,
        )

    def to_original_box(self, box):
        left, top = self.offset
        sx, sy = (self.original_size[i] / self.content_size[i] for i in (0, 1))
        width, height = self.original_size
        x1 = max(0, min(width, math.floor((box[0] - left) * sx)))
        y1 = max(0, min(height, math.floor((box[1] - top) * sy)))
        x2 = max(0, min(width, math.ceil((box[2] - left) * sx)))
        y2 = max(0, min(height, math.ceil((box[3] - top) * sy)))
        return None if x1 >= x2 or y1 >= y2 else (x1, y1, x2, y2)

    def restore(self, image: Image.Image, *, nearest=False):
        x, y = self.offset
        w, h = self.content_size
        return image.crop((x, y, x + w, y + h)).resize(
            self.original_size,
            Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
        )

    def trim_mask(self, mask):
        output = np.zeros_like(mask, dtype=bool)
        x, y = self.offset
        w, h = self.content_size
        output[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
        return output


@dataclass
class Candidate:
    box: Box
    model_box: tuple
    mask: np.ndarray | None = None
    quality: float = 0.0
    crop: object | None = None
    embedding: tuple | None = None
    patches: np.ndarray | None = None
    validity: np.ndarray | None = None
    validity50: np.ndarray | None = None
    cache_key: str = ''


def cache_binding():
    from pathlib import Path
    from .session_cache import digest, file_hash
    base = Path(__file__).resolve().parents[2]
    paths = ['visual_peer_inspector/web/' + name for name in ('pipeline.py', 'model_runtime.py', 'session_cache.py','encoder_worker.py')]
    paths += ['visual_peer_inspector/representation/crops.py', 'visual_peer_inspector/models/visual_encoder.py']
    import importlib.metadata
    runtime={name:importlib.metadata.version(name) for name in ('numpy','scipy','pillow')}
    return digest({'configuration': pipeline_config(), 'runtime':runtime,'sources': {p: file_hash(base/p) for p in paths}})


class UploadPipeline:
    def __init__(self, models: ModelStages, *, review_policy_path=None, cache_root=None):
        import os, tempfile
        from .session_cache import SessionCache
        self.models = models
        self.config = pipeline_config()
        self.policy, self.policy_version = load_review_policy(review_policy_path)
        self._temporary = None
        root = cache_root or os.environ.get('VPI_SESSION_CACHE_ROOT')
        if root is None:
            self._temporary = tempfile.TemporaryDirectory(prefix='vpi-candidates-')
            root = self._temporary.name
        self.cache = SessionCache(root)

    def clear_session(self, session_id, cache_token=None):
        self.cache.clear(session_id, cache_token)

    def inspect(self, request: InspectionRequest, progress) -> InspectionResult:
        import os
        from .session_cache import digest, pixels
        from visual_peer_inspector.scoring.forward_patch import patch_policy
        if not MIN_VIEWS <= len(request.views) <= MAX_VIEWS:
            raise InspectionError('Upload between two and five views.')
        if request.decision_mode not in ('automatic', 'assisted'):
            raise InspectionError('Choose automatic or assisted inspection.')
        if len({v.view_id for v in request.views}) != len(request.views):
            raise InspectionError('View IDs must be unique.')
        if request.corrections is not None:
            validate_boxes(request.corrections, request.views)
            if any(sum(b.view_id == v.view_id for b in request.corrections) > 8 for v in request.views):
                raise InspectionError('Keep at most eight objects in each view.')
        binding = cache_binding()
        path, state, token = self.cache.open(request.session_id, request.cache_token, binding,
            allow_create=os.environ.get('VPI_CREATE_SESSION_CACHE') == '1')
        by_view = {v.view_id: [] for v in request.views}
        prepared = {v.view_id: ImageTransform.prepare(v.image) for v in request.views}
        reasons, reused, computed = [], 0, 0
        try:
            progress('Finding objects')
            for view in sorted(request.views, key=lambda v: v.view_id):
                canvas, transform = prepared[view.view_id]
                image_key = pixels(view.image)
                if request.corrections is not None:
                    boxes = [b for b in request.corrections if b.view_id == view.view_id]
                elif image_key in state['views']:
                    boxes = [Box(view.view_id, b['object_id'], '', tuple(b['xyxy'])) for b in state['views'][image_key]]
                else:
                    proposals, notes = self.models.propose(canvas, view.view_id)
                    reasons.extend(notes)
                    boxes = []
                    for index, coords in enumerate(proposals):
                        original = transform.to_original_box(coords)
                        if original:
                            boxes.append(Box(view.view_id, f'object_{index+1}', '', original))
                    state['views'][image_key] = [{'object_id': b.object_id, 'xyxy': b.xyxy} for b in boxes]
                    self.cache.save(path, state)
                for box in boxes:
                    c = Candidate(box, transform.to_model_box(box.xyxy))
                    c.cache_key = digest({'image': image_key, 'box': box.xyxy, 'model_box': c.model_box, 'binding': binding})
                    saved = self.cache.load(path, state, c.cache_key)
                    if saved is not None:
                        c.mask = saved['mask'].astype(bool)
                        if c.mask.shape != (view.image.height, view.image.width):
                            raise InspectionError('Cached mask dimensions differ. Reset this session.')
                        c.quality = saved['quality']
                        c.embedding = tuple(saved['embedding']) if len(saved['embedding']) else None
                        c.patches = saved['patches']
                        reused += 1
                    by_view[view.view_id].append(c)
            self.models.release()
            progress('Creating masks')
            for view in request.views:
                missing = [c for c in by_view[view.view_id] if c.mask is None]
                if missing:
                    canvas, transform = prepared[view.view_id]
                    masks, qualities = self.models.segment(canvas, [c.model_box for c in missing])
                    if np.asarray(masks).shape != (len(missing), 256, 256) or np.asarray(qualities).shape != (len(missing),) or not np.isfinite(qualities).all():
                        raise ValueError('Invalid SAM mask or quality schema.')
                    for c, mask, quality in zip(missing, masks, qualities, strict=True):
                        model_mask = transform.trim_mask(mask)
                        c.mask = np.asarray(transform.restore(Image.fromarray(model_mask.astype(np.uint8)*255), nearest=True)) > 0
                        c.quality = float(quality)
                for c in by_view[view.view_id]:
                    if c.mask.sum() >= self.config['crop']['minimum_foreground_pixels']:
                        c.crop = prepare_masked_crop(np.asarray(view.image), c.mask, configuration=CropConfiguration(**self.config['crop']))
                        c.validity = patch_policy(c.crop, 16, .70, fallback='all')[0]
                        c.validity50 = patch_policy(c.crop, 16, .50, fallback='largest')[0]
                    else:
                        reasons.append('An object mask is empty or too small; it remains available for correction.')
            self.models.release()
            progress('Extracting features')
            missing = [c for cs in by_view.values() for c in cs if c.crop is not None and c.patches is None]
            for start in range(0, len(missing), 4):
                batch = missing[start:start+4]
                encoded = self.models.encode([c.crop.rgb for c in batch])
                if len(encoded.patch_features) != len(batch) or encoded.global_features.shape != (len(batch), encoded.feature_dimension) or encoded.patch_grid_size != 16:
                    raise ValueError('Encoder output does not match the fixed patch contract.')
                for c, vector, patches in zip(batch, encoded.global_features, encoded.patch_features, strict=True):
                    if np.asarray(patches).shape != (256, encoded.feature_dimension) or not np.isfinite(patches).all() or not np.isfinite(vector).all():
                        raise ValueError('Invalid encoder tensor.')
                    c.embedding, c.patches = tuple(float(x) for x in vector), np.asarray(patches, dtype=np.float32)
                    computed += 1
            self.models.release()
            dimensions = next((c.patches.shape[1] for cs in by_view.values() for c in cs if c.patches is not None and len(c.patches)),384)
            for cs in by_view.values():
                for c in cs:
                    if c.patches is None:
                        c.patches = np.empty((0, dimensions), dtype=np.float32)
                    self.cache.put(path, state, c.cache_key, {'quality': c.quality},
                        {'mask': c.mask, 'patches': c.patches, 'embedding': np.asarray(c.embedding or (), dtype=np.float32)})
            progress('Comparing peers')
            result = self._decide(request, by_view, reasons, token, reused, computed)
            progress('Combining views')
            return result
        finally:
            self.models.release()

    def _decide(self, request, by_view, reasons, token, reused, computed):
        from .decision import component_scores, decide_cached
        from visual_peer_inspector.scoring.peer_support import mask_shape_descriptor
        candidates = [c for v in sorted(by_view) for c in sorted(by_view[v], key=lambda c:c.box.object_id)]
        records = [{'view_id': c.box.view_id, 'candidate_id': c.box.object_id, 'embedding': c.embedding,
                    'patches': c.patches, 'valid50': c.validity50, 'valid70': c.validity,
                    'geometry': mask_shape_descriptor(c.crop.foreground) if c.crop else None} for c in candidates]
        rows, heatmaps = component_scores(records)
        views = sorted(by_view)
        for view in views:
            rows.setdefault(view, {'objects': []})
        bank = [{'view_id': c.box.view_id, 'candidate_id': c.box.object_id,
                 'object_id': (c.box.identity_id or None) if request.decision_mode == 'assisted' else None,
                 'patches': c.patches[c.validity.ravel()] if c.validity is not None else np.empty((0, c.patches.shape[1]))} for c in candidates]
        decision = decide_cached(rows, bank, views=views, mode=request.decision_mode)
        raw_rankings = decision['ranking']
        ranking = []
        for obj in raw_rankings:
            if obj['score'] is None:
                continue
            members = obj['members']
            contributors = [{**m, 'normalized_rank': decision['component_rank'].get((m['view_id'],m['candidate_id']))} for m in members]
            usable = [m for m in contributors if m['normalized_rank'] is not None]
            used = sorted(usable, key=lambda m:(-m['normalized_rank'],m['view_id'],m['candidate_id']))[:2] if request.decision_mode == 'assisted' else usable
            used_keys = {(m['view_id'],m['candidate_id']) for m in used}
            for member in contributors:
                member['used_in_fusion'] = (member['view_id'],member['candidate_id']) in used_keys
            ranking.append(RankedObject(obj['id'], obj['score'], tuple(sorted(m['view_id'] for m in usable)), tuple(contributors),
                'object_bank_top2' if request.decision_mode == 'assisted' else 'winner_cluster_consensus'))
        ranking.sort(key=lambda r:(-r.relative_score,r.identity_id))
        tied = bool(len(ranking)>1 and np.isclose(ranking[0].relative_score,ranking[1].relative_score,atol=1e-12,rtol=0))
        selected = ranking[0].identity_id if ranking and not tied else None
        if tied:
            reasons.append('The leading combined scores are tied; no unique object is selected.')
        if not ranking:
            reasons.append('At least three usable peers and sufficient identity evidence are required.')
        if request.decision_mode == 'assisted' and request.corrections is None:
            reasons.append('Confirm neutral identities across all views before assisted scoring.')
        if selected and len(ranking[0].view_ids) < len(views):
            reasons.append('The selected object is not linked in every uploaded view; absence is not established.')
        if request.decision_mode == 'assisted' and any(not c.box.identity_id for c in candidates):
            reasons.append('Unknown identities remain visible and unscored; they are excluded from trusted peer banks.')
        reasons.append('Scores compare relative unusualness; they are not correctness probabilities.')
        lookup = {(v,o['candidate_id']):o['scores'] for v,row in rows.items() for o in row['objects']}
        scored = {(o['view_id'],o['candidate_id']):o for o in decision['scored_observations'] or []}
        output = []
        for index,c in enumerate(candidates):
            key = (c.box.view_id,c.box.object_id)
            identity = decision['identities'].get(key, f'unlinked_{index+1}')
            components = dict(lookup[key])
            score = decision['component_rank'].get(key)
            patchmap = heatmaps.get(key)
            status = 'supported_winner' if key in decision['identities'] else 'unlinked'
            source = 'predicted'
            if request.decision_mode == 'assisted':
                item = scored[key]
                score = item['object_bank']['score']
                components['object_bank'] = score
                status = 'confirmed' if c.box.identity_id else 'unknown'
                source = 'user_confirmed' if c.box.identity_id else 'unknown'
                if item['object_bank']['patch_scores']:
                    patchmap = np.full((16,16),np.nan)
                    patchmap[c.validity] = item['object_bank']['patch_scores']
                else:
                    patchmap = None
            heatmap = None
            if patchmap is not None and c.crop is not None:
                projected = project_crop_heatmap(patchmap,c.crop,c.mask)
                image = next(v.image for v in request.views if v.view_id == c.box.view_id)
                heatmap = heatmap_overlay(image, projected, c.mask)
            output.append(Observation(Box(c.box.view_id,c.box.object_id,identity,c.box.xyxy),components.get('P1_current'),
                Image.fromarray(c.mask.astype(np.uint8)*255),heatmap,score,
                'object_bank' if request.decision_mode=='assisted' else 'winner_consensus_components', components,
                decision['component_rank'].get(key),status,source))
        evidence = tuple(output)
        signals = self.policy.signals(tuple(ranking), request.decision_mode, len(request.views))
        state, policy_reasons = self.policy.decide(signals,
            correspondence_confirmed=request.decision_mode == 'assisted' and request.corrections is not None)
        result = InspectionResult(tuple(ranking), evidence, state,
            tuple(dict.fromkeys([*policy_reasons, *reasons])) if state != 'result_ready' else (), self.config['method_version']+':'+pipeline_fingerprint(),self.policy_version,request.decision_mode,
            {'pipeline':self.config,'pipeline_sha256':pipeline_fingerprint(),'prediction_only':True,
             'review_signals':signals, 'review_policy':{'source':'src/visual_peer_inspector/web/frozen_review_policy.json',
                'sha256':self.policy.source_sha256,'schema':'vpi-current-two-mode-review-policy/v1',
                'version':self.policy.version},
             'correspondence_source':'user_confirmed' if request.decision_mode=='assisted' and request.corrections is not None else 'predicted',
             'cache':{'reused_candidates':reused,'encoded_candidates':computed},
             'identity_tracks':decision['tracks'],'score_kind':'object_bank_top2' if request.decision_mode=='assisted' else 'winner_cluster_consensus'},
            selected,request.session_id,token,
            {'method_label':'Assisted patch bank' if request.decision_mode=='assisted' else 'Automatic winner consensus',
             'selection_status':'tied' if tied else ('unique' if selected else 'insufficient'),
             'missing_links':[v for v in views if selected and v not in ranking[0].view_ids],
             'suggested_identities':decision['suggested_identities'] if request.decision_mode=='automatic' or request.corrections is None else
                [{'view_id':c.box.view_id,'candidate_id':c.box.object_id,'identity_id':c.box.identity_id,'status':'confirmed' if c.box.identity_id else 'unknown'} for c in candidates],
             'all_object_rankings':raw_rankings,'heatmap_scope':'coarse foreground patch evidence; independently scaled for display',
             'heatmap_style':HEATMAP_STYLE})
        validate_result(result,request)
        return result
