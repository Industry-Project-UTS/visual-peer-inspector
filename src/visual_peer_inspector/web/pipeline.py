"""Real upload orchestration using the existing crop, P1 and identity algorithms.

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
from visual_peer_inspector.review_policy import (
    apply_review_policy,
    build_prediction_review_signals,
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
    score: float | None = None
    normalized_rank: float | None = None
    heatmap: np.ndarray | None = None


class UploadPipeline:
    def __init__(self, models: ModelStages, *, review_policy_path=None):
        self.models = models
        self.config = pipeline_config()
        self.policy, self.policy_version = load_review_policy(review_policy_path)

    def inspect(self, request: InspectionRequest, progress) -> InspectionResult:
        if not MIN_VIEWS <= len(request.views) <= MAX_VIEWS:
            raise InspectionError("Upload between two and five views.")
        if len({v.view_id for v in request.views}) != len(request.views):
            raise InspectionError("View IDs must be unique.")
        if request.corrections is not None:
            validate_boxes(request.corrections, request.views)
            if any(
                sum(b.view_id == v.view_id for b in request.corrections) > 8
                for v in request.views
            ):
                raise InspectionError("Keep at most eight objects in each view.")
        reasons = []
        prepared = {v.view_id: ImageTransform.prepare(v.image) for v in request.views}
        by_view = {v.view_id: [] for v in request.views}
        try:
            progress("Finding objects")
            for view in sorted(request.views, key=lambda v: v.view_id):
                canvas, transform = prepared[view.view_id]
                if request.corrections is None:
                    boxes, notes = self.models.propose(canvas, view.view_id)
                    reasons.extend(notes)
                    for index, coords in enumerate(boxes):
                        original = transform.to_original_box(coords)
                        if original is not None:
                            box = Box(
                                view.view_id,
                                f"object_{index + 1}",
                                f"pending_{index + 1}",
                                original,
                            )
                            by_view[view.view_id].append(Candidate(box, tuple(coords)))
                else:
                    for box in request.corrections:
                        if box.view_id == view.view_id:
                            by_view[view.view_id].append(
                                Candidate(box, transform.to_model_box(box.xyxy))
                            )
            self.models.release()
            progress("Creating masks")
            for view in request.views:
                candidates = by_view[view.view_id]
                if not candidates:
                    reasons.append("No objects were found in one or more views.")
                    continue
                canvas, transform = prepared[view.view_id]
                masks, qualities = self.models.segment(
                    canvas, [c.model_box for c in candidates]
                )
                if np.asarray(masks).shape != (len(candidates), 256, 256):
                    raise ValueError("SAM mask dimensions do not match the request.")
                if (
                    np.asarray(qualities).shape != (len(candidates),)
                    or not np.isfinite(qualities).all()
                ):
                    raise ValueError("SAM returned invalid quality scores.")
                for candidate, mask, quality in zip(
                    candidates, masks, qualities, strict=True
                ):
                    candidate.mask = transform.trim_mask(mask)
                    candidate.quality = float(quality)
                    if (
                        candidate.mask.sum()
                        < self.config["crop"]["minimum_foreground_pixels"]
                    ):
                        reasons.append(
                            "An object mask is empty or too small. Review its box."
                        )
                        continue
                    candidate.crop = prepare_masked_crop(
                        canvas,
                        candidate.mask,
                        configuration=CropConfiguration(**self.config["crop"]),
                    )
            self.models.release()
            progress("Extracting features")
            usable = [
                c for values in by_view.values() for c in values if c.crop is not None
            ]
            size = self.config["encoder"]["batch_size"]
            for start in range(0, len(usable), size):
                batch = usable[start : start + size]
                encoded = self.models.encode([c.crop.rgb for c in batch])
                if len(encoded.global_features) != len(batch) or len(
                    encoded.patch_features
                ) != len(batch):
                    raise ValueError(
                        "Encoder batch dimensions differ from object count."
                    )
                for candidate, vector, patches in zip(
                    batch, encoded.global_features, encoded.patch_features, strict=True
                ):
                    if not np.isfinite(vector).all() or not np.isfinite(patches).all():
                        raise ValueError("Encoder returned non-finite features.")
                    candidate.embedding = tuple(float(v) for v in vector)
                    candidate.patches = patches
                    candidate.validity = candidate.crop.patch_validity(
                        encoded.patch_grid_size,
                        minimum_fraction=self.config["patch_validity_minimum_fraction"],
                    )
            self.models.release()
            progress("Comparing peers")
            for candidates in by_view.values():
                valid = [
                    c for c in candidates if c.validity is not None and c.validity.any()
                ]
                if len(valid) < 3:
                    reasons.append(
                        "P1 needs at least three usable peer objects in a view."
                    )
                    continue
                result = p1_scores(
                    [c.patches for c in valid],
                    [c.validity for c in valid],
                    top_fraction=self.config["p1_top_fraction"],
                )
                order = deterministic_ranking(
                    result.scores, [c.box.object_id for c in valid]
                )
                if len({round(float(s), 12) for s in result.scores}) < len(
                    result.scores
                ):
                    reasons.append(
                        "Some peer scores are tied; their order is uncertain."
                    )
                for rank, index in enumerate(order, 1):
                    valid[index].score = float(result.scores[index])
                    valid[index].normalized_rank = (len(valid) - rank) / (
                        len(valid) - 1
                    )
                    valid[index].heatmap = project_crop_heatmap(
                        result.heatmaps[index], valid[index].crop, valid[index].mask
                    )
            progress("Combining views")
            return self._combine(request, by_view, prepared, reasons)
        finally:
            self.models.release()

    def _combine(self, request, by_view, prepared, reasons):
        local = {
            (c.box.view_id, c.box.object_id): c
            for values in by_view.values()
            for c in values
        }
        identities = {}
        tracks = []
        if request.corrections is None:
            views = [
                ViewObservation(
                    view_id,
                    tuple(
                        CandidateObservation(c.box.object_id, c.embedding)
                        for c in candidates
                        if c.embedding is not None
                    ),
                )
                for view_id, candidates in by_view.items()
            ]
            matched = match_cross_view_identities(
                views, configuration=IdentityConfiguration(**self.config["identity"])
            )
            for track in matched.tracks:
                members = [(m.view_id, m.candidate_id) for m in track.members]
                identities.update({key: track.identity_id for key in members})
                tracks.append(
                    {
                        "identity_id": track.identity_id,
                        "identity_status": track.status,
                        "cycle_consistent": track.cycle_consistent,
                        "members": members,
                    }
                )
        else:
            for key, candidate in local.items():
                identities[key] = candidate.box.identity_id
            for identity in sorted(set(identities.values())):
                members = [
                    key for key, value in identities.items() if value == identity
                ]
                tracks.append(
                    {
                        "identity_id": identity,
                        "identity_status": "matched"
                        if len(members) >= 2
                        else "unmatched",
                        "cycle_consistent": True,
                        "members": members,
                    }
                )
        for index, key in enumerate(sorted(local)):
            if key not in identities:
                identity = f"unmatched_{index + 1}"
                identities[key] = identity
                tracks.append(
                    {
                        "identity_id": identity,
                        "identity_status": "unmatched",
                        "cycle_consistent": False,
                        "members": [key],
                    }
                )
        ranked = []
        votes = {}
        for track in tracks:
            scored = [
                local[key] for key in track["members"] if local[key].score is not None
            ]
            if scored:
                ranked.append(
                    RankedObject(
                        track["identity_id"],
                        float(np.mean([c.normalized_rank for c in scored])),
                        tuple(sorted(c.box.view_id for c in scored)),
                    )
                )
                votes[track["identity_id"]] = sum(
                    c.normalized_rank == 1 for c in scored
                ) / len(scored)
        ranked.sort(key=lambda r: (-r.relative_score, r.identity_id))
        winner = ranked[0] if ranked else None
        decision = {
            "tracks": tracks,
            "fused_odd_one_out": winner.identity_id if winner else None,
            "margin": ranked[0].relative_score - ranked[1].relative_score
            if len(ranked) > 1
            else None,
            "agreement": votes.get(winner.identity_id) if winner else None,
            "insufficient_evidence": winner is None or len(winner.view_ids) < 2,
        }
        rows = [
            {
                "objects": [
                    {
                        "box_xyxy": c.model_box,
                        "foreground_pixels": int(c.mask.sum()),
                        "predicted_quality_score": c.quality,
                    }
                    for c in candidates
                ]
            }
            for candidates in by_view.values()
        ]
        signals = build_prediction_review_signals(decision, rows)
        decision.update(signals)
        if signals["insufficient_evidence"]:
            state = "insufficient_evidence"
            reasons.append(
                "There is not enough evidence to compare the same object across at least two views."
            )
        elif self.policy is None:
            state = "review_recommended"
            reasons.append(
                "The review policy is provisional. Check the highlighted objects and matches."
            )
        else:
            state, policy_reasons = apply_review_policy(decision, self.policy)
            state = "review_recommended" if state == "review_required" else state
            reasons.extend(
                reason.replace("_", " ").capitalize() + "." for reason in policy_reasons
            )
        if signals["identity_unmatched_or_ambiguous"]:
            reasons.append("Some cross-view matches are uncertain or unmatched.")
        if any(c.score is None for c in local.values()):
            reasons.append(
                "Some objects have no usable peer score; they remain available for review."
            )
        if winner and len(winner.view_ids) < len(request.views):
            reasons.append(
                "The top object has scoring evidence in only some uploaded views."
            )
        if decision["margin"] is not None and decision["margin"] <= 1e-12:
            reasons.append("The top combined scores are tied.")
        if reasons and state == "result_ready":
            state = "review_recommended"
        observations = []
        finite = [
            c.heatmap[np.isfinite(c.heatmap)]
            for c in local.values()
            if c.heatmap is not None
        ]
        limit = max(
            (float(values.max()) for values in finite if values.size), default=1.0
        )
        for key, candidate in local.items():
            view = next(v for v in request.views if v.view_id == key[0])
            _, transform = prepared[key[0]]
            mask = transform.restore(
                Image.fromarray(candidate.mask.astype(np.uint8) * 255), nearest=True
            )
            heatmap = None
            if candidate.heatmap is not None:
                strength = np.clip(
                    np.nan_to_num(candidate.heatmap, nan=0.0) / max(limit, 1e-12), 0, 1
                )
                alpha = transform.restore(
                    Image.fromarray((strength * 255).astype(np.uint8))
                )
                heatmap = Image.composite(
                    Image.new("RGB", view.image.size, "#ec7c36"), view.image, alpha
                )
            box = Box(key[0], key[1], identities[key], candidate.box.xyxy)
            observations.append(Observation(box, candidate.score, mask, heatmap))
        result = InspectionResult(
            tuple(ranked),
            tuple(observations),
            state,
            tuple(dict.fromkeys(reasons)),
            f"{self.config['method_version']}:{pipeline_fingerprint()}",
            self.policy_version,
            "assisted" if request.corrections is not None else "automatic",
            provenance={
                "pipeline": self.config,
                "pipeline_sha256": pipeline_fingerprint(),
                "score_kind": "mean_normalized_rank",
                "prediction_only": True,
                "review_signals": signals,
                "identity_tracks": [
                    {
                        **track,
                        "members": [
                            {"view_id": key[0], "object_id": key[1]}
                            for key in track["members"]
                        ],
                    }
                    for track in tracks
                ],
            },
        )
        validate_result(result, request)
        return result
