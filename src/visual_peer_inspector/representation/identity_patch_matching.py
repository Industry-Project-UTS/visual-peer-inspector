"""Build cross-view identity descriptors from fixed DINO object features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from visual_peer_inspector.models.visual_encoder import EncodedBatch
from visual_peer_inspector.representation.crops import PreparedCrop
from visual_peer_inspector.representation.cross_view_identity import (
    CandidateObservation,
    IdentityConfiguration,
    ViewObservation,
    match_cross_view_identities,
)


PATCH_COVERAGES = (0.05, 0.25, 0.50)
UNMATCHED_COSTS = (0.20, 0.35, 0.50)


@dataclass(frozen=True)
class DescriptorVariant:
    """One prediction-only identity descriptor configuration."""

    name: str
    local_weight: float
    patch_coverage: float | None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.local_weight) <= 1.0:
            raise ValueError("local_weight must be in [0, 1]")
        if self.patch_coverage is None:
            if self.local_weight != 0.0:
                raise ValueError("A local descriptor requires patch coverage")
        elif not 0.0 < float(self.patch_coverage) <= 1.0:
            raise ValueError("patch_coverage must be in (0, 1]")


DESCRIPTOR_VARIANTS = (
    DescriptorVariant("global_current", 0.0, None),
    DescriptorVariant("local_cov05", 1.0, 0.05),
    DescriptorVariant("local_cov25", 1.0, 0.25),
    DescriptorVariant("local_cov50", 1.0, 0.50),
    DescriptorVariant("hybrid25_cov25", 0.25, 0.25),
    DescriptorVariant("hybrid50_cov25", 0.50, 0.25),
    DescriptorVariant("hybrid75_cov25", 0.75, 0.25),
)


def unit_vector(values: Sequence[float]) -> np.ndarray:
    """Return one finite unit vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("A descriptor must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("A descriptor must have non-zero norm")
    return (array / norm).astype(np.float32)


def foreground_patch_descriptor(
    patch_features: np.ndarray,
    crop: PreparedCrop,
    *,
    minimum_coverage: float,
) -> tuple[np.ndarray, bool, int]:
    """Pool foreground patch tokens and report whether a fallback was needed."""

    patches = np.asarray(patch_features, dtype=np.float32)
    if patches.ndim != 2 or not patches.size or not np.isfinite(patches).all():
        raise ValueError("Patch features must be a finite two-dimensional matrix")
    grid = int(round(np.sqrt(patches.shape[0])))
    if grid * grid != patches.shape[0]:
        raise ValueError("Patch features must form a square spatial grid")
    selected = crop.patch_validity(grid, minimum_fraction=minimum_coverage).reshape(-1)
    fallback = False
    if not selected.any():
        coverage = crop.foreground.reshape(
            grid,
            crop.foreground.shape[0] // grid,
            grid,
            crop.foreground.shape[1] // grid,
        ).mean(axis=(1, 3)).reshape(-1)
        selected[int(np.argmax(coverage))] = True
        fallback = True
    descriptor = unit_vector(patches[selected].mean(axis=0))
    return descriptor, fallback, int(selected.sum())


def descriptors_from_batch(
    batch: EncodedBatch,
    crops: Sequence[PreparedCrop],
) -> tuple[list[dict[str, np.ndarray]], dict[str, int]]:
    """Prepare global, local and hybrid descriptors for one object batch."""

    if len(crops) != len(batch.global_features) or len(crops) != len(batch.patch_features):
        raise ValueError("The encoder batch does not match the prepared crops")
    if batch.patch_grid_size is None:
        raise ValueError("Identity refinement requires spatial patch features")
    output: list[dict[str, np.ndarray]] = []
    fallback_counts = {f"coverage_{coverage:.2f}": 0 for coverage in PATCH_COVERAGES}
    for index, crop in enumerate(crops):
        global_descriptor = unit_vector(batch.global_features[index])
        local = {}
        for coverage in PATCH_COVERAGES:
            descriptor, fallback, _selected_count = foreground_patch_descriptor(
                batch.patch_features[index],
                crop,
                minimum_coverage=coverage,
            )
            local[coverage] = descriptor
            fallback_counts[f"coverage_{coverage:.2f}"] += int(fallback)
        variants: dict[str, np.ndarray] = {"global_current": global_descriptor}
        for variant in DESCRIPTOR_VARIANTS[1:]:
            assert variant.patch_coverage is not None
            local_descriptor = local[variant.patch_coverage]
            if variant.local_weight == 1.0:
                variants[variant.name] = local_descriptor
                continue
            combined = np.concatenate(
                [
                    np.sqrt(1.0 - variant.local_weight) * global_descriptor,
                    np.sqrt(variant.local_weight) * local_descriptor,
                ]
            )
            variants[variant.name] = unit_vector(combined)
        output.append(variants)
    return output, fallback_counts


def match_descriptor_variant(
    views: Sequence[dict[str, Any]],
    *,
    descriptor_name: str,
    unmatched_cost: float,
) -> dict[str, Any]:
    """Match one descriptor variant across two to five views."""

    observations = []
    for row in views:
        candidates = tuple(
            CandidateObservation(
                candidate_id=str(item["candidate_id"]),
                embedding=tuple(float(value) for value in item["descriptors"][descriptor_name]),
            )
            for item in row["objects"]
        )
        observations.append(ViewObservation(view_id=str(row["view"]), candidates=candidates))
    result = match_cross_view_identities(
        observations,
        configuration=IdentityConfiguration(unmatched_cost=unmatched_cost),
    )
    return result.to_record()


__all__ = [
    "DESCRIPTOR_VARIANTS",
    "PATCH_COVERAGES",
    "UNMATCHED_COSTS",
    "DescriptorVariant",
    "descriptors_from_batch",
    "foreground_patch_descriptor",
    "match_descriptor_variant",
    "unit_vector",
]
