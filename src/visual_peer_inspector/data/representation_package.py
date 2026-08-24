"""Join verified RGB media and label-free mask routes for Notebook 07."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from visual_peer_inspector.data.mask_route_package import verify_mask_route_package
from visual_peer_inspector.data.model_safe_package import (
    file_sha256,
    read_jsonl,
    verify_model_input_package,
)


EXPECTED_ROUTE_COUNTS = {
    "supplied_mask": 1_640,
    "reference_box": 1_640,
    "automatic_box": 1_616,
}


@dataclass(frozen=True)
class RepresentationCandidate:
    """One route-local object candidate with no evaluator information."""

    candidate_id: str
    opaque_scene_id: str
    view_id: str
    route: str
    prompt_key: str
    image_path: Path
    image_sha256: str
    box_xyxy: tuple[int, int, int, int]
    foreground_pixels: int
    mask_sha256: str
    mask_rle: Mapping[str, Any]
    predicted_quality_score: float | None


@dataclass(frozen=True)
class RepresentationInputAudit:
    scene_count: int
    image_count: int
    route_counts: Mapping[str, int]
    insufficient_peer_images: Mapping[str, int]
    membership_sha256: str
    rgb_manifest_sha256: str
    route_manifest_sha256: str
    join_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_count": self.scene_count,
            "image_count": self.image_count,
            "route_counts": dict(self.route_counts),
            "insufficient_peer_images": dict(self.insufficient_peer_images),
            "membership_sha256": self.membership_sha256,
            "rgb_manifest_sha256": self.rgb_manifest_sha256,
            "route_manifest_sha256": self.route_manifest_sha256,
            "join_sha256": self.join_sha256,
        }


def _candidate_id(scene_id: str, view_id: str, route: str, prompt_key: str) -> str:
    payload = f"{scene_id}|{view_id}|{route}|{prompt_key}".encode("utf-8")
    return "C" + sha256(payload).hexdigest()[:20]


def load_representation_candidates(
    rgb_root: Path,
    mask_route_root: Path,
) -> tuple[dict[tuple[str, str, str], list[RepresentationCandidate]], RepresentationInputAudit]:
    """Verify both packages and return candidates grouped by image and route."""

    rgb_root = rgb_root.expanduser().resolve()
    mask_route_root = mask_route_root.expanduser().resolve()
    rgb_audit = verify_model_input_package(rgb_root)
    route_record = verify_mask_route_package(mask_route_root)
    if rgb_audit.membership_sha256 != route_record["membership_sha256"]:
        raise ValueError("RGB and mask-route packages contain different scenes")

    rgb_rows = read_jsonl(rgb_root / "neutral_corpus_manifest.jsonl")
    expected_images: dict[tuple[str, str], Path] = {}
    for row in rgb_rows:
        scene_id = str(row["opaque_scene_id"])
        for view_id in row["view_ids"]:
            key = (scene_id, str(view_id))
            path = rgb_root / "media" / scene_id / "rgb" / f"{view_id}.png"
            expected_images[key] = path

    grouped: dict[tuple[str, str, str], list[RepresentationCandidate]] = defaultdict(list)
    route_counts: Counter[str] = Counter()
    join_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    route_rows = read_jsonl(mask_route_root / "mask_route_manifest.jsonl")
    if len(route_rows) != len(expected_images):
        raise ValueError("Mask-route manifest does not cover every RGB image")
    for row in route_rows:
        scene_id = str(row["opaque_scene_id"])
        view_id = str(row["view_id"])
        image_key = (scene_id, view_id)
        if image_key not in expected_images:
            raise ValueError("Mask-route row refers to an unknown RGB image")
        image_path = expected_images[image_key]
        image_digest = file_sha256(image_path)
        if image_digest != str(row["image_sha256"]):
            raise ValueError("RGB image hash differs from its mask-route row")
        for route, items in sorted(row["routes"].items()):
            for item in items:
                prompt_key = str(item["prompt_key"])
                identifier = _candidate_id(scene_id, view_id, route, prompt_key)
                if identifier in seen_ids:
                    raise ValueError("Derived representation candidate IDs are not unique")
                seen_ids.add(identifier)
                score = item.get("predicted_quality_score")
                candidate = RepresentationCandidate(
                    candidate_id=identifier,
                    opaque_scene_id=scene_id,
                    view_id=view_id,
                    route=route,
                    prompt_key=prompt_key,
                    image_path=image_path,
                    image_sha256=image_digest,
                    box_xyxy=tuple(int(value) for value in item["box_xyxy"]),
                    foreground_pixels=int(item["foreground_pixels"]),
                    mask_sha256=str(item["mask_sha256"]),
                    mask_rle=dict(item["mask_rle"]),
                    predicted_quality_score=None if score is None else float(score),
                )
                grouped[(scene_id, view_id, route)].append(candidate)
                route_counts[route] += 1
                join_rows.append(
                    {
                        "candidate_id": identifier,
                        "image_sha256": image_digest,
                        "mask_sha256": candidate.mask_sha256,
                    }
                )
    if dict(route_counts) != EXPECTED_ROUTE_COUNTS:
        raise ValueError("Representation candidates have unexpected route counts")
    actual_images = {(scene, view) for scene, view, _route in grouped}
    if actual_images != set(expected_images):
        raise ValueError("Representation candidates do not cover every RGB image")
    insufficient = Counter()
    for (_scene, _view, route), candidates in grouped.items():
        if len(candidates) < 3:
            insufficient[route] += 1
    join_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(join_rows, key=lambda value: value["candidate_id"])
    ).encode("utf-8")
    audit = RepresentationInputAudit(
        scene_count=rgb_audit.scene_count,
        image_count=rgb_audit.image_count,
        route_counts=dict(route_counts),
        insufficient_peer_images=dict(insufficient),
        membership_sha256=rgb_audit.membership_sha256,
        rgb_manifest_sha256=rgb_audit.manifest_sha256,
        route_manifest_sha256=str(route_record["manifest_sha256"]),
        join_sha256=sha256(join_payload).hexdigest(),
    )
    return dict(grouped), audit


__all__ = [
    "EXPECTED_ROUTE_COUNTS",
    "RepresentationCandidate",
    "RepresentationInputAudit",
    "load_representation_candidates",
]
