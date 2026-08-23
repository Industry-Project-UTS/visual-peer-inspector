"""Load a validated one-view sample from a selectively extracted ToysAD scene."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .scene_catalog import anomaly_variant, scene2d_anomaly_flags


@dataclass(frozen=True)
class ToysObjectView:
    object_index: int
    object_name: str
    blend_path: str
    is_anomaly: bool
    anomaly_variant: str | None
    segmentation_path: Path


@dataclass(frozen=True)
class ToysSceneView:
    scene_id: str
    split: str
    view_id: str
    rgb_path: Path
    silhouette_path: Path
    objects: tuple[ToysObjectView, ...]

    @property
    def anomaly_count(self) -> int:
        return sum(item.is_anomaly for item in self.objects)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required extracted member is missing: {path}")
    return path


def load_toys_scene_view(
    extraction_root: Path,
    scene_id: str,
    *,
    split: str,
    view_id: str = "0000",
) -> ToysSceneView:
    """Load paths and labels for one view, enforcing metadata alignment."""

    if not (len(view_id) == 4 and view_id.isdigit()):
        raise ValueError(f"Expected a four-digit view id, found {view_id!r}")
    if "/" in scene_id or "\\" in scene_id or scene_id in {"", ".", ".."}:
        raise ValueError(f"Unsafe scene id: {scene_id!r}")

    scene_root = extraction_root / "toysad8k" / scene_id
    scene2d_path = _require_file(scene_root / "scene2d.metadata.json")
    scene3d_path = _require_file(scene_root / "scene3d.metadata.json")
    with scene2d_path.open("r", encoding="utf-8") as handle:
        scene2d = json.load(handle)
    with scene3d_path.open("r", encoding="utf-8") as handle:
        scene3d = json.load(handle)

    raw_labels = [int(value) for value in scene2d.get("anomaly_labels", [])]
    blend_paths = [str(value) for value in scene2d.get("blend_paths", [])]
    objects = scene3d.get("objects", [])
    if not raw_labels or len(raw_labels) != len(blend_paths):
        raise ValueError(f"Invalid scene2d labels/paths for {scene_id}")
    if not isinstance(objects, list) or len(objects) != len(raw_labels):
        raise ValueError(f"scene2d/scene3d object count mismatch for {scene_id}")

    anomaly_flags = scene2d_anomaly_flags(raw_labels)
    object_views: list[ToysObjectView] = []
    for index, (item, blend_path, is_anomaly) in enumerate(
        zip(objects, blend_paths, anomaly_flags, strict=True)
    ):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid scene3d object {index} for {scene_id}")
        object_name = str(item.get("name", ""))
        object_path = str(item.get("path", ""))
        if object_path != blend_path:
            raise ValueError(
                f"scene2d/scene3d path mismatch for {scene_id} object {index}"
            )
        if not object_name or "/" in object_name or "\\" in object_name:
            raise ValueError(f"Unsafe object name for {scene_id}: {object_name!r}")
        variant = anomaly_variant(blend_path)
        if bool(variant) != bool(is_anomaly):
            raise ValueError(
                f"Label/path anomaly mismatch for {scene_id} object {index}"
            )
        segmentation = _require_file(
            scene_root
            / "segmentations"
            / object_name
            / f"{object_name}_{view_id}.png"
        )
        object_views.append(
            ToysObjectView(
                object_index=index,
                object_name=object_name,
                blend_path=blend_path,
                is_anomaly=bool(is_anomaly),
                anomaly_variant=variant if is_anomaly else None,
                segmentation_path=segmentation,
            )
        )

    return ToysSceneView(
        scene_id=scene_id,
        split=split,
        view_id=view_id,
        rgb_path=_require_file(scene_root / "RGB" / f"{view_id}.png"),
        silhouette_path=_require_file(scene_root / "masks" / f"{view_id}.png"),
        objects=tuple(object_views),
    )
