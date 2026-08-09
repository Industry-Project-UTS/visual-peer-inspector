from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class SceneObject:
    index: int
    name: str
    is_anomaly: bool
    anomaly_variant: str | None
    mask_path: Path


@dataclass(frozen=True)
class SceneView:
    scene_id: str
    split: str
    view_id: str
    image_path: Path
    objects: tuple[SceneObject, ...]

    @property
    def anomaly_count(self) -> int:
        return sum(item.is_anomaly for item in self.objects)


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def _variant_from_path(blend_path: str) -> str | None:
    path = PurePosixPath(blend_path)
    asset_name = path.parent.name
    file_name = path.stem
    if file_name == asset_name:
        return None
    prefix = f"{asset_name}-"
    if file_name.startswith(prefix):
        return file_name[len(prefix) :]
    raise ValueError(f"Unexpected asset path: {blend_path}")


def load_scene_view(
    data_root: Path,
    scene_id: str,
    *,
    split: str,
    view_id: str = "0000",
) -> SceneView:
    if not scene_id or any(value in scene_id for value in ("/", "\\")):
        raise ValueError(f"Unsafe scene id: {scene_id!r}")
    if len(view_id) != 4 or not view_id.isdigit():
        raise ValueError(f"Invalid view id: {view_id!r}")

    scene_root = data_root / "toysad8k" / scene_id
    scene2d_path = _required_file(scene_root / "scene2d.metadata.json")
    scene3d_path = _required_file(scene_root / "scene3d.metadata.json")
    with scene2d_path.open("r", encoding="utf-8") as handle:
        scene2d = json.load(handle)
    with scene3d_path.open("r", encoding="utf-8") as handle:
        scene3d = json.load(handle)

    labels = [int(value) for value in scene2d.get("anomaly_labels", [])]
    blend_paths = [str(value) for value in scene2d.get("blend_paths", [])]
    objects = scene3d.get("objects", [])
    if not labels or any(value not in (0, 1) for value in labels):
        raise ValueError(f"Invalid anomaly labels for {scene_id}")
    if len(labels) != len(blend_paths) or len(labels) != len(objects):
        raise ValueError(f"Object metadata is not aligned for {scene_id}")

    loaded_objects = []
    for index, (label, blend_path, item) in enumerate(
        zip(labels, blend_paths, objects, strict=True)
    ):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid object metadata for {scene_id}")
        name = str(item.get("name", ""))
        object_path = str(item.get("path", ""))
        if not name or any(value in name for value in ("/", "\\")):
            raise ValueError(f"Unsafe object name in {scene_id}")
        if object_path != blend_path:
            raise ValueError(f"Object path mismatch in {scene_id}")

        variant = _variant_from_path(blend_path)
        is_anomaly = label == 0
        if is_anomaly != (variant is not None):
            raise ValueError(f"Label and asset path disagree in {scene_id}")
        mask_path = _required_file(
            scene_root / "segmentations" / name / f"{name}_{view_id}.png"
        )
        loaded_objects.append(
            SceneObject(
                index=index,
                name=name,
                is_anomaly=is_anomaly,
                anomaly_variant=variant,
                mask_path=mask_path,
            )
        )

    return SceneView(
        scene_id=scene_id,
        split=split,
        view_id=view_id,
        image_path=_required_file(scene_root / "RGB" / f"{view_id}.png"),
        objects=tuple(loaded_objects),
    )
