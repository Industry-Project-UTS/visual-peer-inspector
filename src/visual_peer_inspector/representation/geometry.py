"""Separate shape and layout descriptors for the Notebook 07 geometry branch."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def geometry_feature_matrices(
    boxes_xyxy: Sequence[Sequence[int]],
    foreground_pixels: Sequence[int],
    *,
    image_width: int = 256,
    image_height: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shape and layout features without mixing them with appearance."""

    if len(boxes_xyxy) != len(foreground_pixels):
        raise ValueError("Boxes and foreground counts must have equal length")
    if not boxes_xyxy:
        raise ValueError("At least one object is required")
    shapes = []
    centroids = []
    for box, area in zip(boxes_xyxy, foreground_pixels, strict=True):
        if len(box) != 4:
            raise ValueError("Every box must contain four coordinates")
        x1, y1, x2, y2 = [float(value) for value in box]
        if not 0 <= x1 < x2 <= image_width or not 0 <= y1 < y2 <= image_height:
            raise ValueError("Box is outside the image")
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        mask_area = float(area) / (image_width * image_height)
        shapes.append([width, height, mask_area, np.log(width / height)])
        centroids.append(
            [
                (x1 + x2) / (2.0 * image_width),
                (y1 + y2) / (2.0 * image_height),
            ]
        )
    centre_array = np.asarray(centroids, dtype=np.float64)
    layouts = []
    for index, centre in enumerate(centre_array):
        distances = np.linalg.norm(centre_array - centre, axis=1)
        distances = np.delete(distances, index)
        layouts.append([*centre, *sorted(distances.tolist())])
    return np.asarray(shapes, dtype=np.float64), np.asarray(layouts, dtype=np.float64)


__all__ = ["geometry_feature_matrices"]
