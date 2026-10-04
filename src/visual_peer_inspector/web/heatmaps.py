"""Display-only colour mapping for projected patch mismatch evidence."""
from __future__ import annotations

import numpy as np
from PIL import Image

HEATMAP_STYLE = "blue-cyan-yellow-red/max-v1"


def heatmap_overlay(image: Image.Image, projected: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Colour finite foreground values once; leave all other pixels unchanged.

    Zero is blue and the largest mismatch within this observation is red. This
    scale makes texture visible; colours cannot be compared between objects.
    """
    values = np.asarray(projected, dtype=np.float64)
    foreground = np.asarray(mask, dtype=bool)
    if values.shape != (image.height, image.width) or foreground.shape != values.shape:
        raise ValueError("Heatmap, mask and original image coordinates must agree.")
    valid = foreground & np.isfinite(values)
    output = np.array(image.convert("RGB"), copy=True)
    if not valid.any():
        return Image.fromarray(output)
    evidence = np.maximum(values[valid], 0)
    normalized = evidence / max(float(evidence.max()), 1e-12)
    stops = np.array([0, .33, .67, 1.])
    colours = np.array([[38, 65, 190], [22, 190, 213], [255, 214, 54], [220, 40, 35]])
    mapped = np.stack([np.interp(normalized, stops, colours[:, channel]) for channel in range(3)], axis=1)
    output[valid] = np.rint(.8 * mapped + .2 * output[valid]).astype(np.uint8)
    return Image.fromarray(output)
