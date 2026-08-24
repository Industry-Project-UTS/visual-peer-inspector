"""Prepare one deterministic masked crop for every frozen visual encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from visual_peer_inspector.models.sam2 import decode_binary_mask


@dataclass(frozen=True)
class CropConfiguration:
    """Fixed crop geometry used by the encoder comparison."""

    input_size: int = 224
    padding_fraction: float = 0.15
    background_rgb: tuple[int, int, int] = (238, 241, 245)
    minimum_foreground_pixels: int = 4

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= self.padding_fraction <= 1.0:
            raise ValueError("padding_fraction must be between zero and one")
        if len(self.background_rgb) != 3 or any(
            not 0 <= int(value) <= 255 for value in self.background_rgb
        ):
            raise ValueError("background_rgb must contain three byte values")
        if self.minimum_foreground_pixels <= 0:
            raise ValueError("minimum_foreground_pixels must be positive")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedCrop:
    """An encoder-ready crop and the transform required for heatmap projection."""

    rgb: np.ndarray
    foreground: np.ndarray
    source_box_xyxy: tuple[int, int, int, int]
    padded_box_xyxy: tuple[int, int, int, int]
    configuration_sha256: str

    @property
    def sha256(self) -> str:
        digest = sha256()
        digest.update(self.rgb.tobytes(order="C"))
        digest.update(self.foreground.astype(np.uint8).tobytes(order="C"))
        digest.update(self.configuration_sha256.encode("ascii"))
        return digest.hexdigest()

    def patch_validity(self, grid_size: int, *, minimum_fraction: float = 0.05) -> np.ndarray:
        """Return the foreground coverage of each spatial patch."""

        if grid_size <= 0 or self.foreground.shape[0] % grid_size:
            raise ValueError("grid_size must divide the prepared crop size")
        if not 0.0 <= minimum_fraction <= 1.0:
            raise ValueError("minimum_fraction must be between zero and one")
        patch = self.foreground.shape[0] // grid_size
        coverage = self.foreground.reshape(
            grid_size, patch, grid_size, patch
        ).mean(axis=(1, 3))
        return coverage >= minimum_fraction


def _validated_rgb(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("RGB input must have shape height x width x 3")
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError("RGB input must contain byte values")
            array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def _validated_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, Mapping):
        array = decode_binary_mask(mask)
    else:
        array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError("Mask dimensions must match the RGB image")
    if array.dtype == bool:
        return np.ascontiguousarray(array)
    if not np.all(np.logical_or(array == 0, array == 1)):
        raise ValueError("Mask values must be binary")
    return np.ascontiguousarray(array.astype(bool))


def _square_padded_box(
    mask: np.ndarray,
    *,
    padding_fraction: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("Object mask cannot be empty")
    source = (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
    x1, y1, x2, y2 = source
    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1.0 + 2.0 * padding_fraction)
    side = max(1, int(np.ceil(side)))
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    padded_x1 = int(np.floor(centre_x - side / 2.0))
    padded_y1 = int(np.floor(centre_y - side / 2.0))
    return source, (padded_x1, padded_y1, padded_x1 + side, padded_y1 + side)


def _extract_square(
    array: np.ndarray,
    box: Sequence[int],
    *,
    fill: Sequence[int] | int,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(value) for value in box]
    side = x2 - x1
    if side <= 0 or y2 - y1 != side:
        raise ValueError("Padded crop box must be a positive square")
    if array.ndim == 3:
        output = np.empty((side, side, array.shape[2]), dtype=array.dtype)
        output[...] = np.asarray(fill, dtype=array.dtype)
    else:
        output = np.full((side, side), fill, dtype=array.dtype)
    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(array.shape[1], x2)
    source_y2 = min(array.shape[0], y2)
    target_x1 = source_x1 - x1
    target_y1 = source_y1 - y1
    output[
        target_y1 : target_y1 + (source_y2 - source_y1),
        target_x1 : target_x1 + (source_x2 - source_x1),
    ] = array[source_y1:source_y2, source_x1:source_x2]
    return output


def prepare_masked_crop(
    image: Any,
    mask: Any,
    *,
    configuration: CropConfiguration = CropConfiguration(),
) -> PreparedCrop:
    """Mask the background, square-pad the object and resize once for all models."""

    rgb = _validated_rgb(image)
    foreground = _validated_mask(mask, rgb.shape[:2])
    if int(foreground.sum()) < configuration.minimum_foreground_pixels:
        raise ValueError("Object mask contains too little foreground")
    source_box, padded_box = _square_padded_box(
        foreground,
        padding_fraction=configuration.padding_fraction,
    )
    neutral = np.asarray(configuration.background_rgb, dtype=np.uint8)
    masked = np.where(foreground[:, :, None], rgb, neutral)
    square_rgb = _extract_square(masked, padded_box, fill=configuration.background_rgb)
    square_mask = _extract_square(foreground, padded_box, fill=False)
    size = (configuration.input_size, configuration.input_size)
    prepared_rgb = np.asarray(
        Image.fromarray(square_rgb, mode="RGB").resize(size, Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    prepared_mask = np.asarray(
        Image.fromarray(square_mask.astype(np.uint8) * 255, mode="L").resize(
            size, Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0
    return PreparedCrop(
        rgb=np.ascontiguousarray(prepared_rgb),
        foreground=np.ascontiguousarray(prepared_mask),
        source_box_xyxy=source_box,
        padded_box_xyxy=padded_box,
        configuration_sha256=configuration.sha256,
    )


__all__ = ["CropConfiguration", "PreparedCrop", "prepare_masked_crop"]
