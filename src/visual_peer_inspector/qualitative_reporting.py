"""Deterministic qualitative panels for model predictions and private evaluation.

The prediction renderer accepts only model-facing fields. Evaluator references are
handled by a separate function so protected labels cannot enter publication panels
by accident.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ImageInput: TypeAlias = str | Path | np.ndarray | Image.Image
MaskInput: TypeAlias = str | Path | np.ndarray | Image.Image
CoordinateSpace: TypeAlias = Literal["pixel", "normalized"]


_PROTECTED_FIELD_PARTS = (
    "anomaly",
    "asset",
    "category",
    "crosswalk",
    "evaluator",
    "ground_truth",
    "label",
    "official_split",
    "protected",
    "reference",
    "source_id",
    "source_path",
)
_PREDICTION_PALETTE = (
    (31, 119, 180),
    (255, 127, 14),
    (148, 103, 189),
    (23, 190, 207),
    (44, 160, 44),
    (227, 119, 194),
)
_FUSED_COLOUR = (214, 39, 40)
_REFERENCE_COLOUR = (20, 145, 64)
_BACKGROUND = (246, 248, 251)
_TEXT = (30, 39, 49)
_MUTED = (90, 101, 114)
_HEADER = (31, 78, 121)


class ProtectedFieldError(ValueError):
    """Raised when evaluator-only information enters prediction-safe content."""


@dataclass(frozen=True)
class PredictedObject:
    """One predicted object and the evidence needed for a visual result."""

    object_id: str
    box_xyxy: tuple[float, float, float, float]
    sam_mask: MaskInput | None
    p1_score: float | None
    p1_rank: int | None

    def __post_init__(self) -> None:
        _require_identifier(self.object_id, "object_id")
        _validate_box(self.box_xyxy)
        if self.p1_score is not None and not math.isfinite(self.p1_score):
            raise ValueError("P1 score must be finite")
        if self.p1_rank is not None and self.p1_rank < 1:
            raise ValueError("P1 rank must be positive")


@dataclass(frozen=True)
class QualitativeView:
    """Prediction evidence for one uploaded view or one declared missing view."""

    view_id: str
    rgb: ImageInput | None
    objects: tuple[PredictedObject, ...] = ()
    heatmap: MaskInput | None = None
    coordinate_space: CoordinateSpace = "pixel"
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.view_id, "view_id")
        if self.coordinate_space not in {"pixel", "normalized"}:
            raise ValueError("coordinate_space must be 'pixel' or 'normalized'")
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"Duplicate object IDs in view {self.view_id}")
        if self.rgb is None:
            if not self.missing_reason:
                raise ValueError("Missing RGB input requires missing_reason")
            if self.objects or self.heatmap is not None:
                raise ValueError("A missing view cannot contain predictions")
        elif self.missing_reason is not None:
            raise ValueError("A present view cannot include missing_reason")


@dataclass(frozen=True)
class QualitativeResult:
    """Model-facing multi-view result suitable for a publication panel."""

    result_id: str
    views: tuple[QualitativeView, ...]
    fused_odd_one_out: str | None
    agreement: float | None
    margin: float | None
    review_state: str
    publication_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.result_id, "result_id")
        if not 2 <= len(self.views) <= 5:
            raise ValueError("Qualitative panels require between 2 and 5 views")
        view_ids = [item.view_id for item in self.views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("Qualitative views must have unique view IDs")
        if self.fused_odd_one_out is not None:
            _require_identifier(self.fused_odd_one_out, "fused_odd_one_out")
            predicted_ids = {
                item.object_id for view in self.views for item in view.objects
            }
            if self.fused_odd_one_out not in predicted_ids:
                raise ValueError("Fused odd-one-out is absent from every view")
        if self.agreement is not None and not 0.0 <= self.agreement <= 1.0:
            raise ValueError("Agreement must be between 0 and 1")
        if self.margin is not None and (
            not math.isfinite(self.margin) or self.margin < 0.0
        ):
            raise ValueError("Margin must be finite and non-negative")
        if not self.review_state.strip():
            raise ValueError("review_state cannot be empty")
        reject_protected_fields(self.publication_metadata)


@dataclass(frozen=True)
class EvaluatorObject:
    """Private reference geometry for one object in one view."""

    object_id: str
    box_xyxy: tuple[float, float, float, float] | None = None
    mask: MaskInput | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.object_id, "object_id")
        if self.box_xyxy is not None:
            _validate_box(self.box_xyxy)


@dataclass(frozen=True)
class EvaluatorView:
    """Private reference objects aligned to one prediction view."""

    view_id: str
    objects: tuple[EvaluatorObject, ...]
    coordinate_space: CoordinateSpace = "pixel"

    def __post_init__(self) -> None:
        _require_identifier(self.view_id, "view_id")
        if self.coordinate_space not in {"pixel", "normalized"}:
            raise ValueError("coordinate_space must be 'pixel' or 'normalized'")


@dataclass(frozen=True)
class EvaluatorOverlay:
    """Private labels that may appear only in evaluator-overlay panels."""

    views: tuple[EvaluatorView, ...]
    reference_odd_one_out: str | None = None
    acceptable_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        view_ids = [item.view_id for item in self.views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("Evaluator views must have unique view IDs")
        if self.reference_odd_one_out is not None:
            _require_identifier(
                self.reference_odd_one_out,
                "reference_odd_one_out",
            )
        for reference_id in self.acceptable_reference_ids:
            _require_identifier(reference_id, "acceptable_reference_ids")
        if len(self.acceptable_reference_ids) != len(
            set(self.acceptable_reference_ids)
        ):
            raise ValueError("Acceptable reference IDs must be unique")

    @property
    def reference_header(self) -> str:
        """Return every acceptable evaluator identity for the private header."""

        if self.acceptable_reference_ids:
            return ", ".join(self.acceptable_reference_ids)
        return self.reference_odd_one_out or "not supplied"


def reject_protected_fields(value: Any, *, path: str = "metadata") -> None:
    """Reject protected field names recursively in publication metadata."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if any(part in key for part in _PROTECTED_FIELD_PARTS):
                raise ProtectedFieldError(
                    f"Prediction metadata contains protected field at {path}.{raw_key}"
                )
            reject_protected_fields(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_protected_fields(child, path=f"{path}[{index}]")


def select_publication_results(
    results: Sequence[QualitativeResult],
    *,
    count: int,
    namespace: str = "visual-peer-inspector-qualitative-v1",
) -> tuple[QualitativeResult, ...]:
    """Select results reproducibly without using labels or evaluator fields."""

    if count < 1:
        raise ValueError("count must be positive")
    if not namespace.strip():
        raise ValueError("namespace cannot be empty")
    identifiers = [item.result_id for item in results]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Result IDs must be unique for deterministic selection")

    def stable_key(item: QualitativeResult) -> tuple[str, str]:
        payload = f"{namespace}|{item.result_id}".encode()
        return sha256(payload).hexdigest(), item.result_id

    return tuple(sorted(results, key=stable_key)[:count])


def render_prediction_panel(
    result: QualitativeResult,
    *,
    panel_size: int = 256,
) -> Image.Image:
    """Render a publication-safe panel containing predictions only."""

    return _render_panel(
        result,
        evaluator=None,
        panel_size=panel_size,
        mode="prediction-only",
    )


def render_evaluator_overlay_panel(
    result: QualitativeResult,
    evaluator: EvaluatorOverlay,
    *,
    panel_size: int = 256,
) -> Image.Image:
    """Render a private panel with prediction and evaluator geometry."""

    prediction_view_ids = {view.view_id for view in result.views}
    unknown = {view.view_id for view in evaluator.views}.difference(prediction_view_ids)
    if unknown:
        raise ValueError(f"Evaluator overlay contains unknown views: {sorted(unknown)}")
    return _render_panel(
        result,
        evaluator=evaluator,
        panel_size=panel_size,
        mode="evaluator-overlay",
    )


def save_panel_png(image: Image.Image, output_path: str | Path) -> Path:
    """Save a panel with fixed PNG settings and no run-specific metadata."""

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("Qualitative panels must be saved as PNG files")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    return path


def _render_panel(
    result: QualitativeResult,
    *,
    evaluator: EvaluatorOverlay | None,
    panel_size: int,
    mode: Literal["prediction-only", "evaluator-overlay"],
) -> Image.Image:
    if panel_size < 128:
        raise ValueError("panel_size must be at least 128 pixels")

    views = tuple(sorted(result.views, key=lambda item: _natural_key(item.view_id)))
    evaluator_by_view = (
        {item.view_id: item for item in evaluator.views}
        if evaluator is not None
        else {}
    )
    font = ImageFont.load_default()
    column_gap = 14
    margin = 18
    label_height = 42
    header_height = 112 if evaluator is not None else 94
    row_gap = 26
    column_width = panel_size
    canvas_width = (
        2 * margin + len(views) * column_width + (len(views) - 1) * column_gap
    )
    row_count = 5
    canvas_height = (
        header_height
        + row_count * (panel_size + label_height)
        + (row_count - 1) * row_gap
        + margin
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_width, 8), fill=_HEADER)
    _draw_text(draw, (margin, 18), f"Result {result.result_id}", font, _TEXT)
    _draw_text(
        draw,
        (margin, 38),
        "Per-view P1 candidates | Cross-view identity is not evaluated"
        if result.review_state == "prediction_only" and result.fused_odd_one_out is None
        else _summary_text(result),
        font,
        _TEXT,
    )
    _draw_text(
        draw,
        (margin, 58),
        f"Mode: {mode}",
        font,
        _MUTED,
    )
    if evaluator is not None:
        _draw_text(
            draw,
            (margin, 78),
            f"Private evaluator references: {evaluator.reference_header}",
            font,
            _REFERENCE_COLOUR,
        )

    for column, view in enumerate(views):
        x = margin + column * (column_width + column_gap)
        reference_view = evaluator_by_view.get(view.view_id)
        if view.rgb is None:
            for row_index, title in enumerate(
                (
                    "Predicted boxes",
                    "SAM mask overlay",
                    "Selected binary mask",
                    "Selected object on black",
                    "Unusual-region heatmap",
                )
            ):
                y = header_height + row_index * (panel_size + label_height + row_gap)
                _draw_missing_panel(
                    canvas,
                    x,
                    y,
                    panel_size,
                    title,
                    view.missing_reason or "View unavailable",
                    font,
                )
            continue

        rgb = _load_rgb(view.rgb)
        fitted, scale, offset = _fit_rgb(rgb, panel_size)
        box_panel = fitted.copy()
        _draw_prediction_boxes(
            box_panel,
            view,
            rgb.size,
            scale,
            offset,
            result.fused_odd_one_out,
            font,
        )
        if reference_view is not None:
            _draw_reference_boxes(
                box_panel,
                reference_view,
                rgb.size,
                scale,
                offset,
            )

        mask_panel = _prediction_mask_overlay(
            fitted,
            view,
            rgb.size,
            scale,
            offset,
            result.fused_odd_one_out,
        )
        if reference_view is not None:
            mask_panel = _add_reference_mask_edges(
                mask_panel,
                reference_view,
                rgb.size,
                scale,
                offset,
            )

        selected_id = result.fused_odd_one_out
        per_view_selection = (
            result.review_state == "prediction_only" and selected_id is None
        )
        if per_view_selection:
            candidates = [item.object_id for item in view.objects if item.p1_rank == 1]
            selected_id = candidates[0] if len(candidates) == 1 else None
        binary_mask_panel, masked_object_panel, selected_object = (
            _selected_object_mask_panels(
                fitted,
                view,
                rgb.size,
                scale,
                offset,
                selected_id,
            )
        )

        heatmap_panel = _heatmap_overlay(
            fitted,
            view.heatmap,
            rgb.size,
            scale,
            offset,
        )
        rows = (
            (box_panel, _box_caption(view)),
            (mask_panel, _mask_caption(view)),
            (
                binary_mask_panel,
                _selected_mask_caption(
                    selected_object,
                    "Per-view top mask" if per_view_selection else "Binary mask",
                ),
            ),
            (
                masked_object_panel,
                _selected_mask_caption(
                    selected_object,
                    "Per-view top on black"
                    if per_view_selection
                    else "Object on black",
                ),
            ),
            (heatmap_panel, _heatmap_caption(view)),
        )
        for row_index, (panel, caption) in enumerate(rows):
            y = header_height + row_index * (panel_size + label_height + row_gap)
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x, y, x + panel_size - 1, y + panel_size - 1),
                outline=(184, 192, 201),
                width=1,
            )
            _draw_text(draw, (x, y + panel_size + 7), caption, font, _TEXT)
    return canvas


def _require_identifier(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    if any(character in value for character in ("\n", "\r", "\t")):
        raise ValueError(f"{field_name} contains control characters")


def _validate_box(box: Sequence[float]) -> None:
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        raise ValueError("Boxes must contain four finite coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box maximum coordinates must exceed minimum coordinates")


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in parts if part
    )


def _load_rgb(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return image.convert("RGB")
    array = np.asarray(source)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("RGB arrays must have shape H x W x 3 or H x W x 4")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("RGB arrays must contain finite values")
        maximum = float(array.max()) if array.size else 0.0
        array = array * 255.0 if maximum <= 1.0 else array
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3], mode="RGB")


def _load_mask(source: MaskInput, expected_size: tuple[int, int]) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source.convert("L")
        array = np.asarray(image)
    elif isinstance(source, (str, Path)):
        with Image.open(source) as image:
            array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(source)
        if array.ndim == 3:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Mask and heatmap arrays must be two-dimensional")
    if array.shape != (expected_size[1], expected_size[0]):
        raise ValueError(
            f"Mask size {array.shape[::-1]} does not match RGB size {expected_size}"
        )
    return array


def _fit_rgb(
    image: Image.Image,
    panel_size: int,
) -> tuple[Image.Image, float, tuple[int, int]]:
    width, height = image.size
    scale = min(panel_size / width, panel_size / height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    offset = (
        (panel_size - resized_size[0]) // 2,
        (panel_size - resized_size[1]) // 2,
    )
    panel = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
    panel.paste(resized, offset)
    return panel, scale, offset


def _scaled_box(
    box: Sequence[float],
    coordinate_space: CoordinateSpace,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    if coordinate_space == "normalized":
        x1, x2 = x1 * original_size[0], x2 * original_size[0]
        y1, y2 = y1 * original_size[1], y2 * original_size[1]
    x1 = min(max(x1, 0.0), float(original_size[0]))
    x2 = min(max(x2, 0.0), float(original_size[0]))
    y1 = min(max(y1, 0.0), float(original_size[1]))
    y2 = min(max(y2, 0.0), float(original_size[1]))
    return (
        round(offset[0] + x1 * scale),
        round(offset[1] + y1 * scale),
        round(offset[0] + x2 * scale),
        round(offset[1] + y2 * scale),
    )


def _object_colour(object_id: str, fused_object_id: str | None) -> tuple[int, int, int]:
    if object_id == fused_object_id:
        return _FUSED_COLOUR
    index = int(sha256(object_id.encode("utf-8")).hexdigest()[:8], 16)
    return _PREDICTION_PALETTE[index % len(_PREDICTION_PALETTE)]


def _ordered_objects(view: QualitativeView) -> tuple[PredictedObject, ...]:
    return tuple(
        sorted(
            view.objects,
            key=lambda item: (
                item.p1_rank if item.p1_rank is not None else 10**9,
                item.object_id,
            ),
        )
    )


def _draw_prediction_boxes(
    panel: Image.Image,
    view: QualitativeView,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
    fused_object_id: str | None,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(panel)
    for item in _ordered_objects(view):
        box = _scaled_box(
            item.box_xyxy,
            view.coordinate_space,
            original_size,
            scale,
            offset,
        )
        colour = _object_colour(item.object_id, fused_object_id)
        width = 4 if item.object_id == fused_object_id else 2
        draw.rectangle(box, outline=colour, width=width)
        score = "NA" if item.p1_score is None else f"{item.p1_score:.3f}"
        rank = "NA" if item.p1_rank is None else str(item.p1_rank)
        label = _truncate(f"{item.object_id}  r{rank}  {score}", 34)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0, min(box[0], panel.width - text_width - 6))
        label_y = max(0, box[1] - text_height - 6)
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + text_width + 6,
                label_y + text_height + 5,
            ),
            fill=colour,
        )
        draw.text((label_x + 3, label_y + 2), label, font=font, fill="white")


def _draw_reference_boxes(
    panel: Image.Image,
    reference_view: EvaluatorView,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(panel)
    for item in sorted(reference_view.objects, key=lambda value: value.object_id):
        if item.box_xyxy is None:
            continue
        box = _scaled_box(
            item.box_xyxy,
            reference_view.coordinate_space,
            original_size,
            scale,
            offset,
        )
        _draw_dashed_rectangle(draw, box, fill=_REFERENCE_COLOUR, width=2)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash: int = 7,
) -> None:
    x1, y1, x2, y2 = box
    for start in range(x1, x2 + 1, 2 * dash):
        draw.line((start, y1, min(start + dash, x2), y1), fill=fill, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=fill, width=width)
    for start in range(y1, y2 + 1, 2 * dash):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=fill, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=fill, width=width)


def _prediction_mask_overlay(
    fitted_rgb: Image.Image,
    view: QualitativeView,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
    fused_object_id: str | None,
) -> Image.Image:
    panel = fitted_rgb.copy()
    for item in reversed(_ordered_objects(view)):
        if item.sam_mask is None:
            continue
        mask = _load_mask(item.sam_mask, original_size) > 0
        colour = _object_colour(item.object_id, fused_object_id)
        alpha = 0.52 if item.object_id == fused_object_id else 0.32
        panel = _blend_mask(panel, mask, colour, alpha, scale, offset)
    if not view.objects:
        draw = ImageDraw.Draw(panel)
        _draw_centered_text(draw, panel.size, "No proposals", ImageFont.load_default())
    return panel


def _selected_object_mask_panels(
    fitted_rgb: Image.Image,
    view: QualitativeView,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
    fused_object_id: str | None,
) -> tuple[Image.Image, Image.Image, PredictedObject | None]:
    """Render the selected prediction as a binary mask and isolated RGB object."""

    selected = next(
        (
            item
            for item in _ordered_objects(view)
            if item.object_id == fused_object_id and item.sam_mask is not None
        ),
        None,
    )
    if selected is None:
        reason = (
            "No fused object in this view"
            if fused_object_id is not None
            else "No fused object selected"
        )
        return (
            _unavailable_prediction_panel(fitted_rgb.size, reason),
            _unavailable_prediction_panel(fitted_rgb.size, reason),
            None,
        )

    mask = _load_mask(selected.sam_mask, original_size) > 0
    full_mask = _fit_binary_mask(mask, fitted_rgb.size, scale, offset)
    binary = np.zeros((fitted_rgb.height, fitted_rgb.width, 3), dtype=np.uint8)
    binary[full_mask] = 255
    source = np.asarray(fitted_rgb)
    isolated = np.zeros_like(source)
    isolated[full_mask] = source[full_mask]
    return (
        Image.fromarray(binary, mode="RGB"),
        Image.fromarray(isolated, mode="RGB"),
        selected,
    )


def _fit_binary_mask(
    source_mask: np.ndarray,
    panel_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
) -> np.ndarray:
    resized_size = (
        max(1, round(source_mask.shape[1] * scale)),
        max(1, round(source_mask.shape[0] * scale)),
    )
    resized = Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").resize(
        resized_size,
        Image.Resampling.NEAREST,
    )
    full_mask = Image.new("L", panel_size, 0)
    full_mask.paste(resized, offset)
    return np.asarray(full_mask) > 0


def _unavailable_prediction_panel(
    panel_size: tuple[int, int],
    reason: str,
) -> Image.Image:
    panel = Image.new("RGB", panel_size, (0, 0, 0))
    draw = ImageDraw.Draw(panel)
    _draw_centered_text(draw, panel_size, reason, ImageFont.load_default())
    return panel


def _add_reference_mask_edges(
    panel: Image.Image,
    reference_view: EvaluatorView,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
) -> Image.Image:
    output = panel.copy()
    for item in sorted(reference_view.objects, key=lambda value: value.object_id):
        if item.mask is None:
            continue
        mask = (_load_mask(item.mask, original_size) > 0).astype(np.uint8) * 255
        edge = Image.fromarray(mask, mode="L").filter(ImageFilter.FIND_EDGES)
        edge_array = np.asarray(edge) > 0
        output = _blend_mask(
            output,
            edge_array,
            _REFERENCE_COLOUR,
            0.92,
            scale,
            offset,
        )
    return output


def _blend_mask(
    panel: Image.Image,
    source_mask: np.ndarray,
    colour: tuple[int, int, int],
    alpha: float,
    scale: float,
    offset: tuple[int, int],
) -> Image.Image:
    resized_size = (
        max(1, round(source_mask.shape[1] * scale)),
        max(1, round(source_mask.shape[0] * scale)),
    )
    resized_mask = Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L")
    resized_mask = resized_mask.resize(resized_size, Image.Resampling.NEAREST)
    full_mask = Image.new("L", panel.size, 0)
    full_mask.paste(resized_mask, offset)
    mask_array = np.asarray(full_mask) > 0
    image_array = np.asarray(panel).astype(np.float32)
    colour_array = np.asarray(colour, dtype=np.float32)
    image_array[mask_array] = (1.0 - alpha) * image_array[
        mask_array
    ] + alpha * colour_array
    return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB")


def _heatmap_overlay(
    fitted_rgb: Image.Image,
    heatmap: MaskInput | None,
    original_size: tuple[int, int],
    scale: float,
    offset: tuple[int, int],
) -> Image.Image:
    if heatmap is None:
        panel = fitted_rgb.copy()
        draw = ImageDraw.Draw(panel)
        _draw_centered_text(
            draw,
            panel.size,
            "Heatmap not provided",
            ImageFont.load_default(),
        )
        return panel
    values = _load_mask(heatmap, original_size).astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Heatmap must contain at least one finite value")
    minimum = float(values[finite].min())
    maximum = float(values[finite].max())
    normalized = np.zeros_like(values, dtype=np.float32)
    if maximum > minimum:
        normalized[finite] = (values[finite] - minimum) / (maximum - minimum)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * normalized - 1.0)
    colour_map = np.stack((red, green, blue), axis=2)
    colour_map = np.clip(colour_map * 255.0, 0, 255).astype(np.uint8)
    resized_size = (
        max(1, round(original_size[0] * scale)),
        max(1, round(original_size[1] * scale)),
    )
    colour_image = Image.fromarray(colour_map, mode="RGB").resize(
        resized_size,
        Image.Resampling.BILINEAR,
    )
    heat_panel = Image.new("RGB", fitted_rgb.size, (255, 255, 255))
    heat_panel.paste(colour_image, offset)
    valid_mask = Image.fromarray(finite.astype(np.uint8) * 255, mode="L").resize(
        resized_size,
        Image.Resampling.NEAREST,
    )
    full_mask = Image.new("L", fitted_rgb.size, 0)
    full_mask.paste(valid_mask.point(lambda value: round(value * 0.55)), offset)
    return Image.composite(heat_panel, fitted_rgb, full_mask)


def _summary_text(result: QualitativeResult) -> str:
    fused = result.fused_odd_one_out or "none"
    agreement = "NA" if result.agreement is None else f"{result.agreement:.3f}"
    margin = "NA" if result.margin is None else f"{result.margin:.3f}"
    return _truncate(
        f"Odd-one-out: {fused} | agreement: {agreement} | "
        f"margin: {margin} | state: {result.review_state}",
        130,
    )


def _box_caption(view: QualitativeView) -> str:
    if not view.objects:
        return f"{view.view_id} | predicted boxes: 0"
    top = _ordered_objects(view)[0]
    return _truncate(
        f"{view.view_id} | boxes: {len(view.objects)} | top: {top.object_id}",
        44,
    )


def _mask_caption(view: QualitativeView) -> str:
    count = sum(item.sam_mask is not None for item in view.objects)
    return f"SAM mask overlay: {count}/{len(view.objects)} objects"


def _selected_mask_caption(
    selected: PredictedObject | None,
    panel_name: str,
) -> str:
    object_id = selected.object_id if selected is not None else "unavailable"
    return _truncate(f"{panel_name}: {object_id}", 44)


def _heatmap_caption(view: QualitativeView) -> str:
    return "Unusual-region heatmap" if view.heatmap is not None else "No heatmap"


def _draw_missing_panel(
    canvas: Image.Image,
    x: int,
    y: int,
    size: int,
    title: str,
    reason: str,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (x, y, x + size - 1, y + size - 1),
        fill=(233, 236, 240),
        outline=(184, 192, 201),
        width=1,
    )
    _draw_centered_text(draw, (size, size), "View unavailable", font, origin=(x, y))
    _draw_text(
        draw,
        (x, y + size + 7),
        _truncate(f"{title}: {reason}", 44),
        font,
        _MUTED,
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    origin: tuple[int, int] = (0, 0),
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    point = (
        origin[0] + (size[0] - width) // 2,
        origin[1] + (size[1] - height) // 2,
    )
    draw.text(point, text, font=font, fill=_MUTED)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(point, text, font=font, fill=fill)


def _truncate(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 3] + "..."


__all__ = [
    "EvaluatorObject",
    "EvaluatorOverlay",
    "EvaluatorView",
    "PredictedObject",
    "ProtectedFieldError",
    "QualitativeResult",
    "QualitativeView",
    "reject_protected_fields",
    "render_evaluator_overlay_panel",
    "render_prediction_panel",
    "save_panel_png",
    "select_publication_results",
]
