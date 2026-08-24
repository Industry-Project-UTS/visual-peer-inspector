"""Small deterministic helpers for notebook tables and visual evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


BLUE = "#24557f"
LIGHT_BLUE = "#dceaf5"
GRID = "#c8d1da"
TEXT = "#1f2933"
WHITE = "#ffffff"


def write_csv_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a stable UTF-8 CSV with an explicit column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_vertical_axis_label(
    image: Image.Image,
    text: str,
    *,
    y_center: int,
    font: ImageFont.ImageFont,
) -> None:
    """Draw a standard vertical y-axis label without clipping the text."""

    bounds = font.getbbox(text)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    label = Image.new("RGBA", (width + 6, height + 6), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (3 - bounds[0], 3 - bounds[1]),
        text,
        fill=TEXT,
        font=font,
    )
    rotated = label.rotate(90, expand=True)
    image.paste(
        rotated,
        (14, round(y_center - rotated.height / 2)),
        rotated,
    )


def save_bar_chart(
    labels: Sequence[str],
    values: Sequence[int | float],
    path: Path,
    *,
    title: str,
    y_label: str,
    integer_ticks: bool = False,
    show_values: bool = False,
) -> None:
    """Save a compact vertical bar chart without a plotting dependency."""

    if not labels or len(labels) != len(values):
        raise ValueError("Bar chart labels and values must be non-empty and aligned")
    width, height = 1400, 760
    left, right, top, bottom = 110, 40, 95, 125
    chart_width = width - left - right
    chart_height = height - top - bottom
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    body_font = _font(18)
    small_font = _font(14)
    draw.text((left, 28), title, fill=TEXT, font=title_font)
    _draw_vertical_axis_label(
        image,
        y_label,
        y_center=top + chart_height // 2,
        font=body_font,
    )
    maximum = max(float(value) for value in values)
    if integer_ticks:
        axis_maximum = float(max(5, math.ceil(maximum / 5) * 5))
    else:
        axis_maximum = maximum if maximum > 0 else 1.0
    for tick in range(6):
        ratio = tick / 5
        y = top + chart_height - round(ratio * chart_height)
        draw.line((left, y, width - right, y), fill=GRID, width=1)
        tick_value = axis_maximum * ratio
        label = str(round(tick_value)) if integer_ticks else (
            f"{tick_value:.1f}".rstrip("0").rstrip(".")
        )
        draw.text((left - 62, y - 9), label, fill=TEXT, font=small_font)
    slot = chart_width / len(values)
    bar_width = max(8, round(slot * 0.68))
    for index, (label, value) in enumerate(zip(labels, values)):
        x_center = left + (index + 0.5) * slot
        bar_height = round(float(value) / axis_maximum * chart_height)
        x1 = round(x_center - bar_width / 2)
        y1 = top + chart_height - bar_height
        x2 = round(x_center + bar_width / 2)
        draw.rounded_rectangle(
            (x1, y1, x2, top + chart_height),
            radius=4,
            fill=BLUE,
        )
        if show_values:
            value_label = (
                str(int(round(float(value))))
                if integer_ticks
                else f"{float(value):.3f}".rstrip("0").rstrip(".")
            )
            draw.text(
                (round(x_center), max(top + 4, y1 - 10)),
                value_label,
                fill=TEXT,
                font=small_font,
                anchor="ms",
            )
        draw.text(
            (round(x_center), top + chart_height + 14),
            str(label),
            fill=TEXT,
            font=small_font,
            anchor="ma",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def save_histogram(
    values: Sequence[int | float],
    path: Path,
    *,
    title: str,
    x_label: str,
    bins: int = 12,
    count_label: str = "Count",
) -> None:
    """Save a histogram with stable bin boundaries."""

    if not values:
        raise ValueError("Histogram values cannot be empty")
    numeric = [float(value) for value in values]
    lower = min(numeric)
    upper = max(numeric)
    if lower == upper:
        labels = [f"{lower:.1f}"]
        counts = [len(numeric)]
    else:
        bin_count = max(1, min(int(bins), len(numeric)))
        width = (upper - lower) / bin_count
        counts = [0] * bin_count
        for value in numeric:
            index = min(bin_count - 1, int((value - lower) / width))
            counts[index] += 1
        labels = [
            f"{lower + (index + 0.5) * width:.2f}"
            for index in range(bin_count)
        ]
    save_bar_chart(
        labels,
        counts,
        path,
        title=title,
        y_label=count_label,
        integer_ticks=True,
    )
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text(
        (image.width // 2, image.height - 36),
        x_label,
        fill=TEXT,
        font=_font(18),
        anchor="mm",
    )
    image.save(path, format="PNG")


def save_scatter_plot(
    x_values: Sequence[int | float],
    y_values: Sequence[int | float],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Save a deterministic scatter plot without a plotting dependency."""

    if not x_values or len(x_values) != len(y_values):
        raise ValueError("Scatter values must be non-empty and have equal length")
    xs = [float(value) for value in x_values]
    ys = [float(value) for value in y_values]
    left, top, right, bottom = 112, 88, 1360, 650
    image = Image.new("RGB", (1440, 760), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((72, 24), title, fill=TEXT, font=_font(28, bold=True))
    draw.line((left, bottom, right, bottom), fill=GRID, width=2)
    draw.line((left, top, left, bottom), fill=GRID, width=2)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    for step in range(6):
        fraction = step / 5
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text(
            (round(x), bottom + 14),
            f"{x_min + fraction * (x_max - x_min):.1f}",
            fill=TEXT,
            font=_font(14),
            anchor="ma",
        )
        draw.text(
            (left - 12, round(y)),
            f"{y_min + fraction * (y_max - y_min):.2f}",
            fill=TEXT,
            font=_font(14),
            anchor="rm",
        )
    for x_value, y_value in zip(xs, ys):
        x = left + (x_value - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=BLUE)
    draw.text(
        ((left + right) // 2, 724),
        x_label,
        fill=TEXT,
        font=_font(18),
        anchor="mm",
    )
    _draw_vertical_axis_label(
        image,
        y_label,
        y_center=(top + bottom) // 2,
        font=_font(18),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def save_image_grid(
    items: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
    columns: int = 5,
    cell_size: int = 220,
) -> None:
    """Save a labelled contact sheet, optionally with XYXY overlays."""

    if not items:
        raise ValueError("Image grid items cannot be empty")
    columns = max(1, int(columns))
    rows = math.ceil(len(items) / columns)
    label_height = 34
    title_height = 72
    margin = 18
    width = margin * 2 + columns * cell_size
    height = title_height + margin + rows * (cell_size + label_height)
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill=TEXT, font=_font(26, bold=True))
    for index, item in enumerate(items):
        source_path = Path(str(item["path"]))
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        original_width, original_height = source.size
        fitted = ImageOps.contain(source, (cell_size - 18, cell_size - 18))
        x = margin + (index % columns) * cell_size
        y = title_height + (index // columns) * (cell_size + label_height)
        paste_x = x + (cell_size - fitted.width) // 2
        paste_y = y + (cell_size - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        scale_x = fitted.width / original_width
        scale_y = fitted.height / original_height
        for box in item.get("boxes", []):
            x1, y1, x2, y2 = [int(value) for value in box]
            draw.rectangle(
                (
                    paste_x + round(x1 * scale_x),
                    paste_y + round(y1 * scale_y),
                    paste_x + round(x2 * scale_x),
                    paste_y + round(y2 * scale_y),
                ),
                outline="#d73a49",
                width=4,
            )
        draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline=GRID)
        draw.text(
            (x + cell_size // 2, y + cell_size + 8),
            str(item["label"]),
            fill=TEXT,
            font=_font(15),
            anchor="ma",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


__all__ = [
    "save_bar_chart",
    "save_histogram",
    "save_image_grid",
    "save_scatter_plot",
    "write_csv_rows",
]
