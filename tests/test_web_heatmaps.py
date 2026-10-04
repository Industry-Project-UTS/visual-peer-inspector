"""Spatial and perceptual checks for display evidence, without neural inference."""
import numpy as np
from PIL import Image
import pytest

from visual_peer_inspector.web.heatmaps import heatmap_overlay


def test_low_and_high_mismatch_are_distinguishable_on_yellow_object():
    image = Image.new("RGB", (12, 10), (215, 195, 30))
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:10] = True
    values = np.full((10, 12), np.nan)
    values[mask] = np.tile(np.linspace(0, 1, 8), 6)
    original_values = values.copy()
    result = np.asarray(heatmap_overlay(image, values, mask))
    assert np.array_equal(result[~mask], np.asarray(image)[~mask])
    low, high = result[4, 2], result[4, 9]
    assert low[2] > low[0] and high[0] > high[2]
    assert np.linalg.norm(low.astype(float) - high) > 150
    np.testing.assert_equal(values, original_values)


def test_zero_mismatch_is_not_painted_as_high_evidence():
    image = Image.new("RGB", (5, 3), "white")
    result = heatmap_overlay(image, np.zeros((3, 5)), np.ones((3, 5), bool))
    red, green, blue = result.getpixel((2, 1))
    assert blue > red and blue > green


def test_invalid_values_and_missing_foreground_leave_photo_unchanged():
    image = Image.new("RGB", (5, 3), "yellow")
    for values, mask in [(np.full((3, 5), np.nan), np.ones((3, 5), bool)),
                         (np.ones((3, 5)), np.zeros((3, 5), bool))]:
        assert heatmap_overlay(image, values, mask).tobytes() == image.tobytes()


def test_incompatible_coordinates_are_rejected():
    with pytest.raises(ValueError, match="coordinates"):
        heatmap_overlay(Image.new("RGB", (5, 3)), np.zeros((5, 3)), np.ones((5, 3), bool))
