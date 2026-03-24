"""Tests for edge detection and contour extraction."""

import numpy as np
from PIL import Image

from src.pipeline.edge_detect import (
    image_to_edges,
    extract_largest_contour,
    normalize_contour,
    process_image,
)


def _make_circle_image(size: int = 256, radius: int = 80) -> Image.Image:
    """Create a test image with a white circle on black background."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    img[mask] = 255
    return Image.fromarray(img)


def test_image_to_edges():
    img = _make_circle_image()
    edges = image_to_edges(img)
    assert edges.shape == (256, 256)
    assert edges.dtype == np.uint8
    # Should have some edge pixels
    assert edges.sum() > 0


def test_extract_largest_contour():
    img = _make_circle_image()
    edges = image_to_edges(img)
    contour = extract_largest_contour(edges)
    assert contour is not None
    assert contour.ndim == 2
    assert contour.shape[1] == 2
    assert len(contour) > 10  # A circle should have many contour points


def test_extract_largest_contour_empty():
    edges = np.zeros((100, 100), dtype=np.uint8)
    contour = extract_largest_contour(edges)
    assert contour is None


def test_normalize_contour():
    contour = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
    normalized = normalize_contour(contour)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    np.testing.assert_allclose(normalized[0], [0.0, 0.0])
    np.testing.assert_allclose(normalized[-1], [1.0, 1.0])


def test_process_image():
    img = _make_circle_image()
    edges, contour = process_image(img)
    assert edges is not None
    assert contour is not None
    assert contour.min() >= 0.0
    assert contour.max() <= 1.0
