"""Canny edge detection and contour extraction from generated images."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def image_to_edges(
    image: Image.Image,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Apply Canny edge detection to a PIL image.

    Returns a binary edge map (H, W) with values 0 or 255.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    if blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    return edges


def extract_largest_contour(edges: np.ndarray) -> np.ndarray | None:
    """Extract the largest contour from a binary edge map.

    Returns an (N, 2) array of contour points in pixel coordinates,
    or None if no contours are found.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    # Reshape from (N, 1, 2) to (N, 2)
    return largest.reshape(-1, 2)


def normalize_contour(contour: np.ndarray) -> np.ndarray:
    """Normalize contour points to [0, 1] x [0, 1] unit coordinates.

    Args:
        contour: (N, 2) array of pixel coordinates (x, y).

    Returns:
        (N, 2) array of normalized coordinates.
    """
    contour = contour.astype(np.float64)
    mins = contour.min(axis=0)
    maxs = contour.max(axis=0)
    span = maxs - mins
    # Avoid division by zero
    span = np.where(span == 0, 1.0, span)
    return (contour - mins) / span


def process_image(
    image: Image.Image,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Full pipeline: image → edges → largest contour (normalized).

    Returns:
        (edges, normalized_contour) — edges is the binary edge map,
        normalized_contour is (N, 2) in [0,1]x[0,1] or None.
    """
    edges = image_to_edges(image, canny_low, canny_high, blur_kernel)
    contour = extract_largest_contour(edges)
    if contour is None:
        return edges, None
    normalized = normalize_contour(contour)
    return edges, normalized
