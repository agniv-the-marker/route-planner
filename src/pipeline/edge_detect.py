"""Canny edge detection and contour extraction from generated images."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class ContourSet:
    """A set of contours with outer boundary and inner detail features."""

    outer: np.ndarray  # (N, 2) — the main outer boundary
    inner: list[np.ndarray] = field(default_factory=list)  # list of (M_i, 2)
    inner_areas: list[float] = field(default_factory=list)  # area of each inner contour


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


def extract_silhouette_contour(
    image: Image.Image,
    threshold: int = 128,
    blur_kernel: int = 5,
    min_area_ratio: float = 0.005,
) -> np.ndarray | None:
    """Extract the contour of the dark silhouette region via thresholding.

    This is more robust than Canny-based contour extraction because it
    finds the actual shape boundary rather than edge fragments.

    Args:
        image: Input PIL image (expected: dark shape on light background).
        threshold: Grayscale threshold. Pixels below this are "shape".
        blur_kernel: Gaussian blur before thresholding (reduces noise).
        min_area_ratio: Minimum contour area as a fraction of image area.

    Returns:
        (N, 2) array of contour points in pixel coordinates, or None.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_h, img_w = gray.shape

    # Use Otsu's threshold for robustness against gray backgrounds
    otsu_thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if otsu_thresh < 30 or otsu_thresh > 230:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Light morphological close to fill small gaps (e.g. anti-aliasing holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Filter out tiny contours AND full-image boundary boxes
    img_area = img_h * img_w
    min_area = img_area * min_area_ratio
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if (w > img_w * 0.9) and (h > img_h * 0.9):
            continue  # skip full-image boundary box
        if area > img_area * 0.85:
            continue  # skip contours covering nearly all pixels
        valid.append(c)
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def extract_contour_set(
    image: Image.Image,
    threshold: int = 128,
    min_inner_area_ratio: float = 0.005,
    max_inner_contours: int = 5,
) -> ContourSet | None:
    """Extract outer contour + significant inner detail contours.

    Uses RETR_TREE hierarchy to find contours nested inside the main
    silhouette boundary (e.g., a dog's eye, house windows).

    Args:
        image: Input PIL image (dark shape on light background).
        threshold: Grayscale threshold for silhouette detection.
        min_inner_area_ratio: Minimum inner contour area as fraction of outer area.
        max_inner_contours: Maximum number of inner contours to keep.

    Returns:
        ContourSet with outer + inner contours in pixel coords, or None.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Use Otsu's method to auto-detect threshold (handles gray/noisy backgrounds)
    # Falls back to fixed threshold if Otsu fails
    otsu_thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # If Otsu gives a weird threshold (too low or too high), use the fixed one
    if otsu_thresh < 30 or otsu_thresh > 230:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Reject contours that span nearly the full image (background artifacts)
    img_h, img_w = gray.shape

    # RETR_TREE gives full hierarchy: [next, prev, first_child, parent]
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]  # shape (N, 4)

    # Find the largest contour that isn't a full-image boundary box
    areas = [cv2.contourArea(c) for c in contours]
    if max(areas) < 100:  # no meaningful contour
        return None

    # Filter out contours that span >90% of the image (background artifacts)
    img_area = img_h * img_w
    valid_indices = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        spans_full = (w > img_w * 0.9) and (h > img_h * 0.9)
        too_big = areas[i] > img_area * 0.85
        if not spans_full and not too_big:
            valid_indices.append(i)

    # If all contours were filtered, use the largest anyway as fallback
    if not valid_indices:
        valid_indices = list(range(len(contours)))

    outer_idx = max(valid_indices, key=lambda i: areas[i])
    outer_contour = contours[outer_idx].reshape(-1, 2)
    outer_area = areas[outer_idx]

    # Find direct children of the outer contour in the hierarchy
    min_inner_area = outer_area * min_inner_area_ratio
    inner_contours = []
    inner_areas = []

    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent == outer_idx and areas[i] > min_inner_area:
            inner_contours.append(contours[i].reshape(-1, 2))
            inner_areas.append(areas[i])

    # Sort by area descending and cap
    if inner_contours:
        sorted_pairs = sorted(
            zip(inner_areas, inner_contours), key=lambda x: x[0], reverse=True
        )
        inner_areas = [a for a, _ in sorted_pairs[:max_inner_contours]]
        inner_contours = [c for _, c in sorted_pairs[:max_inner_contours]]

    return ContourSet(
        outer=outer_contour,
        inner=inner_contours,
        inner_areas=inner_areas,
    )


def normalize_contour_set(contour_set: ContourSet) -> ContourSet:
    """Normalize all contours using the outer contour's bounding box.

    This preserves spatial relationships between outer and inner contours.
    """
    outer = contour_set.outer.astype(np.float64)
    mins = outer.min(axis=0)
    maxs = outer.max(axis=0)
    span = maxs - mins
    span = np.where(span == 0, 1.0, span)

    norm_outer = (outer - mins) / span
    norm_inner = [(c.astype(np.float64) - mins) / span for c in contour_set.inner]

    return ContourSet(
        outer=norm_outer,
        inner=norm_inner,
        inner_areas=contour_set.inner_areas,
    )


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
    multi_contour: bool = False,
    min_inner_area_ratio: float = 0.005,
    max_inner_contours: int = 5,
) -> tuple[np.ndarray, np.ndarray | ContourSet | None]:
    """Full pipeline: image → edges + contour(s) (normalized).

    Args:
        multi_contour: If True, returns a ContourSet with outer + inner contours.
            If False, returns a single normalized contour (backward compatible).

    Returns:
        (edges, contour_data) — edges is the binary Canny edge map.
        contour_data is either:
          - np.ndarray (N, 2) in [0,1]x[0,1] when multi_contour=False
          - ContourSet with normalized contours when multi_contour=True
          - None if no contour found
    """
    edges = image_to_edges(image, canny_low, canny_high, blur_kernel)

    if multi_contour:
        cs = extract_contour_set(
            image,
            min_inner_area_ratio=min_inner_area_ratio,
            max_inner_contours=max_inner_contours,
        )
        if cs is not None:
            return edges, normalize_contour_set(cs)
        # Fallback: try single contour, wrap in ContourSet
        contour = extract_silhouette_contour(image)
        if contour is None:
            contour = extract_largest_contour(edges)
        if contour is None:
            return edges, None
        return edges, ContourSet(outer=normalize_contour(contour))

    # Single contour mode (backward compatible)
    contour = extract_silhouette_contour(image, blur_kernel=blur_kernel)
    if contour is None:
        contour = extract_largest_contour(edges)
    if contour is None:
        return edges, None
    return edges, normalize_contour(contour)
