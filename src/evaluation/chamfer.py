"""Chamfer distance between binary edge images."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def extract_edge_points(edge_image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Extract nonzero pixel coordinates from a binary edge image.

    Args:
        edge_image: (H, W) array with edge pixels.
        threshold: Pixel value threshold for edge detection.

    Returns:
        (M, 2) array of (row, col) coordinates.
    """
    ys, xs = np.where(edge_image > threshold)
    if len(ys) == 0:
        return np.empty((0, 2))
    return np.stack([ys, xs], axis=1).astype(np.float64)


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute the symmetric Chamfer distance between two point sets.

    Chamfer = mean(min_dist(A→B)) + mean(min_dist(B→A)) / 2

    Args:
        points_a: (M, 2) array of points.
        points_b: (N, 2) array of points.

    Returns:
        Chamfer distance (lower is better, 0 is perfect).
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")

    tree_b = cKDTree(points_b)
    tree_a = cKDTree(points_a)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    return (dist_a_to_b.mean() + dist_b_to_a.mean()) / 2.0


def chamfer_score(
    reference_edges: np.ndarray,
    route_edges: np.ndarray,
    threshold: int = 128,
) -> float:
    """Compute normalized Chamfer score between two edge images.

    Args:
        reference_edges: (H, W) binary edge image from the generated silhouette.
        route_edges: (H, W) binary edge image from the rendered GPX route.
        threshold: Pixel threshold for edge extraction.

    Returns:
        Score in [0, 1] where 1 is perfect match.
        Computed as 1 / (1 + chamfer_distance).
    """
    pts_ref = extract_edge_points(reference_edges, threshold)
    pts_route = extract_edge_points(route_edges, threshold)

    dist = chamfer_distance(pts_ref, pts_route)

    if dist == float("inf"):
        return 0.0

    return 1.0 / (1.0 + dist)
