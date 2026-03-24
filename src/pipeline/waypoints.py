"""Uniform waypoint sampling along a contour."""

from __future__ import annotations

import numpy as np


def compute_arc_lengths(contour: np.ndarray) -> np.ndarray:
    """Compute cumulative arc lengths along a contour.

    Args:
        contour: (N, 2) array of points.

    Returns:
        (N,) array of cumulative arc lengths, starting at 0.
    """
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def sample_uniform_waypoints(
    contour: np.ndarray,
    num_points: int = 65,
    close_loop: bool = True,
) -> np.ndarray:
    """Sample uniformly-spaced waypoints along a contour by arc length.

    Args:
        contour: (N, 2) array of (lat, lon) or any 2D coordinates.
        num_points: Number of waypoints to sample.
        close_loop: If True, ensure the last waypoint matches the first.

    Returns:
        (num_points, 2) array of sampled waypoints.
    """
    if close_loop:
        # Close the contour if not already closed
        if not np.allclose(contour[0], contour[-1], atol=1e-8):
            contour = np.vstack([contour, contour[0:1]])

    arc_lengths = compute_arc_lengths(contour)
    total_length = arc_lengths[-1]

    if total_length == 0:
        # Degenerate contour — return repeated point
        return np.tile(contour[0], (num_points, 1))

    # Target arc lengths for uniform sampling
    if close_loop:
        target_lengths = np.linspace(0, total_length, num_points, endpoint=False)
    else:
        target_lengths = np.linspace(0, total_length, num_points)

    # Interpolate points at target arc lengths
    waypoints = np.zeros((num_points, 2))
    for i, target in enumerate(target_lengths):
        # Find the segment containing this arc length
        idx = np.searchsorted(arc_lengths, target, side="right") - 1
        idx = np.clip(idx, 0, len(contour) - 2)

        # Interpolate within the segment
        seg_start = arc_lengths[idx]
        seg_end = arc_lengths[idx + 1]
        seg_len = seg_end - seg_start

        if seg_len == 0:
            t = 0.0
        else:
            t = (target - seg_start) / seg_len

        waypoints[i] = contour[idx] + t * (contour[idx + 1] - contour[idx])

    return waypoints
