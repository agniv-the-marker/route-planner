"""Waypoint sampling along a contour (uniform and curvature-adaptive)."""

from __future__ import annotations

import math

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


def _compute_curvature(contour: np.ndarray) -> np.ndarray:
    """Estimate discrete curvature at each interior point of a contour.

    Uses the angle between successive edge vectors. Returns an array of
    length ``len(contour)`` with curvature values >= 0 (radians).  The
    first and last points get zero curvature.
    """
    curvature = np.zeros(len(contour))
    for i in range(1, len(contour) - 1):
        v1 = contour[i] - contour[i - 1]
        v2 = contour[i + 1] - contour[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        curvature[i] = np.arccos(cos_angle)  # 0 = straight, pi = reversal
    return curvature


def sample_adaptive_waypoints(
    contour: np.ndarray,
    num_points: int = 65,
    close_loop: bool = True,
    curvature_weight: float = 2.0,
) -> np.ndarray:
    """Sample waypoints with more density near high-curvature regions.

    Instead of spacing points uniformly by arc length, this function
    builds a density function that is higher near sharp corners (star
    tips, inner vertices) and lower along straight edges.  This keeps
    the total waypoint budget the same but preserves sharp features.

    Args:
        contour: (N, 2) array of points (geo-coordinates or normalised).
        num_points: Total waypoints to emit.
        close_loop: Whether the route should form a closed loop.
        curvature_weight: How strongly curvature biases the density.
            0 = fully uniform, higher = more points at corners.

    Returns:
        (num_points, 2) array of sampled waypoints.
    """
    if close_loop:
        if not np.allclose(contour[0], contour[-1], atol=1e-8):
            contour = np.vstack([contour, contour[0:1]])

    arc_lengths = compute_arc_lengths(contour)
    total_length = arc_lengths[-1]

    if total_length == 0:
        return np.tile(contour[0], (num_points, 1))

    # Compute curvature at each contour vertex
    curvature = _compute_curvature(contour)

    # Build a per-segment density: base (uniform) + curvature boost.
    # Assign each segment the average curvature of its two endpoints.
    n_seg = len(contour) - 1
    seg_lengths = np.diff(arc_lengths)
    seg_curvature = (curvature[:-1] + curvature[1:]) / 2.0

    # density = base_uniform + curvature_weight * curvature_component
    # We want the integral of density over arc length = 1
    base = np.ones(n_seg)  # uniform component
    curv_component = seg_curvature / (seg_curvature.sum() + 1e-12) * n_seg

    density_per_seg = base + curvature_weight * curv_component
    # Weighted arc-length: how much "budget" each segment consumes
    weighted_lengths = seg_lengths * density_per_seg
    cum_weighted = np.concatenate([[0.0], np.cumsum(weighted_lengths)])
    total_weighted = cum_weighted[-1]

    if total_weighted == 0:
        return sample_uniform_waypoints(contour, num_points, close_loop=False)

    # Sample uniformly in the *weighted* arc-length space
    if close_loop:
        targets = np.linspace(0, total_weighted, num_points, endpoint=False)
    else:
        targets = np.linspace(0, total_weighted, num_points)

    waypoints = np.zeros((num_points, 2))
    for i, target in enumerate(targets):
        idx = np.searchsorted(cum_weighted, target, side="right") - 1
        idx = np.clip(idx, 0, n_seg - 1)

        seg_start_w = cum_weighted[idx]
        seg_end_w = cum_weighted[idx + 1]
        seg_w = seg_end_w - seg_start_w

        if seg_w == 0:
            t = 0.0
        else:
            t = (target - seg_start_w) / seg_w

        waypoints[i] = contour[idx] + t * (contour[idx + 1] - contour[idx])

    return waypoints


def densify_waypoints(
    waypoints: np.ndarray,
    max_gap_km: float = 0.2,
) -> np.ndarray:
    """Insert intermediate waypoints so no two consecutive points are farther
    than *max_gap_km* apart.

    Long gaps between waypoints force the road router to take multi-block
    detours that zig-zag across the street grid.  Keeping gaps short
    (~200 m) means the router only needs one or two turns per segment,
    producing a path that looks like something a person would actually ride.

    Args:
        waypoints: (N, 2) array of (lat, lon) waypoints.
        max_gap_km: Maximum allowed gap in kilometres.

    Returns:
        Densified (M, 2) array with M >= N.
    """
    if len(waypoints) < 2:
        return waypoints

    DEG_TO_KM_LAT = 111.0

    result = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]

        dlat = (p2[0] - p1[0]) * DEG_TO_KM_LAT
        avg_lat = (p1[0] + p2[0]) / 2.0
        dlon = (p2[1] - p1[1]) * DEG_TO_KM_LAT * math.cos(math.radians(avg_lat))
        dist_km = math.sqrt(dlat ** 2 + dlon ** 2)

        if dist_km > max_gap_km:
            n_subdivisions = math.ceil(dist_km / max_gap_km)
            for j in range(1, n_subdivisions):
                t = j / n_subdivisions
                result.append(p1 + t * (p2 - p1))

        result.append(p2)

    return np.array(result)
