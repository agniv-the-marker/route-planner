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


def allocate_waypoint_budget(
    outer_perimeter: float,
    inner_perimeters: list[float],
    total_budget: int = 80,
    outer_budget_min: float = 0.6,
    min_inner_waypoints: int = 4,
) -> tuple[int, list[int]]:
    """Allocate waypoint budget across outer and inner contours.

    Returns (outer_count, [inner_count_1, inner_count_2, ...]).
    """
    if not inner_perimeters:
        return total_budget, []

    total_perim = outer_perimeter + sum(inner_perimeters)
    if total_perim == 0:
        return total_budget, [min_inner_waypoints] * len(inner_perimeters)

    # Proportional allocation
    outer_prop = int(total_budget * outer_perimeter / total_perim)
    outer_count = max(int(total_budget * outer_budget_min), outer_prop)

    remaining = total_budget - outer_count
    inner_counts = []
    for p in inner_perimeters:
        prop = max(min_inner_waypoints, int(remaining * p / max(sum(inner_perimeters), 1e-9)))
        inner_counts.append(prop)

    # Scale down if over budget
    inner_total = sum(inner_counts)
    if inner_total > remaining and inner_total > 0:
        scale = remaining / inner_total
        inner_counts = [max(min_inner_waypoints, int(c * scale)) for c in inner_counts]

    return outer_count, inner_counts


def build_bridged_path(
    outer_waypoints: np.ndarray,
    inner_waypoint_lists: list[np.ndarray],
) -> np.ndarray:
    """Stitch outer + inner contour waypoints into a single connected path.

    For each inner contour, finds the closest point on the outer contour,
    inserts a bridge (out to inner, trace inner loop, bridge back), then
    continues along the outer contour.

    Args:
        outer_waypoints: (N, 2) waypoints along the outer contour (closed loop).
        inner_waypoint_lists: List of (M_i, 2) waypoints for each inner contour.

    Returns:
        (P, 2) single connected path array.
    """
    if not inner_waypoint_lists:
        return outer_waypoints

    from scipy.spatial import cKDTree

    n_outer = len(outer_waypoints)

    # For each inner contour, find the closest outer waypoint
    outer_tree = cKDTree(outer_waypoints)
    bridges = []  # (outer_idx, inner_waypoints, inner_entry_idx)

    for inner_wps in inner_waypoint_lists:
        if len(inner_wps) < 2:
            continue
        # Find closest pair
        dists, outer_indices = outer_tree.query(inner_wps)
        best_inner_idx = int(np.argmin(dists))
        best_outer_idx = int(outer_indices[best_inner_idx])
        bridges.append((best_outer_idx, inner_wps, best_inner_idx))

    if not bridges:
        return outer_waypoints

    # Sort bridges by outer index (order of encounter along outer boundary)
    bridges.sort(key=lambda b: b[0])

    # Build the unified path
    path = []
    outer_pos = 0

    for outer_idx, inner_wps, inner_entry_idx in bridges:
        # Add outer waypoints up to the bridge point
        if outer_idx >= outer_pos:
            path.extend(outer_waypoints[outer_pos:outer_idx + 1].tolist())
        outer_pos = outer_idx + 1

        # Bridge to inner contour
        n_inner = len(inner_wps)
        # Trace inner contour starting from entry point, full loop
        for j in range(n_inner):
            idx = (inner_entry_idx + j) % n_inner
            path.append(inner_wps[idx].tolist())
        # Close the inner loop back to entry
        path.append(inner_wps[inner_entry_idx].tolist())

        # Bridge back to outer contour
        path.append(outer_waypoints[min(outer_idx, n_outer - 1)].tolist())

    # Add remaining outer waypoints
    if outer_pos < n_outer:
        path.extend(outer_waypoints[outer_pos:].tolist())

    return np.array(path)


def sample_multi_contour_waypoints(
    contour_set: "ContourSet",
    num_points: int = 80,
    curvature_weight: float = 2.0,
    outer_budget_min: float = 0.6,
    min_inner_waypoints: int = 4,
    max_gap_km: float = 0.15,
) -> np.ndarray:
    """Sample waypoints from a ContourSet using bridge-and-trace.

    Traces the outer boundary with detours into inner features,
    producing a single connected waypoint path.
    """
    outer = contour_set.outer
    inner_contours = contour_set.inner

    # Compute perimeters for budget allocation
    outer_perim = float(compute_arc_lengths(outer)[-1]) if len(outer) > 1 else 0
    inner_perims = [
        float(compute_arc_lengths(c)[-1]) if len(c) > 1 else 0
        for c in inner_contours
    ]

    outer_budget, inner_budgets = allocate_waypoint_budget(
        outer_perim, inner_perims,
        total_budget=num_points,
        outer_budget_min=outer_budget_min,
        min_inner_waypoints=min_inner_waypoints,
    )

    # Sample each contour independently
    outer_wps = sample_adaptive_waypoints(
        outer, num_points=outer_budget, curvature_weight=curvature_weight, close_loop=True
    )

    inner_wps_list = []
    for c, budget in zip(inner_contours, inner_budgets):
        if len(c) < 3:
            continue
        wps = sample_adaptive_waypoints(
            c, num_points=budget, curvature_weight=curvature_weight, close_loop=True
        )
        inner_wps_list.append(wps)

    # Stitch into single path
    bridged = build_bridged_path(outer_wps, inner_wps_list)

    # Densify the final path
    bridged = densify_waypoints(bridged, max_gap_km=max_gap_km)

    return bridged
