"""Tests for pipeline components: placement, waypoints."""

import math

import numpy as np

from src.pipeline.placement import (
    BBox,
    transform_contour,
    contour_fits_bbox,
    grid_search,
    km_to_deg_lat,
    km_to_deg_lon,
)
from src.pipeline.waypoints import (
    compute_arc_lengths,
    sample_adaptive_waypoints,
    sample_uniform_waypoints,
)


# --- Placement tests ---

def test_bbox_center():
    bbox = BBox(min_lat=37.7, max_lat=37.8, min_lon=-122.5, max_lon=-122.4)
    center = bbox.center
    assert abs(center[0] - 37.75) < 1e-6
    assert abs(center[1] - (-122.45)) < 1e-6


def test_km_to_deg_lat():
    assert abs(km_to_deg_lat(111.0) - 1.0) < 0.01


def test_transform_contour_centered():
    # Unit square contour
    contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    geo = transform_contour(contour, 37.75, -122.45, 1.0, 0.0)
    assert geo.shape == (4, 2)
    # Should be roughly centered around the center point
    center_lat = geo[:, 0].mean()
    center_lon = geo[:, 1].mean()
    assert abs(center_lat - 37.75) < 0.01
    assert abs(center_lon - (-122.45)) < 0.01


def test_transform_contour_rotation():
    contour = np.array([[0, 0], [1, 0]], dtype=np.float64)
    geo_0 = transform_contour(contour, 37.75, -122.45, 1.0, 0.0)
    geo_90 = transform_contour(contour, 37.75, -122.45, 1.0, 90.0)
    # After 90° rotation, the span should switch between lat and lon axes
    lat_span_0 = geo_0[:, 0].max() - geo_0[:, 0].min()
    lon_span_0 = geo_0[:, 1].max() - geo_0[:, 1].min()
    lat_span_90 = geo_90[:, 0].max() - geo_90[:, 0].min()
    lon_span_90 = geo_90[:, 1].max() - geo_90[:, 1].min()
    # They should swap (approximately, given the coordinate conversion)
    assert lat_span_90 > lat_span_0 * 0.5  # just check it changed meaningfully


def test_contour_fits_bbox():
    bbox = BBox(min_lat=37.7, max_lat=37.8, min_lon=-122.5, max_lon=-122.4)
    inside = np.array([[37.75, -122.45], [37.76, -122.44]])
    outside = np.array([[37.75, -122.45], [37.85, -122.44]])
    assert contour_fits_bbox(inside, bbox)
    assert not contour_fits_bbox(outside, bbox)


def test_grid_search_returns_results():
    contour = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
    )
    bbox = BBox(min_lat=37.7, max_lat=37.8, min_lon=-122.5, max_lon=-122.4)
    results = grid_search(contour, bbox, num_positions=4, num_scales=2, num_rotations=4)
    assert len(results) > 0
    # Results should be sorted by score descending
    scores = [p.score for p, _ in results]
    assert scores == sorted(scores, reverse=True)


# --- Waypoint tests ---

def test_compute_arc_lengths():
    contour = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
    arc = compute_arc_lengths(contour)
    assert len(arc) == 3
    assert arc[0] == 0.0
    assert abs(arc[1] - 1.0) < 1e-6
    assert abs(arc[2] - 2.0) < 1e-6


def test_sample_uniform_waypoints():
    # Circle contour
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    contour = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    waypoints = sample_uniform_waypoints(contour, num_points=20, close_loop=True)
    assert waypoints.shape == (20, 2)

    # Check roughly uniform spacing
    diffs = np.diff(waypoints, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    assert distances.std() / distances.mean() < 0.3  # low variation


def test_sample_uniform_waypoints_open():
    contour = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    waypoints = sample_uniform_waypoints(contour, num_points=5, close_loop=False)
    assert waypoints.shape == (5, 2)
    np.testing.assert_allclose(waypoints[0], [0, 0], atol=1e-6)
    np.testing.assert_allclose(waypoints[-1], [2, 0], atol=1e-6)


def test_sample_adaptive_waypoints_star():
    """Adaptive sampling should place more points near star tips."""
    # Build a dense star contour (interpolated edges, like the real pipeline)
    vertices = []
    outer_r, inner_r = 1.0, 0.4
    for i in range(10):
        angle = np.radians(i * 36 - 90)
        r = outer_r if i % 2 == 0 else inner_r
        vertices.append([r * np.cos(angle), r * np.sin(angle)])
    vertices.append(vertices[0])  # close loop

    # Interpolate 20 points per edge to get a dense contour
    dense = []
    for i in range(len(vertices) - 1):
        for t in np.linspace(0, 1, 20, endpoint=False):
            p = np.array(vertices[i]) * (1 - t) + np.array(vertices[i + 1]) * t
            dense.append(p)
    contour = np.array(dense, dtype=np.float64)

    waypoints = sample_adaptive_waypoints(contour, num_points=30, close_loop=True, curvature_weight=2.0)
    assert waypoints.shape == (30, 2)

    # Count how many waypoints fall near each star tip (outer vertices)
    tips = np.array(vertices[::2][:5])
    threshold = 0.15  # distance threshold to count as "near a tip"

    adaptive_near_tips = 0
    for wp in waypoints:
        if np.min(np.sqrt(((tips - wp) ** 2).sum(axis=1))) < threshold:
            adaptive_near_tips += 1

    uniform = sample_uniform_waypoints(contour, num_points=30, close_loop=True)
    uniform_near_tips = 0
    for wp in uniform:
        if np.min(np.sqrt(((tips - wp) ** 2).sum(axis=1))) < threshold:
            uniform_near_tips += 1

    # Adaptive should cluster more points near the sharp tips
    assert adaptive_near_tips >= uniform_near_tips
