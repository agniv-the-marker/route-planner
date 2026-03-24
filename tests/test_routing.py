"""Tests for routing and GPX utilities (offline/unit tests)."""

import numpy as np
import tempfile
from pathlib import Path

from src.pipeline.gpx_utils import route_to_gpx, save_gpx, load_gpx, gpx_to_points


def test_route_to_gpx():
    route = [(37.78, -122.42), (37.79, -122.43), (37.80, -122.41)]
    gpx = route_to_gpx(route, name="Test Route")
    assert len(gpx.tracks) == 1
    assert len(gpx.tracks[0].segments) == 1
    assert len(gpx.tracks[0].segments[0].points) == 3


def test_save_and_load_gpx():
    route = [(37.78, -122.42), (37.79, -122.43)]
    gpx = route_to_gpx(route)

    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as f:
        filepath = f.name

    save_gpx(gpx, filepath)
    loaded = load_gpx(filepath)

    assert len(loaded.tracks) == 1
    points = loaded.tracks[0].segments[0].points
    assert len(points) == 2
    assert abs(points[0].latitude - 37.78) < 1e-6

    Path(filepath).unlink()


def test_gpx_to_points():
    route = [(37.78, -122.42), (37.79, -122.43), (37.80, -122.41)]
    gpx = route_to_gpx(route)
    points = gpx_to_points(gpx)
    assert points.shape == (3, 2)
    np.testing.assert_allclose(points[0], [37.78, -122.42], atol=1e-6)


def test_gpx_to_points_empty():
    import gpxpy.gpx

    gpx = gpxpy.gpx.GPX()
    points = gpx_to_points(gpx)
    assert points.shape == (0, 2)
