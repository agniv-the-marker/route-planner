"""GPX file creation and parsing utilities."""

from __future__ import annotations

from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np


def route_to_gpx(
    route: list[tuple[float, float]],
    name: str = "Route Sculptor Route",
    description: str = "",
) -> gpxpy.gpx.GPX:
    """Convert a list of (lat, lon) points to a GPX object.

    Args:
        route: List of (lat, lon) tuples.
        name: Name for the GPX track.
        description: Optional description.

    Returns:
        A gpxpy GPX object.
    """
    gpx = gpxpy.gpx.GPX()
    gpx.name = name
    gpx.description = description

    track = gpxpy.gpx.GPXTrack(name=name)
    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for lat, lon in route:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    return gpx


def save_gpx(gpx: gpxpy.gpx.GPX, filepath: str | Path) -> None:
    """Save a GPX object to a file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(gpx.to_xml())


def load_gpx(filepath: str | Path) -> gpxpy.gpx.GPX:
    """Load a GPX file and return a GPX object."""
    with open(filepath) as f:
        return gpxpy.parse(f)


def gpx_to_points(gpx: gpxpy.gpx.GPX) -> np.ndarray:
    """Extract all track points from a GPX object as (N, 2) array of (lat, lon)."""
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append((point.latitude, point.longitude))
    if not points:
        return np.empty((0, 2))
    return np.array(points)
