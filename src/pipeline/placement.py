"""Grid search for placing an outline contour on a geographic bounding box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BBox:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2,
        )

    @property
    def lat_span(self) -> float:
        return self.max_lat - self.min_lat

    @property
    def lon_span(self) -> float:
        return self.max_lon - self.min_lon


@dataclass
class Placement:
    """A specific placement of an outline on the map."""
    center_lat: float
    center_lon: float
    scale_km: float
    rotation_deg: float
    score: float = 0.0  # road density or heuristic score


def km_to_deg_lat(km: float) -> float:
    """Convert kilometers to degrees latitude (approximate)."""
    return km / 111.0


def km_to_deg_lon(km: float, lat: float) -> float:
    """Convert kilometers to degrees longitude at a given latitude."""
    return km / (111.0 * math.cos(math.radians(lat)))


def transform_contour(
    contour: np.ndarray,
    center_lat: float,
    center_lon: float,
    scale_km: float,
    rotation_deg: float,
) -> np.ndarray:
    """Transform a normalized [0,1]x[0,1] contour to geo-coordinates.

    Args:
        contour: (N, 2) normalized contour, columns are (x, y) in [0, 1].
        center_lat: Center latitude for placement.
        center_lon: Center longitude for placement.
        scale_km: Size of the outline footprint in km.
        rotation_deg: Rotation angle in degrees.

    Returns:
        (N, 2) array of (lat, lon) coordinates.
    """
    # Center the contour around origin
    centered = contour - 0.5  # now in [-0.5, 0.5]

    # Apply rotation
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = centered @ rotation_matrix.T

    # Scale to geographic coordinates
    dlat = km_to_deg_lat(scale_km)
    dlon = km_to_deg_lon(scale_km, center_lat)

    geo = np.zeros_like(rotated)
    geo[:, 0] = center_lat + rotated[:, 1] * dlat  # y -> lat
    geo[:, 1] = center_lon + rotated[:, 0] * dlon  # x -> lon

    return geo


def contour_fits_bbox(geo_contour: np.ndarray, bbox: BBox) -> bool:
    """Check if a geo-transformed contour fits within the bounding box."""
    lats, lons = geo_contour[:, 0], geo_contour[:, 1]
    return (
        lats.min() >= bbox.min_lat
        and lats.max() <= bbox.max_lat
        and lons.min() >= bbox.min_lon
        and lons.max() <= bbox.max_lon
    )


def grid_search(
    contour: np.ndarray,
    bbox: BBox,
    num_positions: int = 10,
    num_scales: int = 5,
    num_rotations: int = 8,
    scale_range_km: tuple[float, float] = (0.3, 5.0),
    max_rotation_deg: float = 90.0,
) -> list[tuple[Placement, np.ndarray]]:
    """Search over position/scale/rotation for best outline placements.

    Args:
        contour: (N, 2) normalized contour in [0, 1]x[0, 1].
        bbox: Geographic bounding box to place the outline within.
        num_positions: Number of position grid points (sqrt determines grid density).
        num_scales: Number of scale values to try.
        num_rotations: Number of rotation angles to try.
        scale_range_km: (min_km, max_km) for outline footprint.
        max_rotation_deg: Maximum rotation in either direction (default ±90°).
            Set to 180 for full rotation, 90 to prevent upside-down shapes.

    Returns:
        List of (Placement, geo_contour) tuples, sorted by score (descending).
        Only placements that fit within the bbox are included.
    """
    # Generate grid of center positions within bbox (with margin)
    grid_side = max(2, int(math.sqrt(num_positions)))
    margin_lat = bbox.lat_span * 0.15
    margin_lon = bbox.lon_span * 0.15
    lats = np.linspace(
        bbox.min_lat + margin_lat, bbox.max_lat - margin_lat, grid_side
    )
    lons = np.linspace(
        bbox.min_lon + margin_lon, bbox.max_lon - margin_lon, grid_side
    )

    scales = np.linspace(scale_range_km[0], scale_range_km[1], num_scales)
    # Limit rotation range to avoid upside-down shapes
    rotations = np.linspace(-max_rotation_deg, max_rotation_deg, num_rotations)

    results: list[tuple[Placement, np.ndarray]] = []

    for lat in lats:
        for lon in lons:
            for scale in scales:
                for rot in rotations:
                    geo_contour = transform_contour(contour, lat, lon, scale, rot)

                    if not contour_fits_bbox(geo_contour, bbox):
                        continue

                    # Score by how well-centered and sized the placement is.
                    # Better heuristic: prefer placements that use more of the bbox
                    # and are centered (simple proxy for road density).
                    lats_col = geo_contour[:, 0]
                    lons_col = geo_contour[:, 1]
                    lat_range = (lats_col.max() - lats_col.min()) / bbox.lat_span
                    lon_range = (lons_col.max() - lons_col.min()) / bbox.lon_span
                    coverage = (lat_range + lon_range) / 2

                    # Prefer centered placements
                    cx, cy = bbox.center
                    dist_to_center = math.sqrt(
                        ((lat - cx) / bbox.lat_span) ** 2
                        + ((lon - cy) / bbox.lon_span) ** 2
                    )
                    centrality = 1.0 / (1.0 + dist_to_center)

                    score = 0.6 * coverage + 0.4 * centrality

                    placement = Placement(
                        center_lat=lat,
                        center_lon=lon,
                        scale_km=scale,
                        rotation_deg=rot,
                        score=score,
                    )
                    results.append((placement, geo_contour))

    results.sort(key=lambda x: x[0].score, reverse=True)
    return results
