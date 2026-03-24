"""Render GPX routes as images for evaluation."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def render_polyline(
    route: list[tuple[float, float]],
    size: int = 512,
    line_width: int = 3,
    bg_color: int = 0,
    line_color: int = 255,
) -> Image.Image:
    """Render a route as a white polyline on a black background.

    Used for Chamfer distance comparison.

    Args:
        route: List of (lat, lon) points.
        size: Output image size (square).
        line_width: Width of the drawn line.
        bg_color: Background pixel value (0 = black).
        line_color: Line pixel value (255 = white).

    Returns:
        Grayscale PIL Image.
    """
    if len(route) < 2:
        return Image.new("L", (size, size), bg_color)

    points = np.array(route)
    lats, lons = points[:, 0], points[:, 1]

    # Normalize to pixel coordinates
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    lat_range = lat_max - lat_min or 1e-6
    lon_range = lon_max - lon_min or 1e-6

    # Add margin
    margin = 0.05
    px = ((lons - lon_min) / lon_range * (1 - 2 * margin) + margin) * size
    # Flip y-axis (lat increases upward, pixels increase downward)
    py = ((1 - (lats - lat_min) / lat_range) * (1 - 2 * margin) + margin) * size

    img = Image.new("L", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    coords = list(zip(px.tolist(), py.tolist()))
    draw.line(coords, fill=line_color, width=line_width)

    return img


def render_map_overlay(
    route: list[tuple[float, float]],
    size: int = 512,
) -> Image.Image:
    """Render a route overlaid on a map tile.

    Used for CLIP scoring. Falls back to a styled polyline render
    if map tiles are unavailable.

    Args:
        route: List of (lat, lon) points.
        size: Output image size (square).

    Returns:
        RGB PIL Image.
    """
    try:
        return _render_with_staticmap(route, size)
    except Exception:
        # Fallback to simple colored render
        return _render_styled_polyline(route, size)


def _render_with_staticmap(
    route: list[tuple[float, float]], size: int
) -> Image.Image:
    """Render using staticmap library for tile-based map background."""
    import staticmap

    m = staticmap.StaticMap(size, size)
    line = staticmap.Line(
        [(lon, lat) for lat, lon in route],
        color="red",
        width=3,
    )
    m.add_line(line)
    return m.render()


def _render_styled_polyline(
    route: list[tuple[float, float]], size: int
) -> Image.Image:
    """Render a colored polyline on a light background (fallback)."""
    if len(route) < 2:
        return Image.new("RGB", (size, size), (240, 240, 240))

    points = np.array(route)
    lats, lons = points[:, 0], points[:, 1]

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_range = lat_max - lat_min or 1e-6
    lon_range = lon_max - lon_min or 1e-6

    margin = 0.05
    px = ((lons - lon_min) / lon_range * (1 - 2 * margin) + margin) * size
    py = ((1 - (lats - lat_min) / lat_range) * (1 - 2 * margin) + margin) * size

    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    coords = list(zip(px.tolist(), py.tolist()))
    draw.line(coords, fill=(220, 50, 50), width=3)

    return img


def route_to_edge_image(
    route: list[tuple[float, float]],
    size: int = 512,
) -> np.ndarray:
    """Render route as binary edge image for Chamfer comparison.

    Returns:
        (size, size) numpy array with values 0 or 255.
    """
    img = render_polyline(route, size=size)
    return np.array(img)
