"""Render the SF road network as an image for ControlNet conditioning.

Creates a 512x512 image of white road lines on black background,
suitable as a Canny-style conditioning image for ControlNet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def render_road_map(
    graph,
    bbox: tuple[float, float, float, float],
    size: int = 512,
    line_width: int = 1,
    min_edge_length: float = 0,
) -> Image.Image:
    """Render the road graph as white lines on black background.

    Args:
        graph: osmnx MultiDiGraph
        bbox: (min_lat, max_lat, min_lon, max_lon)
        size: Output image size (square)
        line_width: Width of road lines in pixels

    Returns:
        PIL Image (size x size) — white roads on black background
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    if lat_span == 0 or lon_span == 0:
        return Image.new("L", (size, size), 0)

    img = Image.new("L", (size, size), 0)  # black background
    draw = ImageDraw.Draw(img)

    def geo_to_pixel(lat: float, lon: float) -> tuple[int, int]:
        x = int((lon - min_lon) / lon_span * (size - 1))
        y = int((max_lat - lat) / lat_span * (size - 1))  # flip y
        return (x, y)

    # Draw edges (optionally filter by minimum length for sparser maps)
    for u, v, data in graph.edges(data=True):
        try:
            # Skip short edges for sparser rendering
            if min_edge_length > 0 and float(data.get("length", 0)) < min_edge_length:
                continue

            u_lat, u_lon = graph.nodes[u]["y"], graph.nodes[u]["x"]
            v_lat, v_lon = graph.nodes[v]["y"], graph.nodes[v]["x"]

            # Skip edges outside bbox
            if (u_lat < min_lat or u_lat > max_lat or
                u_lon < min_lon or u_lon > max_lon):
                continue

            p1 = geo_to_pixel(u_lat, u_lon)
            p2 = geo_to_pixel(v_lat, v_lon)
            draw.line([p1, p2], fill=255, width=line_width)
        except (KeyError, ValueError):
            continue

    logger.info(f"Rendered road map: {size}x{size}, {len(graph.edges)} edges")
    return img


def render_local_map(
    graph,
    center_lat: float,
    center_lon: float,
    scale_km: float,
    size: int = 512,
    line_width: int = 1,
    sparse: bool = True,
) -> Image.Image:
    """Render a local road map centered on a specific position and scale.

    Use this for per-placement ControlNet conditioning — each placement
    candidate gets its own map showing the exact local street grid.

    Args:
        graph: osmnx MultiDiGraph
        center_lat, center_lon: Center of the view
        scale_km: Radius of the view in km
        size: Output image size
        line_width: Road line width

    Returns:
        PIL Image of local road network
    """
    # Convert km to degrees
    dlat = scale_km / 111.0
    dlon = scale_km / (111.0 * np.cos(np.radians(center_lat)))

    bbox = (
        center_lat - dlat,
        center_lat + dlat,
        center_lon - dlon,
        center_lon + dlon,
    )
    # Sparse: only render roads longer than 50m (filters out alleys/driveways)
    min_len = 50.0 if sparse else 0
    return render_road_map(graph, bbox, size, line_width, min_edge_length=min_len)


def render_road_map_from_file(
    graph_path: str | Path = "data/sf_bike_graph.graphml",
    bbox: tuple[float, float, float, float] | None = None,
    size: int = 512,
) -> Image.Image:
    """Load graph and render road map.

    Args:
        graph_path: Path to graphml file
        bbox: Optional bbox override. If None, uses graph extent.
        size: Output image size

    Returns:
        PIL Image of road network
    """
    import osmnx as ox

    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    G = ox.load_graphml(graph_path)

    if bbox is None:
        # Use graph extent
        lats = [G.nodes[n]["y"] for n in G.nodes]
        lons = [G.nodes[n]["x"] for n in G.nodes]
        bbox = (min(lats), max(lats), min(lons), max(lons))

    return render_road_map(G, bbox, size)
