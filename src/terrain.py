"""Prepared terrain in the same projected frame as the route and street map."""
import base64
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from src.geo import SF

DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_terrain():
    metadata = json.loads((DATA / "sf-terrain.json").read_text())
    if metadata["frame"] != repr(SF):
        raise ValueError("Terrain needs rebuilding for this map frame.")
    elevations = np.load(DATA / "sf-elevation.npy", allow_pickle=False)
    image = base64.b64encode((DATA / "sf-terrain.png").read_bytes()).decode("ascii")
    return elevations, image


def terrain_image():
    try:
        _, image = load_terrain()
    except (OSError, ValueError):
        return ""
    return f'<image class="terrain" x="0" y="0" width="511" height="511" href="data:image/png;base64,{image}"/>'


def profile_markup(route):
    if route is None or len(route) < 2:
        return ""
    try:
        elevations, _ = load_terrain()
    except (OSError, ValueError):
        return ""
    # Sample at uniform distance so the chart's x-axis represents ride distance.
    from shapely.geometry import LineString
    line = LineString(route)
    distance = np.linspace(0, line.length, 200)
    xy = np.array([line.interpolate(d).coords[0] for d in distance])
    pixels = SF.meters_to_pixels(xy) * (len(elevations) - 1) / (SF.size - 1)
    heights = map_coordinates(elevations, [pixels[:, 1], pixels[:, 0]], order=1, mode="nearest")
    low, high = float(heights.min()), float(heights.max())
    plot_x = np.linspace(0, 511, len(heights))
    plot_y = 65 - (heights - low) / max(30, high - low) * 50
    path = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in zip(plot_x, plot_y))
    ticks = ''.join(f'<line x1="{511*i/4:.1f}" x2="{511*i/4:.1f}" y1="4" y2="66" stroke="currentColor" opacity=".18"/><text x="{511*i/4:.1f}" y="68" text-anchor="{"start" if i == 0 else "end" if i == 4 else "middle"}">{line.length/1609.344*i/4:.1f} mi</text>' for i in range(5))
    return f'''<div class="elevation-profile"><div class="map-eyebrow">
      <span>elevation · approx. {low * 3.28084:.0f}–{high * 3.28084:.0f} ft</span><span>{line.length/1609.344:.1f} mi</span></div>
      <svg viewBox="0 0 511 70" role="img" aria-label="Approximate ground elevation along the route">
      <path d="{path}L511,70L0,70Z" fill="var(--route-color, #637baa)" opacity=".14"/>
      <path class="route-elevation-line" d="{path}" fill="none" stroke="var(--route-color, #637baa)" stroke-width="1.5"/>
      <g class="elevation-ticks" font-family="JetBrains Mono, monospace" font-size="8" fill="currentColor">{ticks}</g>
      </svg></div>'''
