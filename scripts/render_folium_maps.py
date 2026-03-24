#!/usr/bin/env python3
"""Render GPX routes on interactive folium maps and export as PNG."""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import gpxpy
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.gpx_utils import load_gpx, gpx_to_points


def render_gpx_on_map(gpx_path: Path, output_html: Path, output_png: Path | None = None) -> None:
    """Render a GPX route on an OpenStreetMap folium map."""
    gpx = load_gpx(gpx_path)
    points = gpx_to_points(gpx)

    if len(points) == 0:
        print(f"  Skipping {gpx_path.name}: no points")
        return

    lats, lons = points[:, 0], points[:, 1]
    center_lat = (lats.min() + lats.max()) / 2
    center_lon = (lons.min() + lons.max()) / 2

    # Create map centered on route
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
    )

    # Add the route as a polyline
    route_coords = [[lat, lon] for lat, lon in zip(lats.tolist(), lons.tolist())]
    folium.PolyLine(
        route_coords,
        color="red",
        weight=4,
        opacity=0.85,
    ).add_to(m)

    # Add start/end markers
    folium.CircleMarker(
        location=route_coords[0],
        radius=6,
        color="green",
        fill=True,
        fill_color="green",
        popup="Start",
    ).add_to(m)
    folium.CircleMarker(
        location=route_coords[-1],
        radius=6,
        color="blue",
        fill=True,
        fill_color="blue",
        popup="End",
    ).add_to(m)

    # Fit bounds to route
    m.fit_bounds([[lats.min(), lons.min()], [lats.max(), lons.max()]])

    # Save HTML
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    print(f"  Saved: {output_html}")

    # Try to export PNG via selenium
    if output_png:
        try:
            _export_png(m, output_png)
            print(f"  Saved: {output_png}")
        except Exception as e:
            print(f"  PNG export skipped ({e})")


def _export_png(m: folium.Map, output_png: Path) -> None:
    """Export folium map to PNG using selenium."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import time

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=800,800")

    driver = webdriver.Chrome(options=options)

    # Save temp HTML
    html_data = m._repr_html_()
    tmp = output_png.with_suffix(".tmp.html")
    m.save(str(tmp))

    driver.get(f"file://{tmp.resolve()}")
    time.sleep(3)  # Wait for tiles to load

    output_png.parent.mkdir(parents=True, exist_ok=True)
    driver.save_screenshot(str(output_png))
    driver.quit()
    tmp.unlink(missing_ok=True)


def main():
    outputs_dir = Path("outputs")

    demos = [
        ("demo", "star"),
        ("demo_heart", "heart"),
        ("demo_horse", "horse"),
    ]

    for demo_dir, shape in demos:
        gpx_path = outputs_dir / demo_dir / f"{shape}_route.gpx"
        if not gpx_path.exists():
            print(f"Skipping {gpx_path}: not found")
            continue

        print(f"\n{shape.upper()}:")
        html_out = outputs_dir / demo_dir / f"{shape}_map_overlay.html"
        png_out = outputs_dir / demo_dir / f"{shape}_map_overlay.png"
        render_gpx_on_map(gpx_path, html_out, png_out)


if __name__ == "__main__":
    main()
