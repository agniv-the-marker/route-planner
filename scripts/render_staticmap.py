#!/usr/bin/env python3
"""Render GPX routes on real map tiles using staticmap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import staticmap
from src.pipeline.gpx_utils import load_gpx, gpx_to_points


def render_gpx_on_tiles(gpx_path: Path, output_png: Path, size: int = 800) -> None:
    """Render a GPX route on OSM tiles."""
    gpx = load_gpx(gpx_path)
    points = gpx_to_points(gpx)

    if len(points) == 0:
        print(f"  Skipping {gpx_path.name}: no points")
        return

    m = staticmap.StaticMap(size, size)

    # Add route as a red line
    coords = [(lon, lat) for lat, lon in points.tolist()]
    line = staticmap.Line(coords, color="red", width=4)
    m.add_line(line)

    # Add start marker (green)
    start = staticmap.CircleMarker(coords[0], color="green", width=10)
    m.add_marker(start)

    # Add end marker (blue)
    end = staticmap.CircleMarker(coords[-1], color="blue", width=10)
    m.add_marker(end)

    img = m.render()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_png))
    print(f"  Saved: {output_png}")


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
        png_out = outputs_dir / demo_dir / f"{shape}_map_overlay.png"
        render_gpx_on_tiles(gpx_path, png_out)


if __name__ == "__main__":
    main()
