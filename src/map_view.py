"""A small, offline SVG map. Rendering never invents road geometry."""

import base64

from functools import lru_cache
from pathlib import Path

from src.geo import SF, TO_METERS
from src.graph import BikeRouter, edge_coordinates
from src.terrain import terrain_image, profile_markup

ASSETS = Path(__file__).parent / "assets"


def svg_path(xy, frame=SF, precision=2, tolerance=0.0):
    points = frame.meters_to_pixels(xy)
    if tolerance:
        points = _drop_collinear(points, tolerance)
    return "M" + "L".join(f"{x:.{precision}f},{y:.{precision}f}" for x, y in points)


def _drop_collinear(points, tolerance):
    """Drop points that sit within `tolerance` pixels of the line they lie on.

    Street geometry carries far more vertices than the 511-pixel frame can show.
    At the 6x zoom ceiling a 0.02 px tolerance is a fifth of a device pixel, so
    the drawn roads are unchanged while the browser re-strokes far less per frame.
    """
    if len(points) < 3:
        return points
    kept = [points[0]]
    anchor = points[0]
    for index in range(1, len(points) - 1):
        nxt = points[index + 1]
        current = points[index]
        dx, dy = nxt[0] - anchor[0], nxt[1] - anchor[1]
        span = (dx * dx + dy * dy) ** .5
        if span == 0:
            continue
        # Perpendicular distance from the candidate to the anchor→next chord.
        offset = abs(dx * (anchor[1] - current[1]) - dy * (anchor[0] - current[0])) / span
        if offset > tolerance:
            kept.append(current)
            anchor = current
    kept.append(points[-1])
    return kept


def _chain(polylines):
    """Join polylines that share an endpoint so one subpath covers a whole run.

    Purely a serialisation change: the same strokes are drawn, but with far fewer
    moveto commands for the renderer to set up.
    """
    starts = {}
    for index, line in enumerate(polylines):
        starts.setdefault(line[0], []).append(index)
    used = [False] * len(polylines)
    chains = []
    for index, line in enumerate(polylines):
        if used[index]:
            continue
        used[index] = True
        chain = list(line)
        while True:
            candidates = starts.get(chain[-1], ())
            following = next((c for c in candidates if not used[c]), None)
            if following is None:
                break
            used[following] = True
            chain.extend(polylines[following][1:])
        chains.append(chain)
    return chains


@lru_cache(maxsize=4)
def street_paths(frame=SF, graph=None):
    graph = BikeRouter.load().graph if graph is None else graph
    seen = set()
    polylines = []
    for u, v, data in graph.edges(data=True):
        xy = edge_coordinates(graph, u, v, data)
        # Reverse directions share a stroke, but distinct parallel roads don't.
        signature = tuple(map(tuple, xy if u <= v else xy[::-1]))
        if signature not in seen:
            seen.add(signature)
            points = _drop_collinear(frame.meters_to_pixels(xy), .02)
            polylines.append(tuple((round(x, 1), round(y, 1)) for x, y in points))
    return " ".join(
        "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in chain) for chain in _chain(polylines))


def markup(route=None, outline=None, frame=SF, graph=None, label='san francisco', terrain=True,
           gpx=None, another=False):
    try:
        roads = street_paths(frame, graph)
    except (RuntimeError, FileNotFoundError):
        roads = ""
    labels = []
    for name, lat, lon in [("richmond", 37.78, -122.478), ("golden gate park", 37.768, -122.48),
                           ("sunset", 37.752, -122.484), ("haight", 37.772, -122.445),
                           ("castro", 37.761, -122.435), ("mission", 37.754, -122.419),
                           ("noe valley", 37.747, -122.433), ("twin peaks", 37.755, -122.447)]:
        x, y = frame.meters_to_pixels([TO_METERS.transform(lon, lat)])[0]
        labels.append(f'<text x="{x:.2f}" y="{y:.2f}">{name}</text>')
    route_path = svg_path(route, frame) if route is not None else ""
    outline_path = svg_path(outline, frame) if outline is not None else ""
    start = ""
    view_box = "0 0 511 511"
    if route is not None:
        pixels = frame.meters_to_pixels(route)
        x, y = pixels[0]
        start = (f'<g class="route-progress-marker" transform="translate({x:.2f} {y:.2f})" '
                 'role="slider" aria-label="Position along route" tabindex="0">'
                 '<circle r="10" fill="transparent"/>'
                 '<text class="route-progress-bike" x="0" y="0" text-anchor="middle" '
                 'dominant-baseline="central" font-size="9">🚲</text></g>')
    gpx_action = ""
    if gpx:
        encoded = base64.b64encode(gpx.encode("utf-8")).decode("ascii")
        gpx_action = (f'<a class="map-action" download="route.gpx" '
                      f'href="data:application/gpx+xml;base64,{encoded}">download GPX ↗</a>')
    another_action = ('<button type="button" class="map-action" data-main-action="another">'
                      'another route ↻</button>') if another else ""
    # Skipped under terrain: the terrain PNG is opaque and covers the whole frame,
    # so these two fills would be rasterised on every repaint and never seen.
    backdrop = "" if terrain else (
        '<defs><pattern id="grain" width="3" height="3" patternUnits="userSpaceOnUse">'
        '<circle cx="1" cy="1" r=".28" fill="#737373" opacity=".18"/></pattern></defs>'
        '<rect x="-1000" y="-1000" width="2511" height="2511" fill="var(--map-paper, #f1f2ec)"/>'
        '<rect width="511" height="511" fill="url(#grain)"/>')
    return f'''<div class="street-map">
      <div class="map-eyebrow"><span>{label} / {frame.width_m / 1609.344:.1f} × {frame.width_m / 1609.344:.1f} mi</span><span>n ↑</span></div>
      <svg class="map-canvas" viewBox="{view_box}" tabindex="0" role="img"
           aria-label="San Francisco bicycle streets. Drag to pan; plus and minus keys to zoom; zero to reset.">
        {backdrop}
        {terrain_image() if terrain else ''}
        <path class="streets" d="{roads}" fill="none" stroke="#7c8780" stroke-width=".55"/>
        <g class="neighborhoods">{''.join(labels)}</g>
        <path class="reference-outline" d="{outline_path}" fill="none" stroke="#bd8e8e" stroke-width="1.1" stroke-dasharray="3 3"/>
        <path class="ride" d="{route_path}" fill="none" stroke="var(--route-color, #637baa)" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
        {start}
      </svg>
      <div class="map-bottom">{'<button type="button" class="route-play" aria-label="Play route direction">▶</button>' if route is not None else ''}<div class="map-actions">{gpx_action}{another_action}</div><div>
        <button type="button" data-map="in" aria-label="Zoom in">+</button>
        <button type="button" data-map="out" aria-label="Zoom out">−</button>
        <button type="button" data-map="reset" aria-label="Reset map">reset</button>
      </div></div>
    </div>{profile_markup(route)}'''
