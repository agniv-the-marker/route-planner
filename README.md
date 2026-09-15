# Route Sculptor

Public site: [routesculptor.bike](https://routesculptor.bike/).
See [hosting and monthly budget](docs/HOSTING.md) for the Modal deployment, persistence, and credentials.

Describe a shape and Route Sculptor generates four Stable Diffusion silhouettes, extracts a
validated outer contour, and searches for a matching SF bicycle-street loop.
then searches for it as a closed bicycle street loop in San Francisco. The site shows the
shape, the route, distance, and a GPX download. The fixed map is 10 × 10 km, centered near
Twin Peaks (37.76, -122.45).

`claude --resume "route-sculptor-local-modal-setup" --dangerously-skip-permissions`

## Run locally

Python 3.12 or newer:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,modal]'
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m src.web
```

Open http://127.0.0.1:7860. `/about` explains the pipeline and data sources. There is no
local model to download: describing an exact catalogue name (see `src/assets/symbols.json`)
resolves instantly and for free; anything else calls a small model deployed on Modal (see
below) that only ever *picks* one of the catalogue shapes, so it never returns something
that fails to parse as a valid outline.

The prepared bicycle network is in `data/sf.graphml`, with its preparation date, settings,
and SHA-256 in `data/sf.json`. To deliberately rebuild it from OSM:

```bash
.venv/bin/python -m scripts.prepare_map
.venv/bin/python -m scripts.prepare_terrain
```

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
ODbL 1.0. No map tiles or routing service are needed at request time.

## Debug and dev views

- **Debug popup (`/?debug=true`):** the interpreted silhouette, the placed outline and
  candidate positions on the map, measured stage durations (interpretation / search / cache),
  and a JSON block with the exact interpretation, transform, and scores needed to reproduce
  the run.
- **Dev popup (`/?dev=true`):** street visibility/opacity, terrain opacity, a toggle for the
  placed reference outline, page palette, headline type, and route color. Appearance changes
  apply only in the browser tab; Reset restores defaults.
- **Map:** drag to pan within the fixed bounds, scroll or use buttons to zoom. At full
  zoom-out panning is locked. Keyboard `+`, `-`, and `0` zoom and reset. Terrain uses
  elevation-driven dither and hill shading adapted from
  [ferryri.de](https://github.com/agniv-the-marker/ferryri.de), with higher-resolution
  Mapzen/USGS terrain tiles and the reference site's MTC/ABAG coastline. A successful route
  includes an approximate ground-elevation profile.

Combine `/?debug=true&dev=true` to enable both popups. Close with Escape or the close button.
Neither popup appears on the normal page.

## How a description becomes a route

1. **Interpret** (`src/symbols.py`): the description is matched to a catalogue key directly
   (case/whitespace-normalized), or, if that fails, sent to a small model
   (Qwen3-4B-Instruct-2507) deployed on Modal that must return one of the catalogue names
   (or `none`) plus a small tilt/aspect adjustment via guided JSON decoding. The catalogue
   itself (`src/assets/symbols.json`, ~150 entries) is compiled offline from pinned Material
   Design Icons SVGs (`scripts/prepare_symbols.py`) — outer boundary only, simplified and
   validated as a simple closed polygon. The model can never hand back a shape that isn't
   already known-good.
2. **Search** (`src/street_search.py`): the outline is tried at many positions, rotations,
   and scales across the map; a directed street graph is matched to its perimeter with a
   cyclic dynamic program that keeps actual street curves and rejects excessive detours,
   disconnected paths, degenerate loops, and routes that reuse more than 20% of the same
   physical street length. Up to three qualifying alternatives are kept.
3. **Export**: GPX, an SVG map overlay, and an elevation profile are built directly from the
   chosen route's coordinates.

Disconnected placements, large detours, and degenerate loops fail with an explanation and no
GPX; the interpreted shape remains visible so you can see what was understood. The map
conservatively filters access and direction tags, including steps and ferries. It does not
implement turn-restriction relations or live closures. Check current access and conditions
before riding.

## Symbol interpretation and Modal

The interpreter model runs as its own Modal app, separate from the website:

```bash
MODAL_PROFILE=nyro-robotics modal deploy scripts/modal_symbols.py   # the LLM symbol picker
MODAL_PROFILE=nyro-robotics modal deploy scripts/modal_app.py       # the CPU website itself
```

`route-sculptor-symbols` (one L4, scales to zero after 5 minutes idle) is deployed today in
the `nyro-robotics` Modal workspace. The CPU website is now deployed at
[routesculptor.bike](https://routesculptor.bike/) with the current diffusion
pipeline and drawing mode; see [HOSTING.md](docs/HOSTING.md). Both the exact-match
path and the model path were validated end to end — see
[SYMBOL_EVALUATION.md](SYMBOL_EVALUATION.md) for measured pass rates, timings, and evidence,
including a real browser run and a full sweep of every catalogue symbol against the live
street graph.

Two earlier approaches were tried: SD 1.5 / ControlNet silhouette generation, and a free-form LLM-drawn vector outline. Both drawing generation and routing lost important geometry; valid routes did not establish recognizable subjects. Their code and evidence are kept for
reference (`outputs/legacy-diffusion-source/`, `src/outline_inference.py`,
`scripts/modal_outlines.py`) — see [OUTLINE_EVALUATION.md](OUTLINE_EVALUATION.md) for what
was tried and why it didn't clear the bar.

## Routing-first research

The catalogue’s 151/151 valid-route result does **not** establish recognition or requested pose fidelity. The separate generated-vector baseline compiled 11 of 120 outputs and produced zero qualifying routes. The experimental stroke router and 12-source approval workflow are documented in [ROUTING_MILESTONE.md](docs/ROUTING_MILESTONE.md). It supports open rides, internal strokes, visible connectors, upright placement, and 15/30/50/80 km limits on a separate SF–Palo Alto graph. Source approval and the controlled comparison are pending. See [PICKUP.md](PICKUP.md) for current state and [the historical index](docs/history/INDEX.md) for earlier evidence.

## Validation

```bash
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q
```

Tests cover coordinate alignment and rotation, directed/disconnected/curved/parallel-edge
routing, symbol selection and caching (exact match, model fallback, unknown descriptions),
street-search matching and placement, GPX round-trips, failure diagnostics, concurrency, and
the Gradio wiring (queueing, isolated downloads, the `/about` route). Tests inject the model
and search where useful and do not call Modal or require network access. For pass rates on
the real model and the real street graph, see [SYMBOL_EVALUATION.md](SYMBOL_EVALUATION.md).

## Experimental text-to-vector milestone

The public site uses the SD 1.5 → raster outline → historical street-search pipeline. It retains
failed images and diagnostics rather than replacing them with a catalogue icon. The generated-vector
pipeline, catalogue selection, and stroke router remain experimental.
