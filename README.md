# Route Sculptor

An RL system that generates GPX bike routes in San Francisco depicting visual concepts.

**Pipeline**: text concept → Stable Diffusion 1.5 → Canny edge detection → outline placement on SF map → OSRM bike routing → GPX → evaluate via CLIP + Chamfer distance → DDPO reward → fine-tune SD 1.5.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Generate a route (inference)

```bash
python scripts/generate.py --concept "horse" --output route.gpx
```

### Launch Gradio UI

```bash
python scripts/generate.py --gradio
```

### Train with DDPO

```bash
python scripts/train.py --config configs/default.yaml
```

## Architecture

```
"horse" → SD 1.5 → Canny Edges → Largest Contour
    → Grid Search (pos/scale/rot) on SF map
    → 50-80 Waypoints → OSRM Bike Routing → GPX
    → Render → CLIP Score + Chamfer Distance
    → Reward = 0.7·CLIP + 0.3·Chamfer → DDPO
```

## Project Structure

```
src/
├── pipeline/       # Image gen, edge detection, placement, routing, GPX
├── evaluation/     # Rendering, CLIP, Chamfer, reward
├── training/       # DDPO trainer, reward wrapper, dataset
└── inference/      # CLI and Gradio UI
```
