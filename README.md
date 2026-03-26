# Route Sculptor

Generate rideable GPX bike routes in San Francisco that trace the shape of any concept. Type "horse" and get a horse-shaped bike route through real streets.

## How It Works

```
"horse" → SD 1.5 silhouette → binary threshold → contour extraction →
         placement on SF map → graph-based road routing → GPX file
```

1. **Image Generation**: SD 1.5 generates a black silhouette from text (or loads a pre-made SVG)
2. **Post-Processing**: Otsu threshold + corner brightness check + small blob removal → pure black on white
3. **Contour Extraction**: OpenCV RETR_TREE hierarchy extracts outer boundary + inner details (eyes, etc.)
4. **Placement**: Grid search over position × scale × rotation on SF map, scored by coverage
5. **Waypoint Sampling**: Curvature-adaptive sampling (more points at sharp corners) + 150m max gap densification
6. **Graph Routing**: Local OSM bike network (104K nodes) with Dijkstra + backtrack penalty
7. **GPX Export**: Standard GPX loadable in Strava, Komoot, or any GPS device

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Download SF bike network (one-time, ~130MB)
python -c "import osmnx as ox; G = ox.graph_from_place('San Francisco, California, USA', network_type='bike', custom_filter='[\"route\"!~\"ferry\"]'); ox.save_graphml(G, 'data/sf_bike_graph.graphml')"
```

## Usage

### Generate a route (GPU via Modal)
```bash
modal run scripts/modal_inference.py --concept horse --preview
# Output: outputs/horse/route.gpx + silhouette.png + map_overlay.png
```

### Batch generation
```bash
# All pre-made silhouettes (no GPU needed)
PYTHONPATH=. python scripts/batch_generate.py --svg-only

# With SD 1.5 (needs GPU or Modal)
PYTHONPATH=. python scripts/batch_generate.py --use-sd horse dog cat star heart
```

### Local CLI
```bash
PYTHONPATH=. python -m src.inference.cli --concept horse --preview
```

## Output Structure
```
outputs/{concept}/
    silhouette.png    # Generated image (binary black/white)
    edges.png         # Canny edge detection
    contour.png       # Extracted contour overlay (red=outer, blue=inner)
    polyline.png      # Route shape
    map_overlay.png   # Route on SF map tiles
    route.gpx         # Rideable GPX file
    metadata.json     # Placement details + stats
```

## Architecture

```
src/pipeline/
    image_gen.py        # SD 1.5 + binary post-processing + SVG loader
    edge_detect.py      # Canny + ContourSet (outer + inner) + Otsu silhouette extraction
    placement.py        # Grid search placement on SF map
    waypoints.py        # Curvature-adaptive sampling + bridge-and-trace + densification
    graph_routing.py    # Local OSM Dijkstra routing with backtrack penalty
    routing.py          # Router factory (graph/valhalla/osrm)
    gpx_utils.py        # GPX I/O
src/evaluation/
    clip_score.py       # CLIP text-image similarity
    chamfer.py          # Chamfer distance
    reward.py           # Combined reward function
    render.py           # Route visualization
src/training/
    ddpo_trainer.py     # DDPO RL training (TRL)
    reward_fn.py        # Pipeline-as-reward for DDPO
    dataset.py          # Concept loading
src/inference/
    cli.py              # CLI interface
    gradio_app.py       # Web UI
```

## Roadmap
- [ ] **Road-snapping**: Snap contours to road grid before routing (eliminates zig-zags)
- [ ] **ControlNet**: Condition SD on rendered SF street map for road-aware generation
- [ ] **CLIP-only RL**: Train SD with DDPO using just CLIP score as reward
