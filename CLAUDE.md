# Route Sculptor — Project Context

## What This Project Does
Generates rideable GPX bike routes in San Francisco that visually trace the shape of a text concept (e.g., "horse" → horse-shaped bike route on real streets).

## Current Pipeline
```
Text concept ("horse")
  → SD 1.5 generates black silhouette (30 steps, GPU via Modal)
  → Binary post-processing (Otsu threshold + corner-check + blob removal)
  → Contour extraction (outer + inner details via RETR_TREE hierarchy)
  → Grid search placement on SF map (position × scale × rotation)
  → Curvature-adaptive waypoint sampling + densification (150m max gap)
  → Graph routing (local OSM bike network, 104K nodes, backtrack penalty)
  → GPX export
```

## What Works Well
- **Graph routing**: 104K-node bike network, instant (~50s/concept), no API dependency
- **Binary post-processing**: Corner-check handles light/dark backgrounds, blob removal kills noise
- **Aspect-ratio normalization**: Preserves shape proportions during contour normalization
- **Backtrack penalty**: Router prefers fresh roads over already-traversed ones
- **Multi-contour bridge-and-trace**: Routes inner features (eyes, etc.) via bridge detours
- **Structured outputs**: `outputs/{concept}/` with silhouette, edges, contour, polyline, map_overlay, route.gpx, metadata.json
- **Batch generation**: `scripts/batch_generate.py` for multiple concepts

## Known Issues
- **SD 1.5 prompt inconsistency**: Some concepts produce bad silhouettes (textured backgrounds, inverted colors). Seed-dependent.
- **Shapes ignore road grid**: Routes zig-zag when the contour doesn't align with streets. Need road-snapping.
- **Grid search always picks max coverage**: All shapes end up at the same scale/position

## Key Bugs Fixed (don't reintroduce!)
- **Y-flip in transform_contour**: Image y=0 is top, geo lat increases north. Line: `geo[:, 0] = center_lat - rotated[:, 1] * dlat`
- **max_rotation_deg not passed from config**: Was hardcoded to 90°, should be 15° from config
- **Aspect ratio squashing**: normalize_contour must use `max(span_x, span_y)` for both axes
- **TRL DDPO API**: Needs `DefaultDDPOStableDiffusionPipeline`, `use_lora=True`, reward returns `(rewards, metadata)` tuple
- **TRL version**: Must use `trl>=0.7,<0.12` (DDPO removed in newer versions)
- **Modal .venv upload**: Must ignore `.venv/` in image build (2GB+ hangs)

## Infrastructure
- **Modal**: GPU inference (A10G) and training (A100). Secrets: `wandb-secret`, `huggingface-secret`
- **W&B**: Experiment tracking at https://wandb.ai/agniv-sarkar-agniv/route-sculptor
- **OSM Graph**: `data/sf_bike_graph.graphml` (130MB, 104K nodes, bike network, no ferries)
- **Config**: `configs/default.yaml` — all hyperparameters

## Next Steps
1. **Road-snapping** (`src/pipeline/road_align.py`): Snap contours to road grid before routing
2. **ControlNet conditioning**: Condition SD on rendered SF street map for road-aware generation
3. **CLIP-only RL**: Simplify reward to just CLIP score, train with DDPO + graph routing
