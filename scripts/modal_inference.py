#!/usr/bin/env python3
"""Generate GPX routes on Modal with GPU-accelerated SD 1.5.

Usage:
    modal run scripts/modal_inference.py --concept horse
    modal run scripts/modal_inference.py --concept star --seed 42 --output star.gpx
    modal run scripts/modal_inference.py --concept heart --checkpoint /data/checkpoints/final
"""

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

IGNORE_PATTERNS = [
    ".venv/**", "__pycache__/**", "*.pyc", ".git/**",
    "outputs/**", "route_cache/**", "checkpoints/**",
    "*.egg-info/**", ".pytest_cache/**",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.1",
        "torchvision>=0.16",
        "diffusers>=0.25",
        "transformers>=4.36",
        "accelerate>=0.25",
        "opencv-python-headless>=4.8",
        "numpy>=1.24",
        "scipy>=1.11",
        "gpxpy>=1.6",
        "Pillow>=10.0",
        "requests>=2.31",
        "aiohttp>=3.9",
        "osmnx>=2.0",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
        "wandb>=0.16",
    )
    .add_local_dir(".", remote_path="/app", ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App("route-sculptor-inference", image=image)
vol = modal.Volume.from_name("route-sculptor-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_route(
    concept: str,
    seed: int | None = None,
    checkpoint: str | None = None,
    preview: bool = False,
) -> dict:
    """Generate a GPX route for a concept on GPU."""
    import sys
    import logging
    import yaml

    sys.path.insert(0, "/app")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("modal_inference")

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    os.makedirs("/data/hf_cache", exist_ok=True)
    os.makedirs("/data/route_cache", exist_ok=True)

    import src.pipeline.routing as routing_mod
    routing_mod.CACHE_DIR = Path("/data/route_cache")

    with open("/app/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["image_gen"]["num_inference_steps"] = 30

    model_id = checkpoint or cfg["image_gen"]["model_id"]

    from src.pipeline.image_gen import ImageGenerator
    from src.pipeline.edge_detect import ContourSet, process_image
    from src.pipeline.placement import BBox, grid_search
    from src.pipeline.waypoints import (
        sample_adaptive_waypoints, densify_waypoints, sample_multi_contour_waypoints,
    )
    from src.pipeline.routing import create_router
    from src.pipeline.gpx_utils import route_to_gpx

    mc_cfg = cfg.get("multi_contour", {})
    gen_cfg = cfg["image_gen"]
    use_controlnet = gen_cfg.get("use_controlnet", False)

    if use_controlnet:
        # INVERTED PIPELINE: placement first → render map → ControlNet generation
        logger.info(f"ControlNet mode: generating '{concept}' conditioned on road map")

        from src.pipeline.image_gen import ControlNetImageGenerator
        from src.pipeline.map_render import render_local_map

        # Step 1: Pick a placement (use a dummy circle contour for grid search)
        import numpy as np
        t = np.linspace(0, 2 * np.pi, 100)
        dummy_contour = np.column_stack([0.5 + 0.4 * np.cos(t), 0.5 + 0.4 * np.sin(t)])

        sf = cfg["sf_bbox"]
        bbox = BBox(**sf)
        p_cfg = cfg["placement"]
        placements = grid_search(
            dummy_contour, bbox,
            num_positions=p_cfg["num_positions"], num_scales=p_cfg["num_scales"],
            num_rotations=1,  # no rotation for ControlNet (map is fixed)
            scale_range_km=tuple(p_cfg["scale_range_km"]),
            max_rotation_deg=0.0,
        )
        best_placement = placements[0][0] if placements else None

        if best_placement is None:
            raise RuntimeError("No valid placement found")

        # Step 2: Render local road map for this placement
        router = create_router(cfg["routing"])
        road_map = render_local_map(
            router.G, best_placement.center_lat, best_placement.center_lon,
            best_placement.scale_km, size=gen_cfg["image_size"],
        )

        # Step 3: Generate silhouette conditioned on road map
        cn_gen = ControlNetImageGenerator(
            model_id=checkpoint or gen_cfg["model_id"],
            controlnet_id=gen_cfg.get("controlnet_model", "lllyasviel/control_v11p_sd15_canny"),
            prompt_template=gen_cfg["prompt_template"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            image_size=gen_cfg["image_size"],
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            controlnet_scale=gen_cfg.get("controlnet_scale", 0.5),
        )
        image = cn_gen.generate(concept, conditioning_image=road_map, seed=seed)

    else:
        # STANDARD PIPELINE: generate first, then place
        logger.info(f"Generating image for '{concept}' with model {model_id}")
        generator = ImageGenerator(
            model_id=model_id,
            prompt_template=gen_cfg["prompt_template"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            image_size=gen_cfg["image_size"],
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
        )
        image = generator.generate(concept, seed=seed)

    logger.info("Extracting edges...")
    edge_cfg = cfg["edge_detect"]
    edges, contour_data = process_image(
        image, edge_cfg["canny_low"], edge_cfg["canny_high"], edge_cfg["blur_kernel"],
        multi_contour=mc_cfg.get("enabled", True),
        min_inner_area_ratio=mc_cfg.get("min_inner_area_ratio", 0.005),
        max_inner_contours=mc_cfg.get("max_inner_contours", 5),
    )
    if contour_data is None:
        raise RuntimeError(f"No contour found in generated image for '{concept}'")

    is_multi = isinstance(contour_data, ContourSet)

    if use_controlnet and best_placement is not None:
        # ControlNet: placement already chosen, use it to transform contour
        from src.pipeline.placement import transform_contour, transform_contour_set
        if is_multi:
            geo_data = transform_contour_set(
                contour_data, best_placement.center_lat, best_placement.center_lon,
                best_placement.scale_km, best_placement.rotation_deg,
            )
        else:
            geo_data = transform_contour(
                contour_data, best_placement.center_lat, best_placement.center_lon,
                best_placement.scale_km, best_placement.rotation_deg,
            )
        best = best_placement
    else:
        # Standard: grid search for placement
        logger.info("Grid search placement...")
        sf = cfg["sf_bbox"]
        bbox = BBox(**sf)
        p_cfg = cfg["placement"]
        placements = grid_search(
            contour_data, bbox,
            num_positions=p_cfg["num_positions"],
            num_scales=p_cfg["num_scales"],
            num_rotations=p_cfg["num_rotations"],
            scale_range_km=tuple(p_cfg["scale_range_km"]),
            max_rotation_deg=p_cfg.get("max_rotation_deg", 15.0),
        )
        if not placements:
            raise RuntimeError(f"No valid placements found for '{concept}'")
        best, geo_data = placements[0]

    logger.info(
        f"Placement: center=({best.center_lat:.4f}, {best.center_lon:.4f}), "
        f"scale={best.scale_km:.1f}km, rotation={best.rotation_deg:.0f}°"
    )

    logger.info("Computing bike route...")
    router = create_router(cfg["routing"])

    # Snap contour to road graph if available
    if hasattr(router, 'G'):
        from src.pipeline.road_align import RoadAligner
        logger.info("Snapping contour to road graph...")
        aligner = RoadAligner(router.G)
        outer = geo_data.outer if (is_multi and isinstance(geo_data, ContourSet)) else geo_data
        waypoints = aligner.snap_contour(outer, subsample=3)
    else:
        wp_cfg = cfg["waypoints"]
        if is_multi and isinstance(geo_data, ContourSet):
            waypoints = sample_multi_contour_waypoints(
                geo_data,
                num_points=wp_cfg["num_points"],
                curvature_weight=wp_cfg.get("curvature_weight", 2.0),
                outer_budget_min=mc_cfg.get("outer_budget_min", 0.6),
                min_inner_waypoints=mc_cfg.get("min_inner_waypoints", 4),
                max_gap_km=wp_cfg.get("max_gap_km", 0.15),
            )
        else:
            waypoints = sample_adaptive_waypoints(
                geo_data,
                num_points=wp_cfg["num_points"],
                curvature_weight=wp_cfg.get("curvature_weight", 2.0),
            )
            waypoints = densify_waypoints(waypoints, max_gap_km=wp_cfg.get("max_gap_km", 0.15))

    route = router.route_waypoints(waypoints)

    if len(route) < 2:
        raise RuntimeError(f"Routing failed for '{concept}'")

    gpx = route_to_gpx(route, name=f"Route Sculptor: {concept}")
    gpx_bytes = gpx.to_xml().encode("utf-8")

    logger.info(f"Generated route: {len(route)} points, {len(gpx_bytes)} bytes GPX")

    result = {
        "gpx_bytes": gpx_bytes,
        "num_points": len(route),
        "concept": concept,
    }

    if preview:
        import io
        import cv2 as _cv2
        import numpy as _np
        from src.evaluation.render import render_polyline
        from src.pipeline.edge_detect import extract_contour_set
        from PIL import Image as _PILImage

        # Raw silhouette (before post-processing)
        raw_image = None
        if use_controlnet and hasattr(cn_gen, '_last_raw'):
            raw_image = cn_gen._last_raw
        elif not use_controlnet and hasattr(generator, '_last_raw'):
            raw_image = generator._last_raw
        if raw_image is not None:
            raw_buf = io.BytesIO()
            raw_image.save(raw_buf, format="PNG")
            result["raw_silhouette_png"] = raw_buf.getvalue()

        # Road map (ControlNet conditioning)
        if use_controlnet and road_map is not None:
            map_buf = io.BytesIO()
            road_map.convert("RGB").save(map_buf, format="PNG")
            result["road_map_png"] = map_buf.getvalue()

        # Processed silhouette
        sil_buf = io.BytesIO()
        image.save(sil_buf, format="PNG")
        result["silhouette_png"] = sil_buf.getvalue()

        # Edges
        edges_buf = io.BytesIO()
        _PILImage.fromarray(edges).save(edges_buf, format="PNG")
        result["edges_png"] = edges_buf.getvalue()

        # Contour overlay (outer=red, inner=blue)
        pixel_cs = extract_contour_set(image)
        if pixel_cs is not None:
            vis = _np.array(image.copy().convert("RGB"))
            _cv2.drawContours(vis, [pixel_cs.outer], -1, (255, 0, 0), 2)
            for c in pixel_cs.inner:
                _cv2.drawContours(vis, [c], -1, (0, 100, 255), 2)
            contour_buf = io.BytesIO()
            _PILImage.fromarray(vis).save(contour_buf, format="PNG")
            result["contour_png"] = contour_buf.getvalue()

        # Route polyline
        polyline_img = render_polyline(route)
        poly_buf = io.BytesIO()
        polyline_img.save(poly_buf, format="PNG")
        result["polyline_png"] = poly_buf.getvalue()

        # Metadata
        result["metadata"] = {
            "concept": concept,
            "placement": {
                "center_lat": best.center_lat, "center_lon": best.center_lon,
                "scale_km": best.scale_km, "rotation_deg": best.rotation_deg,
            },
            "route_points": len(route),
            "waypoints": len(waypoints),
            "inner_contours": len(contour_data.inner) if is_multi else 0,
        }

    vol.commit()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    concept: str = "horse",
    output: str = "output.gpx",
    seed: int = 42,
    checkpoint: str = None,
    preview: bool = False,
):
    """Generate a GPX route using Modal GPU."""
    print(f"Generating route for '{concept}' (seed={seed}) on Modal A10G...")

    result = generate_route.remote(
        concept=concept,
        seed=seed,
        checkpoint=checkpoint,
        preview=preview,
    )

    # Save to structured directory: outputs/{concept}/
    output_path = Path(output)
    if output_path.suffix == ".gpx":
        concept_dir = output_path.parent
    else:
        concept_dir = output_path / concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    gpx_path = concept_dir / "route.gpx"
    gpx_path.write_bytes(result["gpx_bytes"])
    print(f"GPX saved to {gpx_path} ({result['num_points']} points)")

    if preview:
        for key, filename in [
            ("raw_silhouette_png", "raw_silhouette.png"),
            ("silhouette_png", "silhouette.png"),
            ("road_map_png", "road_map.png"),
            ("edges_png", "edges.png"),
            ("contour_png", "contour.png"),
            ("polyline_png", "polyline.png"),
        ]:
            if key in result:
                (concept_dir / filename).write_bytes(result[key])

        if "metadata" in result:
            import json
            with open(concept_dir / "metadata.json", "w") as f:
                json.dump(result["metadata"], f, indent=2)

        print(f"All outputs saved to {concept_dir}/")
