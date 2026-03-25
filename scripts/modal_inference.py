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

    logger.info(f"Generating image for '{concept}' with model {model_id}")
    generator = ImageGenerator(
        model_id=model_id,
        prompt_template=cfg["image_gen"]["prompt_template"],
        negative_prompt=cfg["image_gen"].get("negative_prompt", ""),
        image_size=cfg["image_gen"]["image_size"],
        num_inference_steps=cfg["image_gen"]["num_inference_steps"],
        guidance_scale=cfg["image_gen"]["guidance_scale"],
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
        f"Best placement: center=({best.center_lat:.4f}, {best.center_lon:.4f}), "
        f"scale={best.scale_km:.1f}km, rotation={best.rotation_deg:.0f}°"
    )

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

    logger.info("Computing bike route...")
    router = create_router(cfg["routing"])
    if hasattr(router, 'route_waypoints_parallel'):
        route = router.route_waypoints_parallel(waypoints)
    else:
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
        from src.evaluation.render import render_polyline, render_map_overlay

        polyline_img = render_polyline(route)
        map_img = render_map_overlay(route)

        poly_buf = io.BytesIO()
        polyline_img.save(poly_buf, format="PNG")
        result["polyline_png"] = poly_buf.getvalue()

        map_buf = io.BytesIO()
        map_img.save(map_buf, format="PNG")
        result["map_png"] = map_buf.getvalue()

        sil_buf = io.BytesIO()
        image.save(sil_buf, format="PNG")
        result["silhouette_png"] = sil_buf.getvalue()

    vol.commit()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    concept: str = "horse",
    output: str = "output.gpx",
    seed: int = None,
    checkpoint: str = None,
    preview: bool = False,
):
    """Generate a GPX route using Modal GPU."""
    print(f"Generating route for '{concept}' on Modal A10G...")

    result = generate_route.remote(
        concept=concept,
        seed=seed,
        checkpoint=checkpoint,
        preview=preview,
    )

    output_path = Path(output)
    output_path.write_bytes(result["gpx_bytes"])
    print(f"GPX saved to {output_path} ({result['num_points']} points)")

    if preview:
        base = output_path.stem
        out_dir = output_path.parent

        if "silhouette_png" in result:
            (out_dir / f"{base}_silhouette.png").write_bytes(result["silhouette_png"])
        if "polyline_png" in result:
            (out_dir / f"{base}_polyline.png").write_bytes(result["polyline_png"])
        if "map_png" in result:
            (out_dir / f"{base}_map.png").write_bytes(result["map_png"])
        print("Preview images saved.")
