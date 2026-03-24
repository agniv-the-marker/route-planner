#!/usr/bin/env python3
"""Demo: Run the full pipeline with a synthetic silhouette image (no GPU needed).

This skips SD 1.5 and instead creates a programmatic silhouette,
then runs through edge detection → placement → waypoints → OSRM routing → GPX.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("demo")


def create_star_silhouette(size: int = 512) -> Image.Image:
    """Create a programmatic star silhouette (no GPU needed)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    outer_r = size * 0.4
    inner_r = size * 0.18
    points = []
    for i in range(10):
        angle = np.radians(i * 36 - 90)
        r = outer_r if i % 2 == 0 else inner_r
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append((x, y))

    draw.polygon(points, fill=(0, 0, 0))
    return img


def create_heart_silhouette(size: int = 512) -> Image.Image:
    """Create a programmatic heart silhouette."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    t = np.linspace(0, 2 * np.pi, 500)
    # Heart parametric equations
    scale = size * 0.025
    x = scale * 16 * np.sin(t) ** 3 + cx
    y = -scale * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) + cy

    points = list(zip(x.tolist(), y.tolist()))
    draw.polygon(points, fill=(0, 0, 0))
    return img


def create_horse_silhouette(size: int = 512) -> Image.Image:
    """Create a rough horse-like silhouette using basic shapes."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Simple horse-like polygon
    s = size / 512
    points = [
        (180*s, 350*s), (160*s, 280*s), (140*s, 220*s), (130*s, 170*s),
        (150*s, 120*s), (180*s, 80*s), (200*s, 60*s), (220*s, 50*s),
        (240*s, 55*s), (250*s, 70*s), (245*s, 90*s), (260*s, 100*s),
        (280*s, 110*s), (300*s, 130*s), (320*s, 160*s), (340*s, 200*s),
        (350*s, 250*s), (355*s, 300*s), (360*s, 350*s), (370*s, 400*s),
        (365*s, 430*s), (340*s, 430*s), (335*s, 380*s), (320*s, 350*s),
        (300*s, 340*s), (280*s, 350*s), (260*s, 380*s), (250*s, 430*s),
        (230*s, 430*s), (225*s, 390*s), (220*s, 360*s), (200*s, 360*s),
    ]
    draw.polygon(points, fill=(0, 0, 0))
    return img


SHAPES = {
    "star": create_star_silhouette,
    "heart": create_heart_silhouette,
    "horse": create_horse_silhouette,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo the full pipeline without GPU")
    parser.add_argument("--concept", default="star", choices=list(SHAPES.keys()))
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--skip-routing", action="store_true",
                        help="Skip OSRM routing (for offline testing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    concept = args.concept
    logger.info(f"=== Route Sculptor Demo: '{concept}' ===")

    # Step 1: Create synthetic silhouette
    logger.info("Step 1: Creating synthetic silhouette...")
    image = SHAPES[concept]()
    image.save(output_dir / f"{concept}_silhouette.png")
    logger.info(f"  Saved silhouette to {output_dir / f'{concept}_silhouette.png'}")

    # Step 2: Edge detection
    logger.info("Step 2: Running Canny edge detection...")
    from src.pipeline.edge_detect import process_image

    edge_cfg = cfg["edge_detect"]
    edges, contour = process_image(
        image, edge_cfg["canny_low"], edge_cfg["canny_high"], edge_cfg["blur_kernel"]
    )
    if contour is None:
        logger.error("No contour found!")
        sys.exit(1)
    logger.info(f"  Extracted contour with {len(contour)} points")

    # Save edge image
    edge_img = Image.fromarray(edges)
    edge_img.save(output_dir / f"{concept}_edges.png")

    # Step 3: Grid search placement
    logger.info("Step 3: Grid search for best placement on SF map...")
    from src.pipeline.placement import BBox, grid_search

    sf = cfg["sf_bbox"]
    bbox = BBox(**sf)
    p_cfg = cfg["placement"]

    placements = grid_search(
        contour, bbox,
        num_positions=p_cfg["num_positions"],
        num_scales=p_cfg["num_scales"],
        num_rotations=p_cfg["num_rotations"],
        scale_range_km=tuple(p_cfg["scale_range_km"]),
    )
    logger.info(f"  Found {len(placements)} valid placements")

    if not placements:
        logger.error("No valid placements!")
        sys.exit(1)

    best, geo_contour = placements[0]
    logger.info(
        f"  Best: center=({best.center_lat:.4f}, {best.center_lon:.4f}), "
        f"scale={best.scale_km:.1f}km, rot={best.rotation_deg:.0f}°, "
        f"score={best.score:.4f}"
    )

    # Step 4: Sample waypoints
    logger.info("Step 4: Sampling waypoints...")
    from src.pipeline.waypoints import sample_uniform_waypoints

    wp_cfg = cfg["waypoints"]
    waypoints = sample_uniform_waypoints(geo_contour, num_points=wp_cfg["num_points"])
    logger.info(f"  Sampled {len(waypoints)} waypoints")

    # Step 5: OSRM Routing
    if args.skip_routing:
        logger.info("Step 5: Skipping OSRM routing (--skip-routing)")
        route = [(float(w[0]), float(w[1])) for w in waypoints]
    else:
        logger.info("Step 5: Routing via OSRM (this may take a minute)...")
        from src.pipeline.routing import OSRMRouter

        r_cfg = cfg["routing"]
        router = OSRMRouter(
            base_url=r_cfg["base_url"],
            profile=r_cfg["profile"],
            request_delay=r_cfg["request_delay"],
        )
        start_time = time.time()
        route = router.route_waypoints(waypoints)
        elapsed = time.time() - start_time
        logger.info(f"  Routed to {len(route)} points in {elapsed:.1f}s")

    # Step 6: Save GPX
    logger.info("Step 6: Saving GPX...")
    from src.pipeline.gpx_utils import route_to_gpx, save_gpx

    gpx = route_to_gpx(route, name=f"Route Sculptor Demo: {concept}")
    gpx_path = output_dir / f"{concept}_route.gpx"
    save_gpx(gpx, gpx_path)
    logger.info(f"  Saved GPX to {gpx_path}")

    # Step 7: Render previews
    logger.info("Step 7: Rendering previews...")
    from src.evaluation.render import render_polyline, render_map_overlay

    polyline_img = render_polyline(route)
    polyline_img.save(output_dir / f"{concept}_polyline.png")

    map_img = render_map_overlay(route)
    map_img.save(output_dir / f"{concept}_map.png")

    # Step 8: Compute Chamfer score (CLIP requires GPU-heavy model, skip in demo)
    logger.info("Step 8: Computing Chamfer distance...")
    from src.evaluation.render import route_to_edge_image
    from src.evaluation.chamfer import chamfer_score

    route_edges = route_to_edge_image(route)
    # Resize reference edges to match
    ref_img = Image.fromarray(edges).resize((512, 512))
    ref_edges = np.array(ref_img)

    cham = chamfer_score(ref_edges, route_edges)
    logger.info(f"  Chamfer score: {cham:.4f}")

    logger.info("")
    logger.info("=== Demo Complete ===")
    logger.info(f"Output files in: {output_dir}/")
    logger.info(f"  {concept}_silhouette.png  - Input silhouette")
    logger.info(f"  {concept}_edges.png       - Canny edges")
    logger.info(f"  {concept}_polyline.png    - Route outline")
    logger.info(f"  {concept}_map.png         - Route on map")
    logger.info(f"  {concept}_route.gpx       - GPX file")
    logger.info(f"  Chamfer score: {cham:.4f}")


if __name__ == "__main__":
    main()
