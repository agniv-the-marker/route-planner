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
    """Create a recognisable horse silhouette (side view, facing left)
    as a single filled outline with proper equine proportions."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    s = size / 512

    # Single outline traced clockwise starting at muzzle.
    # Horse facing left, standing, with head up.
    points = [
        # Muzzle & lower lip
        (55*s, 105*s), (48*s, 100*s), (40*s, 92*s), (38*s, 82*s),
        # Nose bridge up to forehead
        (42*s, 72*s), (50*s, 62*s), (62*s, 54*s), (76*s, 48*s),
        # Ear 1
        (88*s, 44*s), (86*s, 26*s), (96*s, 36*s),
        # Ear 2
        (104*s, 32*s), (112*s, 22*s), (114*s, 40*s),
        # Poll, crest of neck (mane side)
        (120*s, 50*s), (132*s, 62*s), (148*s, 82*s),
        (168*s, 108*s), (188*s, 138*s), (205*s, 162*s),
        # Withers (highest point of back)
        (218*s, 172*s), (228*s, 168*s), (238*s, 166*s),
        # Back (slightly dipped then rising to croup)
        (260*s, 170*s), (285*s, 172*s), (310*s, 170*s),
        (335*s, 168*s), (355*s, 172*s),
        # Croup / top of rump
        (370*s, 180*s), (380*s, 192*s),
        # Tail
        (390*s, 190*s), (410*s, 178*s), (430*s, 175*s),
        (448*s, 182*s), (460*s, 200*s), (465*s, 225*s),
        (460*s, 255*s), (448*s, 278*s), (432*s, 290*s),
        (418*s, 288*s), (410*s, 275*s), (405*s, 255*s),
        (398*s, 232*s), (392*s, 212*s),
        # Buttock descending
        (385*s, 220*s), (380*s, 240*s), (375*s, 262*s),
        (370*s, 280*s), (365*s, 295*s),
        # Hind-right thigh → hock → cannon → hoof
        (368*s, 315*s), (372*s, 340*s), (375*s, 365*s),
        (374*s, 390*s), (370*s, 415*s), (368*s, 438*s),
        (366*s, 452*s),
        # Hind-right hoof
        (376*s, 458*s), (382*s, 455*s), (384*s, 448*s),
        # Hind-right leg back side up to stifle
        (382*s, 425*s), (378*s, 400*s), (376*s, 378*s),
        (378*s, 358*s), (380*s, 340*s),
        # Gap between hind legs
        (375*s, 325*s), (365*s, 312*s),
        # Hind-left thigh
        (355*s, 320*s), (348*s, 342*s), (342*s, 365*s),
        (338*s, 390*s), (334*s, 415*s), (332*s, 438*s),
        (330*s, 452*s),
        # Hind-left hoof
        (340*s, 458*s), (348*s, 455*s), (350*s, 448*s),
        # Hind-left leg back side
        (348*s, 425*s), (345*s, 402*s), (340*s, 378*s),
        (335*s, 355*s), (328*s, 335*s), (318*s, 318*s),
        # Belly (curving forward under body)
        (300*s, 308*s), (275*s, 312*s), (250*s, 314*s),
        (225*s, 312*s), (205*s, 308*s),
        # Front-right upper leg
        (200*s, 315*s), (198*s, 335*s), (195*s, 358*s),
        (192*s, 382*s), (188*s, 408*s), (186*s, 432*s),
        (184*s, 452*s),
        # Front-right hoof
        (194*s, 458*s), (202*s, 455*s), (204*s, 448*s),
        # Front-right leg back side
        (202*s, 428*s), (200*s, 405*s), (202*s, 382*s),
        (206*s, 358*s), (210*s, 338*s), (215*s, 320*s),
        # Gap between front legs
        (210*s, 310*s), (200*s, 305*s),
        # Front-left upper leg
        (188*s, 310*s), (180*s, 330*s), (174*s, 355*s),
        (168*s, 380*s), (163*s, 408*s), (160*s, 432*s),
        (158*s, 452*s),
        # Front-left hoof
        (168*s, 458*s), (176*s, 455*s), (178*s, 448*s),
        # Front-left leg back side
        (176*s, 428*s), (175*s, 405*s), (178*s, 382*s),
        (182*s, 358*s), (185*s, 335*s), (188*s, 315*s),
        # Chest rising
        (182*s, 298*s), (170*s, 278*s), (158*s, 258*s),
        # Throat (up to jaw)
        (145*s, 238*s), (130*s, 215*s), (115*s, 192*s),
        (100*s, 168*s), (88*s, 148*s), (78*s, 130*s),
        # Jaw line → back to muzzle
        (70*s, 118*s), (62*s, 112*s),
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

    # Step 4: Sample waypoints (adaptive: more points at sharp corners)
    logger.info("Step 4: Sampling waypoints (curvature-adaptive)...")
    from src.pipeline.waypoints import sample_adaptive_waypoints, densify_waypoints

    wp_cfg = cfg["waypoints"]
    waypoints = sample_adaptive_waypoints(
        geo_contour,
        num_points=wp_cfg["num_points"],
        curvature_weight=wp_cfg.get("curvature_weight", 2.0),
    )
    logger.info(f"  Sampled {len(waypoints)} waypoints")

    # Densify: ensure no gap > max_gap_km so the router follows roads naturally
    max_gap = wp_cfg.get("max_gap_km", 0.15)
    waypoints = densify_waypoints(waypoints, max_gap_km=max_gap)
    logger.info(f"  Densified to {len(waypoints)} waypoints (max gap {max_gap*1000:.0f}m)")

    # Step 5: Routing
    if args.skip_routing:
        logger.info("Step 5: Skipping routing (--skip-routing)")
        route = [(float(w[0]), float(w[1])) for w in waypoints]
    else:
        logger.info("Step 5: Routing via Valhalla (this may take a minute)...")
        from src.pipeline.routing import ValhallaRouter

        router = ValhallaRouter(
            costing="bicycle",
            request_delay=1.0,
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
