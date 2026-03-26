#!/usr/bin/env python3
"""Batch generate routes for multiple concepts.

Usage:
    PYTHONPATH=. python scripts/batch_generate.py horse dog cat star heart
    PYTHONPATH=. python scripts/batch_generate.py --all          # all concepts from data/concepts.txt
    PYTHONPATH=. python scripts/batch_generate.py --svg-only     # only concepts with pre-made silhouettes
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("batch")


def main():
    parser = argparse.ArgumentParser(description="Batch generate routes")
    parser.add_argument("concepts", nargs="*", help="Concepts to generate")
    parser.add_argument("--all", action="store_true", help="Use all concepts from data/concepts.txt")
    parser.add_argument("--svg-only", action="store_true", help="Only concepts with pre-made silhouettes")
    parser.add_argument("--use-sd", action="store_true", help="Force SD 1.5 generation (ignore pre-made SVGs)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Determine concepts
    if args.all:
        from src.training.dataset import load_concepts
        concepts = load_concepts()
    elif args.svg_only:
        svg_dir = Path("data/silhouettes")
        concepts = [p.stem for p in sorted(svg_dir.glob("*.png"))]
    elif args.concepts:
        concepts = args.concepts
    else:
        parser.error("Provide concept names, --all, or --svg-only")

    logger.info(f"Generating {len(concepts)} concepts: {', '.join(concepts)}")

    # Import pipeline components
    from src.pipeline.image_gen import ImageGenerator, load_silhouette
    from src.pipeline.edge_detect import ContourSet, process_image, extract_contour_set
    from src.pipeline.placement import BBox, grid_search
    from src.pipeline.waypoints import (
        sample_adaptive_waypoints, densify_waypoints, sample_multi_contour_waypoints,
    )
    from src.pipeline.routing import create_router
    from src.pipeline.gpx_utils import route_to_gpx, save_gpx
    from src.evaluation.render import render_polyline, render_map_overlay
    import json
    import numpy as np
    import cv2
    from PIL import Image

    # Load config sections
    bbox = BBox(**cfg["sf_bbox"])
    p_cfg = cfg["placement"]
    wp_cfg = cfg["waypoints"]
    mc_cfg = cfg.get("multi_contour", {})
    edge_cfg = cfg["edge_detect"]

    # Initialize router once (graph loading is expensive)
    router = create_router(cfg["routing"])

    # Optional: lazy-load SD generator
    sd_generator = None

    results = []

    for i, concept in enumerate(concepts):
        t0 = time.time()
        concept_dir = Path(args.output_dir) / concept
        concept_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Image
            image = None if args.use_sd else load_silhouette(concept)
            source = "svg" if image else "sd"
            if image is None:
                if sd_generator is None:
                    gen_cfg = cfg["image_gen"]
                    sd_generator = ImageGenerator(
                        model_id=gen_cfg["model_id"],
                        prompt_template=gen_cfg["prompt_template"],
                        negative_prompt=gen_cfg.get("negative_prompt", ""),
                        image_size=gen_cfg["image_size"],
                        num_inference_steps=gen_cfg["num_inference_steps"],
                        guidance_scale=gen_cfg["guidance_scale"],
                    )
                image = sd_generator.generate(concept, seed=args.seed)
                source = "sd"

            image.save(concept_dir / "silhouette.png")

            # Step 2: Contour
            edges, contour_data = process_image(
                image, edge_cfg["canny_low"], edge_cfg["canny_high"],
                edge_cfg["blur_kernel"], multi_contour=mc_cfg.get("enabled", True),
            )
            if contour_data is None:
                raise RuntimeError("No contour found")

            Image.fromarray(edges).save(concept_dir / "edges.png")

            # Contour overlay
            pixel_cs = extract_contour_set(image)
            if pixel_cs is not None:
                vis = np.array(image.copy().convert("RGB"))
                cv2.drawContours(vis, [pixel_cs.outer], -1, (255, 0, 0), 2)
                for c in pixel_cs.inner:
                    cv2.drawContours(vis, [c], -1, (0, 100, 255), 2)
                Image.fromarray(vis).save(concept_dir / "contour.png")

            is_multi = isinstance(contour_data, ContourSet)
            n_inner = len(contour_data.inner) if is_multi else 0

            # Step 3: Placement
            placements = grid_search(
                contour_data, bbox,
                num_positions=p_cfg["num_positions"], num_scales=p_cfg["num_scales"],
                num_rotations=p_cfg["num_rotations"],
                scale_range_km=tuple(p_cfg["scale_range_km"]),
                max_rotation_deg=p_cfg.get("max_rotation_deg", 15.0),
            )
            if not placements:
                raise RuntimeError("No valid placements")

            best, geo_data = placements[0]

            # Step 4: Waypoints
            if is_multi and isinstance(geo_data, ContourSet):
                waypoints = sample_multi_contour_waypoints(
                    geo_data, num_points=wp_cfg["num_points"],
                    curvature_weight=wp_cfg.get("curvature_weight", 2.0),
                    max_gap_km=wp_cfg.get("max_gap_km", 0.15),
                )
            else:
                waypoints = sample_adaptive_waypoints(
                    geo_data, num_points=wp_cfg["num_points"],
                    curvature_weight=wp_cfg.get("curvature_weight", 2.0),
                )
                waypoints = densify_waypoints(waypoints, max_gap_km=wp_cfg.get("max_gap_km", 0.15))

            # Step 5: Route
            route = router.route_waypoints(waypoints)
            if len(route) < 2:
                raise RuntimeError("Routing failed")

            # Step 6: Save
            gpx = route_to_gpx(route, name=f"Route Sculptor: {concept}")
            save_gpx(gpx, str(concept_dir / "route.gpx"))

            render_polyline(route).save(concept_dir / "polyline.png")
            render_map_overlay(route).save(concept_dir / "map_overlay.png")

            elapsed = time.time() - t0

            metadata = {
                "concept": concept, "source": source,
                "placement": {"center_lat": best.center_lat, "center_lon": best.center_lon,
                              "scale_km": best.scale_km, "rotation_deg": best.rotation_deg},
                "route_points": len(route), "waypoints": len(waypoints),
                "inner_contours": n_inner, "time_seconds": round(elapsed, 1),
            }
            with open(concept_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            results.append({"concept": concept, "status": "ok", "points": len(route),
                            "inner": n_inner, "time": elapsed, "source": source})
            logger.info(f"[{i+1}/{len(concepts)}] {concept}: {len(route)} pts, {n_inner} inner, {elapsed:.1f}s ({source})")

        except Exception as e:
            elapsed = time.time() - t0
            results.append({"concept": concept, "status": "error", "error": str(e), "time": elapsed})
            logger.error(f"[{i+1}/{len(concepts)}] {concept}: FAILED - {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Concept':<15} {'Status':<8} {'Points':<8} {'Inner':<6} {'Time':<8} {'Source'}")
    print(f"{'-'*60}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['concept']:<15} {'OK':<8} {r['points']:<8} {r['inner']:<6} {r['time']:<8.1f} {r['source']}")
        else:
            print(f"{r['concept']:<15} {'FAIL':<8} {'-':<8} {'-':<6} {r['time']:<8.1f} {r['error'][:30]}")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"{ok}/{len(results)} succeeded. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
