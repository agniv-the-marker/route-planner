"""CLI for generating GPX routes from concepts."""

from __future__ import annotations

import logging

import click

from src.pipeline.placement import BBox

logger = logging.getLogger(__name__)


@click.command()
@click.option("--concept", required=True, help="Text concept (e.g., 'horse')")
@click.option("--output", default="output.gpx", help="Output GPX file path")
@click.option(
    "--bbox",
    default=None,
    help="Bounding box as 'min_lat,min_lon,max_lat,max_lon'. Defaults to all of SF.",
)
@click.option("--checkpoint", default=None, help="Path to trained model checkpoint")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--config", default="configs/default.yaml", help="Config file path")
@click.option("--preview", is_flag=True, help="Save preview images alongside GPX")
@click.option("--multi-contour/--no-multi-contour", default=None,
              help="Include internal details (eyes, etc). Default: from config.")
@click.option("--use-svg/--use-sd", default=True,
              help="Use pre-made SVG silhouettes (default) or SD 1.5 generation.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(
    concept: str,
    output: str,
    bbox: str | None,
    checkpoint: str | None,
    seed: int | None,
    config: str,
    preview: bool,
    multi_contour: bool | None,
    use_svg: bool,
    verbose: bool,
):
    """Generate a GPX bike route shaped like a concept."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    import yaml
    from src.pipeline.image_gen import ImageGenerator
    from src.pipeline.edge_detect import ContourSet, process_image
    from src.pipeline.placement import grid_search
    from src.pipeline.waypoints import (
        sample_adaptive_waypoints, densify_waypoints, sample_multi_contour_waypoints,
    )
    from src.pipeline.gpx_utils import route_to_gpx, save_gpx
    from src.evaluation.render import render_polyline, render_map_overlay

    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Resolve multi-contour: CLI flag overrides config
    mc_cfg = cfg.get("multi_contour", {})
    use_multi = multi_contour if multi_contour is not None else mc_cfg.get("enabled", True)

    # Parse bounding box
    if bbox:
        parts = [float(x) for x in bbox.split(",")]
        sf_bbox = BBox(
            min_lat=parts[0], min_lon=parts[1],
            max_lat=parts[2], max_lon=parts[3],
        )
    else:
        sf = cfg["sf_bbox"]
        sf_bbox = BBox(**sf)

    # Step 1: Get silhouette image
    from src.pipeline.image_gen import load_silhouette
    image = None
    if use_svg:
        image = load_silhouette(concept)
        if image:
            logger.info(f"Loaded pre-made silhouette for '{concept}'")
        else:
            logger.info(f"No pre-made silhouette for '{concept}', falling back to SD 1.5")

    if image is None:
        logger.info(f"Generating image for concept: '{concept}' with SD 1.5")
        gen_cfg = cfg["image_gen"]
        generator = ImageGenerator(
            model_id=checkpoint or gen_cfg["model_id"],
            prompt_template=gen_cfg["prompt_template"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            image_size=gen_cfg["image_size"],
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
        )
        image = generator.generate(concept, seed=seed)

    # Set up structured output directory: outputs/{concept}/
    from pathlib import Path as _Path
    output_path = _Path(output)
    if output_path.suffix == ".gpx":
        # Old-style single file path → convert to structured dir
        concept_dir = output_path.parent / concept
    else:
        concept_dir = output_path / concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    if preview:
        image.save(concept_dir / "silhouette.png")

    # Step 2: Edge detection + contour extraction
    logger.info("Extracting edges and contour...")
    edge_cfg = cfg["edge_detect"]
    edges, contour_data = process_image(
        image, edge_cfg["canny_low"], edge_cfg["canny_high"], edge_cfg["blur_kernel"],
        multi_contour=use_multi,
        min_inner_area_ratio=mc_cfg.get("min_inner_area_ratio", 0.005),
        max_inner_contours=mc_cfg.get("max_inner_contours", 5),
    )
    if contour_data is None:
        click.echo("Error: No contour found in generated image.", err=True)
        raise SystemExit(1)

    is_multi = isinstance(contour_data, ContourSet)
    if is_multi:
        logger.info(f"Found {len(contour_data.inner)} inner contours")

    # Step 3: Grid search placement
    logger.info("Searching for best placement...")
    p_cfg = cfg["placement"]
    placements = grid_search(
        contour_data, sf_bbox,
        num_positions=p_cfg["num_positions"],
        num_scales=p_cfg["num_scales"],
        num_rotations=p_cfg["num_rotations"],
        scale_range_km=tuple(p_cfg["scale_range_km"]),
    )
    if not placements:
        click.echo("Error: No valid placements found.", err=True)
        raise SystemExit(1)

    best, geo_data = placements[0]
    logger.info(
        f"Best placement: center=({best.center_lat:.4f}, {best.center_lon:.4f}), "
        f"scale={best.scale_km:.1f}km, rotation={best.rotation_deg:.0f}°"
    )

    # Step 4: Snap contour to road graph + sample waypoints
    from src.pipeline.routing import create_router
    router = create_router(cfg["routing"])

    # If graph router available, snap contour to roads
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
            waypoints = densify_waypoints(
                waypoints, max_gap_km=wp_cfg.get("max_gap_km", 0.15),
            )

    # Step 5: Route via routing engine
    logger.info("Computing bike route...")
    route = router.route_waypoints(waypoints)

    if len(route) < 2:
        click.echo("Error: Routing returned insufficient points.", err=True)
        raise SystemExit(1)

    # Step 6: Save GPX + previews to structured directory
    gpx = route_to_gpx(route, name=f"Route Sculptor: {concept}")
    gpx_path = concept_dir / "route.gpx"
    save_gpx(gpx, str(gpx_path))
    click.echo(f"GPX saved to {gpx_path} ({len(route)} points)")

    if preview:
        from PIL import Image as PILImage
        import cv2 as _cv2
        from src.pipeline.edge_detect import extract_contour_set

        # Save edges
        PILImage.fromarray(edges).save(concept_dir / "edges.png")

        # Save contour overlay
        pixel_cs = extract_contour_set(image)
        if pixel_cs is not None:
            import numpy as _np
            vis = _np.array(image.copy().convert("RGB"))
            _cv2.drawContours(vis, [pixel_cs.outer], -1, (255, 0, 0), 2)
            for inner_c in pixel_cs.inner:
                _cv2.drawContours(vis, [inner_c], -1, (0, 100, 255), 2)
            PILImage.fromarray(vis).save(concept_dir / "contour.png")

        # Save route renders
        polyline_img = render_polyline(route)
        polyline_img.save(concept_dir / "polyline.png")

        map_img = render_map_overlay(route)
        map_img.save(concept_dir / "map_overlay.png")

        # Save metadata
        import json
        metadata = {
            "concept": concept,
            "placement": {
                "center_lat": best.center_lat,
                "center_lon": best.center_lon,
                "scale_km": best.scale_km,
                "rotation_deg": best.rotation_deg,
                "score": best.score,
            },
            "route_points": len(route),
            "waypoints": len(waypoints),
            "inner_contours": len(contour_data.inner) if is_multi else 0,
        }
        with open(concept_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        click.echo(f"All outputs saved to {concept_dir}/")


if __name__ == "__main__":
    main()
