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
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(
    concept: str,
    output: str,
    bbox: str | None,
    checkpoint: str | None,
    seed: int | None,
    config: str,
    preview: bool,
    verbose: bool,
):
    """Generate a GPX bike route shaped like a concept."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    import yaml
    from src.pipeline.image_gen import ImageGenerator
    from src.pipeline.edge_detect import process_image
    from src.pipeline.placement import grid_search
    from src.pipeline.waypoints import sample_uniform_waypoints
    from src.pipeline.gpx_utils import route_to_gpx, save_gpx
    from src.evaluation.render import render_polyline, render_map_overlay

    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)

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

    # Step 1: Generate image
    logger.info(f"Generating image for concept: '{concept}'")
    gen_cfg = cfg["image_gen"]
    generator = ImageGenerator(
        model_id=checkpoint or gen_cfg["model_id"],
        prompt_template=gen_cfg["prompt_template"],
        image_size=gen_cfg["image_size"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
    )
    image = generator.generate(concept, seed=seed)

    if preview:
        image.save(output.replace(".gpx", "_generated.png"))

    # Step 2: Edge detection
    logger.info("Extracting edges and contour...")
    edge_cfg = cfg["edge_detect"]
    edges, contour = process_image(
        image, edge_cfg["canny_low"], edge_cfg["canny_high"], edge_cfg["blur_kernel"]
    )
    if contour is None:
        click.echo("Error: No contour found in generated image.", err=True)
        raise SystemExit(1)

    # Step 3: Grid search placement
    logger.info("Searching for best placement...")
    p_cfg = cfg["placement"]
    placements = grid_search(
        contour, sf_bbox,
        num_positions=p_cfg["num_positions"],
        num_scales=p_cfg["num_scales"],
        num_rotations=p_cfg["num_rotations"],
        scale_range_km=tuple(p_cfg["scale_range_km"]),
    )
    if not placements:
        click.echo("Error: No valid placements found.", err=True)
        raise SystemExit(1)

    best, geo_contour = placements[0]
    logger.info(
        f"Best placement: center=({best.center_lat:.4f}, {best.center_lon:.4f}), "
        f"scale={best.scale_km:.1f}km, rotation={best.rotation_deg:.0f}°"
    )

    # Step 4: Sample waypoints
    wp_cfg = cfg["waypoints"]
    waypoints = sample_uniform_waypoints(geo_contour, num_points=wp_cfg["num_points"])

    # Step 5: Route via routing engine
    logger.info("Computing bike route...")
    from src.pipeline.routing import create_router

    router = create_router(cfg["routing"])
    route = router.route_waypoints(waypoints)

    if len(route) < 2:
        click.echo("Error: Routing returned insufficient points.", err=True)
        raise SystemExit(1)

    # Step 6: Save GPX
    gpx = route_to_gpx(route, name=f"Route Sculptor: {concept}")
    save_gpx(gpx, output)
    click.echo(f"GPX saved to {output} ({len(route)} points)")

    # Step 7: Optional previews
    if preview:
        polyline_img = render_polyline(route)
        polyline_img.save(output.replace(".gpx", "_polyline.png"))

        map_img = render_map_overlay(route)
        map_img.save(output.replace(".gpx", "_map.png"))

        click.echo("Preview images saved.")


if __name__ == "__main__":
    main()
