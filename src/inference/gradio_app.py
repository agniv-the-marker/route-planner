"""Gradio web UI for Route Sculptor."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def create_app(config_path: str = "configs/default.yaml"):
    """Create and return the Gradio app."""
    import gradio as gr

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Lazy-loaded components (initialized on first use)
    _components = {}

    def _get_components():
        if "generator" not in _components:
            from src.pipeline.image_gen import ImageGenerator
            from src.pipeline.routing import create_router
            from src.evaluation.clip_score import CLIPScorer

            gen_cfg = cfg["image_gen"]
            _components["generator"] = ImageGenerator(
                model_id=cfg.get("inference", {}).get("checkpoint") or gen_cfg["model_id"],
                prompt_template=gen_cfg["prompt_template"],
                image_size=gen_cfg["image_size"],
                num_inference_steps=gen_cfg["num_inference_steps"],
                guidance_scale=gen_cfg["guidance_scale"],
            )

            _components["router"] = create_router(cfg["routing"])

            eval_cfg = cfg["evaluation"]
            _components["clip_scorer"] = CLIPScorer(
                model_id=eval_cfg["clip_model"],
                prompt_template=eval_cfg["clip_prompt_template"],
            )

        return _components

    def generate_route(
        concept: str,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        seed: int | None,
    ):
        """Generate a route for the given concept."""
        from src.pipeline.edge_detect import process_image
        from src.pipeline.placement import BBox, grid_search
        from src.pipeline.waypoints import sample_adaptive_waypoints, densify_waypoints
        from src.pipeline.gpx_utils import route_to_gpx, save_gpx
        from src.evaluation.render import render_polyline, render_map_overlay
        from src.evaluation.chamfer import chamfer_score
        from src.evaluation.render import route_to_edge_image

        components = _get_components()
        generator = components["generator"]
        router = components["router"]
        clip_scorer = components["clip_scorer"]

        bbox = BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

        # Generate image
        seed_val = seed if seed and seed > 0 else None
        image = generator.generate(concept, seed=seed_val)

        # Edge detection
        edge_cfg = cfg["edge_detect"]
        edges, contour = process_image(
            image, edge_cfg["canny_low"], edge_cfg["canny_high"], edge_cfg["blur_kernel"]
        )
        if contour is None:
            return image, None, None, None, "No contour found in generated image."

        # Grid search
        p_cfg = cfg["placement"]
        placements = grid_search(
            contour, bbox,
            num_positions=p_cfg["num_positions"],
            num_scales=p_cfg["num_scales"],
            num_rotations=p_cfg["num_rotations"],
            scale_range_km=tuple(p_cfg["scale_range_km"]),
        )
        if not placements:
            return image, None, None, None, "No valid placements found."

        best, geo_contour = placements[0]

        # Waypoints
        wp_cfg = cfg["waypoints"]
        waypoints = sample_adaptive_waypoints(
            geo_contour,
            num_points=wp_cfg["num_points"],
            curvature_weight=wp_cfg.get("curvature_weight", 2.0),
        )
        waypoints = densify_waypoints(
            waypoints, max_gap_km=wp_cfg.get("max_gap_km", 0.15),
        )

        # Routing
        route = router.route_waypoints(waypoints)

        if len(route) < 2:
            return image, None, None, None, "Routing failed."

        # Render
        polyline_img = render_polyline(route)
        map_img = render_map_overlay(route)

        # Score
        route_edges = route_to_edge_image(route)
        if edges.shape[0] != 512:
            from PIL import Image as PILImage
            ref_img = PILImage.fromarray(edges).resize((512, 512))
            edges_resized = np.array(ref_img)
        else:
            edges_resized = edges

        cham = chamfer_score(edges_resized, route_edges)
        clip = clip_scorer.score(map_img, concept)
        reward = 0.7 * clip + 0.3 * cham

        # Save GPX
        gpx = route_to_gpx(route, name=f"Route Sculptor: {concept}")
        tmp = tempfile.NamedTemporaryFile(suffix=".gpx", delete=False)
        save_gpx(gpx, tmp.name)

        info = (
            f"Placement: ({best.center_lat:.4f}, {best.center_lon:.4f}), "
            f"scale={best.scale_km:.1f}km, rot={best.rotation_deg:.0f}°\n"
            f"CLIP Score: {clip:.4f}\n"
            f"Chamfer Score: {cham:.4f}\n"
            f"Combined Reward: {reward:.4f}\n"
            f"Route Points: {len(route)}"
        )

        return image, polyline_img, map_img, tmp.name, info

    # Build UI
    sf = cfg["sf_bbox"]

    with gr.Blocks(title="Route Sculptor") as app:
        gr.Markdown("# Route Sculptor")
        gr.Markdown("Generate GPX bike routes shaped like concepts using RL-tuned Stable Diffusion.")

        with gr.Row():
            with gr.Column(scale=1):
                concept_input = gr.Textbox(label="Concept", placeholder="e.g., horse")
                seed_input = gr.Number(label="Seed (optional)", value=0, precision=0)

                gr.Markdown("### Bounding Box")
                min_lat = gr.Number(label="Min Latitude", value=sf["min_lat"])
                min_lon = gr.Number(label="Min Longitude", value=sf["min_lon"])
                max_lat = gr.Number(label="Max Latitude", value=sf["max_lat"])
                max_lon = gr.Number(label="Max Longitude", value=sf["max_lon"])

                generate_btn = gr.Button("Generate Route", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    gen_image = gr.Image(label="Generated Silhouette", type="pil")
                    polyline_image = gr.Image(label="Route Polyline", type="pil")
                    map_image = gr.Image(label="Route on Map", type="pil")

                gpx_file = gr.File(label="Download GPX")
                info_text = gr.Textbox(label="Info", lines=5, interactive=False)

        generate_btn.click(
            fn=generate_route,
            inputs=[concept_input, min_lat, min_lon, max_lat, max_lon, seed_input],
            outputs=[gen_image, polyline_image, map_image, gpx_file, info_text],
        )

    return app


def launch(config_path: str = "configs/default.yaml", port: int = 7860):
    """Launch the Gradio app."""
    app = create_app(config_path)
    app.launch(server_port=port)


if __name__ == "__main__":
    launch()
