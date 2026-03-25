#!/usr/bin/env python3
"""Preview silhouettes locally on CPU before committing to a training run.

Usage:
    PYTHONPATH=. python scripts/preview_silhouettes.py horse dog star bell sword
    PYTHONPATH=. python scripts/preview_silhouettes.py --all  # preview all concepts
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("preview")


def main():
    parser = argparse.ArgumentParser(description="Preview SD 1.5 silhouettes locally")
    parser.add_argument("concepts", nargs="*", help="Concepts to preview")
    parser.add_argument("--all", action="store_true", help="Preview all concepts from data/concepts.txt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/previews")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (min 20 for quality)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.all:
        from src.training.dataset import load_concepts
        concepts = load_concepts()
    elif args.concepts:
        concepts = args.concepts
    else:
        parser.error("Provide concept names or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SD 1.5
    logger.info("Loading Stable Diffusion 1.5 (this takes a minute on CPU)...")
    from src.pipeline.image_gen import ImageGenerator

    gen_cfg = cfg["image_gen"]
    generator = ImageGenerator(
        model_id=gen_cfg["model_id"],
        prompt_template=gen_cfg["prompt_template"],
        negative_prompt=gen_cfg.get("negative_prompt", ""),
        image_size=gen_cfg["image_size"],
        num_inference_steps=max(args.steps, gen_cfg["num_inference_steps"]),
        guidance_scale=gen_cfg["guidance_scale"],
    )

    from src.pipeline.edge_detect import (
        process_image, extract_contour_set, ContourSet,
    )
    import cv2

    for i, concept in enumerate(concepts):
        logger.info(f"[{i+1}/{len(concepts)}] Generating '{concept}'...")

        image = generator.generate(concept, seed=args.seed)
        image.save(output_dir / f"{concept}_silhouette.png")

        # Edge detection
        edge_cfg = cfg["edge_detect"]
        mc_cfg = cfg.get("multi_contour", {})
        edges, contour_data = process_image(
            image, edge_cfg["canny_low"], edge_cfg["canny_high"],
            edge_cfg["blur_kernel"],
            multi_contour=mc_cfg.get("enabled", True),
        )

        # Save edges
        Image.fromarray(edges).save(output_dir / f"{concept}_edges.png")

        # Save contour overlay (outer=red, inner=blue)
        pixel_cs = extract_contour_set(image)
        if pixel_cs is not None:
            vis = np.array(image.copy().convert("RGB"))
            cv2.drawContours(vis, [pixel_cs.outer], -1, (255, 0, 0), 2)
            for inner_c in pixel_cs.inner:
                cv2.drawContours(vis, [inner_c], -1, (0, 100, 255), 2)
            Image.fromarray(vis).save(output_dir / f"{concept}_contour.png")

            n_inner = len(pixel_cs.inner)
            logger.info(
                f"  Saved: silhouette, edges, contour "
                f"(outer={len(pixel_cs.outer)} pts, inner={n_inner} contours)"
            )
        else:
            logger.warning(f"  No contour found for '{concept}'")

    logger.info(f"\nAll previews saved to {output_dir}/")
    logger.info("Review the images, then run training when satisfied with the silhouettes.")


if __name__ == "__main__":
    main()
