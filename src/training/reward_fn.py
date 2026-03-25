"""Reward function wrapper for DDPO training.

This wraps the full pipeline (edge detection → placement → routing → evaluation)
into a callable that the DDPO trainer can use.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.pipeline.edge_detect import ContourSet, process_image, extract_contour_set
from src.pipeline.placement import BBox, grid_search
from src.pipeline.waypoints import (
    sample_adaptive_waypoints, densify_waypoints, sample_multi_contour_waypoints,
)
from src.pipeline.routing import ValhallaRouter
from src.evaluation.reward import RewardFunction

logger = logging.getLogger(__name__)

# Directory for saving sample images during training
TRAIN_SAMPLES_DIR = Path(os.environ.get("TRAIN_SAMPLES_DIR", "training_samples"))


class DDPORewardWrapper:
    """Wraps the full route-sculptor pipeline as a DDPO reward function.

    Given a generated image and the concept that prompted it,
    computes the full pipeline and returns a scalar reward.
    """

    def __init__(
        self,
        bbox: BBox | None = None,
        router: ValhallaRouter | None = None,
        reward_fn: RewardFunction | None = None,
        num_waypoints: int = 80,
        num_positions: int = 10,
        num_scales: int = 5,
        num_rotations: int = 8,
        scale_range_km: tuple[float, float] = (0.3, 5.0),
        canny_low: int = 50,
        canny_high: int = 150,
        curvature_weight: float = 2.0,
        max_gap_km: float = 0.15,
        max_rotation_deg: float = 15.0,
        multi_contour: bool = True,
        min_inner_area_ratio: float = 0.005,
        max_inner_contours: int = 5,
        outer_budget_min: float = 0.6,
        min_inner_waypoints: int = 4,
    ):
        # Default SF bounding box
        self.bbox = bbox or BBox(
            min_lat=37.7080, max_lat=37.8120,
            min_lon=-122.5150, max_lon=-122.3570,
        )
        self.router = router or ValhallaRouter()
        self.reward_fn = reward_fn or RewardFunction()
        self.num_waypoints = num_waypoints
        self.num_positions = num_positions
        self.num_scales = num_scales
        self.num_rotations = num_rotations
        self.scale_range_km = scale_range_km
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.curvature_weight = curvature_weight
        self.max_gap_km = max_gap_km
        self.max_rotation_deg = max_rotation_deg
        self.multi_contour = multi_contour
        self.min_inner_area_ratio = min_inner_area_ratio
        self.max_inner_contours = max_inner_contours
        self.outer_budget_min = outer_budget_min
        self.min_inner_waypoints = min_inner_waypoints

    def __call__(
        self,
        images: list[Image.Image],
        prompts: list[str],
        concepts: list[str],
    ) -> list[dict[str, float]]:
        """Compute rewards for a batch of generated images.

        Args:
            images: List of generated PIL images.
            prompts: List of prompts used to generate (unused but passed by DDPO).
            concepts: List of concept words corresponding to each image.

        Returns:
            List of reward dicts, each with "clip_score", "chamfer_score", "reward".
        """
        results = []
        for image, concept in zip(images, concepts):
            try:
                # DDPO may pass GPU tensors — convert to PIL
                if not isinstance(image, Image.Image):
                    import torch
                    if isinstance(image, torch.Tensor):
                        img_np = image.detach().cpu().float()
                        # Handle (C, H, W) or (H, W, C) formats
                        if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
                            img_np = img_np.permute(1, 2, 0)
                        # Normalize to 0-255
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).clamp(0, 255)
                        image = Image.fromarray(img_np.numpy().astype(np.uint8))
                    else:
                        image = Image.fromarray(np.array(image))
                result = self._compute_single(image, concept)
            except Exception as e:
                logger.error(f"Reward computation failed for '{concept}': {e}")
                result = {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}
            results.append(result)
        return results

    def _compute_single(
        self, image: Image.Image, concept: str
    ) -> dict[str, float]:
        """Compute reward for a single image.

        Returns dict with scores and optionally intermediate images for logging.
        """
        t0 = time.time()

        # Step 1: Edge detection + contour extraction
        edges, contour_data = process_image(
            image, self.canny_low, self.canny_high,
            multi_contour=self.multi_contour,
            min_inner_area_ratio=self.min_inner_area_ratio,
            max_inner_contours=self.max_inner_contours,
        )
        if contour_data is None:
            logger.warning(f"No contour found for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        # Step 2: Grid search for best placement (handles both array and ContourSet)
        placements = grid_search(
            contour_data, self.bbox,
            num_positions=self.num_positions,
            num_scales=self.num_scales,
            num_rotations=self.num_rotations,
            scale_range_km=self.scale_range_km,
            max_rotation_deg=self.max_rotation_deg,
        )
        if not placements:
            logger.warning(f"No valid placements found for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        best_placement, geo_data = placements[0]

        # Step 3: Sample waypoints
        is_multi = isinstance(geo_data, ContourSet)
        if is_multi:
            n_inner = len(geo_data.inner)
            waypoints = sample_multi_contour_waypoints(
                geo_data,
                num_points=self.num_waypoints,
                curvature_weight=self.curvature_weight,
                outer_budget_min=self.outer_budget_min,
                min_inner_waypoints=self.min_inner_waypoints,
                max_gap_km=self.max_gap_km,
            )
        else:
            n_inner = 0
            waypoints = sample_adaptive_waypoints(
                geo_data,
                num_points=self.num_waypoints,
                curvature_weight=self.curvature_weight,
            )
            waypoints = densify_waypoints(waypoints, max_gap_km=self.max_gap_km)

        # Step 4: Route via routing engine (parallel if available)
        if hasattr(self.router, 'route_waypoints_parallel'):
            route = self.router.route_waypoints_parallel(waypoints)
        else:
            route = self.router.route_waypoints(waypoints)

        if len(route) < 2:
            logger.warning(f"Routing failed for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        # Step 5: Compute reward
        result = self.reward_fn.compute(route, concept, edges)

        elapsed = time.time() - t0
        logger.info(
            f"  [{concept}] reward={result['reward']:.3f} "
            f"(clip={result['clip_score']:.3f}, chamfer={result['chamfer_score']:.3f}) "
            f"route_pts={len(route)} inner_contours={n_inner} time={elapsed:.1f}s"
        )

        # Attach intermediate artifacts for W&B logging
        result["_image"] = image
        result["_edges"] = Image.fromarray(edges)
        result["_route"] = route
        result["_concept"] = concept
        result["_num_inner_contours"] = n_inner
        result["_num_waypoints"] = len(waypoints)
        result["_num_route_points"] = len(route)
        result["_placement"] = best_placement

        # Create contour overlay with outer (red) + inner (blue)
        import cv2 as _cv2
        pixel_cs = extract_contour_set(image)
        if pixel_cs is not None:
            vis = np.array(image.copy().convert("RGB"))
            _cv2.drawContours(vis, [pixel_cs.outer], -1, (255, 0, 0), 2)
            for inner_c in pixel_cs.inner:
                _cv2.drawContours(vis, [inner_c], -1, (0, 100, 255), 2)
            result["_contour_overlay"] = Image.fromarray(vis)

        return result
