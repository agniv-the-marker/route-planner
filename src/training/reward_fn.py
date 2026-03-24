"""Reward function wrapper for DDPO training.

This wraps the full pipeline (edge detection → placement → routing → evaluation)
into a callable that the DDPO trainer can use.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from src.pipeline.edge_detect import process_image
from src.pipeline.placement import BBox, grid_search
from src.pipeline.waypoints import sample_uniform_waypoints
from src.pipeline.routing import ValhallaRouter
from src.evaluation.reward import RewardFunction

logger = logging.getLogger(__name__)


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
        num_waypoints: int = 65,
        num_positions: int = 10,
        num_scales: int = 5,
        num_rotations: int = 8,
        scale_range_km: tuple[float, float] = (0.3, 5.0),
        canny_low: int = 50,
        canny_high: int = 150,
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
                result = self._compute_single(image, concept)
            except Exception as e:
                logger.error(f"Reward computation failed for '{concept}': {e}")
                result = {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}
            results.append(result)
        return results

    def _compute_single(
        self, image: Image.Image, concept: str
    ) -> dict[str, float]:
        """Compute reward for a single image."""
        # Step 1: Edge detection
        edges, contour = process_image(
            image, self.canny_low, self.canny_high
        )
        if contour is None:
            logger.warning(f"No contour found for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        # Step 2: Grid search for best placement
        placements = grid_search(
            contour, self.bbox,
            num_positions=self.num_positions,
            num_scales=self.num_scales,
            num_rotations=self.num_rotations,
            scale_range_km=self.scale_range_km,
        )
        if not placements:
            logger.warning(f"No valid placements found for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        # Take the best placement
        best_placement, geo_contour = placements[0]

        # Step 3: Sample waypoints
        waypoints = sample_uniform_waypoints(
            geo_contour, num_points=self.num_waypoints
        )

        # Step 4: Route via OSRM
        route = self.router.route_waypoints(waypoints)

        if len(route) < 2:
            logger.warning(f"Routing failed for '{concept}'")
            return {"clip_score": 0.0, "chamfer_score": 0.0, "reward": 0.0}

        # Step 5: Compute reward
        return self.reward_fn.compute(route, concept, edges)
