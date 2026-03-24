"""Combined reward function for RL training."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.evaluation.chamfer import chamfer_score
from src.evaluation.clip_score import CLIPScorer
from src.evaluation.render import render_polyline, render_map_overlay, route_to_edge_image


class RewardFunction:
    """Computes combined CLIP + Chamfer reward for a generated route."""

    def __init__(
        self,
        clip_weight: float = 0.7,
        chamfer_weight: float = 0.3,
        clip_scorer: CLIPScorer | None = None,
        render_size: int = 512,
    ):
        self.clip_weight = clip_weight
        self.chamfer_weight = chamfer_weight
        self.clip_scorer = clip_scorer or CLIPScorer()
        self.render_size = render_size

    def compute(
        self,
        route: list[tuple[float, float]],
        concept: str,
        reference_edges: np.ndarray,
    ) -> dict[str, float]:
        """Compute the full reward for a route.

        Args:
            route: List of (lat, lon) points from OSRM routing.
            concept: The text concept (e.g., "horse").
            reference_edges: Binary edge image from the original generated silhouette.

        Returns:
            Dict with keys: "clip_score", "chamfer_score", "reward".
        """
        # Render the route for evaluation
        map_img = render_map_overlay(route, size=self.render_size)
        route_edges = route_to_edge_image(route, size=self.render_size)

        # CLIP score: does the route look like the concept?
        clip = self.clip_scorer.score(map_img, concept)

        # Chamfer score: does the route outline match the reference?
        # Resize reference edges to match render size if needed
        if reference_edges.shape[0] != self.render_size:
            from PIL import Image as PILImage
            ref_img = PILImage.fromarray(reference_edges)
            ref_img = ref_img.resize((self.render_size, self.render_size))
            reference_edges = np.array(ref_img)

        cham = chamfer_score(reference_edges, route_edges)

        reward = self.clip_weight * clip + self.chamfer_weight * cham

        return {
            "clip_score": clip,
            "chamfer_score": cham,
            "reward": reward,
        }
