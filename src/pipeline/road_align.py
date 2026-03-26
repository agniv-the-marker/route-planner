"""Road-aware contour snapping using OSM graph.

Snaps geographic contour points to the nearest road network nodes
so the resulting route follows actual streets by construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class RoadAligner:
    """Snaps contour points to the nearest road graph nodes."""

    def __init__(self, graph, max_snap_km: float = 0.3):
        self.G = graph
        self.max_snap_km = max_snap_km

        # Build spatial index of all graph nodes
        self.node_ids = list(graph.nodes)
        self.node_coords = np.array([
            (graph.nodes[n]["y"], graph.nodes[n]["x"])
            for n in self.node_ids
        ])
        self.tree = cKDTree(self.node_coords)

        logger.info(f"RoadAligner: {len(self.node_ids)} nodes, max_snap={max_snap_km}km")

    def snap_contour(
        self,
        geo_contour: np.ndarray,
        subsample: int = 3,
    ) -> np.ndarray:
        """Snap contour points to nearest road nodes.

        Simple approach: for each point, pick the nearest graph node
        within max_snap_km. Skip points with no nearby node (water/park).
        Deduplicate consecutive identical nodes.

        Args:
            geo_contour: (N, 2) array of (lat, lon)
            subsample: Only snap every Nth point to reduce density

        Returns:
            (M, 2) array of snapped (lat, lon) on the road graph
        """
        # Subsample
        indices = list(range(0, len(geo_contour), subsample))
        if indices[-1] != len(geo_contour) - 1:
            indices.append(len(geo_contour) - 1)
        points = geo_contour[indices]

        snapped = []
        prev_node_id = None

        for lat, lon in points:
            dist, idx = self.tree.query([lat, lon])
            dist_km = dist * 111.0  # rough degree-to-km

            if dist_km > self.max_snap_km:
                continue  # no nearby road — skip (water/park)

            node_id = self.node_ids[idx]

            # Deduplicate consecutive identical nodes
            if node_id == prev_node_id:
                continue

            nlat = self.G.nodes[node_id]["y"]
            nlon = self.G.nodes[node_id]["x"]
            snapped.append((nlat, nlon))
            prev_node_id = node_id

        if not snapped:
            logger.warning("No road nodes found near contour — returning original")
            return geo_contour

        # Densify: if consecutive snapped nodes are >200m apart, insert midpoints
        densified = [snapped[0]]
        for i in range(1, len(snapped)):
            prev = snapped[i - 1]
            curr = snapped[i]
            dist_km = np.sqrt(
                ((curr[0] - prev[0]) * 111.0) ** 2 +
                ((curr[1] - prev[1]) * 111.0 * np.cos(np.radians(prev[0]))) ** 2
            )
            if dist_km > 0.2:  # >200m gap
                n_inserts = int(dist_km / 0.15)  # insert every ~150m
                for j in range(1, n_inserts + 1):
                    t = j / (n_inserts + 1)
                    mid_lat = prev[0] + t * (curr[0] - prev[0])
                    mid_lon = prev[1] + t * (curr[1] - prev[1])
                    # Snap the midpoint to nearest node too
                    d, idx = self.tree.query([mid_lat, mid_lon])
                    if d * 111.0 <= self.max_snap_km:
                        node_id = self.node_ids[idx]
                        densified.append((
                            self.G.nodes[node_id]["y"],
                            self.G.nodes[node_id]["x"],
                        ))
            densified.append(curr)

        result = np.array(densified)
        logger.info(f"Snapped {len(points)} → {len(snapped)} → {len(result)} (densified) road nodes")
        return result
