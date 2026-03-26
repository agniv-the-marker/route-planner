"""Local graph-based routing using OSM road network.

Routes waypoints by finding shortest paths on the pre-downloaded OSM bike
graph. Fast (~30-60s for a full concept) and fully offline.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sf_bike_graph.graphml"


class GraphRouter:
    """Routes waypoints using local OSM graph pathfinding."""

    def __init__(
        self,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
        max_detour_ratio: float = 3.0,
    ):
        import osmnx as ox

        graph_path = Path(graph_path)
        candidates = [graph_path, DEFAULT_GRAPH_PATH, Path("/app") / graph_path]
        resolved = None
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        graph_path = resolved
        logger.info(f"Loading OSM graph from {graph_path}...")
        self.G = ox.load_graphml(graph_path)
        self._ox = ox
        self.max_detour_ratio = max_detour_ratio
        logger.info(f"Graph loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")

    def route_waypoints(
        self, waypoints: np.ndarray
    ) -> list[tuple[float, float]]:
        """Route through all waypoints using shortest path.

        Uses standard Dijkstra (weight=length) for speed on large graphs.
        Rejects segments where the routed path is >max_detour_ratio times
        the straight-line distance (prevents long detours).
        """
        if len(waypoints) < 2:
            return [(float(waypoints[0, 0]), float(waypoints[0, 1]))]

        n_segments = len(waypoints) - 1
        full_route: list[tuple[float, float]] = []
        routed_count = 0
        straight_count = 0

        for i in range(n_segments):
            start_lat, start_lon = float(waypoints[i, 0]), float(waypoints[i, 1])
            end_lat, end_lon = float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1])

            try:
                start_node = self._ox.nearest_nodes(self.G, start_lon, start_lat)
                end_node = self._ox.nearest_nodes(self.G, end_lon, end_lat)

                # Check snap distance
                s_nlat = self.G.nodes[start_node]["y"]
                s_nlon = self.G.nodes[start_node]["x"]
                e_nlat = self.G.nodes[end_node]["y"]
                e_nlon = self.G.nodes[end_node]["x"]

                if (_haversine(start_lat, start_lon, s_nlat, s_nlon) > 0.3 or
                    _haversine(end_lat, end_lon, e_nlat, e_nlon) > 0.3):
                    segment = [(start_lat, start_lon), (end_lat, end_lon)]
                    straight_count += 1
                elif start_node == end_node:
                    segment = [(s_nlat, s_nlon)]
                    routed_count += 1
                else:
                    path = nx.shortest_path(
                        self.G, start_node, end_node, weight="length"
                    )
                    segment = [
                        (self.G.nodes[n]["y"], self.G.nodes[n]["x"])
                        for n in path
                    ]

                    # Detour check: reject if route is way longer than straight line
                    straight_dist = _haversine(start_lat, start_lon, end_lat, end_lon)
                    if straight_dist > 0.01:  # >10m
                        route_dist = sum(
                            _haversine(segment[j][0], segment[j][1],
                                       segment[j+1][0], segment[j+1][1])
                            for j in range(len(segment) - 1)
                        )
                        if route_dist > straight_dist * self.max_detour_ratio:
                            segment = [(start_lat, start_lon), (end_lat, end_lon)]
                            straight_count += 1
                        else:
                            routed_count += 1
                    else:
                        routed_count += 1

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                segment = [(start_lat, start_lon), (end_lat, end_lon)]
                straight_count += 1

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

        # Post-processing: remove immediate reversals
        full_route = _remove_reversals(full_route)

        if straight_count > 0:
            logger.warning(
                "%d/%d segments used straight-line fallback",
                straight_count, n_segments,
            )
        logger.info(
            f"Graph-routed {len(waypoints)} waypoints → {len(full_route)} points "
            f"({routed_count} routed, {straight_count} straight)"
        )
        return full_route

    # Alias for compatibility
    route_waypoints_parallel = route_waypoints


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km between two lat/lon points."""
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def _remove_reversals(
    route: list[tuple[float, float]], tolerance: float = 1e-5
) -> list[tuple[float, float]]:
    """Remove immediate A→B→A backtracking."""
    if len(route) < 3:
        return route

    cleaned = [route[0], route[1]]
    for i in range(2, len(route)):
        prev2 = cleaned[-2] if len(cleaned) >= 2 else None
        if prev2 is not None:
            dlat = abs(route[i][0] - prev2[0])
            dlon = abs(route[i][1] - prev2[1])
            if dlat < tolerance and dlon < tolerance:
                cleaned.pop()
                continue
        cleaned.append(route[i])

    return cleaned
