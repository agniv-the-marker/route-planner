"""Local graph-based routing using OSM road network.

Routes waypoints by finding shortest paths on the pre-downloaded OSM bike
graph. This is ~200x faster than the Valhalla API (instant vs 194s/concept)
and works fully offline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sf_bike_graph.graphml"


class GraphRouter:
    """Routes waypoints using local OSM graph pathfinding.

    Loads a pre-downloaded OSM bike network graph and uses Dijkstra's
    shortest path to route between consecutive waypoints.
    """

    def __init__(
        self,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
    ):
        import osmnx as ox

        graph_path = Path(graph_path)
        # Try multiple locations: given path, project root, /app (Modal)
        candidates = [
            graph_path,
            DEFAULT_GRAPH_PATH,
            Path("/app") / graph_path,
        ]
        resolved = None
        for p in candidates:
            if p.exists():
                resolved = p
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Graph file not found: {graph_path}. "
                "Run: PYTHONPATH=. python -c \"import osmnx as ox; "
                "G = ox.graph_from_bbox(bbox=(37.812, 37.708, -122.357, -122.515), network_type='bike'); "
                "ox.save_graphml(G, 'data/sf_bike_graph.graphml')\""
            )

        graph_path = resolved
        logger.info(f"Loading OSM graph from {graph_path}...")
        self.G = ox.load_graphml(graph_path)
        self._ox = ox
        logger.info(f"Graph loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")

    def route_waypoints(
        self, waypoints: np.ndarray
    ) -> list[tuple[float, float]]:
        """Route through all waypoints using local graph pathfinding.

        Args:
            waypoints: (N, 2) array of (lat, lon) waypoints.

        Returns:
            List of (lat, lon) points forming the complete route.
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

                if start_node == end_node:
                    # Same node — just add it
                    lat = self.G.nodes[start_node]["y"]
                    lon = self.G.nodes[start_node]["x"]
                    segment = [(lat, lon)]
                else:
                    path = nx.shortest_path(self.G, start_node, end_node, weight="length")
                    segment = [
                        (self.G.nodes[n]["y"], self.G.nodes[n]["x"])
                        for n in path
                    ]
                routed_count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No path exists — use straight line
                segment = [(start_lat, start_lon), (end_lat, end_lon)]
                straight_count += 1

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

        if straight_count > 0:
            logger.warning(
                "%d/%d segments used straight-line fallback (no path in graph)",
                straight_count, n_segments,
            )
        logger.info(
            f"Graph-routed {len(waypoints)} waypoints → {len(full_route)} points "
            f"({routed_count} routed, {straight_count} straight)"
        )
        return full_route

    # Alias for compatibility with parallel routing interface
    route_waypoints_parallel = route_waypoints
