"""Routing succeeds only when every segment follows directed street edges."""

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from src.geo import CRS, SF

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "sf.graphml"
PREPARATION_VERSION = "sf-bike-v1"
FORBIDDEN_ACCESS = {"no", "private", "permit", "customers", "destination", "residents",
                    "emergency", "employees", "restricted", "delivery"}
FORBIDDEN_HIGHWAYS = {"steps", "motorway", "motorway_link", "construction", "proposed",
                      "footway", "platform", "corridor", "elevator", "escalator"}


def values(value):
    return set(map(str, value if isinstance(value, (list, tuple, set)) else [value]))


def allowed_edge(data):
    """Conservative exclusions in addition to the OSMnx bicycle preset."""
    return bool(data.get("highway")) and not (
        values(data.get("highway")) & FORBIDDEN_HIGHWAYS
        or values(data.get("access")) & FORBIDDEN_ACCESS
        or values(data.get("bicycle")) & (FORBIDDEN_ACCESS | {"dismount", "use_sidepath"})
        or "ferry" in values(data.get("route"))
        or data.get("access:conditional") or data.get("bicycle:conditional")
        or data.get("oneway:conditional") or data.get("oneway:bicycle:conditional")
    )


class RouteError(ValueError):
    """An expected failure to form a valid street loop."""


@dataclass
class StreetRoute:
    xy: np.ndarray
    edges: list[tuple[int, int, int]]
    distance_m: float


def edge_coordinates(graph, u, v, data):
    start = np.array([graph.nodes[u]["x"], graph.nodes[u]["y"]])
    end = np.array([graph.nodes[v]["x"], graph.nodes[v]["y"]])
    geometry = data.get("geometry")
    points = np.array(geometry.coords if geometry is not None else [start, end], dtype=float)
    if np.linalg.norm(points[-1] - start) < np.linalg.norm(points[0] - start):
        points = points[::-1]
    if not (np.allclose(points[0], start, atol=0.01, rtol=0)
            and np.allclose(points[-1], end, atol=0.01, rtol=0)):
        raise RouteError("The street data contains a broken edge geometry.")
    return points


class BikeRouter:
    def __init__(self, graph):
        if not graph.is_directed() or not graph.is_multigraph() or not graph.nodes:
            raise ValueError("A nonempty directed street graph is required.")
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.coordinates = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"])
                                     for n in self.nodes], dtype=float)
        self.tree = cKDTree(self.coordinates)

    @classmethod
    def load(cls, path=GRAPH_PATH):
        import osmnx as ox

        if not Path(path).exists():
            raise RuntimeError("The SF street map is unavailable. Run python -m scripts.prepare_map.")
        graph = ox.load_graphml(path)
        if (graph.graph.get("preparation") != PREPARATION_VERSION
                or str(graph.graph.get("crs")) != CRS
                or graph.graph.get("frame") != repr(SF)):
            raise RuntimeError("The street map needs rebuilding with scripts.prepare_map.")
        if any(not allowed_edge(d) for _, _, d in graph.edges(data=True)):
            raise RuntimeError("The street map contains excluded edges; rebuild it.")
        return cls(graph)

    def route(self, points, max_snap_m=300, max_detour_ratio=3):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise RouteError("The image did not produce a usable outline.")
        if not np.allclose(points[0], points[-1], atol=0.01, rtol=0):
            points = np.vstack([points, points[0]])
        distances, indices = self.tree.query(points)
        if distances.max() > max_snap_m:
            raise RouteError("Part of this shape is too far from bicycle-accessible streets. Try another shape.")
        nodes = [self.nodes[i] for i in indices]
        nodes = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
        if len(set(nodes)) < 3:
            raise RouteError("This shape is too small to form a street loop.")
        route_edges = []
        coordinates = []
        distance_m = 0.0
        for start, end in zip(nodes, nodes[1:]):
            try:
                path = nx.shortest_path(self.graph, start, end, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                raise RouteError("This outline cannot be connected using bicycle-accessible streets.") from exc
            segment = []
            segment_length = 0.0
            for u, v in zip(path, path[1:]):
                key, data = min(self.graph[u][v].items(), key=lambda item: float(item[1]["length"]))
                segment.append((u, v, key, data))
                segment_length += float(data["length"])
            a, b = self.graph.nodes[start], self.graph.nodes[end]
            direct = np.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if segment_length > max(10, direct) * max_detour_ratio:
                raise RouteError("Following this shape would require too large a street detour. Try again.")
            for u, v, key, data in segment:
                xy = edge_coordinates(self.graph, u, v, data)
                coordinates.extend(xy if not coordinates else xy[1:])
                route_edges.append((u, v, key))
            distance_m += segment_length
        xy = np.array(coordinates)
        if (len(xy) < 4 or not np.allclose(xy[0], xy[-1], atol=0.01, rtol=0)
                or LineString(xy).convex_hull.area < 10000):
            raise RouteError("This outline collapsed instead of forming a useful loop.")
        return StreetRoute(xy, route_edges, distance_m)
