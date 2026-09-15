"""One-time map preparation; never download streets during a user request."""

import hashlib
import json
from datetime import datetime, timezone

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

from src.geo import CRS, SF
from src.graph import GRAPH_PATH, PREPARATION_VERSION, allowed_edge, edge_coordinates


def prepare():
    ox.settings.use_cache = True
    ox.settings.cache_folder = ".cache/osmnx"
    ox.settings.log_console = True
    ox.settings.requests_timeout = 120
    ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | {
        "bicycle", "route", "access:conditional", "bicycle:conditional", "oneway:bicycle",
        "bicycle:forward", "bicycle:backward", "oneway:conditional", "oneway:bicycle:conditional",
    })
    # custom_filter REPLACES the bike preset; deliberately do not provide one.
    graph = ox.graph_from_polygon(SF.geographic_polygon, network_type="bike", simplify=False,
                                  retain_all=True, truncate_by_edge=True)
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        reverse = data.get("reversed", False)
        bicycle_direction = data.get("oneway:bicycle")
        prohibited_direction = (
            (reverse and bicycle_direction in {"yes", "1", "true"})
            or (not reverse and bicycle_direction == "-1")
            or data.get("bicycle:backward" if reverse else "bicycle:forward") in {"no", "dismount"}
        )
        if not allowed_edge(data) or prohibited_direction:
            graph.remove_edge(u, v, key)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph = ox.simplify_graph(graph, edge_attrs_differ=["access", "bicycle", "oneway:bicycle"])
    graph = ox.project_graph(graph, to_crs=CRS)
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        xy = edge_coordinates(graph, u, v, data)
        geometry = LineString(xy)
        if not SF.polygon.covers(geometry):
            graph.remove_edge(u, v, key)
        else:
            data["geometry"] = geometry
            data["length"] = geometry.length
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if not graph.nodes or any(not allowed_edge(d) for _, _, d in graph.edges(data=True)):
        raise RuntimeError("No usable bicycle network, or excluded edges survived preparation.")
    graph.graph.update(preparation=PREPARATION_VERSION, frame=repr(SF),
                       downloaded_at=datetime.now(timezone.utc).isoformat())
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GRAPH_PATH.with_suffix(".pending.graphml")
    ox.save_graphml(graph, temporary)
    temporary.replace(GRAPH_PATH)
    metadata = {"preparation": PREPARATION_VERSION, "frame": repr(SF), "crs": CRS,
                "downloaded_at": graph.graph["downloaded_at"], "osmnx": ox.__version__,
                "network_type": "bike", "custom_filter": None,
                "nodes": len(graph), "edges": graph.number_of_edges(),
                "sha256": hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest(),
                "attribution": "© OpenStreetMap contributors, ODbL 1.0",
                "limitations": "Conservative direction/access filtering; no turn-restriction relations or live closures."}
    GRAPH_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    prepare()
