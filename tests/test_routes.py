import numpy as np
import networkx as nx
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import LineString

from src.geo import SF
from src.graph import BikeRouter, RouteError, allowed_edge


@pytest.fixture
def square():
    graph = nx.MultiDiGraph()
    for node, (x, y) in enumerate([(0, 0), (200, 0), (200, 200), (0, 200)]):
        graph.add_node(node, x=x, y=y)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        graph.add_edge(u, v, length=200, highway="residential")
    return graph


def test_frame_round_trip_and_north():
    pixels = np.array([[0, 0], [511, 511], [135, 241]])
    xy = SF.pixels_to_meters(pixels)
    np.testing.assert_allclose(SF.meters_to_pixels(xy), pixels, atol=1e-7)
    assert xy[0, 0] < xy[1, 0] and xy[0, 1] > xy[1, 1]
    np.testing.assert_allclose(xy[1] - xy[0], [SF.width_m, -SF.width_m])


def test_curved_parallel_edge_geometry_and_closure(square):
    # Geometry may be stored backwards; select the shorter parallel edge.
    curve = [(0, 0), (100, -40), (200, 0)]
    square[0][1][0]["length"] = 400
    key = square.add_edge(0, 1, length=LineString(curve).length,
                          geometry=LineString(curve[::-1]))
    result = BikeRouter(square).route([(0, 0), (200, 0), (200, 200), (0, 200)])
    assert result.edges == [(0, 1, key), (1, 2, 0), (2, 3, 0), (3, 0, 0)]
    np.testing.assert_allclose(result.xy, curve + [(200, 200), (0, 200), (0, 0)])
    assert result.distance_m == pytest.approx(600 + LineString(curve).length)


def test_disconnected_loop_never_gets_shortcut(square):
    square.remove_edge(3, 0)
    with pytest.raises(RouteError, match="cannot be connected"):
        BikeRouter(square).route([(0, 0), (200, 0), (200, 200), (0, 200)])


def test_direction_is_respected(square):
    result = BikeRouter(square).route([(0, 0), (0, 200), (200, 200), (200, 0)])
    assert len(result.edges) == 12
    assert all(square.has_edge(*edge) for edge in result.edges)


@pytest.mark.parametrize("points, message", [
    ([(1000, 1000), (200, 0), (200, 200)], "too far"),
    ([(0, 0), (1, 0), (0, 1)], "too small"),
    ([(0, 0), (float("nan"), 0), (0, 200)], "usable outline"),
])
def test_invalid_snaps_and_shapes(square, points, message):
    with pytest.raises(RouteError, match=message):
        BikeRouter(square).route(points)


def test_excessive_detour(square):
    square[0][1][0]["length"] = 601
    with pytest.raises(RouteError, match="detour"):
        BikeRouter(square).route([(0, 0), (200, 0), (200, 200), (0, 200)])


def test_bad_edge_geometry_fails(square):
    square[0][1][0]["geometry"] = LineString([(10, 10), (200, 0)])
    with pytest.raises(RouteError, match="broken edge"):
        BikeRouter(square).route([(0, 0), (200, 0), (200, 200), (0, 200)])


@pytest.mark.parametrize("tag", [
    {"access": "private"}, {"bicycle": "no"}, {"highway": "steps"},
    {"route": "ferry"}, {"bicycle:conditional": "no @ (Mo-Fr)"},
])
def test_access_exclusions(tag):
    assert not allowed_edge({"highway": "residential", **tag})


def test_prepared_map_integrity():
    import hashlib
    import json
    from src.graph import GRAPH_PATH

    metadata = json.loads(GRAPH_PATH.with_suffix(".json").read_text())
    assert hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest() == metadata["sha256"]
    graph = BikeRouter.load().graph
    assert len(graph) == metadata["nodes"]
    assert graph.number_of_edges() == metadata["edges"]
    assert all(SF.polygon.covers(d["geometry"]) for _, _, d in graph.edges(data=True))
