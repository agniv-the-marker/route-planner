import json

import gpxpy
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.drawing_service import DrawingService
from src.geo import SF, TO_GEO
from src.stroke_router import SearchLimit, StrokeRouter


class FakeRouter:
    def __init__(self, status="completed"):
        self.status = status
        self.seen = None
        self.started = None

    def start(self, config):
        self.started = config

    def match(self, target, variant="paper"):
        self.seen = np.asarray(target)
        return ({"xy": target.tolist(), "target_xy": target.tolist(),
                 "distance_m": float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum()),
                 "scores": {"line_distance_m": 0.0}}, None)

    def search(self, drawing, config, variant="paper"):
        p, _ = drawing.place([SF.bounds[0] + 100, SF.bounds[1] + 100], 200, 0)
        return {"status": self.status, "routes": [{"edges": [[1, 2, 0]], "xy": p.tolist(), "target_xy": p.tolist(),
                  "distance_m": 200.0, "scores": {"feature_max_m": 1.0}}],
                "rejections": {}, "seconds": 0.1, "work_used": 3, "limit": "wall_timeout" if self.status == "incomplete" else None}


def drawing(mode="map"):
    return {"mode": mode, "max_distance_km": 15,
            "strokes": [[[.1, .1], [.1, .5], [.5, .5], [.5, .1]]]}


def test_map_result_converts_full_511_pixels_and_is_json_serializable():
    router = FakeRouter()
    result = DrawingService(router).fit(drawing())
    assert result["status"] == "completed"
    route = result["routes"][0]
    assert route["target_pixels"][0] == pytest.approx([51.1, 51.1])
    assert route["target_pixels"][-1] == pytest.approx([255.5, 51.1])
    assert len(route["xy_pixels"]) == 4
    assert "<gpx" in route["gpx"]
    json.dumps(result)
    np.testing.assert_allclose(router.seen, SF.pixels_to_meters(np.asarray(route["target_pixels"])))
    assert router.started.candidates == router.started.beam_width == 1


def test_canvas_reports_incomplete_and_keeps_candidate():
    result = DrawingService(FakeRouter("incomplete")).fit(drawing("canvas"))
    assert result["status"] == "incomplete"
    assert result["routes"]
    assert "candidate" in result["message"]
    assert result["seconds"] == .1 and result["work_used"] == 3


def test_canvas_deduplicates_identical_edge_sequences():
    class Duplicates(FakeRouter):
        def search(self, drawing, config, variant="paper"):
            p, _ = drawing.place([SF.bounds[0] + 100, SF.bounds[1] + 100], 200, 0)
            route = {"edges": [[1, 2, 0]], "xy": p.tolist(), "target_xy": p.tolist(),
                     "distance_m": 200., "scores": {}}
            return {"status": "completed", "routes": [route, dict(route)], "rejections": {}}
    result = DrawingService(Duplicates()).fit(drawing("canvas"))
    assert len(result["routes"]) == 1


def test_canvas_filters_near_identical_routes_but_keeps_distinct_candidates():
    base = {"edges": [[1, 2, 0], [2, 3, 0]], "xy": [[0, 0], [100, 0], [200, 0]], "target_xy": [[0, 0], [200, 0]], "distance_m": 200, "scores": {}}
    near = {**base, "xy": [[20, 0], [120, 0], [220, 0]], "edges": [[1, 2, 0], [2, 3, 0]]}
    distinct = {**base, "xy": [[2000, 0], [2100, 0], [2200, 0]], "edges": [[8, 9, 0], [9, 10, 0]]}
    selected = DrawingService._select_routes([base, near, distinct])
    assert selected == [base, distinct]


def test_interactive_config_uses_small_spans_and_research_distance_bands():
    service = DrawingService(FakeRouter())
    for km in (15, 30, 50, 80):
        config = service._config(km)
        assert config.max_distance_m == km * 1000
        assert config.span_fractions == (.04, .07, .10)
        assert config.min_distance_m == (20_000 if km == 80 else 1)



def test_map_search_limit_is_explicitly_incomplete():
    class Limited(FakeRouter):
        def match(self, target, variant="paper"):
            raise SearchLimit("work_budget")
    result = DrawingService(Limited()).fit(drawing())
    assert result["status"] == "incomplete"
    assert result["routes"] == []
    assert result["rejections"] == {"work_budget": 1}


def test_real_directed_router_preserves_connector_and_gpx_geometry():
    left, bottom, _, _ = SF.bounds
    points = np.asarray([(left + 100, bottom + 100), (left + 100, bottom + 300),
                         (left + 300, bottom + 300), (left + 500, bottom + 100)])
    graph = nx.MultiDiGraph()
    for i, (x, y) in enumerate(points):
        graph.add_node(i, x=x, y=y)
    for i in range(len(points) - 1):
        geom = LineString(points[i:i + 2])
        graph.add_edge(i, i + 1, geometry=geom, length=geom.length, highway="residential")
    service = DrawingService(StrokeRouter(graph, region=SF), simplify_tolerance_px=0)
    normalized = (SF.meters_to_pixels(points) / 511).tolist()
    payload = {"mode": "map", "max_distance_km": 15,
               "strokes": [normalized[:2], normalized[2:]]}
    result = service.fit(payload)
    assert result['status'] == 'completed'
    route = result['routes'][0]
    # The middle edge is the pen-lift transfer, included in target and export.
    np.testing.assert_allclose(route['target_pixels'], SF.meters_to_pixels(points))
    np.testing.assert_allclose(route['xy_pixels'], SF.meters_to_pixels(points))
    assert route['distance_m'] == pytest.approx(400 + np.hypot(200, 200))
    gps = gpxpy.parse(route['gpx']).tracks[0].segments[0].points
    lon, lat = TO_GEO.transform(points[:, 0], points[:, 1])
    np.testing.assert_allclose([p.latitude for p in gps], lat)
    np.testing.assert_allclose([p.longitude for p in gps], lon)
    reverse = service.fit({**payload, 'strokes': [normalized[::-1]]})
    assert reverse['status'] == 'failed'
    assert not reverse['routes']


@pytest.mark.parametrize("payload", [
    {"mode": "bad", "strokes": [[[0, 0], [1, 1]]]},
    {"mode": "map", "max_distance_km": 10, "strokes": [[[0, 0], [1, 1]]]},
    {"mode": "map", "strokes": [[[0, 0], [float("nan"), 1]]]},
])
def test_rejects_invalid_payload(payload):
    with pytest.raises(ValueError):
        DrawingService(FakeRouter()).fit(payload)


def test_rejects_oversized_input():
    points = [[i / 4096, .5] for i in range(4097)]
    with pytest.raises(ValueError, match="4096"):
        DrawingService(FakeRouter()).fit({"mode": "map", "strokes": [points]})
