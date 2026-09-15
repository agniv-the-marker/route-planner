import time
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box
from scripts.outline_benchmark import fixtures
from src.outlines import OutlineSpec, prompt_key
from src.street_search import StreetSearch

class Frame:
    bounds = (-1000, -1000, 3000, 3000)
    polygon = box(*bounds)


def loop_graph():
    g = nx.MultiDiGraph()
    for i, p in enumerate([(0,0), (1000,0), (1000,1000), (0,1000)]):
        g.add_node(i, x=p[0], y=p[1])
    for u,v in [(0,1),(1,2),(2,3),(3,0)]:
        g.add_edge(u,v,length=1000)
    return g


def test_fixture_validation():
    assert len(fixtures()) == 6
    for shape in fixtures().values():
        assert OutlineSpec.parse({'interpretation': shape.interpretation, 'points': shape.points}) == shape

@pytest.mark.parametrize('change', ['open', 'crossed', 'nan', 'out_of_bounds', 'boolean'])
def test_invalid_outline(change):
    p = np.asarray(fixtures()['circle'].points).tolist()
    if change == 'open': p[-1] = [.4, .4]
    if change == 'crossed': p[2],p[16] = p[16],p[2]
    if change == 'nan': p[2][0] = float('nan')
    if change == 'out_of_bounds': p[2][0] = 2
    if change == 'boolean': p = [[True,False]]*8
    with pytest.raises(ValueError):
        OutlineSpec.parse({'interpretation': 'shape', 'points': p})


def test_directed_loop_curves_and_parallel_edges():
    g = loop_graph()
    g[0][1][0]['length'] = 1500
    curve = LineString([(0,0),(500,-50),(1000,0)])
    g[0][1][0]['geometry'] = curve
    g.add_edge(0,1,key=1,length=curve.length,geometry=curve)
    s = StreetSearch(g, Frame())
    s.deadline = time.perf_counter()+8
    p = np.array([(0,0),(1000,0),(1000,1000),(0,1000),(0,0)])
    r = s.match(p)
    assert r and (0,1,1) in r['edges'] and (0,1,0) not in r['edges']
    assert min(y for x,y in r['xy']) == -50
    assert r['xy'][0] == r['xy'][-1]
    assert r['scores']['repeated'] == pytest.approx(0)
    for a,b in zip(r['index_edges'], r['index_edges'][1:]):
        assert s.edges[a][1] == s.edges[b][0]


def test_crossings_are_not_junctions_and_disconnected():
    g = nx.MultiDiGraph()
    for i,(x,y) in enumerate([(0,500),(1000,500),(500,0),(500,1000)]):
        g.add_node(i,x=x,y=y)
    g.add_edge(0,1,length=1000)
    g.add_edge(2,3,length=1000)
    s = StreetSearch(g, Frame())
    s.deadline = time.perf_counter()+1
    assert not s.paths(0,{3},3000)
    assert not s.paths(1,{0},3000)


def test_timeout_and_hash_invalidation():
    g = loop_graph()
    s = StreetSearch(g, Frame())
    result = s.search(fixtures()['circle'], budget=0)
    assert result.failure_reason and result.diagnostics['timed_out']
    assert result.timings['search_seconds'] < .1
    g[0][1][0]['length'] += 1
    assert StreetSearch(g, Frame()).graph_hash != s.graph_hash
    assert prompt_key(' A CAT  ') == prompt_key('a cat')
    assert prompt_key('a dog') != prompt_key('a cat')


def test_rotation_matches_north_up_map_without_mirroring():
    from src.geo import SF
    from src.map_view import svg_path
    s = StreetSearch(loop_graph(), Frame())
    spec = fixtures()['fish']  # asymmetric: tail at right, nose at left
    left, bottom, right, top = SF.bounds
    center = ((left+right)/2, (bottom+top)/2)
    original = np.asarray(spec.points)
    baseline = s.transform(spec, *center, 0, 3000)
    pixels = SF.meters_to_pixels(baseline)
    # Screen orientation survives normalized → projected → screen conversion.
    scale = np.linalg.norm(pixels[1]-pixels[0]) / np.linalg.norm(original[1]-original[0])
    np.testing.assert_allclose(pixels-pixels[0], (original-original[0])*scale, atol=1e-8)
    rotated = s.transform(spec, *center, 90, 3000)
    delta = baseline - center
    np.testing.assert_allclose(rotated-center, np.column_stack((-delta[:,1], delta[:,0])), atol=1e-8)
    # Positive rotation is counterclockwise on the displayed north-up map.
    screen_delta = SF.meters_to_pixels(rotated) - SF.meters_to_pixels([center])
    base_delta = pixels - SF.meters_to_pixels([center])
    np.testing.assert_allclose(screen_delta, np.column_stack((base_delta[:,1], -base_delta[:,0])), atol=1e-8)
    assert svg_path(rotated).startswith('M')
    np.testing.assert_allclose(s.transform(spec, *center, 360, 3000), baseline, atol=1e-8)


def test_path_limit_cache_and_real_detour():
    s = StreetSearch(loop_graph(), Frame())
    s.deadline = time.perf_counter()+2
    assert not s.paths(0, {1}, 999)
    path = s.paths(0, {1}, 1000)
    assert path[1][0] == pytest.approx(1000)
    assert s.paths(0, {1}, 1000) is path
    # The only reverse connection travels three sides; never add a reverse shortcut.
    assert not s.paths(1, {0}, 2999)
    assert s.paths(1, {0}, 3000)[0][0] == pytest.approx(3000)


def test_collapse_and_minimum_ride_distance():
    g = loop_graph()
    for _,d in g.nodes(data=True):
        d['x'] /= 10
        d['y'] /= 10
    for _,_,d in g.edges(data=True):
        d['length'] /= 10
    s = StreetSearch(g, Frame())
    s.deadline = time.perf_counter()+2
    assert s.match(np.array([(0,0),(100,0),(100,100),(0,100),(0,0)])) is None


def test_curved_result_gpx_uses_the_actual_route():
    import gpxpy
    from src.generation import to_gpx
    from src.geo import TO_GEO
    g = loop_graph()
    g[0][1][0]['geometry'] = LineString([(0,0),(500,-50),(1000,0)])
    g[0][1][0]['length'] = g[0][1][0]['geometry'].length
    s = StreetSearch(g, Frame())
    s.deadline = time.perf_counter()+2
    route = s.match(np.array([(0,0),(1000,0),(1000,1000),(0,1000),(0,0)]))
    assert route
    xy = np.asarray(route['xy'])
    lon,lat = TO_GEO.transform(xy[:,0],xy[:,1])
    gpx = gpxpy.parse(to_gpx(list(zip(lat,lon)), 'curve'))
    points = gpx.tracks[0].segments[0].points
    np.testing.assert_allclose([(p.latitude,p.longitude) for p in points], np.column_stack((lat,lon)), atol=1e-9)
    assert len(points) == len(xy) and min(xy[:,1]) == -50


def test_prepared_index_roundtrip_and_source_hash_invalidation(tmp_path,monkeypatch):
    from src.graph import BikeRouter
    source=tmp_path/'graph.graphml'
    source.write_text('fixture graph version one')
    calls=[]
    def load(path):
        calls.append(path)
        return BikeRouter(loop_graph())
    monkeypatch.setattr(BikeRouter,'load',load)
    one=StreetSearch.load_prepared(source,tmp_path/'index',Frame())
    two=StreetSearch.load_prepared(source,tmp_path/'index',Frame())
    assert len(calls)==1
    np.testing.assert_array_equal(one.xy,two.xy)
    assert one.graph_hash==two.graph_hash and len(one.edges)==len(two.edges)
    for a,b in zip(one.edges,two.edges):
        assert a[:3]==b[:3] and a[4:]==b[4:]
        np.testing.assert_array_equal(a[3],b[3])
    source.write_text('fixture graph version two')
    StreetSearch.load_prepared(source,tmp_path/'index',Frame())
    assert len(calls)==2


def test_parallel_reverse_street_pieces_share_physical_identity():
    graph=loop_graph()
    curve=LineString([(0,0),(350,-25),(500,-50),(1000,0)])
    graph[0][1][0].update(geometry=curve,length=curve.length)
    graph.add_edge(1,0,geometry=LineString(list(curve.coords)[::-1]),length=curve.length)
    search=StreetSearch(graph,Frame())
    forward={e[5] for e in search.edges if e[4][:2]==(0,1)}
    reverse={e[5] for e in search.edges if e[4][:2]==(1,0)}
    assert forward==reverse

def test_deterministic_work_failure_cache(tmp_path):
    s = StreetSearch(loop_graph(),Frame())
    s.route_cache_dir=tmp_path
    spec=fixtures()['circle']
    a=s.search(spec,work_budget=0)
    assert a.diagnostics['work_exhausted'] and not a.diagnostics['timed_out']
    b=s.search(spec,work_budget=0)
    assert b.failure_reason==a.failure_reason and 'cache_seconds' in b.timings
    s.route_cache.clear()
    c=s.search(spec,work_budget=0)
    assert c.failure_reason==a.failure_reason and 'cache_seconds' in c.timings

def test_feature_anchors_include_concavities_and_respect_cap():
    s=StreetSearch(loop_graph(),Frame())
    captured=[]
    class Capture:
        def query(self,anchors,**kwargs):
            captured.extend(anchors)
            raise RuntimeError('captured anchors')
    s.junction_tree=Capture()
    p=np.array([(0,0),(450,0),(450,700),(480,700),(480,0),(1000,0),
                (1000,1000),(0,1000),(0,0)])
    with pytest.raises(RuntimeError,match='captured anchors'): s.match(p)
    assert 32<=len(captured)<=64
    assert all(any(np.array_equal(vertex,anchor) for anchor in captured) for vertex in p[:-1])


def test_distance_band_is_configurable_and_defaults_to_the_historical_one():
    from src.street_search import DEFAULT_DISTANCE_RANGE, DEFAULT_PERIMETERS
    s = StreetSearch(loop_graph(), Frame())
    assert s.distance_range == DEFAULT_DISTANCE_RANGE == (3000, 15000)
    assert s.perimeters == DEFAULT_PERIMETERS
    # Middle-out, reproducing the order the placement ladder used to hardcode.
    assert s.perimeter_preference == (8000, 11000, 5000, 14000, 3000)
    s.set_distance_band(perimeters=(1, 2, 3, 4))
    assert s.perimeter_preference == (3, 4, 2, 1)
    assert s.distance_label == '3–15 km'
    s.set_distance_band(distance_range=(2000, 25000))
    assert s.distance_range == (2000, 25000)
    assert s.distance_label == '2–25 km'


def test_distance_band_takes_part_in_the_route_cache_identity():
    """A wider band accepts loops the narrow one rejected, so it cannot share a key."""
    import hashlib, json
    from src.street_search import VERSION
    s = StreetSearch(loop_graph(), Frame())

    def key():
        return hashlib.sha256(json.dumps(
            [spec.key(), s.graph_hash, VERSION, s.frame.bounds, 8, None,
             s.perimeters, s.distance_range]).encode()).hexdigest()

    spec = fixtures()['fish']
    narrow = key()
    s.set_distance_band(distance_range=(2000, 25000))
    assert key() != narrow


def test_distance_band_rejects_nonsense():
    s = StreetSearch(loop_graph(), Frame())
    for bad in [(15000, 3000), (0, 15000), (-1, 10)]:
        with pytest.raises(ValueError):
            s.set_distance_band(distance_range=bad)
    with pytest.raises(ValueError):
        s.set_distance_band(perimeters=())
