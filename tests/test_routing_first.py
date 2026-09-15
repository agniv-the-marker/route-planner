import json
from dataclasses import replace
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString
from src.strokes import Drawing
from src.stroke_router import StrokeRouter, RoutingConfig
from src.experiment_journal import Journal, Budget, atomic_json, digest


def graph():
    g=nx.MultiDiGraph()
    for n,p in enumerate([(0,0),(0,100),(100,100),(100,0)]):
        g.add_node(n,x=p[0],y=p[1])
    def add(u,v,k=0,**kw):
        p=[(g.nodes[u]['x'],g.nodes[u]['y']),(g.nodes[v]['x'],g.nodes[v]['y'])]
        geom=kw.pop('geometry',LineString(p))
        g.add_edge(u,v,key=k,geometry=geom,length=geom.length,highway='residential',**kw)
    for u,v in [(0,1),(1,2),(2,3),(0,3),(3,2),(2,1),(1,0)]:add(u,v)
    return g


def test_longer_shape_following_path_beats_shortcut_and_parallel_curve():
    g=graph()
    router=StrokeRouter(g)
    router.start(RoutingConfig(work_budget=10000))
    target=LineString([(0,0),(0,100),(100,100),(100,0)])
    shortest=router.paths(0,{3},target,500,'expanded')[3][0]
    following=router.paths(0,{3},target,500,'paper')[3][0]
    assert shortest[1]==100 and following[1]==300
    assert following[2]==[(0,1,0),(1,2,0),(2,3,0)]
    g.add_edge(0,3,key=1,highway='residential',length=300,geometry=target)
    router=StrokeRouter(g);router.start(RoutingConfig(work_budget=10000))
    path=router.paths(0,{3},target,500,'paper')[3][0]
    rendered=router.render_route(path[2],np.array(target.coords))
    assert rendered['distance_m']==300
    assert max(y for x,y in rendered['xy'])==100


def test_open_retracing_and_gpx_agree():
    from src.generation import to_gpx
    from src.geo import TO_GEO
    import gpxpy
    router=StrokeRouter(graph());router.start(RoutingConfig())
    edges=[(0,1,0),(1,2,0),(2,1,0),(1,0,0),(0,3,0)]
    target=np.array([(0,0),(0,100),(100,100),(0,100),(0,0),(100,0)])
    r=router.render_route(edges,target)
    assert r['distance_m']==500 and r['scores']['repeated']==pytest.approx(.4)
    assert r['xy'][0]!=r['xy'][-1]
    xy=np.array(r['xy']);lon,lat=TO_GEO.transform(xy[:,0],xy[:,1])
    parsed=gpxpy.parse(to_gpx(list(zip(lat,lon)),'test'))
    pts=parsed.tracks[0].segments[0].points
    assert len(pts)==len(xy)
    assert np.allclose([p.latitude for p in pts],lat)


def test_overpass_is_not_a_junction_and_access_excluded():
    g=nx.MultiDiGraph()
    for n,(x,y) in enumerate([(-100,0),(100,0),(0,-100),(0,100)]):g.add_node(n,x=x,y=y)
    for u,v in [(0,1),(2,3)]:g.add_edge(u,v,highway='residential',length=200)
    g.add_edge(1,2,highway='residential',access='private',length=200)
    router=StrokeRouter(g);router.start(RoutingConfig())
    assert not router.paths(0,{3},LineString([(-100,0),(0,100)]),1000,'paper')
    with pytest.raises(ValueError,match='Disconnected'):router.render_route([(0,1,0),(2,3,0)],np.array([[-100,0],[0,100]]))


def test_connectors_upright_and_no_mirroring():
    d=Drawing.parse({'name':'test','strokes':[[[0,0],[0,1]],[[1,1],[1,0]]]})
    p,connectors=d.place([0,0],100,0)
    assert connectors==[1]
    assert np.allclose(p,[[-50,50],[-50,-50],[50,-50],[50,50]])
    with pytest.raises(ValueError):d.place([0,0],100,16)
    for angle in (-15,15):
        q,_=d.place([0,0],100,angle)
        assert np.linalg.det(np.vstack([q[1]-q[0],q[2]-q[1]]))>0


def test_configurable_distance_and_partial_reporting():
    router=StrokeRouter(graph())
    cfg=RoutingConfig(max_distance_m=500,snap_m=1,candidates=1,anchor_spacing_m=100,work_budget=10000)
    router.start(cfg)
    p=np.array([(0,0),(0,100),(100,100),(100,0)])
    r,reason=router.match(p)
    assert reason is None and r['distance_m']==300
    router.start(replace(cfg,max_distance_m=250))
    r,reason=router.match(p)
    assert r is None
    d=Drawing.parse({'name':'v','strokes':[[[0,0],[0,1],[1,1]]]})
    result=router.search(d,replace(cfg,work_budget=1,grid_m=10))
    assert result['status']=='incomplete' and result['limit']=='work_budget'
    for km in (15,30,50,80): assert RoutingConfig(max_distance_m=km*1000).max_distance_m==km*1000


def test_journal_resume_stale_cache_immutable_raw_and_interruption(tmp_path):
    j=Journal(tmp_path);calls=[]
    identity={'model':'v1','compiler':'v1'}
    run=lambda: calls.append(1) or {'raw':'abc'}
    assert j.run('a','infer',identity,run)=={'raw':'abc'}
    j.run('a','infer',identity,run)
    assert len(calls)==1
    j.run('a','compile',{**identity,'compiler':'v2'},run)
    assert len(calls)==2
    j.raw('a',{'raw':'abc'})
    with pytest.raises(ValueError):j.raw('a',{'raw':'changed'})
    def crash():raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):j.run('b','infer',identity,crash,remote=True,reservation='test')
    with pytest.raises(RuntimeError,match='Reconcile'):j.run('b','infer',identity,run,remote=True,reservation='test')
    key=digest(['b','infer',identity])
    j.reconcile(key,'failed','Remote job confirmed never started','job-123')
    j.run('b','infer',identity,run,remote=True,reservation='test')
    artifact=next((tmp_path/'artifacts').glob('*.json'))
    artifact.write_text('{}')
    stage=next(row for p in (tmp_path/'stages').glob('*.json') if (row:=json.loads(p.read_text())).get('artifact')==str(artifact.relative_to(tmp_path)))
    with pytest.raises(ValueError,match='hash mismatch'):j.run(stage['attempt'],stage['stage'],stage['identity'],run)


def test_budget_reserves_preview_and_uncertain_spend(tmp_path):
    b=Budget(tmp_path/'budget.json')
    b.reserve('research',90,'research','price checked','timeout')
    with pytest.raises(ValueError):b.reserve('more',.01,'research','price','timeout')
    b.reserve('preview',10,'preview','price checked','timeout')
    with pytest.raises(ValueError):b.reserve('excess',.01,'preview','price','timeout')


def test_region_boundary_filters_whole_edges():
    from shapely.geometry import box
    class Region:polygon=box(-1,-1,50,101)
    router=StrokeRouter(graph(),Region())
    assert set(router.edges)=={(0,1,0),(1,0,0)}


def test_reference_contract_and_approval_required(tmp_path):
    from scripts.routing_references import prepare
    manifest=prepare(tmp_path/'refs')
    assert len(manifest['drawings'])==12
    assert sum(r['split']=='comparison' for r in manifest['drawings'])==8
    assert manifest['source_acceptance'] is None
    assert all(Drawing.parse(r['drawing']).key() for r in manifest['drawings'])


def test_review_persistence_and_explicit_labels(tmp_path):
    from scripts.routing_references import prepare
    from src.review_preview import create_review_site
    from fastapi.testclient import TestClient
    prepare(tmp_path/'artifacts')
    with TestClient(create_review_site(tmp_path/'artifacts',tmp_path/'state')) as client:
        assert client.get('/health').json()['inference_enabled'] is False
        assert client.get('/').status_code==200
        assert client.get('/01-source.svg').status_code==200
        assert client.put('/api/reviews/01',json={'label':'uncertain','guess':'heart'}).status_code==200
        assert client.put('/api/reviews/01',json={'label':'maybe'}).status_code==422
    with TestClient(create_review_site(tmp_path/'artifacts',tmp_path/'state')) as client:
        assert client.get('/api/reviews').json()['01']['label']=='uncertain'


def test_comparison_gate_rejects_unapproved_and_changed_sources(tmp_path):
    from scripts.routing_references import prepare
    from scripts.routing_comparison import approved_manifest,accept
    prepare(tmp_path/'refs')
    with pytest.raises(ValueError,match='acceptance'):approved_manifest(tmp_path/'refs')
    accept(tmp_path/'refs','Synthetic test acceptance')
    approved_manifest(tmp_path/'refs')
    p=tmp_path/'refs'/'references.json';m=json.loads(p.read_text())
    m['drawings'][0]['drawing']['name']='changed';atomic_json(p,m)
    with pytest.raises(ValueError,match='changed'):approved_manifest(tmp_path/'refs')


def test_generated_cache_recompiles_without_repeating_inference(tmp_path,monkeypatch):
    from src.generative import VectorGenerator
    import src.generative as module
    calls=[]
    def complete(*args,**kwargs):calls.append(kwargs['seed']);return 'invalid original raw'
    list(VectorGenerator(complete,cache_dir=tmp_path).candidates('test'))
    monkeypatch.setattr(module,'compile_program',lambda *args: (_ for _ in ()).throw(ValueError('new compiler rejection')))
    # Extract also validates, so inject just the raw parser for this cache regression.
    import src.vector_program
    monkeypatch.setattr(src.vector_program,'extract_program',lambda raw: {})
    rows=list(VectorGenerator(complete,cache_dir=tmp_path).candidates('test'))
    assert len(calls)==4
    assert all(r.validation=='new compiler rejection' for r in rows)


def test_budget_caps_come_from_the_ledger_and_null_means_no_ceiling(tmp_path):
    import json
    path = tmp_path / 'budget.json'
    b = Budget(path)
    b.reserve('first', 90, 'research', 'price checked', 'timeout')
    with pytest.raises(ValueError):
        b.reserve('over-research', .01, 'research', 'price', 'timeout')
    # Lifting only the research cap still leaves the hard cap in force.
    data = json.loads(path.read_text())
    data['research_cap_usd'] = None
    path.write_text(json.dumps(data))
    b.reserve('now-allowed', 9, 'research', 'price checked', 'timeout')
    with pytest.raises(ValueError):
        b.reserve('over-total', 2, 'research', 'price', 'timeout')
    # Removing both caps stops refusals, and reservations are still recorded.
    data = json.loads(path.read_text())
    data['cap_usd'] = None
    path.write_text(json.dumps(data))
    b.reserve('uncapped', 500, 'research', 'price checked', 'timeout')
    assert json.loads(path.read_text())['allocations']['uncapped']['reserved_usd'] == 500
