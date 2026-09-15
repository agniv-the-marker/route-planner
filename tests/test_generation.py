import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gpxpy
import numpy as np
from scripts.outline_benchmark import fixtures
from src.generation import RouteService
from src.geo import SF
from src.street_search import RouteSearchResult, StreetSearch

class FakeGenerator:
    def generate(self,prompt):
        return fixtures()['heart'], {'source':'test'}

class FakeSearch:
    def __init__(self,failed=False):
        self.failed=failed
        self.calls=0

    transform = StreetSearch.transform

    def search(self,spec):
        self.calls+=1
        result=RouteSearchResult(spec)
        left,bottom,right,top=SF.bounds
        x,y=(left+right)/2,(bottom+top)/2
        if self.failed:
            result.failure_reason='No qualifying 3–15 km street loop found within the search budget.'
        else:
            for shift in (0,300):
                xy=np.array([[-600,-600],[600,-600],[600,600],[-600,600],[-600,-600]])+[x+shift,y]
                result.routes.append({'xy':xy.tolist(),'edges':[(0,1,0),(1,2,0),(2,3,0),(3,0,0)],
                                      'distance_m':4800,'scores':{'rank':.1},
                                      'transform':{'x':x+shift,'y':y,'angle':0,'perimeter':5000,'direction':1}})
        result.timings={'search_seconds':.01}
        return result


def service(failed=False):
    return RouteService(FakeGenerator(),FakeSearch(failed))


def test_outline_is_yielded_before_search_and_gpx_is_closed():
    runner=service()
    stream=runner.stream('heart')
    preview=next(stream)
    assert preview.image and preview.gpx is None and runner.search.calls == 0
    result=next(stream)
    points=gpxpy.parse(result.gpx).tracks[0].segments[0].points
    assert [(p.latitude,p.longitude) for p in points] == result.coordinates
    assert result.coordinates[0] == result.coordinates[-1]
    assert result.distance_m == 4800
    assert result.diagnostics['chosen_transform']['angle'] == 0
    stream.close()


def test_failed_route_keeps_outline_without_gpx():
    result=service(True).generate('heart')
    assert result.image and result.gpx is None and result.coordinates == []
    assert 'search budget' in result.status and 'Try another shape' not in result.status


def test_model_failure_reported():
    runner=service()
    def fail(prompt): raise RuntimeError('worker unavailable')
    runner.generator.generate=fail
    result=runner.generate('anything')
    assert result.gpx is None and 'interpreter is unavailable' in result.status


def test_concurrent_calls_serialize_search():
    runner=service()
    active=threading.Lock()
    original=runner.search.search
    def guarded(spec):
        assert active.acquire(blocking=False)
        try: return original(spec)
        finally: active.release()
    runner.search.search=guarded
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(runner.generate,['one','two']))
    assert all(r.gpx for r in results)


def test_alternative_reuses_search_and_has_separate_gpx():
    runner=service()
    first=runner.generate('heart')
    second=runner.select_route(first,1)
    assert first.gpx != second.gpx and first.alternative_index == 0
    assert runner.search.calls == 1


def test_downloads_are_isolated_and_cleaned():
    from src.web import generate_outputs
    first,second=generate_outputs(service(),'one'),generate_outputs(service(),'two')
    next(first);next(first)
    next(second);next(second)
    one,two=next(first),next(second)
    assert one[2] != two[2]
    assert 'one' in Path(one[2]).read_text() and 'two' in Path(two[2]).read_text()
    first.close()
    assert not Path(one[2]).exists() and Path(two[2]).exists()
    second.close()
    assert not Path(two[2]).exists()


def test_empty_description_clears_outputs():
    from src.web import generate_outputs
    rows=list(generate_outputs(service(),''))
    assert rows[-1][0] is None and rows[-1][2] is None
    assert 'Describe a shape' in rows[-1][3]


def test_app_builds_with_one_generation_queue_and_no_diffusion_controls():
    from src.web import create_app
    app=create_app(service())
    generation=[d for d in app.config['dependencies'] if d.get('api_name') == 'generate']
    assert len(generation)==1 and len(generation[0]['outputs'])==10
    assert len(generation[0]['inputs'])==1
    labels=[c['props'].get('label') for c in app.config['components']]
    assert 'inference steps' not in labels and 'street conditioning' not in labels


def test_gradio_copies_download_before_generator_cleanup():
    import asyncio
    from gradio.state_holder import SessionState
    from src.web import create_app
    async def exercise():
        app=create_app(service())
        fn=next(fn for fn in app.fns.values() if fn.api_name=='generate')
        state=SessionState(app)
        first=await app.process_api(fn,['heart'],state=state)
        assert first['data'][2]['value'] is None
        preview=await app.process_api(fn,[],state=state,iterator=first['iterator'])
        assert not preview['data'][0]['visible'] and preview['data'][2]['value'] is None
        routed=await app.process_api(fn,[],state=state,iterator=preview['iterator'])
        cached=Path(routed['data'][2]['value']['path'])
        assert cached.is_file()
        done=await app.process_api(fn,[],state=state,iterator=routed['iterator'])
        assert not done['is_generating'] and cached.is_file()
    asyncio.run(exercise())


def test_about_route_is_separate_from_gradio():
    from fastapi.testclient import TestClient
    from src.web import create_site
    with TestClient(create_site(service())) as client:
        response=client.get('/about')
        assert response.status_code==200
        assert 'OpenStreetMap' in response.text and 'Waschk' in response.text
        assert 'Generation trace' not in response.text

def test_four_generated_previews_precede_each_search_and_no_catalogue():
    import json
    from src.generative import VectorGenerator
    program=[['M',100,100],['L',900,100],['L',900,900],['L',100,900],['Z']]
    runner=RouteService(VectorGenerator(lambda *a,**k:json.dumps(program)),FakeSearch())
    stream=runner.stream('an asymmetric subject')
    for i in range(4):
        preview=next(stream)
        assert preview.image and preview.gpx is None and runner.search.calls==i
    result=next(stream)
    assert result.gpx and len(result.diagnostics['candidates'])==4
    assert all(c['prompt']=='an asymmetric subject' for c in result.diagnostics['candidates'])
    stream.close()


def test_invalid_generated_attempts_never_call_catalogue_or_router():
    from src.generative import VectorGenerator
    runner=RouteService(VectorGenerator(lambda *a,**k:'bad'),FakeSearch())
    result=runner.generate('horse')
    assert runner.search.calls==0 and result.gpx is None
    assert len(result.diagnostics['candidates'])==4
