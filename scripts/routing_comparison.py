"""Approval-gated paired routing comparison. No inference or paid compute."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from src.experiment_journal import Journal, atomic_json, digest, source_identity, now
from src.strokes import Drawing
from src.stroke_router import StrokeRouter, RoutingConfig
from src.regions import REGIONS


def approved_manifest(output):
    manifest=json.loads((output/'references.json').read_text())
    if digest(manifest['drawings']) != manifest['drawings_hash']:
        raise ValueError('Source drawings changed since freeze')
    approval=manifest.get('source_acceptance')
    if not approval or approval.get('drawings_hash') != manifest['drawings_hash'] or not approval.get('evidence'):
        raise ValueError('Source drawings require explicit user acceptance before routing')
    return manifest


def accept(output,evidence):
    if not evidence.strip():raise ValueError('Record the actual user acceptance message')
    path=output/'references.json'
    manifest=json.loads(path.read_text())
    if digest(manifest['drawings']) != manifest['drawings_hash']:
        raise ValueError('Source drawings changed')
    manifest.update(status='sources_accepted',source_acceptance={'drawings_hash':manifest['drawings_hash'],'evidence':evidence,'recorded':now()})
    atomic_json(path,manifest)


def run(output,split,config_path):
    manifest=approved_manifest(output)
    config_data=json.loads(config_path.read_text())
    config=RoutingConfig(**config_data)
    freeze_path=output/'comparison-freeze.json'
    if split=='comparison':
        freeze={'config':config_data,'drawings_hash':manifest['drawings_hash']}
        if not freeze_path.exists() or json.loads(freeze_path.read_text()) != freeze:
            raise ValueError('Freeze tuned parameters before running the eight comparison drawings')
    region=REGIONS['sf-palo-alto']
    graph=region.load()
    router=StrokeRouter(graph,region)
    journal=Journal(output/'journal')
    source=source_identity()
    from src.street_search import StreetSearch
    from src.outlines import OutlineSpec
    historic=StreetSearch.load_prepared()
    results=[]
    for row in manifest['drawings']:
        if row['split']!=split:continue
        drawing=Drawing.parse(row['drawing'])
        raw_hash=journal.raw({'reference':row['id'],'version':manifest['version']},row)
        for km in manifest['distance_km']:
            from dataclasses import replace
            cfg=replace(config,max_distance_m=km*1000)
            for variant in ('historical','expanded','paper'):
                identity={'source':source,'graph_hash':historic.graph_hash if variant=='historical' else router.graph_hash,
                          'config':{'wall_seconds':8,'work_budget':None,'settings':'street-search-v8-historical'} if variant=='historical' else asdict(cfg),
                          'raw_hash':raw_hash,'variant':variant,'region':'historical-sf' if variant=='historical' else asdict(region),'seed':0}
                attempt={'drawing':row['id'],'distance_km':km,'variant':variant}
                def operation():
                    if variant=='historical':
                        if len(drawing.strokes)!=1:
                            return {'status':'excluded','reason':'Historical closed-outline contract cannot represent internal strokes or components','routes':[]}
                        try:spec=OutlineSpec.parse({'interpretation':drawing.name,'points':list(drawing.strokes[0])})
                        except ValueError as exc:return {'status':'excluded','reason':str(exc),'routes':[]}
                        # Historical configuration remains exactly the original 3–15km, 8s search.
                        historic.route_cache.clear();historic.route_cache_dir=None
                        found=historic.search(spec,budget=8)
                        result=asdict(found);result.pop('outline')
                        result['status']='incomplete' if found.diagnostics.get('timed_out') else 'completed'
                        for route in result['routes']:
                            t=route['transform']
                            route['target_xy']=historic.transform(spec,t['x'],t['y'],t['angle'],t['perimeter']).tolist()
                        return result
                    return router.search(drawing,cfg,variant)
                result=journal.run(attempt,'route',identity,operation)
                row_result={'attempt':attempt,'identity_hash':digest(identity),'result':result,'review':{'recognition':'unreviewed','features':'unreviewed'}}
                directory=output/'comparisons'/digest([attempt,identity])
                directory.mkdir(parents=True,exist_ok=True)
                export(result,directory,graph)
                atomic_json(directory/'result.json',row_result)
                results.append(row_result)
                atomic_json(output/f'{split}-results.json',results)
                print(attempt,result['status'],len(result['routes']),flush=True)
    return results


def export(result,output,graph):
    from scripts.routing_references import svg
    from src.generation import to_gpx
    from src.geo import TO_GEO
    from src.graph import edge_coordinates
    from shapely.geometry import box, LineString
    cards=[]
    for i,route in enumerate(result['routes']):
        xy=np.asarray(route['xy']);target=np.asarray(route['target_xy'])
        (output/f'{i}-plain.svg').write_text(svg([xy*[1,-1]]))
        (output/f'{i}-overlay.svg').write_text(svg([xy*[1,-1]],[target*[1,-1]]))
        bounds=box(*(LineString(xy).buffer(500).bounds))
        streets=[edge_coordinates(graph,u,v,d)*[1,-1] for u,v,d in graph.edges(data=True) if bounds.intersects(LineString(edge_coordinates(graph,u,v,d)))]
        (output/f'{i}-map.svg').write_text(svg([xy*[1,-1]],[target*[1,-1]],streets))
        lon,lat=TO_GEO.transform(xy[:,0],xy[:,1])
        (output/f'{i}.gpx').write_text(to_gpx(list(zip(lat,lon)),'Anonymous routing comparison'))
        cards.append(f'<article><img width="400" src="{i}-plain.svg"><details><summary>Reveal overlay and map</summary><img width="400" src="{i}-overlay.svg"><img width="400" src="{i}-map.svg"><p>{route["distance_m"]/1000:.2f} km; repeated fraction {route["scores"]["repeated"]:.3f}</p><a href="{i}.gpx">GPX</a></details></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Anonymous route review</title>'+''.join(cards))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['accept','freeze','tuning','comparison']);p.add_argument('--output',type=Path,default=Path('outputs/routing-first-v2'));p.add_argument('--config',type=Path,default=Path('configs/routing-v1.json'));p.add_argument('--evidence')
    a=p.parse_args()
    if a.action=='accept':accept(a.output,a.evidence or '')
    elif a.action=='freeze':
        manifest=approved_manifest(a.output)
        cfg=json.loads(a.config.read_text());RoutingConfig(**cfg)
        if (a.output/'comparison-freeze.json').exists():raise ValueError('Comparison is already frozen; create a new run for revisions')
        atomic_json(a.output/'comparison-freeze.json',{'config':cfg,'drawings_hash':manifest['drawings_hash']})
    else:run(a.output,a.action,a.config)
