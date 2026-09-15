"""Prepare the Peninsula separately; preserve data/sf.graphml and historical helpers."""
import argparse
from dataclasses import asdict
import hashlib
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from src.geo import CRS
from src.regions import REGIONS
from src.graph import allowed_edge, edge_coordinates
from src.experiment_journal import atomic_json, now, source_identity

def prepare(region, source=None):
    path = region.graph_path
    if path.exists():
        region.load()
        print(f'Validated existing {path}; immutable region versions are never overwritten.')
        return
    if source is not None:
        graph = ox.load_graphml(source)
        if str(graph.graph.get('crs')) != CRS:
            raise ValueError('Parent graph must use the projected regional CRS')
    else:
        ox.settings.use_cache = True
        ox.settings.cache_folder = '.cache/osmnx'
        ox.settings.requests_timeout = 120
        ox.settings.log_console = True
        ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | {'bicycle','route','access:conditional','bicycle:conditional','oneway:bicycle','bicycle:forward','bicycle:backward','oneway:conditional','oneway:bicycle:conditional'})
        graph = ox.graph_from_polygon(region.geographic_polygon,network_type='bike',simplify=False,retain_all=True,truncate_by_edge=True)
        for u,v,k,d in list(graph.edges(keys=True,data=True)):
            reverse = d.get('reversed',False)
            direction = d.get('oneway:bicycle')
            if not allowed_edge(d) or (reverse and direction in {'yes','1','true'}) or (not reverse and direction == '-1') or d.get('bicycle:backward' if reverse else 'bicycle:forward') in {'no','dismount'}:
                graph.remove_edge(u,v,k)
        graph.remove_nodes_from(list(nx.isolates(graph)))
        graph = ox.simplify_graph(graph,edge_attrs_differ=['access','bicycle','oneway:bicycle'])
        graph = ox.project_graph(graph,to_crs=CRS)
    for u,v,k,d in list(graph.edges(keys=True,data=True)):
        geometry = LineString(edge_coordinates(graph,u,v,d))
        if not allowed_edge(d) or not region.polygon.covers(geometry):
            graph.remove_edge(u,v,k)
        else:
            d.update(geometry=geometry,length=geometry.length)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if not len(graph):
        raise ValueError('Empty regional bicycle graph')
    graph.graph.update(region=region.name,region_version=region.version,downloaded_at=now())
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.pending.graphml')
    ox.save_graphml(graph,tmp)
    tmp.replace(path)
    atomic_json(path.with_suffix('.json'),{'region':asdict(region),'parent_graph':str(source) if source else None,'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source else None,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':source_identity(),'nodes':len(graph),'edges':graph.number_of_edges(),'downloaded_at':now(),'osmnx':ox.__version__,'crs':CRS,'network_type':'bike','attribution':'© OpenStreetMap contributors, ODbL 1.0','limitations':'No turn restriction relations or live closures; conservative access exclusions.'})
    from scripts.routing_references import svg
    strokes = [edge_coordinates(graph,u,v,d)*[1,-1] for u,v,d in graph.edges(data=True)]
    path.with_suffix('.svg').write_text(svg(strokes))
    print(path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--region',choices=REGIONS,default='sf-palo-alto')
    from pathlib import Path
    p.add_argument('--source',type=Path,help='Clip an already filtered projected parent graph without another download')
    args=p.parse_args()
    prepare(REGIONS[args.region],args.source)
