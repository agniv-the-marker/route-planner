"""Named immutable geographic regions; historical SF files retain their identities."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from shapely.geometry import box, Polygon
from shapely.ops import transform
from src.geo import CRS, SF, TO_METERS
from src.graph import allowed_edge

@dataclass(frozen=True)
class Region:
    name: str
    version: str
    bounds_lonlat: tuple
    boundary_lonlat: tuple = ()

    @property
    def geographic_polygon(self):
        return Polygon(self.boundary_lonlat) if self.boundary_lonlat else box(*self.bounds_lonlat)

    @property
    def polygon(self):
        return transform(TO_METERS.transform,self.geographic_polygon)

    @property
    def graph_path(self):
        return Path('data') / f'{self.name}-{self.version}.graphml'

    def load(self):
        import osmnx as ox
        path = self.graph_path
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Region graph hash mismatch')
        graph = ox.load_graphml(path)
        if graph.graph.get('region') != self.name or graph.graph.get('region_version') != self.version or str(graph.graph.get('crs')) != CRS:
            raise ValueError('Region identity mismatch')
        if any(not allowed_edge(d) for _,_,d in graph.edges(data=True)):
            raise ValueError('Excluded access in regional graph')
        return graph

REGIONS = {'sf-palo-alto': Region('sf-palo-alto','v2',(-122.56,37.38,-122.08,37.84),
    ((-122.56,37.84),(-122.36,37.84),(-122.35,37.70),(-122.20,37.56),
     (-122.08,37.46),(-122.08,37.38),(-122.56,37.38)))}
