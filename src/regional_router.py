"""Ordered corridor matching on the Peninsula, with explicit fit rejection.

Uses directed, edge-interior anchors and shape-weighted sparse shortest paths.
This is a heuristic inspired by GPS-art and ordered curve matching literature,
not an implementation of the exact continuous Frechet algorithm.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString
from shapely import points, distance, covers

from src.geo import MapFrame, TO_GEO
from src.graph import allowed_edge, edge_coordinates
from src.regions import REGIONS
from src.strokes import Drawing
from src.street_search import RouteSearchResult, sample_line

VERSION = 'regional-ordered-corridor-v1'
REGION = REGIONS['sf-palo-alto']
_bounds = REGION.polygon.bounds
_lon, _lat = TO_GEO.transform((_bounds[0]+_bounds[2])/2, (_bounds[1]+_bounds[3])/2)
REGIONAL_FRAME = MapFrame(latitude=_lat, longitude=_lon,
                          width_m=max(_bounds[2]-_bounds[0], _bounds[3]-_bounds[1])*1.03)


@dataclass(frozen=True)
class FitConfig:
    min_distance_m: float = 50000
    max_distance_m: float = 80000
    target_lengths: tuple = (65000, 55000, 45000)
    angles: tuple = (-15, -7.5, 0, 7.5, 15)
    grid_m: float = 1000
    placements: int = 100
    seconds: float = 90
    anchor_spacing_m: float = 1200
    candidates: int = 6
    max_relative_error: float = .035
    mean_relative_error: float = .012


def discrete_frechet(a, b):
    """Ordered sample distance; unlike nearest-line distance it detects shortcuts."""
    distances = cdist(a, b)
    previous = np.full(len(b), np.inf)
    for i, row in enumerate(distances):
        current = np.full(len(b), np.inf)
        for j, value in enumerate(row):
            if i == j == 0:
                current[j] = value
            else:
                current[j] = max(value, min(previous[j], previous[j-1] if j else np.inf,
                                            current[j-1] if j else np.inf))
        previous = current
    return float(previous[-1])


class FitTimeout(Exception):
    pass


class RegionalOutlineSearch:
    def __init__(self, graph, frame=REGIONAL_FRAME, region=REGION, config=None):
        if not graph.is_directed() or not graph.is_multigraph() or not len(graph):
            raise ValueError('A nonempty directed multigraph is required')
        self.frame, self.region, self.graph = frame, region, graph
        self.config = config or FitConfig()
        node_ids = {n: i for i, n in enumerate(graph.nodes)}
        xy = [[d['x'], d['y']] for _, d in graph.nodes(data=True)]
        self.edges = []
        digest = hashlib.sha256(VERSION.encode())
        for u, v, k, data in graph.edges(keys=True, data=True):
            if not allowed_edge(data):
                continue
            geom = edge_coordinates(graph, u, v, data)
            ds = np.r_[0, np.cumsum(np.linalg.norm(np.diff(geom, axis=0), axis=1))]
            length = float(data['length'])
            if ds[-1] <= 0 or length <= 0 or not np.isfinite(length):
                raise ValueError('Invalid road geometry or length')
            digest.update(repr((u, v, k, length, geom.tolist())).encode())
            # Split only within each original directed edge. Nearby bridges and
            # parallel carriageways never acquire an artificial connection.
            count = max(1, int(np.ceil(ds[-1]/180)))
            cuts = np.linspace(0, ds[-1], count+1)
            q = np.column_stack([np.interp(cuts, ds, geom[:, axis]) for axis in (0, 1)])
            start = node_ids[u]
            for i in range(count):
                end = node_ids[v] if i == count-1 else len(xy)
                if i != count-1:
                    xy.append(q[i+1].tolist())
                lo, hi = np.searchsorted(ds, cuts[i], side='right'), np.searchsorted(ds, cuts[i+1], side='left')
                piece = np.vstack([q[i], geom[lo:hi], q[i+1]])
                physical = tuple(map(tuple, np.round(piece, 2)))
                self.edges.append((start, end, length/count, piece, (u, v, k), min(physical, physical[::-1])))
                start = end
        self.xy = np.asarray(xy, dtype=float)
        self.graph_hash = digest.hexdigest()
        self._prepare_arrays()

    def _prepare_arrays(self):
        self.tree = cKDTree(self.xy)
        self.uv = np.asarray([(e[0], e[1]) for e in self.edges], dtype=int)
        self.lengths = np.asarray([e[2] for e in self.edges])
        # Actual curved geometry is sampled for fitting; rendering retains all vertices.
        self.samples = np.asarray([sample_line(e[3], 5) for e in self.edges])
        self.midpoints = self.samples[:, 2]
        self.edge_tree = cKDTree(self.midpoints)
        self.edge_radius = float(np.linalg.norm(self.samples-self.midpoints[:,None,:], axis=2).max())
        self.road_tree = cKDTree(self.samples.reshape(-1, 2))
        self.rejections = {}

    @classmethod
    @lru_cache(maxsize=1)
    def load_prepared(cls):
        return cls(REGION.load())

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise FitTimeout('wall_timeout')

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0)+1
        return None

    @staticmethod
    def transform(spec, x, y, angle, perimeter):
        if abs(angle) > 15:
            raise ValueError('Orientation must stay within ±15°')
        p = np.asarray(spec.points, dtype=float).copy()
        p -= (p.max(0)+p.min(0))/2
        p[:, 1] *= -1
        a = np.deg2rad(angle)
        p = p @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return p*perimeter/np.linalg.norm(np.diff(p, axis=0), axis=1).sum()+[x,y]

    def _transitions(self, a, b, sources, targets, leash):
        """Joint endpoint alternatives, with deviation paid *inside* path search."""
        self.check()
        vector = b-a
        arc = float(np.linalg.norm(vector))
        idx = np.asarray(self.edge_tree.query_ball_point((a+b)/2, arc/2+leash+self.edge_radius), dtype=int)
        if not len(idx):
            return {}
        q = self.samples[idx]
        progress = np.clip(((q-a)*vector).sum(2)/max(arc*arc,1e-12),0,1)
        deviations = np.linalg.norm(q-(a+progress[:,:,None]*vector), axis=2)
        keep = deviations.max(1) <= leash
        idx, deviations = idx[keep], deviations[keep]
        if not len(idx):
            return {}
        tangent = self.samples[idx,-1]-self.samples[idx,0]
        cosine = (tangent*vector).sum(1)/np.maximum(np.linalg.norm(tangent,axis=1)*arc,1e-12)
        costs = self.lengths[idx]*(1+12*(deviations.mean(1)/max(leash*.45,1))**2+1.5*(1-cosine))
        # scipy sparse sums parallel entries: explicitly select the cheapest
        # directed edge for each node pair for this particular target segment.
        uv = self.uv[idx]
        order = np.lexsort((costs, uv[:,1], uv[:,0]))
        unique = np.r_[True, np.any(np.diff(uv[order],axis=0),axis=1)]
        chosen = order[unique]
        uv, idx, costs = uv[chosen], idx[chosen], costs[chosen]
        nodes = np.unique(np.r_[uv.ravel(), sources, targets])
        local_uv = np.searchsorted(nodes, uv)
        matrix = csr_matrix((costs,(local_uv[:,0],local_uv[:,1])),shape=(len(nodes),len(nodes)))
        local_sources, local_targets = np.searchsorted(nodes,sources),np.searchsorted(nodes,targets)
        distances, predecessors = dijkstra(matrix, directed=True, indices=local_sources, return_predecessors=True)
        lookup = {(int(u),int(v)):int(e) for (u,v),e in zip(local_uv,idx)}
        transitions = {}
        for row, source in enumerate(sources):
            for target, local_target in zip(targets,local_targets):
                cost = distances[row,local_target]
                if not np.isfinite(cost):
                    continue
                current, path = int(local_target), []
                while current != local_sources[row]:
                    previous = int(predecessors[row,current])
                    if previous < 0:
                        raise ValueError('Broken directed predecessor chain')
                    path.append(lookup[previous,current]);current=previous
                path.reverse()
                length = float(self.lengths[path].sum())
                if length > arc*2+2*leash:
                    continue
                transitions[int(source),int(target)] = (float(cost),length,path)
        return transitions

    def match(self, p):
        config = self.config
        span = float(np.ptp(p,axis=0).max())
        leash = min(450, max(100, span*.025))
        anchors = [p[0]]
        for a,b in zip(p,p[1:]):
            anchors.extend(np.linspace(a,b,max(1,int(np.ceil(np.linalg.norm(b-a)/config.anchor_spacing_m)))+1)[1:])
        anchors = np.asarray(anchors)
        ds, ids = self.tree.query(anchors,k=min(config.candidates,len(self.xy)),distance_upper_bound=leash*.8)
        ds, ids = np.asarray(ds).reshape(len(anchors),-1),np.asarray(ids).reshape(len(anchors),-1)
        candidates = [[int(i) for d,i in zip(dd,ii) if np.isfinite(d)] for dd,ii in zip(ds,ids)]
        if any(not c for c in candidates):
            return self.reject('anchor_outside_corridor')
        closed = np.linalg.norm(p[-1]-p[0]) < 1
        if closed:
            candidates[-1] = candidates[0]
        # Keep start identity for exact closure; retain independent alternatives
        # at every anchor instead of greedily snapping to one street.
        states = {(s,s):(0.,0.,[]) for s in candidates[0]}
        for i,(a,b) in enumerate(zip(anchors,anchors[1:])):
            transitions = self._transitions(a,b,candidates[i],candidates[i+1],leash)
            next_states = {}
            for (origin,source),(cost,length,path) in states.items():
                for target in candidates[i+1]:
                    if closed and i == len(anchors)-2 and target != origin:
                        continue
                    option = transitions.get((source,target))
                    if option is None:
                        continue
                    extra,travel,segment = option
                    if length+travel > config.max_distance_m:
                        continue
                    key = (origin,target)
                    proposal = (cost+extra,length+travel,path+segment)
                    if key not in next_states or proposal[0] < next_states[key][0]:
                        next_states[key] = proposal
            states = next_states
            if not states:
                return self.reject('no_ordered_directed_path')
        options = sorted(states.values(),key=lambda s:s[0])
        for _,length,path in options:
            if path and config.min_distance_m <= length <= config.max_distance_m:
                return self._route(path,p)
        return self.reject('ride_outside_50_80km')

    def _route(self, path, p):
        if any(self.edges[a][1] != self.edges[b][0] for a,b in zip(path,path[1:])):
            raise ValueError('Disconnected route')
        xy = np.vstack([self.edges[e][3][:-1] for e in path]+[self.edges[path[-1]][3][-1:]])
        length = float(self.lengths[path].sum())
        source, line = LineString(p),LineString(xy)
        a, b = sample_line(p,512),sample_line(xy,512)
        forward, backward = distance(points(b),source),distance(points(a),line)
        feature = distance(points(p),line)
        span = float(np.ptp(p,axis=0).max())
        mean = float((forward.mean()+backward.mean())/2)
        maximum = float(max(forward.max(),backward.max(),feature.max()))
        ordered = discrete_frechet(a,b)
        physical = {self.edges[e][5]:self.edges[e][2] for e in path}
        repeated = 1-sum(physical.values())/length
        if maximum/span > self.config.max_relative_error or mean/span > self.config.mean_relative_error:
            return self.reject('shape_deviation')
        if ordered/span > .05:
            return self.reject('ordered_shape_deviation')
        directed = []
        for e in path:
            original = self.edges[e][4]
            if not directed or directed[-1] != original:
                directed.append(original)
        scores = {'line_distance_m':mean,'feature_max_m':float(feature.max()),'max_distance_m':maximum,
                  'relative_mean_error':mean/span,'relative_max_error':maximum/span,
                  'discrete_frechet_m':ordered,'relative_frechet':ordered/span,'repeated':repeated,
                  'rank':mean/span+.15*ordered/span}
        return {'xy':xy.tolist(),'target_xy':p.tolist(),'edges':directed,'index_edges':path,
                'distance_m':length,'scores':scores}

    @staticmethod
    def rank(route):
        # Larger rides win only within a comparable geometric fit band.
        return (round(route['scores']['rank']/.002),-route['distance_m'],route['scores']['rank'])

    def _placements(self,spec):
        config = self.config
        bounds = self.region.polygon.bounds if self.region else self.frame.bounds
        left,bottom,right,top = bounds
        gx,gy = np.meshgrid(np.arange(left,right,config.grid_m),np.arange(bottom,top,config.grid_m))
        centers = np.column_stack([gx.ravel(),gy.ravel()])
        placements = []
        for length in config.target_lengths:
            for angle in config.angles:
                self.check()
                p = self.transform(spec,0,0,angle,length)
                valid = ((centers+p.min(0)>=[left,bottom]).all(1)&(centers+p.max(0)<=[right,top]).all(1))
                cs = centers[valid]
                samples = sample_line(p,96)
                for start in range(0,len(cs),128):
                    self.check()
                    chunk = cs[start:start+128]
                    q = chunk[:,None,:]+samples
                    ds = self.road_tree.query(q.reshape(-1,2))[0].reshape(len(chunk),-1)
                    span = np.ptp(p,axis=0).max()
                    score = (ds.mean(1)+.5*np.quantile(ds,.95,axis=1)+.15*ds.max(1))/span
                    placements.extend((float(s),float(x),float(y),float(angle),float(length)) for s,(x,y) in zip(score,chunk))
        selected = []
        for item in sorted(placements):
            if all(np.hypot(item[1]-v[1],item[2]-v[2])>=800 or item[4]!=v[4] or abs(item[3]-v[3])>=10 for v in selected):
                selected.append(item)
            if len(selected)>=config.placements:
                break
        return selected

    def search(self,spec,budget=None):
        started = time.perf_counter()
        self.deadline = started+min(self.config.seconds,budget if budget is not None else self.config.seconds)
        self.rejections = {}
        result = RouteSearchResult(spec)
        result.graph = self.graph
        tested = 0
        try:
            proposals = self._placements(spec)
            result.timings['placement_seconds']=time.perf_counter()-started
            result.placements = [dict(zip(('field_distance','x','y','angle','perimeter'),q)) for q in proposals]
            for _,x,y,angle,length in proposals:
                self.check()
                p = self.transform(spec,x,y,angle,length)
                if self.region and not self.region.polygon.covers(LineString(p)):
                    self.reject('outside_region');continue
                for reverse in (False,True):
                    self.check();tested+=1
                    target = p[::-1] if reverse else p
                    route = self.match(target)
                    if route:
                        route['transform']={'x':x,'y':y,'angle':angle,'perimeter':length,'mirrored':False,'reversed':reverse}
                        if all(route['index_edges']!=r['index_edges'] for r in result.routes):
                            result.routes.append(route)
                            result.routes=sorted(result.routes,key=self.rank)[:3]
            result.diagnostics['status']='completed'
        except FitTimeout:
            result.diagnostics.update(status='incomplete',limit='wall_timeout')
        result.timings['search_seconds']=time.perf_counter()-started
        result.diagnostics.update(version=VERSION,graph_hash=self.graph_hash,region='sf-palo-alto-v2',
                                  candidates_tested=tested,rejections=dict(self.rejections),
                                  min_distance_m=self.config.min_distance_m,max_distance_m=self.config.max_distance_m)
        if not result.routes:
            result.failure_reason='No sufficiently close 50–80 km fit found within this search. Your drawing has been retained.'
        return result

    def search_drawing(self,drawing,budget=None):
        from types import SimpleNamespace
        p,_ = drawing.continuous()
        return self.search(SimpleNamespace(points=p),budget)
