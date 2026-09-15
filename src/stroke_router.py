"""Directed adaptation of Waschk/Krüger: deviation enters path search itself.

Whole original edges are retained, including parallel edges and curved geometry.
Anchor layers preserve stroke order; a beam jointly chooses connected transitions.
This is a bounded heuristic, not an exact reproduction or an optimality claim.
"""
from dataclasses import dataclass, asdict
import heapq
import time
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely import points, distance
from src.graph import allowed_edge, edge_coordinates
from src.street_search import sample_line
from src.experiment_journal import digest

VERSION = 'stroke-router-v1'

@dataclass(frozen=True)
class RoutingConfig:
    version: str = 'routing-v1'
    max_distance_m: float = 30000
    min_distance_m: float = 0
    wall_seconds: float = 120
    work_budget: int = 600000
    angles: tuple = (-15, 0, 15)
    span_fractions: tuple = (.08, .15, .25)
    anchor_spacing_m: float = 400
    snap_m: float = 300
    candidates: int = 3
    beam_width: int = 12
    placements: int = 12
    grid_m: float = 2000
    deviation_weight: float = 16
    deviation_scale_m: float = 100
    closed: bool = False
    max_repeated: float | None = None
    quality_tie_m: float = 10
    # Interactive callers may retain a larger ranked pool for diversity
    # selection. The research/default workflow remains top-three.
    diverse_candidates: bool = False
    candidate_pool: int = 3

    def __post_init__(self):
        if not (0 <= self.min_distance_m < self.max_distance_m and 0 < self.wall_seconds <= 120 and self.work_budget > 0):
            raise ValueError('Invalid ride or search limits')
        if not self.angles or any(abs(a) > 15 for a in self.angles):
            raise ValueError('Angles must stay within ±15°')
        if any(v <= 0 for v in (self.anchor_spacing_m,self.snap_m,self.candidates,self.beam_width,self.placements,self.grid_m,self.deviation_scale_m,self.quality_tie_m,self.candidate_pool)) or self.deviation_weight < 0:
            raise ValueError('Search costs and resolutions must be nonnegative/positive')
        if not self.span_fractions or any(f <= 0 for f in self.span_fractions):
            raise ValueError('Positive footprint scales required')

class SearchLimit(Exception):
    pass

class StrokeRouter:
    def __init__(self, graph, region=None):
        if not graph.is_directed() or not graph.is_multigraph() or not len(graph):
            raise ValueError('Nonempty directed multigraph required')
        self.graph, self.region = graph, region
        self.nodes = sorted(graph.nodes)
        self.xy = np.array([[graph.nodes[n]['x'],graph.nodes[n]['y']] for n in self.nodes])
        self.tree = cKDTree(self.xy)
        self.edges, self.adj = {}, {n: [] for n in self.nodes}
        for u,v,k,d in sorted(graph.edges(keys=True,data=True), key=lambda e: repr(e[:3])):
            if not allowed_edge(d):
                continue
            p = edge_coordinates(graph,u,v,d)
            length = float(d['length'])
            if not np.isfinite(length) or length <= 0:
                raise ValueError('Edge lengths must be positive and finite')
            geom = LineString(p)
            if region is not None and not region.polygon.covers(geom):
                continue
            eid = (u,v,k)
            physical = tuple(map(tuple, np.round(p,2)))
            self.edges[eid] = (length,p, sample_line(p,max(3,int(np.ceil(geom.length/40))+1)), min(physical,physical[::-1]))
            self.adj[u].append(eid)
        self.graph_hash = digest([(e,d[0],d[1].tolist()) for e,d in self.edges.items()])

    def check(self):
        self.used += 1
        if self.used > self.config.work_budget:
            raise SearchLimit('work_budget')
        if time.perf_counter() >= self.deadline:
            raise SearchLimit('wall_timeout')

    def start(self, config):
        self.config = config
        self.used = 0
        self.deadline = time.perf_counter()+config.wall_seconds

    def paths(self, start, targets, arc, remaining, variant):
        """Pareto labels preserve cheaper/longer and dearer/shorter alternatives."""
        # Keep Dijkstra in a corridor around the requested stroke segment. This
        # prevents a regional graph search from expanding across the entire
        # Peninsula while still allowing a kilometre-scale bicycle detour.
        minx, miny, maxx, maxy = arc.bounds
        corridor = max(1000.0, self.config.snap_m * 3)
        minx -= corridor; miny -= corridor; maxx += corridor; maxy += corridor
        labels = [(0.,0.,start,None,None)]
        active = {start: [0]}
        queue = [(0.,0)]
        found, costs = {}, {}
        while queue:
            self.check()
            cost, lid = heapq.heappop(queue)
            _, traveled, u, parent, edge = labels[lid]
            if lid not in active.get(u, []):
                continue
            if u in targets:
                path, cur = [], lid
                while labels[cur][3] is not None:
                    path.append(labels[cur][4]); cur = labels[cur][3]
                found.setdefault(u, []).append((cost,traveled,path[::-1]))
                # Keep searching for feasible shorter alternatives under the ride bound.
            for eid in self.adj[u]:
                self.check()
                length,xy,samples,_ = self.edges[eid]
                endpoint = self.xy[self.nodes_index(eid[1])]
                if not (minx <= endpoint[0] <= maxx and miny <= endpoint[1] <= maxy):
                    continue
                nl = traveled+length
                if nl > remaining:
                    continue
                if eid not in costs:
                    deviation = float(distance(points(samples),arc).mean())
                    costs[eid] = length*(1 + (self.config.deviation_weight*deviation/self.config.deviation_scale_m if variant == 'paper' else 0))
                nc, v = cost+costs[eid], eid[1]
                old = active.get(v, [])
                if any(labels[i][0] <= nc and labels[i][1] <= nl for i in old):
                    continue
                old = [i for i in old if not (nc <= labels[i][0] and nl <= labels[i][1])]
                nid = len(labels)
                labels.append((nc,nl,v,lid,eid))
                active[v] = old+[nid]
                heapq.heappush(queue,(nc,nid))
            # Once every target has a cheapest solution, return these solutions.
            # Additional Pareto labels found before that point remain available.
            if targets <= found.keys():
                break
        return found

    def match(self, p, variant='paper'):
        if variant not in {'paper','expanded'}:
            raise ValueError('Unknown expanded solver')
        anchors = [p[0]]
        # Split each stroke segment separately: retain every corner and connector.
        for a,b in zip(p,p[1:]):
            anchors.extend(np.linspace(a,b,max(1,int(np.ceil(np.linalg.norm(b-a)/self.config.anchor_spacing_m)))+1)[1:])
        if self.config.closed and not np.allclose(anchors[0],anchors[-1]):
            anchors.append(anchors[0])
        ds, ids = self.tree.query(anchors,k=min(self.config.candidates,len(self.nodes)),distance_upper_bound=self.config.snap_m)
        ds, ids = np.asarray(ds).reshape(len(anchors),-1), np.asarray(ids).reshape(len(anchors),-1)
        candidates = [[self.nodes[j] for d,j in zip(dd,ii) if np.isfinite(d)] for dd,ii in zip(ds,ids)]
        if any(not c for c in candidates):
            return None, 'no_anchor_within_snap_limit'
        states = [(0.,0.,n,[],n) for n in candidates[0]]
        for layer,(a,b) in enumerate(zip(anchors,anchors[1:])):
            self.check()
            arc = LineString([a,b])
            choices = []
            for cost, length, source, path, origin in states:
                targets = set(candidates[layer+1])
                if self.config.closed and layer == len(anchors)-2:
                    targets &= {origin}
                # Arc search bound controls detours, but never truncates the ride limit.
                remaining = min(self.config.max_distance_m-length, arc.length*4+2*self.config.snap_m)
                for target, options in self.paths(source,targets,arc,remaining,variant).items():
                    for extra,travel,segment in options:
                        if variant == 'expanded':
                            q = np.vstack([self.edges[e][2] for e in segment]) if segment else np.array([a])
                            extra = float(distance(points(q),arc).mean())+abs(travel-arc.length)*.35
                        # Penalize missing anchors even when consecutive anchors snap to one node.
                        snap = np.linalg.norm(self.xy[self.nodes_index(target)]-b)
                        choices.append((cost+extra+snap,length+travel,target,path+segment,origin))
            states = sorted(choices, key=lambda s:(s[0],s[1],s[2]))[:self.config.beam_width]
            if not states:
                return None, 'no_directed_transition'
        routes = []
        for _,length,_,edges,_ in states:
            if not edges or not self.config.min_distance_m <= length <= self.config.max_distance_m:
                continue
            route = self.render_route(edges,p)
            if self.config.max_repeated is None or route['scores']['repeated'] <= self.config.max_repeated:
                routes.append(route)
        return (min(routes,key=self.rank),None) if routes else (None,'distance_or_repetition_limit')

    def nodes_index(self, node):
        if not hasattr(self,'node_index'):
            self.node_index = {n:i for i,n in enumerate(self.nodes)}
        return self.node_index[node]

    def rank(self, route):
        s = route['scores']
        return (round(s['line_distance_m']/self.config.quality_tie_m),round(s['feature_max_m']/self.config.quality_tie_m),route['distance_m'],s['repeated'])

    def render_route(self, edges, target):
        if any(a[1] != b[0] for a,b in zip(edges,edges[1:])):
            raise ValueError('Disconnected directed ride')
        xy = np.vstack([self.edges[e][1][:-1] for e in edges]+[self.edges[edges[-1]][1][-1:]])
        line, reference = LineString(xy), LineString(target)
        length = sum(self.edges[e][0] for e in edges)
        physical = {self.edges[e][3]:self.edges[e][0] for e in edges}
        forward = float(distance(points(sample_line(xy,512)),reference).mean())
        backward = float(distance(points(sample_line(target,512)),line).mean())
        features = distance(points(target),line)
        return {'xy':xy.tolist(),'edges':[list(e) for e in edges], 'distance_m':length,
                'target_xy':target.tolist(), 'scores':{'line_distance_m':(forward+backward)/2,
                'route_to_target_m':forward,'target_to_route_m':backward,
                'feature_max_m':float(features.max()), 'feature_distances_m':features.tolist(),
                'repeated':1-sum(physical.values())/length,
                'footprint_width_m':float(np.ptp(xy[:,0])), 'footprint_height_m':float(np.ptp(xy[:,1])),
                'footprint_area_m2':float(line.convex_hull.area)}}

    def search(self, drawing, config, variant='paper'):
        self.start(config)
        started = time.perf_counter()
        result = {'version':VERSION,'variant':variant,'config':asdict(config),'graph_hash':self.graph_hash,
                  'drawing_hash':drawing.key(),'routes':[], 'rejections':{},'status':'completed'}
        try:
            retained = config.candidate_pool if config.diverse_candidates else 3
            low, high = self.xy.min(0),self.xy.max(0)
            if self.region is not None:
                left,bottom,right,top = self.region.polygon.bounds
                low,high = np.array([left,bottom]),np.array([right,top])
            # Overlapping coarse windows nominate placements; refinement uses the full graph.
            candidates = []
            for x in np.arange(low[0],high[0]+1,config.grid_m):
                for y in np.arange(low[1],high[1]+1,config.grid_m):
                    for fraction in config.span_fractions:
                        for angle in config.angles:
                            self.check()
                            span = config.max_distance_m*fraction
                            p,_ = drawing.place([x,y],span,angle)
                            if (p.min(0)<low).any() or (p.max(0)>high).any():
                                continue
                            if self.region is not None and not self.region.polygon.covers(LineString(p)):
                                continue
                            score = float(self.tree.query(sample_line(p,64))[0].mean())/span
                            candidates.append((score,x,y,span,angle))
            selected = []
            for item in sorted(candidates):
                if all(np.hypot(item[1]-q[1],item[2]-q[2]) >= config.grid_m/2 or item[3] != q[3] for q in selected):
                    selected.append(item)
                if len(selected) >= config.placements:
                    break
            result['placements'] = [list(q) for q in selected]
            for _,x,y,span,angle in selected:
                # Re-match nearby placements: connected transitions and local subpaths are recomputed.
                for dx,dy in ((0,0),(-config.snap_m/2,0),(config.snap_m/2,0),(0,-config.snap_m/2),(0,config.snap_m/2)):
                    self.check()
                    p,connectors = drawing.place([x+dx,y+dy],span,angle)
                    if self.region is not None and not self.region.polygon.covers(LineString(p)):
                        continue
                    route,reason = self.match(p,variant)
                    if route:
                        route.update(transform={'x':x+dx,'y':y+dy,'span_m':span,'angle':angle,'mirrored':False},connector_segments=connectors)
                        result['routes'].append(route)
                        result['routes'] = sorted(result['routes'],key=self.rank)[:retained]
                    else:
                        result['rejections'][reason] = result['rejections'].get(reason,0)+1
        except SearchLimit as exc:
            result.update(status='incomplete',limit=str(exc))
        result.update(work_used=self.used,seconds=time.perf_counter()-started)
        if not result['routes']:
            result['failure_reason'] = 'No feasible ride found within the recorded search limits.'
        return result
