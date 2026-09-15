"""Bounded placement and joint directed matching, independent of model inference."""
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import heapq
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.ops import substring
from src.geo import SF
from src.graph import edge_coordinates

VERSION = 'street-search-v8'

def sample_line(p, count, endpoint=True):
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    d = np.linspace(0, lengths[-1], count, endpoint=endpoint)
    return np.column_stack([np.interp(d, lengths, p[:, i]) for i in (0, 1)])

@dataclass
class RouteSearchResult:
    outline: object
    routes: list = field(default_factory=list)
    placements: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    failure_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

class Deadline(Exception):
    pass

#: Historical website/experiment settings. Callers that want longer rides pass their own.
DEFAULT_PERIMETERS = (3000, 5000, 8000, 11000, 14000)
DEFAULT_DISTANCE_RANGE = (3000, 15000)


class StreetSearch:
    def __init__(self, graph, frame=SF, perimeters=DEFAULT_PERIMETERS,
                 distance_range=DEFAULT_DISTANCE_RANGE, prefer='balanced'):
        started = time.perf_counter()
        if not graph.is_directed() or not graph.is_multigraph() or not graph.nodes:
            raise ValueError('A nonempty directed multigraph is required.')
        self.frame = frame
        self.set_distance_band(perimeters, distance_range, prefer)
        self.xy = [[float(d['x']), float(d['y'])] for _, d in graph.nodes(data=True)]
        self.node_count = len(self.xy)
        nodes = dict(zip(graph.nodes, range(len(self.xy))))
        self.adj = [[] for _ in self.xy]
        self.edges = []
        digest = hashlib.sha256(VERSION.encode())
        # Each directed edge owns its interior vertices: geometric crossings never connect.
        for u, v, key, data in graph.edges(keys=True, data=True):
            xy = edge_coordinates(graph, u, v, data)
            line = LineString(xy)
            if not np.isfinite(float(data['length'])) or float(data['length']) <= 0 or line.length <= 0:
                raise ValueError('Street edges must have positive finite lengths.')
            digest.update(repr((u, v, key, float(data['length']), xy.tolist())).encode())
            cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            count = max(1, int(np.ceil(cumulative[-1] / 50)))
            cuts = np.linspace(0, cumulative[-1], count + 1)
            samples = np.column_stack([np.interp(cuts, cumulative, xy[:, axis]) for axis in (0, 1)])
            previous = nodes[u]
            for i in range(count):
                end = nodes[v] if i == count - 1 else len(self.xy)
                if i != count - 1:
                    self.xy.append(samples[i + 1].tolist())
                    self.adj.append([])
                lo = np.searchsorted(cumulative, cuts[i], side='right')
                hi = np.searchsorted(cumulative, cuts[i + 1], side='left')
                geom = np.vstack((samples[i:i+1], xy[lo:hi], samples[i+1:i+2]))
                length = float(data['length']) / count
                # Canonical geometry identifies reverse traversal of the same physical piece.
                rounded = tuple(map(tuple, np.round(geom, 2)))
                physical = min(rounded, rounded[::-1])
                eid = len(self.edges)
                self.edges.append((previous, end, length, geom, (u, v, key), physical))
                self.adj[previous].append((end, length, eid))
                previous = end
        self.xy = np.asarray(self.xy)
        self.tree = cKDTree(self.xy)
        self.junction_tree = cKDTree(self.xy[:self.node_count])
        self.graph_hash = digest.hexdigest()
        left, bottom, right, top = frame.bounds
        self.field_step = 50
        gx, gy = np.meshgrid(np.arange(left, right + 50, 50), np.arange(bottom, top + 50, 50))
        self.field = self.tree.query(np.column_stack((gx.ravel(), gy.ravel())))[0].reshape(gx.shape)
        self.path_cache = OrderedDict()
        self.route_cache = OrderedDict()
        self.rejections = {}
        self.index_seconds = time.perf_counter() - started

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return None

    def check(self):
        if getattr(self, 'work_remaining', None) is not None:
            self.work_remaining -= 1
            if self.work_remaining < 0:
                raise Deadline('work_budget')
        if time.perf_counter() >= self.deadline:
            raise Deadline('wall_timeout')

    def set_distance_band(self, perimeters=DEFAULT_PERIMETERS, distance_range=DEFAULT_DISTANCE_RANGE,
                          prefer='balanced'):
        """Target outline perimeters to try, and the loop lengths that are acceptable.

        `prefer='balanced'` walks the ladder middle-out, keeping mid-length candidates
        before the extremes when the placement budget runs short. `prefer='longest'`
        walks it from the top instead and reports the longest qualifying loops first.
        """
        if prefer not in ('balanced', 'longest'):
            raise ValueError("prefer must be 'balanced' or 'longest'.")
        self.prefer = prefer
        self.perimeters = tuple(sorted(float(p) for p in perimeters))
        if not self.perimeters:
            raise ValueError('At least one target perimeter is required.')
        low, high = (float(v) for v in distance_range)
        if not 0 < low < high:
            raise ValueError('distance_range must be an increasing positive pair.')
        self.distance_range = (low, high)
        if prefer == 'longest':
            self.perimeter_preference = tuple(reversed(self.perimeters))
            return
        # Walk outward from the middle rung, longer side first — the order the ladder
        # used to hardcode as (8000, 11000, 5000, 14000, 3000).
        middle = len(self.perimeters) // 2
        order, step = [middle], 1
        while len(order) < len(self.perimeters):
            for index in (middle + step, middle - step):
                if 0 <= index < len(self.perimeters):
                    order.append(index)
            step += 1
        self.perimeter_preference = tuple(self.perimeters[i] for i in order)

    @property
    def distance_label(self):
        low, high = self.distance_range
        return f'{low / 1000:g}\u2013{high / 1000:g} km'

    def transform(self, outline, x, y, angle, perimeter):
        """Angle is counterclockwise in projected meters; flip screen y exactly once."""
        p = np.asarray(outline.points).copy()
        p -= (p.max(axis=0) + p.min(axis=0)) / 2
        p[:, 1] *= -1
        a = np.deg2rad(angle)
        p = p @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return p * perimeter / np.linalg.norm(np.diff(p, axis=0), axis=1).sum() + [x, y]

    def placement_score(self, p):
        left, bottom, right, top = self.frame.bounds
        if (p.min(axis=0) < [left, bottom]).any() or (p.max(axis=0) > [right, top]).any():
            return float('inf')
        q = sample_line(p, 48)
        ij = np.rint((q - [left, bottom]) / self.field_step).astype(int)
        return float(np.mean(self.field[ij[:, 1], ij[:, 0]])) / np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

    def paths(self, start, targets, limit):
        key = (start, tuple(sorted(targets)), round(limit, 6))
        if key in self.path_cache:
            self.path_cache.move_to_end(key)
            return self.path_cache[key]
        dist, prev, queue, found = {start: 0.0}, {}, [(0.0, start)], {}
        while queue:
            self.check()
            d, u = heapq.heappop(queue)
            if d != dist[u]:
                continue
            if u in targets:
                path, current = [], u
                while current != start:
                    parent, eid = prev[current]
                    path.append(eid)
                    current = parent
                found[u] = (d, path[::-1])
                if len(found) == len(targets):
                    break
            for v, length, eid in self.adj[u]:
                nd = d + length
                if nd <= limit and nd < dist.get(v, float('inf')):
                    dist[v], prev[v] = nd, (u, eid)
                    heapq.heappush(queue, (nd, v))
        self.path_cache[key] = found
        if len(self.path_cache) > 20000:
            self.path_cache.popitem(last=False)
        return found

    def match(self, p):
        line = LineString(p)
        # Preserve every compiler vertex, including sharp bends and concavities.
        distances_along = list(np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))][:-1])
        desired = int(np.clip(np.ceil(line.length / 180), max(32, len(distances_along)), 64))
        while len(distances_along) < desired:
            ends = distances_along[1:] + [line.length]
            i = int(np.argmax(np.array(ends)-distances_along))
            distances_along.insert(i+1, (distances_along[i]+ends[i])/2)
        n = len(distances_along)
        ends = distances_along[1:] + [line.length]
        anchors = np.array([line.interpolate(d).coords[0] for d in distances_along])
        distances, indices = self.junction_tree.query(anchors, k=4, distance_upper_bound=300)
        candidates = [list(dict.fromkeys(int(i) for d,i in zip(ds,ids) if np.isfinite(d)))
                      for ds,ids in zip(distances,indices)]
        # Long edges may have no nearby junction: retain directed interior locations there.
        for i, options in enumerate(candidates):
            if len(options) < 2:
                ds, ids = self.tree.query(anchors[i], k=4, distance_upper_bound=300)
                candidates[i] = list(dict.fromkeys(options + [int(j) for d,j in zip(ds,ids) if np.isfinite(d)]))[:4]
        if any(not c for c in candidates):
            return self.reject('no_anchor_within_300m')
        transitions = []
        for layer in range(n):
            self.check()
            next_layer = (layer + 1) % n
            arc = ends[layer] - distances_along[layer]
            target_arc = substring(line, distances_along[layer], ends[layer])
            matrix = {}
            for source in candidates[layer]:
                for target, (length, path) in self.paths(source, set(candidates[next_layer]), 3 * arc).items():
                    if not path:
                        cost = arc
                    else:
                        q = self.xy[[self.edges[e][1] for e in path]]
                        from shapely import points, distance
                        deviation = float(np.mean(distance(points(q), target_arc)))
                        cost = deviation + abs(length - arc) * 0.35
                    matrix[source, target] = (cost, path)
            if not matrix:
                return self.reject('no_directed_transition_within_arc_limits')
            transitions.append(matrix)
        best = None
        for start in candidates[0]:
            states = {start: (0.0, [])}
            for layer, matrix in enumerate(transitions):
                next_states = {}
                targets = [start] if layer == n - 1 else candidates[layer + 1]
                for source, (cost, path) in states.items():
                    for target in targets:
                        if (source, target) in matrix:
                            extra, segment = matrix[source, target]
                            if cost + extra < next_states.get(target, (float('inf'),))[0]:
                                next_states[target] = (cost + extra, path + segment)
                states = next_states
            if start in states and (best is None or states[start][0] < best[0]):
                best = states[start]
        if best is None or not best[1]:
            return self.reject('no_directed_closure_within_arc_limits')
        edges = best[1]
        length = sum(self.edges[e][2] for e in edges)
        low, high = self.distance_range
        if not low <= length <= high:
            return self.reject(f'distance_outside_{low / 1000:g}_{high / 1000:g}km')
        physical = {}
        for e in edges:
            physical[self.edges[e][5]] = self.edges[e][2]
        repeated = 1 - sum(physical.values()) / length
        xy = np.vstack([self.edges[e][3][:-1] for e in edges] + [self.edges[edges[-1]][3][-1:]])
        if repeated > .2 or not np.allclose(xy[0], xy[-1], atol=.01, rtol=0):
            return self.reject('repeated_streets_or_open_loop')
        polygon = Polygon(xy).buffer(0)
        if polygon.area < 10000 or polygon.area < Polygon(p).area * .15:
            return self.reject('collapsed_loop')
        if not self.frame.polygon.covers(LineString(xy)):
            return self.reject('route_outside_map')
        target = Polygon(p)
        overlap = polygon.intersection(target).area / polygon.union(target).area
        from shapely import points, distance
        distortion = float(np.mean(distance(points(xy), line))) / (line.length / (2 * np.pi))
        directed = []
        for e in edges:
            original = self.edges[e][4]
            if not directed or directed[-1] != original:
                directed.append(original)
        return {'xy': xy.tolist(), 'edges': directed, 'index_edges': edges, 'distance_m': length,
                'scores': {'outline_distance': distortion, 'overlap': overlap, 'repeated': repeated,
                           'rank': distortion + 1 - overlap + repeated}}

    def search(self, outline, budget=8, work_budget=None):
        started = time.perf_counter()
        self.deadline = started + max(0, min(8, budget))
        self.work_remaining = work_budget
        if work_budget is not None:
            self.path_cache.clear()  # Cache warmth must not change deterministic work accounting.
        result = RouteSearchResult(outline)
        self.rejections = {}
        key = hashlib.sha256(json.dumps([outline.key(), self.graph_hash, VERSION, self.frame.bounds, budget,
                                         work_budget, self.perimeters, self.distance_range, self.prefer]).encode()).hexdigest()
        if key in self.route_cache:
            import copy
            result = copy.deepcopy(self.route_cache[key])
            result.timings = {'cache_seconds': time.perf_counter() - started}
            return result
        disk = getattr(self,'route_cache_dir',None)
        cache_path = disk / f'{key}.json' if disk else None
        if cache_path and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                if payload['key'] != key: raise ValueError('Stale route cache')
                result = RouteSearchResult(outline,**payload['result'])
                result.timings = {'cache_seconds':time.perf_counter()-started}
                return result
            except (ValueError,KeyError,TypeError,OSError): pass
        placements, tested = [], 0
        try:
            left, bottom, right, top = self.frame.bounds
            # Vectorized distance-field lookup for all translations at each angle/scale.
            gx, gy = np.meshgrid(np.arange(left + 250, right, 500), np.arange(bottom + 250, top, 500))
            centers = np.column_stack((gx.ravel(), gy.ravel()))
            for perimeter in self.perimeters:
                # Generated raster coordinates are upright screen coordinates.  The
                # production path permits only a small physical adjustment, never a
                # mirror or an arbitrary rotation that changes the requested subject.
                for angle in (-15, 0, 15):
                    self.check()
                    p = self.transform(outline, 0, 0, angle, perimeter)
                    samples = sample_line(p, 48)
                    valid = ((centers + p.min(axis=0) >= [left, bottom]).all(axis=1)
                             & (centers + p.max(axis=0) <= [right, top]).all(axis=1))
                    cs = centers[valid]
                    ij = np.rint((cs[:, None, :] + samples - [left, bottom]) / 50).astype(int)
                    scores = self.field[ij[:, :, 1], ij[:, :, 0]].mean(axis=1) / perimeter + .001 * min(angle, 360-angle) / 180
                    placements.extend((float(s), float(c[0]), float(c[1]), angle, perimeter) for s, c in zip(scores, cs))
            distinct = []
            ordered = sorted(placements)
            smallest = self.perimeters[0]
            for scale in self.perimeter_preference:
                selected = []
                for item in ordered:
                    if item[4] == scale and all(np.hypot(item[1]-q[1], item[2]-q[2]) >= 500 for q in selected):
                        selected.append(item)
                        if len(selected) == (4 if scale == smallest else 5):
                            break
                distinct.extend(selected)
            refined = []
            for _, x, y, angle, perimeter in distinct:
                self.check()
                local = []
                for dx, dy in ((0, 0), (-125, 0), (125, 0), (0, -125), (0, 125)):
                    for da in (-10, 0, 10):
                        for scale in (.9, 1, 1.05):
                            self.check()
                            p = self.transform(outline, x + dx, y + dy, angle + da, perimeter * scale)
                            local.append((self.placement_score(p) + .001 * abs((angle + da + 180) % 360 - 180) / 180, x + dx, y + dy, angle + da, perimeter * scale))
                refined.extend(sorted(local)[:4])
            result.placements = [dict(zip(('field_distance', 'x', 'y', 'angle', 'perimeter'), item)) for item in sorted(refined)]
            result.timings['placement_seconds'] = time.perf_counter() - started
            for item in result.placements:
                p = self.transform(outline, item['x'], item['y'], item['angle'], item['perimeter'])
                for direction in (1, -1):
                    self.check()
                    route = self.match(p[::direction])
                    tested += 1
                    if route:
                        route['transform'] = {**item, 'direction': direction}
                        route['scores']['orientation_penalty'] = .15 * abs((item['angle'] + 180) % 360 - 180) / 180
                        route['scores']['rank'] += route['scores']['orientation_penalty']
                        result.routes.append(route)
                if (work_budget is None and time.perf_counter()-started >= 3 and len(result.routes) >= 3
                        and min(r['scores']['rank'] for r in result.routes) < .3):
                    result.diagnostics['early_quality_stop'] = True
                    break
        except Deadline as exc:
            result.diagnostics['timed_out'] = str(exc) != 'work_budget'
            result.diagnostics['work_exhausted'] = str(exc) == 'work_budget'
        result.routes.sort(key=lambda r: r['scores']['rank'])
        unique, signatures = [], []
        for route in result.routes:
            physical = frozenset(self.edges[e][5] for e in route['index_edges'])
            if any(len(physical & seen) / len(physical | seen) > .85 for seen in signatures):
                continue
            unique.append(route)
            signatures.append(physical)
            if len(unique) == 3: break
        if self.prefer == 'longest':
            unique.sort(key=lambda route: -route['distance_m'])
        result.routes = unique
        result.timings['search_seconds'] = time.perf_counter() - started
        result.diagnostics['rejections'] = dict(self.rejections)
        result.diagnostics.update(coarse_placements=len(placements), matched_placements=tested,
                                  index_seconds=self.index_seconds, graph_hash=self.graph_hash)
        if not result.routes:
            result.failure_reason = (f'No qualifying {self.distance_label} street loop found '
                                     'within the search budget.')
        if not result.diagnostics.get('timed_out'):
            import copy
            self.route_cache[key] = copy.deepcopy(result)
            if cache_path:
                import os,tempfile
                from dataclasses import asdict
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                payload = asdict(result)
                del payload['outline']
                with tempfile.NamedTemporaryFile(mode='w',dir=cache_path.parent,delete=False) as temp:
                    json.dump({'key':key,'result':payload},temp)
                os.replace(temp.name,cache_path)
            if len(self.route_cache) > 128:
                self.route_cache.popitem(last=False)
        return result

    def save_index(self, path, source_hash):
        """Only numeric arrays and JSON: loading the derived index never unpickles code."""
        import os, tempfile
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(e[3]) for e in self.edges])]
        manifest = json.dumps({'version': VERSION, 'source_hash': source_hash,
                               'graph_hash': self.graph_hash, 'bounds': self.frame.bounds,
                               'node_count': self.node_count})
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            np.savez_compressed(temp, xy=self.xy, offsets=offsets,
                                geometry=np.vstack([e[3] for e in self.edges]),
                                links=np.array([e[:3] for e in self.edges]),
                                originals=json.dumps([e[4] for e in self.edges]),
                                field=self.field, manifest=manifest)
        os.replace(temp.name,path)

    @classmethod
    def load_prepared(cls, graph_path=None, cache_dir=None, frame=SF,
                      perimeters=DEFAULT_PERIMETERS, distance_range=DEFAULT_DISTANCE_RANGE,
                      prefer='balanced'):
        from pathlib import Path
        from src.graph import GRAPH_PATH, BikeRouter
        graph_path = Path(graph_path or GRAPH_PATH)
        cache_dir = Path(cache_dir or graph_path.parent / 'search-index')
        source_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        key = hashlib.sha256(json.dumps([source_hash,VERSION,frame.bounds]).encode()).hexdigest()
        path = cache_dir / f'{key}.npz'
        if path.exists():
            try:
                started = time.perf_counter()
                with np.load(path,allow_pickle=False) as data:
                    meta = json.loads(str(data['manifest']))
                    if (meta['version'] != VERSION or meta['source_hash'] != source_hash
                            or meta['bounds'] != list(frame.bounds)):
                        raise ValueError('Stale search index')
                    instance = cls.__new__(cls)
                    instance.frame, instance.node_count = frame, meta['node_count']
                    instance.set_distance_band(perimeters, distance_range, prefer)
                    instance.graph_hash = meta['graph_hash']
                    instance.xy = data['xy']
                    instance.field = data['field']
                    instance.field_step = 50
                    offsets, geometry = data['offsets'], data['geometry']
                    originals = json.loads(str(data['originals']))
                    instance.adj = [[] for _ in instance.xy]
                    instance.edges = []
                    for i,(u,v,length) in enumerate(data['links']):
                        u,v = int(u),int(v)
                        geom = geometry[offsets[i]:offsets[i+1]]
                        rounded = tuple(map(tuple,np.round(geom,2)))
                        instance.edges.append((u,v,float(length),geom,tuple(originals[i]),min(rounded,rounded[::-1])))
                        instance.adj[u].append((v,float(length),i))
                instance.tree = cKDTree(instance.xy)
                instance.junction_tree = cKDTree(instance.xy[:instance.node_count])
                instance.path_cache, instance.route_cache, instance.rejections = OrderedDict(), OrderedDict(), {}
                instance.route_cache_dir = Path('route_cache/routes')
                instance.index_seconds = time.perf_counter()-started
                return instance
            except (ValueError,KeyError,OSError,EOFError):
                pass
        instance = cls(BikeRouter.load(graph_path).graph, frame, perimeters, distance_range, prefer)
        instance.save_index(path,source_hash)
        instance.route_cache_dir = Path('route_cache/routes')
        return instance
