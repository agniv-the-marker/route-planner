"""Restricted, non-executable drawing language and conservative curve compiler."""
import json
import numpy as np
from shapely.geometry import Polygon, LineString
from src.outlines import OutlineSpec

MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
VERSION = 'vector-v1'
SEEDS = (17, 29, 43, 71)
SYSTEM = '''Draw the user's exact subject, pose and proportions as one closed silhouette.
Syntax example (a triangle only to demonstrate formatting): [[\"M\",128,128],[\"L\",896,128],[\"L\",512,896],[\"Z\"]]
Do not copy this triangle for the user's subject.
Return only a JSON array of commands: ["M",x,y], ["L",x,y],
["Q",cx,cy,x,y], ["C",c1x,c1y,c2x,c2y,x,y], ["Z"].
Use integer coordinates 0–1024, x right and y down. Start with M, end with Z.
At most 32 commands total, one subpath, no intersections, no internal details.
No prose, SVG, code, images or template names. Preserve distinguishing features.'''


def extract_program(raw):
    """Decode one JSON array from an otherwise unusable completion, preserving raw text."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != '[':
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    raise ValueError('Completion did not contain a JSON command array.')


def compile_program(raw, prompt):
    if isinstance(raw, str):
        if len(raw) > 32768:
            raise ValueError('Program too large.')
        raw = extract_program(raw)
    if not isinstance(raw, list) or not 4 <= len(raw) <= 32:
        raise ValueError('Expected 4–32 commands.')
    sizes = {'M': 3, 'L': 3, 'Q': 5, 'C': 7, 'Z': 1}
    for i, c in enumerate(raw):
        if not isinstance(c, list) or not c or not isinstance(c[0], str) or c[0] not in sizes or len(c) != sizes[c[0]]:
            raise ValueError('Invalid command.')
        if any(type(v) is not int or not 0 <= v <= 1024 for v in c[1:]):
            raise ValueError('Coordinates must be integers in 0–1024.')
        if (c[0] == 'M') != (i == 0) or (c[0] == 'Z') != (i == len(raw)-1):
            raise ValueError('One explicitly closed subpath required.')
    points = [np.array(raw[0][1:], dtype=float)]
    # De Casteljau subdivision: each chord stays within .25 grid units of its curve.
    def flatten(control, depth=0):
        chord = LineString([control[0], control[-1]])
        from shapely.geometry import Point
        if max(chord.distance(Point(p)) for p in control) <= .25:
            points.append(control[-1]); return
        if depth >= 16:
            raise ValueError('Curve subdivision limit exceeded.')
        levels = [control]
        while len(levels[-1]) > 1:
            levels.append((levels[-1][:-1] + levels[-1][1:])/2)
        flatten(np.array([a[0] for a in levels]), depth+1)
        flatten(np.array([a[-1] for a in levels[::-1]]), depth+1)
    for c in raw[1:-1]:
        control = np.vstack([points[-1], np.array(c[1:]).reshape(-1,2)])
        flatten(control)
    if not np.array_equal(points[-1], points[0]):
        points.append(points[0])
    poly = Polygon(points)
    if not poly.is_valid or poly.area <= 1 or not poly.exterior.is_simple:
        raise ValueError('Collapsed or self-intersecting silhouette.')
    width = poly.bounds[2] - poly.bounds[0]
    tolerance = width * .005 - .25
    if tolerance <= 0:
        raise ValueError('Silhouette too narrow.')
    simplified = poly.simplify(tolerance, preserve_topology=True)
    if poly.boundary.hausdorff_distance(simplified.boundary) > tolerance:
        raise ValueError('Simplification loses features.')
    p = list(simplified.exterior.coords)
    if len(p) > 48:
        raise ValueError('Cannot preserve features within 48 points.')
    while len(p) < 8:
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        i = int(np.argmax(lengths))
        p.insert(i+1, tuple((np.array(p[i])+p[i+1])/2))
    return OutlineSpec.parse({'interpretation': prompt[:160], 'points': (np.array(p)/1024).tolist()})
