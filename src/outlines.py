"""Validated data-only outline contract. Coordinates use x-right, y-down."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from shapely.geometry import Polygon, LinearRing

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
VERSION = 'outline-v1'
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['interpretation', 'points'],
          'properties': {'interpretation': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                         'points': {'type': 'array', 'minItems': 8, 'maxItems': 48,
                                    'items': {'type': 'array', 'minItems': 2, 'maxItems': 2,
                                              'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}}
SYSTEM = '''Interpret the description as ONE recognizable simple silhouette, without internal details.
Return only JSON: interpretation (short description of your symbol), points (8–48 ordered [x,y]
vertices of its outer boundary). Coordinates are 0 to 1, x right and y down. Preserve proportions.
Trace the perimeter in order; never cross or retrace an edge. Repeat the first point as the last
point to close the polygon. Use at most 32 vertices when possible. No code, SVG, or prose.'''

@dataclass(frozen=True)
class OutlineSpec:
    interpretation: str
    points: tuple

    @classmethod
    def parse(cls, value):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict) or set(value) != {'interpretation', 'points'}:
            raise ValueError('Expected interpretation and points only.')
        label = value['interpretation']
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 160:
            raise ValueError('Missing short interpretation.')
        points = np.asarray(value['points'])
        if points.dtype.kind not in 'ifu' or points.ndim != 2 or points.shape[1] != 2 or not 8 <= len(points) <= 48:
            raise ValueError('Expected 8–48 numeric coordinate pairs.')
        points = points.astype(float)
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError('Coordinates must be finite and normalized.')
        if not np.array_equal(points[0], points[-1]):
            raise ValueError('Outline must explicitly close.')
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 1e-6 or not LinearRing(points).is_simple:
            raise ValueError('Outline is empty or self-intersecting.')
        if (np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-8).any():
            raise ValueError('Consecutive vertices must differ.')
        return cls(label.strip(), tuple(map(tuple, points)))

    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def prompt_key(prompt):
    return hashlib.sha256(json.dumps([MODEL, VERSION, ' '.join(prompt.casefold().split())]).encode()).hexdigest()
