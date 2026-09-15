"""Ordered strokes in x-right/y-down unit coordinates; transfers are explicit lines."""
from dataclasses import dataclass, asdict
import numpy as np
from src.experiment_journal import digest

@dataclass(frozen=True)
class Drawing:
    name: str
    strokes: tuple
    features: tuple = ()
    version: str = 'strokes-v1'

    @classmethod
    def parse(cls, data):
        strokes = []
        for stroke in data['strokes']:
            p = np.asarray(stroke, dtype=float)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError('Strokes need at least two finite points in [0,1]')
            if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 1e-9):
                raise ValueError('Consecutive stroke points must differ')
            strokes.append(tuple(map(tuple, p.tolist())))
        if not strokes:
            raise ValueError('At least one stroke is required')
        features = tuple(data.get('features', ()))
        for feature in features:
            si, vi = feature['stroke'], feature['vertex']
            if not (0 <= si < len(strokes) and 0 <= vi < len(strokes[si])):
                raise ValueError('Feature must reference a stroke vertex')
        return cls(data['name'], tuple(strokes), features)

    def key(self):
        return digest(asdict(self))

    def continuous(self):
        points, connector_segments = [], []
        for stroke in self.strokes:
            if points and not np.allclose(points[-1], stroke[0]):
                connector_segments.append(len(points)-1)
                points.append(stroke[0])
            points.extend(stroke if not points else stroke[1:])
        return np.asarray(points, dtype=float), connector_segments

    def place(self, center, span_m, angle):
        if not -15 <= angle <= 15 or span_m <= 0:
            raise ValueError('Placement must be upright ±15°, with positive scale')
        p, connectors = self.continuous()
        midpoint = (p.min(0)+p.max(0))/2
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return (p-midpoint)*[1,-1] @ rotation * (span_m / np.ptp(p, axis=0).max()) + center, connectors
