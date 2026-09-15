"""Small JSON boundary for interactive, user supplied drawings.

The service deliberately contains no inference.  It turns normalized pointer
strokes into a :class:`Drawing`, fits them with the directed stroke router,
and converts the resulting projected geometry back to screen pixels and GPX.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from shapely.geometry import LineString

from src.geo import SF, TO_GEO
from src.graph import BikeRouter
from src.generation import to_gpx
from src.terrain import profile_markup
from src.strokes import Drawing
from src.stroke_router import RoutingConfig, SearchLimit, StrokeRouter


class DrawingService:
    """Fit normalized drawings against the historical SF bicycle graph.

    ``router`` is injectable and is expected to expose ``match`` and
    ``search`` like :class:`StrokeRouter`; this keeps UI tests independent of
    the large map file.  With no router, the graph is loaded on first call.
    """

    MAX_POINTS = 4096
    MAX_STROKES = 64
    PIXEL_SIZE = 511
    # Candidate routes are already ranked by the router.  These guards only
    # stop the UI from presenting several nearby re-matches of that same ride.
    DIVERSITY_CENTER_M = 750.0
    DIVERSITY_EDGE_OVERLAP = 0.75
    # Keep the UI aligned with the research ride bands.  The router still
    # receives a hard cap for every request; larger caps are useful when a
    # small canvas sketch needs a longer connecting ride.
    DISTANCE_KM = frozenset((15, 30, 50, 80))

    def __init__(self, router: Any | None = None, frame=SF, simplify_tolerance_px: float = 1.5):
        if simplify_tolerance_px < 0 or not np.isfinite(simplify_tolerance_px):
            raise ValueError("simplify_tolerance_px must be finite and nonnegative")
        self._router = router
        self.frame = frame
        self.simplify_tolerance_px = float(simplify_tolerance_px)
        self._load_lock = threading.Lock()
        self._fit_lock = threading.Lock()

    @property
    def router(self):
        if self._router is None:
            with self._load_lock:
                if self._router is None:
                    # BikeRouter performs the historical graph integrity checks;
                    # StrokeRouter supplies the shape-aware fitting operation.
                    self._router = StrokeRouter(BikeRouter.load().graph, region=self.frame)
        return self._router

    def _drawing(self, payload: dict) -> tuple[Drawing, np.ndarray]:
        if not isinstance(payload, dict):
            raise ValueError("Payload must be an object")
        raw = payload.get("strokes")
        if not isinstance(raw, (list, tuple)) or not raw or len(raw) > self.MAX_STROKES:
            raise ValueError(f"strokes must contain 1–{self.MAX_STROKES} strokes")
        total = 0
        normalized = []
        # Feature indices refer to compiler vertices.  Keep those vertices
        # stable until a feature-aware simplifier exists.
        has_features = bool(payload.get("features"))
        for stroke in raw:
            try:
                p = np.asarray(stroke, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Stroke points must be numeric") from exc
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
                raise ValueError("Each stroke needs at least two points")
            total += len(p)
            if total > self.MAX_POINTS:
                raise ValueError(f"Drawing exceeds {self.MAX_POINTS} points")
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError("Stroke points must be finite and normalized to [0,1]")
            # Simplify in pixels so the tolerance has a predictable UI meaning.
            px = p * self.PIXEL_SIZE
            if self.simplify_tolerance_px and not has_features:
                px = np.asarray(LineString(px).simplify(self.simplify_tolerance_px,
                                                       preserve_topology=False).coords)
            if len(px) < 2:
                px = np.asarray([p[0] * self.PIXEL_SIZE, p[-1] * self.PIXEL_SIZE])
            normalized.append((px / self.PIXEL_SIZE).tolist())
        drawing = Drawing.parse({"name": str(payload.get("name", "drawing")),
                                 "strokes": normalized,
                                 "features": payload.get("features", ())})
        target_pixels, _ = drawing.continuous()
        return drawing, target_pixels * self.PIXEL_SIZE

    def _config(self, max_distance_km: int) -> RoutingConfig:
        # Prefer a longer loop in the dedicated long-ride mode. SF's bounded
        # graph may still return a shorter candidate when no 80 km loop follows
        # the requested outline.
        minimum = 1 if max_distance_km < 80 else 20_000
        return RoutingConfig(max_distance_m=max_distance_km * 1000,
                             min_distance_m=minimum, candidates=1, beam_width=1,
                             # Small spans fit dense SF streets much more
                             # reliably.  The old .20/.35/.50 candidates often
                             # made a single turn several kilometres apart.
                             span_fractions=(.04, .07, .10),
                             placements=24,
                             work_budget=600000, wall_seconds=20,
                             diverse_candidates=True, candidate_pool=12)

    def _route_json(self, route: dict, target_pixels: np.ndarray) -> dict:
        xy = np.asarray(route.get("xy", ()), dtype=float)
        target_xy = np.asarray(route.get("target_xy", ()), dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("Router returned an invalid route geometry")
        if target_xy.ndim != 2 or target_xy.shape[1] != 2 or len(target_xy) < 2:
            target_xy = self.frame.pixels_to_meters(target_pixels)
        lon, lat = TO_GEO.transform(xy[:, 0], xy[:, 1])
        gpx = to_gpx(list(zip(lat.tolist(), lon.tolist())), "drawing")
        # ``to_gpx`` is shared with the legacy closed-loop UI; interactive
        # drawings may be open, so avoid promising closure in this export.
        gpx = gpx.replace("SF bicycle street loop.",
                          "SF bicycle street route for an open or closed drawing.")
        return {"xy_pixels": self.frame.meters_to_pixels(xy).tolist(),
                "target_pixels": self.frame.meters_to_pixels(target_xy).tolist(),
                "distance_m": float(route.get("distance_m", 0)),
                "gpx": gpx,
                "scores": route.get("scores", {}),
                "profile_html": profile_markup(xy)}

    def fit(self, payload: dict) -> dict:
        """Fit a drawing and return a JSON serializable result."""
        if not isinstance(payload, dict):
            raise ValueError("Payload must be an object")
        mode = payload.get("mode", "canvas")
        if mode not in {"map", "canvas"}:
            raise ValueError("mode must be 'map' or 'canvas'")
        max_km = payload.get("max_distance_km", 80)
        if isinstance(max_km, bool) or max_km not in self.DISTANCE_KM:
            raise ValueError("max_distance_km must be one of 15, 30, 50, or 80 km")
        drawing, target_pixels = self._drawing(payload)
        config = self._config(int(max_km))
        with self._fit_lock:
            if mode == "map":
                target_m = self.frame.pixels_to_meters(target_pixels)
                if hasattr(self.router, "start"):
                    self.router.start(config)
                started = time.perf_counter()
                try:
                    route, reason = self.router.match(target_m, variant="paper")
                    raw = {"status": "completed" if route else "failed",
                           "routes": [self._route_json(route, target_pixels)] if route else [],
                           "rejections": {} if route else {str(reason): 1}}
                except SearchLimit as exc:
                    raw = {"status": "incomplete", "routes": [],
                           "rejections": {str(exc): 1}, "limit": str(exc)}
                raw.update(seconds=time.perf_counter() - started,
                           work_used=getattr(self.router, 'used', 0))
            else:
                result = self.router.search(drawing, config, variant="paper")
                raw = dict(result)
                raw["routes"] = [self._route_json(route, target_pixels)
                                 for route in self._select_routes(result.get("routes", []))]
        routes = raw["routes"]
        raw_status = raw.get("status", "completed")
        # A completed search with no candidate is a failed fit.  Preserve an
        # incomplete status even when the bounded search found partial routes.
        status = raw_status if raw_status == "incomplete" else ("completed" if routes else "failed")
        if routes and status == "incomplete":
            message = f"Search incomplete; found {len(routes)} candidate route(s) before the limit."
        elif routes:
            message = f"Found {len(routes)} candidate route(s)."
        elif status == "incomplete":
            message = "Search incomplete within the configured work or time limit."
        else:
            message = "No route found within this search's placement, snap and ride limits. Try adjusting the drawing."
        return {"status": status, "message": message, "routes": routes,
                "rejections": raw.get("rejections", {}),
                "failure_reason": raw.get("failure_reason"),
                "config": config.__dict__,
                **{key: raw[key] for key in ("seconds", "work_used", "limit") if key in raw}}

    @staticmethod
    def _unique_routes(routes):
        """Remove repeated candidate traversals while retaining route order."""
        seen = set()
        unique = []
        for route in routes:
            edges = route.get("edges")
            key = ("edges", tuple(tuple(e) for e in edges)) if edges else (
                "xy", tuple(tuple(round(float(v), 6) for v in p) for p in route.get("xy", ())))
            if key in seen:
                continue
            seen.add(key)
            unique.append(route)
        return unique

    @classmethod
    def _select_routes(cls, routes):
        """Keep the router's best route first while removing near-identical ones.

        The router may return distinct placements that snap to the same local
        street traversal. Exact edge deduplication alone made the result cards
        look like copies, so a second guard compares route centers and shared
        directed edges. A route survives when it is spatially distinct or
        follows a meaningfully different edge set.
        """
        unique = cls._unique_routes(routes)
        selected = []
        for route in unique:
            xy = np.asarray(route.get("xy", ()), dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
                continue
            center = xy[:, :2].mean(axis=0)
            edges = {tuple(edge) for edge in route.get("edges", ())}
            too_similar = False
            for prior, prior_center, prior_edges in selected:
                center_close = float(np.linalg.norm(center - prior_center)) < cls.DIVERSITY_CENTER_M
                if edges and prior_edges:
                    overlap = len(edges & prior_edges) / min(len(edges), len(prior_edges))
                else:
                    overlap = 0.0
                if center_close and overlap >= cls.DIVERSITY_EDGE_OVERLAP:
                    too_similar = True
                    break
            if too_similar:
                continue
            selected.append((route, center, edges))
            if len(selected) >= 3:
                break
        return [route for route, _, _ in selected]
