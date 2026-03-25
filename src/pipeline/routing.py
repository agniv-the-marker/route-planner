"""Routing clients for computing bike routes between waypoints.

Supports Valhalla (default, free public server) and OSRM backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

VALHALLA_BASE_URL = "https://valhalla1.openstreetmap.de"
OSRM_BASE_URL = "https://router.project-osrm.org"
CACHE_DIR = Path("route_cache")


class ValhallaRouter:
    """Routes waypoints through Valhalla to get bikeable road paths.

    Uses the free OpenStreetMap.de Valhalla instance by default.
    """

    def __init__(
        self,
        base_url: str = VALHALLA_BASE_URL,
        costing: str = "bicycle",
        request_delay: float = 1.0,
        cache_dir: Path | str = CACHE_DIR,
    ):
        self.base_url = base_url.rstrip("/")
        self.costing = costing
        self.request_delay = request_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def _cache_key(self, waypoints: np.ndarray) -> str:
        data = waypoints.tobytes() + self.costing.encode()
        return hashlib.sha256(data).hexdigest()

    def _load_cache(self, key: str) -> list[tuple[float, float]] | None:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return [tuple(p) for p in json.load(f)]
        return None

    def _save_cache(self, key: str, route: list[tuple[float, float]]) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(route, f)

    def _segment_cache_key(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> str:
        """Cache key for a single segment (pair of points + costing)."""
        data = f"{start[0]:.6f},{start[1]:.6f}-{end[0]:.6f},{end[1]:.6f}-{self.costing}"
        return "seg_" + hashlib.sha256(data.encode()).hexdigest()[:16]

    def _load_segment_cache(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        key = self._segment_cache_key(start, end)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return [tuple(p) for p in json.load(f)]
        return None

    def _save_segment_cache(
        self, start: tuple[float, float], end: tuple[float, float],
        segment: list[tuple[float, float]],
    ) -> None:
        key = self._segment_cache_key(start, end)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(segment, f)

    def route_segment(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        """Route between two points (lat, lon) via Valhalla.

        Returns the routed path, or *None* if the server could not
        produce a route (caller should try a fallback).
        """
        # Check segment cache first
        cached = self._load_segment_cache(start, end)
        if cached is not None:
            return cached

        url = f"{self.base_url}/route"
        payload = {
            "locations": [
                {"lat": start[0], "lon": start[1]},
                {"lat": end[0], "lon": end[1]},
            ],
            "costing": self.costing,
            "directions_options": {"units": "km"},
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=30)
                if resp.status_code == 400:
                    logger.warning(f"Valhalla 400 error: {resp.text[:200]}")
                    return None
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"Valhalla retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.warning(f"Valhalla request failed after {max_retries} retries: {e}")
                    return None

        # Extract the shape from the response
        try:
            shape = data["trip"]["legs"][0]["shape"]
            points = self._decode_polyline(shape)
            # Cache the successful segment result
            self._save_segment_cache(start, end, points)
            return points
        except (KeyError, IndexError) as e:
            logger.warning(f"Valhalla response parsing failed: {e}")
            return None

    @staticmethod
    def _decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
        """Decode a Valhalla encoded polyline string to (lat, lon) points."""
        inv = 1.0 / (10 ** precision)
        decoded = []
        previous = [0, 0]
        i = 0
        while i < len(encoded):
            for dim in range(2):
                shift = 0
                result = 0
                while True:
                    char_code = ord(encoded[i]) - 63
                    i += 1
                    result |= (char_code & 0x1F) << shift
                    shift += 5
                    if char_code < 0x20:
                        break
                if result & 1:
                    result = ~result
                result >>= 1
                previous[dim] += result
            decoded.append((previous[0] * inv, previous[1] * inv))
        return decoded

    def route_waypoints(
        self, waypoints: np.ndarray
    ) -> list[tuple[float, float]]:
        """Route through all waypoints sequentially.

        If Valhalla fails for a segment, automatically falls back to OSRM.
        If both fail, uses a straight line but warns loudly.

        Args:
            waypoints: (N, 2) array of (lat, lon) waypoints.

        Returns:
            List of (lat, lon) points forming the complete route.
        """
        cache_key = self._cache_key(waypoints)
        cached = self._load_cache(cache_key)
        if cached is not None:
            # Reject cached straight-line results (route should have more
            # points than the input waypoints if it was actually routed).
            if len(cached) > len(waypoints) * 1.5:
                logger.info("Using cached route (%d points)", len(cached))
                return cached
            logger.warning(
                "Cached route looks unrouted (%d points for %d waypoints) "
                "— re-routing.",
                len(cached), len(waypoints),
            )

        # Lazy-init an OSRM fallback router
        osrm_fallback = OSRMRouter(
            profile="bike",
            request_delay=self.request_delay,
        )

        full_route: list[tuple[float, float]] = []
        n_segments = len(waypoints) - 1
        routed_count = 0
        fallback_count = 0
        straight_count = 0

        for i in range(n_segments):
            start = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            end = (float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1]))

            # Try Valhalla first
            segment = self.route_segment(start, end)

            # Fallback to OSRM if Valhalla failed
            if segment is None:
                segment = osrm_fallback.route_segment(start, end)
                if segment is not None:
                    fallback_count += 1
                else:
                    # Both routers failed — straight line as last resort
                    segment = [start, end]
                    straight_count += 1
            else:
                routed_count += 1

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

            if self.request_delay > 0 and i < n_segments - 1:
                time.sleep(self.request_delay)

            if (i + 1) % 10 == 0:
                logger.info(f"  Routed {i + 1}/{n_segments} segments")

        # Report routing quality
        if straight_count == n_segments:
            logger.error(
                "ALL %d segments fell back to straight lines — "
                "neither Valhalla nor OSRM was reachable. "
                "The route will NOT follow real roads.",
                n_segments,
            )
        elif straight_count > 0:
            logger.warning(
                "%d/%d segments used straight-line fallback "
                "(Valhalla routed %d, OSRM routed %d)",
                straight_count, n_segments, routed_count, fallback_count,
            )
        else:
            logger.info(
                "Routed %d segments (Valhalla: %d, OSRM fallback: %d)",
                n_segments, routed_count, fallback_count,
            )

        self._save_cache(cache_key, full_route)
        logger.info(f"Routed {len(waypoints)} waypoints → {len(full_route)} route points")
        return full_route


def create_router(config: dict) -> "ValhallaRouter | OSRMRouter":
    """Create a router from a routing config dict.

    Config keys: backend, valhalla_url, valhalla_costing,
                 osrm_url, osrm_profile, request_delay.
    """
    backend = config.get("backend", "valhalla")
    delay = config.get("request_delay", 1.0)

    if backend == "valhalla":
        return ValhallaRouter(
            base_url=config.get("valhalla_url", VALHALLA_BASE_URL),
            costing=config.get("valhalla_costing", "bicycle"),
            request_delay=delay,
        )
    return OSRMRouter(
        base_url=config.get("osrm_url", OSRM_BASE_URL),
        profile=config.get("osrm_profile", "driving"),
        request_delay=delay,
    )


class OSRMRouter:
    """Routes waypoints through OSRM. Fallback if Valhalla is unavailable."""

    def __init__(
        self,
        base_url: str = OSRM_BASE_URL,
        profile: str = "driving",
        request_delay: float = 1.0,
        cache_dir: Path | str = CACHE_DIR,
    ):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.request_delay = request_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def _cache_key(self, waypoints: np.ndarray) -> str:
        data = waypoints.tobytes()
        return hashlib.sha256(data).hexdigest()

    def _load_cache(self, key: str) -> list[tuple[float, float]] | None:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return [tuple(p) for p in json.load(f)]
        return None

    def _save_cache(self, key: str, route: list[tuple[float, float]]) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(route, f)

    def _segment_cache_key(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> str:
        data = f"{start[0]:.6f},{start[1]:.6f}-{end[0]:.6f},{end[1]:.6f}-{self.profile}"
        return "seg_osrm_" + hashlib.sha256(data.encode()).hexdigest()[:16]

    def _load_segment_cache(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        key = self._segment_cache_key(start, end)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return [tuple(p) for p in json.load(f)]
        return None

    def _save_segment_cache(
        self, start: tuple[float, float], end: tuple[float, float],
        segment: list[tuple[float, float]],
    ) -> None:
        key = self._segment_cache_key(start, end)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(segment, f)

    def route_segment(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        """Route between two points (lat, lon) via OSRM.

        Returns the routed path, or *None* if the server could not
        produce a route.
        """
        # Check segment cache first
        cached = self._load_segment_cache(start, end)
        if cached is not None:
            return cached

        coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"
        params = {"overview": "full", "geometries": "geojson"}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"OSRM retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.warning(f"OSRM request failed after {max_retries} retries: {e}")
                    return None

        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM returned no route: {data.get('code')}")
            return None

        geometry = data["routes"][0]["geometry"]["coordinates"]
        points = [(pt[1], pt[0]) for pt in geometry]
        self._save_segment_cache(start, end, points)
        return points

    def route_waypoints(
        self, waypoints: np.ndarray
    ) -> list[tuple[float, float]]:
        """Route through all waypoints sequentially."""
        cache_key = self._cache_key(waypoints)
        cached = self._load_cache(cache_key)
        if cached is not None:
            if len(cached) > len(waypoints) * 1.5:
                logger.info("Using cached route (%d points)", len(cached))
                return cached
            logger.warning(
                "Cached route looks unrouted (%d points for %d waypoints) "
                "— re-routing.",
                len(cached), len(waypoints),
            )

        full_route: list[tuple[float, float]] = []
        n_segments = len(waypoints) - 1
        straight_count = 0

        for i in range(n_segments):
            start = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            end = (float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1]))

            segment = self.route_segment(start, end)
            if segment is None:
                segment = [start, end]
                straight_count += 1

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

            if self.request_delay > 0 and i < n_segments - 1:
                time.sleep(self.request_delay)

        if straight_count == n_segments:
            logger.error(
                "ALL %d segments fell back to straight lines — "
                "OSRM was not reachable. The route will NOT follow real roads.",
                n_segments,
            )
        elif straight_count > 0:
            logger.warning(
                "%d/%d segments used straight-line fallback",
                straight_count, n_segments,
            )

        self._save_cache(cache_key, full_route)
        logger.info(f"Routed {len(waypoints)} waypoints → {len(full_route)} route points")
        return full_route
