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

    def route_segment(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Route between two points (lat, lon) via Valhalla."""
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
                resp = self._session.get(
                    url, params={"json": json.dumps(payload)}, timeout=30
                )
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
                    return [start, end]

        # Extract the shape from the response
        try:
            shape = data["trip"]["legs"][0]["shape"]
            return self._decode_polyline(shape)
        except (KeyError, IndexError) as e:
            logger.warning(f"Valhalla response parsing failed: {e}")
            return [start, end]

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

        Args:
            waypoints: (N, 2) array of (lat, lon) waypoints.

        Returns:
            List of (lat, lon) points forming the complete route.
        """
        cache_key = self._cache_key(waypoints)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Using cached route")
            return cached

        full_route: list[tuple[float, float]] = []

        for i in range(len(waypoints) - 1):
            start = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            end = (float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1]))

            segment = self.route_segment(start, end)

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

            if self.request_delay > 0 and i < len(waypoints) - 2:
                time.sleep(self.request_delay)

            if (i + 1) % 10 == 0:
                logger.info(f"  Routed {i + 1}/{len(waypoints) - 1} segments")

        self._save_cache(cache_key, full_route)
        logger.info(f"Routed {len(waypoints)} waypoints → {len(full_route)} route points")
        return full_route


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

    def route_segment(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Route between two points (lat, lon) via OSRM."""
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
                    return [start, end]

        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM returned no route: {data.get('code')}")
            return [start, end]

        geometry = data["routes"][0]["geometry"]["coordinates"]
        return [(pt[1], pt[0]) for pt in geometry]

    def route_waypoints(
        self, waypoints: np.ndarray
    ) -> list[tuple[float, float]]:
        """Route through all waypoints sequentially."""
        cache_key = self._cache_key(waypoints)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Using cached route")
            return cached

        full_route: list[tuple[float, float]] = []

        for i in range(len(waypoints) - 1):
            start = (float(waypoints[i, 0]), float(waypoints[i, 1]))
            end = (float(waypoints[i + 1, 0]), float(waypoints[i + 1, 1]))

            segment = self.route_segment(start, end)

            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

            if self.request_delay > 0 and i < len(waypoints) - 2:
                time.sleep(self.request_delay)

        self._save_cache(cache_key, full_route)
        logger.info(f"Routed {len(waypoints)} waypoints → {len(full_route)} route points")
        return full_route
