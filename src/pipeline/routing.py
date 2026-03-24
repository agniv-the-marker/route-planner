"""OSRM routing client for computing bike routes between waypoints."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

OSRM_BASE_URL = "https://router.project-osrm.org"
CACHE_DIR = Path("route_cache")


class OSRMRouter:
    """Routes waypoints through OSRM to get bikeable road paths."""

    def __init__(
        self,
        base_url: str = OSRM_BASE_URL,
        profile: str = "bike",
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
        """Generate a cache key from waypoints."""
        data = waypoints.tobytes()
        return hashlib.sha256(data).hexdigest()

    def _load_cache(self, key: str) -> list[tuple[float, float]] | None:
        """Load cached route if available."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return [tuple(p) for p in json.load(f)]
        return None

    def _save_cache(self, key: str, route: list[tuple[float, float]]) -> None:
        """Save route to cache."""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(route, f)

    def route_segment(
        self, start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Route between two points (lat, lon).

        Returns list of (lat, lon) points along the route.
        """
        # OSRM expects lon,lat order in the URL
        coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"
        params = {
            "overview": "full",
            "geometries": "geojson",
        }

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"OSRM request failed for segment: {e}")
            # Fallback: straight line
            return [start, end]

        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM returned no route: {data.get('code')}")
            return [start, end]

        # Extract geometry (GeoJSON is [lon, lat])
        geometry = data["routes"][0]["geometry"]["coordinates"]
        return [(pt[1], pt[0]) for pt in geometry]

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

            # Avoid duplicating junction points
            if full_route and segment:
                full_route.extend(segment[1:])
            else:
                full_route.extend(segment)

            # Rate limiting
            if self.request_delay > 0 and i < len(waypoints) - 2:
                time.sleep(self.request_delay)

        self._save_cache(cache_key, full_route)
        logger.info(f"Routed {len(waypoints)} waypoints → {len(full_route)} route points")
        return full_route
