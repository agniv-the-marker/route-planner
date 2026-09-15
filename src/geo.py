"""The single coordinate frame used by images, streets, routes and previews."""

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class MapFrame:
    latitude: float = 37.76
    longitude: float = -122.45
    width_m: float = 10000.0
    size: int = 512

    @property
    def bounds(self):
        x, y = TO_METERS.transform(self.longitude, self.latitude)
        half = self.width_m / 2
        return x - half, y - half, x + half, y + half

    @property
    def polygon(self):
        return box(*self.bounds)

    @property
    def geographic_polygon(self):
        return Polygon([TO_GEO.transform(x, y) for x, y in self.polygon.exterior.coords])

    def pixels_to_meters(self, points):
        points = np.asarray(points, dtype=float)
        left, bottom, right, top = self.bounds
        return np.column_stack((left + points[:, 0] * self.width_m / (self.size - 1),
                                top - points[:, 1] * self.width_m / (self.size - 1)))

    def meters_to_pixels(self, points):
        points = np.asarray(points, dtype=float)
        left, bottom, right, top = self.bounds
        return np.column_stack(((points[:, 0] - left) * (self.size - 1) / self.width_m,
                                (top - points[:, 1]) * (self.size - 1) / self.width_m))


CRS = "EPSG:32610"
TO_METERS = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
TO_GEO = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
SF = MapFrame()
