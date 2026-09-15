"""Build fixed-frame shaded relief, adapting ferryri.de's paper/ink treatment.

Terrain: Mapzen Terrarium / USGS via AWS Open Data. Coast: MTC/ABAG county
shoreline, prepared in the user's ferryri.de repository. No runtime tile service.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates, sobel
from shapely.geometry import shape
from shapely.ops import transform

from src.geo import SF, TO_GEO, TO_METERS

DATA = Path(__file__).resolve().parent.parent / "data"
COAST_URL = ("https://raw.githubusercontent.com/agniv-the-marker/ferryri.de/"
             "46e43796cc9f869723de921bd94d350795a217d0/public/data/coast.json")
TILE_ROOT = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"


def fetch(url):
    with urlopen(url, timeout=45) as response:
        return response.read()


def decode_terrarium(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def prepare():
    resolution, zoom = 1024, 13
    axis = np.linspace(0, SF.size - 1, resolution)
    px, py = np.meshgrid(axis, axis)
    xy = SF.pixels_to_meters(np.column_stack([px.ravel(), py.ravel()]))
    lon, lat = TO_GEO.transform(xy[:, 0], xy[:, 1])
    world_x = (lon + 180) / 360 * 2**zoom * 256
    radians = np.deg2rad(lat)
    world_y = (1 - np.arcsinh(np.tan(radians)) / np.pi) / 2 * 2**zoom * 256
    x0, x1 = int(world_x.min() // 256), int(world_x.max() // 256)
    y0, y1 = int(world_y.min() // 256), int(world_y.max() // 256)
    cache = DATA.parent / ".cache" / "terrain"
    cache.mkdir(parents=True, exist_ok=True)

    def tile(pair):
        x, y = pair
        path = cache / f"{zoom}-{x}-{y}.png"
        if not path.exists():
            path.write_bytes(fetch(f"{TILE_ROOT}/{zoom}/{x}/{y}.png"))
        return x, y, decode_terrarium(Image.open(path).convert("RGB"))

    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
    pairs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for x, y, heights in pool.map(tile, pairs):
            mosaic[(y-y0)*256:(y-y0+1)*256, (x-x0)*256:(x-x0+1)*256] = heights
    elevation = map_coordinates(mosaic, [world_y-y0*256, world_x-x0*256], order=1,
                                mode="nearest").reshape(resolution, resolution)
    if not np.isfinite(elevation).all() or elevation.min() < -500:
        raise RuntimeError("Unexpected elevation values in SF terrain data.")

    # Project the reference shoreline before clipping; all layers share SF's frame.
    land = transform(TO_METERS.transform, shape(json.loads(fetch(COAST_URL))))
    land = land.intersection(SF.polygon)
    mask_image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(mask_image)
    polygons = list(land.geoms) if hasattr(land, "geoms") else [land]
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            continue
        for ring, fill in [(polygon.exterior, 255), *[(r, 0) for r in polygon.interiors]]:
            pixels = SF.meters_to_pixels(ring.coords) * (resolution - 1) / (SF.size - 1)
            draw.polygon([tuple(p) for p in pixels], fill=fill)
    mask = np.asarray(mask_image) / 255

    # Horn derivatives and NW lighting; retain the reference's soft-knee density
    # and four-level ordered dither, rendered statically for the small map.
    cell = SF.width_m / (resolution - 1)
    height = gaussian_filter(elevation, 1)
    dx, dy = sobel(height, axis=1) / (8 * cell), sobel(height, axis=0) / (8 * cell)
    shade = (dx * .5 + dy * .5 + 2**-.5) / np.sqrt(1 + dx*dx + dy*dy)
    relative = np.maximum(0, .72 - shade) * 6
    shade_tone = relative / (1 + relative)
    elevation_tone = np.clip(np.maximum(elevation, 0) / 850, 0, 1)**.75 * .8
    shore_tone = (1 - gaussian_filter(mask, 5)) * .12
    density = (1 - (1-shade_tone)*(1-elevation_tone)*(1-shore_tone)) * .55
    bayer = (np.array([[0,8,2,10], [12,4,14,6], [3,11,1,9], [15,7,13,5]]) + .5) / 16
    threshold = np.tile(bayer, (resolution//4, resolution//4))
    tone = np.clip(np.floor(density * 3 + threshold) / 3, 0, 1) * .38
    paper = np.array([224, 232, 222])
    ink = np.array([53, 72, 59])
    land_rgb = paper + tone[..., None] * (ink - paper)
    yy, xx = np.indices(mask.shape)
    water_tone = .5 + .12*np.sin(xx / 26 + yy / 41) + .06*np.sin(xx / 9 - yy / 17)
    water_tone = (water_tone > threshold)[..., None]
    water_rgb = np.where(water_tone, [191, 207, 218], [176, 195, 209])
    rgb = np.where(mask[..., None] > .5, land_rgb, water_rgb)
    # A fine shoreline stroke follows the actual land mask.
    coast_edge = (abs(sobel(mask, axis=0)) + abs(sobel(mask, axis=1))) > 0
    rgb[coast_edge] = rgb[coast_edge] * .75 + np.array([71, 88, 76]) * .25
    path = DATA / "sf-terrain.png"
    Image.fromarray(np.uint8(np.clip(rgb, 0, 255))).save(path, optimize=True)
    np.save(DATA / "sf-elevation.npy", elevation.astype(np.float32))
    metadata = {"frame": repr(SF), "resolution": resolution, "zoom": zoom,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "tiles": pairs,
                "source": TILE_ROOT, "coast_source": COAST_URL,
                "style_reference": "https://github.com/agniv-the-marker/ferryri.de",
                "attribution": "Terrain: Mapzen / USGS / NOAA. Coast: MTC/ABAG (TIGER-derived).",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "elevation_sha256": hashlib.sha256((DATA / "sf-elevation.npy").read_bytes()).hexdigest()}
    (DATA / "sf-terrain.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {resolution}px terrain from {len(pairs)} tiles; elevations {elevation.min():.1f}–{elevation.max():.1f}m")


if __name__ == "__main__":
    prepare()
