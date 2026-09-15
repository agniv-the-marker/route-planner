import hashlib
import json

import numpy as np

from scripts.prepare_terrain import decode_terrarium
from src.geo import SF
from src.terrain import DATA, load_terrain, profile_markup


def test_terrarium_units():
    pixels = np.array([[[128, 0, 0], [128, 100, 128], [127, 255, 0]]])
    np.testing.assert_allclose(decode_terrarium(pixels), [[0, 100.5, -1]])


def test_terrain_matches_frame_and_provenance():
    elevations, image = load_terrain()
    metadata = json.loads((DATA / "sf-terrain.json").read_text())
    assert metadata["frame"] == repr(SF)
    assert elevations.shape == (1024, 1024)
    assert np.isfinite(elevations).all() and 200 < elevations.max() < 400
    assert metadata["sha256"] == hashlib.sha256((DATA / "sf-terrain.png").read_bytes()).hexdigest()
    assert image


def test_elevation_profile_has_real_distance_and_empty_state():
    assert profile_markup(None) == ""
    route = SF.pixels_to_meters([[200, 200], [250, 200], [250, 250], [200, 200]])
    html = profile_markup(route)
    assert "Approximate ground elevation" in html
    assert "mi" in html and "ft" in html and "km" not in html and "nan" not in html
