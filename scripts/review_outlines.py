"""Render numbered review sheets and street/outline alignment overlays."""
import json
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from scripts.outline_benchmark import contact_sheet
from src.geo import SF
from src.graph import BikeRouter
from src.generation import render_streets
from src.outlines import OutlineSpec
from src.street_search import StreetSearch


def render_proof(output=Path('outputs/vector-routing-proof-v2')):
    records = json.loads((output / 'routing.json').read_text())
    contact_sheet(records, output / 'outlines.png')
    contact_sheet(records, output / 'routes.png', route=True)
    background = render_streets(BikeRouter.load().graph).resize((1024, 1024))
    for i, record in enumerate(records):
        if not record['routes']:
            continue
        route = record['routes'][0]
        t = route['transform']
        # transform only uses OutlineSpec; no graph/index construction needed.
        p = StreetSearch.transform(None, OutlineSpec.parse(record['outline']),
                                   t['x'], t['y'], t['angle'], t['perimeter'])
        image = background.copy()
        draw = ImageDraw.Draw(image)
        draw.line([tuple(q) for q in SF.meters_to_pixels(p)*2], fill='#cc5966', width=3)
        draw.line([tuple(q) for q in SF.meters_to_pixels(route['xy'])*2], fill='#234dab', width=3)
        draw.text((20,20), f"{i+1:02}  angle={t['angle']} deg CCW  {route['distance_m']/1000:.2f} km", fill='black')
        draw.text((20,40), 'Pink: transformed outline / blue: directed street route / north up', fill='black')
        image.save(output / f'{i+1:02}-map.png')

if __name__ == '__main__':
    render_proof()
