"""Compile trusted, pinned MDI SVGs to validated data-only outer silhouettes."""
import json
import hashlib
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from svgpathtools import parse_path
from src.outlines import OutlineSpec

# Object silhouettes, excluding interface/status glyphs whose meaning requires internal detail.
NAMES = '''heart star fish butterfly cat circle kettle umbrella sail-boat guitar-acoustic
 elephant turtle mushroom key shoe-boot apple rocket-launch crown dog bird duck rabbit
 horse pig cow sheep dolphin shark owl penguin bat bee snail snake spider dinosaur
 bear jellyfish kangaroo koala panda octopus paw flower leaf pine-tree cactus palm-tree
 clover tree carrot chili-hot corn egg eggplant cherry pear pineapple strawberry watermelon
 lemon banana pepper food-croissant bread-slice pizza cheese ice-cream cake cupcake candy
 cookie coffee bottle-soda wine-glass beer glass-cocktail spoon fork knife scissors hammer
 wrench screwdriver axe saw-blade shovel magnet anchor bell lightbulb candle flashlight
 camera television radio headphones microphone music-note trumpet violin piano chess-pawn
 chess-knight chess-rook chess-king dice-5 basketball football baseball tennis-ball puzzle
 airplane helicopter car truck bus train bicycle motorbike scooter boat submarine
 home castle tent church lighthouse bridge wind-turbine factory hospital school
 mountain volcano earth moon-waning-crescent weather-cloudy weather-sunny lightning-bolt
 water fire snowflake diamond hexagon triangle square arrow-up cross infinity
 account human-handup hand-wave hand-heart hand-peace hand-thumb-up hand-fist
 baby-face robot alien ghost skull shield sword trophy medal flag gift balloon
 book-open-variant pencil pen paintbrush palette feather cup sunglasses tie t-shirt-crew
 hanger hat-fedora wizard-hat briefcase bag-personal suitcase wallet watch clock'''.split()


def compile_symbols(root, output):
    symbols, skipped = {}, {}
    for name in NAMES:
        path = root / 'svg' / f'{name}.svg'
        if not path.exists():
            skipped[name] = 'not in pinned collection'
            continue
        polygons = []
        for elem in ElementTree.parse(path).getroot().iter('{http://www.w3.org/2000/svg}path'):
            for subpath in parse_path(elem.attrib['d']).continuous_subpaths():
                pts = []
                for segment in subpath:
                    count = max(2, int(np.ceil(segment.length() / .15)))
                    pts.extend((segment.point(t).real, segment.point(t).imag) for t in np.linspace(0,1,count,endpoint=False))
                if len(subpath): pts.append((subpath[-1].end.real, subpath[-1].end.imag))
                if len(pts) < 3: continue
                poly = Polygon(pts)
                if poly.is_valid and poly.area > .01: polygons.append(poly)
        if not polygons:
            skipped[name] = 'no simple closed outer path'; continue
        combined = unary_union(polygons)
        if combined.geom_type != 'Polygon':
            for gap in (.3, .55, .8, 1.05):
                closed = combined.buffer(gap, join_style=2).buffer(-gap, join_style=2)
                if closed.geom_type == 'Polygon':
                    combined = closed
                    break
        if combined.geom_type != 'Polygon':
            skipped[name] = 'substantial disconnected parts'; continue
        outer = Polygon(combined.exterior)
        tolerance = .015
        simple = outer.simplify(tolerance, preserve_topology=True)
        while len(simple.exterior.coords) > 48:
            tolerance *= 1.2
            simple = outer.simplify(tolerance, preserve_topology=True)
        pts = np.array(simple.exterior.coords)
        # Subdivide straight edges for the 8-point contract, preserving every original corner.
        while len(pts) < 8:
            delta = np.linalg.norm(np.diff(pts,axis=0),axis=1)
            i = int(np.argmax(delta))
            pts = np.insert(pts, i+1, (pts[i]+pts[i+1])/2, axis=0)
        pts = (pts - pts.min(axis=0)) / np.ptp(pts,axis=0).max()
        spec = OutlineSpec.parse({'interpretation':name.replace('-',' '), 'points':pts.tolist()})
        symbols[name] = {'interpretation':spec.interpretation,'points':spec.points,
                         'source': f'https://github.com/Templarian/MaterialDesign-SVG/blob/v7.4.47/svg/{name}.svg',
                         'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    symbols['boot'] = {'interpretation':'boot',
                       'points':[[.1,0],[.55,0],[.5,.7],[.7,.73],[.97,.85],[1,.95],[.95,1],[.2,1],[.1,.88],[.1,0]],
                       'source':'Original Route Sculptor vector outline'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'version':'mdi-7.4.47-outlines-v1','symbols':symbols,'excluded':skipped},indent=2))
    print(f'{len(symbols)} symbols; {len(skipped)} excluded')

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    compile_symbols(args.source, Path('src/assets/symbols.json'))
