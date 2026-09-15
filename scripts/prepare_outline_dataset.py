"""Build licensed text -> closed-outline SFT records from the MDI catalogue.

The catalogue is a bootstrap corpus, not evidence of arbitrary-text recognition.
Every record retains the upstream icon URL/hash and a subject-family split.
"""
import argparse
import hashlib
import json
from pathlib import Path
from shapely.geometry import Polygon

from scripts.vector_training import subject_split

MODEL_SYSTEM = (
    'Draw the exact subject as one closed silhouette. Return only a JSON array of '
    'M/L/Q/C/Z commands with integer coordinates from 0 to 1024. Start with M and '
    'end with Z. Use one subpath, at most 32 commands, and preserve distinguishing '
    'features. Do not return prose or SVG.'
)

TEMPLATES = (
    'a {name}',
    'a simple {name} silhouette',
    'the outline of a {name}',
    'a recognizable {name} icon',
)


def points_to_program(points):
    polygon = Polygon(points)
    tolerance = 0.0005
    while len(polygon.exterior.coords) - 1 > 31 and tolerance < .2:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if simplified.geom_type == 'Polygon' and simplified.is_valid:
            polygon = simplified
        tolerance *= 1.6
    if len(polygon.exterior.coords) - 1 > 31:
        raise ValueError('Catalogue outline cannot be represented in 32 commands.')
    scaled = [[int(round(float(x) * 1024)), int(round(float(y) * 1024))]
              for x, y in polygon.exterior.coords]
    if scaled[-1] != scaled[0]:
        scaled.append(scaled[0])
    return [['M', *scaled[0]]]+[['L', *p] for p in scaled[1:-1]]+[['Z']]


def build(output, held_out=()):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh dataset directory: {output}')
    catalogue = json.loads(Path('src/assets/symbols.json').read_text())
    held = {x.casefold().strip() for x in held_out}
    rows = []
    for name, symbol in sorted(catalogue['symbols'].items()):
        family = name.casefold().replace('-', ' ')
        program = points_to_program(symbol['points'])
        for template in TEMPLATES:
            prompt = template.format(name=name.replace('-', ' '))
            split = 'test' if family in held else subject_split(family)
            rows.append({
                'prompt': prompt, 'subject_family': family, 'program': program,
                'license': 'Pictogrammers Free License / Apache-2.0 icon corpus',
                'source': symbol['source'],
                'source_sha256': symbol.get('source_sha256', hashlib.sha256(
                    json.dumps(symbol['points'], separators=(',', ':')).encode()).hexdigest()),
                'split': split, 'system': MODEL_SYSTEM,
            })
    output.mkdir(parents=True)
    with (output / 'examples.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    manifest = {
        'version': 'outline-sft-v1', 'catalogue_version': catalogue['version'],
        'templates': TEMPLATES, 'system': MODEL_SYSTEM, 'examples': len(rows),
        'subject_families': len(catalogue['symbols']), 'held_out_families': sorted(held),
        'license': 'Pictogrammers Free License / Apache-2.0 icon corpus',
        'source_file_sha256': hashlib.sha256(Path('src/assets/symbols.json').read_bytes()).hexdigest(),
    }
    manifest['examples_sha256'] = hashlib.sha256((output / 'examples.jsonl').read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--held-out', nargs='*', default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.held_out), indent=2))
