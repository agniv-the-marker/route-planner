"""Frozen quality gate; anonymous SVGs are review artifacts, never model input."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from src.outlines import OutlineSpec

DESCRIPTIONS = [
 'heart', 'star', 'fish', 'butterfly', 'cat', 'circle',
 'teapot', 'umbrella', 'sailboat', 'guitar', 'elephant', 'turtle',
 'mushroom', 'key', 'boot', 'apple', 'rocket', 'crown',
 'a heart leaning to the left', 'a tall narrow five-pointed star',
 'a fish swimming to the right with a broad tail', 'a butterfly with rounded wings',
 'a sitting cat with pointed ears', 'a slightly flattened circle',
 'a cheerful symbol of love', 'a little boat with one triangular sail',
 'a sturdy rain boot with a wide toe', 'a mushroom with a broad cap',
 'a rocket ready to explore the moon', 'a royal crown with three peaks',
]

def fixtures():
    t = np.linspace(0, 2*np.pi, 33)
    heart = np.column_stack((16*np.sin(t)**3, -(13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))))
    angles = np.arange(10)*np.pi/5 - np.pi/2
    star = np.column_stack((np.cos(angles), np.sin(angles))) * np.tile([1, .45], 5)[:, None]
    shapes = {'heart': heart[:-1], 'star': star,
              'fish': [(0,.5),(.2,.25),(.55,.15),(.8,.4),(1,.15),(.95,.5),(1,.85),(.8,.6),(.55,.85),(.2,.75)],
              'butterfly': [(.5,.3),(.3,0),(0,.05),(.05,.4),(.3,.5),(.05,.65),(.1,1),(.4,.85),(.5,.65),(.6,.85),(.9,1),(.95,.65),(.7,.5),(.95,.4),(1,.05),(.7,0)],
              'cat': [(.2,1),(.15,.8),(.25,.5),(.25,0),(.45,.2),(.65,.2),(.85,0),(.85,.45),(.75,.6),(.8,.9),(.95,1)],
              'circle': np.column_stack((np.cos(t[:-1]), np.sin(t[:-1])))}
    result = {}
    for name, p in shapes.items():
        p = np.asarray(p, dtype=float)
        p = (p - p.min(axis=0)) / np.ptp(p, axis=0).max()
        p = np.vstack((p, p[0]))
        result[name] = OutlineSpec.parse({'interpretation': name, 'points': p.tolist()})
    return result


def svg(points, path):
    p = np.asarray(points, dtype=float)
    span = max(np.ptp(p, axis=0).max(), 1e-9)
    p = (p - (p.min(axis=0) + p.max(axis=0)) / 2) / span * 440 + 250
    coords = ' '.join(f'{x:.2f},{y:.2f}' for x,y in p)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500"><rect width="500" height="500" fill="white"/><polyline points="{coords}" fill="none" stroke="black" stroke-width="3"/></svg>')


def routing_proof(output):
    from src.graph import BikeRouter
    from src.street_search import StreetSearch
    if (output / 'routing.json').exists():
        raise ValueError('Choose a fresh output directory to preserve previous evidence.')
    output.mkdir(parents=True, exist_ok=True)
    search = StreetSearch(BikeRouter.load().graph)
    records = []
    for i, (name, spec) in enumerate(fixtures().items()):
        result = search.search(spec)
        record = asdict(result)
        record['name'] = name
        records.append(record)
        svg(spec.points, output / f'{i+1:02}-outline.svg')
        if result.routes:
            p = np.asarray(result.routes[0]['xy']) * [1, -1]
            svg(p, output / f'{i+1:02}-route.svg')
        (output / 'routing.json').write_text(json.dumps(records, indent=2))
        print(name, len(result.routes), result.timings, result.diagnostics, flush=True)
    return records

def complete_routes(output=Path('outputs/outline-benchmark-v1'), follow=False):
    """Evaluate saved model outputs on CPU; no further model calls."""
    import time
    from src.graph import BikeRouter
    from src.street_search import StreetSearch
    from src.geo import TO_GEO
    from src.generation import to_gpx
    search = StreetSearch(BikeRouter.load().graph)
    records = []
    for record in saved_records(output / 'results.json', follow):
        row = {'id': record['id'], 'prompt': record['prompt']}
        if record['outline']:
            spec = OutlineSpec.parse(record['outline'])
            result = search.search(spec)
            row.update(asdict(result))
            if result.routes:
                xy = np.asarray(result.routes[0]['xy'])
                svg(xy * [1, -1], output / f"{record['id']:02}-route.svg")
                lon, lat = TO_GEO.transform(xy[:, 0], xy[:, 1])
                (output / f"{record['id']:02}.gpx").write_text(to_gpx(list(zip(lat, lon)), record['prompt']))
                started = time.perf_counter()
                cached = search.search(spec)
                row['route_cache_seconds'] = time.perf_counter() - started
                assert cached.routes == result.routes
        else:
            row['failure_reason'] = record.get('failure_reason', 'Outline unavailable')
        records.append(row)
        (output / 'routes.json').write_text(json.dumps(records, indent=2))
        print(row['id'], len(row.get('routes', [])), flush=True)
    return records


def contact_sheet(records, output, route=False):
    """Labels are numeric review IDs only; interpretation stays in separate JSON."""
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (1000, 200 * int(np.ceil(len(records) / 5))), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, record in enumerate(records):
        x, y = (i % 5) * 200, (i // 5) * 200
        draw.text((x + 8, y + 8), str(record.get('id', i + 1)), fill='black')
        value = record.get('routes', []) if route else record.get('outline')
        if not value:
            draw.text((x + 45, y + 95), 'Unavailable', fill='gray')
            continue
        p = np.asarray(value[0]['xy'] if route else value['points'], dtype=float)
        if route: p *= [1, -1]
        span = max(np.ptp(p, axis=0).max(), 1e-9)
        p = (p - (p.min(axis=0) + p.max(axis=0)) / 2) / span * 165 + [x + 100, y + 110]
        draw.line([tuple(q) for q in p], fill='black', width=2)
    canvas.save(output)


def saved_records(path, follow=False):
    import time
    deadline = time.monotonic() + 1200
    seen = 0
    while True:
        try:
            records = json.loads(path.read_text())['records']
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        for record in records[seen:]:
            yield record
            seen += 1
        if not follow or seen == len(DESCRIPTIONS) or path.with_name('STOPPED.json').exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Bounded benchmark did not finish; partial artifacts retained.')
        time.sleep(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--routes', action='store_true')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.routes:
        complete_routes(args.output_dir or Path('outputs/outline-benchmark-v1'), follow=args.follow)
    else:
        routing_proof(args.output_dir or Path('outputs/vector-routing-proof-v2'))
