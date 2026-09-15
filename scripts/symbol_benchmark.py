"""Full symbol-catalogue routability sweep against the real prepared SF bike graph.

Every entry in src/assets/symbols.json is a hand-vetted, pre-validated polygon, so this
measures the StreetSearch matching/placement stage in isolation, without any model call.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.street_search import StreetSearch
from src.symbols import catalogue, outline_from_selection


def sweep(output, budget=8):
    if (output / 'results.json').exists():
        raise ValueError('Choose a fresh output directory to preserve previous evidence.')
    output.mkdir(parents=True, exist_ok=True)
    search = StreetSearch.load_prepared()
    names = sorted(catalogue())
    records = []
    for i, name in enumerate(names):
        spec = outline_from_selection({'symbol': name, 'tilt': 0, 'aspect': 'normal'})
        started = time.perf_counter()
        result = search.search(spec, budget=budget)
        record = {'name': name, 'routes': len(result.routes),
                  'seconds': round(time.perf_counter() - started, 2),
                  'failure_reason': result.failure_reason,
                  'rejections': result.diagnostics.get('rejections')}
        if result.routes:
            record['distance_m'] = result.routes[0]['distance_m']
        records.append(record)
        (output / 'results.json').write_text(json.dumps(records, indent=2))
        print(i + 1, len(names), name, record, flush=True)
    passed = sum(1 for r in records if r['routes'])
    summary = {'version': 'symbol-routability-v1', 'graph_hash': search.graph_hash,
               'index_seconds': round(search.index_seconds, 2), 'budget_seconds': budget,
               'total': len(records), 'passed': passed,
               'median_seconds': float(np.median([r['seconds'] for r in records])),
               'failures': [r['name'] for r in records if not r['routes']]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)
    return records, summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/symbol-routability-v1'))
    parser.add_argument('--budget', type=float, default=8)
    args = parser.parse_args()
    sweep(args.output_dir, args.budget)
