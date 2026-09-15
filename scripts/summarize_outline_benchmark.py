"""Summarize measured stages and hard upper bounds, without inventing review scores."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.outline_benchmark import DESCRIPTIONS, contact_sheet


def summarize(output=Path('outputs/outline-benchmark-v1')):
    payload = json.loads((output / 'results.json').read_text())
    records = payload['records']
    routes = json.loads((output / 'routes.json').read_text()) if (output / 'routes.json').exists() else []
    invalid = sum(not r['outline'] for r in records)
    valid_routes = sum(bool(r.get('routes')) for r in routes)
    warm = [r['wall_seconds'] for r in records if not r['cold_load_seconds']]
    search = [r['timings']['search_seconds'] for r in routes if 'timings' in r]
    errors = Counter(a['error'] for r in records for a in r['attempts'] if 'error' in a)
    summary = {
        'frozen_descriptions': DESCRIPTIONS, 'evaluated': len(records),
        'valid_outlines': len(records)-invalid, 'invalid_outlines': invalid,
        'maximum_possible_recognizable_outlines_out_of_30': 30-invalid,
        'outline_target': 24, 'route_target': 18,
        'valid_routes': valid_routes, 'routes_evaluated': len(routes),
        'recognizability': 'See numbered sheets and review.json; validity is not recognizability.',
        'outline_gate_impossible': 30-invalid < 24,
        'model_warm_wall_seconds': {'median': float(np.median(warm)), 'p95': float(np.percentile(warm,95))} if warm else None,
        'cpu_search_seconds': {'median': float(np.median(search)), 'p95': float(np.percentile(search,95))} if search else None,
        'model_cold_load_seconds': records[0]['cold_load_seconds'],
        'first_model_wall_seconds': records[0]['wall_seconds'],
        'validation_errors_by_attempt': dict(errors),
        'timing_note': 'Model RPC wall time and local CPU search measured separately. Not deployed end-to-end latency.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'descriptions.json').write_text(json.dumps(DESCRIPTIONS, indent=2))
    contact_sheet(records, output / 'outlines.png')
    contact_sheet(routes, output / 'routes.png', route=True)
    print(json.dumps({k:v for k,v in summary.items() if k != 'frozen_descriptions'}, indent=2))
    return summary

if __name__ == '__main__':
    summarize()
