"""Recognition gates require explicit human evidence, reported first and best-of-four."""
import argparse
import json
from pathlib import Path
from src.route_reward import calibration


def summarize(output, reviews=None, comparisons=None):
    manifest = json.loads((output/'manifest.json').read_text())
    records = {int(p.stem):json.loads(p.read_text()) for p in output.glob('[0-9][0-9][0-9].json')}
    labels = {(r['id'],r['kind']):r['recognizable'] is True for r in (reviews or [])}
    report = {'attempts':len(records),'expected_attempts':120,'prompts':len(manifest['prompts']),
              'human_labels':len(labels),'recognition_status':'unreviewed' if not labels else 'human labels supplied',
              'first_sample':{},'best_of_four':{},'geometry_and_routing':{},
              'infrastructure_errors':sum(bool(r['infrastructure_error']) for r in records.values())}
    for kind in ('outline','route'):
        def passes(i):
            r=records.get(i,{})
            return bool(r.get(kind)) and labels.get((i,kind),False)
        report['first_sample'][kind]=sum(passes(i) for i in range(1,121,4))
        report['best_of_four'][kind]=sum(any(passes(i+j) for j in range(4)) for i in range(1,121,4))
    for kind in ('outline','route'):
        report['geometry_and_routing'][kind] = {
            'first_sample':sum(bool(records.get(i,{}).get(kind)) for i in range(1,121,4)),
            'best_of_four_upper_bound':sum(any(records.get(i+j,{}).get(kind) for j in range(4))
                                          for i in range(1,121,4))}
    from collections import Counter
    report['validation_failures']=dict(Counter(r['validation'] for r in records.values() if r['validation']!='valid'))
    report['calibration']=calibration(comparisons or [])
    report['promotion_allowed']=(len(records)==120 and report['best_of_four']['outline']>=24
                                 and report['best_of_four']['route']>=18)
    report['rl_pilot_allowed']=report['promotion_allowed'] and report['calibration']['rl_gate']
    report['timings']={key:sum(r['timings'].get(key,0) for r in records.values())
                       for key in {k for r in records.values() for k in r['timings']}}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--comparisons',type=Path)
    a=p.parse_args()
    result=summarize(a.output,json.loads(a.reviews.read_text()) if a.reviews else None,
                     json.loads(a.comparisons.read_text()) if a.comparisons else None)
    (a.output/'recognition-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
