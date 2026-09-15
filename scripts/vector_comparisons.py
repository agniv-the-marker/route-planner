"""Balanced same-prompt route comparisons for human/judge calibration."""
import argparse
import html
import itertools
import json
from pathlib import Path


def create(output):
    records={int(p.stem):json.loads(p.read_text()) for p in output.glob('[0-9][0-9][0-9].json')}
    pairs=[]
    for start in range(1,121,4):
        available=[i for i in range(start,start+4) if records.get(i,{}).get('route')]
        for a,b in itertools.combinations(available,2):
            if len(pairs)%2: a,b=b,a
            pairs.append({'id':len(pairs)+1,'a':a,'b':b,'prompt':records[a]['prompt'],
                          'human':None,'judge':None})
    pairs=pairs[:50]
    path=output/'comparisons.json'
    if path.exists():
        raise ValueError('Existing comparisons retained; choose a fresh review location.')
    path.write_text(json.dumps(pairs,indent=2))
    rows=[]
    for pair in pairs:
        i,a,b=pair['id'],pair['a'],pair['b']
        rows.append(f'<article><h2>{i}: {html.escape(pair["prompt"])}</h2><img width="300" src="{a:03}-route.png"><img width="300" src="{b:03}-route.png"><p>Which street drawing better preserves the subject, pose and proportions?</p><select data-id="{i}"><option value="">Unreviewed</option><option value="a">Left</option><option value="b">Right</option><option value="tie">Tie</option></select></article>')
    payload=json.dumps(pairs).replace('<','\\u003c')
    script='''<button onclick="save()">Export comparisons</button><script>const pairs=PAYLOAD;
function save(){for(const s of document.querySelectorAll('select'))pairs.find(p=>p.id===+s.dataset.id).human=s.value||null;
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(pairs,null,2)],{type:'application/json'}));a.download='comparisons-reviewed.json';a.click();URL.revokeObjectURL(a.href);}</script>'''.replace('PAYLOAD',payload)
    (output/'comparisons.html').write_text('<!doctype html><meta charset="utf-8"><h1>Blind route comparison</h1>'+''.join(rows)+script)
    return len(pairs)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    print('Available comparisons:',create(parser.parse_args().output))
