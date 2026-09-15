"""Frozen offline evaluation and artifact export. Never invokes training."""
import hashlib
import html
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.outlines import OutlineSpec
from src.route_reward import reward
from src.experiment_journal import Journal, digest, source_identity, atomic_json

SUBJECTS = ['a pangolin with a long tapered tail', 'a narwhal with a long straight tusk',
'a pitcher plant with a flared opening', 'a horseshoe crab with a pointed tail',
'a seahorse with a curled tail', 'a ginkgo leaf with a central notch',
'a toucan with an oversized beak', 'a manta ray with broad triangular fins',
'a baobab with a thick trunk and small crown', 'a curled fiddlehead fern']
PAIRS = [('a standing horse','a galloping horse'), ('a sitting rabbit','a leaping rabbit'),
('a bird with folded wings','a bird with spread wings'), ('a tall narrow cactus','a short wide cactus'),
('a dog with its tail raised','a dog with its tail lowered'), ('a swan with a straight neck','a swan with a curved neck'),
('a left-facing fox','a right-facing fox'), ('a squat teapot with a short spout','a tall teapot with a long spout'),
('a closed umbrella','an open umbrella'), ('a standing bear','a rearing bear')]
PROMPTS = SUBJECTS + [p for pair in PAIRS for p in pair]


def freeze(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = {'version':'prompts-v1', 'prompts':PROMPTS, 'pairs':PAIRS,
            'attempt_limit':120, 'gpu_seconds_limit':7200,
            'recognizable_outline_target':24, 'recognizable_route_target':18}
    data['sha256'] = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(data, indent=2))
    return output


def route_canvas(xy):
    p = np.asarray(xy).copy()
    p[:,1] *= -1
    low, high = p.min(0), p.max(0)
    p = (p-(low+high)/2)*430/max(high-low)+256
    im = Image.new('RGB',(512,512),'white')
    ImageDraw.Draw(im).line(list(map(tuple,p)),fill='black',width=4)
    return im


def evaluate(record, search, output, number):
    from src.generative import Candidate
    from src.street_search import VERSION as router_version
    from src.generation import render_outline, to_gpx
    from src.geo import TO_GEO
    record = Candidate(**record) if isinstance(record,dict) else record
    stem = f'{number:03}'
    record.scores['routing_config'] = {'router_version':router_version,
                                       'work_budget':150000,'wall_seconds':8}
    if record.outline:
        spec = OutlineSpec.parse(record.outline)
        render_outline(spec).save(output/f'{stem}-outline.png')
        try:
            found = search.search(spec, budget=8, work_budget=150000)
        except Exception as exc:
            record.infrastructure_error = str(exc)
            record.scores['reward'] = reward(record)
            (output/f'{stem}.json').write_text(json.dumps(asdict(record),indent=2))
            return asdict(record)
        record.timings.update(found.timings)
        if found.routes:
            record.route = found.routes[0]
            route_canvas(record.route['xy']).save(output/f'{stem}-route.png')
            xy = np.array(record.route['xy'])
            lon,lat = TO_GEO.transform(xy[:,0],xy[:,1])
            (output/f'{stem}.gpx').write_text(to_gpx(list(zip(lat,lon)),record.prompt))
        record.scores['routing'] = found.diagnostics
        record.scores['routing_failure'] = found.failure_reason
    record.scores['reward'] = reward(record)
    (output/f'{stem}.json').write_text(json.dumps(asdict(record),indent=2))
    return asdict(record)


def review(output, records):
    rows = []
    for i,r in enumerate(records,1):
        images = ''.join(f'<img width="250" src="{i:03}-{kind}.png">' for kind in ('outline','route') if (output/f'{i:03}-{kind}.png').exists())
        rows.append(f'<article><h3>{i:03}: {html.escape(r["prompt"])}</h3>{images}<p>{html.escape(r["validation"])}</p><label>Recognizable outline <input type="checkbox" data-id="{i}" data-kind="outline"></label><label>Recognizable route including distinguishing features <input type="checkbox" data-id="{i}" data-kind="route"></label></article>')
    # Numbered plain-canvas sheets; failures remain visible as empty cells.
    for kind in ('outline','route'):
        for offset in range(0,len(records),20):
            sheet=Image.new('RGB',(1280,1500),'white')
            draw=ImageDraw.Draw(sheet)
            for cell in range(min(20,len(records)-offset)):
                i=offset+cell+1
                x,y=(cell%4)*320,(cell//4)*300
                path=output/f'{i:03}-{kind}.png'
                if path.exists(): sheet.paste(Image.open(path).resize((256,256)),(x,y+25))
                draw.text((x+5,y+5),f'{i:03} {kind}',fill='black')
            sheet.save(output/f'{kind}-sheet-{offset//20+1:02}.png')
    script = '''<button onclick="save()">Export human reviews</button><script>
function save(){const rows=[...document.querySelectorAll('input')].map(x=>({id:+x.dataset.id,kind:x.dataset.kind,recognizable:x.checked}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='human-reviews.json';a.click();URL.revokeObjectURL(a.href);}</script>'''
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Generated drawing review</title><h1>Experimental generator — human recognition review required</h1>'+''.join(rows)+script)
    summary = {'attempts':len(records), 'valid_geometry':sum(r['validation']=='valid' for r in records),
               'qualifying_routes':sum(r['route'] is not None for r in records),
               'recognition':'unreviewed', 'promotion_allowed':False,
               'first_sample_ids':list(range(1,len(records)+1,4)),
               'best_of_four':'Requires prompt fidelity judgments, not geometry alone'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    import argparse
    from src.street_search import StreetSearch
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    search=StreetSearch.load_prepared()
    search.route_cache_dir=Path('route_cache/generative-v1')
    records=[]
    artifacts=[]
    for path in sorted(args.output.glob('*-raw.json')):
        i=int(path.name.split('-')[0])
        raw = json.loads(path.read_text())
        journal = Journal(args.output/'journal')
        raw_hash = journal.raw({'attempt':i},raw)
        identity = {'raw_hash':raw_hash,'source':source_identity(),'graph':search.graph_hash,
                    'work_budget':150000,'wall_seconds':8}
        stage_output = args.output/'evaluations'/digest(identity)
        def operation():
            # Cached compiler output in historical raw records is not authoritative.
            from src.vector_program import compile_program, extract_program
            from src.generative import Candidate
            record = Candidate(**raw)
            record.outline, record.route = None, None
            if record.raw is not None:
                try:
                    record.program = extract_program(record.raw)
                    record.outline = asdict(compile_program(record.raw,record.prompt))
                    record.validation = 'valid'
                except (ValueError,TypeError,OverflowError) as exc:
                    record.validation = str(exc)
            stage_output.mkdir(parents=True,exist_ok=True)
            return evaluate(record,search,stage_output,i)
        record = journal.run({'attempt':i},'compile-route',identity,operation)
        records.append(record)
        artifacts.append(stage_output)
    # A fresh report cannot accidentally display historical route thumbnails.
    import shutil
    report_output = args.output/'reports'/digest([str(p) for p in artifacts])
    report_output.mkdir(parents=True,exist_ok=True)
    for stage_output in artifacts:
        for artifact in stage_output.iterdir():
            if artifact.is_file():
                shutil.copy2(artifact,report_output/artifact.name)
    review(report_output,records)
    print(report_output/'review.html')
