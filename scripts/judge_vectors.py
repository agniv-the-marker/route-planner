"""Offline vision evaluation with an injected frozen judge; never generates samples."""
import json
import time
from pathlib import Path
from src.generative import Candidate, cache_key


def judge_saved(output, judge, cache_dir=Path('route_cache/generative-evaluations')):
    output, cache_dir = Path(output), Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in sorted(output.glob('[0-9][0-9][0-9].json')):
        record=Candidate(**json.loads(path.read_text()))
        graph_hash=record.scores.get('routing',{}).get('graph_hash')
        key=cache_key(record,graph_hash,[judge.model,judge.revision])
        cached=cache_dir/f'{key}.json'
        start=time.perf_counter()
        if cached.exists():
            scores=json.loads(cached.read_text())
            timing={'cache_seconds':time.perf_counter()-start}
        else:
            scores=judge(record)
            timing={'judging_seconds':time.perf_counter()-start}
            if not scores.get('excluded'):
                import os, tempfile
                with tempfile.NamedTemporaryFile(mode='w',dir=cache_dir,delete=False) as temporary:
                    json.dump(scores,temporary,indent=2)
                os.replace(temporary.name,cached)
        result={'id':int(path.stem),'candidate_key':key,'scores':scores,'timings':timing}
        (output/f'{path.stem}-judgment.json').write_text(json.dumps(result,indent=2))
        results.append(result)
    return results
