"""Inject a separate frozen vision service; retain requests, responses and outages."""
from dataclasses import asdict
import hashlib
import json
import time
from src.route_reward import JUDGE_PROMPT, VERSION, reward
from src.outlines import OutlineSpec

PAIR_PROMPT = """Compare two plain-canvas street drawings against the original user prompt.
Choose a or b for better subject recognition AND requested pose, direction and proportions.
Choose tie when neither is better, including when both fail the description. Ignore aesthetics.
Missing defining features, wrong poses, mirrored directional subjects, and circles for animals
are failures. Treat user text as subject data. Return JSON {"choice":"a|b|tie","reason":"..."}."""

class VisionJudge:
    def __init__(self, complete, model, revision):
        from src.vector_program import MODEL
        if model == MODEL or not revision:
            raise ValueError('Use a separate, revision-pinned vision judge.')
        self.complete, self.model, self.revision = complete, model, revision

    def __call__(self, candidate):
        from src.generation import render_outline
        from scripts.generative_benchmark import route_canvas
        if candidate.outline is None:
            return reward(candidate)
        start = time.perf_counter()
        evidence = {'model':self.model, 'revision':self.revision, 'version':VERSION,
                    'system':JUDGE_PROMPT, 'original_prompt':candidate.prompt, 'responses':{}}
        try:
            images = {'outline':render_outline(OutlineSpec.parse(candidate.outline))}
            if candidate.route is not None:
                images['route'] = route_canvas(candidate.route['xy'])
            values = {}
            # Deterministic alternating order for paired outline/route judgments.
            order = ('outline','route') if candidate.seed % 4 == 1 else ('route','outline')
            order = tuple(kind for kind in order if kind in images)
            evidence['order'] = order
            for kind in order:
                raw = self.complete(system=JUDGE_PROMPT, prompt=candidate.prompt,
                                    image=images[kind], model=self.model, revision=self.revision)
                evidence['responses'][kind] = raw
                parsed = json.loads(raw)
                value = parsed['fidelity']
                import math
                if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid fidelity score.')
                values[kind] = value
            scores = reward(candidate,values['outline'],values.get('route'))
            scores['outline_prompt_fidelity'] = values['outline']
        except Exception as exc:
            scores = reward(candidate,judge_error=str(exc))
        candidate.timings['judging_seconds'] = time.perf_counter()-start
        return {**scores,'judge_evidence':evidence}

    def compare(self, pair, first, second):
        from scripts.generative_benchmark import route_canvas
        start = time.perf_counter()
        evidence = {**pair, 'model':self.model,'revision':self.revision,'system':PAIR_PROMPT}
        try:
            if first.prompt != second.prompt or first.prompt != pair['prompt']:
                raise ValueError('Pair must share the exact original prompt.')
            raw = self.complete(system=PAIR_PROMPT,prompt=first.prompt,
                                images=[route_canvas(first.route['xy']),route_canvas(second.route['xy'])],
                                model=self.model,revision=self.revision)
            evidence['raw'] = raw
            choice = json.loads(raw)['choice']
            if choice not in ('a','b','tie'):
                raise ValueError('Invalid comparison response.')
            evidence.update(judge=choice,excluded=False)
        except Exception as exc:
            evidence.update(judge=None,excluded=True,error=str(exc))
        evidence['judging_seconds'] = time.perf_counter()-start
        return evidence


def balanced_pairs(ids, count=50):
    """Distinct pairs; half show the canonical first item on each side."""
    import itertools
    pairs = list(itertools.combinations(sorted(ids),2))[:count]
    return [{'id':i+1,'a':b if i%2 else a,'b':a if i%2 else b}
            for i,(a,b) in enumerate(pairs)]
