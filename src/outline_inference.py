"""One geometry repair at most, with an injectable JSON completion function."""
import time
from dataclasses import asdict
from src.outlines import OutlineSpec, SYSTEM, prompt_key


def generate_outline(prompt, complete, cache):
    started = time.perf_counter()
    key = prompt_key(prompt)
    if key in cache:
        return {**cache[key], 'cached': True, 'cache_seconds': time.perf_counter()-started}
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    attempts = []
    for _ in range(2):
        raw, finish_reason = complete(messages)
        try:
            spec = OutlineSpec.parse(raw)
            result = {'outline': asdict(spec), 'attempts': attempts + [{'raw': raw}],
                      'generation_seconds': time.perf_counter()-started, 'cached': False}
            cache[key] = result
            if len(cache) > 128:
                del cache[next(iter(cache))]
            return result
        except (ValueError, TypeError) as exc:
            attempts.append({'raw': raw, 'error': str(exc), 'finish_reason': finish_reason})
            messages += [{'role': 'assistant', 'content': raw},
                         {'role': 'user', 'content': f'Repair the geometry: {exc}. Return the complete corrected JSON.'}]
    return {'outline': None, 'attempts': attempts, 'generation_seconds': time.perf_counter()-started,
            'cached': False, 'failure_reason': 'Invalid outline after one repair.'}
