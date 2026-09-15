import json
import pytest
from scripts.outline_benchmark import fixtures
from src.outline_inference import generate_outline
from dataclasses import asdict


def test_repair_once_and_cache_only_success():
    responses = iter([('not JSON', 'length'), (json.dumps(asdict(fixtures()['fish'])), 'stop')])
    calls = []
    def complete(messages):
        calls.append(list(messages))
        return next(responses)
    cache = {}
    result = generate_outline('a fish', complete, cache)
    assert result['outline'] and len(calls) == 2 and len(result['attempts']) == 2
    assert calls[1][-1]['content'].startswith('Repair the geometry:')
    cached = generate_outline(' A FISH ', complete, cache)
    assert cached['cached'] and len(calls) == 2


def test_invalid_model_geometry_stops_after_one_repair():
    calls = []
    def complete(messages):
        calls.append(messages)
        return ('{"interpretation":"shape","points":[[0,0],[1,1],[0,1],[1,0],[0,0],[1,1],[0,1],[0,0]]}', 'stop')
    cache = {}
    result = generate_outline('something unusual', complete, cache)
    assert result['outline'] is None and result['failure_reason']
    assert len(calls) == 2 and not cache


def test_model_transport_failure_is_not_cached_or_repaired():
    def complete(messages):
        raise RuntimeError('worker unavailable')
    cache = {}
    with pytest.raises(RuntimeError, match='worker unavailable'):
        generate_outline('cat', complete, cache)
    assert not cache
