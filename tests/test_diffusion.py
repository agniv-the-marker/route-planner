import io
import json
import sys
import types

from PIL import Image, ImageDraw

from src.diffusion import DiffusionGenerator
from src.generation import RouteService
from src.generation import GenerationResult


def _png():
    image = Image.new('RGB', (128, 128), 'white')
    ImageDraw.Draw(image).ellipse((28, 18, 100, 110), fill='black')
    output = io.BytesIO()
    image.save(output, 'PNG')
    return output.getvalue()


class EmptySearch:
    def search(self, spec):
        from src.street_search import RouteSearchResult
        return RouteSearchResult(spec)


def test_raster_candidate_is_extracted_before_search(tmp_path):
    generator = DiffusionGenerator(lambda prompt, seed: {'image': _png()}, tmp_path)
    result = RouteService(generator, EmptySearch()).generate('circle')

    assert result.outline is None
    assert result.gpx is None
    assert len(result.diagnostics['candidates']) == 4
    assert all(candidate['extraction_error'] is None for candidate in result.diagnostics['candidates'])
    assert all(candidate['outline'] is not None for candidate in result.diagnostics['candidates'])


def test_interim_result_outputs_handles_unplaced_outline(monkeypatch):
    """The streamed generation result has no map placement until a route wins."""
    import src.web as web

    monkeypatch.setattr(web, 'markup', lambda *args, **kwargs: '<map/>')
    result = GenerationResult(image=Image.new('RGB', (8, 8)), status='Searching…')
    values = list(web.result_outputs(result))
    assert values and values[0][1] == '<map/>'
    assert values[0][2] is None


def test_worker_failure_is_reported_as_generation_failure(tmp_path):
    def fail(prompt, seed):
        raise RuntimeError('worker unavailable')

    result = RouteService(DiffusionGenerator(fail, tmp_path), EmptySearch()).generate('circle')

    assert result.gpx is None
    assert result.status == 'Text generation failed: worker unavailable'
    assert len(result.diagnostics['candidates']) == 4


def test_modal_dispatch_is_reconciled_without_spawning_again(tmp_path, monkeypatch):
    calls = []

    class Call:
        object_id = 'fc-reconcile-1'
        attempts = 0

        def get(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError('client disconnected while waiting')
            return {'image': _png()}

    call = Call()

    class Function:
        @staticmethod
        def from_id(job_id, **kwargs):
            assert job_id == call.object_id
            return call

    class Generate:
        def spawn(self, prompt, seed):
            calls.append((prompt, seed))
            return call

    modal = types.SimpleNamespace(
        is_local=lambda: False,
        Cls=types.SimpleNamespace(from_name=lambda *args, **kwargs: (lambda: types.SimpleNamespace(generate=Generate()))),
        FunctionCall=Function,
    )
    monkeypatch.setitem(sys.modules, 'modal', modal)
    cache = tmp_path / 'cache'
    journal = tmp_path / 'journal'
    budget = tmp_path / 'budget.json'
    first = DiffusionGenerator(cache_dir=cache, journal_dir=journal, budget_path=budget)
    candidate = next(first.candidates('circle', request_id='reconcile'))
    assert candidate.error == 'client disconnected while waiting'
    assert len(calls) == 1
    second = DiffusionGenerator(cache_dir=cache, journal_dir=journal, budget_path=budget)
    candidate = next(second.candidates('circle', request_id='reconcile'))
    assert candidate.error is None
    assert candidate.remote_job_id == call.object_id
    assert len(calls) == 1
    assert len(json.loads(budget.read_text())['allocations']) == 1


def test_persisted_cache_is_complete_before_yield(tmp_path):
    snapshots = []
    def persist():
        metadata = list(tmp_path.glob('*.json'))
        assert all(path.with_suffix('.png').exists() for path in metadata)
        snapshots.append(len(metadata))
    generator = DiffusionGenerator(lambda prompt, seed: {'image': _png()},
                                   tmp_path, persist=persist)
    assert all(not c.error for c in generator.candidates('heart'))
    assert snapshots == [1, 2, 3, 4]
    assert all(c.cached for c in generator.candidates('heart'))
    assert snapshots == [1, 2, 3, 4]


def test_failed_persistence_prevents_paid_dispatch(tmp_path, monkeypatch):
    calls = []
    class Generate:
        def spawn(self, *args):
            calls.append(args)
            raise AssertionError('Must persist before dispatch')
    monkeypatch.setitem(sys.modules, 'modal', types.SimpleNamespace(
        is_local=lambda: False,
        Cls=types.SimpleNamespace(from_name=lambda *a, **kw: lambda: types.SimpleNamespace(generate=Generate()))))
    def persist():
        row = json.loads(next((tmp_path / 'journal/stages').glob('*.json')).read_text())
        assert row['state'] == 'running' and row['reservation']
        assert json.loads((tmp_path / 'budget.json').read_text())['allocations']
        raise RuntimeError('Storage unavailable')
    generator = DiffusionGenerator(cache_dir=tmp_path / 'cache',
        journal_dir=tmp_path / 'journal', budget_path=tmp_path / 'budget.json', persist=persist)
    result = next(generator.candidates('fish'))
    assert result.error == 'Storage unavailable'
    assert calls == []
