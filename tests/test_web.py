"""Event wiring for the text page. No diffusion, no street search."""

import types

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from src.generation import GenerationResult
from src.web import create_app


class StubService:
    """Stands in for RouteService: records calls, returns shaped results."""

    def __init__(self):
        self.generator = types.SimpleNamespace(candidates=lambda prompt: [])
        self.fit_calls = []

    def generate_outlines(self, prompt):
        result = GenerationResult(prompt=prompt, interpretation=prompt)
        result.diagnostics = {'valid_candidates': [0, 1],
                              'outline_previews': [(Image.new('RGB', (8, 8)), 'Candidate 1'),
                                                   (Image.new('RGB', (8, 8)), 'Candidate 2')],
                              'candidates': [{'outline': {}}, {'outline': {}}]}
        result.status = '2 candidates. Select one, then fit it to streets.'
        return result

    def fit_selected(self, result, index):
        self.fit_calls.append(index)
        if index is None:
            raise ValueError('Choose a valid silhouette before fitting a route.')
        result.selected_candidate = int(index)
        angle = np.linspace(0, 2 * np.pi, 16)
        result.route_xy = np.stack([551000 + 900 * np.cos(angle), 4181000 + 900 * np.sin(angle)], axis=1)
        result.gpx = '<gpx></gpx>'
        result.status = f'fitted {index}'
        return result


def named_fns(app):
    return {name: fn for name, fn in
            ((getattr(fn, 'api_name', None), fn) for fn in app.fns.values())}


@pytest.fixture
def app():
    service = StubService()
    built = create_app(service)
    built.service = service
    return built


def test_fit_button_starts_hidden(app):
    fit = next(block for block in app.blocks.values() if getattr(block, 'elem_id', None) == 'fit-button')
    assert fit.visible is False
    assert fit.interactive is False


def test_fit_visibility_has_its_own_single_output_event(app):
    """Gradio 6 drops a component's `visible` update when several are batched into one
    event, so the fit button's visibility must not ride along with the others."""
    fit = next(block for block in app.blocks.values() if getattr(block, 'elem_id', None) == 'fit-button')
    dedicated = [fn for fn in app.fns.values() if list(fn.outputs) == [fit]]
    assert dedicated, 'fit button visibility needs a dependency that outputs only the button'
    assert dedicated[0].fn(None)['visible'] is False
    result = GenerationResult()
    result.diagnostics = {'valid_candidates': [0]}
    assert dedicated[0].fn(result)['visible'] is True


def test_batched_generate_updates_never_touch_fit_visibility(app):
    fit = next(block for block in app.blocks.values() if getattr(block, 'elem_id', None) == 'fit-button')
    generate = named_fns(app)['generate']
    index = list(generate.outputs).index(fit)
    updates = list(generate.fn('a heart'))[-1]
    assert 'visible' not in updates[index]
    # Enabled because the first candidate is preselected; visibility is a separate event.
    assert updates[index]['interactive'] is True


def test_fit_runs_the_search_for_the_selected_candidate(app):
    fit_fn = named_fns(app)['fit_selected']
    generate = named_fns(app)['generate']
    result = list(generate.fn('a heart'))[-1][-1]
    fitted = list(fit_fn.fn(result, 1))[-1]
    assert app.service.fit_calls == [1]
    assert fitted[-1].status == 'fitted 1'
    assert 'class="ride"' in fitted[1]


def test_fit_without_a_selection_reports_instead_of_failing(app):
    fit_fn = named_fns(app)['fit_selected']
    generate = named_fns(app)['generate']
    result = list(generate.fn('a heart'))[-1][-1]
    fitted = list(fit_fn.fn(result, None))[-1]
    assert 'Choose a valid silhouette' in fitted[3]


def test_loading_frame_keeps_the_existing_map(app):
    """The street map is ~800 KB; re-sending it just to blank it stalls the page."""
    generate = named_fns(app)['generate']
    frames = list(generate.fn('a heart'))
    assert isinstance(frames[0][1], type(gr.skip()))


def test_first_candidate_is_preselected_so_fit_works_on_the_first_click(app):
    """A visible fit button that does nothing until a thumbnail is clicked reads as broken."""
    fit = next(block for block in app.blocks.values() if getattr(block, 'elem_id', None) == 'fit-button')
    gallery = next(block for block in app.blocks.values() if getattr(block, 'elem_id', None) == 'outline-choices')
    generate = named_fns(app)['generate']
    updates = list(generate.fn('a heart'))[-1]
    assert updates[list(generate.outputs).index(gallery)]['selected_index'] == 0
    assert updates[list(generate.outputs).index(fit)]['interactive'] is True


def test_the_preselection_reset_matches_the_gallery(app):
    state = next(fn for fn in app.fns.values()
                 if len(fn.outputs) == 1 and type(fn.outputs[0]).__name__ == 'State'
                 and fn.fn.__code__.co_argcount == 1)
    result = GenerationResult()
    result.diagnostics = {'valid_candidates': [0, 1]}
    assert state.fn(result) == 0
    assert state.fn(GenerationResult()) is None
