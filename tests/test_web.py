"""Event wiring for the text page. No diffusion, no street search."""

import types

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from src.generation import GenerationResult
from src.map_view import ASSETS
from src.site_shell import header
from src.web import boot_document, create_app, link_preview


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


def test_the_alternative_route_clears_its_busy_button_when_the_map_is_unchanged(app):
    """ui.js marks the clicked map button busy; something has to release it."""
    another = next(block for block in app.blocks.values()
                   if getattr(block, 'elem_id', None) == 'another-button')
    alternative = next(fn for fn in app.fns.values() if another._id in fn.targets[0])
    release = next(fn for fn in app.fns.values() if fn.js and 'finishAction' in fn.js)
    assert release.trigger_after == alternative._id
    assert "finishAction('another')" in release.js


def test_the_map_button_and_its_proxy_agree_on_the_action_name():
    from src.map_view import markup
    assert 'data-main-action="another"' in markup(another=True)
    assert 'data-main-action="another"' not in markup()
    ui = (ASSETS / 'ui.js').read_text()
    assert 'RouteSculptorBusy.mark(action' in ui
    assert "getElementById('another-button')" in ui


def test_the_served_page_carries_the_stylesheet_and_masthead_before_gradio_boots():
    """Gradio applies `css=`/`head=` from its client config, which is exactly the window
    its full-page 'Loading…' overlay covers. Both must be in the served document."""
    document = boot_document('<html><head><title>x</title></head>'
                             '<body><gradio-app></gradio-app></body></html>')
    head, body = document.split('</head>')
    assert '[data-testid="status-tracker"]{display:none !important;}' in head
    assert 'gradio-container' in head and '#masthead' in head  # the whole stylesheet
    assert body.index('id="boot-shell"') < body.index('<gradio-app')
    assert '#masthead-block #masthead' in body  # the hand-over condition
    assert header('text') in body


def test_the_boot_shell_is_only_added_once():
    twice = boot_document(boot_document(
        '<html><head></head><body><gradio-app></gradio-app></body></html>'))
    assert twice.count('id="boot-shell"') == 2 and twice.count('<gradio-app') == 1


def test_the_link_preview_replaces_gradios_own_card():
    """Scrapers take the first og:image they find, and Gradio's template ships one
    pointing at a Gradio banner."""
    served = ('<html><head>'
              '<meta\n\t\tproperty="og:image"\n\t\tcontent="https://raw.githubusercontent.com/x.jpg"\n\t/>'
              '<meta property="og:title" content="Gradio"/>'
              '<meta name="twitter:image" content="https://raw.githubusercontent.com/x.jpg"/>'
              '<meta charset="utf-8"/>'
              '</head><body><gradio-app></gradio-app></body></html>')
    document = boot_document(served, 'https://example.test/')
    assert 'raw.githubusercontent' not in document
    assert document.count('property="og:image"') == 1
    assert document.count('name="twitter:card"') == 1
    assert '<meta charset="utf-8"/>' in document  # only the card tags are stripped
    assert 'content="https://example.test/drawing-assets/preview.jpg"' in document


def test_the_preview_is_absolute_https_and_the_declared_size():
    from PIL import Image
    card = link_preview('http://route-sculptor.example/')
    assert 'content="https://route-sculptor.example/drawing-assets/preview.jpg"' in card
    assert 'content="summary_large_image"' in card
    # A localhost run stays on http, so a developer's own card still resolves.
    assert 'content="http://127.0.0.1:7860/drawing-assets/preview.jpg"' in link_preview('http://127.0.0.1:7860/')
    width = next(p for p in card.split('<meta ') if 'og:image:width' in p)
    height = next(p for p in card.split('<meta ') if 'og:image:height' in p)
    with Image.open(ASSETS / 'preview.jpg') as image:
        assert image.size == (1200, 630), 'the declared card size must match the file'
        assert f'content="{image.width}"' in width and f'content="{image.height}"' in height
