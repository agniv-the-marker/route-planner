"""Prompt-only app with an optional diagnostic view and development controls."""

import logging
import tempfile
import threading
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.generation import RouteService
from src.diffusion import MODEL as DIFFUSION_MODEL
from src.map_view import ASSETS, markup
from src.geo import SF
from src.graph import BikeRouter
from src.site_shell import header, footer

CSS = (ASSETS / "site.css").read_text()
THEME = gr.themes.Base(primary_hue="slate", neutral_hue="stone", radius_size="none")


def loading_status(message):
    return f'<span class="loading-message">{message}</span><span class="loading-bike" aria-hidden="true">🚲</span>'


def result_outputs(result):
    times = [[t['stage'],t['seconds'],t['status']] for t in result.timings]
    times.append(['Total (excludes queue)',result.total_seconds,'done' if result.gpx else 'pending or failed'])
    details_diagnostics = {k: v for k, v in result.diagnostics.items() if k != 'outline_previews'}
    details = {'pipeline':'z-image-raster-outline-street-search-v2','model':DIFFUSION_MODEL,'interpretation':result.interpretation,**details_diagnostics}
    common = (result.image,markup(result.route_xy,result.outline,frame=SF,
                                 graph=getattr(result.search_result,'graph',None),
                                 label='san francisco', gpx=result.gpx,
                                 another=bool(result.search_result and len(result.search_result.routes) > 1)))
    gallery = result.diagnostics.get('outline_previews', result.debug_images)
    tail = (result.status,gallery,times,details,result)
    if result.gpx:
        # Gradio copies the yielded file before this generator resumes or closes.
        with tempfile.TemporaryDirectory(prefix='route-sculptor-') as directory:
            path = Path(directory)/'route.gpx'
            path.write_text(result.gpx)
            yield (*common,str(path),*tail)
    else:
        yield (*common,None,*tail)


def generate_outputs(service,prompt):
    # Keep injected legacy/vector generators usable for scripts and API callers while
    # diffusion-backed UI requests use the explicit human outline gate below.
    if not hasattr(service.generator, 'candidates'):
        yield (None,gr.skip(),None,loading_status('Interpreting your description…'),[],[],{},None)
        for result in service.stream(prompt):
            yield from result_outputs(result)
        return
    yield (None,gr.skip(),None,loading_status('Generating silhouettes…'),[],[],{},None)
    result = service.generate_outlines(prompt)
    yield from result_outputs(result)


def create_app(service=None):
    service = service if service is not None else RouteService()
    with gr.Blocks(title="Route Sculptor", delete_cache=(300, 3600), analytics_enabled=False) as app:
        gr.HTML(header('text') + '<div id="popup-backdrop" hidden></div>',
                elem_id="masthead-block", apply_default_css=False)
        with gr.Row(elem_id="prompt-bar"):
            prompt = gr.Textbox(label="Describe a shape", show_label=False, placeholder="A heart, a fish, a butterfly…", max_length=200,
                                lines=1, scale=4, min_width=180, elem_id="prompt-box")
            button = gr.Button("generate silhouettes\u00a0↗", variant="primary", scale=1,
                               min_width=120, elem_id="generate-button")
        status = gr.Markdown("", elem_id="status")
        outline_choices = gr.Gallery(label="choose the outline to fit", columns=4, height=220,
                                     format="png", visible=False, elem_id="outline-choices")
        fit = gr.Button("fit selected outline ↗", variant="primary", visible=False, interactive=False, elem_id="fit-button")
        route_map = gr.HTML(value=markup(), js_on_load=(ASSETS / "map.js").read_text(),
                            elem_id="route-map", apply_default_css=False)
        last_result = gr.State(value=None, time_to_live=3600)
        selected_candidate = gr.State(value=None, time_to_live=3600)
        with gr.Row(elem_id="results"):
            generated = gr.Image(label="the shape", interactive=False, format="png", height=160,
                                 visible=False, elem_id="shape-preview")
            download = gr.DownloadButton(label="download GPX ↗", size="sm", visible=False)
            another = gr.Button("another route ↻", size="sm", visible=False, elem_id="another-button")
        with gr.Column(elem_id="tools"):
            with gr.Column(elem_id="debug-popup", elem_classes=["popup"]):
                gr.HTML('<div class="popup-header"><h2 id="debug-title">Generation trace</h2>'
                        '<button type="button" data-close aria-label="Close debug">×</button></div>')
                timing = gr.Dataframe(headers=["stage", "seconds", "status"], datatype=["str", "number", "str"],
                                      interactive=False, label="measured timings", value=[])
                details = gr.JSON(label="reproduce this run")
            with gr.Column(elem_id="dev-popup", elem_classes=["popup"]):
                gr.HTML('<div class="popup-header"><h2 id="dev-title">Search & appearance</h2>'
                        '<button type="button" data-close aria-label="Close dev">×</button></div>')
                gr.Markdown("Search tries placements across SF for up to eight seconds. "
                            "Debug shows the chosen angle, candidate placements, scores and cache timings. "
                            "Appearance changes are local to this browser tab.")
                with gr.Row(elem_id="map-controls"):
                    streets = gr.Checkbox(value=True, label="show SF bicycle streets")
                    opacity = gr.Slider(0, 1, value=.43, step=.01, label="street opacity")
                    terrain = gr.Slider(0, 1, value=1, step=.05, label="terrain opacity")
                    outline = gr.Checkbox(value=False, label="show placed outline")
                with gr.Row():
                    palette = gr.Radio(["paper", "warm", "blue"], value="paper", label="page palette")
                    typography = gr.Radio(["serif", "sans"], value="serif", label="headline type")
                    color = gr.ColorPicker(value="#637baa", label="route & button color")
                reset = gr.Button("reset dev settings", size="sm")
        gr.HTML(footer(), elem_id="credits-block", apply_default_css=False)

        def present(values):
            result = values[7]
            alternatives = len(result.search_result.routes) if result and result.search_result else 0
            valid = bool(result and result.diagnostics.get('valid_candidates'))
            # Preselect the first candidate: a visible "fit selected outline" button that
            # does nothing until you also click a thumbnail just reads as a dead button.
            chosen = result.selected_candidate if result and result.selected_candidate is not None else (0 if valid else None)
            return (gr.update(value=None,visible=False),values[1],
                    gr.update(value=values[2],visible=values[2] is not None),
                    "" if "loading-message" in values[3] else values[3],
                    gr.update(value=values[4],visible=valid, selected_index=chosen),
                    gr.update(interactive=chosen is not None),values[5],values[6],
                    gr.update(visible=alternatives > 1),result)

        def show_fit(result):
            return gr.update(visible=bool(result and result.diagnostics.get('valid_candidates')))

        def generate(prompt):
            for values in generate_outputs(service,prompt):
                yield present(values)

        def remember_candidate(evt: gr.SelectData):
            index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            return int(index), gr.update(interactive=True)

        def fit_selected(result, index):
            if result is None:
                return
            try:
                fitted = service.fit_selected(result, index)
            except (ValueError, TypeError, RuntimeError) as exc:
                fitted = result
                fitted.status = str(exc)
            yield from (present(values) for values in result_outputs(fitted))

        def alternative(result):
            if result is None:
                return
            result = service.select_route(result,result.alternative_index+1)
            for values in result_outputs(result):
                yield present(values)

        outputs = [generated,route_map,download,status,outline_choices,fit,timing,details,another,last_result]
        generation_event = gr.on(triggers=[button.click,prompt.submit],fn=generate,inputs=[prompt],outputs=outputs,
              concurrency_limit=1,concurrency_id='generation',trigger_mode='once',api_name='generate',
              show_progress='hidden', js="(...args) => { window.RouteSculptorBusy.start('generate-button', 'generating silhouettes'); return args; }")
        generation_event.then(fn=None, js="() => window.RouteSculptorBusy.finish('generate-button')", queue=False)
        # Matches the gallery's preselected thumbnail, so the first fit click works.
        generation_event.then(fn=lambda result: 0 if result and result.diagnostics.get('valid_candidates') else None,
                              inputs=[last_result], outputs=[selected_candidate], queue=False)
        # Gradio 6 drops a component's `visible` update when several are batched into one
        # event, which left the fit button permanently hidden. Show it on its own.
        generation_event.then(fn=show_fit, inputs=[last_result], outputs=[fit], queue=False)

        outline_choices.select(fn=remember_candidate, outputs=[selected_candidate, fit], queue=False, show_progress='hidden')
        fitting_event = fit.click(fn=fit_selected, inputs=[last_result, selected_candidate], outputs=outputs,
                  concurrency_limit=1, concurrency_id='generation', api_name='fit_selected', show_progress='hidden',
                  js="(...args) => { window.RouteSculptorBusy.start('fit-button', 'fitting route'); return args; }")
        fitting_event.then(fn=None, js="() => window.RouteSculptorBusy.finish('fit-button')", queue=False)
        fitting_event.then(fn=show_fit, inputs=[last_result], outputs=[fit], queue=False)
        another.click(fn=alternative,inputs=[last_result],outputs=outputs,
                      concurrency_limit=1,concurrency_id='generation',api_name=False, show_progress='hidden')
        appearance = [streets, opacity, terrain, outline, palette, typography, color]
        gr.on(triggers=[c.change for c in appearance], fn=None, inputs=appearance, queue=False,
              js="""(streets, opacity, terrain, outline, palette, type, color) => {
                const style = document.documentElement.style;
                style.setProperty('--street-display', streets ? 'block' : 'none');
                style.setProperty('--street-opacity', opacity);
                style.setProperty('--terrain-opacity', terrain);
                style.setProperty('--outline-display', outline ? 'block' : 'none');
                style.setProperty('--page-paper', {paper:'#fafafa',warm:'#f8f3ec',blue:'#f1f4f8'}[palette]);
                style.setProperty('--map-paper', {paper:'#f1f2ec',warm:'#eee9de',blue:'#e9eff1'}[palette]);
                style.setProperty('--title-font', type === 'serif' ? "'EB Garamond', Georgia, serif" : 'Arial, sans-serif');
                style.setProperty('--route-color', color);
              }""")
        reset.click(fn=lambda: (True,.43,1,False,"paper","serif","#637baa"),
                    outputs=appearance,queue=False)
        app.load(fn=None, js=(ASSETS / "ui.js").read_text())
    app.queue(max_size=4, default_concurrency_limit=1)
    return app


def create_site(service=None):
    site = FastAPI()
    from src.drawing_service import DrawingService
    # Keep both interactive workflows on the same SF street map.
    drawing_service = DrawingService(frame=SF)
    # Building the stroke router takes ~23 s; paying that inside the first request put
    # a cold "find a ride" past the browser's abort timeout.
    threading.Thread(target=lambda: drawing_service.router, daemon=True).start()
    site.mount('/drawing-assets', StaticFiles(directory=ASSETS), name='drawing-assets')

    @site.get('/favicon.ico', include_in_schema=False)
    def favicon():
        return FileResponse(ASSETS / 'bike-mark.webp', media_type='image/webp')

    @site.get('/draw', response_class=HTMLResponse)
    def draw():
        return (ASSETS / 'drawing.html').read_text().replace(
            '{{HEADER}}', header('draw')).replace('{{FOOTER}}', footer()).replace(
            '{{MAP}}', markup(frame=SF,
                              label='san francisco', terrain=True))

    @site.post('/api/drawing/fit')
    def fit_drawing(payload: dict):
        try:
            return drawing_service.fit(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @site.get('/health')
    def health():
        return {'status': 'ok', 'drawing_modes': ['canvas'], 'drawing_region': 'sf',
                'drawing_inference_enabled': False}

    @site.get("/about", response_class=HTMLResponse)
    def about():
        return (ASSETS / "about.html").read_text().replace("{{HEADER}}", header("about")).replace("{{FOOTER}}", footer())

    return gr.mount_gradio_app(site, create_app(service), path="/", css=CSS, theme=THEME,
                              show_error=False, favicon_path=str(ASSETS / 'bike-mark.webp'),
                              head='<script src="/drawing-assets/busy.js"></script><script src="/drawing-assets/playback.js"></script>')


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_site(), host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
