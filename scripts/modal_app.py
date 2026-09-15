"""Public CPU website; inference uses the workspace's existing GPU worker.

Deploy from the repository root with MODAL_PROFILE=nyro-robotics.
No account token belongs in this image or the public website.
"""
import modal

state = modal.Volume.from_name('route-sculptor-web-state', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_pyproject('pyproject.toml', optional_dependencies=['modal'])
         .add_local_dir('src', '/app/src', copy=True, ignore=['**/__pycache__/**'])
         .workdir('/app')
         .env({'GRADIO_ANALYTICS_ENABLED': 'False', 'PYTHONPATH': '/app'}))
# Only the historical SF map and terrain are required by the public application.
for name in ('sf.graphml', 'sf.json', 'sf-elevation.npy', 'sf-terrain.json', 'sf-terrain.png'):
    image = image.add_local_file(f'data/{name}', f'/app/data/{name}', copy=True)
app = modal.App('route-sculptor', image=image)


@app.function(cpu=2, memory=4096, timeout=600, max_containers=1,
              scaledown_window=120, volumes={'/state': state})
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import os
    from src.diffusion import DiffusionGenerator
    from src.hosting import MonthlyBudget
    from src.generation import RouteService
    from src.web import create_site

    # Generated files persist; source geodata stays in /app.
    os.chdir('/state')
    generator = DiffusionGenerator(persist=state.commit)
    generator.budget = MonthlyBudget(cap_usd=100)
    site = create_site(RouteService(generator=generator))

    @site.middleware('http')
    async def embedding_policy(request, call_next):
        response = await call_next(request)
        # routesculptor.bike reaches this app through its own Cloudflare Worker rather
        # than an iframe, so nothing but the site itself needs to frame these pages.
        response.headers['Content-Security-Policy'] = "frame-ancestors 'self'"
        return response

    return site
