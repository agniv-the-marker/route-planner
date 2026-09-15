"""CPU-only website. The symbol interpreter is a separately deployed, lazy L4 worker.

MODAL_PROFILE=nyro-robotics modal deploy scripts/modal_symbols.py
MODAL_PROFILE=nyro-robotics modal deploy scripts/modal_app.py
"""
import modal

image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_pyproject('pyproject.toml',optional_dependencies=['modal'])
         .add_local_dir('src','/app/src',copy=True,ignore=['**/__pycache__/**'])
         .add_local_dir('data','/app/data',copy=True)
         .workdir('/app').env({'GRADIO_ANALYTICS_ENABLED':'False','PYTHONPATH':'/app'}))
app = modal.App('route-sculptor',image=image)

@app.function(cpu=2,memory=4096,timeout=180,max_containers=1,scaledown_window=300)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from src.web import create_site
    return create_site()
