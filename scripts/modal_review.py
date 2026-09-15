"""Bounded CPU review preview: MODAL_PROFILE=nyro-robotics modal serve --timeout 3600 scripts/modal_review.py"""
import modal
import os

if modal.is_local():
    from src.experiment_journal import source_identity, digest
    release=digest(source_identity())[:16]
else:
    release=os.environ['REVIEW_RELEASE']
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('fastapi>=0.115,<1')
       .add_local_file('src/review_preview.py','/app/src/review_preview.py',copy=True)
       .add_local_dir('outputs/routing-first-v2','/app/artifacts',copy=True)
       .env({'PYTHONPATH':'/app','REVIEW_RELEASE':release}))
app=modal.App('route-sculptor-routing-review',image=image)
volume=modal.Volume.from_name('route-sculptor-routing-review',create_if_missing=True)

@app.function(cpu=(.25,.25),memory=(256,256),timeout=60,max_containers=1,scaledown_window=30,volumes={'/state':volume})
@modal.asgi_app()
def web():
    from src.review_preview import create_review_site
    return create_review_site('/app/artifacts','/state',volume.commit,release)
