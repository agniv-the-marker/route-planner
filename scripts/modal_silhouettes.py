"""Pretrained Z-Image silhouette worker. No training or automatic inference at deploy."""
import modal

image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'diffusers==0.36.0', 'transformers==4.57.1',
                 'accelerate==1.12.0', 'Pillow==11.3.0', 'sentencepiece==0.2.1')
    .env({'HF_HOME': '/model-cache', 'HF_HUB_DISABLE_TELEMETRY': '1'})
    .add_local_python_source('src'))
app = modal.App('route-sculptor-outlines-v2')
cache = modal.Volume.from_name('route-sculptor-silhouette-models', create_if_missing=True)


@app.cls(image=image, gpu='L40S', cpu=(2, 2), memory=(32768, 32768),
         timeout=180, startup_timeout=600, scaledown_window=60,
         max_containers=1, retries=0, volumes={'/model-cache': cache})
class SilhouetteWorker:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import ZImagePipeline
        from src.diffusion import MODEL, REVISION
        self.pipe = ZImagePipeline.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.bfloat16).to('cuda')
        cache.commit()

    @modal.method()
    def generate(self, prompt, seed):
        import io
        import torch
        from src.diffusion import PROMPT_TEMPLATE, STEPS, GUIDANCE, IMAGE_SIZE
        image = self.pipe(prompt=PROMPT_TEMPLATE.format(prompt=prompt),
            height=IMAGE_SIZE, width=IMAGE_SIZE, num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator('cuda').manual_seed(seed)).images[0]
        output = io.BytesIO()
        image.save(output, 'PNG')
        return {'image': output.getvalue()}
