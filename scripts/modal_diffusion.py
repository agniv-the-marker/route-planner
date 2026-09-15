"""Dedicated GPU worker for the public SD 1.5 silhouette path."""
import modal

image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch>=2.1', 'diffusers>=0.25', 'transformers>=4.36', 'accelerate>=0.25', 'Pillow>=10')
    .add_local_file('src/__init__.py', '/app/src/__init__.py', copy=True)
    .add_local_file('src/diffusion.py', '/app/src/diffusion.py', copy=True).workdir('/app'))
app = modal.App('route-sculptor-diffusion')

@app.cls(image=image, gpu='L4', timeout=300, scaledown_window=300, max_containers=1,
         secrets=[modal.Secret.from_name('huggingface-secret')])
class DiffusionWorker:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import StableDiffusionPipeline
        MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        self.pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None).to('cuda')

    @modal.method()
    def generate(self, prompt, seed):
        import io, torch
        # Preserve this historical worker independently of the current website model.
        PROMPT_TEMPLATE = 'a single centered filled black silhouette of {prompt}, flat simple icon, pure white background, no internal detail'
        NEGATIVE_PROMPT = 'text, watermark, letters, border, frame, multiple objects, background, scenery, photograph, shading, gradient, outline, sketch, gray, color'
        STEPS, GUIDANCE = 30, 8.0
        image = self.pipe(PROMPT_TEMPLATE.format(prompt=prompt), negative_prompt=NEGATIVE_PROMPT,
            height=512, width=512, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
        out = io.BytesIO(); image.save(out, 'PNG')
        return {'image': out.getvalue(), 'remote_job_id': None}
