"""Bounded outline benchmark. Explicitly run with MODAL_PROFILE=nyro-robotics."""
import modal

app = modal.App('route-sculptor-outlines')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('vllm==0.10.2', 'shapely>=2,<3')
         .pip_install('transformers==4.55.4', 'huggingface-hub<1')
         .add_local_file('src/__init__.py', '/app/src/__init__.py', copy=True)
         .add_local_file('src/outlines.py', '/app/src/outlines.py', copy=True)
         .add_local_file('src/outline_inference.py', '/app/src/outline_inference.py', copy=True)
         .workdir('/app').env({'HF_HOME': '/models', 'PYTHONPATH': '/app'}))
models = modal.Volume.from_name('route-sculptor-outline-models', create_if_missing=True)

@app.cls(image=image, gpu='L4', max_containers=1, scaledown_window=300,
         timeout=1200, volumes={'/models': models})
class OutlineWorker:
    @modal.enter()
    def load(self):
        import time
        from vllm import LLM
        from src.outlines import MODEL
        started = time.perf_counter()
        self.llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=.9,
                       enforce_eager=True, trust_remote_code=False)
        self.load_seconds = time.perf_counter() - started
        self.first = True
        self.cache = {}

    @modal.method()
    def generate(self, prompt):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from src.outlines import SCHEMA
        from src.outline_inference import generate_outline
        cold = self.load_seconds if self.first else 0
        self.first = False

        def complete(messages):
            params = SamplingParams(temperature=0, seed=0, max_tokens=768,
                                    guided_decoding=GuidedDecodingParams(json=SCHEMA))
            response = self.llm.chat(messages, sampling_params=params, use_tqdm=False)[0].outputs[0]
            return response.text, response.finish_reason

        return {**generate_outline(prompt, complete, self.cache), 'cold_load_seconds': cold}

@app.local_entrypoint()
def main(output_dir: str = 'outputs/outline-benchmark-v1'):
    import json
    import time
    from pathlib import Path
    from scripts.outline_benchmark import DESCRIPTIONS, svg
    from src.outlines import MODEL, VERSION
    output = Path(output_dir)
    if (output / 'results.json').exists():
        raise ValueError('Choose a fresh --output-dir to preserve previous evidence.')
    output.mkdir(parents=True, exist_ok=True)
    worker = OutlineWorker()
    records = []
    for i, prompt in enumerate(DESCRIPTIONS):
        started = time.perf_counter()
        result = worker.generate.remote(prompt)
        result.update(id=i+1, prompt=prompt, wall_seconds=time.perf_counter()-started)
        records.append(result)
        if result['outline']:
            svg(result['outline']['points'], output / f'{i+1:02}.svg')
        (output / 'results.json').write_text(json.dumps({'model': MODEL, 'version': VERSION, 'records': records}, indent=2))
        print(f"{i+1}/30: {'valid' if result['outline'] else 'invalid'}, {result['wall_seconds']:.2f}s", flush=True)
        if result['outline'] and not (output / 'cache-check.json').exists():
            cache_started = time.perf_counter()
            cached = worker.generate.remote(prompt)
            cached['wall_seconds'] = time.perf_counter() - cache_started
            (output / 'cache-check.json').write_text(json.dumps(cached, indent=2))
        invalid = sum(not r['outline'] for r in records)
        if len(DESCRIPTIONS) - invalid < 24:
            (output / 'STOPPED.json').write_text(json.dumps({
                'reason': '24/30 outline gate is mathematically unreachable.',
                'evaluated': len(records), 'invalid': invalid,
                'maximum_possible_recognizable': len(DESCRIPTIONS)-invalid}, indent=2))
            break
