"""One capped L4 allocation, pinned baseline; 120 attempts, no repair or escalation."""
import modal
app = modal.App('route-sculptor-vector-baseline')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('vllm==0.10.2','transformers==4.55.4','huggingface-hub<1','shapely>=2,<3')
         .add_local_python_source('src', 'scripts'))

@app.function(image=image, gpu='L4', timeout=7100, max_containers=1)
def benchmark():
    import time
    from dataclasses import asdict
    from vllm import LLM, SamplingParams
    from src.vector_program import MODEL, REVISION
    from src.generative import VectorGenerator
    from scripts.generative_benchmark import PROMPTS
    start = time.perf_counter()
    llm = LLM(model=MODEL, revision=REVISION, max_model_len=4096,
              gpu_memory_utilization=.9, enforce_eager=True, trust_remote_code=False)
    cold = time.perf_counter()-start
    def complete(messages, **kwargs):
        return llm.chat(messages,sampling_params=SamplingParams(**kwargs),use_tqdm=False)[0].outputs[0].text
    generator = VectorGenerator(complete)
    for prompt in PROMPTS:
        if time.perf_counter()-start > 6800:
            break
        for record in generator.candidates(prompt):
            record.timings['cold_start_seconds'] = cold
            cold = 0
            yield asdict(record)

@app.local_entrypoint()
def main(output_dir: str = 'outputs/generative-v1'):
    import json
    from pathlib import Path
    from scripts.generative_benchmark import freeze, evaluate, review
    from src.street_search import StreetSearch
    output = freeze(output_dir)
    records = []
    try:
        # Persist all raw attempts before routing; CPU work cannot idle the GPU.
        for i,record in enumerate(benchmark.remote_gen(),1):
            records.append(record)
            (output/f'{i:03}-raw.json').write_text(json.dumps(record,indent=2))
    except Exception as exc:
        (output/'infrastructure-error.json').write_text(json.dumps({'error':str(exc)}))
    if records:
        search = StreetSearch.load_prepared()
        search.route_cache_dir = Path('route_cache/generative-v1')
        records = [evaluate(r,search,output,i) for i,r in enumerate(records,1)]
    review(output,records)
