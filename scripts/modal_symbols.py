"""CPU clients use a separate, lazy L4 symbol interpreter."""
import modal
base_image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('vllm==0.10.2', 'shapely>=2,<3')
         .pip_install('transformers==4.55.4', 'huggingface-hub<1')
         .add_local_file('src/__init__.py', '/app/src/__init__.py', copy=True)
         .add_local_file('src/outlines.py', '/app/src/outlines.py', copy=True)
         .add_local_file('src/outline_inference.py', '/app/src/outline_inference.py', copy=True)
         .workdir('/app').env({'HF_HOME': '/models', 'PYTHONPATH': '/app'}))
models = modal.Volume.from_name('route-sculptor-outline-models', create_if_missing=True)


app = modal.App('route-sculptor-symbols')
image = (base_image.add_local_file('src/symbols.py','/app/src/symbols.py',copy=True)
         .add_local_file('src/assets/symbols.json','/app/src/assets/symbols.json',copy=True))

@app.cls(image=image,gpu='L4',max_containers=1,scaledown_window=300,timeout=600,volumes={'/models':models})
class SymbolWorker:
    @modal.enter()
    def load(self):
        import time
        from vllm import LLM
        from src.outlines import MODEL
        started=time.perf_counter()
        self.llm=LLM(model=MODEL,max_model_len=4096,enforce_eager=True,trust_remote_code=False)
        self.cold_seconds=time.perf_counter()-started
        self.cache={}

    @modal.method()
    def select(self,prompt):
        import json,time
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from src.symbols import selection_schema,selection_prompt,outline_from_selection
        started=time.perf_counter()
        key=' '.join(prompt.casefold().split())
        if key in self.cache:
            return {'selection':self.cache[key],'timings':{'cache_seconds':time.perf_counter()-started,'cold_load_seconds':0}}
        params=SamplingParams(temperature=0,seed=0,max_tokens=96,guided_decoding=GuidedDecodingParams(json=selection_schema()))
        output=self.llm.chat([{'role':'system','content':selection_prompt()},{'role':'user','content':prompt}],sampling_params=params,use_tqdm=False)[0].outputs[0]
        selection=json.loads(output.text)
        # Unknown descriptions may legitimately map to none; the CPU client retains a clear failure.
        if selection['symbol'] != 'none':
            outline_from_selection(selection)
            self.cache[key]=selection
            if len(self.cache)>128: del self.cache[next(iter(self.cache))]
        cold=self.cold_seconds
        self.cold_seconds=0
        return {'selection':selection,'timings':{'generation_seconds':time.perf_counter()-started,'cold_load_seconds':cold}}

@app.local_entrypoint()
def main(output_dir:str='outputs/symbol-benchmark-v1'):
    import json,time
    from pathlib import Path
    from dataclasses import asdict
    from scripts.outline_benchmark import DESCRIPTIONS,svg
    from src.symbols import outline_from_selection,VERSION
    from src.outlines import MODEL
    out=Path(output_dir)
    if (out/'results.json').exists(): raise ValueError('Choose a fresh output directory.')
    out.mkdir(parents=True,exist_ok=True)
    worker=SymbolWorker()
    records=[]
    for i,prompt in enumerate(DESCRIPTIONS):
        started=time.perf_counter()
        response=worker.select.remote(prompt)
        record={'id':i+1,'prompt':prompt,'wall_seconds':time.perf_counter()-started,**response['timings'],'selection':response['selection'],'attempts':[]}
        try:
            spec=outline_from_selection(response['selection'])
            record['outline']=asdict(spec)
            svg(spec.points,out/f'{i+1:02}.svg')
        except ValueError as exc:
            record.update(outline=None,failure_reason=str(exc))
        records.append(record)
        (out/'results.json').write_text(json.dumps({'model':MODEL,'version':VERSION,'records':records},indent=2))
        print(i+1,response['selection'],round(record['wall_seconds'],3),flush=True)
        if i == 0:
            started=time.perf_counter()
            cached=worker.select.remote(prompt)
            cached['wall_seconds']=time.perf_counter()-started
            (out/'cache-check.json').write_text(json.dumps(cached,indent=2))
