"""Four independent attempts; failures and provenance are first-class data."""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import time
from src.vector_program import MODEL, REVISION, VERSION, SEEDS, SYSTEM, compile_program

@dataclass
class Candidate:
    prompt: str
    seed: int
    model: str = MODEL
    revision: str = REVISION
    version: str = VERSION
    raw: str | None = None
    program: object = None
    validation: str = 'pending'
    outline: object = None
    route: object = None
    scores: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    infrastructure_error: str | None = None

class VectorGenerator:
    def __init__(self, complete, judge=None, cache_dir=None):
        self.complete = complete
        self.judge = judge
        self.cache_dir = cache_dir
        self.last_candidates = []

    def candidates(self, prompt):
        self.last_candidates = []
        for seed in SEEDS:
            record = Candidate(prompt, seed)
            start = time.perf_counter()
            cache_path = None
            if self.cache_dir is not None:
                from pathlib import Path
                key = hashlib.sha256(json.dumps([MODEL,REVISION,SYSTEM,prompt,seed,.7,.95,1536]).encode()).hexdigest()
                legacy_key = hashlib.sha256(json.dumps([MODEL,REVISION,VERSION,SYSTEM,prompt,seed,.7,.95,1536]).encode()).hexdigest()
                cache_path = Path(self.cache_dir)/'generated-v1'/f'{key}.json'
                legacy_path = Path(self.cache_dir)/'generated-v1'/f'{legacy_key}.json'
                read_path = cache_path if cache_path.exists() else legacy_path
                if read_path.exists():
                    try:
                        payload = json.loads(read_path.read_text())
                        if payload['key'] != (key if read_path == cache_path else legacy_key):
                            raise ValueError('Cache provenance mismatch')
                        # Reuse immutable inference only; compile against today's validator.
                        record.raw = payload.get('raw', payload.get('candidate', {}).get('raw'))
                        record.timings = {'cache_seconds':time.perf_counter()-start}
                    except (OSError,ValueError,TypeError,KeyError):
                        pass
            try:
                if record.raw is None:
                    record.raw = self.complete([{'role':'system','content':SYSTEM},
                                            {'role':'user','content':prompt}],
                                           seed=seed, temperature=.7, top_p=.95, max_tokens=1536)
            except Exception as exc:
                record.infrastructure_error = str(exc)
            record.timings['generation_seconds'] = time.perf_counter()-start
            if record.raw is not None:
                start = time.perf_counter()
                try:
                    from src.vector_program import extract_program
                    record.program = extract_program(record.raw)
                    record.outline = asdict(compile_program(record.raw, prompt))
                    record.validation = 'valid'
                except (ValueError, TypeError, OverflowError) as exc:
                    record.validation = str(exc)
                record.timings['validation_seconds'] = time.perf_counter()-start
            if cache_path is not None and not record.infrastructure_error:
                import os, tempfile
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w',dir=cache_path.parent,delete=False) as tmp:
                    json.dump({'key':key,'raw':record.raw,'model':MODEL,'revision':REVISION,'seed':seed,'prompt':prompt},tmp)
                os.replace(tmp.name,cache_path)
            self.last_candidates.append(record)
            yield record

    def generate(self, prompt):
        from src.outlines import OutlineSpec
        records = list(self.candidates(prompt))
        valid = next((r for r in records if r.outline), None)
        if valid is None:
            raise ValueError('No valid generated silhouette; all four attempts retained.')
        return OutlineSpec.parse(valid.outline), {'candidates':[asdict(r) for r in records]}


def cache_key(record, graph_hash, judge_revision):
    """Evaluation identity excludes mutable timings, scores and cache-hit diagnostics."""
    from src.street_search import VERSION as ROUTER_VERSION
    from src.route_reward import VERSION as REWARD_VERSION
    identity = [record.prompt,record.seed,record.model,record.revision,record.version,
                record.raw,record.outline,record.route,record.scores.get('routing_config'),
                graph_hash,ROUTER_VERSION,REWARD_VERSION,judge_revision]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
