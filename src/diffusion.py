"""Versioned silhouette inference with bounded spending and durable call IDs."""
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from PIL import Image

from src.experiment_journal import Budget, Journal, atomic_json, digest, locked, now

MODEL = 'Tongyi-MAI/Z-Image-Turbo'
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'
WORKER_APP = 'route-sculptor-outlines-v2'
STEPS = 9
GUIDANCE = 0.0
IMAGE_SIZE = 1024
PROMPT_TEMPLATE = ('A crisp two-tone vector silhouette pictogram. Subject: {prompt}. '
    'Depict exactly this subject and its requested pose, with recognizable proportions and distinctive outer features. '
    'One complete solid jet-black cut-paper shape centered on a pure white square canvas. '
    'The subject occupies the central 70 percent of the canvas, with a wide empty white margin on every side. '
    'Keep the whole subject visible, including extremities. Use a clean, simple continuous outer boundary. '
    'Flat black fill throughout the subject; blank white outside. The image contains only the subject silhouette, '
    'without lettering, decorations, a frame, scenery, shadows, texture or a surrounding badge.')
NEGATIVE_PROMPT = 'text, watermark, letters, border, frame, multiple objects, background, scenery, photograph, shading, gradient, outline, sketch, gray, color'

def normalize_prompt(prompt):
    return ' '.join((prompt or '').casefold().split())

def candidate_seeds(request_id):
    return [int.from_bytes(hashlib.sha256(f'{request_id}:{i}'.encode()).digest()[:4], 'big') for i in range(4)]

@dataclass
class DiffusionCandidate:
    prompt: str
    seed: int
    image: Image.Image | None = None
    remote_job_id: str | None = None
    timings: dict = field(default_factory=dict)
    error: str | None = None
    cached: bool = False
    outline: object = None
    extraction: dict = field(default_factory=dict)
    extraction_error: str | None = None
    route: object = None
    scores: dict = field(default_factory=dict)
    infrastructure_error: str | None = None

class DiffusionGenerator:
    """Four independent silhouettes. ``remote`` is injectable for tests."""
    def __init__(self, remote=None, cache_dir='route_cache/silhouettes-v2',
                 journal_dir='experiments/silhouettes-v2/journal',
                 budget_path='experiments/budget.json', persist=None):
        self.remote, self.cache_dir = remote, Path(cache_dir)
        self.journal = Journal(journal_dir)
        self.budget = Budget(budget_path)
        self.persist = persist or (lambda: None)

    def _remote_stage(self, prompt, seed, request_id):
        identity = {'model': MODEL, 'revision': REVISION, 'prompt': normalize_prompt(prompt),
                    'seed': seed, 'steps': STEPS, 'guidance': GUIDANCE, 'request_id': request_id,
                    'template': PROMPT_TEMPLATE, 'image_size': IMAGE_SIZE}
        key = self.journal.pending(request_id, 'diffusion-inference', identity)
        path = self.journal.root / 'stages' / f'{key}.json'
        return key, path, identity

    def _save_remote_job(self, path, job_id, reservation):
        """Record the call handle before waiting; this is the durable retry boundary."""
        with locked(self.journal.root / '.lock'):
            row = json.loads(path.read_text()) if path.exists() else {}
            row.update(state='running', remote=True, remote_job_id=job_id,
                       reservation=reservation, started=row.get('started', now()))
            atomic_json(path, row)
        self.persist()

    def _finish_remote_stage(self, path, value):
        artifact = f'artifacts/{path.stem}-{digest(value)}.json'
        atomic_json(self.journal.root / artifact, value)
        with locked(self.journal.root / '.lock'):
            row = json.loads(path.read_text())
            row.update(state='completed', artifact=artifact, artifact_hash=digest(value), ended=now())
            atomic_json(path, row)

    def _interrupt_remote_stage(self, path, exc):
        with locked(self.journal.root / '.lock'):
            if not path.exists():
                return
            row = json.loads(path.read_text())
            row.update(state='interrupted', error=str(exc), ended=now())
            atomic_json(path, row)

    @staticmethod
    def _modal_client():
        import modal
        if modal.is_local():
            from modal.config import config
            return modal.Client.from_credentials(
                config.get('token_id', profile='nyro-robotics', use_env=False),
                config.get('token_secret', profile='nyro-robotics', use_env=False))
        return None

    @classmethod
    def _modal_call(cls, job_id, client):
        import modal
        return modal.FunctionCall.from_id(job_id, client=client)

    def _generate_remote(self, prompt, seed, request_id):
        key, path, identity = self._remote_stage(prompt, seed, request_id)
        row = json.loads(path.read_text())
        job_id = row.get('remote_job_id')
        reservation = row.get('reservation')
        if not job_id and row.get('state') in {'running', 'interrupted'}:
            raise RuntimeError('Reconcile interrupted Modal dispatch before retrying: ' + key)
        if not job_id:
            # A reservation is made only for actual Modal calls. Injected remotes are
            # deliberately free and remain the unit-test/local deployment seam.
            reservation = f'diffusion-{key[:24]}'
            if reservation not in self._budget_allocations():
                self.budget.reserve(reservation, 1.0, 'research',
                                    {'url': 'https://modal.com/pricing', 'checked': '2026-09-14',
                                     'l40s_gpu_second_usd': 0.000542,
                                     'cpu_core_second_usd': 0.0000131,
                                     'gib_second_usd': 0.00000222},
                                    {'timeout_seconds': 180, 'startup_timeout_seconds': 600,
                                     'scaledown_window_seconds': 60, 'max_containers': 1,
                                     'cpu_limit': 2, 'memory_limit_gib': 32,
                                     'compute_upper_usd': 0.537, 'overhead_allowance_usd': 0.463})
            import modal
            client = self._modal_client()
            worker = modal.Cls.from_name(WORKER_APP, 'SilhouetteWorker', client=client)()
            # Mark dispatch as running before spawn. If the process dies during spawn,
            # the next attempt must reconcile this uncertain dispatch explicitly.
            with locked(self.journal.root / '.lock'):
                row = json.loads(path.read_text())
                row.update(state='running', remote=True, reservation=reservation, started=row.get('started', now()))
                atomic_json(path, row)
            # Persist reservation and uncertain-dispatch marker before paid work.
            self.persist()
            call = worker.generate.spawn(prompt, seed)
            job_id = getattr(call, 'object_id', None) or getattr(call, 'id', None)
            if not job_id:
                raise RuntimeError('Modal spawn returned no durable job id')
            self._save_remote_job(path, job_id, reservation)
        try:
            response = self._modal_call(job_id, self._modal_client()).get() if row.get('remote_job_id') else call.get()
            if isinstance(response, Image.Image):
                response = {'image': response}
            response['remote_job_id'] = job_id
            self._finish_remote_stage(path, {'remote_job_id': job_id, 'status': 'completed'})
            return response
        except BaseException as exc:
            self._interrupt_remote_stage(path, exc)
            raise

    def _budget_allocations(self):
        if not self.budget.path.exists():
            return {}
        try:
            return json.loads(self.budget.path.read_text()).get('allocations', {})
        except (OSError, ValueError):
            return {}

    def _path(self, prompt, seed):
        key = hashlib.sha256(json.dumps([MODEL, REVISION, PROMPT_TEMPLATE, IMAGE_SIZE,
            normalize_prompt(prompt), seed, STEPS, GUIDANCE]).encode()).hexdigest()
        return self.cache_dir / f'{key}.png', self.cache_dir / f'{key}.json'

    def candidates(self, prompt, request_id=None):
        request_id = request_id or normalize_prompt(prompt)
        for seed in candidate_seeds(request_id):
            started = time.perf_counter(); png, meta = self._path(prompt, seed)
            candidate = DiffusionCandidate(prompt, seed)
            try:
                if png.exists() and meta.exists():
                    data = json.loads(meta.read_text())
                    if data.get('model') != MODEL or data.get('revision') != REVISION or data.get('seed') != seed:
                        raise ValueError('stale candidate artifact')
                    candidate.image = Image.open(png).convert('RGB').copy()
                    candidate.remote_job_id = data.get('remote_job_id')
                    candidate.cached = True
                else:
                    if self.remote is None:
                        response = self._generate_remote(prompt, seed, request_id)
                        # The worker response is normalized below.
                        if isinstance(response, Image.Image): response = {'image': response}
                    else:
                        response = self.remote(prompt, seed)
                    if isinstance(response, Image.Image): response = {'image': response}
                    payload = response['image']
                    candidate.image = Image.open(io.BytesIO(payload)).convert('RGB') if isinstance(payload, bytes) else payload.convert('RGB')
                    candidate.remote_job_id = response.get('remote_job_id')
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    candidate.image.save(png, 'PNG')
                    fd, name = tempfile.mkstemp(dir=self.cache_dir, suffix='.json'); os.close(fd)
                    Path(name).write_text(json.dumps({'model': MODEL, 'revision': REVISION, 'seed': seed,
                        'prompt': normalize_prompt(prompt), 'template': PROMPT_TEMPLATE,
                        'steps': STEPS, 'guidance': GUIDANCE, 'image_size': IMAGE_SIZE,
                        'remote_job_id': candidate.remote_job_id}))
                    os.replace(name, meta)
                    self.persist()
            except Exception as exc:
                candidate.error = str(exc)
            candidate.timings['generation_seconds'] = round(time.perf_counter() - started, 3)
            yield candidate
