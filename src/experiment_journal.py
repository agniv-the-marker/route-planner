"""Content-addressed stage journal. A crashed remote call is never silently retried."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time

STATES = {'pending', 'running', 'completed', 'failed', 'interrupted', 'excluded'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            os.unlink(f.name)
            raise
    os.replace(f.name, path)

@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield

def source_identity():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0')
    files = {}
    for name in sorted(set(paths)):
        if name and (name.startswith(('src/', 'scripts/', 'configs/', 'tests/')) or name == 'pyproject.toml'):
            p = Path(name)
            files[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'deleted'
    return {'revision': revision, 'dirty_source_hash': digest(files), 'source_files': files}

class Journal:
    def __init__(self, root):
        self.root = Path(root)

    def raw(self, attempt, payload):
        path = self.root / 'raw' / f'{digest(attempt)}.json'
        with locked(self.root / '.lock'):
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError('Raw attempt is immutable; create a child attempt.')
            else:
                atomic_json(path, payload)
        return digest(payload)

    def pending(self, attempt, stage, identity):
        key = digest([attempt, stage, identity])
        path = self.root / 'stages' / f'{key}.json'
        with locked(self.root / '.lock'):
            if not path.exists():
                atomic_json(path, {'key':key,'attempt':attempt,'stage':stage,'identity':identity,'state':'pending','created':now()})
        return key

    def run(self, attempt, stage, identity, operation, remote=False, reservation=None):
        key = digest([attempt, stage, identity])
        path = self.root / 'stages' / f'{key}.json'
        with locked(self.root / '.lock'):
            previous = json.loads(path.read_text()) if path.exists() else {}
            if previous.get('state') in {'completed', 'excluded'} and previous.get('artifact'):
                artifact = self.root / previous['artifact']
                value = json.loads(artifact.read_text())
                if digest(value) != previous['artifact_hash']:
                    raise ValueError('Completed artifact hash mismatch')
                return value
            if previous.get('state') in {'running', 'interrupted'}:
                raise RuntimeError('Reconcile interrupted/running stage before retrying: ' + key)
            if remote and reservation is None:
                raise ValueError('Paid calls require a recorded reservation')
            if previous:
                atomic_json(self.root/'history'/f'{key}-{digest(previous)}.json',previous)
            row = {'key': key, 'attempt': attempt, 'stage': stage, 'identity': identity,
                   'state': 'running', 'started': now(), 'remote': remote,
                   'reservation': reservation, 'remote_job_id': None}
            atomic_json(path, row)
        started = time.perf_counter()
        try:
            value = operation()
            artifact = f'artifacts/{key}-{digest(value)}.json'
            atomic_json(self.root / artifact, value)
            row.update(state='excluded' if isinstance(value,dict) and value.get('status') == 'excluded' else 'completed', artifact=artifact, artifact_hash=digest(value), ended=now(),seconds=time.perf_counter()-started)
        except BaseException as exc:
            row.update(state='interrupted' if remote or not isinstance(exc, Exception) else 'failed',
                       error=str(exc), ended=now(),seconds=time.perf_counter()-started)
            atomic_json(path, row)
            raise
        atomic_json(path, row)
        return value

    def reconcile(self, key, state, evidence, remote_job_id=None):
        if state not in {'failed', 'interrupted', 'excluded'} or not evidence:
            raise ValueError('Reconciliation requires an explicit state and evidence')
        path = self.root / 'stages' / f'{key}.json'
        with locked(self.root / '.lock'):
            row = json.loads(path.read_text())
            if row['state'] == 'completed':
                raise ValueError('Completed stages are immutable')
            atomic_json(self.root/'history'/f'{key}-{digest(row)}.json',row)
            row.update(state=state, reconciliation=evidence, remote_job_id=remote_job_id, reconciled=now())
            atomic_json(path, row)

class Budget:
    """Reservations count at their full upper bound until reconciled to actual spend."""
    def __init__(self, path):
        self.path = Path(path)

    def reserve(self, name, usd, category, pricing, bound):
        if usd <= 0 or category not in {'research', 'preview'} or not pricing or not bound:
            raise ValueError('Reservation requires a positive bound, category and pricing evidence')
        with locked(self.path.with_suffix('.lock')):
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                'cap_usd': 100, 'research_cap_usd': 90, 'preview_reserve_usd': 10, 'allocations': {}}
            if name in data['allocations']:
                raise ValueError('Allocation already exists; reconcile it before creating another')
            rows = list(data['allocations'].values())
            cost = lambda r: r.get('actual_usd', r['reserved_usd'])
            # Caps live in the ledger, not in this code. A null cap means no ceiling:
            # reservations are still recorded, they just stop refusing new work.
            cap, research_cap = data.get('cap_usd', 100), data.get('research_cap_usd', 90)
            spent = sum(map(cost, rows))
            researched = sum(cost(r) for r in rows if r['category'] == 'research')
            if cap is not None and spent + usd > cap:
                raise ValueError(f'Experiment budget exhausted: ${spent:.2f} of ${cap:.2f} reserved')
            if research_cap is not None and category == 'research' and researched + usd > research_cap:
                raise ValueError(f'Research budget exhausted: ${researched:.2f} of ${research_cap:.2f} reserved')
            data['allocations'][name] = dict(reserved_usd=usd, category=category, pricing=pricing, bound=bound, created=now())
            atomic_json(self.path, data)
        return name

    def settle(self, name, actual_usd, evidence):
        if actual_usd < 0 or not evidence:
            raise ValueError('Measured spend requires nonnegative cost and evidence')
        with locked(self.path.with_suffix('.lock')):
            data = json.loads(self.path.read_text())
            row = data['allocations'][name]
            row.update(actual_usd=actual_usd, reconciliation=evidence, settled=now())
            atomic_json(self.path,data)
