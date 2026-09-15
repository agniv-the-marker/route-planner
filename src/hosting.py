"""Persistent monthly inference allowance for the public website."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.experiment_journal import Budget, atomic_json, locked


class MonthlyBudget:
    """Each month has an independent ledger; previous allocations are retained."""
    def __init__(self, root='experiments/monthly', cap_usd=100, clock=None):
        self.root = Path(root)
        self.cap_usd = cap_usd
        self.clock = clock or (lambda: datetime.now(ZoneInfo('America/Los_Angeles')))

    @property
    def path(self):
        return self.root / (self.clock().strftime('%Y-%m') + '.json')

    def reserve(self, name, usd, category, pricing, bound):
        path = self.path
        with locked(path.with_suffix('.init.lock')):
            if not path.exists():
                atomic_json(path, {'cap_usd': self.cap_usd,
                                   'research_cap_usd': self.cap_usd,
                                   'preview_reserve_usd': 0, 'allocations': {},
                                   'timezone': 'America/Los_Angeles'})
        return Budget(path).reserve(name, usd, category, pricing, bound)
