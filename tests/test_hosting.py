from datetime import datetime
import json

import pytest
from src.hosting import MonthlyBudget


def test_monthly_budget_preserves_previous_ledger_and_resets(tmp_path):
    current = [datetime(2026, 9, 30, 23, 59)]
    budget = MonthlyBudget(tmp_path, cap_usd=100, clock=lambda: current[0])
    budget.reserve('first', 100, 'research', {'source': 'test'}, {'seconds': 1})
    with pytest.raises(ValueError, match='exhausted'):
        budget.reserve('extra', 1, 'research', {'source': 'test'}, {'seconds': 1})
    current[0] = datetime(2026, 10, 1)
    budget.reserve('next', 1, 'research', {'source': 'test'}, {'seconds': 1})
    assert json.loads((tmp_path / '2026-09.json').read_text())['allocations']['first']['reserved_usd'] == 100
    assert list(json.loads(budget.path.read_text())['allocations']) == ['next']
    # A new container must see the same month's reservations.
    other = MonthlyBudget(tmp_path, cap_usd=100, clock=lambda: current[0])
    with pytest.raises(ValueError, match='exhausted'):
        other.reserve('too-much', 100, 'research', {'source': 'test'}, {'seconds': 1})
