import json
import pytest

from scripts.prepare_outline_dataset import build
from scripts.train_outline import train


def test_bootstrap_dataset_has_provenance_and_compilable_programs(tmp_path):
    from src.vector_program import compile_program
    manifest = build(tmp_path / 'dataset', ['heart', 'fish'])
    assert manifest['examples'] == 604
    assert manifest['held_out_families'] == ['fish', 'heart']
    rows = [json.loads(line) for line in (tmp_path / 'dataset' / 'examples.jsonl').read_text().splitlines()]
    assert len(rows) == 604
    assert all(row['license'] and row['source'] and row['source_sha256'] for row in rows)
    assert {row['split'] for row in rows if row['subject_family'] in {'fish', 'heart'}} == {'test'}
    for row in rows:
        compile_program(row['program'], row['prompt'])


def test_training_requires_explicit_reservation(tmp_path):
    build(tmp_path / 'dataset')
    with pytest.raises(PermissionError, match='allow-training'):
        train(tmp_path / 'dataset', tmp_path / 'checkpoint', 'model', 'revision')
