from pathlib import Path
import json
import pytest
import numpy as np
from src.symbols import SymbolGenerator,catalogue,outline_from_selection,selection_schema
from src.outlines import OutlineSpec


def test_entire_catalogue_is_valid_geometry():
    assert len(catalogue()) >= 100
    for asset in catalogue().values():
        OutlineSpec.parse({k:asset[k] for k in ('interpretation','points')})


def test_exact_symbol_never_starts_remote_worker(tmp_path):
    def remote(prompt): raise AssertionError('An exact symbol needs no GPU')
    spec,timings=SymbolGenerator(remote,tmp_path).generate('HEART')
    assert spec.interpretation=='heart' and timings['source']=='exact symbol'


def test_arbitrary_description_interpretation_and_persistent_cache(tmp_path):
    calls=[]
    def remote(prompt):
        calls.append(prompt)
        return {'selection':{'symbol':'heart','tilt':0,'aspect':'normal'}}
    generator=SymbolGenerator(remote,tmp_path)
    first,_=generator.generate('a warm welcome for a friend')
    second,timing=SymbolGenerator(remote,tmp_path).generate(' A warm  welcome for a friend ')
    assert first==second and len(calls)==1 and 'cache_seconds' in timing
    # A catalogue change changes the key: different descriptions do not share stale interpretations.
    generator.generate('joy and love')
    assert len(calls)==2


def test_unknown_symbol_and_malformed_response_do_not_cache(tmp_path):
    def remote(prompt): return {'selection':{'symbol':'none','tilt':0,'aspect':'normal'}}
    with pytest.raises(ValueError,match='No suitable'):
        SymbolGenerator(remote,tmp_path).generate('some impossible description')
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        outline_from_selection({'symbol':'heart','tilt':90,'aspect':'normal'})


def test_left_tilt_and_aspect_preserve_valid_outline():
    normal=outline_from_selection({'symbol':'heart','tilt':0,'aspect':'normal'})
    left=outline_from_selection({'symbol':'heart','tilt':20,'aspect':'tall'})
    assert 'leaning left' in left.interpretation
    assert not np.allclose(normal.points,left.points)
    assert 'none' in selection_schema()['properties']['symbol']['enum']
