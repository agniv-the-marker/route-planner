"""Free-form descriptions select vetted vector silhouettes; models never draw coordinates."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from src.outlines import OutlineSpec

VERSION = 'symbol-selection-v1'
ASSET = Path(__file__).parent / 'assets' / 'symbols.json'
ALIASES = {'teapot':'kettle','sailboat':'sail-boat','boat':'sail-boat','guitar':'guitar-acoustic',
           'rocket':'rocket-launch','love':'heart','round':'circle', 'boot':'boot'}

@lru_cache(maxsize=1)
def catalogue():
    return json.loads(ASSET.read_text())['symbols']


def selection_schema():
    return {'type':'object', 'additionalProperties':False, 'required':['symbol','tilt','aspect'],
            'properties':{'symbol':{'type':'string','enum':[*catalogue(), 'none']},
                          'tilt':{'type':'integer','enum':[-20,0,20]},
                          'aspect':{'type':'string','enum':['normal','tall','wide']}}}


def selection_prompt():
    return ('Choose ONE available symbol that best interprets the user description. '
            'Use none if nothing is relevant. Output JSON only with symbol, tilt, aspect. '
            'tilt is degrees COUNTERCLOCKWISE: leaning left=20, right=-20, otherwise 0. '
            'aspect is normal unless explicitly tall/narrow or wide/flattened. '
            'Ignore extra detail that cannot be shown by an outer silhouette. '
            'Available symbols: '+', '.join(catalogue()))


def outline_from_selection(selection):
    if not isinstance(selection,dict) or set(selection) != {'symbol','tilt','aspect'}:
        raise ValueError('The description could not be interpreted.')
    key, tilt, aspect = selection['symbol'], selection['tilt'], selection['aspect']
    if key not in catalogue() or tilt not in (-20,0,20) or aspect not in ('normal','tall','wide'):
        raise ValueError('No suitable simple symbol was found for this description.')
    asset = catalogue()[key]
    p = np.asarray(asset['points'], dtype=float) - .5
    if aspect == 'tall': p[:,0] *= .65
    if aspect == 'wide': p[:,1] *= .65
    a = np.deg2rad(tilt)
    p = p @ np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    p = (p-p.min(axis=0)) / np.ptp(p,axis=0).max()
    label = asset['interpretation'] + (' · '+aspect if aspect != 'normal' else '')
    if tilt: label += ' · leaning '+('left' if tilt > 0 else 'right')
    return OutlineSpec.parse({'interpretation':label,'points':p.tolist()})


class SymbolGenerator:
    def __init__(self, remote=None, cache_dir=None):
        self.remote = remote
        self.cache_dir = Path(cache_dir) if cache_dir else Path('route_cache/symbols')

    def generate(self, prompt):
        normalized = ' '.join(prompt.casefold().split())
        key = hashlib.sha256((VERSION + ASSET.read_text() + normalized).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json'
        started = time.perf_counter()
        if path.exists():
            try:
                selection = json.loads(path.read_text())
                return outline_from_selection(selection), {'cache_seconds':time.perf_counter()-started}
            except (ValueError,TypeError): pass
        literal = ALIASES.get(normalized, normalized.replace(' ','-'))
        if literal in catalogue():
            selection = {'symbol':literal,'tilt':0,'aspect':'normal'}
            timings = {'interpretation_seconds':time.perf_counter()-started,'source':'exact symbol'}
        else:
            if self.remote is None:
                import modal
                from modal.config import config
                # Local calls explicitly use the authorized workspace, never the active default.
                if modal.is_local():
                    client = modal.Client.from_credentials(
                        config.get('token_id',profile='nyro-robotics',use_env=False),
                        config.get('token_secret',profile='nyro-robotics',use_env=False))
                else:
                    client = None  # Modal container identity belongs to the deployed workspace.
                worker = modal.Cls.from_name('route-sculptor-symbols','SymbolWorker',client=client)()
                response = worker.select.remote(prompt)
            else:
                response = self.remote(prompt)
            selection = response['selection']
            timings = {**response.get('timings',{}),'interpretation_wall_seconds':time.perf_counter()-started}
        spec = outline_from_selection(selection)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', dir=self.cache_dir, delete=False) as temp:
            json.dump(selection,temp)
        os.replace(temp.name,path)
        return spec,timings
