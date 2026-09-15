"""CPU artifact review. Human labels are explicit and persist separately from evidence."""
from pathlib import Path
import json
import sqlite3
import html
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

class Review(BaseModel):
    label: Literal['unreviewed','pass','fail','uncertain']
    guess: str = Field(default='',max_length=500)
    details: str = Field(default='',max_length=2000)
    revealed: bool = False


def create_review_site(artifacts, state, commit=lambda:None, release='local'):
    artifacts,state=Path(artifacts),Path(state)
    state.mkdir(parents=True,exist_ok=True)
    db=state/'reviews.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, body TEXT NOT NULL)')
    manifest=json.loads((artifacts/'references.json').read_text())
    rows={r['id']:r for r in manifest['drawings']}
    review_namespace = manifest['version'] + ':'
    app=FastAPI()
    @app.get('/health')
    def health():return {'status':'ok','release':release,'source_approval':manifest['status'],'count':len(rows),'inference_enabled':False}
    @app.get('/routes', response_class=HTMLResponse)
    def route_review():
        cards = []
        for result_path in sorted((artifacts / 'comparisons').glob('*/result.json')):
            try:
                record = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            result, attempt = record.get('result', {}), record.get('attempt', {})
            rel = result_path.parent.relative_to(artifacts).as_posix()
            if not result.get('routes'):
                cards.append(f'<article><h3>Anonymous {html.escape(str(attempt))}</h3>'
                             f'<p>{html.escape(result.get("status", "unknown"))}: '
                             f'{html.escape(result.get("failure_reason", "no route"))}</p></article>')
                continue
            sections = []
            for i, route in enumerate(result['routes']):
                distance_km = float(route.get('distance_m', 0)) / 1000
                sections.append(
                    f'<section><h4>Route {i + 1}: {distance_km:.2f} km</h4>'
                    f'<img src="/{rel}/{i}-plain.svg" alt="plain route drawing">'
                    f'<details><summary>Reveal map overlay</summary>'
                    f'<img src="/{rel}/{i}-overlay.svg" alt="route over target">'
                    f'<img src="/{rel}/{i}-map.svg" alt="route with streets"></details>'
                    f'<p><a href="/{rel}/{i}.gpx">Download GPX</a>; '
                    f'retracing {route.get("scores", {}).get("repeated", 0):.3f}</p></section>')
            cards.append(f'<article><h3>Anonymous {html.escape(str(attempt))}</h3>{"".join(sections)}</article>')
        return ('<!doctype html><meta charset="utf-8"><title>Saved GPS routes</title>'
                '<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{border:1px solid #ddd;padding:1rem}img{width:100%;max-height:480px;object-fit:contain;background:#fff}details img{margin-top:.5rem}</style>'
                '<h1>Saved GPS route evidence</h1><p>Plain drawings appear first. Overlays, street maps and GPX are available per route.</p><p><a href="/">Back to source approval</a></p><main>' + ''.join(cards) + '</main>')
    @app.get('/api/reviews')
    def reviews():
        with sqlite3.connect(db) as conn:
            return {k.removeprefix(review_namespace): json.loads(v)
                    for k,v in conn.execute('SELECT id,body FROM reviews WHERE id LIKE ?',
                                             (review_namespace + '%',))}
    @app.put('/api/reviews/{item}')
    def save(item:str,review:Review):
        if item not in rows:raise HTTPException(404)
        with sqlite3.connect(db) as conn:
            conn.execute('INSERT OR REPLACE INTO reviews VALUES (?,?)',
                         (review_namespace + item,review.model_dump_json()))
        commit()
        return {'saved':item}
    @app.get('/',response_class=HTMLResponse)
    def index():
        page=(artifacts/'index.html').read_text()
        script='''<style>label{display:block;margin:.6rem 0}input,textarea,select,button{font:inherit;max-width:100%;padding:.4rem}#status{position:sticky;top:0;background:white;padding:1rem}</style><p id="status">Reviews are saved on this preview. Inspect each plain drawing, enter a guess, then reveal its subject. Acceptance is confirmed separately in the conversation.</p><script>
(async()=>{const saved=await (await fetch('/api/reviews')).json();
for(const article of document.querySelectorAll('article')){
const id=article.querySelector('h2').textContent.trim(), previous=saved[id]||{};
const form=document.createElement('div');form.innerHTML='<label>Your guess before reveal <input class="guess"></label><label>Source drawing <select><option>unreviewed</option><option>pass</option><option>fail</option><option>uncertain</option></select></label><label>Feature notes <textarea></textarea></label><button>Save review</button><span aria-live="polite"></span>';
article.append(form);form.querySelector('input').value=previous.guess||'';form.querySelector('textarea').value=previous.details||'';form.querySelector('select').value=previous.label||'unreviewed';
let revealed=previous.revealed||false;article.querySelector('details').addEventListener('toggle',()=>{revealed=true});
form.querySelector('button').onclick=async()=>{const response=await fetch('/api/reviews/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:form.querySelector('select').value,guess:form.querySelector('input').value,details:form.querySelector('textarea').value,revealed})});form.querySelector('span').textContent=response.ok?' Saved':' Save failed';};}
})();</script>'''
        return page.replace('<h1>12 revised source drawings</h1>', '<h1>12 revised source drawings</h1><p><a href="/routes">View saved GPS routes and GPX downloads</a></p>')+script
    app.mount('/',StaticFiles(directory=artifacts),name='artifacts')
    return app
