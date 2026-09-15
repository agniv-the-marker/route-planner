"""Twelve hand-authored source drawings; approval precedes any routing evaluation."""
from pathlib import Path
from dataclasses import asdict
import argparse
import html
import numpy as np
from scripts.outline_benchmark import fixtures
from src.strokes import Drawing
from src.experiment_journal import atomic_json, digest, now

def svg(strokes, overlay=(), streets=()):
    strokes = [np.asarray(s) for s in strokes]
    all_points = np.vstack(strokes + [np.asarray(s) for s in overlay])
    low,high = all_points.min(0),all_points.max(0)
    scale = 440/max(max(high-low),1e-9)
    def lines(items,color,width):
        return ''.join('<polyline fill="none" stroke="'+color+'" stroke-width="'+str(width)+'" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in (np.asarray(p)-(low+high)/2)*scale+250)+'"/>' for p in items)
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500"><rect width="500" height="500" fill="white"/>'+lines(streets,'#ddd',.6)+lines(overlay,'#e9a49b',2)+lines(strokes,'#111',3)+'</svg>'

def references():
    basic = fixtures()
    items = [(name,[basic[name].points],features) for name,features in [('heart',['two lobes and central notch']),('star',['five sharp points']),('fish',['broad tail']),('cat',['pointed ears and sitting body'])]]
    items += [
      # These are each one intentional pen stroke.  Where a shape branches, its
      # retracing is part of the drawing rather than an invisible component hop.
      ('house with a door', [[(.4,.9),(.4,.62),(.6,.62),(.6,.9),(.9,.9),(.9,.45),(.5,.08),(.1,.45),(.1,.9),(.4,.9)]], ['roof peak','door']),
      ('sailboat', [[(.1,.6),(.9,.6),(.74,.9),(.28,.9),(.1,.6),(.45,.6),(.45,.16),(.1,.6),(.5,.6),(.5,.08),(.9,.6),(.5,.6)]], ['two triangular sails joined by a boom','hull']),
      ('umbrella with curved handle', [[(.25,.86),(.34,.95),(.46,.92),(.5,.8),(.5,.46),(.3,.39),(.08,.46),(.14,.26),(.3,.12),(.5,.07),(.7,.12),(.86,.26),(.92,.46),(.7,.39),(.5,.46)]], ['canopy','hooked handle']),
      ('butterfly', [basic['butterfly'].points,[(.5,.65),(.5,.25),(.4,.08)],[(.5,.25),(.6,.08)]], ['four wings','antennae']),
      ('bicycle', [[*( [(.27+.18*np.cos(a),.75+.18*np.sin(a)) for a in np.linspace(-np.pi/2,3*np.pi/2,17)] ),(.27,.75),(.48,.4),(.73,.75),(.73,.57),*( [(.73+.18*np.cos(a),.75+.18*np.sin(a)) for a in np.linspace(-np.pi/2,3*np.pi/2,17)][1:] ),(.73,.75),(.48,.4),(.39,.4),(.48,.4),(.64,.42),(.68,.32),(.8,.32)]], ['two widely spaced wheels','triangular frame','handlebar']),
      ('flower with two leaves', [[*( [(.5+.24*np.cos(a)*(1+.28*np.cos(5*a)),.32+.24*np.sin(a)*(1+.28*np.cos(5*a))) for a in np.linspace(np.pi/2,5*np.pi/2,41)] ),(.5,.8),(.25,.6),(.18,.75),(.5,.87),(.5,.8),(.78,.65),(.8,.82),(.5,.92),(.5,.8),(.5,.95)]], ['petals touching the stem','two leaves']),
      ('running person', [[*( [(.53+.1*np.cos(a),.15+.1*np.sin(a)) for a in np.linspace(np.pi/2,5*np.pi/2,13)] ),(.48,.36),(.25,.3),(.15,.43),(.25,.3),(.48,.36),(.67,.4),(.8,.26),(.67,.4),(.48,.36),(.43,.53),(.3,.75),(.1,.77),(.3,.75),(.43,.53),(.68,.65),(.8,.87),(.68,.65),(.43,.53)]], ['head','bent arms','striding legs with deliberate retracing']),
      ('two mountains and sun', [[*( [(.34+.11*np.cos(a),.15+.11*np.sin(a)) for a in np.linspace(np.pi/2,5*np.pi/2,17)] ),(.05,.88),(.34,.26),(.52,.7),(.72,.18),(.95,.88)]], ['two distinct peaks','sun touching the left peak'])]
    # Tune across representation types; retain three silhouettes among the eight held-out sources.
    items = [items[i] for i in (0,4,8,10,1,2,3,5,6,7,9,11)]
    result=[]
    for i,(name,strokes,features) in enumerate(items,1):
        drawing=Drawing.parse({'name':name,'strokes':strokes})
        result.append({'id':f'{i:02}', 'split':'tuning' if i<=4 else 'comparison','drawing':asdict(drawing),'requested_features':features,'review':'unreviewed'})
    return result

def prepare(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=references()
    manifest={'version':'references-v2','created':now(),'status':'awaiting_source_approval','drawings':rows,'drawings_hash':digest(rows),'distance_km':[15,30,50,80],'comparison_gate':{'recognizable':6,'total':8},'source_acceptance':None}
    atomic_json(output/'references.json',manifest)
    cards=[]
    for row in rows:
        drawing=Drawing.parse(row['drawing'])
        path,connectors=drawing.continuous()
        (output/f'{row["id"]}-source.svg').write_text(svg(drawing.strokes))
        (output/f'{row["id"]}-continuous.svg').write_text(svg([path]))
        cards.append(f'<article><h2>{row["id"]}</h2><img src="{row["id"]}-source.svg"><details><summary>Reveal subject and details</summary><p>{html.escape(drawing.name)}: {html.escape(", ".join(row["requested_features"]))}</p><p>{row["split"]}; this is the entire proposed ride. Repeated segments are intentional and visible.</p><img src="{row["id"]}-continuous.svg"></details></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Source drawing approval v2</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}img{width:100%}article{border:1px solid #ddd;padding:1rem}</style><h1>12 revised source drawings</h1><p>Revised after review: every branch now joins an intended line directly. Retracing is deliberate and shown in the source. Inspect plain drawings first, then reveal the subject. IDs 01–04 tune parameters; 05–12 form the comparison set.</p><main>'+''.join(cards)+'</main>')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path)
    prepare(p.parse_args().output)
