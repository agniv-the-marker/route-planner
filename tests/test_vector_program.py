import json
import pytest
from src.vector_program import compile_program
from src.generative import Candidate, VectorGenerator
from src.route_reward import reward, calibration

BOX = [['M',100,100],['L',900,100],['L',900,900],['L',100,900],['Z']]

def test_compiler_closure_and_orientation():
    spec = compile_program(BOX,'box')
    assert 8 <= len(spec.points) <= 48
    assert spec.points[0] == spec.points[-1]
    assert spec.points[0][1] < .2

@pytest.mark.parametrize('program',[
    [['M',True,1],['L',900,100],['L',900,900],['Z']],
    [['M',0,0],['L',1025,0],['L',0,100],['Z']],
    [['M',0,0],['L',900,900],['L',0,900],['L',900,0],['Z']],
    BOX[:-1], BOX[:-1]+[['M',50,50],['Z']],
    [['M',0,0],['eval','bad'],['L',1,1],['Z']],
])
def test_rejects_unsafe_or_invalid(program):
    with pytest.raises(ValueError): compile_program(program,'x')

def test_curves_and_no_repair():
    program = [['M',100,500],['C',100,0,900,0,900,500],['Q',500,1000,100,500],['Z']]
    assert compile_program(program,'curved').points
    calls=[]
    def complete(messages,**kw):
        calls.append(kw); return 'not json'
    generator=VectorGenerator(complete)
    assert len(list(generator.candidates('horse'))) == 4
    assert len(calls)==4 and len({c['seed'] for c in calls})==4
    assert all(c['max_tokens']==1536 for c in calls)

def test_fenced_completion_is_extracted_without_changing_raw_attempt():
    from src.vector_program import extract_program
    raw='```json\n[["M",100,100],["L",900,100],["L",500,900],["Z"]]\n```'
    assert extract_program(raw)[0] == ['M',100,100]

def test_reward_semantics_and_failure_cases():
    c=Candidate('horse',17,validation='bad')
    assert reward(c)['reward']==-1
    c.validation='valid'
    assert reward(c)['reward']==0
    c.route={'scores':{'overlap':1.,'repeated':0.,'outline_distance':0.}}
    # A circle, wrong pose, mirrored directional animal, or lost features must
    # receive low judge fidelity even when street geometry is perfect.
    for failure in ('circle','wrong pose','mirrored','lost features'):
        assert reward(c,0,0)['reward']==pytest.approx(.3)
    assert reward(c,1,1)['reward']==pytest.approx(1)
    assert reward(c,judge_error='unavailable')['reward'] is None
    assert not calibration([{'id':i,'human':'a','judge':'a'} for i in range(49)])['rl_gate']
    assert calibration([{'id':i,'human':'a','judge':'a' if i<40 else 'b'} for i in range(50)])['rl_gate']

def test_vision_judge_uses_original_prompt_and_excludes_outages():
    from dataclasses import asdict
    from src.vector_judge import VisionJudge, balanced_pairs
    c = Candidate('a right-facing horse',17,validation='valid',outline=asdict(compile_program(BOX,'different interpretation')),
                  route={'xy':[[0,0],[1,0],[1,1],[0,0]],'scores':{'overlap':1,'repeated':0,'outline_distance':0}})
    calls=[]
    def complete(**kwargs):
        calls.append(kwargs)
        return '{"fidelity":0.2,"reason":"wrong subject"}'
    result=VisionJudge(complete,'separate-vision-model','frozen-revision')(c)
    assert all(x['prompt']==c.prompt for x in calls)
    assert result['route_prompt_fidelity']==.2
    def outage(**kwargs): raise RuntimeError('offline')
    assert VisionJudge(outage,'vision','rev')(c)['excluded']
    pairs=balanced_pairs(range(11))
    assert len(pairs)==50 and sum(p['a']<p['b'] for p in pairs)==25


def test_training_family_split_and_license_requirements():
    from scripts.vector_training import prepare_examples, configuration
    rows=[{'license':'CC0','source':'fixture','subject_family':'horse','prompt':p,'program':BOX}
          for p in ('standing horse','galloping horse')]
    assert {r['split'] for r in prepare_examples(rows,['horse'])}=={'test'}
    assert not configuration('grpo')['training_enabled']
    with pytest.raises(ValueError): prepare_examples([{'prompt':'horse'}])

def test_generation_cache_retains_invalid_and_separates_prompts(tmp_path):
    calls=[]
    def complete(messages,**kwargs):
        calls.append(messages); return 'invalid raw'
    generator=VectorGenerator(complete,cache_dir=tmp_path)
    list(generator.candidates('standing horse'))
    cached=list(generator.candidates('standing horse'))
    assert len(calls)==4 and all(c.raw=='invalid raw' and 'cache_seconds' in c.timings for c in cached)
    list(generator.candidates('galloping horse'))
    assert len(calls)==8

def test_evaluation_key_ignores_timings_but_tracks_judge_and_raw():
    from src.generative import cache_key
    c=Candidate('horse',17,raw='invalid')
    key=cache_key(c,'graph','judge-v1')
    c.timings['generation_seconds']=12
    c.scores['reward']=0
    assert cache_key(c,'graph','judge-v1')==key
    assert cache_key(c,'graph','judge-v2')!=key
    c.raw='another invalid attempt'
    assert cache_key(c,'graph','judge-v1')!=key


def test_parser_command_cap_and_feature_preservation():
    from shapely.geometry import Polygon
    import numpy as np
    program=[['M',50,50]]+[['L',50+i,100] for i in range(31)]+[['Z']]
    with pytest.raises(ValueError): compile_program(program,'too long')
    # A narrow, deep concavity must survive simplification.
    notch=[['M',100,100],['L',450,100],['L',450,700],['L',480,700],
           ['L',480,100],['L',900,100],['L',900,900],['L',100,900],['Z']]
    spec=compile_program(notch,'notched block')
    original=Polygon([c[1:] for c in notch[:-1]])
    actual=Polygon(np.array(spec.points)*1024)
    assert original.boundary.hausdorff_distance(actual.boundary)<=4
    assert (450/1024,700/1024) in spec.points

def test_comparison_judge_uses_balanced_pair_images_and_excludes_errors():
    from src.vector_judge import VisionJudge
    c=Candidate('horse',17,route={'xy':[[0,0],[1,0],[1,1],[0,0]]})
    calls=[]
    def complete(**kwargs):
        calls.append(kwargs);return '{"choice":"b","reason":"pose preserved"}'
    pair={'id':1,'a':2,'b':1,'prompt':'horse','human':'b'}
    result=VisionJudge(complete,'vision','rev').compare(pair,c,c)
    assert result['judge']=='b' and len(calls[0]['images'])==2
    assert calls[0]['prompt']=='horse'
    assert calibration([result]*50)['reviewed']==1
    c.prompt='different'
    assert VisionJudge(complete,'vision','rev').compare(pair,c,c)['excluded']

def test_judge_tie_counts_as_disagreement_with_decisive_human():
    result=calibration([{'id':i,'human':'a','judge':'tie'} for i in range(50)])
    assert result['reviewed']==50 and result['agreement']==0

def test_outline_judgment_is_retained_even_when_routing_fails():
    from dataclasses import asdict
    from src.vector_judge import VisionJudge
    calls=[]
    def complete(**kwargs):
        calls.append(kwargs);return '{"fidelity":0.4,"reason":"partial subject"}'
    c=Candidate('horse',17,validation='valid',outline=asdict(compile_program(BOX,'horse')))
    scores=VisionJudge(complete,'vision','rev')(c)
    assert len(calls)==1 and scores['reward']==0
    assert scores['outline_prompt_fidelity']==.4 and not scores['excluded']
