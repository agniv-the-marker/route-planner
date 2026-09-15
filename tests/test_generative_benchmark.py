import json
from dataclasses import asdict
from pathlib import Path
from scripts.generative_benchmark import freeze, PROMPTS
from scripts.outline_benchmark import DESCRIPTIONS
from scripts.summarize_vectors import summarize
from src.generative import Candidate
from src.vector_program import SEEDS


def test_frozen_manifest_has_thirty_new_prompts_and_refuses_overwrite(tmp_path):
    import pytest
    output=freeze(tmp_path/'run')
    manifest=json.loads((output/'manifest.json').read_text())
    assert len(PROMPTS)==len(set(PROMPTS))==30
    assert not set(PROMPTS)&set(DESCRIPTIONS)
    assert len(SEEDS)*len(PROMPTS)==manifest['attempt_limit']==120
    assert len(manifest['pairs'])==10 and manifest['gpu_seconds_limit']==7200
    with pytest.raises(FileExistsError): freeze(output)


def test_summary_separates_first_sample_best_of_four_and_unreviewed(tmp_path):
    output=freeze(tmp_path/'run')
    for i in range(1,5):
        c=Candidate(PROMPTS[0],SEEDS[i-1],validation='valid' if i==2 else 'bad',
                    outline={'points':[]} if i==2 else None)
        (output/f'{i:03}.json').write_text(json.dumps(asdict(c)))
    report=summarize(output)
    assert report['recognition_status']=='unreviewed'
    assert not report['promotion_allowed']
    report=summarize(output,[{'id':2,'kind':'outline','recognizable':True}])
    assert report['first_sample']['outline']==0
    assert report['best_of_four']['outline']==1
    assert report['geometry_and_routing']['outline']['best_of_four_upper_bound']==1

def test_judge_cache_reuses_evaluation_but_not_infrastructure_errors(tmp_path):
    from scripts.judge_vectors import judge_saved
    output=freeze(tmp_path/'run')
    (output/'001.json').write_text(json.dumps(asdict(Candidate('horse',17,validation='invalid'))))
    class Judge:
        model='vision'
        revision='v1'
        calls=0
        def __call__(self,record):
            self.calls+=1
            return {'reward':-1,'excluded':False}
    judge=Judge()
    judge_saved(output,judge,tmp_path/'cache')
    rows=judge_saved(output,judge,tmp_path/'cache')
    assert judge.calls==1 and 'cache_seconds' in rows[0]['timings']
    class Outage(Judge):
        revision='v2'
        def __call__(self,record):
            self.calls+=1
            return {'reward':None,'excluded':True,'error':'service unavailable'}
    outage=Outage()
    judge_saved(output,outage,tmp_path/'cache')
    judge_saved(output,outage,tmp_path/'cache')
    assert outage.calls==2
