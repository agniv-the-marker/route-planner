"""Data validation and SFT/GRPO LoRA configuration only. No training entrypoint."""
import hashlib
from src.vector_program import MODEL, REVISION, compile_program


def subject_split(family):
    bucket = int(hashlib.sha256(family.strip().casefold().encode()).hexdigest(),16)%10
    return 'test' if bucket==0 else 'validation' if bucket==1 else 'train'


def prepare_examples(rows, benchmark_families=()):
    held_out = {x.strip().casefold() for x in benchmark_families}
    examples = []
    for row in rows:
        if not all(row.get(key) for key in ('license','source','subject_family','prompt','program')):
            raise ValueError('Licensed source and subject family provenance required.')
        compile_program(row['program'],row['prompt'])
        family = row['subject_family'].strip().casefold()
        split = 'test' if family in held_out else subject_split(family)
        examples.append({**row,'split':split})
    return examples


def configuration(method):
    if method not in ('sft','grpo'):
        raise ValueError('Expected sft or grpo.')
    return {'method':method,'model':MODEL,'revision':REVISION,
            'lora':{'r':16,'alpha':32,'target_modules':['q_proj','v_proj']},
            'num_generations':4,'max_completion_length':1536,
            'router_processes':'one StreetSearch instance per CPU process',
            'reward':'src.route_reward.reward','training_enabled':False,
            'requires':['subject-family split','licensed sources',
                        'human-calibrated reward >=80% on >=50 comparisons',
                        '24/30 outlines and 18/30 routes recognizable']}
