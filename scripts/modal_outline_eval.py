"""Held-out text->outline evaluation for the trained adapter."""
import json
import modal

MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
HELD_OUT = ('heart', 'dinosaur', 'horse', 'cat', 'fish', 'butterfly')
TEMPLATES = ('a {name}', 'a simple {name} silhouette', 'the outline of a {name}', 'a recognizable {name} icon')

app = modal.App('route-sculptor-outline-eval')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.51.3', 'peft==0.15.2',
                      'bitsandbytes==0.45.5', 'accelerate==1.6.0', 'sentencepiece==0.2.0',
                      'shapely==2.0.7')
         .add_local_file('src/vector_program.py', '/app/src/vector_program.py', copy=True)
         .add_local_file('src/outlines.py', '/app/src/outlines.py', copy=True)
         .add_local_file('scripts/modal_outline_eval.py', '/app/scripts/modal_outline_eval.py', copy=True)
         .env({'PYTHONPATH': '/app'}))
volume = modal.Volume.from_name('route-sculptor-outline-checkpoints', create_if_missing=True)


@app.function(image=image, gpu='L4', timeout=1800, max_containers=1,
              volumes={'/checkpoints': volume})
def evaluate():
    import torch
    from datetime import datetime, timezone
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.vector_program import compile_program, extract_program
    checkpoint = '/checkpoints/outline-sft-v4'
    status_path = '/checkpoints/outline-sft-v4-heldout-status.json'
    with open(status_path, 'w') as f:
        json.dump({'stage': 'starting', 'updated_at': datetime.now(timezone.utc).isoformat()}, f)
    volume.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                quantization_config=quant, device_map='auto',
                                                trust_remote_code=False)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    with open(status_path, 'w') as f:
        json.dump({'stage': 'generating', 'updated_at': datetime.now(timezone.utc).isoformat()}, f)
    volume.commit()
    system = 'Draw the exact subject as one closed silhouette. Return only a JSON array of M/L/Q/C/Z commands with integer coordinates from 0 to 1024. Start with M and end with Z. Use one subpath, at most 32 commands, and preserve distinguishing features. Do not return prose or SVG.'
    rows = []
    for name in HELD_OUT:
        for template in TEMPLATES:
            prompt = template.format(name=name)
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
            encoded = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(encoded, max_new_tokens=256, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0][encoded.shape[-1]:], skip_special_tokens=True)
            row = {'prompt': prompt, 'raw': raw, 'validation': 'invalid'}
            try:
                compile_program(extract_program(raw), prompt)
                row['validation'] = 'valid'
            except Exception as exc:
                row['error'] = str(exc)
            rows.append(row)
            with open(status_path, 'w') as f:
                json.dump({'stage': 'generating', 'completed': len(rows), 'total': len(HELD_OUT) * len(TEMPLATES),
                           'updated_at': datetime.now(timezone.utc).isoformat()}, f)
            volume.commit()
    path = '/checkpoints/outline-sft-v4-heldout.json'
    with open(path, 'w') as f:
        json.dump({'model': MODEL, 'revision': REVISION, 'held_out': HELD_OUT, 'rows': rows}, f, indent=2)
    volume.commit()
    return {'path': path, 'valid': sum(r['validation'] == 'valid' for r in rows), 'total': len(rows)}


@app.local_entrypoint()
def main():
    print(evaluate.remote())
