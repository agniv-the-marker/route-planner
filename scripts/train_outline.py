"""Bounded LoRA SFT for the text-to-outline bootstrap model.

This module is intentionally explicit: it refuses missing provenance and does not
silently turn a failed training job into a usable checkpoint.
"""
import argparse
import json
from pathlib import Path


def train(dataset, output, model, revision, max_steps=300, allow_training=False):
    if not allow_training:
        raise PermissionError('Pass --allow-training after reserving a bounded experiment.')
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('license') is None or manifest.get('examples', 0) < 100:
        raise ValueError('Training requires the licensed, provenance-bearing bootstrap dataset.')
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError('Install the pinned training image dependencies first.') from exc
    # Read JSONL ourselves: Arrow's JSON reader tries to unify mixed command
    # arrays ("M" plus numeric coordinates) into a scalar column and rejects
    # otherwise valid programs.
    rows = [json.loads(line) for line in (dataset / 'examples.jsonl').read_text().splitlines()]
    for row in rows:
        row['program'] = json.dumps(row['program'], separators=(',', ':'))
    data = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                                                quantization_config=quant,
                                                device_map='auto', trust_remote_code=False)
    def format_row(row):
        messages = [{'role': 'system', 'content': row['system']},
                    {'role': 'user', 'content': row['prompt']},
                    {'role': 'assistant', 'content': row['program']}]
        return {'text': tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    data = data.map(format_row, remove_columns=data.column_names)
    output.mkdir(parents=True, exist_ok=False)
    trainer = SFTTrainer(
        model=base, processing_class=tokenizer, train_dataset=data,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                               target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'),
        args=SFTConfig(output_dir=str(output), max_steps=max_steps,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8,
                       learning_rate=2e-4, logging_steps=10, save_strategy='steps',
                       save_steps=max_steps, report_to=[], bf16=False, fp16=True,
                       dataset_text_field='text', max_length=1536),
    )
    trainer.train()
    trainer.save_model(str(output))
    (output / 'training-manifest.json').write_text(json.dumps({
        'model': model, 'revision': revision, 'dataset_manifest': manifest,
        'max_steps': max_steps, 'method': 'sft-lora', 'status': 'completed',
    }, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--allow-training', action='store_true')
    args = parser.parse_args()
    train(args.dataset, args.output, args.model, args.revision, args.max_steps, args.allow_training)
