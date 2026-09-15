"""One bounded QLoRA outline-training allocation in nyro-robotics."""
import modal

MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'
REVISION = 'c03e6d358207e414f1eca0bb1891e29f1db0e242'

app = modal.App('route-sculptor-outline-sft')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.6.0', 'transformers==4.51.3', 'datasets==3.5.0',
                      'peft==0.15.2', 'trl==0.16.1', 'bitsandbytes==0.45.5',
                      'accelerate==1.6.0', 'sentencepiece==0.2.0')
         .add_local_dir('scripts', '/app/scripts', copy=True)
         .add_local_dir('experiments/outline-sft-v2', '/app/dataset', copy=True)
         .env({'PYTHONPATH': '/app'}))
volume = modal.Volume.from_name('route-sculptor-outline-checkpoints', create_if_missing=True)


@app.function(image=image, gpu='L4', timeout=3600, max_containers=1,
              volumes={'/checkpoints': volume})
def train_outline():
    from scripts.train_outline import train
    result = train('/app/dataset', '/checkpoints/outline-sft-v4', MODEL, REVISION,
                   max_steps=300, allow_training=True)
    volume.commit()
    return str(result)


@app.local_entrypoint()
def main():
    print(train_outline.remote())
