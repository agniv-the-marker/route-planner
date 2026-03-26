#!/usr/bin/env python3
"""DDPO training on Modal with A100 GPU.

Usage:
    modal run scripts/modal_train.py
    modal run scripts/modal_train.py --num-epochs 10  # quick test run
"""

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

IGNORE_PATTERNS = [
    ".venv/**", "__pycache__/**", "*.pyc", ".git/**",
    "outputs/**", "route_cache/**", "checkpoints/**",
    "*.egg-info/**", ".pytest_cache/**",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Core ML
        "torch>=2.1",
        "torchvision>=0.16",
        "diffusers>=0.25",
        "transformers>=4.36",
        "accelerate>=0.25",
        "trl>=0.7,<0.12",
        "peft>=0.6",
        # Vision / science
        "opencv-python-headless>=4.8",
        "osmnx>=2.0",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "scipy>=1.11",
        "Pillow>=10.0",
        # Routing & data
        "gpxpy>=1.6",
        "requests>=2.31",
        "aiohttp>=3.9",
        "pyyaml>=6.0",
        # Logging
        "wandb>=0.16",
    )
    .add_local_dir(".", remote_path="/app", ignore=IGNORE_PATTERNS, copy=True)
)

app = modal.App("route-sculptor-train", image=image)

# Persistent storage for checkpoints, HF model cache, and route cache
vol = modal.Volume.from_name("route-sculptor-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=6 * 3600,  # 6 hours max
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train(
    config_path: str = "configs/default.yaml",
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
):
    """Run DDPO training on a cloud GPU.

    Args:
        config_path: Path to config YAML (relative to project root).
        num_epochs: Override number of training epochs.
        batch_size: Override batch size.
        learning_rate: Override learning rate.
    """
    import sys
    import yaml

    sys.path.insert(0, "/app")

    # Point HuggingFace and route caches to persistent volume
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    os.makedirs("/data/hf_cache", exist_ok=True)
    os.makedirs("/data/route_cache", exist_ok=True)
    os.makedirs("/data/checkpoints", exist_ok=True)
    os.makedirs("/data/training_samples", exist_ok=True)

    # Point training outputs to persistent volume
    os.environ["TRAIN_SAMPLES_DIR"] = "/data/training_samples"
    os.environ["CHECKPOINT_DIR"] = "/data/checkpoints"

    # Ensure SF bike graph is available (download to volume if missing)
    graph_path = "/data/sf_bike_graph.graphml"
    if not os.path.exists(graph_path):
        print("Downloading SF bike graph to volume (one-time)...")
        import osmnx as ox
        G = ox.graph.graph_from_place(
            "San Francisco, California, USA",
            network_type="bike",
            custom_filter='["route"!~"ferry"]',
            retain_all=False, simplify=True,
        )
        ox.save_graphml(G, graph_path)
        print(f"Graph saved: {len(G.nodes)} nodes, {len(G.edges)} edges")
        vol.commit()

    # Load and optionally override config
    with open(f"/app/{config_path}") as f:
        config = yaml.safe_load(f)

    # Bump inference steps back to 30 for GPU training
    config["image_gen"]["num_inference_steps"] = 30

    # Use graph routing with volume-cached graph
    config["routing"]["backend"] = "graph"
    config["routing"]["graph_path"] = graph_path

    if num_epochs is not None:
        config["training"]["num_epochs"] = num_epochs
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate

    # Use persistent volume paths
    config["routing"]["request_delay"] = 0.5  # faster for training

    # Write the overridden config to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_config = f.name

    # Patch route cache dir to use persistent volume
    import src.pipeline.routing as routing_mod
    routing_mod.CACHE_DIR = Path("/data/route_cache")

    # Run training
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from src.training.ddpo_trainer import train as run_train
    run_train(tmp_config)

    # Persist everything
    vol.commit()
    print("Training complete. Checkpoints saved to /data/checkpoints/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    config: str = "configs/default.yaml",
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
):
    """Launch DDPO training on Modal A100."""
    print("Launching DDPO training on Modal A100...")
    print(f"  Config: {config}")
    if num_epochs:
        print(f"  Epochs override: {num_epochs}")
    if batch_size:
        print(f"  Batch size override: {batch_size}")

    train.remote(
        config_path=config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
