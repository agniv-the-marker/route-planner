"""DDPO training loop using HuggingFace TRL."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_ddpo_config(config: dict):
    """Build TRL DDPOConfig from our config dict."""
    from trl import DDPOConfig

    train_cfg = config["training"]
    log_cfg = config.get("logging", {})

    return DDPOConfig(
        num_epochs=train_cfg["num_epochs"],
        train_gradient_accumulation_steps=1,
        sample_num_steps=config["image_gen"]["num_inference_steps"],
        sample_batch_size=train_cfg["batch_size"],
        train_batch_size=train_cfg["batch_size"],
        sample_num_batches_per_epoch=train_cfg.get("num_samples_per_concept", 4),
        per_prompt_stat_tracking=True,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        tracker_project_name=log_cfg.get("wandb_project", "route-sculptor"),
        log_with="wandb" if log_cfg.get("wandb_project") else None,
        project_kwargs={
            "project_dir": "checkpoints",
            "logging_dir": "checkpoints/logs",
        },
    )


def create_reward_fn(config: dict):
    """Create the reward function from config."""
    from src.pipeline.placement import BBox
    from src.pipeline.routing import OSRMRouter
    from src.evaluation.clip_score import CLIPScorer
    from src.evaluation.reward import RewardFunction
    from src.training.reward_fn import DDPORewardWrapper

    sf = config["sf_bbox"]
    bbox = BBox(
        min_lat=sf["min_lat"],
        max_lat=sf["max_lat"],
        min_lon=sf["min_lon"],
        max_lon=sf["max_lon"],
    )

    routing_cfg = config["routing"]
    router = OSRMRouter(
        base_url=routing_cfg["base_url"],
        profile=routing_cfg["profile"],
        request_delay=routing_cfg["request_delay"],
    )

    eval_cfg = config["evaluation"]
    clip_scorer = CLIPScorer(
        model_id=eval_cfg["clip_model"],
        prompt_template=eval_cfg["clip_prompt_template"],
    )

    reward_fn = RewardFunction(
        clip_weight=eval_cfg["reward_clip_weight"],
        chamfer_weight=eval_cfg["reward_chamfer_weight"],
        clip_scorer=clip_scorer,
        render_size=eval_cfg["render_size"],
    )

    placement_cfg = config["placement"]
    waypoint_cfg = config["waypoints"]

    return DDPORewardWrapper(
        bbox=bbox,
        router=router,
        reward_fn=reward_fn,
        num_waypoints=waypoint_cfg["num_points"],
        num_positions=placement_cfg["num_positions"],
        num_scales=placement_cfg["num_scales"],
        num_rotations=placement_cfg["num_rotations"],
        scale_range_km=tuple(placement_cfg["scale_range_km"]),
        canny_low=config["edge_detect"]["canny_low"],
        canny_high=config["edge_detect"]["canny_high"],
    )


def train(config_path: str = "configs/default.yaml"):
    """Run DDPO training.

    This sets up the SD 1.5 pipeline, reward function, and TRL DDPOTrainer,
    then runs the training loop.
    """
    config = load_config(config_path)

    logger.info("Building DDPO config...")
    ddpo_config = build_ddpo_config(config)

    logger.info("Loading Stable Diffusion pipeline...")
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    sd_model_id = config["image_gen"]["model_id"]
    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    logger.info("Setting up reward function...")
    reward_wrapper = create_reward_fn(config)

    logger.info("Loading concepts dataset...")
    from src.training.dataset import load_concepts, build_prompts

    concepts = load_concepts()
    prompts = build_prompts(concepts, config["image_gen"]["prompt_template"])

    logger.info(f"Training with {len(concepts)} concepts")

    # Build the reward function that TRL expects
    # TRL DDPOTrainer expects: reward_fn(images, prompts, metadata) -> rewards, metadata
    def trl_reward_fn(images, prompts_batch, metadata):
        """Reward function in the format TRL expects."""
        # Extract concepts from metadata or parse from prompts
        concept_list = []
        for m in metadata:
            concept_list.append(m.get("concept", "unknown"))

        results = reward_wrapper(images, prompts_batch, concept_list)
        rewards = torch.tensor([r["reward"] for r in results])
        # Return updated metadata with scores
        for m, r in zip(metadata, results):
            m.update(r)
        return rewards, metadata

    # Build the prompt function that generates prompts for each training step
    import random

    def prompt_fn():
        """Generate a random concept prompt for training."""
        idx = random.randint(0, len(concepts) - 1)
        concept = concepts[idx]
        prompt = prompts[idx]
        return prompt, {"concept": concept}

    logger.info("Initializing DDPOTrainer...")
    from trl import DDPOTrainer

    trainer = DDPOTrainer(
        config=ddpo_config,
        reward_function=trl_reward_fn,
        prompt_function=prompt_fn,
        sd_pipeline=pipe,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    save_dir = Path("checkpoints") / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.sd_pipeline.save_pretrained(str(save_dir))
    logger.info(f"Training complete. Model saved to {save_dir}")
