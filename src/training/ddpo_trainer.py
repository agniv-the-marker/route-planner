"""DDPO training loop using HuggingFace TRL."""

from __future__ import annotations

import logging
import os
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

    # Resolve checkpoint dir (use /data/checkpoints on Modal, local otherwise)
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return DDPOConfig(
        num_epochs=train_cfg["num_epochs"],
        train_learning_rate=train_cfg["learning_rate"],
        train_gradient_accumulation_steps=1,
        sample_num_steps=config["image_gen"]["num_inference_steps"],
        sample_batch_size=train_cfg["batch_size"],
        train_batch_size=train_cfg["batch_size"],
        sample_num_batches_per_epoch=train_cfg.get("num_samples_per_concept", 4),
        per_prompt_stat_tracking=True,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        tracker_project_name=log_cfg.get("wandb_project", "route-sculptor"),
        log_with="wandb" if log_cfg.get("wandb_project") else None,
        logdir=checkpoint_dir,
        save_freq=train_cfg.get("save_every", 10),
        project_kwargs={
            "project_dir": checkpoint_dir,
            "logging_dir": os.path.join(checkpoint_dir, "logs"),
        },
    )


def create_reward_fn(config: dict):
    """Create the reward function from config."""
    from src.pipeline.placement import BBox
    from src.pipeline.routing import create_router
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

    router = create_router(config["routing"])

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
        curvature_weight=waypoint_cfg.get("curvature_weight", 2.0),
        max_gap_km=waypoint_cfg.get("max_gap_km", 0.15),
        max_rotation_deg=placement_cfg.get("max_rotation_deg", 15.0),
        multi_contour=config.get("multi_contour", {}).get("enabled", True),
        min_inner_area_ratio=config.get("multi_contour", {}).get("min_inner_area_ratio", 0.005),
        max_inner_contours=config.get("multi_contour", {}).get("max_inner_contours", 5),
        outer_budget_min=config.get("multi_contour", {}).get("outer_budget_min", 0.6),
        min_inner_waypoints=config.get("multi_contour", {}).get("min_inner_waypoints", 4),
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
    from trl import DefaultDDPOStableDiffusionPipeline

    sd_model_id = config["image_gen"]["model_id"]
    pipeline = DefaultDDPOStableDiffusionPipeline(
        sd_model_id,
        use_lora=True,
    )

    logger.info("Setting up reward function...")
    reward_wrapper = create_reward_fn(config)

    logger.info("Loading concepts dataset...")
    from src.training.dataset import load_concepts, build_prompts

    concepts = load_concepts()
    prompts = build_prompts(concepts, config["image_gen"]["prompt_template"])

    logger.info(f"Training with {len(concepts)} concepts")

    # W&B logging for images and metrics
    step_counter = [0]  # mutable counter for closure
    self_ref = reward_wrapper  # reference for accessing params in closures

    def log_training_step(results, epoch_num):
        """Log images, routes, and metrics to W&B."""
        try:
            import wandb
            if wandb.run is None:
                return

            log_dict = {}
            images_to_log = []

            for i, r in enumerate(results):
                concept = r.get("_concept", "unknown")

                # Log scores as metrics
                log_dict[f"reward/{concept}"] = r["reward"]
                log_dict[f"clip_score/{concept}"] = r["clip_score"]
                log_dict[f"chamfer_score/{concept}"] = r["chamfer_score"]

                # Log generated silhouette image
                if "_image" in r:
                    images_to_log.append(
                        wandb.Image(r["_image"], caption=f"{concept} silhouette (r={r['reward']:.3f})")
                    )

                # Log Canny edge detection output
                if "_edges" in r:
                    images_to_log.append(
                        wandb.Image(r["_edges"], caption=f"{concept} edges (canny {self_ref.canny_low}/{self_ref.canny_high})")
                    )

                # Log contour overlay on original image
                if "_contour_overlay" in r:
                    n_pts = r.get("_num_contour_points", "?")
                    images_to_log.append(
                        wandb.Image(r["_contour_overlay"], caption=f"{concept} contour ({n_pts} pts)")
                    )

                # Log route polyline and styled map view
                if "_route" in r and len(r["_route"]) > 2:
                    from src.evaluation.render import render_polyline, _render_styled_polyline
                    polyline_img = render_polyline(r["_route"])
                    images_to_log.append(
                        wandb.Image(polyline_img, caption=f"{concept} route ({r['_num_route_points']} pts)")
                    )
                    map_img = _render_styled_polyline(r["_route"], size=512)
                    images_to_log.append(
                        wandb.Image(map_img, caption=f"{concept} map (r={r['reward']:.3f})")
                    )

            if images_to_log:
                log_dict["samples"] = images_to_log

            # Aggregate metrics
            all_rewards = [r["reward"] for r in results]
            log_dict["reward/mean"] = sum(all_rewards) / len(all_rewards)
            log_dict["reward/max"] = max(all_rewards)
            log_dict["reward/min"] = min(all_rewards)
            log_dict["epoch"] = epoch_num

            wandb.log(log_dict, step=step_counter[0])
            step_counter[0] += 1

        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")

    # Save sample images to disk for inspection
    samples_dir = Path(os.environ.get("TRAIN_SAMPLES_DIR", "training_samples"))
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Build the reward function that TRL expects
    # TRL DDPOTrainer expects: reward_fn(images, prompts, metadata) -> (reward, metadata)
    def trl_reward_fn(images, prompts_batch, metadata):
        """Reward function in the format TRL expects."""
        concept_list = []
        for m in metadata:
            concept_list.append(m.get("concept", "unknown"))

        results = reward_wrapper(images, prompts_batch, concept_list)
        rewards = torch.tensor([r["reward"] for r in results])

        # Log to W&B and save sample images
        epoch = step_counter[0]
        log_training_step(results, epoch)

        # Save sample images to disk every step
        for i, r in enumerate(results):
            concept = r.get("_concept", "unknown")
            prefix = samples_dir / f"epoch{epoch:04d}_{concept}_{i}"
            if "_image" in r:
                r["_image"].save(str(prefix) + "_silhouette.png")
            if "_edges" in r:
                r["_edges"].save(str(prefix) + "_edges.png")
            if "_contour_overlay" in r:
                r["_contour_overlay"].save(str(prefix) + "_contour.png")
            if "_route" in r and len(r["_route"]) > 2:
                from src.evaluation.render import render_polyline, _render_styled_polyline
                polyline_img = render_polyline(r["_route"])
                polyline_img.save(str(prefix) + "_route.png")
                map_img = _render_styled_polyline(r["_route"], size=512)
                map_img.save(str(prefix) + "_map.png")

        # Update metadata with scores
        for m, r in zip(metadata, results):
            m["reward"] = r["reward"]
            m["clip_score"] = r["clip_score"]
            m["chamfer_score"] = r["chamfer_score"]

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
        sd_pipeline=pipeline,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model to the same checkpoint dir used during training
    save_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints")) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.sd_pipeline.save_pretrained(str(save_dir))
    logger.info(f"Training complete. Model saved to {save_dir}")
