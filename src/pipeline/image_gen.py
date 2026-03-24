"""Stable Diffusion 1.5 image generation wrapper."""

from __future__ import annotations

import torch
from PIL import Image


DEFAULT_PROMPT_TEMPLATE = (
    "a simple black silhouette of a {concept} on a white background, "
    "minimal, clean outline, no background details"
)


class ImageGenerator:
    """Wraps SD 1.5 for generating silhouette images from text concepts."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        image_size: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

    def _load_pipeline(self):
        """Lazy-load the diffusion pipeline."""
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionPipeline

        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self._pipe = self._pipe.to(self.device)

    @property
    def pipe(self):
        self._load_pipeline()
        return self._pipe

    def generate(
        self,
        concept: str,
        seed: int | None = None,
        prompt_template: str | None = None,
    ) -> Image.Image:
        """Generate a silhouette image for the given concept.

        Args:
            concept: The text concept (e.g., "horse").
            seed: Optional random seed for reproducibility.
            prompt_template: Override the default prompt template.

        Returns:
            A PIL Image of size (image_size, image_size).
        """
        template = prompt_template or self.prompt_template
        prompt = template.format(concept=concept)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        return result.images[0]

    def update_pipeline(self, pipe):
        """Replace the internal pipeline (used during RL training)."""
        self._pipe = pipe
