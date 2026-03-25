"""Image generation: SVG/PNG silhouette loader with SD 1.5 fallback."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


SILHOUETTES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "silhouettes"


def load_silhouette(concept: str, silhouettes_dir: Path | str = SILHOUETTES_DIR) -> Image.Image | None:
    """Load a pre-made silhouette PNG for a concept.

    Returns None if no silhouette exists for this concept.
    """
    path = Path(silhouettes_dir) / f"{concept}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


DEFAULT_PROMPT_TEMPLATE = (
    "simple 2d black silhouette of a {concept}, full body, solid filled shape "
    "on plain white background, flat graphic, logo style, no detail, no shading"
)

DEFAULT_NEGATIVE_PROMPT = (
    "3d, depth, perspective, realistic, photograph, detailed, complex, intricate, "
    "outline, lines, strokes, sketch, hatching, texture, shading, gradient, gray, "
    "person, people, human, hands, text, watermark, color, multiple objects, "
    "cropped, partial, head only, close up, blurry, noise"
)


class ImageGenerator:
    """Wraps SD 1.5 for generating silhouette images from text concepts."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        image_size: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 10.0,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.negative_prompt = negative_prompt
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
            negative_prompt=self.negative_prompt,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        return self._force_binary(image)

    @staticmethod
    def _force_binary(image: Image) -> Image:
        """Force SD output to pure black silhouette on white background.

        SD 1.5 often produces brown/gradient backgrounds instead of pure white.
        This threshold step removes all ambiguity, giving contour extraction
        a clean binary image to work with.
        """
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Fill small holes in the silhouette (anti-aliasing artifacts)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Invert: white background, black silhouette
        clean = 255 - binary
        return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB))

    def update_pipeline(self, pipe):
        """Replace the internal pipeline (used during RL training)."""
        self._pipe = pipe
