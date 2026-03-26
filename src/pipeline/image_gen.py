"""Image generation: SVG/PNG silhouette loader with SD 1.5 fallback."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


SILHOUETTES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "silhouettes"

# Disambiguation for concepts that SD 1.5 misinterprets
CONCEPT_OVERRIDES = {
    "cross": "plus sign shape, medical cross symbol",
    "diamond": "diamond shape, rhombus, playing card diamond suit",
    "arrow": "right-pointing arrow, direction arrow pointing right",
    "snake": "long coiled snake, serpent side view",
    "whale": "whale side view, large marine mammal",
    "tree": "single tree with trunk and round crown, deciduous tree",
    "moon": "crescent moon shape",
    "turtle": "sea turtle side view",
}


def load_silhouette(concept: str, silhouettes_dir: Path | str = SILHOUETTES_DIR) -> Image.Image | None:
    """Load a pre-made silhouette PNG for a concept.

    Returns None if no silhouette exists for this concept.
    """
    path = Path(silhouettes_dir) / f"{concept}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


DEFAULT_PROMPT_TEMPLATE = (
    "black silhouette of a single {concept}, solid black shape, pure white background, "
    "simple flat icon, minimal, centered, one object only"
)

DEFAULT_NEGATIVE_PROMPT = (
    "circle around, encircled, ring, border, frame, multiple, two, many, pair, group, "
    "pattern, texture, ornate, decorated, background, scenery, environment, water, ocean, "
    "sky, ground, grass, nature, landscape, 3d, realistic, photo, detailed, complex, "
    "lines, sketch, hatching, shading, gradient, gray, person, human, hands, "
    "text, watermark, color, noise, blurry"
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
        # Use disambiguation override if available
        concept_text = CONCEPT_OVERRIDES.get(concept, concept)
        prompt = template.format(concept=concept_text)

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
        raw = result.images[0]
        self._last_raw = raw  # save for debugging
        return self._force_binary(raw)

    @staticmethod
    def _force_binary(image: Image) -> Image:
        """Force SD output to pure black silhouette on white background.

        Tries both orientations (normal + inverted) and picks the one
        that produces the most centered, reasonably-sized silhouette.
        This handles any background color — light, dark, or mixed.
        """
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        cx, cy = w / 2, h / 2
        img_area = h * w
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        def _score_orientation(g):
            """Score a grayscale image: higher = better silhouette candidate."""
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Remove tiny blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            min_area = img_area * 0.01
            for lid in range(1, num_labels):
                if stats[lid, cv2.CC_STAT_AREA] < min_area:
                    binary[labels == lid] = 0

            # Recompute after cleanup
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            if num_labels <= 1:
                return -1, binary  # no foreground

            # Find the largest remaining component
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            area = stats[largest, cv2.CC_STAT_AREA]
            centroid_x, centroid_y = centroids[largest]

            # Score: prefer centered (close to image center) and medium-sized (10-70% of image)
            dist_from_center = np.sqrt((centroid_x - cx) ** 2 + (centroid_y - cy) ** 2)
            center_score = 1.0 / (1.0 + dist_from_center / (w * 0.3))

            area_ratio = area / img_area
            if area_ratio < 0.05 or area_ratio > 0.85:
                size_score = 0.1  # too small or too big
            else:
                size_score = 1.0 - abs(area_ratio - 0.3)  # prefer ~30% of image

            return center_score * size_score, binary

        # Try both orientations
        score_normal, binary_normal = _score_orientation(gray)
        score_inverted, binary_inverted = _score_orientation(255 - gray)

        binary = binary_normal if score_normal >= score_inverted else binary_inverted

        # Final: silhouette=black(0), background=white(255)
        clean = 255 - binary
        return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB))

    def update_pipeline(self, pipe):
        """Replace the internal pipeline (used during RL training)."""
        self._pipe = pipe


class ControlNetImageGenerator:
    """SD 1.5 + ControlNet for road-map-conditioned silhouette generation.

    Takes a conditioning image (white roads on black) and generates
    silhouettes whose edges are biased toward the road grid pattern.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11p_sd15_canny",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        image_size: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 8.0,
        controlnet_scale: float = 0.5,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.prompt_template = prompt_template
        self.negative_prompt = negative_prompt
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is not None:
            return
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
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
        conditioning_image: Image,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate a silhouette conditioned on a road map image.

        Args:
            concept: Text concept (e.g., "horse").
            conditioning_image: Road map image (white roads on black, 512x512).
            seed: Random seed.

        Returns:
            Binary silhouette image (black shape on white).
        """
        concept_text = CONCEPT_OVERRIDES.get(concept, concept)
        prompt = self.prompt_template.format(concept=concept_text)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure conditioning image is RGB and correct size
        cond = conditioning_image.convert("RGB").resize(
            (self.image_size, self.image_size)
        )

        result = self.pipe(
            prompt,
            negative_prompt=self.negative_prompt,
            image=cond,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_scale,
            generator=generator,
        )
        raw = result.images[0]
        self._last_raw = raw
        return ImageGenerator._force_binary(raw)
