"""CLIP-based text-image similarity scoring."""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_PROMPT_TEMPLATE = "a bike route shaped like a {concept}"


class CLIPScorer:
    """Computes CLIP similarity between a concept text and a route image."""

    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    @torch.no_grad()
    def score(self, image: Image.Image, concept: str) -> float:
        """Compute CLIP similarity between an image and a concept.

        Args:
            image: RGB PIL Image (e.g., map overlay render of the route).
            concept: The text concept (e.g., "horse").

        Returns:
            Cosine similarity score in [0, 1].
        """
        text = self.prompt_template.format(concept=concept)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Normalize embeddings and compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).item()

        # Map from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    @torch.no_grad()
    def score_batch(
        self, images: list[Image.Image], concepts: list[str]
    ) -> list[float]:
        """Score a batch of images against their concepts.

        Args:
            images: List of RGB PIL Images.
            concepts: List of concept strings (same length as images).

        Returns:
            List of similarity scores in [0, 1].
        """
        texts = [self.prompt_template.format(concept=c) for c in concepts]

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        # Diagonal similarities (each image with its own text)
        similarities = (image_embeds * text_embeds).sum(dim=-1)

        return [((s.item() + 1.0) / 2.0) for s in similarities]
