"""Concept prompts dataset for DDPO training."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_concepts(filepath: str | Path | None = None) -> list[str]:
    """Load concept words from a text file (one per line).

    Args:
        filepath: Path to the concepts file.

    Returns:
        List of concept strings.
    """
    if filepath is None:
        filepath = PROJECT_ROOT / "data" / "concepts.txt"
    filepath = Path(filepath)
    concepts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                concepts.append(line)
    return concepts


def build_prompts(
    concepts: list[str],
    template: str = "a simple black silhouette of a {concept} on a white background, minimal, clean outline, no background details",
) -> list[str]:
    """Build SD prompts from concepts using a template.

    Args:
        concepts: List of concept words.
        template: Prompt template with {concept} placeholder.

    Returns:
        List of formatted prompts.
    """
    return [template.format(concept=c) for c in concepts]
