#!/usr/bin/env python3
"""Inference entrypoint for Route Sculptor."""

import sys

import click


@click.command()
@click.option("--gradio", is_flag=True, help="Launch Gradio web UI instead of CLI")
@click.option("--concept", default=None, help="Text concept for CLI mode")
@click.option("--output", default="output.gpx", help="Output GPX file path")
@click.option("--bbox", default=None, help="Bounding box as 'min_lat,min_lon,max_lat,max_lon'")
@click.option("--checkpoint", default=None, help="Path to trained model checkpoint")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--config", default="configs/default.yaml", help="Config file path")
@click.option("--preview", is_flag=True, help="Save preview images alongside GPX")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--port", default=7860, type=int, help="Gradio server port")
def main(
    gradio: bool,
    concept: str | None,
    output: str,
    bbox: str | None,
    checkpoint: str | None,
    seed: int | None,
    config: str,
    preview: bool,
    verbose: bool,
    port: int,
):
    """Generate GPX bike routes from concepts, or launch the web UI."""
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if gradio:
        from src.inference.gradio_app import launch
        launch(config_path=config, port=port)
    else:
        if not concept:
            click.echo("Error: --concept is required in CLI mode.", err=True)
            raise SystemExit(1)

        # Delegate to the CLI module
        from src.inference.cli import main as cli_main

        # Invoke with the same args
        ctx = click.Context(cli_main)
        ctx.invoke(
            cli_main,
            concept=concept,
            output=output,
            bbox=bbox,
            checkpoint=checkpoint,
            seed=seed,
            config=config,
            preview=preview,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
