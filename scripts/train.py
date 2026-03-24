#!/usr/bin/env python3
"""Training entrypoint for Route Sculptor DDPO training."""

import logging
import sys

import click


@click.command()
@click.option("--config", default="configs/default.yaml", help="Config file path")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(config: str, verbose: bool):
    """Train Route Sculptor with DDPO."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from src.training.ddpo_trainer import train

    train(config)


if __name__ == "__main__":
    main()
