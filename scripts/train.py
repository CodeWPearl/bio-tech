"""CLI entry point for training a pathogenicity model.

Usage:
    train --config configs/default.yaml [--override key.path=value ...]

This is a scaffold stub: it loads and validates the config so the pipeline is
wired end-to-end, while the actual training loop is implemented in a later step
(see ``src/training``).
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a pathogenicity model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="key.path=value",
        help="Optional config overrides applied after loading.",
    )
    return parser.parse_args()


def main() -> None:
    """Console-script entry point for ``train``."""
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    logger = setup_logging(level="INFO", name=__name__)
    seed_everything(cfg.data.random_seed)
    logger.info("Loaded experiment '%s'", cfg.experiment.name)
    logger.warning("Training pipeline not yet implemented (scaffold stub).")


if __name__ == "__main__":
    main()
