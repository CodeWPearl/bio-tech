"""CLI entry point for running inference on new mutations.

Usage:
    inference --checkpoint results/checkpoints/best_model.ckpt --input variants.tsv

Scaffold stub: argument parsing and logging are wired up; prediction is
implemented in a later step.
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Predict pathogenicity for new variants.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input file of variants to score.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Console-script entry point for ``inference``."""
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(level="INFO", name=__name__)
    logger.info(
        "Running inference with checkpoint '%s' on '%s' (%d classes)",
        args.checkpoint,
        args.input,
        cfg.model.num_classes,
    )
    logger.warning("Inference pipeline not yet implemented (scaffold stub).")


if __name__ == "__main__":
    main()
