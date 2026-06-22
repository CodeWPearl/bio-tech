"""CLI entry point for evaluating a trained pathogenicity model.

Usage:
    evaluate --checkpoint results/checkpoints/best_model.ckpt [--config ...]

Scaffold stub: argument parsing and logging are wired up; metric computation is
implemented in a later step (see ``src/evaluation``).
"""

from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a pathogenicity model.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint to evaluate.",
    )
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
    """Console-script entry point for ``evaluate``."""
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    logger = setup_logging(level="INFO", name=__name__)
    logger.info("Evaluating checkpoint '%s' (experiment '%s')", args.checkpoint, cfg.experiment.name)
    logger.warning("Evaluation pipeline not yet implemented (scaffold stub).")


if __name__ == "__main__":
    main()
