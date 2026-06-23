"""CLI entry point for training a pathogenicity model.

Usage::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --experiment_name my_exp --gpus 1
    python scripts/train.py --config configs/default.yaml --override training.learning_rate=0.0005
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import PathogenicityDataModule
from src.models.full_model import PathogenicityPredictor
from src.training.callbacks import (
    EarlyStoppingWithPatience,
    GradientMonitor,
    MetricLogger,
)
from src.training.lightning_module import PathogenicityLightningModule
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a pathogenicity model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="MLflow experiment name (overrides config).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect).",
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
    """Train the pathogenicity prediction model end-to-end."""
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    log = setup_logging(level="INFO", name=__name__)
    seed_everything(cfg.data.random_seed)

    experiment_name = args.experiment_name or cfg.experiment.name
    log.info("Starting experiment: %s", experiment_name)

    # --- data ----------------------------------------------------------------
    datamodule = PathogenicityDataModule(cfg)
    datamodule.setup("fit")

    class_weights = None
    if datamodule._class_weights is not None:
        class_weights = torch.tensor(
            datamodule._class_weights, dtype=torch.float32,
        )
        log.info("Class weights: %s", class_weights.tolist())

    # --- model ---------------------------------------------------------------
    model = PathogenicityPredictor.from_config(cfg)
    log.info(model.summary())

    lightning_module = PathogenicityLightningModule(
        config=cfg, model=model, class_weights=class_weights,
    )

    # --- MLflow logging ------------------------------------------------------
    tracking_uri = cfg.experiment.tracking_uri
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    # --- callbacks -----------------------------------------------------------
    checkpoint_dir = Path("results") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_auroc",
            mode="max",
            save_top_k=3,
            filename="pathogenicity-{epoch:02d}-{val_auroc:.4f}",
        ),
        EarlyStoppingWithPatience(
            patience=int(cfg.training.patience),
        ),
        LearningRateMonitor(logging_interval="epoch"),
        GradientMonitor(log_every_n_steps=50),
        MetricLogger(),
    ]

    # --- trainer -------------------------------------------------------------
    accelerator = "auto"
    devices: int | str = "auto"
    if args.gpus is not None:
        accelerator = "gpu" if args.gpus > 0 else "cpu"
        devices = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=mlf_logger,
        deterministic=True,
        log_every_n_steps=10,
    )

    # --- train ---------------------------------------------------------------
    log.info("Starting training for up to %d epochs", cfg.training.max_epochs)
    trainer.fit(lightning_module, datamodule=datamodule)

    # --- test ----------------------------------------------------------------
    log.info("Running test evaluation with best checkpoint")
    datamodule.setup("test")
    test_results = trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")

    # --- save final metrics --------------------------------------------------
    results_dir = Path("results") / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    if test_results:
        metrics_path = results_dir / "final_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(test_results[0], fh, indent=2)
        log.info("Saved final metrics to %s", metrics_path)

        try:
            import mlflow

            for key, value in test_results[0].items():
                mlflow.log_metric(f"final_{key}", value)
        except ImportError:
            pass

    log.info("Training complete. Best checkpoint: %s", checkpoint_dir)


if __name__ == "__main__":
    main()
