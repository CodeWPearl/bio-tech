"""Ablation study runner for the pathogenicity prediction model.

Iterates through ablation config variants, trains each model with the
same seed and data splits, evaluates on the test set, and produces a
comparison table showing each component's contribution.

Usage::

    python scripts/run_ablation.py
    python scripts/run_ablation.py --configs-dir configs/ablation
    python scripts/run_ablation.py --max-epochs 50 --skip-training
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule import PathogenicityDataModule
from src.evaluation.metrics import compute_all_metrics
from src.models.full_model import PathogenicityPredictor
from src.training.callbacks import EarlyStoppingWithPatience
from src.training.lightning_module import PathogenicityLightningModule
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)

ABLATION_DISPLAY_NAMES: dict[str, str] = {
    "default": "Full Model",
    "no_mutation": "No Mutation",
    "no_expression": "No Expression",
    "no_methylation": "No Methylation",
    "no_cnv": "No CNV",
    "no_attention": "No Attention (Early Fusion)",
    "single_mutation_only": "Mutation Only",
    "single_expression_only": "Expression Only",
    "no_focal_loss": "No Focal Loss (CE)",
}

KEY_METRICS: list[tuple[str, str]] = [
    ("Accuracy", "accuracy"),
    ("F1-Macro", "f1_macro"),
    ("AUROC", "roc_auc_macro"),
    ("PR-AUC", "pr_auc_macro"),
    ("MCC", "mcc"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation study.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run ablation study for pathogenicity prediction.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/default.yaml",
        help="Path to the full-model (baseline) config.",
    )
    parser.add_argument(
        "--configs-dir",
        default="configs/ablation",
        help="Directory containing ablation YAML configs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for output CSV and JSON files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/ablation_checkpoints",
        help="Directory for per-ablation model checkpoints.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for all ablation runs.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; load existing checkpoints and evaluate.",
    )
    return parser.parse_args()


def _train_and_evaluate(
    config_path: Path,
    config_name: str,
    checkpoint_dir: Path,
    max_epochs_override: int | None,
    gpus: int | None,
    skip_training: bool,
) -> dict[str, Any]:
    """Train a model from a config and evaluate on the test set.

    Args:
        config_path: Path to the YAML config file.
        config_name: Short name for this ablation variant.
        checkpoint_dir: Where to save/load checkpoints.
        max_epochs_override: Optional epoch limit override.
        gpus: Number of GPUs.
        skip_training: If True, load checkpoint instead of training.

    Returns:
        Dict of computed metrics on the test set.
    """
    cfg = load_config(config_path)
    seed = int(cfg.data.random_seed)
    seed_everything(seed)

    variant_ckpt_dir = checkpoint_dir / config_name
    variant_ckpt_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PathogenicityDataModule(cfg)
    datamodule.setup("fit")

    class_weights = None
    if datamodule._class_weights is not None:
        class_weights = torch.tensor(
            datamodule._class_weights, dtype=torch.float32,
        )

    model = PathogenicityPredictor.from_config(cfg)
    lightning_module = PathogenicityLightningModule(
        config=cfg, model=model, class_weights=class_weights,
    )

    best_ckpt_path: Path | None = None

    if not skip_training:
        max_epochs = max_epochs_override or int(cfg.training.max_epochs)

        callbacks: list[pl.Callback] = [
            ModelCheckpoint(
                dirpath=str(variant_ckpt_dir),
                monitor="val_auroc",
                mode="max",
                save_top_k=1,
                filename=f"{config_name}-{{epoch:02d}}-{{val_auroc:.4f}}",
            ),
            EarlyStoppingWithPatience(
                patience=int(cfg.training.patience),
            ),
        ]

        accelerator = "auto"
        devices: int | str = "auto"
        if gpus is not None:
            accelerator = "gpu" if gpus > 0 else "cpu"
            devices = gpus if gpus > 0 else 1

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=False,
            deterministic=True,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        logger.info(
            "Training ablation variant '%s' for up to %d epochs...",
            config_name, max_epochs,
        )
        trainer.fit(lightning_module, datamodule=datamodule)

        ckpt_callback = callbacks[0]
        if isinstance(ckpt_callback, ModelCheckpoint) and ckpt_callback.best_model_path:
            best_ckpt_path = Path(ckpt_callback.best_model_path)
    else:
        ckpts = sorted(variant_ckpt_dir.glob("*.ckpt"))
        if ckpts:
            best_ckpt_path = ckpts[-1]
        else:
            logger.warning(
                "No checkpoint found for '%s'; training from scratch.",
                config_name,
            )
            return _train_and_evaluate(
                config_path, config_name, checkpoint_dir,
                max_epochs_override, gpus, skip_training=False,
            )

    if best_ckpt_path is not None and best_ckpt_path.is_file():
        logger.info("Loading best checkpoint: %s", best_ckpt_path)
        loaded_module = PathogenicityLightningModule.load_from_checkpoint(
            str(best_ckpt_path), config=cfg, model=model,
        )
    else:
        loaded_module = lightning_module

    loaded_module.eval()
    datamodule.setup("test")

    return _evaluate_model(loaded_module, datamodule)


def _evaluate_model(
    module: PathogenicityLightningModule,
    datamodule: PathogenicityDataModule,
) -> dict[str, Any]:
    """Collect predictions and compute all metrics.

    Args:
        module: The trained Lightning module.
        datamodule: Data module with test data set up.

    Returns:
        Dict of all computed metrics.
    """
    device = next(module.parameters()).device

    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"]

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_device = module._expand_modality_mask(batch_device)

            outputs = module.model(batch_device)

            all_labels.append(labels.numpy())
            all_preds.append(outputs["predicted_class"].cpu().numpy())
            all_probs.append(outputs["probabilities"].cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return compute_all_metrics(y_true, y_pred, y_prob)


def _build_comparison_table(
    all_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build the ablation comparison DataFrame.

    Args:
        all_results: Mapping from config name to metrics dict.

    Returns:
        DataFrame with one row per ablation variant and delta columns.
    """
    rows: list[dict[str, Any]] = []

    full_metrics = all_results.get("default")

    for config_name, metrics in all_results.items():
        display_name = ABLATION_DISPLAY_NAMES.get(config_name, config_name)

        row: dict[str, Any] = {"Configuration": display_name}
        for col_name, metric_key in KEY_METRICS:
            row[col_name] = metrics.get(metric_key, float("nan"))

        if full_metrics is not None and config_name != "default":
            deltas: list[float] = []
            for _, metric_key in KEY_METRICS:
                full_val = full_metrics.get(metric_key, 0.0)
                ablation_val = metrics.get(metric_key, 0.0)
                if full_val != 0.0:
                    delta_pct = ((ablation_val - full_val) / abs(full_val)) * 100
                else:
                    delta_pct = 0.0
                deltas.append(delta_pct)
            avg_delta = float(np.mean(deltas))
            row["Δ vs Full"] = f"{avg_delta:+.1f}%"
        else:
            row["Δ vs Full"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> None:
    """Print the ablation comparison table in formatted form.

    Args:
        df: The comparison DataFrame.
    """
    logger.info("")
    logger.info("=" * 90)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 90)

    header = (
        f"{'Configuration':<30s}"
        f"{'Accuracy':>10s}"
        f"{'F1-Macro':>10s}"
        f"{'AUROC':>10s}"
        f"{'PR-AUC':>10s}"
        f"{'MCC':>10s}"
        f"{chr(916)+' vs Full':>12s}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for _, row in df.iterrows():
        line = f"{row['Configuration']:<30s}"
        for col_name, _ in KEY_METRICS:
            val = row[col_name]
            line += f"{val:>10.4f}"
        line += f"{row[chr(916)+' vs Full']:>12s}"
        logger.info(line)

    logger.info("=" * 90)
    logger.info("")


def main() -> None:
    """Run the full ablation study."""
    args = parse_args()
    log = setup_logging(level="INFO", name=__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_config = Path(args.base_config)
    configs_dir = Path(args.configs_dir)

    configs: list[tuple[str, Path]] = [("default", base_config)]

    if configs_dir.is_dir():
        for yaml_file in sorted(configs_dir.glob("*.yaml")):
            config_name = yaml_file.stem
            configs.append((config_name, yaml_file))

    log.info(
        "Starting ablation study with %d configurations:", len(configs),
    )
    for name, path in configs:
        display = ABLATION_DISPLAY_NAMES.get(name, name)
        log.info("  - %s (%s)", display, path)

    all_results: dict[str, dict[str, Any]] = {}

    for i, (config_name, config_path) in enumerate(configs):
        display = ABLATION_DISPLAY_NAMES.get(config_name, config_name)
        log.info(
            "\n[%d/%d] Running: %s", i + 1, len(configs), display,
        )

        try:
            metrics = _train_and_evaluate(
                config_path=config_path,
                config_name=config_name,
                checkpoint_dir=checkpoint_dir,
                max_epochs_override=args.max_epochs,
                gpus=args.gpus,
                skip_training=args.skip_training,
            )
            all_results[config_name] = metrics
            log.info(
                "  Accuracy=%.4f  F1=%.4f  AUROC=%.4f  MCC=%.4f",
                metrics.get("accuracy", 0),
                metrics.get("f1_macro", 0),
                metrics.get("roc_auc_macro", 0),
                metrics.get("mcc", 0),
            )
        except Exception:
            log.exception("Failed on ablation variant '%s'", config_name)
            continue

    if not all_results:
        log.error("No ablation results collected. Exiting.")
        return

    comparison_df = _build_comparison_table(all_results)
    _print_table(comparison_df)

    csv_path = output_dir / "ablation_results.csv"
    comparison_df.to_csv(csv_path, index=False)
    log.info("Saved ablation results to %s", csv_path)

    json_path = output_dir / "ablation_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log.info("Saved raw metrics to %s", json_path)

    log.info("Ablation study complete.")


if __name__ == "__main__":
    main()
