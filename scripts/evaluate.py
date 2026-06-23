"""CLI entry point for evaluating a trained pathogenicity model.

Loads a checkpoint, runs inference on the test set, computes comprehensive
metrics with confidence intervals, runs baseline comparisons, performs
biological validation, and saves all results to ``results/tables/``.

Usage::

    python scripts/evaluate.py --checkpoint results/checkpoints/best_model.ckpt
    python scripts/evaluate.py --checkpoint best.ckpt --config configs/default.yaml
    python scripts/evaluate.py --checkpoint best.ckpt --skip-baselines
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.datamodule import PathogenicityDataModule
from src.evaluation.benchmarks import run_baselines
from src.evaluation.biological_validation import run_biological_validation
from src.evaluation.metrics import (
    classification_report_df,
    compute_all_metrics,
    compute_ci,
    get_confusion_matrix,
)
from src.models.full_model import PathogenicityPredictor
from src.training.lightning_module import PathogenicityLightningModule
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "Pathogenic",
    "Likely Pathogenic",
    "Benign",
    "Likely Benign",
]


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
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for result CSVs and JSONs.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline model comparison (faster).",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap confidence intervals (faster).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI computation.",
    )
    parser.add_argument(
        "--cosmic-path",
        default=None,
        help="Path to COSMIC Cancer Gene Census CSV (optional).",
    )
    return parser.parse_args()


def _collect_predictions(
    module: PathogenicityLightningModule,
    datamodule: PathogenicityDataModule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on the test set and collect predictions.

    Args:
        module: The Lightning module with loaded weights.
        datamodule: The data module with test data set up.

    Returns:
        Tuple of ``(y_true, y_pred, y_prob)`` numpy arrays.
    """
    module.eval()
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

    return y_true, y_pred, y_prob


def _flatten_features(
    datamodule: PathogenicityDataModule, split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened feature vectors for baseline models.

    Args:
        datamodule: The data module with data set up.
        split: Which split to extract (``"train"``, ``"val"``, ``"test"``).

    Returns:
        Tuple of ``(X, y)`` numpy arrays.
    """
    dataset = getattr(datamodule, f"{split}_dataset")
    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' not set up.")

    features: list[np.ndarray] = []
    labels: list[int] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        feature_parts = [
            sample["mutation"].numpy(),
            sample["expression"].numpy(),
            sample["methylation"].numpy(),
            sample["cnv"].numpy(),
            sample["clinical"].numpy(),
        ]
        features.append(np.concatenate(feature_parts))
        labels.append(int(sample["label"].item()))

    return np.array(features), np.array(labels)


def _save_results(
    output_dir: Path,
    metrics: dict[str, Any],
    ci_results: dict[str, tuple[float, float, float]] | None,
    cm: np.ndarray,
    report_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    bio_results: dict[str, Any] | None,
) -> None:
    """Save all evaluation results to disk.

    Args:
        output_dir: Directory for output files.
        metrics: Dict of all computed metrics.
        ci_results: Optional bootstrap CI results.
        cm: Confusion matrix.
        report_df: Classification report DataFrame.
        baseline_df: Optional baseline comparison DataFrame.
        bio_results: Optional biological validation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Saved test metrics to %s", output_dir / "test_metrics.json")

    if ci_results is not None:
        ci_df = pd.DataFrame(
            {k: {"lower": v[0], "mean": v[1], "upper": v[2]} for k, v in ci_results.items()}
        ).T
        ci_df.to_csv(output_dir / "confidence_intervals.csv")
        logger.info("Saved CIs to %s", output_dir / "confidence_intervals.csv")

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    logger.info("Saved confusion matrix to %s", output_dir / "confusion_matrix.csv")

    report_df.to_csv(output_dir / "classification_report.csv")
    logger.info("Saved classification report to %s", output_dir / "classification_report.csv")

    if baseline_df is not None and len(baseline_df) > 0:
        baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
        logger.info("Saved baseline comparison to %s", output_dir / "baseline_comparison.csv")

    if bio_results is not None:
        with (output_dir / "biological_validation.json").open("w", encoding="utf-8") as fh:
            json.dump(bio_results, fh, indent=2, default=str)
        logger.info("Saved biological validation to %s", output_dir / "biological_validation.json")


def _print_summary(
    metrics: dict[str, Any],
    ci_results: dict[str, tuple[float, float, float]] | None,
    cm: np.ndarray,
) -> None:
    """Print a formatted evaluation summary to the logger.

    Args:
        metrics: Dict of all computed metrics.
        ci_results: Optional bootstrap CI results.
        cm: Confusion matrix.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    key_metrics = [
        ("Accuracy", "accuracy"),
        ("F1 (macro)", "f1_macro"),
        ("F1 (weighted)", "f1_weighted"),
        ("ROC-AUC (macro)", "roc_auc_macro"),
        ("PR-AUC (macro)", "pr_auc_macro"),
        ("MCC", "mcc"),
        ("Cohen's Kappa", "cohen_kappa"),
        ("Top-1 Accuracy", "top_1_accuracy"),
        ("Top-2 Accuracy", "top_2_accuracy"),
        ("ECE", "ece"),
    ]

    for display_name, key in key_metrics:
        value = metrics.get(key, float("nan"))
        if ci_results and key in ci_results:
            lower, mean, upper = ci_results[key]
            logger.info(
                "  %-20s: %.4f  [95%% CI: %.4f - %.4f]",
                display_name, value, lower, upper,
            )
        else:
            logger.info("  %-20s: %.4f", display_name, value)

    logger.info("-" * 60)
    logger.info("Per-class metrics:")
    for i, name in enumerate(CLASS_NAMES):
        p = metrics.get(f"precision_{name}", 0)
        r = metrics.get(f"recall_{name}", 0)
        f = metrics.get(f"f1_{name}", 0)
        logger.info("  %-20s: P=%.4f  R=%.4f  F1=%.4f", name, p, r, f)

    logger.info("-" * 60)
    logger.info("Confusion Matrix:")
    header = "           " + "  ".join(f"{n[:8]:>8s}" for n in CLASS_NAMES)
    logger.info(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name[:8]:>8s}" + "  ".join(f"{cm[i, j]:>8d}" for j in range(len(CLASS_NAMES)))
        logger.info(row)

    logger.info("=" * 60)


def main() -> None:
    """Evaluate a trained pathogenicity model end-to-end."""
    args = parse_args()
    log = setup_logging(level="INFO", name=__name__)
    cfg = load_config(args.config, overrides=args.override)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        log.error("Checkpoint not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = PathogenicityPredictor.from_config(cfg)
    module = PathogenicityLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=cfg,
        model=model,
    )
    module.eval()

    log.info("Setting up data module...")
    datamodule = PathogenicityDataModule(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    log.info("Collecting predictions on test set...")
    y_true, y_pred, y_prob = _collect_predictions(module, datamodule)
    log.info("Test set size: %d samples", len(y_true))

    log.info("Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    cm = get_confusion_matrix(y_true, y_pred, num_classes=cfg.model.num_classes)
    report_df = classification_report_df(y_true, y_pred, class_names=CLASS_NAMES)

    ci_results = None
    if not args.skip_bootstrap:
        log.info("Computing bootstrap confidence intervals (%d resamples)...", args.n_bootstrap)
        ci_results = compute_ci(y_true, y_pred, y_prob, n_bootstrap=args.n_bootstrap)

    baseline_df = None
    if not args.skip_baselines:
        log.info("Running baseline comparisons...")
        try:
            x_train, y_train = _flatten_features(datamodule, "train")
            x_test, y_test = _flatten_features(datamodule, "test")
            baseline_df = run_baselines(
                x_train, y_train, x_test, y_test, our_model_metrics=metrics,
            )
        except Exception:
            log.exception("Baseline comparison failed")

    bio_results = None
    if datamodule.test_dataset is not None:
        log.info("Running biological validation...")
        try:
            import pickle

            cache_path = Path.cwd() / "data" / "processed" / "feature_cache.pkl"
            if cache_path.is_file():
                with cache_path.open("rb") as fh:
                    cached = pickle.load(fh)  # noqa: S301
                test_data = cached.get("test", {})

                gene_symbols = test_data.get("gene_symbols")
                review_status = test_data.get("review_status")

                if gene_symbols is not None:
                    cosmic_path = Path(args.cosmic_path) if args.cosmic_path else None
                    bio_results = run_biological_validation(
                        gene_symbols=gene_symbols,
                        y_true=y_true,
                        y_pred=y_pred,
                        y_prob=y_prob,
                        review_status=review_status,
                        cosmic_path=cosmic_path,
                    )
                else:
                    log.warning(
                        "Gene symbols not in feature cache; skipping biological validation."
                    )
            else:
                log.warning("Feature cache not found; skipping biological validation.")
        except Exception:
            log.exception("Biological validation failed")

    _print_summary(metrics, ci_results, cm)

    _save_results(
        output_dir, metrics, ci_results, cm, report_df, baseline_df, bio_results,
    )

    log.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
