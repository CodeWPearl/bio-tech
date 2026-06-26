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
from src.evaluation.external_tools import run_external_comparison
from src.evaluation.metrics import (
    classification_report_df,
    compute_all_metrics,
    compute_ci,
    get_confusion_matrix,
)
from src.models.full_model import PathogenicityPredictor
from src.training.lightning_module import PathogenicityLightningModule
from src.uncertainty.calibration import (
    TemperatureScaling,
    apply_calibration,
    compute_ece,
    compute_reliability_diagram,
)
from src.uncertainty.mc_dropout import MCDropoutPredictor
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
    parser.add_argument(
        "--dbnsfp-path",
        default=None,
        help="Path to dbNSFP scores file (TSV/CSV) for external tool comparison.",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip external tool comparison (SIFT, PolyPhen-2, CADD, REVEL).",
    )
    parser.add_argument(
        "--skip-uncertainty",
        action="store_true",
        help="Skip MC Dropout uncertainty estimation (faster).",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=50,
        help="Number of MC Dropout forward passes.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.1,
        help="Epistemic uncertainty threshold for flagging manual review.",
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


def _collect_uncertainty(
    model: PathogenicityPredictor,
    datamodule: PathogenicityDataModule,
    n_forward_passes: int = 50,
    expand_mask_fn: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout uncertainty estimation on the test set.

    Args:
        model: The trained model.
        datamodule: The data module with test data set up.
        n_forward_passes: Number of stochastic forward passes.
        expand_mask_fn: Function to expand the modality mask.

    Returns:
        Tuple of ``(epistemic_uncertainty, predictive_entropy, mean_probs)``.
    """
    mc_predictor = MCDropoutPredictor(model, n_forward_passes=n_forward_passes)
    device = next(model.parameters()).device

    all_uncertainty: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_mean_probs: list[np.ndarray] = []

    test_loader = datamodule.test_dataloader()

    for batch in test_loader:
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if expand_mask_fn is not None:
            batch_device = expand_mask_fn(batch_device)

        result = mc_predictor.predict_with_uncertainty(batch_device)
        all_uncertainty.append(result["epistemic_uncertainty"].cpu().numpy())
        all_entropy.append(result["predictive_entropy"].cpu().numpy())
        all_mean_probs.append(result["mean_probs"].cpu().numpy())

    return (
        np.concatenate(all_uncertainty),
        np.concatenate(all_entropy),
        np.concatenate(all_mean_probs),
    )


def _collect_logits(
    module: PathogenicityLightningModule,
    datamodule: PathogenicityDataModule,
    split: str = "val",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and labels from a data split for calibration.

    Args:
        module: The Lightning module.
        datamodule: Data module with data set up.
        split: Which split (``"val"`` or ``"test"``).

    Returns:
        Tuple of ``(logits, labels)`` numpy arrays.
    """
    module.eval()
    device = next(module.parameters()).device

    loader_fn = getattr(datamodule, f"{split}_dataloader")
    loader = loader_fn()

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"]
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_device = module._expand_modality_mask(batch_device)
            outputs = module.model(batch_device)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


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
    uncertainty_results: dict[str, Any] | None = None,
    external_df: pd.DataFrame | None = None,
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
        uncertainty_results: Optional uncertainty estimation results.
        external_df: Optional external tool comparison DataFrame.
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

    if external_df is not None and len(external_df) > 0:
        external_df.to_csv(output_dir / "external_comparison.csv", index=False)
        logger.info(
            "Saved external comparison to %s",
            output_dir / "external_comparison.csv",
        )

    if uncertainty_results is not None:
        with (output_dir / "uncertainty_results.json").open("w", encoding="utf-8") as fh:
            json.dump(uncertainty_results, fh, indent=2, default=str)
        logger.info("Saved uncertainty results to %s", output_dir / "uncertainty_results.json")

        if "predictions_df" in uncertainty_results:
            pred_df = pd.DataFrame(uncertainty_results["predictions_df"])
            pred_df.to_csv(
                output_dir / "uncertainty_augmented_predictions.csv", index=False,
            )
            logger.info(
                "Saved uncertainty-augmented predictions to %s",
                output_dir / "uncertainty_augmented_predictions.csv",
            )


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

    external_df = None
    if not args.skip_external:
        dbnsfp_path = (
            Path(args.dbnsfp_path)
            if args.dbnsfp_path
            else Path.cwd() / "data" / "processed" / "dbnsfp_scores.tsv"
        )
        if dbnsfp_path.is_file():
            log.info("Running external tool comparison...")
            try:
                external_df = run_external_comparison(
                    dbnsfp_path=dbnsfp_path,
                    y_true_4class=y_true,
                    y_pred_4class=y_pred,
                    y_prob_4class=y_prob,
                    output_dir=output_dir,
                )
            except Exception:
                log.exception("External tool comparison failed")
        else:
            log.warning(
                "dbNSFP scores not found at %s; skipping external comparison.",
                dbnsfp_path,
            )

    uncertainty_results: dict[str, Any] | None = None
    if not args.skip_uncertainty:
        log.info("Running MC Dropout uncertainty estimation (%d passes)...", args.mc_passes)
        try:
            epistemic_unc, pred_entropy, mc_probs = _collect_uncertainty(
                module.model,
                datamodule,
                n_forward_passes=args.mc_passes,
                expand_mask_fn=module._expand_modality_mask,
            )

            ece_before = compute_ece(y_true, y_prob, n_bins=15)

            log.info("Fitting temperature scaling on validation set...")
            val_logits, val_labels = _collect_logits(module, datamodule, split="val")
            scaler = TemperatureScaling()
            val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
            val_labels_t = torch.tensor(val_labels, dtype=torch.long)
            optimal_temp = scaler.optimize_temperature(val_logits_t, val_labels_t)

            test_logits_t = torch.tensor(
                np.concatenate([
                    module.model(
                        {k: v.to(next(module.parameters()).device)
                         if isinstance(v, torch.Tensor) else v
                         for k, v in module._expand_modality_mask(batch).items()}
                    )["logits"].cpu().detach().numpy()
                    for batch in datamodule.test_dataloader()
                ]),
                dtype=torch.float32,
            )
            calibrated_probs = scaler(test_logits_t).detach().numpy()
            ece_after = compute_ece(y_true, calibrated_probs, n_bins=15)

            high_unc_mask = epistemic_unc > args.uncertainty_threshold
            n_flagged = int(high_unc_mask.sum())

            per_class_stats: dict[str, dict[str, float]] = {}
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                cls_mask = y_true == cls_idx
                if cls_mask.sum() > 0:
                    per_class_stats[cls_name] = {
                        "mean_uncertainty": float(epistemic_unc[cls_mask].mean()),
                        "std_uncertainty": float(epistemic_unc[cls_mask].std()),
                        "mean_entropy": float(pred_entropy[cls_mask].mean()),
                        "std_entropy": float(pred_entropy[cls_mask].std()),
                    }

            predictions_df_data: dict[str, list[Any]] = {
                "true_label": y_true.tolist(),
                "predicted_label": y_pred.tolist(),
                "epistemic_uncertainty": epistemic_unc.tolist(),
                "predictive_entropy": pred_entropy.tolist(),
                "flagged_for_review": high_unc_mask.tolist(),
            }
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                predictions_df_data[f"prob_{cls_name}"] = mc_probs[:, cls_idx].tolist()
                predictions_df_data[f"calibrated_prob_{cls_name}"] = (
                    calibrated_probs[:, cls_idx].tolist()
                )

            uncertainty_results = {
                "ece_before_calibration": ece_before,
                "ece_after_calibration": ece_after,
                "optimal_temperature": optimal_temp,
                "mc_dropout_passes": args.mc_passes,
                "uncertainty_threshold": args.uncertainty_threshold,
                "n_flagged_for_review": n_flagged,
                "pct_flagged_for_review": float(n_flagged / len(y_true) * 100),
                "per_class_uncertainty": per_class_stats,
                "predictions_df": predictions_df_data,
            }

            metrics["ece_before_calibration"] = ece_before
            metrics["ece_after_calibration"] = ece_after
            metrics["optimal_temperature"] = optimal_temp

            log.info("ECE before calibration: %.4f", ece_before)
            log.info("ECE after calibration:  %.4f", ece_after)
            log.info("Optimal temperature:    %.4f", optimal_temp)
            log.info(
                "Flagged for manual review: %d / %d (%.1f%%)",
                n_flagged, len(y_true), n_flagged / len(y_true) * 100,
            )
            for cls_name, stats in per_class_stats.items():
                log.info(
                    "  %-20s: unc=%.4f +/- %.4f  ent=%.4f +/- %.4f",
                    cls_name,
                    stats["mean_uncertainty"],
                    stats["std_uncertainty"],
                    stats["mean_entropy"],
                    stats["std_entropy"],
                )
        except Exception:
            log.exception("Uncertainty estimation failed")

    _print_summary(metrics, ci_results, cm)

    _save_results(
        output_dir, metrics, ci_results, cm, report_df, baseline_df,
        bio_results, uncertainty_results, external_df,
    )

    log.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
