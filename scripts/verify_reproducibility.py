"""Verify reproducibility by training the model twice with the same config and seed.

Trains the model two independent times with identical configuration and random
seed, then compares the outputs.  On CPU the predictions should be bit-for-bit
identical.  On GPU, small non-determinism from cuDNN/CUDA atomics may introduce
tiny differences — the script reports the max absolute difference so the user can
judge whether reproducibility is acceptable.

Usage::

    python scripts/verify_reproducibility.py --config configs/default.yaml
    python scripts/verify_reproducibility.py --config configs/default.yaml --max-epochs 5
    python scripts/verify_reproducibility.py --config configs/default.yaml --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule import PathogenicityDataModule
from src.models.full_model import PathogenicityPredictor
from src.training.lightning_module import PathogenicityLightningModule
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify training reproducibility by comparing two identical runs.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Number of epochs per run (default: 5, keep small for speed).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Device to train on (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory to save the reproducibility report.",
    )
    return parser.parse_args()


def train_once(
    cfg: Any,
    run_id: int,
    max_epochs: int,
    device: str,
    checkpoint_dir: Path,
) -> tuple[dict[str, float], list[np.ndarray]]:
    """Train the model once and return test metrics and predictions.

    Args:
        cfg: Configuration object.
        run_id: Integer identifier for this run (1 or 2).
        max_epochs: Maximum number of training epochs.
        device: Accelerator to use (cpu, gpu, auto).
        checkpoint_dir: Directory for saving checkpoints.

    Returns:
        Tuple of (test_metrics_dict, list_of_prediction_arrays).
    """
    logger.info("=" * 60)
    logger.info("Starting Run %d", run_id)
    logger.info("=" * 60)

    seed_everything(cfg.data.random_seed)

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

    run_ckpt_dir = checkpoint_dir / f"run_{run_id}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(run_ckpt_dir),
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            filename=f"repro-run{run_id}-{{epoch:02d}}-{{val_auroc:.4f}}",
        ),
    ]

    accelerator = device if device != "auto" else "auto"
    devices_val: int | str = "auto"
    if device == "cpu":
        accelerator = "cpu"
        devices_val = 1
    elif device == "gpu":
        accelerator = "gpu"
        devices_val = 1

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices_val,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
        logger=False,
    )

    trainer.fit(lightning_module, datamodule=datamodule)

    datamodule.setup("test")
    test_results = trainer.test(
        lightning_module, datamodule=datamodule, ckpt_path="best",
    )
    test_metrics = test_results[0] if test_results else {}

    lightning_module.eval()
    all_predictions: list[np.ndarray] = []
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            output = lightning_module.model(batch)
            probs = output["probabilities"].cpu().numpy()
            all_predictions.append(probs)

    logger.info("Run %d complete. Test metrics: %s", run_id, test_metrics)
    return test_metrics, all_predictions


def compare_runs(
    metrics_1: dict[str, float],
    metrics_2: dict[str, float],
    preds_1: list[np.ndarray],
    preds_2: list[np.ndarray],
) -> dict[str, Any]:
    """Compare two training runs for reproducibility.

    Args:
        metrics_1: Test metrics from run 1.
        metrics_2: Test metrics from run 2.
        preds_1: Prediction arrays from run 1.
        preds_2: Prediction arrays from run 2.

    Returns:
        Dictionary with comparison results.
    """
    all_preds_1 = np.concatenate(preds_1, axis=0)
    all_preds_2 = np.concatenate(preds_2, axis=0)

    abs_diff = np.abs(all_preds_1 - all_preds_2)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    median_abs_diff = float(np.median(abs_diff))

    classes_1 = np.argmax(all_preds_1, axis=1)
    classes_2 = np.argmax(all_preds_2, axis=1)
    class_agreement = float(np.mean(classes_1 == classes_2))

    metric_diffs = {}
    for key in sorted(set(metrics_1.keys()) | set(metrics_2.keys())):
        v1 = metrics_1.get(key, float("nan"))
        v2 = metrics_2.get(key, float("nan"))
        metric_diffs[key] = {
            "run_1": v1,
            "run_2": v2,
            "abs_diff": abs(v1 - v2),
        }

    is_exact = max_abs_diff == 0.0
    is_acceptable = max_abs_diff < 1e-4

    report = {
        "prediction_comparison": {
            "num_samples": int(all_preds_1.shape[0]),
            "max_absolute_difference": max_abs_diff,
            "mean_absolute_difference": mean_abs_diff,
            "median_absolute_difference": median_abs_diff,
            "class_agreement_rate": class_agreement,
            "exact_match": is_exact,
        },
        "metric_comparison": metric_diffs,
        "verdict": {
            "reproducible": is_acceptable,
            "exact": is_exact,
            "note": (
                "Predictions are bit-for-bit identical."
                if is_exact
                else (
                    f"Predictions differ by at most {max_abs_diff:.2e}. "
                    "This is within acceptable GPU non-determinism."
                    if is_acceptable
                    else (
                        f"Predictions differ by {max_abs_diff:.2e}, which "
                        "exceeds the acceptable threshold of 1e-4. "
                        "Check seed and deterministic settings."
                    )
                )
            ),
        },
    }
    return report


def main() -> None:
    """Run the reproducibility verification."""
    args = parse_args()
    setup_logging(level="INFO", name=__name__)
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("results") / "checkpoints" / "reproducibility"

    logger.info("Reproducibility Verification")
    logger.info("Config: %s", args.config)
    logger.info("Max epochs: %d", args.max_epochs)
    logger.info("Device: %s", args.device)

    metrics_1, preds_1 = train_once(
        cfg, run_id=1, max_epochs=args.max_epochs,
        device=args.device, checkpoint_dir=checkpoint_dir,
    )

    metrics_2, preds_2 = train_once(
        cfg, run_id=2, max_epochs=args.max_epochs,
        device=args.device, checkpoint_dir=checkpoint_dir,
    )

    report = compare_runs(metrics_1, metrics_2, preds_1, preds_2)

    import json
    report_path = output_dir / "reproducibility_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("REPRODUCIBILITY REPORT")
    logger.info("=" * 60)

    pc = report["prediction_comparison"]
    logger.info("Samples compared:           %d", pc["num_samples"])
    logger.info("Max absolute difference:    %.2e", pc["max_absolute_difference"])
    logger.info("Mean absolute difference:   %.2e", pc["mean_absolute_difference"])
    logger.info("Median absolute difference: %.2e", pc["median_absolute_difference"])
    logger.info("Class agreement rate:       %.4f", pc["class_agreement_rate"])
    logger.info("Exact match:                %s", pc["exact_match"])
    logger.info("")

    verdict = report["verdict"]
    if verdict["exact"]:
        logger.info("VERDICT: EXACT REPRODUCIBILITY — predictions are bit-for-bit identical")
    elif verdict["reproducible"]:
        logger.info("VERDICT: ACCEPTABLE — within GPU non-determinism tolerance")
    else:
        logger.warning("VERDICT: NOT REPRODUCIBLE — differences exceed tolerance")

    logger.info("")
    logger.info("Full report saved to: %s", report_path)


if __name__ == "__main__":
    main()
