"""Comprehensive evaluation metrics for pathogenicity classification.

Provides :func:`compute_all_metrics` for computing a full suite of
classification metrics, :func:`get_confusion_matrix` and
:func:`classification_report_df` for detailed breakdowns, and
:func:`compute_ci` for bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "Pathogenic",
    "Likely Pathogenic",
    "Benign",
    "Likely Benign",
]


def _top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Compute top-k accuracy.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        k: Number of top predictions to consider.

    Returns:
        Fraction of samples where the true label is among the top-k predictions.
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(correct.mean())


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: True labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        n_bins: Number of bins for calibration.

    Returns:
        ECE value (lower is better).
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        ece += (n_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def _pr_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro-averaged PR-AUC (one-vs-rest).

    Args:
        y_true: True labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.

    Returns:
        Macro-averaged area under the precision-recall curve.
    """
    num_classes = y_prob.shape[1]
    pr_aucs: list[float] = []

    for cls in range(num_classes):
        binary_true = (y_true == cls).astype(int)
        if binary_true.sum() == 0:
            continue
        precision_vals, recall_vals, _ = precision_recall_curve(
            binary_true, y_prob[:, cls],
        )
        pr_auc = float(np.trapezoid(precision_vals, recall_vals))
        pr_aucs.append(abs(pr_auc))

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute a comprehensive suite of classification metrics.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.

    Returns:
        Dict with descriptive keys for each metric.
    """
    num_classes = y_prob.shape[1]
    labels = list(range(num_classes))

    metrics: dict[str, Any] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    per_class_precision = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )

    for i in range(num_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"
        metrics[f"precision_{name}"] = float(per_class_precision[i])
        metrics[f"recall_{name}"] = float(per_class_recall[i])
        metrics[f"f1_{name}"] = float(per_class_f1[i])

    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )

    metrics["precision_weighted"] = float(
        precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    metrics["recall_weighted"] = float(
        recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    try:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_macro"] = float("nan")

    metrics["pr_auc_macro"] = _pr_auc_ovr(y_true, y_prob)

    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    metrics["top_1_accuracy"] = _top_k_accuracy(y_true, y_prob, k=1)
    metrics["top_2_accuracy"] = _top_k_accuracy(y_true, y_prob, k=2)

    metrics["ece"] = _expected_calibration_error(y_true, y_prob, n_bins=10)

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    return sklearn_confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes)),
    )


def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a classification report as a DataFrame.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        class_names: Optional list of class names.

    Returns:
        DataFrame with precision, recall, F1, and support per class.
    """
    names = class_names or CLASS_NAMES
    report_dict = classification_report(
        y_true, y_pred, target_names=names, output_dict=True, zero_division=0,
    )
    return pd.DataFrame(report_dict).transpose()


def compute_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for all metrics.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_pred: Predicted labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping metric name to ``(lower, mean, upper)`` tuples.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    alpha = (1.0 - ci) / 2.0

    all_bootstrap_metrics: list[dict[str, Any]] = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, size=n_samples)
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        boot_prob = y_prob[indices]

        if len(np.unique(boot_true)) < 2:
            continue

        try:
            boot_metrics = compute_all_metrics(boot_true, boot_pred, boot_prob)
            all_bootstrap_metrics.append(boot_metrics)
        except Exception:
            continue

    if not all_bootstrap_metrics:
        logger.warning("No valid bootstrap samples; returning point estimates.")
        point = compute_all_metrics(y_true, y_pred, y_prob)
        return {k: (v, v, v) for k, v in point.items() if isinstance(v, float)}

    result: dict[str, tuple[float, float, float]] = {}
    metric_keys = [
        k for k in all_bootstrap_metrics[0] if isinstance(all_bootstrap_metrics[0][k], float)
    ]

    for key in metric_keys:
        values = np.array([m[key] for m in all_bootstrap_metrics if not np.isnan(m.get(key, 0))])
        if len(values) == 0:
            continue
        lower = float(np.percentile(values, 100 * alpha))
        mean = float(np.mean(values))
        upper = float(np.percentile(values, 100 * (1 - alpha)))
        result[key] = (lower, mean, upper)

    return result
