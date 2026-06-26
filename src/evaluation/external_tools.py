"""Compare model predictions against external pathogenicity predictors.

Downloads precomputed dbNSFP scores for test-set variants and evaluates
SIFT, PolyPhen-2, CADD, and REVEL using published classification
thresholds.  Produces a comparison table saved to
``results/tables/external_comparison.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

EXTERNAL_THRESHOLDS: dict[str, dict[str, Any]] = {
    "SIFT": {
        "column": "SIFT_score",
        "threshold": 0.05,
        "direction": "below",
        "description": "SIFT score < 0.05 → Damaging (Pathogenic)",
    },
    "PolyPhen-2": {
        "column": "Polyphen2_HDIV_score",
        "threshold": 0.957,
        "direction": "above",
        "description": "PolyPhen-2 HDIV score > 0.957 → Probably Damaging",
    },
    "CADD": {
        "column": "CADD_phred",
        "threshold": 20.0,
        "direction": "above",
        "description": "CADD PHRED score > 20 → Pathogenic",
    },
    "REVEL": {
        "column": "REVEL_score",
        "threshold": 0.5,
        "direction": "above",
        "description": "REVEL score > 0.5 → Pathogenic",
    },
}


def map_to_binary(y: np.ndarray) -> np.ndarray:
    """Map 4-class labels to binary: pathogenic (1) vs benign (0).

    Classes 0 (Pathogenic) and 1 (Likely Pathogenic) → 1.
    Classes 2 (Benign) and 3 (Likely Benign) → 0.

    Args:
        y: Label array with values in {0, 1, 2, 3}.

    Returns:
        Binary array where 1 = pathogenic, 0 = benign.
    """
    return ((np.asarray(y) == 0) | (np.asarray(y) == 1)).astype(int)


def _apply_threshold(
    scores: np.ndarray,
    threshold: float,
    direction: str,
) -> np.ndarray:
    """Convert continuous scores to binary predictions using a threshold.

    Args:
        scores: Continuous predictor scores.
        threshold: Decision threshold.
        direction: ``"above"`` (score > threshold → pathogenic) or
            ``"below"`` (score < threshold → pathogenic).

    Returns:
        Binary prediction array (1 = pathogenic, 0 = benign).
    """
    if direction == "below":
        return (scores < threshold).astype(int)
    return (scores > threshold).astype(int)


def _pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary PR-AUC.

    Args:
        y_true: True binary labels.
        y_score: Continuous scores (higher → more pathogenic).

    Returns:
        Area under the precision-recall curve.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    return float(abs(np.trapezoid(precision_vals, recall_vals)))


def _compute_binary_metrics(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true_bin: True binary labels.
        y_pred_bin: Predicted binary labels.
        y_score: Optional continuous scores for AUROC/PR-AUC.

    Returns:
        Dict with accuracy, F1, AUROC, PR-AUC.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true_bin)) >= 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true_bin, y_score))
        except ValueError:
            metrics["auroc"] = float("nan")
        metrics["pr_auc"] = _pr_auc_binary(y_true_bin, y_score)
    else:
        metrics["auroc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def load_dbnsfp_scores(
    path: Path,
    variant_ids: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Load precomputed dbNSFP scores from a TSV or CSV file.

    Expects columns for at least some of: ``SIFT_score``,
    ``Polyphen2_HDIV_score``, ``CADD_phred``, ``REVEL_score``.
    Optionally filters to a set of variant identifiers.

    Args:
        path: Path to the dbNSFP scores file.
        variant_ids: Optional variant identifiers to filter to.

    Returns:
        DataFrame with predictor score columns.
    """
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".gz"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    for col in EXTERNAL_THRESHOLDS.values():
        col_name = col["column"]
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    if variant_ids is not None and "variant_id" in df.columns:
        ids_set = set(np.asarray(variant_ids, dtype=str))
        df = df[df["variant_id"].astype(str).isin(ids_set)].copy()

    logger.info(
        "Loaded dbNSFP scores: %d rows, columns: %s",
        len(df), list(df.columns),
    )
    return df


def evaluate_external_tool(
    tool_name: str,
    scores: np.ndarray,
    y_true_bin: np.ndarray,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    """Evaluate a single external predictor against binary ground truth.

    Args:
        tool_name: Human-readable tool name.
        scores: Raw continuous scores from the predictor.
        y_true_bin: True binary labels (1 = pathogenic, 0 = benign).
        threshold: Decision threshold for classification.
        direction: ``"above"`` or ``"below"``.

    Returns:
        Dict with tool name and all binary metrics.
    """
    valid_mask = ~np.isnan(scores)
    n_valid = int(valid_mask.sum())
    n_missing = int((~valid_mask).sum())

    if n_valid == 0:
        logger.warning("No valid scores for %s; skipping.", tool_name)
        return {
            "tool": tool_name,
            "n_scored": 0,
            "n_missing": n_missing,
            "accuracy": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "pr_auc": float("nan"),
        }

    valid_scores = scores[valid_mask]
    valid_true = y_true_bin[valid_mask]
    y_pred_bin = _apply_threshold(valid_scores, threshold, direction)

    if direction == "below":
        y_score_for_auc = 1.0 - valid_scores
    else:
        y_score_for_auc = valid_scores

    metrics = _compute_binary_metrics(valid_true, y_pred_bin, y_score_for_auc)
    metrics["tool"] = tool_name
    metrics["n_scored"] = n_valid
    metrics["n_missing"] = n_missing

    logger.info(
        "%s: accuracy=%.4f  F1=%.4f  AUROC=%.4f  PR-AUC=%.4f  (%d scored, %d missing)",
        tool_name,
        metrics["accuracy"],
        metrics["f1"],
        metrics["auroc"],
        metrics["pr_auc"],
        n_valid,
        n_missing,
    )
    return metrics


def compare_external_tools(
    dbnsfp_df: pd.DataFrame,
    y_true_4class: np.ndarray,
    y_pred_4class: np.ndarray,
    y_prob_4class: np.ndarray,
) -> pd.DataFrame:
    """Compare our model against SIFT, PolyPhen-2, CADD, and REVEL.

    Maps 4-class labels to binary (pathogenic vs benign) for a fair
    comparison, then computes accuracy, F1, AUROC, and PR-AUC for each
    tool and our model.

    Args:
        dbnsfp_df: DataFrame with external predictor score columns.
        y_true_4class: True 4-class labels of shape ``(n,)``.
        y_pred_4class: Our model's 4-class predictions of shape ``(n,)``.
        y_prob_4class: Our model's 4-class probabilities of shape
            ``(n, 4)``.

    Returns:
        Comparison DataFrame with one row per tool/model.
    """
    y_true_bin = map_to_binary(y_true_4class)

    all_results: list[dict[str, Any]] = []

    for tool_name, tool_cfg in EXTERNAL_THRESHOLDS.items():
        col = tool_cfg["column"]
        if col not in dbnsfp_df.columns:
            logger.warning(
                "Column '%s' not found for %s; skipping.", col, tool_name,
            )
            continue

        scores = dbnsfp_df[col].values
        if len(scores) != len(y_true_bin):
            logger.warning(
                "%s: score count (%d) != label count (%d); skipping.",
                tool_name, len(scores), len(y_true_bin),
            )
            continue

        result = evaluate_external_tool(
            tool_name,
            scores.astype(float),
            y_true_bin,
            tool_cfg["threshold"],
            tool_cfg["direction"],
        )
        all_results.append(result)

    our_pred_bin = map_to_binary(y_pred_4class)
    our_prob_pathogenic = y_prob_4class[:, 0] + y_prob_4class[:, 1]
    our_binary_metrics = _compute_binary_metrics(
        y_true_bin, our_pred_bin, our_prob_pathogenic,
    )
    our_binary_metrics["tool"] = "Our Model (binary)"
    our_binary_metrics["n_scored"] = len(y_true_bin)
    our_binary_metrics["n_missing"] = 0
    all_results.append(our_binary_metrics)

    from src.evaluation.metrics import compute_all_metrics

    our_4class_metrics = compute_all_metrics(y_true_4class, y_pred_4class, y_prob_4class)
    all_results.append({
        "tool": "Our Model (4-class)",
        "accuracy": our_4class_metrics["accuracy"],
        "f1": our_4class_metrics["f1_macro"],
        "auroc": our_4class_metrics.get("roc_auc_macro", float("nan")),
        "pr_auc": our_4class_metrics.get("pr_auc_macro", float("nan")),
        "n_scored": len(y_true_4class),
        "n_missing": 0,
    })

    if not all_results:
        logger.warning("No external tools evaluated successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    cols = ["tool", "accuracy", "f1", "auroc", "pr_auc", "n_scored", "n_missing"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    logger.info(
        "External tool comparison:\n%s",
        df.to_string(index=False),
    )
    return df


def run_external_comparison(
    dbnsfp_path: Path,
    y_true_4class: np.ndarray,
    y_pred_4class: np.ndarray,
    y_prob_4class: np.ndarray,
    output_dir: Path | None = None,
    variant_ids: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the full external tool comparison pipeline.

    Loads dbNSFP scores, evaluates each external tool, compares with
    our model, and optionally saves results.

    Args:
        dbnsfp_path: Path to the dbNSFP scores file.
        y_true_4class: True 4-class labels.
        y_pred_4class: Our model's 4-class predictions.
        y_prob_4class: Our model's 4-class probabilities.
        output_dir: Optional directory to save comparison CSV.
        variant_ids: Optional variant IDs for filtering dbNSFP.

    Returns:
        Comparison DataFrame.
    """
    logger.info("Loading external predictor scores from %s", dbnsfp_path)
    dbnsfp_df = load_dbnsfp_scores(dbnsfp_path, variant_ids=variant_ids)

    comparison_df = compare_external_tools(
        dbnsfp_df, y_true_4class, y_pred_4class, y_prob_4class,
    )

    if output_dir is not None and len(comparison_df) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "external_comparison.csv"
        comparison_df.to_csv(out_path, index=False)
        logger.info("Saved external comparison to %s", out_path)

    return comparison_df
