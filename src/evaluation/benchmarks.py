"""Baseline model benchmarks for comparison with the deep learning model.

Trains and evaluates Logistic Regression, Random Forest, XGBoost, LightGBM,
and MLP classifiers on flattened multi-omics feature vectors, using 5-fold
cross-validation for hyperparameter selection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _fit_and_evaluate(
    name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Train a model with CV, evaluate on test, and return metrics.

    Args:
        name: Human-readable model name.
        model: A scikit-learn compatible estimator.
        x_train: Training features of shape ``(n_train, n_features)``.
        y_train: Training labels of shape ``(n_train,)``.
        x_test: Test features of shape ``(n_test, n_features)``.
        y_test: Test labels of shape ``(n_test,)``.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with model name, CV score, and all test metrics.
    """
    logger.info("Training baseline: %s", name)

    cv_scores = cross_val_score(
        model, x_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1,
    )
    logger.info(
        "%s CV accuracy: %.4f (+/- %.4f)",
        name, cv_scores.mean(), cv_scores.std(),
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    metrics = compute_all_metrics(y_test, y_pred, y_prob)
    metrics["model"] = name
    metrics["cv_accuracy_mean"] = float(cv_scores.mean())
    metrics["cv_accuracy_std"] = float(cv_scores.std())

    logger.info("%s test accuracy: %.4f", name, metrics["accuracy"])
    return metrics


def _build_baselines() -> list[tuple[str, Any]]:
    """Build the list of baseline models with hyperparameters.

    Returns:
        List of ``(name, estimator)`` tuples.
    """
    baselines: list[tuple[str, Any]] = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                C=1.0,
                random_state=42,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        baselines.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            ),
        ))
    except ImportError:
        logger.warning("XGBoost not installed; skipping XGBoost baseline.")

    try:
        from lightgbm import LGBMClassifier

        baselines.append((
            "LightGBM",
            LGBMClassifier(
                n_estimators=500,
                max_depth=-1,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
        ))
    except ImportError:
        logger.warning("LightGBM not installed; skipping LightGBM baseline.")

    baselines.append((
        "MLP",
        MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    ))

    return baselines


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    our_model_metrics: dict[str, Any] | None = None,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Train and evaluate all baseline models, returning a comparison table.

    Args:
        x_train: Training features of shape ``(n_train, n_features)``.
        y_train: Training labels of shape ``(n_train,)``.
        x_test: Test features of shape ``(n_test, n_features)``.
        y_test: Test labels of shape ``(n_test,)``.
        our_model_metrics: Optional dict of our deep model's test metrics,
            added as a row for comparison.
        cv_folds: Number of cross-validation folds.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    baselines = _build_baselines()
    all_results: list[dict[str, Any]] = []

    for name, model in baselines:
        try:
            result = _fit_and_evaluate(
                name, model, x_train_scaled, y_train,
                x_test_scaled, y_test, cv_folds=cv_folds,
            )
            all_results.append(result)
        except Exception:
            logger.exception("Failed to train baseline '%s'", name)

    if our_model_metrics is not None:
        our_row = dict(our_model_metrics)
        our_row["model"] = "Our Model (Deep Learning)"
        our_row.setdefault("cv_accuracy_mean", float("nan"))
        our_row.setdefault("cv_accuracy_std", float("nan"))
        all_results.append(our_row)

    if not all_results:
        logger.warning("No baselines completed successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols]

    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    logger.info(
        "Baseline comparison (sorted by accuracy):\n%s",
        df[["model", "accuracy", "f1_macro", "roc_auc_macro", "mcc"]].to_string(
            index=False
        ),
    )

    return df
