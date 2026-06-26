"""Tests for the baseline benchmark module.

Verifies :func:`run_baselines` and :func:`_build_baselines` using small
synthetic datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmarks import _build_baselines, _fit_and_evaluate, run_baselines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a small synthetic dataset for baseline testing."""
    np.random.seed(42)
    n_train, n_test, n_features = 120, 40, 20
    num_classes = 4

    x_train = np.random.randn(n_train, n_features)
    y_train = np.random.randint(0, num_classes, n_train)
    x_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, num_classes, n_test)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# _build_baselines tests
# ---------------------------------------------------------------------------

class TestBuildBaselines:
    """Tests for _build_baselines."""

    def test_returns_list(self) -> None:
        """Should return a list of (name, estimator) tuples."""
        baselines = _build_baselines()
        assert isinstance(baselines, list)
        assert len(baselines) >= 3

    def test_logistic_regression_present(self) -> None:
        """Logistic Regression should be in the baselines."""
        names = [name for name, _ in _build_baselines()]
        assert "Logistic Regression" in names

    def test_random_forest_present(self) -> None:
        """Random Forest should be in the baselines."""
        names = [name for name, _ in _build_baselines()]
        assert "Random Forest" in names

    def test_mlp_present(self) -> None:
        """MLP should be in the baselines."""
        names = [name for name, _ in _build_baselines()]
        assert "MLP" in names

    def test_all_have_fit_predict(self) -> None:
        """Every baseline should have fit and predict methods."""
        for name, model in _build_baselines():
            assert hasattr(model, "fit"), f"{name} has no fit method"
            assert hasattr(model, "predict"), f"{name} has no predict method"
            assert hasattr(model, "predict_proba"), f"{name} has no predict_proba method"


# ---------------------------------------------------------------------------
# _fit_and_evaluate tests
# ---------------------------------------------------------------------------

class TestFitAndEvaluate:
    """Tests for _fit_and_evaluate."""

    def test_returns_dict(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a dict with metrics."""
        x_train, y_train, x_test, y_test = small_dataset
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, random_state=42)
        result = _fit_and_evaluate(
            "LR", model, x_train, y_train, x_test, y_test, cv_folds=3,
        )
        assert isinstance(result, dict)

    def test_has_model_name(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include the model name."""
        x_train, y_train, x_test, y_test = small_dataset
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, random_state=42)
        result = _fit_and_evaluate(
            "TestModel", model, x_train, y_train, x_test, y_test, cv_folds=3,
        )
        assert result["model"] == "TestModel"

    def test_has_cv_scores(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include CV accuracy mean and std."""
        x_train, y_train, x_test, y_test = small_dataset
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, random_state=42)
        result = _fit_and_evaluate(
            "LR", model, x_train, y_train, x_test, y_test, cv_folds=3,
        )
        assert "cv_accuracy_mean" in result
        assert "cv_accuracy_std" in result
        assert 0.0 <= result["cv_accuracy_mean"] <= 1.0
        assert result["cv_accuracy_std"] >= 0.0

    def test_has_accuracy(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include test accuracy."""
        x_train, y_train, x_test, y_test = small_dataset
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, random_state=42)
        result = _fit_and_evaluate(
            "LR", model, x_train, y_train, x_test, y_test, cv_folds=3,
        )
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# run_baselines tests
# ---------------------------------------------------------------------------

class TestRunBaselines:
    """Tests for run_baselines."""

    def test_returns_dataframe(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a DataFrame."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        assert isinstance(result, pd.DataFrame)

    def test_has_multiple_rows(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should have at least 3 baseline models."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        assert len(result) >= 3

    def test_has_model_column(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should have a 'model' column."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        assert "model" in result.columns

    def test_sorted_by_accuracy(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Results should be sorted by accuracy descending."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        accuracies = result["accuracy"].tolist()
        assert accuracies == sorted(accuracies, reverse=True)

    def test_includes_our_model(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include our model when our_model_metrics is provided."""
        x_train, y_train, x_test, y_test = small_dataset
        our_metrics = {"accuracy": 0.95, "f1_macro": 0.94, "mcc": 0.90}
        result = run_baselines(
            x_train, y_train, x_test, y_test,
            our_model_metrics=our_metrics, cv_folds=3,
        )
        assert "Our Model (Deep Learning)" in result["model"].values

    def test_has_metric_columns(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should contain key metric columns."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        for col in ["accuracy", "f1_macro", "mcc"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_accuracy_values_in_range(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """All accuracy values should be in [0, 1]."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        for acc in result["accuracy"]:
            assert 0.0 <= acc <= 1.0

    def test_without_our_model(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should work without our_model_metrics."""
        x_train, y_train, x_test, y_test = small_dataset
        result = run_baselines(x_train, y_train, x_test, y_test, cv_folds=3)
        assert "Our Model (Deep Learning)" not in result["model"].values
