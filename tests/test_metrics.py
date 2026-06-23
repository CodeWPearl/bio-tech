"""Tests for the evaluation metrics module.

Verifies :func:`compute_all_metrics`, :func:`get_confusion_matrix`,
:func:`classification_report_df`, and :func:`compute_ci` against sklearn
on known inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metrics import (
    _expected_calibration_error,
    _pr_auc_ovr,
    _top_k_accuracy,
    classification_report_df,
    compute_all_metrics,
    compute_ci,
    get_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def known_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small deterministic dataset for testing."""
    np.random.seed(42)
    n = 100
    num_classes = 4

    y_true = np.random.randint(0, num_classes, size=n)

    y_prob = np.random.dirichlet(np.ones(num_classes), size=n)
    for i in range(n):
        y_prob[i, y_true[i]] += 0.5
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


@pytest.fixture()
def perfect_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a perfectly-predicted dataset."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = y_true.copy()
    y_prob = np.eye(4)[y_true]
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# compute_all_metrics tests
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_dict(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a dict."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert isinstance(result, dict)

    def test_accuracy_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Accuracy should match sklearn's accuracy_score."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = accuracy_score(y_true, y_pred)
        assert abs(result["accuracy"] - expected) < 1e-10

    def test_f1_macro_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Macro F1 should match sklearn."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
        assert abs(result["f1_macro"] - expected) < 1e-10

    def test_f1_weighted_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Weighted F1 should match sklearn."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        assert abs(result["f1_weighted"] - expected) < 1e-10

    def test_precision_macro_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Macro precision should match sklearn."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = precision_score(y_true, y_pred, average="macro", zero_division=0)
        assert abs(result["precision_macro"] - expected) < 1e-10

    def test_recall_macro_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Macro recall should match sklearn."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = recall_score(y_true, y_pred, average="macro", zero_division=0)
        assert abs(result["recall_macro"] - expected) < 1e-10

    def test_mcc_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """MCC should match sklearn's matthews_corrcoef."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = matthews_corrcoef(y_true, y_pred)
        assert abs(result["mcc"] - expected) < 1e-10

    def test_roc_auc_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """ROC-AUC should match sklearn's roc_auc_score."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        assert abs(result["roc_auc_macro"] - expected) < 1e-10

    def test_cohen_kappa_matches_sklearn(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Cohen's kappa should match sklearn's cohen_kappa_score."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        expected = cohen_kappa_score(y_true, y_pred)
        assert abs(result["cohen_kappa"] - expected) < 1e-10

    def test_per_class_keys_present(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Per-class precision, recall, F1 keys should be present."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        for name in ["Pathogenic", "Likely Pathogenic", "Benign", "Likely Benign"]:
            assert f"precision_{name}" in result
            assert f"recall_{name}" in result
            assert f"f1_{name}" in result

    def test_top_k_accuracy_keys(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Top-1 and top-2 accuracy should be present."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert "top_1_accuracy" in result
        assert "top_2_accuracy" in result
        assert result["top_2_accuracy"] >= result["top_1_accuracy"]

    def test_ece_present(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """ECE should be present and non-negative."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert "ece" in result
        assert result["ece"] >= 0.0

    def test_pr_auc_present(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """PR-AUC should be present and positive."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert "pr_auc_macro" in result
        assert result["pr_auc_macro"] > 0.0

    def test_all_values_are_float(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """All metric values should be Python floats."""
        y_true, y_pred, y_prob = known_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, expected float"


# ---------------------------------------------------------------------------
# Perfect predictions tests
# ---------------------------------------------------------------------------

class TestPerfectPredictions:
    """Tests with perfectly predicted data."""

    def test_perfect_accuracy(
        self, perfect_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Perfect predictions should yield accuracy=1.0."""
        y_true, y_pred, y_prob = perfect_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert result["accuracy"] == 1.0

    def test_perfect_f1(
        self, perfect_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Perfect predictions should yield F1=1.0."""
        y_true, y_pred, y_prob = perfect_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert result["f1_macro"] == 1.0
        assert result["f1_weighted"] == 1.0

    def test_perfect_mcc(
        self, perfect_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Perfect predictions should yield MCC=1.0."""
        y_true, y_pred, y_prob = perfect_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert result["mcc"] == 1.0

    def test_perfect_kappa(
        self, perfect_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Perfect predictions should yield Cohen's kappa=1.0."""
        y_true, y_pred, y_prob = perfect_data
        result = compute_all_metrics(y_true, y_pred, y_prob)
        assert result["cohen_kappa"] == 1.0

    def test_perfect_confusion_matrix(
        self, perfect_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Perfect predictions should yield a diagonal confusion matrix."""
        y_true, y_pred, _ = perfect_data
        cm = get_confusion_matrix(y_true, y_pred, num_classes=4)
        off_diagonal = cm.sum() - np.trace(cm)
        assert off_diagonal == 0


# ---------------------------------------------------------------------------
# top_k_accuracy tests
# ---------------------------------------------------------------------------

class TestTopKAccuracy:
    """Tests for the top-k accuracy helper."""

    def test_top_1_equals_accuracy(self) -> None:
        """Top-1 accuracy should equal standard accuracy."""
        y_true = np.array([0, 1, 2, 3])
        y_prob = np.eye(4)
        assert _top_k_accuracy(y_true, y_prob, k=1) == 1.0

    def test_top_2_at_least_top_1(self) -> None:
        """Top-2 accuracy should be >= top-1 accuracy."""
        np.random.seed(0)
        y_true = np.random.randint(0, 4, 50)
        y_prob = np.random.dirichlet(np.ones(4), 50)
        top1 = _top_k_accuracy(y_true, y_prob, k=1)
        top2 = _top_k_accuracy(y_true, y_prob, k=2)
        assert top2 >= top1

    def test_top_4_is_always_1(self) -> None:
        """Top-4 accuracy with 4 classes should always be 1.0."""
        np.random.seed(0)
        y_true = np.random.randint(0, 4, 50)
        y_prob = np.random.dirichlet(np.ones(4), 50)
        assert _top_k_accuracy(y_true, y_prob, k=4) == 1.0


# ---------------------------------------------------------------------------
# ECE tests
# ---------------------------------------------------------------------------

class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration(self) -> None:
        """ECE of perfectly calibrated predictions should be low."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_prob = np.eye(4)[y_true]
        ece = _expected_calibration_error(y_true, y_prob)
        assert ece < 0.2

    def test_ece_non_negative(self) -> None:
        """ECE should always be non-negative."""
        np.random.seed(0)
        y_true = np.random.randint(0, 4, 100)
        y_prob = np.random.dirichlet(np.ones(4), 100)
        ece = _expected_calibration_error(y_true, y_prob)
        assert ece >= 0.0

    def test_ece_bounded(self) -> None:
        """ECE should be in [0, 1]."""
        np.random.seed(0)
        y_true = np.random.randint(0, 4, 100)
        y_prob = np.random.dirichlet(np.ones(4), 100)
        ece = _expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# PR-AUC tests
# ---------------------------------------------------------------------------

class TestPRAUC:
    """Tests for PR-AUC computation."""

    def test_pr_auc_positive(self) -> None:
        """PR-AUC should be positive for random predictions."""
        np.random.seed(0)
        y_true = np.random.randint(0, 4, 100)
        y_prob = np.random.dirichlet(np.ones(4), 100)
        pr_auc = _pr_auc_ovr(y_true, y_prob)
        assert pr_auc > 0.0

    def test_pr_auc_perfect(self) -> None:
        """PR-AUC for perfect predictions should be high."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_prob = np.eye(4)[y_true]
        pr_auc = _pr_auc_ovr(y_true, y_prob)
        assert pr_auc > 0.9


# ---------------------------------------------------------------------------
# confusion_matrix tests
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Tests for get_confusion_matrix."""

    def test_shape(self) -> None:
        """Confusion matrix should be (num_classes, num_classes)."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 0])
        cm = get_confusion_matrix(y_true, y_pred, num_classes=4)
        assert cm.shape == (4, 4)

    def test_sum_equals_n(self) -> None:
        """Total of confusion matrix should equal sample count."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 1, 0, 3, 2])
        cm = get_confusion_matrix(y_true, y_pred, num_classes=4)
        assert cm.sum() == len(y_true)

    def test_diagonal_is_correct(self) -> None:
        """Diagonal entries should count correct predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        cm = get_confusion_matrix(y_true, y_pred, num_classes=4)
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 2


# ---------------------------------------------------------------------------
# classification_report_df tests
# ---------------------------------------------------------------------------

class TestClassificationReportDf:
    """Tests for classification_report_df."""

    def test_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 0])
        df = classification_report_df(y_true, y_pred)
        assert hasattr(df, "columns")

    def test_contains_class_names(self) -> None:
        """Index should contain all class names."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 0])
        df = classification_report_df(y_true, y_pred)
        for name in ["Pathogenic", "Likely Pathogenic", "Benign", "Likely Benign"]:
            assert name in df.index

    def test_contains_metric_columns(self) -> None:
        """Should contain precision, recall, f1-score columns."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        df = classification_report_df(y_true, y_pred)
        for col in ["precision", "recall", "f1-score"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# compute_ci tests
# ---------------------------------------------------------------------------

class TestComputeCI:
    """Tests for bootstrap confidence intervals."""

    def test_returns_dict(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a dict."""
        y_true, y_pred, y_prob = known_data
        result = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50)
        assert isinstance(result, dict)

    def test_ci_tuples(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Each CI should be a (lower, mean, upper) tuple."""
        y_true, y_pred, y_prob = known_data
        result = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50)
        for key, (lower, mean, upper) in result.items():
            assert lower <= mean <= upper or np.isnan(lower), (
                f"{key}: {lower} <= {mean} <= {upper}"
            )

    def test_lower_le_upper(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Lower bound should be <= upper bound."""
        y_true, y_pred, y_prob = known_data
        result = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50)
        for key, (lower, _, upper) in result.items():
            assert lower <= upper or np.isnan(lower), f"{key}: {lower} > {upper}"

    def test_accuracy_ci_contains_point(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """The point estimate should roughly fall within the CI."""
        y_true, y_pred, y_prob = known_data
        point = compute_all_metrics(y_true, y_pred, y_prob)["accuracy"]
        ci = compute_ci(y_true, y_pred, y_prob, n_bootstrap=200)
        if "accuracy" in ci:
            lower, _, upper = ci["accuracy"]
            assert lower <= point + 0.05
            assert upper >= point - 0.05

    def test_reproducibility(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Same seed should produce the same CIs."""
        y_true, y_pred, y_prob = known_data
        ci1 = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50, seed=123)
        ci2 = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50, seed=123)
        for key in ci1:
            if key in ci2:
                assert ci1[key] == ci2[key], f"Mismatch for {key}"

    def test_key_metrics_present(
        self, known_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Key metrics should be present in CI output."""
        y_true, y_pred, y_prob = known_data
        result = compute_ci(y_true, y_pred, y_prob, n_bootstrap=50)
        for key in ["accuracy", "f1_macro", "mcc"]:
            assert key in result, f"{key} missing from CI results"
