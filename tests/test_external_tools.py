"""Tests for the external tool comparison module.

Verifies binary mapping, threshold application, per-tool evaluation,
and the full comparison pipeline using synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.external_tools import (
    EXTERNAL_THRESHOLDS,
    _apply_threshold,
    _compute_binary_metrics,
    _pr_auc_binary,
    compare_external_tools,
    evaluate_external_tool,
    load_dbnsfp_scores,
    map_to_binary,
    run_external_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_4class() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic 4-class dataset."""
    np.random.seed(42)
    n = 80
    y_true = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
    y_prob = np.random.dirichlet(np.ones(4), size=n)
    for i in range(n):
        y_prob[i, y_true[i]] += 0.6
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


@pytest.fixture()
def dbnsfp_df(synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray]) -> pd.DataFrame:
    """Create a synthetic dbNSFP DataFrame with all four score columns."""
    np.random.seed(42)
    n = len(synthetic_4class[0])
    return pd.DataFrame({
        "SIFT_score": np.random.uniform(0.0, 1.0, n),
        "Polyphen2_HDIV_score": np.random.uniform(0.0, 1.0, n),
        "CADD_phred": np.random.uniform(0.0, 40.0, n),
        "REVEL_score": np.random.uniform(0.0, 1.0, n),
    })


# ---------------------------------------------------------------------------
# map_to_binary tests
# ---------------------------------------------------------------------------

class TestMapToBinary:
    """Tests for map_to_binary."""

    def test_pathogenic_maps_to_1(self) -> None:
        """Classes 0 and 1 should map to 1 (pathogenic)."""
        y = np.array([0, 1])
        result = map_to_binary(y)
        assert (result == np.array([1, 1])).all()

    def test_benign_maps_to_0(self) -> None:
        """Classes 2 and 3 should map to 0 (benign)."""
        y = np.array([2, 3])
        result = map_to_binary(y)
        assert (result == np.array([0, 0])).all()

    def test_all_classes(self) -> None:
        """All four classes should map correctly."""
        y = np.array([0, 1, 2, 3])
        result = map_to_binary(y)
        expected = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_output_is_int(self) -> None:
        """Output should be integer type."""
        y = np.array([0, 2])
        result = map_to_binary(y)
        assert result.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# _apply_threshold tests
# ---------------------------------------------------------------------------

class TestApplyThreshold:
    """Tests for _apply_threshold."""

    def test_above_direction(self) -> None:
        """Scores above threshold should be classified as pathogenic."""
        scores = np.array([0.1, 0.5, 0.9])
        result = _apply_threshold(scores, threshold=0.5, direction="above")
        expected = np.array([0, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_below_direction(self) -> None:
        """Scores below threshold should be classified as pathogenic."""
        scores = np.array([0.01, 0.05, 0.9])
        result = _apply_threshold(scores, threshold=0.05, direction="below")
        expected = np.array([1, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_output_is_binary(self) -> None:
        """Output should only contain 0s and 1s."""
        scores = np.random.uniform(0, 1, 100)
        result = _apply_threshold(scores, threshold=0.5, direction="above")
        assert set(np.unique(result)).issubset({0, 1})


# ---------------------------------------------------------------------------
# _pr_auc_binary tests
# ---------------------------------------------------------------------------

class TestPRAUCBinary:
    """Tests for _pr_auc_binary."""

    def test_positive_value(self) -> None:
        """PR-AUC should be positive for non-trivial data."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = _pr_auc_binary(y_true, y_score)
        assert result > 0.0

    def test_perfect_predictions(self) -> None:
        """PR-AUC should be high for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        result = _pr_auc_binary(y_true, y_score)
        assert result > 0.9

    def test_single_class_returns_nan(self) -> None:
        """PR-AUC should return NaN when only one class is present."""
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        result = _pr_auc_binary(y_true, y_score)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# _compute_binary_metrics tests
# ---------------------------------------------------------------------------

class TestComputeBinaryMetrics:
    """Tests for _compute_binary_metrics."""

    def test_returns_dict(self) -> None:
        """Should return a dict with expected keys."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = _compute_binary_metrics(y_true, y_pred)
        assert "accuracy" in result
        assert "f1" in result

    def test_accuracy_correct(self) -> None:
        """Accuracy should be correct for known input."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = _compute_binary_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0

    def test_with_scores(self) -> None:
        """Should compute AUROC and PR-AUC when scores are provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = _compute_binary_metrics(y_true, y_pred, y_score)
        assert "auroc" in result
        assert "pr_auc" in result
        assert result["auroc"] > 0.5

    def test_without_scores(self) -> None:
        """AUROC/PR-AUC should be NaN without scores."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = _compute_binary_metrics(y_true, y_pred)
        assert np.isnan(result["auroc"])
        assert np.isnan(result["pr_auc"])


# ---------------------------------------------------------------------------
# evaluate_external_tool tests
# ---------------------------------------------------------------------------

class TestEvaluateExternalTool:
    """Tests for evaluate_external_tool."""

    def test_returns_dict_with_tool_name(self) -> None:
        """Result should include the tool name."""
        scores = np.array([0.01, 0.02, 0.5, 0.8])
        y_true_bin = np.array([1, 1, 0, 0])
        result = evaluate_external_tool("SIFT", scores, y_true_bin, 0.05, "below")
        assert result["tool"] == "SIFT"

    def test_n_scored_and_missing(self) -> None:
        """Should report scored and missing counts."""
        scores = np.array([0.1, np.nan, 0.5, 0.8])
        y_true_bin = np.array([1, 1, 0, 0])
        result = evaluate_external_tool("Test", scores, y_true_bin, 0.5, "above")
        assert result["n_scored"] == 3
        assert result["n_missing"] == 1

    def test_all_nan_scores(self) -> None:
        """All NaN scores should yield n_scored=0 and NaN metrics."""
        scores = np.array([np.nan, np.nan, np.nan])
        y_true_bin = np.array([1, 0, 1])
        result = evaluate_external_tool("Bad", scores, y_true_bin, 0.5, "above")
        assert result["n_scored"] == 0
        assert np.isnan(result["accuracy"])

    def test_accuracy_in_range(self) -> None:
        """Accuracy should be between 0 and 1."""
        np.random.seed(0)
        scores = np.random.uniform(0, 1, 50)
        y_true_bin = np.random.randint(0, 2, 50)
        result = evaluate_external_tool("T", scores, y_true_bin, 0.5, "above")
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# EXTERNAL_THRESHOLDS tests
# ---------------------------------------------------------------------------

class TestExternalThresholds:
    """Tests for the EXTERNAL_THRESHOLDS configuration."""

    def test_four_tools_defined(self) -> None:
        """Should have exactly 4 external tools configured."""
        assert len(EXTERNAL_THRESHOLDS) == 4

    def test_expected_tools_present(self) -> None:
        """SIFT, PolyPhen-2, CADD, REVEL should all be present."""
        for tool in ["SIFT", "PolyPhen-2", "CADD", "REVEL"]:
            assert tool in EXTERNAL_THRESHOLDS

    def test_each_tool_has_required_keys(self) -> None:
        """Each tool config should have column, threshold, direction."""
        for tool_name, cfg in EXTERNAL_THRESHOLDS.items():
            assert "column" in cfg, f"{tool_name} missing 'column'"
            assert "threshold" in cfg, f"{tool_name} missing 'threshold'"
            assert "direction" in cfg, f"{tool_name} missing 'direction'"
            assert cfg["direction"] in ("above", "below"), (
                f"{tool_name} has invalid direction: {cfg['direction']}"
            )

    def test_sift_threshold(self) -> None:
        """SIFT threshold should be 0.05 with 'below' direction."""
        cfg = EXTERNAL_THRESHOLDS["SIFT"]
        assert cfg["threshold"] == 0.05
        assert cfg["direction"] == "below"

    def test_revel_threshold(self) -> None:
        """REVEL threshold should be 0.5 with 'above' direction."""
        cfg = EXTERNAL_THRESHOLDS["REVEL"]
        assert cfg["threshold"] == 0.5
        assert cfg["direction"] == "above"


# ---------------------------------------------------------------------------
# compare_external_tools tests
# ---------------------------------------------------------------------------

class TestCompareExternalTools:
    """Tests for compare_external_tools."""

    def test_returns_dataframe(
        self,
        dbnsfp_df: pd.DataFrame,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a DataFrame."""
        y_true, y_pred, y_prob = synthetic_4class
        result = compare_external_tools(dbnsfp_df, y_true, y_pred, y_prob)
        assert isinstance(result, pd.DataFrame)

    def test_has_our_model_rows(
        self,
        dbnsfp_df: pd.DataFrame,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include our model in binary and 4-class forms."""
        y_true, y_pred, y_prob = synthetic_4class
        result = compare_external_tools(dbnsfp_df, y_true, y_pred, y_prob)
        tools = result["tool"].tolist()
        assert "Our Model (binary)" in tools
        assert "Our Model (4-class)" in tools

    def test_has_external_tools(
        self,
        dbnsfp_df: pd.DataFrame,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include at least one external tool."""
        y_true, y_pred, y_prob = synthetic_4class
        result = compare_external_tools(dbnsfp_df, y_true, y_pred, y_prob)
        external_tools = [t for t in result["tool"] if "Our Model" not in t]
        assert len(external_tools) >= 1

    def test_has_metric_columns(
        self,
        dbnsfp_df: pd.DataFrame,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should have accuracy, f1, auroc, pr_auc columns."""
        y_true, y_pred, y_prob = synthetic_4class
        result = compare_external_tools(dbnsfp_df, y_true, y_pred, y_prob)
        for col in ["accuracy", "f1", "auroc", "pr_auc"]:
            assert col in result.columns

    def test_missing_column_skips_tool(
        self,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Tool with missing score column should be skipped."""
        y_true, y_pred, y_prob = synthetic_4class
        partial_df = pd.DataFrame({
            "SIFT_score": np.random.uniform(0, 1, len(y_true)),
        })
        result = compare_external_tools(partial_df, y_true, y_pred, y_prob)
        tools = result["tool"].tolist()
        assert "SIFT" in tools
        assert "CADD" not in tools


# ---------------------------------------------------------------------------
# load_dbnsfp_scores tests
# ---------------------------------------------------------------------------

class TestLoadDbNSFPScores:
    """Tests for load_dbnsfp_scores."""

    def test_loads_csv(self, tmp_path: Path) -> None:
        """Should load a CSV file."""
        df = pd.DataFrame({
            "SIFT_score": [0.01, 0.5],
            "CADD_phred": [25.0, 10.0],
        })
        path = tmp_path / "test_scores.csv"
        df.to_csv(path, index=False)
        result = load_dbnsfp_scores(path)
        assert len(result) == 2
        assert "SIFT_score" in result.columns

    def test_loads_tsv(self, tmp_path: Path) -> None:
        """Should load a TSV file."""
        df = pd.DataFrame({
            "REVEL_score": [0.3, 0.8],
            "variant_id": ["v1", "v2"],
        })
        path = tmp_path / "test_scores.tsv"
        df.to_csv(path, sep="\t", index=False)
        result = load_dbnsfp_scores(path)
        assert len(result) == 2

    def test_filter_by_variant_ids(self, tmp_path: Path) -> None:
        """Should filter to specified variant IDs."""
        df = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "SIFT_score": [0.01, 0.5, 0.9],
        })
        path = tmp_path / "test_scores.csv"
        df.to_csv(path, index=False)
        result = load_dbnsfp_scores(path, variant_ids=np.array(["v1", "v3"]))
        assert len(result) == 2

    def test_numeric_coercion(self, tmp_path: Path) -> None:
        """Non-numeric score values should be coerced to NaN."""
        df = pd.DataFrame({
            "SIFT_score": ["0.01", ".", "0.9"],
        })
        path = tmp_path / "test_scores.csv"
        df.to_csv(path, index=False)
        result = load_dbnsfp_scores(path)
        assert np.isnan(result["SIFT_score"].iloc[1])


# ---------------------------------------------------------------------------
# run_external_comparison tests
# ---------------------------------------------------------------------------

class TestRunExternalComparison:
    """Tests for run_external_comparison."""

    def test_saves_csv(
        self,
        tmp_path: Path,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should save external_comparison.csv when output_dir is given."""
        y_true, y_pred, y_prob = synthetic_4class
        n = len(y_true)
        scores_df = pd.DataFrame({
            "SIFT_score": np.random.uniform(0, 1, n),
            "Polyphen2_HDIV_score": np.random.uniform(0, 1, n),
            "CADD_phred": np.random.uniform(0, 40, n),
            "REVEL_score": np.random.uniform(0, 1, n),
        })
        scores_path = tmp_path / "scores.csv"
        scores_df.to_csv(scores_path, index=False)

        output_dir = tmp_path / "results"
        result = run_external_comparison(
            dbnsfp_path=scores_path,
            y_true_4class=y_true,
            y_pred_4class=y_pred,
            y_prob_4class=y_prob,
            output_dir=output_dir,
        )
        assert isinstance(result, pd.DataFrame)
        assert (output_dir / "external_comparison.csv").is_file()

    def test_returns_dataframe_without_save(
        self,
        tmp_path: Path,
        synthetic_4class: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a DataFrame even without saving."""
        y_true, y_pred, y_prob = synthetic_4class
        n = len(y_true)
        scores_df = pd.DataFrame({
            "SIFT_score": np.random.uniform(0, 1, n),
            "REVEL_score": np.random.uniform(0, 1, n),
        })
        scores_path = tmp_path / "scores.csv"
        scores_df.to_csv(scores_path, index=False)

        result = run_external_comparison(
            dbnsfp_path=scores_path,
            y_true_4class=y_true,
            y_pred_4class=y_pred,
            y_prob_4class=y_prob,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
