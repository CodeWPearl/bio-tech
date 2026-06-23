"""Tests for the biological validation module.

Verifies COSMIC gene loading, cancer driver prediction validation,
ClinVar confidence analysis, gene-level accuracy, and the classification
report for driver mutations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.biological_validation import (
    COSMIC_CENSUS_GENES,
    REVIEW_STAR_MAP,
    _map_review_stars,
    cancer_driver_classification_report,
    gene_level_accuracy,
    load_cosmic_genes,
    run_biological_validation,
    validate_cancer_driver_predictions,
    validate_clinvar_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def driver_data() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Create synthetic data with known driver and non-driver genes."""
    genes = np.array([
        "TP53", "TP53", "BRCA1", "BRCA1", "BRCA2",
        "MYGENE1", "MYGENE1", "MYGENE2", "MYGENE2", "MYGENE3",
    ])
    y_true = np.array([0, 0, 1, 1, 0, 2, 3, 2, 3, 2])
    y_pred = np.array([0, 0, 1, 0, 0, 2, 3, 2, 3, 2])
    y_prob = np.zeros((10, 4))
    for i in range(10):
        y_prob[i, y_pred[i]] = 0.8
        for j in range(4):
            if j != y_pred[i]:
                y_prob[i, j] = 0.2 / 3
    return genes, y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# COSMIC gene loading
# ---------------------------------------------------------------------------

class TestLoadCosmicGenes:
    """Tests for load_cosmic_genes."""

    def test_builtin_genes_not_empty(self) -> None:
        """Built-in COSMIC gene list should have hundreds of genes."""
        genes = load_cosmic_genes()
        assert len(genes) > 500

    def test_known_genes_present(self) -> None:
        """Well-known cancer genes should be in the list."""
        genes = load_cosmic_genes()
        for gene in ["TP53", "BRCA1", "BRCA2", "KRAS", "EGFR", "BRAF", "APC"]:
            assert gene in genes

    def test_returns_frozenset(self) -> None:
        """Should return a frozenset."""
        genes = load_cosmic_genes()
        assert isinstance(genes, frozenset)

    def test_nonexistent_path_uses_builtin(self, tmp_path: object) -> None:
        """Non-existent file path should fall back to built-in."""
        from pathlib import Path
        genes = load_cosmic_genes(Path("nonexistent_file.csv"))
        assert len(genes) > 500


# ---------------------------------------------------------------------------
# Review star mapping
# ---------------------------------------------------------------------------

class TestReviewStarMapping:
    """Tests for _map_review_stars."""

    def test_practice_guideline_4_stars(self) -> None:
        """'practice guideline' should map to 4 stars."""
        result = _map_review_stars(pd.Series(["practice guideline"]))
        assert result.iloc[0] == 4

    def test_expert_panel_3_stars(self) -> None:
        """'reviewed by expert panel' should map to 3 stars."""
        result = _map_review_stars(pd.Series(["reviewed by expert panel"]))
        assert result.iloc[0] == 3

    def test_multiple_submitters_2_stars(self) -> None:
        """Multiple submitters, no conflicts should map to 2 stars."""
        result = _map_review_stars(
            pd.Series(["criteria provided, multiple submitters, no conflicts"])
        )
        assert result.iloc[0] == 2

    def test_no_assertion_0_stars(self) -> None:
        """'no assertion criteria provided' should map to 0 stars."""
        result = _map_review_stars(pd.Series(["no assertion criteria provided"]))
        assert result.iloc[0] == 0

    def test_unknown_status_0_stars(self) -> None:
        """Unknown review status should default to 0 stars."""
        result = _map_review_stars(pd.Series(["totally unknown status"]))
        assert result.iloc[0] == 0


# ---------------------------------------------------------------------------
# Cancer driver prediction validation
# ---------------------------------------------------------------------------

class TestValidateCancerDriverPredictions:
    """Tests for validate_cancer_driver_predictions."""

    def test_has_driver_count(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should report the number of driver gene samples."""
        genes, y_true, y_pred, y_prob = driver_data
        result = validate_cancer_driver_predictions(genes, y_true, y_pred, y_prob)
        assert result["n_driver_genes"] == 5
        assert result["n_non_driver_genes"] == 5

    def test_driver_accuracy(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Driver accuracy should be computed correctly."""
        genes, y_true, y_pred, y_prob = driver_data
        result = validate_cancer_driver_predictions(genes, y_true, y_pred, y_prob)
        assert 0.0 <= result["driver_accuracy"] <= 1.0

    def test_driver_pathogenic_recall(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Driver pathogenic recall should be computed."""
        genes, y_true, y_pred, y_prob = driver_data
        result = validate_cancer_driver_predictions(genes, y_true, y_pred, y_prob)
        assert "driver_pathogenic_recall" in result
        assert 0.0 <= result["driver_pathogenic_recall"] <= 1.0

    def test_nondriver_benign_recall(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Non-driver benign recall should be 1.0 for perfect benign predictions."""
        genes, y_true, y_pred, y_prob = driver_data
        result = validate_cancer_driver_predictions(genes, y_true, y_pred, y_prob)
        assert result["nondriver_benign_recall"] == 1.0

    def test_total_samples(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Total samples should equal input size."""
        genes, y_true, y_pred, y_prob = driver_data
        result = validate_cancer_driver_predictions(genes, y_true, y_pred, y_prob)
        assert result["n_total"] == 10


# ---------------------------------------------------------------------------
# ClinVar confidence validation
# ---------------------------------------------------------------------------

class TestValidateClinvarConfidence:
    """Tests for validate_clinvar_confidence."""

    def test_per_star_output(self) -> None:
        """Should have per-star confidence breakdown."""
        review = pd.Series([
            "practice guideline",
            "reviewed by expert panel",
            "criteria provided, single submitter",
            "no assertion criteria provided",
        ])
        y_prob = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.1, 0.05, 0.05],
            [0.6, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ])
        result = validate_clinvar_confidence(review, y_prob)
        assert "per_star" in result
        assert len(result["per_star"]) > 0

    def test_correlation_present(self) -> None:
        """Should compute star-confidence correlation."""
        review = pd.Series([
            "practice guideline",
            "reviewed by expert panel",
            "criteria provided, single submitter",
            "no assertion criteria provided",
        ])
        y_prob = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.1, 0.05, 0.05],
            [0.6, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ])
        result = validate_clinvar_confidence(review, y_prob)
        assert "star_confidence_correlation" in result

    def test_higher_stars_higher_confidence(self) -> None:
        """Higher star variants should ideally have higher model confidence."""
        review = pd.Series([
            "practice guideline",
            "practice guideline",
            "no assertion criteria provided",
            "no assertion criteria provided",
        ])
        y_prob = np.array([
            [0.95, 0.02, 0.02, 0.01],
            [0.90, 0.05, 0.03, 0.02],
            [0.30, 0.30, 0.20, 0.20],
            [0.28, 0.28, 0.22, 0.22],
        ])
        result = validate_clinvar_confidence(review, y_prob)
        high_star = result["per_star"][4]["mean_confidence"]
        low_star = result["per_star"][0]["mean_confidence"]
        assert high_star > low_star


# ---------------------------------------------------------------------------
# Gene-level accuracy
# ---------------------------------------------------------------------------

class TestGeneLevelAccuracy:
    """Tests for gene_level_accuracy."""

    def test_returns_dataframe(self) -> None:
        """Should return a DataFrame."""
        genes = np.array(["A"] * 10 + ["B"] * 10)
        y_true = np.random.randint(0, 4, 20)
        y_pred = y_true.copy()
        df = gene_level_accuracy(genes, y_true, y_pred, min_samples=5)
        assert isinstance(df, pd.DataFrame)

    def test_perfect_gene_accuracy(self) -> None:
        """Gene with perfect predictions should have accuracy=1.0."""
        genes = np.array(["PERFECT"] * 10 + ["BAD"] * 10)
        y_true = np.array([0] * 10 + [0] * 10)
        y_pred = np.array([0] * 10 + [1] * 10)
        df = gene_level_accuracy(genes, y_true, y_pred, min_samples=5)
        perfect_row = df[df["gene"] == "PERFECT"]
        assert len(perfect_row) == 1
        assert perfect_row["accuracy"].iloc[0] == 1.0

    def test_min_samples_filter(self) -> None:
        """Genes below min_samples should be excluded."""
        genes = np.array(["COMMON"] * 20 + ["RARE"] * 3)
        y_true = np.zeros(23, dtype=int)
        y_pred = np.zeros(23, dtype=int)
        df = gene_level_accuracy(genes, y_true, y_pred, min_samples=5)
        assert "RARE" not in df["gene"].values

    def test_sorted_by_accuracy(self) -> None:
        """Output should be sorted by accuracy ascending."""
        genes = np.array(["LOW"] * 10 + ["HIGH"] * 10)
        y_true = np.array([0] * 20)
        y_pred = np.array([1] * 10 + [0] * 10)
        df = gene_level_accuracy(genes, y_true, y_pred, min_samples=5)
        if len(df) > 1:
            assert df["accuracy"].iloc[0] <= df["accuracy"].iloc[-1]


# ---------------------------------------------------------------------------
# Cancer driver classification report
# ---------------------------------------------------------------------------

class TestCancerDriverClassificationReport:
    """Tests for cancer_driver_classification_report."""

    def test_returns_dict(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a dict with precision, recall, F1."""
        genes, y_true, y_pred, _ = driver_data
        result = cancer_driver_classification_report(genes, y_true, y_pred)
        assert "driver_precision" in result
        assert "driver_recall" in result
        assert "driver_f1" in result

    def test_values_in_range(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Precision, recall, F1 should be in [0, 1]."""
        genes, y_true, y_pred, _ = driver_data
        result = cancer_driver_classification_report(genes, y_true, y_pred)
        for key in ["driver_precision", "driver_recall", "driver_f1"]:
            assert 0.0 <= result[key] <= 1.0

    def test_no_driver_genes_returns_nan(self) -> None:
        """If no driver genes are present, should return NaN."""
        genes = np.array(["NONDRIVER1", "NONDRIVER2"])
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = cancer_driver_classification_report(genes, y_true, y_pred)
        assert np.isnan(result["driver_f1"])


# ---------------------------------------------------------------------------
# Full biological validation
# ---------------------------------------------------------------------------

class TestRunBiologicalValidation:
    """Tests for run_biological_validation."""

    def test_returns_dict(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should return a dict with expected sections."""
        genes, y_true, y_pred, y_prob = driver_data
        result = run_biological_validation(genes, y_true, y_pred, y_prob)
        assert "driver_validation" in result
        assert "driver_classification" in result
        assert "gene_level_accuracy" in result

    def test_with_review_status(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should include ClinVar confidence when review_status is provided."""
        genes, y_true, y_pred, y_prob = driver_data
        review = pd.Series([
            "practice guideline", "reviewed by expert panel",
            "criteria provided, single submitter", "criteria provided, single submitter",
            "criteria provided, single submitter",
            "no assertion criteria provided", "no assertion criteria provided",
            "no assertion criteria provided", "no assertion criteria provided",
            "no assertion criteria provided",
        ])
        result = run_biological_validation(
            genes, y_true, y_pred, y_prob, review_status=review,
        )
        assert "clinvar_confidence" in result

    def test_without_review_status(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Should skip ClinVar confidence when review_status is None."""
        genes, y_true, y_pred, y_prob = driver_data
        result = run_biological_validation(genes, y_true, y_pred, y_prob)
        assert "clinvar_confidence" not in result

    def test_gene_level_section(
        self,
        driver_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Gene-level accuracy section should be present."""
        genes, y_true, y_pred, y_prob = driver_data
        result = run_biological_validation(genes, y_true, y_pred, y_prob)
        gl = result["gene_level_accuracy"]
        assert "n_genes_evaluated" in gl
        assert "mean_gene_accuracy" in gl
