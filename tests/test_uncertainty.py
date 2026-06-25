"""Tests for MC Dropout uncertainty and calibration modules.

Covers prediction shapes, variance behaviour, temperature scaling,
ECE computation, reliability diagrams, and the calibrated wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.full_model import PathogenicityPredictor
from src.uncertainty.calibration import (
    CalibratedModelWrapper,
    TemperatureScaling,
    compute_ece,
    compute_reliability_diagram,
)
from src.uncertainty.mc_dropout import MCDropoutPredictor
from src.utils.config import Config

BATCH_SIZE = 8
MUTATION_DIM = 42
MUTATION_EMBED = 64
EXPRESSION_DIM = 100
EXPRESSION_EMBED = 64
METHYLATION_DIM = 100
METHYLATION_EMBED = 64
CNV_DIM = 50
CNV_EMBED = 32
CLINICAL_DIM = 16
CLINICAL_EMBED = 16
FUSION_DIM = 64
NUM_CLASSES = 4
N_PASSES = 10


def _make_config(fusion_type: str = "cross_attention") -> Config:
    """Build a minimal valid Config for testing."""
    return Config({
        "data": {
            "clinvar_url": "https://example.com",
            "cbioportal_url": "https://example.com",
            "studies": [],
            "test_size": 0.15,
            "val_size": 0.15,
            "random_seed": 42,
        },
        "model": {
            "mutation_input_dim": MUTATION_DIM,
            "mutation_embed_dim": MUTATION_EMBED,
            "expression_input_dim": EXPRESSION_DIM,
            "expression_embed_dim": EXPRESSION_EMBED,
            "methylation_input_dim": METHYLATION_DIM,
            "methylation_embed_dim": METHYLATION_EMBED,
            "cnv_input_dim": CNV_DIM,
            "cnv_embed_dim": CNV_EMBED,
            "clinical_input_dim": CLINICAL_DIM,
            "clinical_embed_dim": CLINICAL_EMBED,
            "fusion_dim": FUSION_DIM,
            "num_classes": NUM_CLASSES,
            "dropout": 0.3,
            "fusion_type": fusion_type,
        },
        "training": {
            "max_epochs": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 5,
            "focal_loss_gamma": 2.0,
            "num_workers": 0,
        },
        "experiment": {
            "name": "test",
            "tracking_uri": "mlruns",
        },
    })


def _make_batch(batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the Dataset format."""
    return {
        "mutation": torch.randn(batch_size, MUTATION_DIM),
        "expression": torch.randn(batch_size, EXPRESSION_DIM),
        "methylation": torch.randn(batch_size, METHYLATION_DIM),
        "cnv": torch.randn(batch_size, CNV_DIM),
        "clinical": torch.randn(batch_size, CLINICAL_DIM),
        "modality_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "label": torch.randint(0, NUM_CLASSES, (batch_size,)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MC Dropout Predictor
# ═══════════════════════════════════════════════════════════════════════════


class TestMCDropoutPredictor:
    """Tests for MC Dropout uncertainty estimation."""

    def test_output_keys(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        batch = _make_batch()
        result = mc.predict_with_uncertainty(batch)
        expected = {
            "mean_probs", "predicted_class", "epistemic_uncertainty",
            "predictive_entropy", "all_predictions",
        }
        assert set(result.keys()) == expected

    def test_mean_probs_shape(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert result["mean_probs"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_predicted_class_shape(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert result["predicted_class"].shape == (BATCH_SIZE,)
        assert (result["predicted_class"] >= 0).all()
        assert (result["predicted_class"] < NUM_CLASSES).all()

    def test_epistemic_uncertainty_shape(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert result["epistemic_uncertainty"].shape == (BATCH_SIZE,)

    def test_predictive_entropy_shape(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert result["predictive_entropy"].shape == (BATCH_SIZE,)

    def test_all_predictions_shape(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert result["all_predictions"].shape == (
            N_PASSES, BATCH_SIZE, NUM_CLASSES,
        )

    def test_mean_probs_sum_to_one(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        sums = result["mean_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-4)

    def test_uncertainty_is_non_negative(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert (result["epistemic_uncertainty"] >= 0).all()

    def test_entropy_is_non_negative(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch())
        assert (result["predictive_entropy"] >= 0).all()

    def test_model_returns_to_eval_mode(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        mc.predict_with_uncertainty(_make_batch())
        assert not model.training

    def test_single_sample_batch(self) -> None:
        model = PathogenicityPredictor(_make_config())
        mc = MCDropoutPredictor(model, n_forward_passes=N_PASSES)
        result = mc.predict_with_uncertainty(_make_batch(batch_size=1))
        assert result["mean_probs"].shape == (1, NUM_CLASSES)
        assert result["predicted_class"].shape == (1,)

    def test_different_n_passes(self) -> None:
        model = PathogenicityPredictor(_make_config())
        for n in [5, 20]:
            mc = MCDropoutPredictor(model, n_forward_passes=n)
            result = mc.predict_with_uncertainty(_make_batch())
            assert result["all_predictions"].shape[0] == n


# ═══════════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════════


class TestTemperatureScaling:
    """Tests for the TemperatureScaling calibration module."""

    def test_forward_output_shape(self) -> None:
        scaler = TemperatureScaling()
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        probs = scaler(logits)
        assert probs.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_probabilities_sum_to_one(self) -> None:
        scaler = TemperatureScaling()
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        probs = scaler(logits)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_temperature_one_matches_softmax(self) -> None:
        scaler = TemperatureScaling(initial_temperature=1.0)
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        expected = torch.softmax(logits, dim=-1)
        actual = scaler(logits)
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_higher_temperature_flattens_distribution(self) -> None:
        logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
        scaler_low = TemperatureScaling(initial_temperature=0.5)
        scaler_high = TemperatureScaling(initial_temperature=5.0)
        probs_low = scaler_low(logits)
        probs_high = scaler_high(logits)
        max_low = probs_low.max(dim=-1).values.mean()
        max_high = probs_high.max(dim=-1).values.mean()
        assert max_low > max_high

    def test_optimize_temperature_returns_positive(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(100, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (100,))
        scaler = TemperatureScaling()
        optimal_t = scaler.optimize_temperature(logits, labels)
        assert optimal_t > 0

    def test_optimize_temperature_changes_value(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(100, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (100,))
        scaler = TemperatureScaling(initial_temperature=5.0)
        initial = scaler.temperature.item()
        scaler.optimize_temperature(logits, labels)
        assert scaler.temperature.item() != initial


# ═══════════════════════════════════════════════════════════════════════════
# ECE Computation
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeECE:
    """Tests for Expected Calibration Error computation."""

    def test_perfect_calibration(self) -> None:
        y_true = np.array([0, 1, 2, 3] * 25)
        y_prob = np.eye(NUM_CLASSES)[y_true]
        ece = compute_ece(y_true, y_prob, n_bins=15)
        assert ece < 0.05

    def test_ece_is_non_negative(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, NUM_CLASSES, 100)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, 100)
        ece = compute_ece(y_true, y_prob, n_bins=15)
        assert ece >= 0.0

    def test_ece_bounded_by_one(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, NUM_CLASSES, 100)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, 100)
        ece = compute_ece(y_true, y_prob, n_bins=15)
        assert ece <= 1.0

    def test_different_n_bins(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, NUM_CLASSES, 100)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, 100)
        for n_bins in [5, 10, 15, 20]:
            ece = compute_ece(y_true, y_prob, n_bins=n_bins)
            assert 0.0 <= ece <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Reliability Diagram
# ═══════════════════════════════════════════════════════════════════════════


class TestReliabilityDiagram:
    """Tests for reliability diagram computation."""

    def test_output_shapes(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, NUM_CLASSES, 100)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, 100)
        mean_pred, true_frac, counts = compute_reliability_diagram(
            y_true, y_prob, n_bins=10,
        )
        assert mean_pred.shape == (10,)
        assert true_frac.shape == (10,)
        assert counts.shape == (10,)

    def test_bin_counts_sum(self) -> None:
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, NUM_CLASSES, n)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, n)
        _, _, counts = compute_reliability_diagram(y_true, y_prob, n_bins=10)
        assert counts.sum() <= n

    def test_true_fraction_range(self) -> None:
        np.random.seed(42)
        y_true = np.random.randint(0, NUM_CLASSES, 100)
        y_prob = np.random.dirichlet([1] * NUM_CLASSES, 100)
        _, true_frac, counts = compute_reliability_diagram(
            y_true, y_prob, n_bins=10,
        )
        for i in range(len(true_frac)):
            if counts[i] > 0:
                assert 0.0 <= true_frac[i] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Calibrated Model Wrapper
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibratedModelWrapper:
    """Tests for the calibrated model wrapper."""

    def test_output_keys(self) -> None:
        model = PathogenicityPredictor(_make_config())
        scaler = TemperatureScaling(initial_temperature=1.5)
        wrapper = CalibratedModelWrapper(model, scaler)
        batch = _make_batch()
        result = wrapper(batch)
        expected = {
            "logits", "probabilities", "predicted_class",
            "fused_embedding", "modality_embeddings", "attention_weights",
        }
        assert set(result.keys()) == expected

    def test_calibrated_probs_sum_to_one(self) -> None:
        model = PathogenicityPredictor(_make_config())
        scaler = TemperatureScaling(initial_temperature=1.5)
        wrapper = CalibratedModelWrapper(model, scaler)
        wrapper.eval()
        result = wrapper(_make_batch())
        sums = result["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_calibrated_probs_differ_from_raw(self) -> None:
        model = PathogenicityPredictor(_make_config())
        model.eval()
        batch = _make_batch()

        raw_result = model(batch)
        raw_probs = raw_result["probabilities"].detach()

        scaler = TemperatureScaling(initial_temperature=2.0)
        wrapper = CalibratedModelWrapper(model, scaler)
        wrapper.eval()
        cal_result = wrapper(batch)
        cal_probs = cal_result["probabilities"].detach()

        assert not torch.allclose(raw_probs, cal_probs, atol=1e-6)

    def test_logits_unchanged(self) -> None:
        model = PathogenicityPredictor(_make_config())
        model.eval()
        batch = _make_batch()

        raw_logits = model(batch)["logits"].detach()

        scaler = TemperatureScaling(initial_temperature=2.0)
        wrapper = CalibratedModelWrapper(model, scaler)
        wrapper.eval()
        cal_logits = wrapper(batch)["logits"].detach()

        assert torch.allclose(raw_logits, cal_logits, atol=1e-6)
