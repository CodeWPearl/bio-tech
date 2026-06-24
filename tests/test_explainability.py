"""Tests for the explainability modules.

Covers SHAP, Integrated Gradients, attention visualisation, and LIME
explainers using tiny synthetic models and data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.full_model import PathogenicityPredictor
from src.utils.config import Config

# ---------------------------------------------------------------------------
# Shared constants (small dims for fast tests)
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
MUTATION_DIM = 10
MUTATION_EMBED = 16
EXPRESSION_DIM = 20
EXPRESSION_EMBED = 16
METHYLATION_DIM = 20
METHYLATION_EMBED = 16
CNV_DIM = 10
CNV_EMBED = 8
CLINICAL_DIM = 8
CLINICAL_EMBED = 8
FUSION_DIM = 16
NUM_CLASSES = 4

MODALITY_DIMS: dict[str, int] = {
    "mutation": MUTATION_DIM,
    "expression": EXPRESSION_DIM,
    "methylation": METHYLATION_DIM,
    "cnv": CNV_DIM,
    "clinical": CLINICAL_DIM,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(fusion_type: str = "attention") -> Config:
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
            "dropout": 0.0,
            "fusion_type": fusion_type,
        },
        "training": {
            "max_epochs": 1,
            "batch_size": BATCH_SIZE,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 5,
            "focal_loss_gamma": 2.0,
            "num_workers": 0,
        },
        "experiment": {
            "name": "test_explainability",
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


def _make_model(fusion_type: str = "attention") -> PathogenicityPredictor:
    """Create a tiny model for testing."""
    config = _make_config(fusion_type)
    model = PathogenicityPredictor(config)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Explainer
# ═══════════════════════════════════════════════════════════════════════════


class TestSHAPExplainer:
    """Tests for the SHAP explainer."""

    def test_init(self) -> None:
        """SHAPExplainer initialises without error."""
        from src.explainability.shap_explainer import SHAPExplainer

        model = _make_model()
        explainer = SHAPExplainer(model, MODALITY_DIMS)
        assert explainer._total_dim == sum(MODALITY_DIMS.values())

    def test_modality_slices(self) -> None:
        """Modality slices cover the full feature dimension."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(_make_model(), MODALITY_DIMS)
        slices = explainer._modality_slices
        assert len(slices) == 5
        last_end = 0
        for name in ["mutation", "expression", "methylation", "cnv", "clinical"]:
            start, end = slices[name]
            assert start == last_end
            assert end - start == MODALITY_DIMS[name]
            last_end = end
        assert last_end == sum(MODALITY_DIMS.values())

    def test_batch_to_flat(self) -> None:
        """Batch dict is correctly flattened to numpy array."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(_make_model(), MODALITY_DIMS)
        batch = _make_batch(2)
        flat = explainer._batch_to_flat(batch)
        assert flat.shape == (2, sum(MODALITY_DIMS.values()))

    def test_build_feature_names_default(self) -> None:
        """Default feature names follow modality_i pattern."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(_make_model(), MODALITY_DIMS)
        names = explainer._build_feature_names(None)
        assert len(names) == sum(MODALITY_DIMS.values())
        assert names[0] == "mutation_0"
        assert names[MUTATION_DIM] == "expression_0"

    def test_build_feature_names_custom(self) -> None:
        """Custom feature names are used when provided."""
        from src.explainability.shap_explainer import SHAPExplainer

        explainer = SHAPExplainer(_make_model(), MODALITY_DIMS)
        custom = {"mutation": [f"mut_{i}" for i in range(MUTATION_DIM)]}
        names = explainer._build_feature_names(custom)
        assert names[0] == "mut_0"
        assert names[MUTATION_DIM] == "expression_0"

    def test_compute_global_importance_runs(self) -> None:
        """Global importance computation runs without error on tiny model."""
        from src.explainability.shap_explainer import SHAPExplainer

        model = _make_model()
        explainer = SHAPExplainer(model, MODALITY_DIMS)
        batch = _make_batch(8)
        result = explainer.compute_global_importance(batch, n_samples=4)
        assert "shap_values" in result
        assert "feature_importance" in result
        assert "modality_importance" in result
        assert len(result["feature_importance"]) == sum(MODALITY_DIMS.values())
        assert set(result["modality_importance"].keys()) == set(MODALITY_DIMS.keys())

    def test_compute_local_explanation_runs(self) -> None:
        """Local explanation runs without error for a single sample."""
        from src.explainability.shap_explainer import SHAPExplainer

        model = _make_model()
        explainer = SHAPExplainer(model, MODALITY_DIMS)
        sample = _make_batch(1)
        result = explainer.compute_local_explanation(sample)
        assert "shap_values" in result
        assert "feature_attributions" in result

    def test_generate_shap_plots(self, tmp_path: Path) -> None:
        """SHAP plots are saved to disk."""
        from src.explainability.shap_explainer import SHAPExplainer

        model = _make_model()
        explainer = SHAPExplainer(model, MODALITY_DIMS)
        batch = _make_batch(8)
        result = explainer.compute_global_importance(batch, n_samples=4)
        saved = explainer.generate_shap_plots(
            result["shap_values"], None, tmp_path,
        )
        assert len(saved) == 3
        for p in saved:
            assert p.exists()
            assert p.suffix == ".png"


# ═══════════════════════════════════════════════════════════════════════════
# Integrated Gradients
# ═══════════════════════════════════════════════════════════════════════════


class TestIGExplainer:
    """Tests for the Integrated Gradients explainer."""

    def test_init(self) -> None:
        """IGExplainer initialises without error."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        assert explainer._total_dim == sum(MODALITY_DIMS.values())

    def test_compute_attributions_shape(self) -> None:
        """IG attributions have correct shape."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batch = _make_batch(2)
        result = explainer.compute_attributions(batch, target_class=0, n_steps=5)
        assert result["attributions"].shape == (2, sum(MODALITY_DIMS.values()))

    def test_per_modality_shapes(self) -> None:
        """Per-modality IG attributions have correct shapes."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batch = _make_batch(2)
        result = explainer.compute_attributions(batch, target_class=0, n_steps=5)
        for mod, dim in MODALITY_DIMS.items():
            assert result["per_modality"][mod].shape == (2, dim)

    def test_modality_importance_keys(self) -> None:
        """Modality importance dict contains all modalities."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batch = _make_batch(2)
        result = explainer.compute_attributions(batch, target_class=0, n_steps=5)
        assert set(result["modality_importance"].keys()) == set(MODALITY_DIMS.keys())
        for val in result["modality_importance"].values():
            assert isinstance(val, float)
            assert val >= 0.0

    def test_attributions_are_finite(self) -> None:
        """IG attributions contain no NaN or Inf values."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batch = _make_batch(2)
        result = explainer.compute_attributions(batch, target_class=0, n_steps=5)
        assert torch.isfinite(result["attributions"]).all()

    def test_different_target_classes(self) -> None:
        """IG works for all target classes."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batch = _make_batch(2)
        for cls in range(NUM_CLASSES):
            result = explainer.compute_attributions(batch, target_class=cls, n_steps=5)
            assert result["attributions"].shape == (2, sum(MODALITY_DIMS.values()))

    def test_compute_modality_importance_from_loader(self) -> None:
        """Modality importance computed over a DataLoader."""
        from src.explainability.integrated_gradients import IGExplainer

        model = _make_model()
        explainer = IGExplainer(model, MODALITY_DIMS)
        batches = [_make_batch(2) for _ in range(3)]
        ranked = explainer.compute_modality_importance(batches, max_batches=3)
        assert set(ranked.keys()) == set(MODALITY_DIMS.keys())
        values = list(ranked.values())
        assert all(v >= 0.0 for v in values)
        assert values == sorted(values, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# Attention Visualiser
# ═══════════════════════════════════════════════════════════════════════════


class TestAttentionVisualizer:
    """Tests for the attention weight visualiser."""

    def test_extract_with_attention_fusion(self) -> None:
        """Attention weights extracted from attention fusion model."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batch = _make_batch()
        weights = viz.extract_attention_weights(batch)
        assert weights is not None
        assert weights.shape[0] == BATCH_SIZE
        assert weights.shape[1] == 5
        assert weights.shape[2] == 5

    def test_extract_without_attention(self) -> None:
        """Returns None for fusion types without attention weights."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("early")
        viz = AttentionVisualizer(model)
        batch = _make_batch()
        weights = viz.extract_attention_weights(batch)
        assert weights is None

    def test_attention_weights_valid(self) -> None:
        """Attention weights are non-negative and rows sum to ~1."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batch = _make_batch()
        weights = viz.extract_attention_weights(batch)
        assert weights is not None
        assert (weights >= -1e-6).all()
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_collect_attention_weights(self) -> None:
        """Weights collected from multiple batches are stacked."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batches = [_make_batch(2) for _ in range(3)]
        stacked = viz.collect_attention_weights(batches, max_batches=3)
        assert stacked.shape == (6, 5, 5)

    def test_plot_heatmap(self, tmp_path: Path) -> None:
        """Attention heatmap is saved to disk."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batch = _make_batch()
        weights = viz.extract_attention_weights(batch)
        assert weights is not None
        path = viz.plot_attention_heatmap(weights, tmp_path / "heatmap.png")
        assert path.exists()

    def test_plot_distribution(self, tmp_path: Path) -> None:
        """Attention distribution box plot is saved to disk."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batches = [_make_batch(2) for _ in range(3)]
        stacked = viz.collect_attention_weights(batches, max_batches=3)
        path = viz.plot_attention_distribution(
            stacked, tmp_path / "distribution.png",
        )
        assert path.exists()

    def test_get_attention_summary(self) -> None:
        """Attention summary returns correct structure."""
        from src.explainability.attention_viz import AttentionVisualizer

        model = _make_model("attention")
        viz = AttentionVisualizer(model)
        batch = _make_batch()
        weights = viz.extract_attention_weights(batch)
        assert weights is not None
        summary = viz.get_attention_summary(weights)
        assert "avg_attention_matrix" in summary
        assert "modality_names" in summary
        assert "per_modality_stats" in summary
        assert len(summary["modality_names"]) == 5
        for name in summary["modality_names"]:
            stats = summary["per_modality_stats"][name]
            assert "mean" in stats
            assert "std" in stats


# ═══════════════════════════════════════════════════════════════════════════
# LIME Explainer
# ═══════════════════════════════════════════════════════════════════════════


class TestLIMEExplainer:
    """Tests for the LIME explainer."""

    def test_init(self) -> None:
        """LIMEExplainer initialises without error."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        assert explainer._total_dim == sum(MODALITY_DIMS.values())

    def test_predict_fn(self) -> None:
        """Internal predict function returns valid probabilities."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        flat = np.random.randn(3, sum(MODALITY_DIMS.values())).astype(np.float32)
        probs = explainer._predict_fn(flat)
        assert probs.shape == (3, NUM_CLASSES)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (probs >= 0).all()

    def test_explain_instance_runs(self) -> None:
        """LIME explanation runs without error for a single sample."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        sample = _make_batch(1)
        result = explainer.explain_instance(
            sample, num_features=5, num_samples=50,
        )
        assert "explanation" in result
        assert "top_features" in result
        assert "predicted_class" in result
        assert 0 <= result["predicted_class"] < NUM_CLASSES

    def test_explain_instance_top_features(self) -> None:
        """Top features returned for each class."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        sample = _make_batch(1)
        result = explainer.explain_instance(
            sample, num_features=5, num_samples=50,
        )
        assert len(result["top_features"]) > 0
        for class_name, features in result["top_features"].items():
            assert isinstance(features, list)
            for feat_name, weight in features:
                assert isinstance(feat_name, str)
                assert isinstance(weight, float)

    def test_explain_with_custom_feature_names(self) -> None:
        """LIME works with custom feature names."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        sample = _make_batch(1)
        custom_names = {
            "mutation": [f"mut_feat_{i}" for i in range(MUTATION_DIM)],
        }
        result = explainer.explain_instance(
            sample, feature_names=custom_names, num_features=5, num_samples=50,
        )
        assert "top_features" in result

    def test_explain_with_training_data(self) -> None:
        """LIME works when training data is provided."""
        from src.explainability.lime_explainer import LIMEExplainer

        model = _make_model()
        explainer = LIMEExplainer(model, MODALITY_DIMS)
        sample = _make_batch(1)
        training_data = np.random.randn(20, sum(MODALITY_DIMS.values())).astype(
            np.float32,
        )
        result = explainer.explain_instance(
            sample, training_data=training_data, num_features=5, num_samples=50,
        )
        assert "predicted_class" in result
