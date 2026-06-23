"""Tests for the classification head and full PathogenicityPredictor model.

Covers forward/backward passes, all five fusion types, missing modalities,
parameter counts, and the ``from_config`` / ``summary`` utility methods.
"""

from __future__ import annotations

import pytest
import torch

from src.models.classifier import ClassificationHead
from src.models.full_model import PathogenicityPredictor
from src.utils.config import Config

# ---------------------------------------------------------------------------
# Shared constants (small dims for fast tests)
# ---------------------------------------------------------------------------

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

ALL_FUSION_TYPES = ["early", "late", "attention", "cross_attention", "transformer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# ClassificationHead
# ═══════════════════════════════════════════════════════════════════════════


class TestClassificationHead:
    """Tests for the standalone classification head."""

    def test_output_shape(self) -> None:
        head = ClassificationHead(fusion_dim=FUSION_DIM, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, FUSION_DIM)
        out = head(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_predict_proba_sums_to_one(self) -> None:
        head = ClassificationHead(fusion_dim=FUSION_DIM, num_classes=NUM_CLASSES)
        head.eval()
        x = torch.randn(BATCH_SIZE, FUSION_DIM)
        probs = head.predict_proba(x)
        assert probs.shape == (BATCH_SIZE, NUM_CLASSES)
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_predict_returns_valid_classes(self) -> None:
        head = ClassificationHead(fusion_dim=FUSION_DIM, num_classes=NUM_CLASSES)
        head.eval()
        x = torch.randn(BATCH_SIZE, FUSION_DIM)
        preds = head.predict(x)
        assert preds.shape == (BATCH_SIZE,)
        assert (preds >= 0).all()
        assert (preds < NUM_CLASSES).all()

    def test_gradient_flow(self) -> None:
        head = ClassificationHead(fusion_dim=FUSION_DIM, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, FUSION_DIM)
        out = head(x)
        loss = out.sum()
        loss.backward()
        for name, p in head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_different_fusion_dim(self) -> None:
        for dim in [32, 128, 256, 512]:
            head = ClassificationHead(fusion_dim=dim, num_classes=NUM_CLASSES)
            x = torch.randn(BATCH_SIZE, dim)
            out = head(x)
            assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_different_num_classes(self) -> None:
        for nc in [2, 4, 10]:
            head = ClassificationHead(fusion_dim=FUSION_DIM, num_classes=nc)
            x = torch.randn(BATCH_SIZE, FUSION_DIM)
            out = head(x)
            assert out.shape == (BATCH_SIZE, nc)


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — forward pass
# ═══════════════════════════════════════════════════════════════════════════


class TestPathogenicityPredictorForward:
    """Tests for the full model forward pass."""

    def test_forward_output_keys(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        expected_keys = {
            "logits", "probabilities", "predicted_class",
            "fused_embedding", "modality_embeddings", "attention_weights",
        }
        assert set(result.keys()) == expected_keys

    def test_forward_logits_shape(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_probabilities_shape(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["probabilities"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_probabilities_sum_to_one(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        model.eval()
        batch = _make_batch()
        result = model(batch)
        row_sums = result["probabilities"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_forward_predicted_class_shape(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["predicted_class"].shape == (BATCH_SIZE,)
        assert (result["predicted_class"] >= 0).all()
        assert (result["predicted_class"] < NUM_CLASSES).all()

    def test_forward_fused_embedding_shape(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["fused_embedding"].shape == (BATCH_SIZE, FUSION_DIM)

    def test_forward_modality_embeddings(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        embs = result["modality_embeddings"]
        assert set(embs.keys()) == {
            "mutation", "expression", "methylation", "cnv", "clinical",
        }
        assert embs["mutation"].shape == (BATCH_SIZE, MUTATION_EMBED)
        assert embs["expression"].shape == (BATCH_SIZE, EXPRESSION_EMBED)
        assert embs["methylation"].shape == (BATCH_SIZE, METHYLATION_EMBED)
        assert embs["cnv"].shape == (BATCH_SIZE, CNV_EMBED)
        assert embs["clinical"].shape == (BATCH_SIZE, CLINICAL_EMBED)


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — backward pass
# ═══════════════════════════════════════════════════════════════════════════


class TestPathogenicityPredictorBackward:
    """Tests for gradient flow through the full model."""

    def test_backward_all_params_have_grad(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(
            result["logits"], batch["label"],
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and ".decoder." not in name:
                assert p.grad is not None, f"No gradient for {name}"

    def test_decoder_params_no_grad_in_classification(self) -> None:
        """Autoencoder decoder params are not in the classification path."""
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(
            result["logits"], batch["label"],
        )
        loss.backward()
        decoder_params = [
            (n, p) for n, p in model.named_parameters()
            if ".decoder." in n
        ]
        assert len(decoder_params) > 0, "Expected decoder params to exist"
        for name, p in decoder_params:
            assert p.grad is None, f"Unexpected gradient for {name}"

    def test_backward_loss_is_scalar(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(
            result["logits"], batch["label"],
        )
        assert loss.dim() == 0

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_backward_all_fusion_types(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(
            result["logits"], batch["label"],
        )
        loss.backward()
        skip = {".decoder.", "embedding_projection."}
        for name, p in model.named_parameters():
            if p.requires_grad and not any(s in name for s in skip):
                assert p.grad is not None, (
                    f"No gradient for {name} with {fusion_type} fusion"
                )


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — all fusion types
# ═══════════════════════════════════════════════════════════════════════════


class TestAllFusionTypes:
    """Verify the model works end-to-end with every fusion strategy."""

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_forward_all_fusion_types(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)
        assert result["probabilities"].shape == (BATCH_SIZE, NUM_CLASSES)
        assert result["predicted_class"].shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_fused_embedding_shape(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["fused_embedding"].shape == (BATCH_SIZE, FUSION_DIM)

    def test_attention_fusion_has_weights(self) -> None:
        config = _make_config(fusion_type="attention")
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        result = model(batch)
        assert result["attention_weights"] is not None

    def test_non_attention_fusion_no_weights(self) -> None:
        for ft in ["early", "cross_attention", "transformer"]:
            config = _make_config(fusion_type=ft)
            model = PathogenicityPredictor(config)
            batch = _make_batch()
            result = model(batch)
            assert result["attention_weights"] is None


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — missing modalities
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingModalities:
    """Verify the model handles absent modalities gracefully."""

    def test_two_modalities_absent(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        batch["modality_mask"][:, 1] = False  # expression absent
        batch["modality_mask"][:, 3] = False  # cnv absent
        batch["expression"].zero_()
        batch["cnv"].zero_()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_only_mutation_present(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        mask = torch.zeros(BATCH_SIZE, 5, dtype=torch.bool)
        mask[:, 0] = True  # only mutation
        batch["modality_mask"] = mask
        batch["expression"].zero_()
        batch["methylation"].zero_()
        batch["cnv"].zero_()
        batch["clinical"].zero_()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_missing_modalities_all_fusions(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        batch["modality_mask"][:, 2] = False  # methylation absent
        batch["modality_mask"][:, 4] = False  # clinical absent
        batch["methylation"].zero_()
        batch["clinical"].zero_()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_backward_with_missing_modalities(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        batch = _make_batch()
        batch["modality_mask"][:, 1] = False
        batch["expression"].zero_()
        result = model(batch)
        loss = torch.nn.functional.cross_entropy(
            result["logits"], batch["label"],
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and ".decoder." not in name:
                assert p.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — parameter count
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterCount:
    """Verify parameter counts are reasonable."""

    def test_total_count_positive(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        total = model.count_parameters(trainable_only=True)
        assert total > 0

    def test_total_count_not_exploding(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        total = model.count_parameters(trainable_only=True)
        assert total < 5_000_000

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_count_per_fusion_type(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor(config)
        total = model.count_parameters(trainable_only=True)
        assert 1_000 < total < 5_000_000


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor — utility methods
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilityMethods:
    """Tests for from_config, summary, get_output_dim, and encode."""

    def test_from_config(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor.from_config(config)
        assert isinstance(model, PathogenicityPredictor)
        assert model.fusion_type == "cross_attention"
        assert model.num_classes == NUM_CLASSES
        assert model.fusion_dim == FUSION_DIM

    @pytest.mark.parametrize("fusion_type", ALL_FUSION_TYPES)
    def test_from_config_all_fusions(self, fusion_type: str) -> None:
        config = _make_config(fusion_type=fusion_type)
        model = PathogenicityPredictor.from_config(config)
        assert model.fusion_type == fusion_type
        batch = _make_batch()
        result = model(batch)
        assert result["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_summary_returns_string(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        text = model.summary()
        assert isinstance(text, str)
        assert "TOTAL" in text
        assert "params" in text

    def test_summary_lists_all_components(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        text = model.summary()
        for name in ["mutation", "expression", "methylation", "cnv", "clinical"]:
            assert name in text
        assert "Fusion" in text
        assert "Classifier" in text

    def test_get_output_dim(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        assert model.get_output_dim() == NUM_CLASSES

    def test_encode_raises(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        with pytest.raises(NotImplementedError):
            model.encode(torch.randn(4, 10))

    def test_get_device(self) -> None:
        config = _make_config()
        model = PathogenicityPredictor(config)
        assert model.get_device() == torch.device("cpu")
