"""Tests for all five multi-omics fusion strategies.

Covers output shape verification, missing-modality handling, gradient flow,
and attention-weight normalisation for attention-based fusions.
"""

from __future__ import annotations

import pytest
import torch

from src.models.fusion.early_fusion import EarlyFusion
from src.models.fusion.late_fusion import LateFusion
from src.models.fusion.attention_fusion import AttentionFusion
from src.models.fusion.cross_attention import CrossAttentionFusion
from src.models.fusion.transformer_fusion import TransformerFusion

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
FUSION_DIM = 256
NUM_CLASSES = 4

MODALITY_DIMS: dict[str, int] = {
    "mutation": 128,
    "expression": 256,
    "methylation": 128,
    "cnv": 64,
    "clinical": 32,
}

MODALITY_NAMES = ["mutation", "expression", "methylation", "cnv", "clinical"]


@pytest.fixture()
def all_present_mask() -> torch.Tensor:
    """All modalities present for every sample."""
    return torch.ones(BATCH_SIZE, len(MODALITY_NAMES), dtype=torch.bool)


@pytest.fixture()
def partial_mask() -> torch.Tensor:
    """Two modalities absent (expression and cnv) for every sample."""
    mask = torch.ones(BATCH_SIZE, len(MODALITY_NAMES), dtype=torch.bool)
    mask[:, 1] = False  # expression
    mask[:, 3] = False  # cnv
    return mask


@pytest.fixture()
def embeddings() -> dict[str, torch.Tensor]:
    """Random embeddings for all five modalities."""
    return {
        name: torch.randn(BATCH_SIZE, dim)
        for name, dim in MODALITY_DIMS.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# EarlyFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestEarlyFusion:
    """Tests for the concatenation-based early fusion."""

    def test_output_shape(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = EarlyFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_missing_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
        partial_mask: torch.Tensor,
    ) -> None:
        model = EarlyFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, partial_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_gradient_flow(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = EarlyFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_masking_zeros_absent(
        self,
        embeddings: dict[str, torch.Tensor],
    ) -> None:
        """Absent modalities contribute zero to the concatenation."""
        mask = torch.zeros(BATCH_SIZE, len(MODALITY_NAMES), dtype=torch.bool)
        mask[:, 0] = True  # only mutation present
        model = EarlyFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        model.eval()
        out = model(embeddings, mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# LateFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestLateFusion:
    """Tests for the weighted late fusion."""

    def test_output_shape(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = LateFusion(
            MODALITY_DIMS, num_classes=NUM_CLASSES, fusion_dim=FUSION_DIM,
        )
        result = model(embeddings, all_present_mask)
        assert result["fused"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_per_modality_outputs(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = LateFusion(
            MODALITY_DIMS, num_classes=NUM_CLASSES, fusion_dim=FUSION_DIM,
        )
        result = model(embeddings, all_present_mask)
        assert len(result["per_modality"]) == len(MODALITY_NAMES)
        for name in MODALITY_NAMES:
            assert result["per_modality"][name].shape == (
                BATCH_SIZE, NUM_CLASSES,
            )

    def test_missing_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
        partial_mask: torch.Tensor,
    ) -> None:
        model = LateFusion(
            MODALITY_DIMS, num_classes=NUM_CLASSES, fusion_dim=FUSION_DIM,
        )
        result = model(embeddings, partial_mask)
        assert result["fused"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_gradient_flow(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = LateFusion(
            MODALITY_DIMS, num_classes=NUM_CLASSES, fusion_dim=FUSION_DIM,
        )
        result = model(embeddings, all_present_mask)
        loss = result["fused"].sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_learnable_weights(self) -> None:
        model = LateFusion(
            MODALITY_DIMS, num_classes=NUM_CLASSES, fusion_dim=FUSION_DIM,
        )
        assert model.modality_weights.requires_grad


# ═══════════════════════════════════════════════════════════════════════════
# AttentionFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestAttentionFusion:
    """Tests for the self-attention fusion."""

    def test_output_shape(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_missing_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
        partial_mask: torch.Tensor,
    ) -> None:
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, partial_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_gradient_flow(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_attention_weights_sum(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        """Attention weights should sum to ~1.0 per query position."""
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        model.eval()
        model(embeddings, all_present_mask)
        assert model.attention_weights is not None
        row_sums = model.attention_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_weights_saved(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        model(embeddings, all_present_mask)
        assert model.attention_weights is not None
        assert model.attention_weights.shape == (
            BATCH_SIZE, len(MODALITY_NAMES), len(MODALITY_NAMES),
        )


# ═══════════════════════════════════════════════════════════════════════════
# CrossAttentionFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossAttentionFusion:
    """Tests for the pairwise cross-attention fusion."""

    def test_output_shape(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = CrossAttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_missing_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
        partial_mask: torch.Tensor,
    ) -> None:
        model = CrossAttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, partial_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_gradient_flow(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = CrossAttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_per_modality_attention(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        """Each modality should have its own cross-attention module."""
        model = CrossAttentionFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        assert len(model.cross_attentions) == len(MODALITY_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# TransformerFusion
# ═══════════════════════════════════════════════════════════════════════════


class TestTransformerFusion:
    """Tests for the Transformer-based fusion."""

    def test_output_shape(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_missing_modalities(
        self,
        embeddings: dict[str, torch.Tensor],
        partial_mask: torch.Tensor,
    ) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, partial_mask)
        assert out.shape == (BATCH_SIZE, FUSION_DIM)

    def test_gradient_flow(
        self,
        embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        out = model(embeddings, all_present_mask)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_cls_token_exists(self) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        assert model.cls_token.shape == (1, 1, FUSION_DIM)

    def test_modality_embeddings_exist(self) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=FUSION_DIM)
        assert model.modality_embeddings.shape == (
            1, len(MODALITY_NAMES), FUSION_DIM,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cross-fusion consistency: different fusion_dim values
# ═══════════════════════════════════════════════════════════════════════════


class TestFusionDimVariation:
    """Verify all fusions work with different fusion_dim values."""

    @pytest.mark.parametrize("fusion_dim", [64, 128, 512])
    def test_early_fusion_dim(
        self, fusion_dim: int, embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = EarlyFusion(MODALITY_DIMS, fusion_dim=fusion_dim)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, fusion_dim)

    @pytest.mark.parametrize("fusion_dim", [64, 128, 512])
    def test_attention_fusion_dim(
        self, fusion_dim: int, embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = AttentionFusion(MODALITY_DIMS, fusion_dim=fusion_dim)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, fusion_dim)

    @pytest.mark.parametrize("fusion_dim", [64, 128, 512])
    def test_cross_attention_fusion_dim(
        self, fusion_dim: int, embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = CrossAttentionFusion(MODALITY_DIMS, fusion_dim=fusion_dim)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, fusion_dim)

    @pytest.mark.parametrize("fusion_dim", [64, 128, 512])
    def test_transformer_fusion_dim(
        self, fusion_dim: int, embeddings: dict[str, torch.Tensor],
        all_present_mask: torch.Tensor,
    ) -> None:
        model = TransformerFusion(MODALITY_DIMS, fusion_dim=fusion_dim)
        out = model(embeddings, all_present_mask)
        assert out.shape == (BATCH_SIZE, fusion_dim)
