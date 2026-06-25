"""Tests for all modality encoders.

Covers output shape verification, batch-size edge cases, gradient flow, and
``get_output_dim`` consistency for every encoder variant.
"""

from __future__ import annotations

import pytest
import torch

from src.models.base import BaseModel
from src.models.encoders.mutation_encoder import (
    MutationEncoder,
    MutationTransformerEncoder,
)
from src.models.encoders.expression_encoder import (
    DenseAutoencoder,
    ExpressionTransformerEncoder,
    VariationalAutoencoder,
)
from src.models.encoders.methylation_encoder import (
    MethylationDenseAutoencoder,
    MethylationTransformerEncoder,
    MethylationVAE,
)
from src.models.encoders.cnv_encoder import CNVAttentionEncoder, CNVFCEncoder
from src.models.encoders.clinical_encoder import ClinicalEncoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MUTATION_DIM = 42
EXPRESSION_DIM = 2000
METHYLATION_DIM = 2000
CNV_DIM = 200
CLINICAL_DIM = 32


@pytest.fixture()
def mutation_batch_1() -> torch.Tensor:
    return torch.randn(1, MUTATION_DIM)


@pytest.fixture()
def mutation_batch_32() -> torch.Tensor:
    return torch.randn(32, MUTATION_DIM)


@pytest.fixture()
def expression_batch_1() -> torch.Tensor:
    return torch.randn(1, EXPRESSION_DIM)


@pytest.fixture()
def expression_batch_32() -> torch.Tensor:
    return torch.randn(32, EXPRESSION_DIM)


@pytest.fixture()
def methylation_batch_1() -> torch.Tensor:
    return torch.randn(1, METHYLATION_DIM)


@pytest.fixture()
def methylation_batch_32() -> torch.Tensor:
    return torch.randn(32, METHYLATION_DIM)


@pytest.fixture()
def cnv_batch_1() -> torch.Tensor:
    return torch.randn(1, CNV_DIM)


@pytest.fixture()
def cnv_batch_32() -> torch.Tensor:
    return torch.randn(32, CNV_DIM)


@pytest.fixture()
def clinical_batch_1() -> torch.Tensor:
    return torch.randn(1, CLINICAL_DIM)


@pytest.fixture()
def clinical_batch_32() -> torch.Tensor:
    return torch.randn(32, CLINICAL_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# BaseModel contract
# ═══════════════════════════════════════════════════════════════════════════


class TestBaseModel:
    """Verify the BaseModel ABC helpers."""

    def test_count_parameters(self) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=128)
        total = enc.count_parameters(trainable_only=True)
        assert total > 0
        assert enc.count_parameters(trainable_only=False) >= total

    def test_get_device(self) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=128)
        assert enc.get_device() == torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
# MutationEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestMutationEncoder:
    """Tests for the MLP-based MutationEncoder."""

    def test_output_shape_batch_1(self, mutation_batch_1: torch.Tensor) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=128)
        enc.eval()
        out = enc(mutation_batch_1)
        assert out.shape == (1, 128)

    def test_output_shape_batch_32(self, mutation_batch_32: torch.Tensor) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=128)
        out = enc(mutation_batch_32)
        assert out.shape == (32, 128)

    def test_get_output_dim(self) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=64)
        assert enc.get_output_dim() == 64

    def test_gradient_flow(self, mutation_batch_32: torch.Tensor) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=128)
        out = enc(mutation_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_custom_embed_dim(self, mutation_batch_32: torch.Tensor) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=256)
        out = enc(mutation_batch_32)
        assert out.shape == (32, 256)
        assert enc.get_output_dim() == 256

    def test_variable_input_size(self) -> None:
        for dim in [10, 42, 100]:
            enc = MutationEncoder(dim, embed_dim=128)
            x = torch.randn(4, dim)
            out = enc(x)
            assert out.shape == (4, 128)


# ═══════════════════════════════════════════════════════════════════════════
# MutationTransformerEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestMutationTransformerEncoder:
    """Tests for the Transformer-based mutation encoder."""

    def test_output_shape_batch_1(self, mutation_batch_1: torch.Tensor) -> None:
        enc = MutationTransformerEncoder(
            MUTATION_DIM, embed_dim=128, group_sizes=(9, 5, 3, 25)
        )
        out = enc(mutation_batch_1)
        assert out.shape == (1, 128)

    def test_output_shape_batch_32(self, mutation_batch_32: torch.Tensor) -> None:
        enc = MutationTransformerEncoder(
            MUTATION_DIM, embed_dim=128, group_sizes=(9, 5, 3, 25)
        )
        out = enc(mutation_batch_32)
        assert out.shape == (32, 128)

    def test_get_output_dim(self) -> None:
        enc = MutationTransformerEncoder(
            MUTATION_DIM, embed_dim=64, group_sizes=(9, 5, 3, 25)
        )
        assert enc.get_output_dim() == 64

    def test_gradient_flow(self, mutation_batch_32: torch.Tensor) -> None:
        enc = MutationTransformerEncoder(
            MUTATION_DIM, embed_dim=128, group_sizes=(9, 5, 3, 25)
        )
        out = enc(mutation_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_auto_group_sizes(self) -> None:
        enc = MutationTransformerEncoder(40, embed_dim=64)
        assert sum(enc.group_sizes) == 40
        out = enc(torch.randn(4, 40))
        assert out.shape == (4, 64)

    def test_invalid_group_sizes(self) -> None:
        with pytest.raises(ValueError, match="group_sizes sum"):
            MutationTransformerEncoder(
                MUTATION_DIM, embed_dim=128, group_sizes=(10, 10, 10, 10)
            )


# ═══════════════════════════════════════════════════════════════════════════
# Expression DenseAutoencoder
# ═══════════════════════════════════════════════════════════════════════════


class TestDenseAutoencoder:
    """Tests for the expression DenseAutoencoder."""

    def test_encode_shape_batch_1(self, expression_batch_1: torch.Tensor) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=256)
        enc.eval()
        z = enc.encode(expression_batch_1)
        assert z.shape == (1, 256)

    def test_encode_shape_batch_32(self, expression_batch_32: torch.Tensor) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=256)
        z = enc.encode(expression_batch_32)
        assert z.shape == (32, 256)

    def test_forward_returns_dict(self, expression_batch_32: torch.Tensor) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=256)
        result = enc(expression_batch_32)
        assert "embedding" in result
        assert "reconstruction" in result
        assert result["embedding"].shape == (32, 256)
        assert result["reconstruction"].shape == (32, EXPRESSION_DIM)

    def test_get_output_dim(self) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=128)
        assert enc.get_output_dim() == 128

    def test_gradient_flow(self, expression_batch_32: torch.Tensor) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=256)
        result = enc(expression_batch_32)
        loss = result["embedding"].sum() + result["reconstruction"].sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_variable_input_size(self) -> None:
        for dim in [500, 1000, 2000]:
            enc = DenseAutoencoder(dim, embed_dim=256)
            x = torch.randn(4, dim)
            z = enc.encode(x)
            assert z.shape == (4, 256)


# ═══════════════════════════════════════════════════════════════════════════
# Expression VariationalAutoencoder
# ═══════════════════════════════════════════════════════════════════════════


class TestVariationalAutoencoder:
    """Tests for the expression VAE."""

    def test_encode_shape_batch_1(self, expression_batch_1: torch.Tensor) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=256)
        enc.eval()
        z = enc.encode(expression_batch_1)
        assert z.shape == (1, 256)

    def test_encode_shape_batch_32(self, expression_batch_32: torch.Tensor) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=256)
        z = enc.encode(expression_batch_32)
        assert z.shape == (32, 256)

    def test_forward_returns_vae_outputs(
        self, expression_batch_32: torch.Tensor
    ) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=256)
        result = enc(expression_batch_32)
        assert "embedding" in result
        assert "mu" in result
        assert "logvar" in result
        assert "reconstruction" in result
        assert result["embedding"].shape == (32, 256)
        assert result["mu"].shape == (32, 256)
        assert result["logvar"].shape == (32, 256)
        assert result["reconstruction"].shape == (32, EXPRESSION_DIM)

    def test_get_output_dim(self) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=128)
        assert enc.get_output_dim() == 128

    def test_gradient_flow(self, expression_batch_32: torch.Tensor) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=256)
        result = enc(expression_batch_32)
        kl = -0.5 * torch.sum(
            1 + result["logvar"] - result["mu"].pow(2) - result["logvar"].exp()
        )
        loss = result["reconstruction"].sum() + kl
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_stochasticity(self, expression_batch_32: torch.Tensor) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=256)
        enc.train()
        z1 = enc.encode(expression_batch_32)
        z2 = enc.encode(expression_batch_32)
        assert not torch.allclose(z1, z2), "VAE samples should differ"


# ═══════════════════════════════════════════════════════════════════════════
# Expression TransformerEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestExpressionTransformerEncoder:
    """Tests for the Transformer-based expression encoder."""

    def test_output_shape_batch_1(self, expression_batch_1: torch.Tensor) -> None:
        enc = ExpressionTransformerEncoder(EXPRESSION_DIM, embed_dim=256)
        out = enc(expression_batch_1)
        assert out.shape == (1, 256)

    def test_output_shape_batch_32(self, expression_batch_32: torch.Tensor) -> None:
        enc = ExpressionTransformerEncoder(EXPRESSION_DIM, embed_dim=256)
        out = enc(expression_batch_32)
        assert out.shape == (32, 256)

    def test_get_output_dim(self) -> None:
        enc = ExpressionTransformerEncoder(EXPRESSION_DIM, embed_dim=128)
        assert enc.get_output_dim() == 128

    def test_gradient_flow(self, expression_batch_32: torch.Tensor) -> None:
        enc = ExpressionTransformerEncoder(EXPRESSION_DIM, embed_dim=256)
        out = enc(expression_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_non_divisible_input(self) -> None:
        enc = ExpressionTransformerEncoder(1999, embed_dim=128, group_size=50)
        x = torch.randn(4, 1999)
        out = enc(x)
        assert out.shape == (4, 128)


# ═══════════════════════════════════════════════════════════════════════════
# Methylation DenseAutoencoder
# ═══════════════════════════════════════════════════════════════════════════


class TestMethylationDenseAutoencoder:
    """Tests for the methylation DenseAutoencoder."""

    def test_encode_shape_batch_1(self, methylation_batch_1: torch.Tensor) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=128)
        enc.eval()
        z = enc.encode(methylation_batch_1)
        assert z.shape == (1, 128)

    def test_encode_shape_batch_32(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=128)
        z = enc.encode(methylation_batch_32)
        assert z.shape == (32, 128)

    def test_forward_returns_dict(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=128)
        result = enc(methylation_batch_32)
        assert "embedding" in result
        assert "reconstruction" in result
        assert result["embedding"].shape == (32, 128)
        assert result["reconstruction"].shape == (32, METHYLATION_DIM)

    def test_get_output_dim(self) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=64)
        assert enc.get_output_dim() == 64

    def test_gradient_flow(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=128)
        result = enc(methylation_batch_32)
        loss = result["embedding"].sum() + result["reconstruction"].sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# Methylation VAE
# ═══════════════════════════════════════════════════════════════════════════


class TestMethylationVAE:
    """Tests for the methylation VAE."""

    def test_encode_shape_batch_1(self, methylation_batch_1: torch.Tensor) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=128)
        enc.eval()
        z = enc.encode(methylation_batch_1)
        assert z.shape == (1, 128)

    def test_encode_shape_batch_32(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=128)
        z = enc.encode(methylation_batch_32)
        assert z.shape == (32, 128)

    def test_forward_returns_vae_outputs(
        self, methylation_batch_32: torch.Tensor
    ) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=128)
        result = enc(methylation_batch_32)
        assert "embedding" in result
        assert "mu" in result
        assert "logvar" in result
        assert "reconstruction" in result
        assert result["embedding"].shape == (32, 128)
        assert result["mu"].shape == (32, 128)
        assert result["logvar"].shape == (32, 128)
        assert result["reconstruction"].shape == (32, METHYLATION_DIM)

    def test_get_output_dim(self) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=64)
        assert enc.get_output_dim() == 64

    def test_gradient_flow(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=128)
        result = enc(methylation_batch_32)
        kl = -0.5 * torch.sum(
            1 + result["logvar"] - result["mu"].pow(2) - result["logvar"].exp()
        )
        loss = result["reconstruction"].sum() + kl
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# Methylation TransformerEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestMethylationTransformerEncoder:
    """Tests for the Transformer-based methylation encoder."""

    def test_output_shape_batch_1(self, methylation_batch_1: torch.Tensor) -> None:
        enc = MethylationTransformerEncoder(METHYLATION_DIM, embed_dim=128)
        out = enc(methylation_batch_1)
        assert out.shape == (1, 128)

    def test_output_shape_batch_32(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationTransformerEncoder(METHYLATION_DIM, embed_dim=128)
        out = enc(methylation_batch_32)
        assert out.shape == (32, 128)

    def test_get_output_dim(self) -> None:
        enc = MethylationTransformerEncoder(METHYLATION_DIM, embed_dim=64)
        assert enc.get_output_dim() == 64

    def test_gradient_flow(self, methylation_batch_32: torch.Tensor) -> None:
        enc = MethylationTransformerEncoder(METHYLATION_DIM, embed_dim=128)
        out = enc(methylation_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ═══════════════════════════════════════════════════════════════════════════
# CNV FC Encoder
# ═══════════════════════════════════════════════════════════════════════════


class TestCNVFCEncoder:
    """Tests for the MLP-based CNV encoder."""

    def test_output_shape_batch_1(self, cnv_batch_1: torch.Tensor) -> None:
        enc = CNVFCEncoder(CNV_DIM, embed_dim=64)
        enc.eval()
        out = enc(cnv_batch_1)
        assert out.shape == (1, 64)

    def test_output_shape_batch_32(self, cnv_batch_32: torch.Tensor) -> None:
        enc = CNVFCEncoder(CNV_DIM, embed_dim=64)
        out = enc(cnv_batch_32)
        assert out.shape == (32, 64)

    def test_get_output_dim(self) -> None:
        enc = CNVFCEncoder(CNV_DIM, embed_dim=32)
        assert enc.get_output_dim() == 32

    def test_gradient_flow(self, cnv_batch_32: torch.Tensor) -> None:
        enc = CNVFCEncoder(CNV_DIM, embed_dim=64)
        out = enc(cnv_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_variable_input_size(self) -> None:
        for dim in [50, 200, 500]:
            enc = CNVFCEncoder(dim, embed_dim=64)
            x = torch.randn(4, dim)
            out = enc(x)
            assert out.shape == (4, 64)


# ═══════════════════════════════════════════════════════════════════════════
# CNV Attention Encoder
# ═══════════════════════════════════════════════════════════════════════════


class TestCNVAttentionEncoder:
    """Tests for the self-attention CNV encoder."""

    def test_output_shape_batch_1(self, cnv_batch_1: torch.Tensor) -> None:
        enc = CNVAttentionEncoder(CNV_DIM, embed_dim=64)
        out = enc(cnv_batch_1)
        assert out.shape == (1, 64)

    def test_output_shape_batch_32(self, cnv_batch_32: torch.Tensor) -> None:
        enc = CNVAttentionEncoder(CNV_DIM, embed_dim=64)
        out = enc(cnv_batch_32)
        assert out.shape == (32, 64)

    def test_get_output_dim(self) -> None:
        enc = CNVAttentionEncoder(CNV_DIM, embed_dim=32)
        assert enc.get_output_dim() == 32

    def test_gradient_flow(self, cnv_batch_32: torch.Tensor) -> None:
        enc = CNVAttentionEncoder(CNV_DIM, embed_dim=64)
        out = enc(cnv_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_variable_input_size(self) -> None:
        for dim in [50, 200, 500]:
            enc = CNVAttentionEncoder(dim, embed_dim=64)
            x = torch.randn(4, dim)
            out = enc(x)
            assert out.shape == (4, 64)


# ═══════════════════════════════════════════════════════════════════════════
# Clinical Encoder
# ═══════════════════════════════════════════════════════════════════════════


class TestClinicalEncoder:
    """Tests for the MLP-based ClinicalEncoder."""

    def test_output_shape_batch_1(self, clinical_batch_1: torch.Tensor) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=32)
        enc.eval()
        out = enc(clinical_batch_1)
        assert out.shape == (1, 32)

    def test_output_shape_batch_32(self, clinical_batch_32: torch.Tensor) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=32)
        out = enc(clinical_batch_32)
        assert out.shape == (32, 32)

    def test_get_output_dim(self) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=16)
        assert enc.get_output_dim() == 16

    def test_gradient_flow(self, clinical_batch_32: torch.Tensor) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=32)
        out = enc(clinical_batch_32)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_custom_embed_dim(self, clinical_batch_32: torch.Tensor) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=64)
        out = enc(clinical_batch_32)
        assert out.shape == (32, 64)
        assert enc.get_output_dim() == 64

    def test_variable_input_size(self) -> None:
        for dim in [8, 16, 32, 64]:
            enc = ClinicalEncoder(dim, embed_dim=32)
            x = torch.randn(4, dim)
            out = enc(x)
            assert out.shape == (4, 32)

    def test_default_embed_dim(self) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM)
        assert enc.get_output_dim() == 32

    def test_count_parameters_small(self) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=32)
        params = enc.count_parameters()
        assert params < 10_000, "Clinical encoder should be compact"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-encoder consistency: get_output_dim matches actual output
# ═══════════════════════════════════════════════════════════════════════════


class TestOutputDimConsistency:
    """Verify that get_output_dim matches the actual forward-pass output."""

    @pytest.mark.parametrize("embed_dim", [64, 128, 256])
    def test_mutation_encoder(self, embed_dim: int) -> None:
        enc = MutationEncoder(MUTATION_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, MUTATION_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [64, 128, 256])
    def test_mutation_transformer(self, embed_dim: int) -> None:
        enc = MutationTransformerEncoder(MUTATION_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, MUTATION_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [128, 256])
    def test_dense_autoencoder(self, embed_dim: int) -> None:
        enc = DenseAutoencoder(EXPRESSION_DIM, embed_dim=embed_dim)
        z = enc.encode(torch.randn(2, EXPRESSION_DIM))
        assert z.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [128, 256])
    def test_vae(self, embed_dim: int) -> None:
        enc = VariationalAutoencoder(EXPRESSION_DIM, embed_dim=embed_dim)
        z = enc.encode(torch.randn(2, EXPRESSION_DIM))
        assert z.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [128, 256])
    def test_expression_transformer(self, embed_dim: int) -> None:
        enc = ExpressionTransformerEncoder(EXPRESSION_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, EXPRESSION_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [64, 128])
    def test_methylation_dense_autoencoder(self, embed_dim: int) -> None:
        enc = MethylationDenseAutoencoder(METHYLATION_DIM, embed_dim=embed_dim)
        z = enc.encode(torch.randn(2, METHYLATION_DIM))
        assert z.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [64, 128])
    def test_methylation_vae(self, embed_dim: int) -> None:
        enc = MethylationVAE(METHYLATION_DIM, embed_dim=embed_dim)
        z = enc.encode(torch.randn(2, METHYLATION_DIM))
        assert z.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [64, 128])
    def test_methylation_transformer(self, embed_dim: int) -> None:
        enc = MethylationTransformerEncoder(METHYLATION_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, METHYLATION_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [32, 64])
    def test_cnv_fc(self, embed_dim: int) -> None:
        enc = CNVFCEncoder(CNV_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, CNV_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [32, 64])
    def test_cnv_attention(self, embed_dim: int) -> None:
        enc = CNVAttentionEncoder(CNV_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, CNV_DIM))
        assert out.shape[1] == enc.get_output_dim()

    @pytest.mark.parametrize("embed_dim", [16, 32, 64])
    def test_clinical_encoder(self, embed_dim: int) -> None:
        enc = ClinicalEncoder(CLINICAL_DIM, embed_dim=embed_dim)
        out = enc(torch.randn(2, CLINICAL_DIM))
        assert out.shape[1] == enc.get_output_dim()
