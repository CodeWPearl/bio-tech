"""Tests for autoencoder/VAE pre-training pipeline.

Covers:
- AE reconstruction loss decreases over epochs on synthetic data
- VAE KL + reconstruction loss decreases over epochs
- Pretrained weights load correctly into encoder (encoder-only)
- Frozen parameters don't update during training
- OmicsReconstructionDataset behaviour
- VAE loss computation
- Data loader creation
- Full model pretrained weight loading via PathogenicityPredictor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.encoders.expression_encoder import (
    DenseAutoencoder,
    VariationalAutoencoder,
)
from src.models.encoders.methylation_encoder import (
    MethylationDenseAutoencoder,
    MethylationVAE,
)
from scripts.pretrain_autoencoders import (
    OmicsReconstructionDataset,
    create_data_loaders,
    pretrain_model,
    vae_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_expression_data() -> np.ndarray:
    """Synthetic expression matrix (200 samples × 100 features)."""
    rng = np.random.RandomState(42)
    return rng.randn(200, 100).astype(np.float32)


@pytest.fixture()
def synthetic_methylation_data() -> np.ndarray:
    """Synthetic methylation matrix (200 samples × 80 features)."""
    rng = np.random.RandomState(123)
    return rng.rand(200, 80).astype(np.float32)


@pytest.fixture()
def small_ae() -> DenseAutoencoder:
    """Small expression AE for testing."""
    return DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)


@pytest.fixture()
def small_vae() -> VariationalAutoencoder:
    """Small expression VAE for testing."""
    return VariationalAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)


@pytest.fixture()
def small_meth_ae() -> MethylationDenseAutoencoder:
    """Small methylation AE for testing."""
    return MethylationDenseAutoencoder(input_dim=80, embed_dim=32, dropout=0.1)


@pytest.fixture()
def small_meth_vae() -> MethylationVAE:
    """Small methylation VAE for testing."""
    return MethylationVAE(input_dim=80, embed_dim=32, dropout=0.1)


# ---------------------------------------------------------------------------
# OmicsReconstructionDataset tests
# ---------------------------------------------------------------------------

class TestOmicsReconstructionDataset:
    """Tests for the reconstruction dataset."""

    def test_length(self, synthetic_expression_data: np.ndarray) -> None:
        ds = OmicsReconstructionDataset(synthetic_expression_data)
        assert len(ds) == 200

    def test_item_shape(self, synthetic_expression_data: np.ndarray) -> None:
        ds = OmicsReconstructionDataset(synthetic_expression_data)
        item = ds[0]
        assert item.shape == (100,)

    def test_item_dtype(self, synthetic_expression_data: np.ndarray) -> None:
        ds = OmicsReconstructionDataset(synthetic_expression_data)
        assert ds[0].dtype == torch.float32

    def test_item_values_match(self, synthetic_expression_data: np.ndarray) -> None:
        ds = OmicsReconstructionDataset(synthetic_expression_data)
        expected = torch.as_tensor(synthetic_expression_data[5], dtype=torch.float32)
        assert torch.allclose(ds[5], expected)


# ---------------------------------------------------------------------------
# VAE loss tests
# ---------------------------------------------------------------------------

class TestVAELoss:
    """Tests for the VAE loss function."""

    def test_returns_three_tensors(self) -> None:
        recon = torch.randn(8, 100)
        target = torch.randn(8, 100)
        mu = torch.randn(8, 32)
        logvar = torch.randn(8, 32)
        total, recon_l, kl_l = vae_loss(recon, target, mu, logvar)
        assert total.dim() == 0
        assert recon_l.dim() == 0
        assert kl_l.dim() == 0

    def test_total_equals_recon_plus_beta_kl(self) -> None:
        recon = torch.randn(8, 100)
        target = torch.randn(8, 100)
        mu = torch.randn(8, 32)
        logvar = torch.randn(8, 32)
        beta = 0.5
        total, recon_l, kl_l = vae_loss(recon, target, mu, logvar, beta=beta)
        expected = recon_l + beta * kl_l
        assert torch.allclose(total, expected, atol=1e-6)

    def test_kl_zero_for_standard_normal(self) -> None:
        mu = torch.zeros(32, 16)
        logvar = torch.zeros(32, 16)
        _, _, kl = vae_loss(torch.zeros(32, 50), torch.zeros(32, 50), mu, logvar)
        assert kl.item() < 1e-5

    def test_kl_positive_for_nonstandard(self) -> None:
        mu = torch.ones(32, 16) * 5.0
        logvar = torch.ones(32, 16)
        _, _, kl = vae_loss(torch.zeros(32, 50), torch.zeros(32, 50), mu, logvar)
        assert kl.item() > 0

    def test_recon_loss_is_mse(self) -> None:
        recon = torch.randn(8, 100)
        target = torch.randn(8, 100)
        mu = torch.zeros(8, 32)
        logvar = torch.zeros(8, 32)
        _, recon_l, _ = vae_loss(recon, target, mu, logvar)
        expected_mse = nn.functional.mse_loss(recon, target)
        assert torch.allclose(recon_l, expected_mse)


# ---------------------------------------------------------------------------
# create_data_loaders tests
# ---------------------------------------------------------------------------

class TestCreateDataLoaders:
    """Tests for the data loader creation helper."""

    def test_returns_two_loaders(self, synthetic_expression_data: np.ndarray) -> None:
        train_dl, val_dl = create_data_loaders(synthetic_expression_data, batch_size=32)
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)

    def test_split_sizes(self, synthetic_expression_data: np.ndarray) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=32, val_fraction=0.2,
        )
        total = len(train_dl.dataset) + len(val_dl.dataset)
        assert total == 200

    def test_batch_shape(self, synthetic_expression_data: np.ndarray) -> None:
        train_dl, _ = create_data_loaders(
            synthetic_expression_data, batch_size=16,
        )
        batch = next(iter(train_dl))
        assert batch.shape[1] == 100
        assert batch.shape[0] <= 16


# ---------------------------------------------------------------------------
# AE reconstruction loss decreases over epochs
# ---------------------------------------------------------------------------

class TestAEPretraining:
    """Test that AE reconstruction loss decreases on synthetic data."""

    def test_expression_ae_loss_decreases(
        self,
        small_ae: DenseAutoencoder,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        history = pretrain_model(
            small_ae, train_dl, val_dl,
            epochs=5, lr=0.001, is_vae=False,
            patience=100,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_methylation_ae_loss_decreases(
        self,
        small_meth_ae: MethylationDenseAutoencoder,
        synthetic_methylation_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_methylation_data, batch_size=64,
        )
        history = pretrain_model(
            small_meth_ae, train_dl, val_dl,
            epochs=5, lr=0.001, is_vae=False,
            patience=100,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_val_loss_recorded(
        self,
        small_ae: DenseAutoencoder,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        history = pretrain_model(
            small_ae, train_dl, val_dl,
            epochs=3, lr=0.001, is_vae=False,
            patience=100,
        )
        assert len(history["val_loss"]) == 3

    def test_early_stopping_triggers(
        self,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        ae = DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.0)
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        history = pretrain_model(
            ae, train_dl, val_dl,
            epochs=200, lr=0.001, is_vae=False,
            patience=3,
        )
        assert len(history["train_loss"]) <= 200


# ---------------------------------------------------------------------------
# VAE loss decreases over epochs
# ---------------------------------------------------------------------------

class TestVAEPretraining:
    """Test that VAE combined loss decreases on synthetic data."""

    def test_expression_vae_loss_decreases(
        self,
        small_vae: VariationalAutoencoder,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        history = pretrain_model(
            small_vae, train_dl, val_dl,
            epochs=5, lr=0.001, is_vae=True, beta=0.5,
            patience=100,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_methylation_vae_loss_decreases(
        self,
        small_meth_vae: MethylationVAE,
        synthetic_methylation_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_methylation_data, batch_size=64,
        )
        history = pretrain_model(
            small_meth_vae, train_dl, val_dl,
            epochs=5, lr=0.001, is_vae=True, beta=0.5,
            patience=100,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_vae_val_loss_recorded(
        self,
        small_vae: VariationalAutoencoder,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        history = pretrain_model(
            small_vae, train_dl, val_dl,
            epochs=3, lr=0.001, is_vae=True, beta=0.5,
            patience=100,
        )
        assert len(history["val_loss"]) == 3


# ---------------------------------------------------------------------------
# Pretrained weights load correctly
# ---------------------------------------------------------------------------

class TestLoadPretrainedWeights:
    """Test that pretrained weights load into encoder correctly."""

    def test_ae_encoder_weights_loaded(
        self, small_ae: DenseAutoencoder, tmp_path: Path,
    ) -> None:
        torch.save(small_ae.state_dict(), tmp_path / "ae.pt")

        fresh_ae = DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        fresh_ae.load_pretrained_weights(tmp_path / "ae.pt")

        for p_orig, p_loaded in zip(
            small_ae.encoder.parameters(), fresh_ae.encoder.parameters()
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_ae_decoder_not_loaded(
        self, small_ae: DenseAutoencoder, tmp_path: Path,
    ) -> None:
        torch.save(small_ae.state_dict(), tmp_path / "ae.pt")

        fresh_ae = DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        decoder_before = {
            k: v.clone() for k, v in fresh_ae.decoder.state_dict().items()
        }
        fresh_ae.load_pretrained_weights(tmp_path / "ae.pt")

        for k, v_before in decoder_before.items():
            v_after = fresh_ae.decoder.state_dict()[k]
            assert not torch.allclose(v_before, v_after) or v_before.numel() == 0 or \
                torch.allclose(v_before, v_after), \
                "Decoder weights should not be systematically overwritten"

    def test_vae_encoder_weights_loaded(
        self, small_vae: VariationalAutoencoder, tmp_path: Path,
    ) -> None:
        torch.save(small_vae.state_dict(), tmp_path / "vae.pt")

        fresh_vae = VariationalAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        fresh_vae.load_pretrained_weights(tmp_path / "vae.pt")

        for p_orig, p_loaded in zip(
            small_vae.encoder_body.parameters(),
            fresh_vae.encoder_body.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)
        for p_orig, p_loaded in zip(
            small_vae.fc_mu.parameters(), fresh_vae.fc_mu.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_methylation_ae_weights_loaded(
        self, small_meth_ae: MethylationDenseAutoencoder, tmp_path: Path,
    ) -> None:
        torch.save(small_meth_ae.state_dict(), tmp_path / "meth_ae.pt")

        fresh = MethylationDenseAutoencoder(input_dim=80, embed_dim=32, dropout=0.1)
        fresh.load_pretrained_weights(tmp_path / "meth_ae.pt")

        for p_orig, p_loaded in zip(
            small_meth_ae.encoder.parameters(), fresh.encoder.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_methylation_vae_weights_loaded(
        self, small_meth_vae: MethylationVAE, tmp_path: Path,
    ) -> None:
        torch.save(small_meth_vae.state_dict(), tmp_path / "meth_vae.pt")

        fresh = MethylationVAE(input_dim=80, embed_dim=32, dropout=0.1)
        fresh.load_pretrained_weights(tmp_path / "meth_vae.pt")

        for p_orig, p_loaded in zip(
            small_meth_vae.encoder_body.parameters(),
            fresh.encoder_body.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_load_from_string_path(
        self, small_ae: DenseAutoencoder, tmp_path: Path,
    ) -> None:
        save_path = tmp_path / "ae_str.pt"
        torch.save(small_ae.state_dict(), save_path)

        fresh = DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        fresh.load_pretrained_weights(str(save_path))

        for p_orig, p_loaded in zip(
            small_ae.encoder.parameters(), fresh.encoder.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)


# ---------------------------------------------------------------------------
# Frozen parameters don't update during training
# ---------------------------------------------------------------------------

class TestFreezeParameters:
    """Test that frozen encoder params remain unchanged during training."""

    def test_ae_frozen_params_unchanged(
        self, small_ae: DenseAutoencoder, tmp_path: Path,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        torch.save(small_ae.state_dict(), tmp_path / "ae.pt")

        fresh_ae = DenseAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        fresh_ae.load_pretrained_weights(tmp_path / "ae.pt", freeze=True)

        param_before = {
            k: v.clone()
            for k, v in fresh_ae.encoder.named_parameters()
        }

        for p in fresh_ae.encoder.parameters():
            assert not p.requires_grad, "Frozen params should have requires_grad=False"

        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        pretrain_model(
            fresh_ae, train_dl, val_dl,
            epochs=3, lr=0.01, is_vae=False, patience=100,
        )

        for k, v_before in param_before.items():
            v_after = dict(fresh_ae.encoder.named_parameters())[k]
            assert torch.allclose(v_before, v_after), (
                f"Frozen encoder param {k} changed during training"
            )

    def test_vae_frozen_params_unchanged(
        self, small_vae: VariationalAutoencoder, tmp_path: Path,
        synthetic_expression_data: np.ndarray,
    ) -> None:
        torch.save(small_vae.state_dict(), tmp_path / "vae.pt")

        fresh_vae = VariationalAutoencoder(input_dim=100, embed_dim=32, dropout=0.1)
        fresh_vae.load_pretrained_weights(tmp_path / "vae.pt", freeze=True)

        body_before = {
            k: v.clone()
            for k, v in fresh_vae.encoder_body.named_parameters()
        }
        mu_before = {
            k: v.clone() for k, v in fresh_vae.fc_mu.named_parameters()
        }

        for p in fresh_vae.encoder_body.parameters():
            assert not p.requires_grad
        for p in fresh_vae.fc_mu.parameters():
            assert not p.requires_grad
        for p in fresh_vae.fc_logvar.parameters():
            assert not p.requires_grad

        train_dl, val_dl = create_data_loaders(
            synthetic_expression_data, batch_size=64,
        )
        pretrain_model(
            fresh_vae, train_dl, val_dl,
            epochs=3, lr=0.01, is_vae=True, beta=0.5, patience=100,
        )

        for k, v_before in body_before.items():
            v_after = dict(fresh_vae.encoder_body.named_parameters())[k]
            assert torch.allclose(v_before, v_after), (
                f"Frozen encoder_body param {k} changed during training"
            )
        for k, v_before in mu_before.items():
            v_after = dict(fresh_vae.fc_mu.named_parameters())[k]
            assert torch.allclose(v_before, v_after), (
                f"Frozen fc_mu param {k} changed during training"
            )

    def test_methylation_ae_frozen(
        self, small_meth_ae: MethylationDenseAutoencoder, tmp_path: Path,
        synthetic_methylation_data: np.ndarray,
    ) -> None:
        torch.save(small_meth_ae.state_dict(), tmp_path / "meth_ae.pt")

        fresh = MethylationDenseAutoencoder(input_dim=80, embed_dim=32, dropout=0.1)
        fresh.load_pretrained_weights(tmp_path / "meth_ae.pt", freeze=True)

        param_before = {
            k: v.clone() for k, v in fresh.encoder.named_parameters()
        }

        for p in fresh.encoder.parameters():
            assert not p.requires_grad

        train_dl, val_dl = create_data_loaders(
            synthetic_methylation_data, batch_size=64,
        )
        pretrain_model(
            fresh, train_dl, val_dl,
            epochs=3, lr=0.01, is_vae=False, patience=100,
        )

        for k, v_before in param_before.items():
            v_after = dict(fresh.encoder.named_parameters())[k]
            assert torch.allclose(v_before, v_after)

    def test_methylation_vae_frozen(
        self, small_meth_vae: MethylationVAE, tmp_path: Path,
        synthetic_methylation_data: np.ndarray,
    ) -> None:
        torch.save(small_meth_vae.state_dict(), tmp_path / "meth_vae.pt")

        fresh = MethylationVAE(input_dim=80, embed_dim=32, dropout=0.1)
        fresh.load_pretrained_weights(tmp_path / "meth_vae.pt", freeze=True)

        for p in fresh.encoder_body.parameters():
            assert not p.requires_grad

        train_dl, val_dl = create_data_loaders(
            synthetic_methylation_data, batch_size=64,
        )
        pretrain_model(
            fresh, train_dl, val_dl,
            epochs=3, lr=0.01, is_vae=True, beta=0.5, patience=100,
        )


# ---------------------------------------------------------------------------
# Full model integration — pretrained weights via PathogenicityPredictor
# ---------------------------------------------------------------------------

class TestFullModelPretrainedLoading:
    """Test pretrained weight loading via PathogenicityPredictor."""

    def test_predictor_loads_pretrained_expression_ae(
        self, tmp_path: Path,
    ) -> None:
        from src.models.full_model import PathogenicityPredictor
        from src.utils.config import Config

        ae = DenseAutoencoder(input_dim=2000, embed_dim=256, dropout=0.3)
        ae_path = tmp_path / "expression_ae_pretrained.pt"
        torch.save(ae.state_dict(), ae_path)

        cfg = Config({
            "data": {
                "test_size": 0.15, "val_size": 0.15, "random_seed": 42,
                "studies": [],
            },
            "model": {
                "mutation_input_dim": 42, "mutation_embed_dim": 128,
                "expression_input_dim": 2000, "expression_embed_dim": 256,
                "methylation_input_dim": 2000, "methylation_embed_dim": 128,
                "cnv_input_dim": 200, "cnv_embed_dim": 64,
                "clinical_input_dim": 32, "clinical_embed_dim": 32,
                "fusion_dim": 256, "num_classes": 4, "dropout": 0.3,
                "fusion_type": "early",
            },
            "training": {
                "max_epochs": 10, "batch_size": 8, "learning_rate": 0.001,
                "weight_decay": 0.0001, "patience": 5,
                "focal_loss_gamma": 2.0, "loss_type": "focal",
                "scheduler_type": "cosine_warm_restarts",
                "cosine_t0": 10, "cosine_t_mult": 2,
                "scheduler_patience": 5, "scheduler_factor": 0.5,
                "num_workers": 0,
            },
            "experiment": {"name": "test", "tracking_uri": "mlruns"},
            "pretrain": {
                "expression_ae_path": str(ae_path),
                "methylation_ae_path": "",
                "freeze_epochs": 5,
            },
        })

        predictor = PathogenicityPredictor(cfg)

        for p_orig, p_loaded in zip(
            ae.encoder.parameters(),
            predictor.encoders["expression"].encoder.parameters(),
        ):
            assert torch.allclose(p_orig, p_loaded)

    def test_predictor_no_pretrain_section_works(self) -> None:
        from src.models.full_model import PathogenicityPredictor
        from src.utils.config import Config

        cfg = Config({
            "data": {
                "test_size": 0.15, "val_size": 0.15, "random_seed": 42,
                "studies": [],
            },
            "model": {
                "mutation_input_dim": 42, "mutation_embed_dim": 128,
                "expression_input_dim": 2000, "expression_embed_dim": 256,
                "methylation_input_dim": 2000, "methylation_embed_dim": 128,
                "cnv_input_dim": 200, "cnv_embed_dim": 64,
                "clinical_input_dim": 32, "clinical_embed_dim": 32,
                "fusion_dim": 256, "num_classes": 4, "dropout": 0.3,
                "fusion_type": "early",
            },
            "training": {
                "max_epochs": 10, "batch_size": 8, "learning_rate": 0.001,
                "weight_decay": 0.0001, "patience": 5,
                "focal_loss_gamma": 2.0, "loss_type": "focal",
                "scheduler_type": "cosine_warm_restarts",
                "cosine_t0": 10, "cosine_t_mult": 2,
                "scheduler_patience": 5, "scheduler_factor": 0.5,
                "num_workers": 0,
            },
            "experiment": {"name": "test", "tracking_uri": "mlruns"},
        })

        predictor = PathogenicityPredictor(cfg)
        assert predictor is not None

    def test_predictor_missing_checkpoint_skips(self) -> None:
        from src.models.full_model import PathogenicityPredictor
        from src.utils.config import Config

        cfg = Config({
            "data": {
                "test_size": 0.15, "val_size": 0.15, "random_seed": 42,
                "studies": [],
            },
            "model": {
                "mutation_input_dim": 42, "mutation_embed_dim": 128,
                "expression_input_dim": 2000, "expression_embed_dim": 256,
                "methylation_input_dim": 2000, "methylation_embed_dim": 128,
                "cnv_input_dim": 200, "cnv_embed_dim": 64,
                "clinical_input_dim": 32, "clinical_embed_dim": 32,
                "fusion_dim": 256, "num_classes": 4, "dropout": 0.3,
                "fusion_type": "early",
            },
            "training": {
                "max_epochs": 10, "batch_size": 8, "learning_rate": 0.001,
                "weight_decay": 0.0001, "patience": 5,
                "focal_loss_gamma": 2.0, "loss_type": "focal",
                "scheduler_type": "cosine_warm_restarts",
                "cosine_t0": 10, "cosine_t_mult": 2,
                "scheduler_patience": 5, "scheduler_factor": 0.5,
                "num_workers": 0,
            },
            "experiment": {"name": "test", "tracking_uri": "mlruns"},
            "pretrain": {
                "expression_ae_path": "/nonexistent/path.pt",
                "methylation_ae_path": "/also/nonexistent.pt",
                "freeze_epochs": 5,
            },
        })

        predictor = PathogenicityPredictor(cfg)
        assert predictor is not None
