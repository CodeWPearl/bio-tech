"""Tests for the training infrastructure.

Covers loss functions, the Lightning module, and a full mini training loop.
"""

from __future__ import annotations

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from src.models.full_model import PathogenicityPredictor
from src.training.losses import FocalLoss, WeightedCrossEntropy
from src.training.callbacks import (
    EarlyStoppingWithPatience,
    GradientMonitor,
    MetricLogger,
)
from src.training.lightning_module import PathogenicityLightningModule
from src.training.scheduler import get_scheduler
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    """Build a minimal config for testing."""
    data = {
        "data": {
            "test_size": 0.15,
            "val_size": 0.15,
            "random_seed": 42,
            "studies": ["test_study"],
        },
        "model": {
            "mutation_input_dim": 10,
            "mutation_embed_dim": 16,
            "expression_input_dim": 20,
            "expression_embed_dim": 16,
            "methylation_input_dim": 20,
            "methylation_embed_dim": 16,
            "cnv_input_dim": 10,
            "cnv_embed_dim": 8,
            "clinical_input_dim": 8,
            "clinical_embed_dim": 8,
            "fusion_dim": 32,
            "num_classes": 4,
            "dropout": 0.1,
            "fusion_type": "early",
        },
        "training": {
            "max_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "patience": 5,
            "focal_loss_gamma": 2.0,
            "loss_type": "focal",
            "scheduler_type": "cosine_warm_restarts",
            "cosine_t0": 1,
            "cosine_t_mult": 1,
            "num_workers": 0,
        },
        "experiment": {
            "name": "test",
            "tracking_uri": "mlruns",
        },
    }
    for key, val in overrides.items():
        parts = key.split(".")
        d = data
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = val
    return Config(data)


def _make_batch(batch_size: int = 4, config: Config | None = None) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the model's expected input."""
    cfg = config or _make_config()
    m = cfg.model
    return {
        "mutation": torch.randn(batch_size, m.mutation_input_dim),
        "expression": torch.randn(batch_size, m.expression_input_dim),
        "methylation": torch.randn(batch_size, m.methylation_input_dim),
        "cnv": torch.randn(batch_size, m.cnv_input_dim),
        "clinical": torch.randn(batch_size, m.clinical_input_dim),
        "modality_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "label": torch.randint(0, m.num_classes, (batch_size,)),
    }


def _make_module(config: Config | None = None) -> PathogenicityLightningModule:
    """Build a Lightning module with a tiny model for testing."""
    cfg = config or _make_config()
    model = PathogenicityPredictor(cfg)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    return PathogenicityLightningModule(cfg, model, class_weights=weights)


# ---------------------------------------------------------------------------
# FocalLoss tests
# ---------------------------------------------------------------------------

class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_output_is_scalar(self) -> None:
        """Loss should return a scalar tensor."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_gradient_flows(self) -> None:
        """Gradients should flow through focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4, requires_grad=True)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_correct_gradient_with_known_inputs(self) -> None:
        """Focal loss with gamma=0 should match cross-entropy."""
        torch.manual_seed(42)
        logits = torch.randn(16, 4, requires_grad=True)
        targets = torch.randint(0, 4, (16,))

        focal_loss = FocalLoss(gamma=0.0)
        ce_loss = torch.nn.CrossEntropyLoss()

        fl_val = focal_loss(logits, targets)
        ce_val = ce_loss(logits, targets)

        assert torch.allclose(fl_val, ce_val, atol=1e-5)

    def test_with_alpha_weights(self) -> None:
        """Focal loss should accept per-class alpha weights."""
        alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_gamma_increases_focus(self) -> None:
        """Higher gamma should down-weight easy examples more."""
        torch.manual_seed(0)
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))

        loss_g0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)
        loss_g5 = FocalLoss(gamma=5.0)(logits, targets)

        assert loss_g0.item() >= loss_g2.item()
        assert loss_g2.item() >= loss_g5.item()

    def test_reduction_none(self) -> None:
        """reduction='none' should return per-sample losses."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (8,)

    def test_reduction_sum(self) -> None:
        """reduction='sum' should return sum of losses."""
        loss_fn_none = FocalLoss(gamma=2.0, reduction="none")
        loss_fn_sum = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        assert torch.allclose(
            loss_fn_sum(logits, targets),
            loss_fn_none(logits, targets).sum(),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# WeightedCrossEntropy tests
# ---------------------------------------------------------------------------

class TestWeightedCrossEntropy:
    """Tests for WeightedCrossEntropy."""

    def test_output_is_scalar(self) -> None:
        """Loss should return a scalar tensor."""
        loss_fn = WeightedCrossEntropy(
            class_weights=torch.tensor([1.0, 2.0, 1.5, 0.5]),
        )
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_from_label_counts(self) -> None:
        """Should compute weights from label counts."""
        loss_fn = WeightedCrossEntropy(label_counts=[1000, 100, 500, 200])
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_no_weights(self) -> None:
        """Should work with no weights (uniform)."""
        loss_fn = WeightedCrossEntropy()
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestScheduler:
    """Tests for the scheduler factory."""

    def test_cosine_warm_restarts(self) -> None:
        """Should create a CosineAnnealingWarmRestarts scheduler."""
        cfg = _make_config()
        opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=0.01)
        result = get_scheduler(opt, cfg)
        assert "scheduler" in result
        assert result["interval"] == "epoch"

    def test_reduce_on_plateau(self) -> None:
        """Should create a ReduceLROnPlateau scheduler."""
        cfg = _make_config(**{"training.scheduler_type": "reduce_on_plateau"})
        opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=0.01)
        result = get_scheduler(opt, cfg)
        assert "scheduler" in result
        assert result["monitor"] == "val_auroc"

    def test_one_cycle(self) -> None:
        """Should create a OneCycleLR scheduler."""
        cfg = _make_config(**{"training.scheduler_type": "one_cycle"})
        opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=0.01)
        result = get_scheduler(opt, cfg)
        assert "scheduler" in result
        assert result["interval"] == "step"

    def test_unknown_raises(self) -> None:
        """Unknown scheduler type should raise ValueError."""
        cfg = _make_config(**{"training.scheduler_type": "nonexistent"})
        opt = torch.optim.AdamW([torch.randn(2, requires_grad=True)], lr=0.01)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler(opt, cfg)


# ---------------------------------------------------------------------------
# Callbacks tests
# ---------------------------------------------------------------------------

class TestCallbacks:
    """Tests for custom callbacks."""

    def test_gradient_monitor_creation(self) -> None:
        """GradientMonitor should instantiate with default settings."""
        gm = GradientMonitor(log_every_n_steps=10)
        assert gm.log_every_n_steps == 10

    def test_early_stopping_creation(self) -> None:
        """EarlyStoppingWithPatience should monitor val_auroc."""
        es = EarlyStoppingWithPatience(patience=10)
        assert es.monitor == "val_auroc"
        assert es.mode == "max"
        assert es.patience == 10

    def test_metric_logger_creation(self) -> None:
        """MetricLogger should instantiate without errors."""
        ml = MetricLogger()
        assert ml is not None


# ---------------------------------------------------------------------------
# Lightning module tests
# ---------------------------------------------------------------------------

class TestLightningModule:
    """Tests for PathogenicityLightningModule."""

    def test_training_step_runs(self) -> None:
        """training_step should execute without error on a synthetic batch."""
        module = _make_module()
        module.train()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_training_step_gradient_flows(self) -> None:
        """Gradients should flow from training_step loss."""
        module = _make_module()
        module.train()
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        grad_count = sum(
            1 for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0

    def test_validation_step_runs(self) -> None:
        """validation_step should execute without error."""
        module = _make_module()
        module.eval()
        batch = _make_batch()
        module.validation_step(batch, batch_idx=0)

    def test_validation_metrics_computed(self) -> None:
        """Validation metrics should be updated after validation_step."""
        module = _make_module()
        module.eval()
        batch = _make_batch(batch_size=8)
        module.validation_step(batch, batch_idx=0)

        acc_val = module.val_accuracy.compute()
        assert 0.0 <= acc_val.item() <= 1.0

        f1_val = module.val_f1.compute()
        assert 0.0 <= f1_val.item() <= 1.0

    def test_test_step_runs(self) -> None:
        """test_step should execute without error."""
        module = _make_module()
        module.eval()
        batch = _make_batch()
        module.test_step(batch, batch_idx=0)

    def test_configure_optimizers(self) -> None:
        """configure_optimizers should return optimizer and scheduler."""
        module = _make_module()
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    @pytest.mark.parametrize("loss_type", ["focal", "weighted_ce", "ce"])
    def test_loss_types(self, loss_type: str) -> None:
        """All loss types should work in training_step."""
        cfg = _make_config(**{"training.loss_type": loss_type})
        module = _make_module(cfg)
        module.train()
        batch = _make_batch(config=cfg)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.item() > 0

    @pytest.mark.parametrize(
        "fusion_type",
        ["early", "late", "attention", "cross_attention", "transformer"],
    )
    def test_fusion_types(self, fusion_type: str) -> None:
        """Lightning module should work with all fusion types."""
        cfg = _make_config(**{"model.fusion_type": fusion_type})
        module = _make_module(cfg)
        module.train()
        batch = _make_batch(config=cfg)
        loss = module.training_step(batch, batch_idx=0)
        assert loss.item() > 0

    def test_modality_mask_expansion(self) -> None:
        """The 3-element mask should be expanded to 5 elements."""
        batch = _make_batch()
        assert batch["modality_mask"].shape[1] == 3
        expanded = PathogenicityLightningModule._expand_modality_mask(batch)
        assert expanded["modality_mask"].shape[1] == 5
        assert expanded["modality_mask"][:, 0].all()
        assert expanded["modality_mask"][:, 4].all()

    def test_unknown_loss_raises(self) -> None:
        """Unknown loss type should raise ValueError."""
        cfg = _make_config(**{"training.loss_type": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown loss type"):
            _make_module(cfg)


# ---------------------------------------------------------------------------
# Full training loop test
# ---------------------------------------------------------------------------

class _SyntheticDataModule(pl.LightningDataModule):
    """Tiny synthetic datamodule for integration testing."""

    def __init__(self, config: Config, n_samples: int = 32) -> None:
        super().__init__()
        self.config = config
        self.n_samples = n_samples
        self._data: dict[str, torch.Tensor] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create synthetic tensors."""
        m = self.config.model
        self._data = {
            "mutation": torch.randn(self.n_samples, m.mutation_input_dim),
            "expression": torch.randn(self.n_samples, m.expression_input_dim),
            "methylation": torch.randn(self.n_samples, m.methylation_input_dim),
            "cnv": torch.randn(self.n_samples, m.cnv_input_dim),
            "clinical": torch.randn(self.n_samples, m.clinical_input_dim),
            "modality_mask": torch.ones(self.n_samples, 3, dtype=torch.bool),
            "label": torch.randint(0, m.num_classes, (self.n_samples,)),
        }

    def _make_loader(self) -> DataLoader:
        """Build a DataLoader from synthetic data."""
        assert self._data is not None
        from src.data.dataset import collate_fn

        class _DictDataset(torch.utils.data.Dataset):
            def __init__(self, data: dict[str, torch.Tensor]) -> None:
                self.data = data
                self.n = data["label"].size(0)

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {k: v[idx] for k, v in self.data.items()}

        return DataLoader(
            _DictDataset(self._data),
            batch_size=int(self.config.training.batch_size),
            collate_fn=collate_fn,
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return self._make_loader()

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return self._make_loader()

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._make_loader()


class TestFullTrainingLoop:
    """Integration test: 2 epochs on synthetic data."""

    def test_two_epoch_training_completes(self) -> None:
        """A 2-epoch training loop on tiny data should complete without error."""
        cfg = _make_config()
        model = PathogenicityPredictor(cfg)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        module = PathogenicityLightningModule(cfg, model, class_weights=weights)

        datamodule = _SyntheticDataModule(cfg, n_samples=32)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            deterministic=True,
        )

        trainer.fit(module, datamodule=datamodule)

        assert trainer.current_epoch == 2
        assert trainer.callback_metrics.get("train_loss") is not None

    def test_two_epoch_with_test(self) -> None:
        """Training + testing should complete without error."""
        cfg = _make_config()
        model = PathogenicityPredictor(cfg)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        module = PathogenicityLightningModule(cfg, model, class_weights=weights)

        datamodule = _SyntheticDataModule(cfg, n_samples=32)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            deterministic=True,
        )

        trainer.fit(module, datamodule=datamodule)
        test_results = trainer.test(module, datamodule=datamodule)

        assert len(test_results) == 1
        assert "test_loss" in test_results[0]

    def test_validation_metrics_after_training(self) -> None:
        """Validation metrics should be present after training."""
        cfg = _make_config()
        model = PathogenicityPredictor(cfg)
        module = PathogenicityLightningModule(cfg, model)

        datamodule = _SyntheticDataModule(cfg, n_samples=32)

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            deterministic=True,
        )

        trainer.fit(module, datamodule=datamodule)

        metrics = trainer.callback_metrics
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "val_auroc" in metrics
