"""PyTorch Lightning module for pathogenicity prediction training.

Wraps :class:`~src.models.full_model.PathogenicityPredictor` with loss
computation, metric tracking, and optimiser configuration.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)

from src.models.full_model import PathogenicityPredictor
from src.training.losses import FocalLoss, WeightedCrossEntropy
from src.training.scheduler import get_scheduler
from src.utils.config import Config

logger = logging.getLogger(__name__)

_NUM_MODALITIES: int = 5


class PathogenicityLightningModule(pl.LightningModule):
    """Lightning wrapper for end-to-end pathogenicity training.

    Args:
        config: Project configuration object.
        model: The assembled predictor model.
        class_weights: Per-class weight tensor for the loss function.
    """

    def __init__(
        self,
        config: Config,
        model: PathogenicityPredictor,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model

        num_classes: int = int(config.model.num_classes)
        self._num_classes = num_classes

        self.loss_fn = self._build_loss(config, class_weights)

        self._build_metrics(num_classes)

        self.save_hyperparameters(config.to_dict(), ignore=["model"])

    def _build_loss(
        self, config: Config, class_weights: torch.Tensor | None,
    ) -> torch.nn.Module:
        """Instantiate the loss function from config.

        Args:
            config: Project configuration.
            class_weights: Per-class weights.

        Returns:
            The loss module.
        """
        loss_type: str = getattr(config.training, "loss_type", "focal")

        if loss_type == "focal":
            gamma = float(config.training.focal_loss_gamma)
            return FocalLoss(alpha=class_weights, gamma=gamma)
        if loss_type == "weighted_ce":
            return WeightedCrossEntropy(class_weights=class_weights)
        if loss_type == "ce":
            return torch.nn.CrossEntropyLoss()

        raise ValueError(f"Unknown loss type: {loss_type}")

    def _build_metrics(self, num_classes: int) -> None:
        """Create torchmetrics instances for train, val, and test.

        Args:
            num_classes: Number of target classes.
        """
        task = "multiclass"

        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)

        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task=task, num_classes=num_classes, average="macro")
        self.val_auroc = AUROC(task=task, num_classes=num_classes)
        self.val_precision = Precision(task=task, num_classes=num_classes, average="macro")
        self.val_recall = Recall(task=task, num_classes=num_classes, average="macro")
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)

        self.test_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.test_f1 = F1Score(task=task, num_classes=num_classes, average="macro")
        self.test_auroc = AUROC(task=task, num_classes=num_classes)
        self.test_precision = Precision(task=task, num_classes=num_classes, average="macro")
        self.test_recall = Recall(task=task, num_classes=num_classes, average="macro")
        self.test_mcc = MatthewsCorrCoef(task=task, num_classes=num_classes)

    @staticmethod
    def _expand_modality_mask(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expand the 3-element modality mask to 5 elements.

        The dataset provides a mask for ``[expression, methylation, cnv]``.
        The full model expects ``[mutation, expression, methylation, cnv,
        clinical]`` — mutation and clinical are always present.

        Args:
            batch: Batch dict with a ``modality_mask`` of shape ``(B, 3)``.

        Returns:
            The same batch dict with ``modality_mask`` expanded to ``(B, 5)``.
        """
        mask_3 = batch["modality_mask"]
        batch_size = mask_3.size(0)
        ones = torch.ones(batch_size, 1, dtype=mask_3.dtype, device=mask_3.device)
        batch["modality_mask"] = torch.cat([ones, mask_3, ones], dim=1)
        return batch

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Run the forward pass.

        Args:
            batch: Dict of modality tensors and modality mask.

        Returns:
            Model output dict with logits, probabilities, etc.
        """
        return self.model(batch)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Execute one training step.

        Args:
            batch: Batch from the training dataloader.
            batch_idx: Index of the current batch.

        Returns:
            Scalar training loss.
        """
        labels = batch["label"]
        batch = self._expand_modality_mask(batch)
        outputs = self.model(batch)
        logits = outputs["logits"]

        loss = self.loss_fn(logits, labels)

        self.train_accuracy(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy", self.train_accuracy,
            on_step=True, on_epoch=True, prog_bar=True,
        )

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int,
    ) -> None:
        """Execute one validation step.

        Args:
            batch: Batch from the validation dataloader.
            batch_idx: Index of the current batch.
        """
        labels = batch["label"]
        batch = self._expand_modality_mask(batch)
        outputs = self.model(batch)
        logits = outputs["logits"]
        probs = outputs["probabilities"]

        loss = self.loss_fn(logits, labels)

        self.val_accuracy(logits, labels)
        self.val_f1(logits, labels)
        self.val_auroc(probs, labels)
        self.val_precision(logits, labels)
        self.val_recall(logits, labels)
        self.val_mcc(logits, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level validation metrics."""
        self.log("val_accuracy", self.val_accuracy, prog_bar=True)
        self.log("val_f1", self.val_f1)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall)
        self.log("val_mcc", self.val_mcc)

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int,
    ) -> None:
        """Execute one test step.

        Args:
            batch: Batch from the test dataloader.
            batch_idx: Index of the current batch.
        """
        labels = batch["label"]
        batch = self._expand_modality_mask(batch)
        outputs = self.model(batch)
        logits = outputs["logits"]
        probs = outputs["probabilities"]

        loss = self.loss_fn(logits, labels)

        self.test_accuracy(logits, labels)
        self.test_f1(logits, labels)
        self.test_auroc(probs, labels)
        self.test_precision(logits, labels)
        self.test_recall(logits, labels)
        self.test_mcc(logits, labels)

        self.log("test_loss", loss, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log epoch-level test metrics."""
        self.log("test_accuracy", self.test_accuracy)
        self.log("test_f1", self.test_f1)
        self.log("test_auroc", self.test_auroc)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)
        self.log("test_mcc", self.test_mcc)

    def configure_optimizers(self) -> dict[str, Any]:
        """Set up AdamW optimiser and LR scheduler.

        Returns:
            Dict with ``"optimizer"`` and ``"lr_scheduler"`` entries.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.training.learning_rate),
            weight_decay=float(self.config.training.weight_decay),
        )

        scheduler_dict = get_scheduler(optimizer, self.config)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
