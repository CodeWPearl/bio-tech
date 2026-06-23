"""Loss functions for pathogenicity classification.

Provides :class:`FocalLoss` for class-imbalance-aware training and
:class:`WeightedCrossEntropy` as a simpler weighted alternative.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weight tensor of shape ``(num_classes,)``.
            If ``None``, all classes are weighted equally.
        gamma: Focusing parameter that down-weights easy examples.
            ``gamma=0`` recovers standard cross-entropy.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        alpha: torch.Tensor | Sequence[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(batch, num_classes)`` raw scores (before softmax).
            targets: ``(batch,)`` integer class labels.

        Returns:
            Scalar loss (or per-sample if ``reduction="none"``).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        p_t = (probs * targets_one_hot).sum(dim=-1)
        log_p_t = (log_probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """Cross-entropy loss with automatic class weight computation.

    Wraps :class:`torch.nn.CrossEntropyLoss` with inverse-frequency weights
    computed from the label distribution.

    Args:
        class_weights: Pre-computed per-class weights as a tensor or sequence.
            If ``None``, weights are computed from ``label_counts`` at
            construction time or left uniform.
        label_counts: Per-class sample counts used to derive inverse-frequency
            weights when ``class_weights`` is not provided.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | Sequence[float] | None = None,
        label_counts: Sequence[int] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            weight = class_weights
        elif label_counts is not None:
            counts = torch.tensor(label_counts, dtype=torch.float32)
            weight = 1.0 / counts
            weight = weight / weight.sum() * len(counts)
        else:
            weight = None

        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

        if weight is not None:
            logger.info("WeightedCrossEntropy weights: %s", weight.tolist())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: ``(batch, num_classes)`` raw scores.
            targets: ``(batch,)`` integer class labels.

        Returns:
            Scalar loss (or per-sample if ``reduction="none"``).
        """
        return self.ce(logits, targets)
