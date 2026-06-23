"""Classification head for pathogenicity prediction.

Maps the fused multi-omics representation to class logits via a three-layer
MLP with batch normalisation and progressive dropout.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Three-layer MLP classification head.

    Args:
        fusion_dim: Dimensionality of the fused input representation.
        num_classes: Number of output classes.
    """

    def __init__(self, fusion_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from the fused representation.

        Args:
            x: ``(batch, fusion_dim)`` fused representation.

        Returns:
            ``(batch, num_classes)`` logits.
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities via softmax.

        Args:
            x: ``(batch, fusion_dim)`` fused representation.

        Returns:
            ``(batch, num_classes)`` probabilities summing to 1.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the most likely class.

        Args:
            x: ``(batch, fusion_dim)`` fused representation.

        Returns:
            ``(batch,)`` predicted class indices.
        """
        logits = self.forward(x)
        return logits.argmax(dim=-1)
