"""Clinical feature encoder.

A small MLP suited to the lower dimensionality of clinical metadata
(age, sex, cancer type, stage, etc.).

* :class:`ClinicalEncoder` — ``input → 64 → 32 → clinical_embed_dim``
  with BatchNorm, ReLU, and Dropout at each hidden layer.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ClinicalEncoder(BaseModel):
    """Two-hidden-layer MLP encoder for clinical features.

    Clinical features are lower-dimensional than omics data, so this
    encoder is intentionally compact.

    Args:
        input_dim: Size of the clinical feature vector.
        embed_dim: Output embedding dimensionality (default 32).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from the clinical feature vector.

        Args:
            x: ``(batch, input_dim)`` tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)
