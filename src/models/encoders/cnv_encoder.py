"""Copy-number variation (CNV) encoders.

Two architectures:

* :class:`CNVFCEncoder` — simple MLP that maps the CNV feature vector into a
  compact embedding.
* :class:`CNVAttentionEncoder` — treats each gene's CNV value as a token,
  applies self-attention to model gene-gene CNV interactions, and mean-pools
  the result.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class CNVFCEncoder(BaseModel):
    """Three-layer MLP encoder for CNV features.

    Args:
        input_dim: Size of the CNV feature vector.
        embed_dim: Output embedding dimensionality (default 64).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from the CNV feature vector.

        Args:
            x: ``(batch, input_dim)`` tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)


class CNVAttentionEncoder(BaseModel):
    """Self-attention encoder for CNV data.

    Each gene's CNV value is treated as a 1-D token and projected to the model
    dimension.  Multi-head self-attention captures gene-gene CNV interactions.
    The output is mean-pooled into a fixed-size embedding.

    Args:
        input_dim: Number of CNV features (genes).
        embed_dim: Output embedding dimensionality (default 64).
        d_model: Internal attention dimension (default 64).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of Transformer encoder layers (default 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self._input_dim = input_dim

        self.token_projection = nn.Linear(1, d_model)

        self.pos_embed = nn.Parameter(
            torch.randn(1, input_dim, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
        )

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from per-gene CNV values.

        Args:
            x: ``(batch, input_dim)`` tensor of CNV values.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        tokens = self.token_projection(x.unsqueeze(-1))  # (B, n_genes, d_model)

        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        out = self.transformer(tokens)  # (B, n_genes, d_model)

        pooled = out.mean(dim=1)  # (B, d_model)

        return self.output_projection(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)
