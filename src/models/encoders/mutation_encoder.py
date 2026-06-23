"""Mutation feature encoders.

Two architectures:

* :class:`MutationEncoder` — two-layer MLP that maps the mutation feature
  vector into a compact embedding.
* :class:`MutationTransformerEncoder` — groups features into semantic tokens
  (variant type, AA properties, gene features, positional) and applies a small
  Transformer encoder, using the ``[CLS]`` token output as the embedding.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# Default feature-group boundaries matching MutationFeatureExtractor output:
# 9 variant-type one-hot, 5 AA properties, 3 gene features, 25 positional
_DEFAULT_GROUP_SIZES: tuple[int, ...] = (9, 5, 3, 25)


class MutationEncoder(BaseModel):
    """Two-layer MLP encoder for mutation feature vectors.

    Args:
        input_dim: Size of the input mutation feature vector.
        embed_dim: Output embedding dimensionality.
        dropout: Dropout rate for the second layer.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
        )

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from the mutation feature vector.

        Args:
            x: ``(batch, input_dim)`` tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)


class MutationTransformerEncoder(BaseModel):
    """Small Transformer encoder over mutation feature groups.

    Each feature group (variant type, AA properties, gene features, positional)
    is projected to a shared token dimension, prepended with a learnable
    ``[CLS]`` token, and fed through a 2-layer Transformer encoder.  The
    ``[CLS]`` output is the embedding.

    Args:
        input_dim: Total mutation feature vector size.
        embed_dim: Output embedding dimensionality.
        group_sizes: Sizes of each feature group.  Must sum to ``input_dim``.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        group_sizes: tuple[int, ...] | None = None,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        if group_sizes is None:
            group_sizes = _infer_group_sizes(input_dim)
        if sum(group_sizes) != input_dim:
            raise ValueError(
                f"group_sizes sum ({sum(group_sizes)}) != input_dim ({input_dim})"
            )

        self.group_sizes = group_sizes
        d_model = embed_dim

        self.group_projections = nn.ModuleList([
            nn.Linear(gs, d_model) for gs in group_sizes
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pos_embed = nn.Parameter(
            torch.randn(1, len(group_sizes) + 1, d_model) * 0.02
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

        self.norm = nn.LayerNorm(d_model)

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from the mutation feature vector.

        Args:
            x: ``(batch, input_dim)`` tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        batch_size = x.size(0)

        tokens: list[torch.Tensor] = []
        offset = 0
        for proj, gs in zip(self.group_projections, self.group_sizes):
            group = x[:, offset : offset + gs]
            tokens.append(proj(group).unsqueeze(1))
            offset += gs

        tokens_cat = torch.cat(tokens, dim=1)  # (B, n_groups, d_model)

        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens_cat], dim=1)  # (B, n_groups+1, d_model)

        seq = seq + self.pos_embed[:, : seq.size(1), :]

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0, :])

        return cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)


def _infer_group_sizes(input_dim: int) -> tuple[int, ...]:
    """Split ``input_dim`` into 4 roughly equal groups."""
    base = input_dim // 4
    remainder = input_dim % 4
    sizes = [base] * 4
    for i in range(remainder):
        sizes[i] += 1
    return tuple(sizes)
