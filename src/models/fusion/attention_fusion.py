"""Attention fusion: multi-head self-attention over modality embeddings.

Projects all modality embeddings to a shared dimension, stacks them as a
sequence, and applies multi-head self-attention.  Modality masking is done
via attention masking (``-inf`` for absent modalities).  Attention weights
are saved for interpretability.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class AttentionFusion(nn.Module):
    """Self-attention fusion over modality embeddings.

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
        fusion_dim: Shared dimension for attention and output.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_names = [m for m in MODALITY_NAMES if m in modality_dims]
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        self.n_heads = n_heads

        self.projections = nn.ModuleDict()
        for name in self.modality_names:
            self.projections[name] = nn.Linear(modality_dims[name], fusion_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights: torch.Tensor | None = None

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse modality embeddings via multi-head self-attention.

        Args:
            embeddings: Mapping from modality name to ``(batch, embed_dim)``
                tensor.
            modality_mask: ``(batch, num_modalities)`` boolean tensor where
                ``True`` means the modality is present.

        Returns:
            ``(batch, fusion_dim)`` fused representation.
        """
        projected: list[torch.Tensor] = []
        for name in self.modality_names:
            projected.append(self.projections[name](embeddings[name]))

        sequence = torch.stack(projected, dim=1)

        key_padding_mask = ~modality_mask.bool()

        attn_out, attn_weights = self.attention(
            sequence, sequence, sequence,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        self.attention_weights = attn_weights.detach()

        residual = sequence + self.dropout(attn_out)
        normed = self.layer_norm(residual)

        mask_float = modality_mask.float().unsqueeze(2)
        weighted = normed * mask_float
        weight_sum = mask_float.sum(dim=1).clamp(min=1e-8)
        pooled = weighted.sum(dim=1) / weight_sum

        return self.output_proj(pooled)
