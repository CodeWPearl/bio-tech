"""Cross-attention fusion: pairwise cross-attention between modalities.

Each modality attends to every other modality via cross-attention (Q from
modality *i*, K/V from all modalities).  This is the most expressive fusion
strategy and the paper's primary contribution.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class CrossAttentionFusion(nn.Module):
    """Pairwise cross-attention fusion.

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
        fusion_dim: Shared dimension for cross-attention and output.
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

        self.projections = nn.ModuleDict()
        for name in self.modality_names:
            self.projections[name] = nn.Linear(modality_dims[name], fusion_dim)

        self.cross_attentions = nn.ModuleDict()
        for name in self.modality_names:
            self.cross_attentions[name] = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.layer_norms = nn.ModuleDict()
        for name in self.modality_names:
            self.layer_norms[name] = nn.LayerNorm(fusion_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse modality embeddings via pairwise cross-attention.

        Args:
            embeddings: Mapping from modality name to ``(batch, embed_dim)``
                tensor.
            modality_mask: ``(batch, num_modalities)`` boolean tensor where
                ``True`` means the modality is present.

        Returns:
            ``(batch, fusion_dim)`` fused representation.
        """
        projected: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            projected[name] = self.projections[name](embeddings[name])

        kv_sequence = torch.stack(
            [projected[n] for n in self.modality_names], dim=1,
        )

        key_padding_mask = ~modality_mask.bool()

        cross_attended: list[torch.Tensor] = []
        for i, name in enumerate(self.modality_names):
            query = projected[name].unsqueeze(1)

            attn_out, _ = self.cross_attentions[name](
                query, kv_sequence, kv_sequence,
                key_padding_mask=key_padding_mask,
            )

            residual = query + attn_out
            normed = self.layer_norms[name](residual)
            cross_attended.append(normed.squeeze(1))

        stacked = torch.stack(cross_attended, dim=1)

        mask_float = modality_mask.float().unsqueeze(2)
        masked = stacked * mask_float
        weight_sum = mask_float.sum(dim=1).clamp(min=1e-8)
        pooled = masked.sum(dim=1) / weight_sum

        return self.output_proj(pooled)
