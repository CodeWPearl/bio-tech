"""Transformer fusion: full Transformer encoder over modality tokens.

Adds learned modality-type embeddings to each modality embedding, prepends
a learnable ``[CLS]`` token, and runs a 2-layer Transformer encoder with
the modality mask as the padding mask.  The ``[CLS]`` output is the fused
representation.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class TransformerFusion(nn.Module):
    """Transformer-based fusion with learned modality embeddings and [CLS].

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
        fusion_dim: Shared dimension for the Transformer and output.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_names = [m for m in MODALITY_NAMES if m in modality_dims]
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        num_modalities = len(self.modality_names)

        self.projections = nn.ModuleDict()
        for name in self.modality_names:
            self.projections[name] = nn.Linear(modality_dims[name], fusion_dim)

        self.modality_embeddings = nn.Parameter(
            torch.randn(1, num_modalities, fusion_dim) * 0.02,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=n_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse modality embeddings via a Transformer encoder.

        Args:
            embeddings: Mapping from modality name to ``(batch, embed_dim)``
                tensor.
            modality_mask: ``(batch, num_modalities)`` boolean tensor where
                ``True`` means the modality is present.

        Returns:
            ``(batch, fusion_dim)`` fused representation from [CLS] output.
        """
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device

        tokens: list[torch.Tensor] = []
        for i, name in enumerate(self.modality_names):
            proj = self.projections[name](embeddings[name])
            proj = proj + self.modality_embeddings[:, i, :]
            tokens.append(proj.unsqueeze(1))

        modality_seq = torch.cat(tokens, dim=1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, modality_seq], dim=1)

        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, modality_mask.bool()], dim=1)
        src_key_padding_mask = ~full_mask

        out = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.norm(out[:, 0, :])

        return cls_out
