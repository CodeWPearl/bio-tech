"""Early fusion: concatenate all modality embeddings and project.

Applies modality masking by zeroing out absent modality embeddings before
concatenation, then projects the concatenated vector to a fixed fusion
dimension via a linear layer with batch normalisation.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class EarlyFusion(nn.Module):
    """Concatenation-based early fusion.

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
        fusion_dim: Output dimensionality of the fused representation.
        dropout: Dropout rate after projection.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.modality_names = [m for m in MODALITY_NAMES if m in modality_dims]
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim

        concat_dim = sum(modality_dims[m] for m in self.modality_names)

        self.projection = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse modality embeddings via concatenation and projection.

        Args:
            embeddings: Mapping from modality name to ``(batch, embed_dim)``
                tensor.
            modality_mask: ``(batch, num_modalities)`` boolean tensor where
                ``True`` means the modality is present.

        Returns:
            ``(batch, fusion_dim)`` fused representation.
        """
        parts: list[torch.Tensor] = []
        for i, name in enumerate(self.modality_names):
            emb = embeddings[name]
            mask = modality_mask[:, i].unsqueeze(1).float()
            parts.append(emb * mask)

        concatenated = torch.cat(parts, dim=1)
        return self.projection(concatenated)
