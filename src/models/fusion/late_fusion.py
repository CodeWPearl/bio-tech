"""Late fusion: per-modality classification heads with learned weighting.

Each modality gets its own independent classification head.  The final
prediction is a weighted average of per-modality logits, where the weights
are learnable and softmax-normalised.  Missing modalities are excluded from
the weighted average.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class LateFusion(nn.Module):
    """Weighted late fusion with per-modality classification heads.

    Args:
        modality_dims: Mapping from modality name to its embedding dimension.
        num_classes: Number of output classes.
        fusion_dim: Intermediate projection dimension for each head.
        dropout: Dropout rate inside each classification head.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        num_classes: int = 4,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.modality_names = [m for m in MODALITY_NAMES if m in modality_dims]
        self.modality_dims = modality_dims
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        num_modalities = len(self.modality_names)

        self.heads = nn.ModuleDict()
        for name in self.modality_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(modality_dims[name], fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes),
            )

        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse modality embeddings via weighted late fusion.

        Args:
            embeddings: Mapping from modality name to ``(batch, embed_dim)``
                tensor.
            modality_mask: ``(batch, num_modalities)`` boolean tensor where
                ``True`` means the modality is present.

        Returns:
            Dict with keys:
            - ``fused``: ``(batch, num_classes)`` weighted-average logits.
            - ``per_modality``: dict mapping modality name to its
              ``(batch, num_classes)`` logits.
        """
        batch_size = next(iter(embeddings.values())).size(0)
        device = next(iter(embeddings.values())).device

        per_modality: dict[str, torch.Tensor] = {}
        all_logits = torch.zeros(
            batch_size, len(self.modality_names), self.num_classes, device=device,
        )

        for i, name in enumerate(self.modality_names):
            logits = self.heads[name](embeddings[name])
            per_modality[name] = logits
            all_logits[:, i, :] = logits

        weights = torch.softmax(self.modality_weights, dim=0)
        weights = weights.unsqueeze(0).expand(batch_size, -1)

        masked_weights = weights * modality_mask.float()
        weight_sum = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalised_weights = masked_weights / weight_sum

        fused = torch.einsum("bm,bmc->bc", normalised_weights, all_logits)

        return {"fused": fused, "per_modality": per_modality}
