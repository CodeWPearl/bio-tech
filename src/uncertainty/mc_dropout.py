"""Monte Carlo Dropout uncertainty estimation.

Enables dropout at inference time and runs multiple stochastic forward passes
to estimate epistemic uncertainty via prediction variance and entropy.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class MCDropoutPredictor:
    """Estimate prediction uncertainty using MC Dropout.

    Args:
        model: A trained model with dropout layers.
        n_forward_passes: Number of stochastic forward passes.
    """

    def __init__(self, model: nn.Module, n_forward_passes: int = 50) -> None:
        self.model = model
        self.n_forward_passes = n_forward_passes

    @staticmethod
    def _enable_dropout(model: nn.Module) -> None:
        """Set all dropout layers to training mode while keeping others in eval."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run MC Dropout inference and compute uncertainty estimates.

        Args:
            batch: Dict with modality feature tensors and ``modality_mask``.

        Returns:
            Dict with ``mean_probs``, ``predicted_class``,
            ``epistemic_uncertainty``, ``predictive_entropy``,
            and ``all_predictions``.
        """
        self.model.eval()
        self._enable_dropout(self.model)

        all_probs: list[torch.Tensor] = []

        for _ in range(self.n_forward_passes):
            outputs = self.model(batch)
            probs = outputs["probabilities"]
            all_probs.append(probs.unsqueeze(0))

        all_predictions = torch.cat(all_probs, dim=0)

        mean_probs = all_predictions.mean(dim=0)
        predicted_class = mean_probs.argmax(dim=-1)

        epistemic_uncertainty = all_predictions.var(dim=0).mean(dim=-1)

        predictive_entropy = -(
            mean_probs * torch.log(mean_probs + 1e-10)
        ).sum(dim=-1)

        self.model.eval()

        return {
            "mean_probs": mean_probs,
            "predicted_class": predicted_class,
            "epistemic_uncertainty": epistemic_uncertainty,
            "predictive_entropy": predictive_entropy,
            "all_predictions": all_predictions,
        }
