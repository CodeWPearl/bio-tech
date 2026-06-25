"""Deep ensemble uncertainty estimation.

Loads multiple independently trained models and aggregates their predictions
to estimate epistemic uncertainty via disagreement and mutual information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.full_model import PathogenicityPredictor
from src.utils.config import Config

logger = logging.getLogger(__name__)


class DeepEnsemblePredictor:
    """Estimate prediction uncertainty using a deep ensemble.

    Args:
        model_paths: Paths to checkpoint files for each ensemble member.
        config: Project configuration for model construction.
    """

    def __init__(
        self,
        model_paths: list[str],
        config: Config,
    ) -> None:
        self.config = config
        self.models: list[nn.Module] = []

        for path_str in model_paths:
            path = Path(path_str)
            model = PathogenicityPredictor.from_config(config)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

        logger.info("Loaded %d ensemble members.", len(self.models))

    @torch.no_grad()
    def predict_with_uncertainty(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run ensemble inference and compute uncertainty estimates.

        Args:
            batch: Dict with modality feature tensors and ``modality_mask``.

        Returns:
            Dict with ``mean_probs``, ``predicted_class``,
            ``epistemic_uncertainty``, ``predictive_entropy``,
            and ``all_predictions``.
        """
        all_probs: list[torch.Tensor] = []

        for model in self.models:
            model.eval()
            outputs = model(batch)
            probs = outputs["probabilities"]
            all_probs.append(probs.unsqueeze(0))

        all_predictions = torch.cat(all_probs, dim=0)

        mean_probs = all_predictions.mean(dim=0)
        predicted_class = mean_probs.argmax(dim=-1)

        epistemic_uncertainty = all_predictions.var(dim=0).mean(dim=-1)

        predictive_entropy = -(
            mean_probs * torch.log(mean_probs + 1e-10)
        ).sum(dim=-1)

        per_model_entropy = -(
            all_predictions * torch.log(all_predictions + 1e-10)
        ).sum(dim=-1)
        expected_entropy = per_model_entropy.mean(dim=0)
        mutual_information = predictive_entropy - expected_entropy

        return {
            "mean_probs": mean_probs,
            "predicted_class": predicted_class,
            "epistemic_uncertainty": epistemic_uncertainty,
            "predictive_entropy": predictive_entropy,
            "mutual_information": mutual_information,
            "all_predictions": all_predictions,
        }
