"""Integrated Gradients explainability for the pathogenicity predictor.

Uses :mod:`captum.attr.IntegratedGradients` to compute per-feature attributions
and aggregate them by omics modality.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class _FlatWrapper(nn.Module):
    """Wraps the multi-omics model to accept a single flat tensor input.

    Captum requires a single tensor input for IG. This wrapper splits the flat
    tensor back into the per-modality batch dict the model expects.

    Args:
        model: The pathogenicity predictor model.
        modality_slices: Mapping from modality name to ``(start, end)`` column
            indices in the flat input.
        num_modalities: Number of modalities for the mask.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_slices: dict[str, tuple[int, int]],
        num_modalities: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_slices = modality_slices
        self.num_modalities = num_modalities

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Forward pass converting flat input to model batch dict.

        Args:
            flat_input: ``(batch, total_features)`` tensor.

        Returns:
            ``(batch, num_classes)`` logits.
        """
        batch: dict[str, torch.Tensor] = {}
        for name, (start, end) in self.modality_slices.items():
            batch[name] = flat_input[:, start:end]
        batch["modality_mask"] = torch.ones(
            flat_input.shape[0], self.num_modalities,
            dtype=torch.bool, device=flat_input.device,
        )
        result = self.model(batch)
        return result["logits"]


class IGExplainer:
    """Integrated Gradients explainability for multi-omics pathogenicity model.

    Args:
        model: Trained :class:`PathogenicityPredictor`.
        modality_dims: Mapping from modality name to its raw input dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_dims: dict[str, int],
    ) -> None:
        self.model = model
        self.modality_dims = modality_dims
        self._modality_slices = self._compute_slices()
        self._total_dim = sum(
            modality_dims[n] for n in MODALITY_NAMES if n in modality_dims
        )

    def _compute_slices(self) -> dict[str, tuple[int, int]]:
        """Compute column ranges for each modality in the flat feature vector."""
        slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in MODALITY_NAMES:
            if name not in self.modality_dims:
                continue
            dim = self.modality_dims[name]
            slices[name] = (offset, offset + dim)
            offset += dim
        return slices

    def _batch_to_flat(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten a model batch dict into a single 2-D tensor."""
        parts: list[torch.Tensor] = []
        for name in MODALITY_NAMES:
            if name in batch and name in self.modality_dims:
                parts.append(batch[name])
        return torch.cat(parts, dim=1)

    def compute_attributions(
        self,
        batch: dict[str, torch.Tensor],
        target_class: int,
        n_steps: int = 50,
    ) -> dict[str, Any]:
        """Compute IG attributions for a batch of samples.

        Args:
            batch: Model batch dict with modality tensors.
            target_class: Class index to attribute to.
            n_steps: Number of interpolation steps for IG.

        Returns:
            Dict with ``attributions`` (flat tensor), ``per_modality``
            (dict of per-modality attribution tensors), and
            ``modality_importance`` (dict of scalar importances).
        """
        from captum.attr import IntegratedGradients

        wrapper = _FlatWrapper(
            self.model, self._modality_slices, len(self._modality_slices),
        )
        flat_input = self._batch_to_flat(batch)
        flat_input.requires_grad_(True)
        baseline = torch.zeros_like(flat_input)

        ig = IntegratedGradients(wrapper)
        attributions = ig.attribute(
            flat_input, baselines=baseline, target=target_class, n_steps=n_steps,
        )

        per_modality: dict[str, torch.Tensor] = {}
        modality_importance: dict[str, float] = {}
        for mod, (start, end) in self._modality_slices.items():
            mod_attr = attributions[:, start:end]
            per_modality[mod] = mod_attr.detach()
            modality_importance[mod] = float(mod_attr.abs().mean().item())

        logger.info(
            "IG attributions computed for %d samples, target class %d",
            flat_input.shape[0], target_class,
        )
        return {
            "attributions": attributions.detach(),
            "per_modality": per_modality,
            "modality_importance": modality_importance,
        }

    def compute_modality_importance(
        self,
        test_loader: torch.utils.data.DataLoader,
        max_batches: int = 50,
    ) -> dict[str, float]:
        """Average IG attributions across the test set per modality.

        Args:
            test_loader: DataLoader yielding batch dicts with ``label`` key.
            max_batches: Maximum number of batches to process.

        Returns:
            Mapping from modality name to its average absolute attribution,
            sorted descending by importance.
        """
        accum: dict[str, list[float]] = {mod: [] for mod in self._modality_slices}

        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            labels = batch["label"]
            target = int(labels[0].item())
            result = self.compute_attributions(batch, target_class=target)
            for mod, importance in result["modality_importance"].items():
                accum[mod].append(importance)

        avg_importance: dict[str, float] = {}
        for mod, values in accum.items():
            avg_importance[mod] = float(np.mean(values)) if values else 0.0

        ranked = dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))
        logger.info("Modality importance (IG): %s", ranked)
        return ranked
