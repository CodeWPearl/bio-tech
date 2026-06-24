"""LIME-based explainability for the pathogenicity predictor.

Uses :mod:`lime.lime_tabular` to produce local, interpretable explanations
for individual predictions.
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


class LIMEExplainer:
    """LIME-based local explanations for the multi-omics model.

    Args:
        model: Trained :class:`PathogenicityPredictor`.
        modality_dims: Mapping from modality name to its raw input dimension.
        class_names: Optional list of class names for display.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_dims: dict[str, int],
        class_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.modality_dims = modality_dims
        self.class_names = class_names or [
            "Pathogenic", "Likely Pathogenic", "Benign", "Likely Benign",
        ]
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

    def _batch_to_flat(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten a model batch dict into a 2-D numpy array."""
        parts: list[np.ndarray] = []
        for name in MODALITY_NAMES:
            if name in batch and name in self.modality_dims:
                t = batch[name]
                parts.append(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t)
        return np.concatenate(parts, axis=1)

    def _predict_fn(self, flat_input: np.ndarray) -> np.ndarray:
        """Convert flat numpy input to model predictions.

        Args:
            flat_input: ``(n_samples, total_features)`` array.

        Returns:
            ``(n_samples, num_classes)`` probability array.
        """
        device = next(self.model.parameters()).device
        batch: dict[str, torch.Tensor] = {}
        for name, (start, end) in self._modality_slices.items():
            batch[name] = torch.tensor(
                flat_input[:, start:end], dtype=torch.float32, device=device,
            )
        mask = torch.ones(
            flat_input.shape[0], len(self._modality_slices),
            dtype=torch.bool, device=device,
        )
        batch["modality_mask"] = mask

        self.model.eval()
        with torch.no_grad():
            result = self.model(batch)
        return result["probabilities"].cpu().numpy()

    def _build_feature_names(
        self, feature_names: dict[str, list[str]] | None,
    ) -> list[str]:
        """Build a flat list of feature names from per-modality names."""
        names: list[str] = []
        for mod in MODALITY_NAMES:
            if mod not in self.modality_dims:
                continue
            dim = self.modality_dims[mod]
            if feature_names and mod in feature_names:
                names.extend(feature_names[mod][:dim])
                if len(feature_names[mod]) < dim:
                    for i in range(len(feature_names[mod]), dim):
                        names.append(f"{mod}_{i}")
            else:
                names.extend(f"{mod}_{i}" for i in range(dim))
        return names

    def explain_instance(
        self,
        sample: dict[str, torch.Tensor],
        feature_names: dict[str, list[str]] | None = None,
        training_data: np.ndarray | None = None,
        num_features: int = 20,
        num_samples: int = 500,
    ) -> dict[str, Any]:
        """Explain a single prediction using LIME.

        Args:
            sample: Single-sample batch dict (batch dim = 1).
            feature_names: Optional per-modality feature name lists.
            training_data: Optional training data for the explainer. If
                ``None``, uses the sample itself as reference.
            num_features: Number of top features to include.
            num_samples: Number of perturbation samples for LIME.

        Returns:
            Dict with ``explanation`` (the LIME Explanation object),
            ``top_features`` (list of (feature, weight) tuples per class),
            and ``predicted_class``.
        """
        from lime.lime_tabular import LimeTabularExplainer

        flat = self._batch_to_flat(sample)
        names = self._build_feature_names(feature_names)

        if training_data is None:
            training_data = flat

        explainer = LimeTabularExplainer(
            training_data,
            feature_names=names,
            class_names=self.class_names,
            mode="classification",
        )

        explanation = explainer.explain_instance(
            flat[0],
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=len(self.class_names),
        )

        probs = self._predict_fn(flat[0:1])
        predicted = int(np.argmax(probs[0]))

        top_features: dict[str, list[tuple[str, float]]] = {}
        for label_idx in explanation.available_labels():
            class_name = self.class_names[label_idx] if label_idx < len(self.class_names) else str(label_idx)
            feature_weights = explanation.as_list(label=label_idx)
            top_features[class_name] = feature_weights

        logger.info(
            "LIME explanation: predicted class %d (%s), %d features",
            predicted, self.class_names[predicted] if predicted < len(self.class_names) else str(predicted),
            num_features,
        )
        return {
            "explanation": explanation,
            "top_features": top_features,
            "predicted_class": predicted,
        }
