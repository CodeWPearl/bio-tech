"""PyTorch Dataset for multi-omics pathogenicity prediction.

Wraps per-modality numpy arrays into a :class:`torch.utils.data.Dataset` that
returns dictionaries of tensors.  Missing modalities are represented as
zero-filled tensors with a corresponding ``False`` entry in the
``modality_mask`` boolean vector, so downstream models can gate on availability
without special-casing ``None`` inputs.

A custom :func:`collate_fn` is provided to stack these heterogeneous dicts
into batched tensors suitable for :class:`torch.utils.data.DataLoader`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Indices into the per-sample modality mask (length-3 boolean vector).
_MASK_EXPRESSION: int = 0
_MASK_METHYLATION: int = 1
_MASK_CNV: int = 2


class MultiOmicsDataset(Dataset):
    """Dataset holding per-modality feature arrays for pathogenicity prediction.

    Args:
        mutation_features: ``(n_samples, n_mutation_feats)`` array.
        expression_features: ``(n_samples, n_expr_feats)`` array or ``None``.
        methylation_features: ``(n_samples, n_meth_feats)`` array or ``None``.
        cnv_features: ``(n_samples, n_cnv_feats)`` array or ``None``.
        clinical_features: ``(n_samples, n_clinical_feats)`` array.
        labels: ``(n_samples,)`` integer label array.
        modality_mask: ``(n_samples, 3)`` boolean array indicating which of
            ``[expression, methylation, cnv]`` are present per sample.
    """

    def __init__(
        self,
        mutation_features: np.ndarray,
        expression_features: np.ndarray | None,
        methylation_features: np.ndarray | None,
        cnv_features: np.ndarray | None,
        clinical_features: np.ndarray,
        labels: np.ndarray,
        modality_mask: np.ndarray,
    ) -> None:
        self.n_samples: int = mutation_features.shape[0]

        self.mutation = torch.as_tensor(mutation_features, dtype=torch.float32)
        self.clinical = torch.as_tensor(clinical_features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.modality_mask = torch.as_tensor(modality_mask, dtype=torch.bool)

        n = self.n_samples
        if expression_features is not None:
            self.expression = torch.as_tensor(
                expression_features, dtype=torch.float32
            )
        else:
            self.expression = torch.zeros(n, 0, dtype=torch.float32)

        if methylation_features is not None:
            self.methylation = torch.as_tensor(
                methylation_features, dtype=torch.float32
            )
        else:
            self.methylation = torch.zeros(n, 0, dtype=torch.float32)

        if cnv_features is not None:
            self.cnv = torch.as_tensor(cnv_features, dtype=torch.float32)
        else:
            self.cnv = torch.zeros(n, 0, dtype=torch.float32)

        logger.info(
            "MultiOmicsDataset created: %d samples, mutation=%s, "
            "expression=%s, methylation=%s, cnv=%s, clinical=%s",
            n,
            list(self.mutation.shape),
            list(self.expression.shape),
            list(self.methylation.shape),
            list(self.cnv.shape),
            list(self.clinical.shape),
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors.

        For modalities flagged as missing in ``modality_mask``, the returned
        tensor is zero-filled (the mask tells downstream layers to ignore it).

        Args:
            idx: Sample index.

        Returns:
            A dict with keys ``mutation``, ``expression``, ``methylation``,
            ``cnv``, ``clinical``, ``modality_mask``, and ``label``.
        """
        mask = self.modality_mask[idx]

        expression = self.expression[idx] if self.expression.shape[1] > 0 else self.expression[idx]
        if not mask[_MASK_EXPRESSION] and self.expression.shape[1] > 0:
            expression = torch.zeros_like(expression)

        methylation = self.methylation[idx] if self.methylation.shape[1] > 0 else self.methylation[idx]
        if not mask[_MASK_METHYLATION] and self.methylation.shape[1] > 0:
            methylation = torch.zeros_like(methylation)

        cnv = self.cnv[idx] if self.cnv.shape[1] > 0 else self.cnv[idx]
        if not mask[_MASK_CNV] and self.cnv.shape[1] > 0:
            cnv = torch.zeros_like(cnv)

        return {
            "mutation": self.mutation[idx],
            "expression": expression,
            "methylation": methylation,
            "cnv": cnv,
            "clinical": self.clinical[idx],
            "modality_mask": mask,
            "label": self.labels[idx],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batched dict of tensors.

    Args:
        batch: List of dicts as returned by
            :meth:`MultiOmicsDataset.__getitem__`.

    Returns:
        A dict with the same keys, each value being a batched tensor with an
        added leading batch dimension.
    """
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0]
    }
