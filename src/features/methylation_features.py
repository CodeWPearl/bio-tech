"""Methylation feature extraction.

Transforms methylation beta-value matrices (samples x top-2000 genes) from
cBioPortal into M-values suitable for statistical analysis:

1. **Clip** beta values to ``[0.001, 0.999]`` to avoid infinite M-values.
2. **M-value transform** — ``M = log2(beta / (1 - beta))``, mapping the
   bounded (0, 1) beta scale onto an unbounded, approximately normal scale
   that is better suited for parametric models.
3. **Z-score standardisation per gene** — centres each gene at zero with unit
   variance (mean and std learned from training data).

Missing values are imputed with the per-gene median from the training set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MethylationFeatureExtractor:
    """Beta-to-M-value pipeline for methylation data.

    Attributes:
        feature_names: Gene column names retained (set after ``fit``).
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._fitted: bool = False
        self._gene_columns: list[str] = []
        self._medians: np.ndarray = np.array([])
        self._means: np.ndarray = np.array([])
        self._stds: np.ndarray = np.array([])

    def fit(self, df: pd.DataFrame) -> MethylationFeatureExtractor:
        """Learn normalisation statistics from training methylation data.

        Args:
            df: Methylation DataFrame. Columns prefixed ``meth_`` are treated
                as gene features; other columns are ignored.

        Returns:
            ``self`` for method chaining.
        """
        self._gene_columns = sorted(
            c for c in df.columns if c.startswith("meth_")
        )
        self.feature_names = list(self._gene_columns)

        if not self._gene_columns:
            self._medians = np.array([])
            self._means = np.array([])
            self._stds = np.array([])
            self._fitted = True
            return self

        raw = df[self._gene_columns].to_numpy(dtype=np.float64, copy=True)
        self._medians = np.nanmedian(raw, axis=0)

        imputed = raw.copy()
        for j in range(imputed.shape[1]):
            mask = np.isnan(imputed[:, j])
            if mask.any():
                imputed[mask, j] = self._medians[j]

        clipped = np.clip(imputed, 0.001, 0.999)
        m_vals = np.log2(clipped / (1.0 - clipped))

        self._means = m_vals.mean(axis=0)
        self._stds = m_vals.std(axis=0)
        self._stds[self._stds == 0] = 1.0

        self._fitted = True
        logger.info(
            "MethylationFeatureExtractor fit on %d samples, %d genes",
            len(df), len(self._gene_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the learned pipeline to methylation data.

        Args:
            df: Methylation DataFrame (same schema as ``fit`` input).

        Returns:
            Array of shape ``(len(df), n_genes)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        n = len(df)
        n_genes = len(self._gene_columns)
        if n_genes == 0:
            return np.zeros((n, 0), dtype=np.float64)

        present = [c for c in self._gene_columns if c in df.columns]
        missing = [c for c in self._gene_columns if c not in df.columns]

        raw = np.full((n, n_genes), np.nan, dtype=np.float64)
        if present:
            col_idx = [self._gene_columns.index(c) for c in present]
            raw[:, col_idx] = df[present].to_numpy(dtype=np.float64)

        if missing:
            logger.warning(
                "%d methylation genes in fit but absent from transform",
                len(missing),
            )

        for j in range(n_genes):
            mask = np.isnan(raw[:, j])
            if mask.any():
                raw[mask, j] = self._medians[j]

        clipped = np.clip(raw, 0.001, 0.999)
        m_vals = np.log2(clipped / (1.0 - clipped))
        z = (m_vals - self._means) / self._stds

        return z

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df).transform(df)
