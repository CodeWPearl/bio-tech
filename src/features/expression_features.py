"""RNA-seq expression feature extraction.

Transforms the wide expression matrix (samples x top-2000 genes, RSEM values
from cBioPortal) into a normalised numeric array:

1. **Log2(x + 1) transform** — compresses the heavy right tail typical of
   RNA-seq count data.
2. **Quantile normalisation** — forces all samples to share the same
   rank-value distribution, removing technical variation.
3. **Z-score standardisation per gene** — centres each gene at zero with unit
   variance (mean and std learned from training data).

Missing values are imputed with the per-gene median from the training set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _quantile_normalise(data: np.ndarray) -> np.ndarray:
    """Quantile-normalise columns of ``data`` (samples x genes).

    Each column is ranked; ranks are then replaced by the mean value at
    that rank across all columns, so every column ends up with the same
    distribution.

    Args:
        data: 2-D array of shape ``(n_samples, n_genes)``.

    Returns:
        Quantile-normalised array of the same shape.
    """
    n_samples, n_genes = data.shape
    if n_samples <= 1 or n_genes == 0:
        return data.copy()

    sorted_vals = np.sort(data, axis=0)
    rank_means = sorted_vals.mean(axis=1)

    ranks = np.argsort(np.argsort(data, axis=0), axis=0)
    normalised = rank_means[ranks]
    return normalised


class ExpressionFeatureExtractor:
    """Log-quantile-zscore pipeline for RNA-seq expression data.

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

    def fit(self, df: pd.DataFrame) -> ExpressionFeatureExtractor:
        """Learn normalisation statistics from training expression data.

        Args:
            df: Expression DataFrame. Columns prefixed ``expr_`` are treated
                as gene features; other columns are ignored.

        Returns:
            ``self`` for method chaining.
        """
        self._gene_columns = sorted(
            c for c in df.columns if c.startswith("expr_")
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

        logged = np.log2(imputed + 1.0)
        normed = _quantile_normalise(logged)

        self._means = normed.mean(axis=0)
        self._stds = normed.std(axis=0)
        self._stds[self._stds == 0] = 1.0

        self._fitted = True
        logger.info(
            "ExpressionFeatureExtractor fit on %d samples, %d genes",
            len(df), len(self._gene_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the learned pipeline to expression data.

        Args:
            df: Expression DataFrame (same schema as ``fit`` input).

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
                "%d expression genes in fit but absent from transform input",
                len(missing),
            )

        for j in range(n_genes):
            mask = np.isnan(raw[:, j])
            if mask.any():
                raw[mask, j] = self._medians[j]

        logged = np.log2(raw + 1.0)
        normed = _quantile_normalise(logged)
        z = (normed - self._means) / self._stds

        return z

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df).transform(df)
