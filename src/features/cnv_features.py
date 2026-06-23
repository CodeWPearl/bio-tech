"""Copy-number variation (CNV) feature extraction.

Transforms the GISTIC2 discrete copy-number matrix (samples x genes, values
in {-2, -1, 0, 1, 2}) into model-ready features.  Two modes are supported:

* **Ordinal** (default) — keeps the integer values directly.  This preserves
  the natural ordering (homozygous deletion → deep amplification) and is
  compact.
* **One-hot** — expands each gene into 5 binary columns (one per GISTIC
  level).  This is appropriate when the model should not assume linearity
  between levels.

In both modes, only genes that overlap with the mutation gene set (learned
from training data in ``fit``) are retained, and missing values are imputed
with 0 (diploid / no change).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GISTIC_LEVELS: tuple[int, ...] = (-2, -1, 0, 1, 2)


class CNVFeatureExtractor:
    """GISTIC copy-number feature extractor.

    Args:
        mode: ``"ordinal"`` keeps raw -2..2 values; ``"onehot"`` expands each
            gene to 5 binary columns.

    Attributes:
        feature_names: Output column names (set after ``fit``).
    """

    def __init__(self, *, mode: str = "ordinal") -> None:
        if mode not in ("ordinal", "onehot"):
            raise ValueError(f"mode must be 'ordinal' or 'onehot', got {mode!r}")
        self.mode = mode
        self.feature_names: list[str] = []
        self._fitted: bool = False
        self._gene_columns: list[str] = []

    def fit(
        self, df: pd.DataFrame, mutation_genes: set[str] | None = None
    ) -> CNVFeatureExtractor:
        """Learn which CNV genes to keep (optionally intersecting with mutations).

        Args:
            df: CNV DataFrame. Columns prefixed ``cnv_`` are gene features.
            mutation_genes: If provided, only CNV genes also present in this
                set are retained.  Gene names should be upper-cased symbols.

        Returns:
            ``self`` for method chaining.
        """
        all_cnv = sorted(c for c in df.columns if c.startswith("cnv_"))

        if mutation_genes is not None:
            norm = {g.upper() for g in mutation_genes}
            self._gene_columns = [
                c for c in all_cnv if c[4:].upper() in norm
            ]
        else:
            self._gene_columns = all_cnv

        if self.mode == "onehot":
            names: list[str] = []
            for col in self._gene_columns:
                for level in GISTIC_LEVELS:
                    names.append(f"{col}_lv{level}")
            self.feature_names = names
        else:
            self.feature_names = list(self._gene_columns)

        self._fitted = True
        logger.info(
            "CNVFeatureExtractor fit: %d genes, mode=%s, %d output features",
            len(self._gene_columns), self.mode, len(self.feature_names),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convert CNV columns to a numeric feature matrix.

        Args:
            df: CNV DataFrame (same schema as ``fit`` input).

        Returns:
            Array of shape ``(len(df), n_cnv_features)``.

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
        raw = np.zeros((n, n_genes), dtype=np.float64)
        if present:
            col_idx = [self._gene_columns.index(c) for c in present]
            vals = df[present].to_numpy(dtype=np.float64, copy=True)
            nan_mask = np.isnan(vals)
            vals[nan_mask] = 0.0
            raw[:, col_idx] = vals

        if self.mode == "ordinal":
            return raw

        out = np.zeros((n, len(self.feature_names)), dtype=np.float64)
        for g_idx in range(n_genes):
            base = g_idx * len(GISTIC_LEVELS)
            for lv_idx, level in enumerate(GISTIC_LEVELS):
                out[:, base + lv_idx] = (raw[:, g_idx] == level).astype(
                    np.float64
                )
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        mutation_genes: set[str] | None = None,
    ) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df, mutation_genes).transform(df)
