"""Feature extraction pipeline orchestrating all modality extractors.

The :class:`FeaturePipeline` wraps the five per-modality extractors
(:mod:`~src.features.mutation_features`,
:mod:`~src.features.expression_features`,
:mod:`~src.features.methylation_features`,
:mod:`~src.features.cnv_features`,
:mod:`~src.features.clinical_features`) into a single ``fit`` / ``transform``
interface.  It automatically selects the correct column subsets from a merged
DataFrame, runs each extractor, and returns per-modality arrays plus an
optional combined matrix for baseline models.

Fitted pipelines can be persisted with :meth:`FeaturePipeline.save` and
restored with :meth:`FeaturePipeline.load` for reproducible inference.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.clinical_features import ClinicalFeatureExtractor
from src.features.cnv_features import CNVFeatureExtractor
from src.features.expression_features import ExpressionFeatureExtractor
from src.features.methylation_features import MethylationFeatureExtractor
from src.features.mutation_features import MutationFeatureExtractor

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrate all modality feature extractors.

    Args:
        cnv_mode: ``"ordinal"`` or ``"onehot"`` — forwarded to
            :class:`~src.features.cnv_features.CNVFeatureExtractor`.

    Attributes:
        mutation: The mutation feature extractor.
        expression: The expression feature extractor.
        methylation: The methylation feature extractor.
        cnv: The CNV feature extractor.
        clinical: The clinical feature extractor.
    """

    def __init__(self, *, cnv_mode: str = "ordinal") -> None:
        self.mutation = MutationFeatureExtractor()
        self.expression = ExpressionFeatureExtractor()
        self.methylation = MethylationFeatureExtractor()
        self.cnv = CNVFeatureExtractor(mode=cnv_mode)
        self.clinical = ClinicalFeatureExtractor()
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        """Fit all extractors on the training split.

        Args:
            df: Merged training DataFrame produced by
                :class:`~src.data.data_merger.DataMerger`.

        Returns:
            ``self`` for method chaining.
        """
        self.mutation.fit(df)

        self.expression.fit(df)
        self.methylation.fit(df)

        mutation_genes: set[str] | None = None
        if "gene_symbol" in df.columns:
            mutation_genes = set(
                df["gene_symbol"].dropna().str.upper().unique()
            )
        self.cnv.fit(df, mutation_genes=mutation_genes)

        self.clinical.fit(df)

        self._fitted = True
        logger.info("FeaturePipeline fit complete")
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Transform a DataFrame into per-modality feature arrays.

        Args:
            df: Merged DataFrame (training, validation, or test split).

        Returns:
            A mapping from modality name (``"mutation"``, ``"expression"``,
            ``"methylation"``, ``"cnv"``, ``"clinical"``) to a 2-D numpy
            array of shape ``(len(df), n_features_for_modality)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        return {
            "mutation": self.mutation.transform(df),
            "expression": self.expression.transform(df),
            "methylation": self.methylation.transform(df),
            "cnv": self.cnv.transform(df),
            "clinical": self.clinical.transform(df),
        }

    def fit_transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df).transform(df)

    def get_combined_matrix(
        self, features: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Horizontally stack all modality arrays into one matrix.

        Args:
            features: Output of :meth:`transform`.

        Returns:
            Array of shape ``(n_samples, total_features)``.
        """
        arrays = [
            features[key]
            for key in ("mutation", "expression", "methylation", "cnv", "clinical")
            if features[key].shape[1] > 0
        ]
        if not arrays:
            n = next(iter(features.values())).shape[0] if features else 0
            return np.zeros((n, 0), dtype=np.float64)
        return np.hstack(arrays)

    def get_feature_names(self) -> dict[str, list[str]]:
        """Return per-modality feature name lists.

        Returns:
            Mapping from modality name to its ordered list of feature names.
        """
        return {
            "mutation": self.mutation.feature_names,
            "expression": self.expression.feature_names,
            "methylation": self.methylation.feature_names,
            "cnv": self.cnv.feature_names,
            "clinical": self.clinical.feature_names,
        }

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline to disk.

        Args:
            path: Destination file (typically ``.pkl``).

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved FeaturePipeline to %s", dest)

    @classmethod
    def load(cls, path: str | Path) -> FeaturePipeline:
        """Load a previously saved pipeline.

        Args:
            path: Path to the ``.pkl`` file written by :meth:`save`.

        Returns:
            The restored :class:`FeaturePipeline` instance.
        """
        src = Path(path)
        with src.open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected FeaturePipeline, got {type(obj).__name__}"
            )
        logger.info("Loaded FeaturePipeline from %s", src)
        return obj
