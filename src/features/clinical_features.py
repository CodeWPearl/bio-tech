"""Clinical feature extraction.

Transforms the standardised clinical columns (age, sex, cancer type, stage)
into a fixed-width numeric array:

* **Age** — min-max normalised to [0, 1] using training-set extremes.
* **Sex** — binary (Female → 0, Male → 1).
* **Cancer type** — one-hot encoded (categories learned from training data).
* **Stage** — ordinal (Stage I → 1 … Stage IV → 4); sub-stages like IIIA
  are mapped to their major stage.

Missing values are imputed with training-set statistics: median for age,
mode for sex and stage.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STAGE_RE = re.compile(
    r"(?:stage\s*)?([IV]{1,3})",
    re.IGNORECASE,
)

_ROMAN_TO_INT: dict[str, int] = {
    "I": 1, "II": 2, "III": 3, "IV": 4,
}


def _parse_stage(value: object) -> float:
    """Convert a clinical stage string to an ordinal integer.

    Returns ``NaN`` if the value is missing or unparseable.
    """
    if not isinstance(value, str) or not value.strip():
        return float("nan")
    m = _STAGE_RE.search(value)
    if m:
        roman = m.group(1).upper()
        return float(_ROMAN_TO_INT.get(roman, float("nan")))
    try:
        val = int(value)
        if 1 <= val <= 4:
            return float(val)
    except (ValueError, TypeError):
        pass
    return float("nan")


def _mode(series: pd.Series) -> object:
    """Return the most common non-null value, or ``None``."""
    clean = series.dropna()
    if clean.empty:
        return None
    return Counter(clean).most_common(1)[0][0]


class ClinicalFeatureExtractor:
    """Encode clinical attributes into a numeric feature vector.

    Attributes:
        feature_names: Output column names (set after ``fit``).
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._fitted: bool = False
        self._age_min: float = 0.0
        self._age_max: float = 100.0
        self._age_median: float = 60.0
        self._sex_mode: str = "Female"
        self._stage_mode: float = 2.0
        self._cancer_types: list[str] = []

    def fit(self, df: pd.DataFrame) -> ClinicalFeatureExtractor:
        """Learn encoding parameters from training clinical data.

        Args:
            df: DataFrame containing clinical columns (``age``, ``sex``,
                ``cancer_type``, ``stage``).  Missing columns are tolerated.

        Returns:
            ``self`` for method chaining.
        """
        if "age" in df.columns:
            age = pd.to_numeric(df["age"], errors="coerce")
            self._age_min = float(age.min()) if age.notna().any() else 0.0
            self._age_max = float(age.max()) if age.notna().any() else 100.0
            self._age_median = float(age.median()) if age.notna().any() else 60.0
            if self._age_max == self._age_min:
                self._age_max = self._age_min + 1.0

        if "sex" in df.columns:
            m = _mode(df["sex"])
            self._sex_mode = str(m) if m is not None else "Female"

        if "stage" in df.columns:
            parsed = df["stage"].map(_parse_stage)
            m = parsed.dropna()
            self._stage_mode = float(m.median()) if not m.empty else 2.0

        if "cancer_type" in df.columns:
            self._cancer_types = sorted(
                df["cancer_type"].dropna().unique().tolist()
            )

        self._build_feature_names()
        self._fitted = True
        logger.info(
            "ClinicalFeatureExtractor fit: %d cancer types, %d features",
            len(self._cancer_types), len(self.feature_names),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode clinical columns into a numeric array.

        Args:
            df: DataFrame with clinical columns.

        Returns:
            Array of shape ``(len(df), n_clinical_features)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        n = len(df)
        parts: list[np.ndarray] = []

        parts.append(self._encode_age(df, n))
        parts.append(self._encode_sex(df, n))
        parts.append(self._encode_stage(df, n))
        parts.append(self._encode_cancer_type(df, n))

        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: ``fit`` then ``transform``."""
        return self.fit(df).transform(df)

    # --- Name registry -------------------------------------------------------

    def _build_feature_names(self) -> None:
        names = ["age_norm", "sex_binary", "stage_ordinal"]
        for ct in self._cancer_types:
            names.append(f"ctype_{ct}")
        self.feature_names = names

    # --- Column encoders -----------------------------------------------------

    def _encode_age(self, df: pd.DataFrame, n: int) -> np.ndarray:
        out = np.full((n, 1), self._age_median, dtype=np.float64)
        if "age" in df.columns:
            vals = pd.to_numeric(df["age"], errors="coerce").to_numpy(
                dtype=np.float64, copy=True,
            )
            mask = np.isnan(vals)
            vals[mask] = self._age_median
            out[:, 0] = (vals - self._age_min) / (self._age_max - self._age_min)
        else:
            out[:, 0] = (self._age_median - self._age_min) / (
                self._age_max - self._age_min
            )
        return out

    def _encode_sex(self, df: pd.DataFrame, n: int) -> np.ndarray:
        out = np.zeros((n, 1), dtype=np.float64)
        if "sex" in df.columns:
            for i, raw in enumerate(df["sex"]):
                if pd.isna(raw):
                    text = self._sex_mode
                else:
                    text = str(raw).strip()
                out[i, 0] = 1.0 if text.lower().startswith("m") else 0.0
        return out

    def _encode_stage(self, df: pd.DataFrame, n: int) -> np.ndarray:
        out = np.full((n, 1), self._stage_mode / 4.0, dtype=np.float64)
        if "stage" in df.columns:
            for i, raw in enumerate(df["stage"]):
                val = _parse_stage(raw)
                if np.isnan(val):
                    val = self._stage_mode
                out[i, 0] = val / 4.0
        return out

    def _encode_cancer_type(self, df: pd.DataFrame, n: int) -> np.ndarray:
        n_types = len(self._cancer_types)
        if n_types == 0:
            return np.zeros((n, 0), dtype=np.float64)

        out = np.zeros((n, n_types), dtype=np.float64)
        if "cancer_type" in df.columns:
            for i, raw in enumerate(df["cancer_type"]):
                if pd.isna(raw):
                    continue
                text = str(raw).strip()
                if text in self._cancer_types:
                    j = self._cancer_types.index(text)
                    out[i, j] = 1.0
        return out
