"""Evaluation modules for pathogenicity prediction.

Public API:
    - :func:`compute_all_metrics` — full metric suite
    - :func:`get_confusion_matrix` — confusion matrix
    - :func:`classification_report_df` — per-class report as DataFrame
    - :func:`compute_ci` — bootstrap confidence intervals
    - :func:`run_baselines` — baseline model comparison
    - :func:`run_biological_validation` — COSMIC and ClinVar validation
    - :func:`run_external_comparison` — compare against SIFT/PolyPhen-2/CADD/REVEL
    - :func:`compare_external_tools` — external tool comparison (in-memory)
"""

from src.evaluation.benchmarks import run_baselines
from src.evaluation.biological_validation import run_biological_validation
from src.evaluation.external_tools import compare_external_tools, run_external_comparison
from src.evaluation.metrics import (
    classification_report_df,
    compute_all_metrics,
    compute_ci,
    get_confusion_matrix,
)

__all__ = [
    "compute_all_metrics",
    "get_confusion_matrix",
    "classification_report_df",
    "compute_ci",
    "run_baselines",
    "run_biological_validation",
    "run_external_comparison",
    "compare_external_tools",
]
