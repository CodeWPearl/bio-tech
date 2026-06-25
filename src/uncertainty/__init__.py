"""Uncertainty estimation and probability calibration modules."""

from src.uncertainty.calibration import (
    CalibratedModelWrapper,
    TemperatureScaling,
    apply_calibration,
    compute_ece,
    compute_reliability_diagram,
)
from src.uncertainty.deep_ensembles import DeepEnsemblePredictor
from src.uncertainty.mc_dropout import MCDropoutPredictor

__all__ = [
    "CalibratedModelWrapper",
    "DeepEnsemblePredictor",
    "MCDropoutPredictor",
    "TemperatureScaling",
    "apply_calibration",
    "compute_ece",
    "compute_reliability_diagram",
]
