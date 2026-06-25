"""Probability calibration via temperature scaling and ECE computation.

Provides :class:`TemperatureScaling` for post-hoc calibration, ECE and
reliability diagram utilities, and :func:`apply_calibration` to fit a
temperature on a validation set and wrap a model for calibrated inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Learn a scalar temperature for post-hoc calibration.

    Args:
        initial_temperature: Starting value for T (default 1.5).
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model logits of shape ``(batch, num_classes)``.

        Returns:
            Calibrated probabilities of shape ``(batch, num_classes)``.
        """
        return F.softmax(logits / self.temperature, dim=-1)

    def optimize_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit the temperature parameter by minimizing NLL on validation data.

        Args:
            logits: Validation set logits ``(n, num_classes)``.
            labels: Validation set labels ``(n,)``.
            lr: Learning rate for L-BFGS optimiser.
            max_iter: Maximum optimiser iterations.

        Returns:
            Optimal temperature value.
        """
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter,
        )
        nll = nn.CrossEntropyLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nll(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        optimal_t = self.temperature.item()
        logger.info("Optimal temperature: %.4f", optimal_t)
        return optimal_t


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (lower is better).
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = float(confidences[in_bin].mean())
        avg_accuracy = float(accuracies[in_bin].mean())
        ece += (n_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute data for a reliability (calibration) diagram.

    Args:
        y_true: True labels of shape ``(n,)``.
        y_prob: Predicted probabilities of shape ``(n, num_classes)``.
        n_bins: Number of equal-width bins.

    Returns:
        Tuple of ``(mean_predicted_prob, true_fraction, bin_counts)``
        arrays each of length ``n_bins``.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    mean_predicted_prob = np.zeros(n_bins)
    true_fraction = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        n_in_bin = int(in_bin.sum())
        bin_counts[i] = n_in_bin

        if n_in_bin > 0:
            mean_predicted_prob[i] = float(confidences[in_bin].mean())
            true_fraction[i] = float(accuracies[in_bin].mean())

    return mean_predicted_prob, true_fraction, bin_counts


class CalibratedModelWrapper(nn.Module):
    """Wraps a model with a fitted :class:`TemperatureScaling` layer.

    Args:
        model: The original trained model.
        temperature_scaler: A fitted TemperatureScaling instance.
    """

    def __init__(
        self,
        model: nn.Module,
        temperature_scaler: TemperatureScaling,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature_scaler = temperature_scaler

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Forward pass with calibrated probabilities.

        Args:
            batch: Dict with modality feature tensors and ``modality_mask``.

        Returns:
            Model output dict with calibrated probabilities.
        """
        outputs = self.model(batch)
        logits = outputs["logits"]
        calibrated_probs = self.temperature_scaler(logits)
        outputs["probabilities"] = calibrated_probs
        outputs["predicted_class"] = calibrated_probs.argmax(dim=-1)
        return outputs


def apply_calibration(
    model: nn.Module,
    val_loader: DataLoader,
    expand_mask_fn: Any | None = None,
) -> CalibratedModelWrapper:
    """Fit temperature scaling on the validation set and wrap the model.

    Args:
        model: Trained model to calibrate.
        val_loader: Validation data loader.
        expand_mask_fn: Optional function to expand modality mask
            (e.g. ``PathogenicityLightningModule._expand_modality_mask``).

    Returns:
        A :class:`CalibratedModelWrapper` producing calibrated probabilities.
    """
    model.eval()
    device = next(model.parameters()).device

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"]
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if expand_mask_fn is not None:
                batch_device = expand_mask_fn(batch_device)

            outputs = model(batch_device)
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    scaler = TemperatureScaling()
    scaler.optimize_temperature(logits, labels)

    logger.info("Calibration complete. Temperature = %.4f", scaler.temperature.item())
    return CalibratedModelWrapper(model, scaler)
