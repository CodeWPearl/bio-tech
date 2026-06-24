"""Attention weight visualisation for the pathogenicity predictor.

Extracts and visualises attention weights from attention-based fusion modules
(attention, cross-attention, transformer).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class AttentionVisualizer:
    """Extract and visualise attention weights from the fusion module.

    Args:
        model: Trained :class:`PathogenicityPredictor` with an attention-based
            fusion module.
        modality_names: List of modality names in order. Defaults to the
            standard five modalities.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.modality_names = modality_names or MODALITY_NAMES

    def extract_attention_weights(
        self,
        batch: dict[str, torch.Tensor],
    ) -> np.ndarray | None:
        """Run a forward pass and extract attention weights.

        Args:
            batch: Model batch dict with modality tensors and
                ``modality_mask``.

        Returns:
            Attention weight array of shape ``(batch, n_modalities,
            n_modalities)`` or ``None`` if the fusion module does not
            expose attention weights.
        """
        self.model.eval()
        with torch.no_grad():
            result = self.model(batch)

        weights = result.get("attention_weights")
        if weights is None:
            logger.warning(
                "No attention weights available (fusion type: %s)",
                getattr(self.model, "fusion_type", "unknown"),
            )
            return None

        return weights.cpu().numpy()

    def collect_attention_weights(
        self,
        test_loader: torch.utils.data.DataLoader,
        max_batches: int = 50,
    ) -> np.ndarray:
        """Collect attention weights across multiple batches.

        Args:
            test_loader: DataLoader yielding batch dicts.
            max_batches: Maximum number of batches to process.

        Returns:
            Stacked attention weights of shape ``(total_samples,
            n_modalities, n_modalities)``.
        """
        all_weights: list[np.ndarray] = []
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            weights = self.extract_attention_weights(batch)
            if weights is not None:
                all_weights.append(weights)

        if not all_weights:
            return np.array([])

        stacked = np.concatenate(all_weights, axis=0)
        logger.info("Collected attention weights for %d samples", stacked.shape[0])
        return stacked

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        output_path: str | Path,
        title: str = "Modality Attention Heatmap",
    ) -> Path:
        """Plot a heatmap of average attention weights across samples.

        Args:
            attention_weights: Array of shape ``(n_samples, n_modalities,
                n_modalities)`` or ``(n_modalities, n_modalities)``.
            output_path: File path to save the plot.
            title: Plot title.

        Returns:
            Path to the saved plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if attention_weights.ndim == 3:
            avg_weights = attention_weights.mean(axis=0)
        else:
            avg_weights = attention_weights

        n_mod = avg_weights.shape[0]
        labels = self.modality_names[:n_mod]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            avg_weights,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            square=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Key Modality")
        ax.set_ylabel("Query Modality")

        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved attention heatmap to %s", save_path)
        return save_path

    def plot_attention_distribution(
        self,
        attention_weights: np.ndarray,
        output_path: str | Path,
        title: str = "Attention Weight Distribution by Modality",
    ) -> Path:
        """Plot box plots of attention weights per modality across samples.

        Args:
            attention_weights: Array of shape ``(n_samples, n_modalities,
                n_modalities)``.
            output_path: File path to save the plot.
            title: Plot title.

        Returns:
            Path to the saved plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_mod = attention_weights.shape[1]
        labels = self.modality_names[:n_mod]

        per_modality_received = attention_weights.mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(
            [per_modality_received[:, j] for j in range(n_mod)],
            tick_labels=labels,
            patch_artist=True,
        )

        colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        for patch, colour in zip(bp["boxes"], colours[:n_mod]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_title(title)
        ax.set_xlabel("Modality")
        ax.set_ylabel("Average Attention Weight Received")

        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved attention distribution plot to %s", save_path)
        return save_path

    def get_attention_summary(
        self,
        attention_weights: np.ndarray,
    ) -> dict[str, Any]:
        """Compute summary statistics of attention weights.

        Args:
            attention_weights: Array of shape ``(n_samples, n_modalities,
                n_modalities)``.

        Returns:
            Dict with per-modality mean, std, and the average attention matrix.
        """
        n_mod = attention_weights.shape[1]
        labels = self.modality_names[:n_mod]

        avg_matrix = attention_weights.mean(axis=0)
        per_modality_received = attention_weights.mean(axis=1)

        summary: dict[str, Any] = {
            "avg_attention_matrix": avg_matrix.tolist(),
            "modality_names": labels,
            "per_modality_stats": {},
        }
        for j, name in enumerate(labels):
            vals = per_modality_received[:, j]
            summary["per_modality_stats"][name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return summary
