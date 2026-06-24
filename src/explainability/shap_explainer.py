"""SHAP-based explainability for the pathogenicity predictor.

Uses SHAP KernelExplainer to compute feature attributions at global and local
levels, grouped by omics modality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


def _model_predict_fn(
    model: torch.nn.Module,
    flat_input: np.ndarray,
    modality_slices: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Convert flat numpy array to model batch dict and return probabilities.

    Args:
        model: The pathogenicity predictor model.
        flat_input: ``(n_samples, total_features)`` array.
        modality_slices: Mapping from modality name to ``(start, end)`` column
            indices in the flat array.

    Returns:
        ``(n_samples, num_classes)`` probability array.
    """
    device = next(model.parameters()).device
    batch: dict[str, torch.Tensor] = {}
    for name, (start, end) in modality_slices.items():
        batch[name] = torch.tensor(
            flat_input[:, start:end], dtype=torch.float32, device=device,
        )
    mask = torch.ones(
        flat_input.shape[0], len(modality_slices), dtype=torch.bool, device=device,
    )
    batch["modality_mask"] = mask

    model.eval()
    with torch.no_grad():
        result = model(batch)
    return result["probabilities"].cpu().numpy()


class SHAPExplainer:
    """SHAP-based feature attribution for the multi-omics model.

    Args:
        model: Trained :class:`PathogenicityPredictor`.
        modality_dims: Mapping from modality name to its raw input dimension.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modality_dims: dict[str, int],
    ) -> None:
        self.model = model
        self.modality_dims = modality_dims
        self._modality_slices = self._compute_slices()
        self._total_dim = sum(modality_dims[n] for n in MODALITY_NAMES if n in modality_dims)

    def _compute_slices(self) -> dict[str, tuple[int, int]]:
        """Compute column ranges for each modality in the flat feature vector."""
        slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in MODALITY_NAMES:
            if name not in self.modality_dims:
                continue
            dim = self.modality_dims[name]
            slices[name] = (offset, offset + dim)
            offset += dim
        return slices

    def _batch_to_flat(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten a model batch dict into a 2-D numpy array."""
        parts: list[np.ndarray] = []
        for name in MODALITY_NAMES:
            if name in batch and name in self.modality_dims:
                t = batch[name]
                parts.append(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t)
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _normalize_shap_values(
        shap_values: np.ndarray | list[np.ndarray],
    ) -> list[np.ndarray]:
        """Normalise SHAP output to a list of 2-D arrays (one per class).

        SHAP versions return different layouts:
        - list of ``(n_samples, n_features)`` arrays (one per class)
        - 3-D array ``(n_samples, n_features, n_classes)``
        """
        if isinstance(shap_values, list):
            return shap_values
        if shap_values.ndim == 3:
            n_classes = shap_values.shape[2]
            return [shap_values[:, :, c] for c in range(n_classes)]
        return [shap_values]

    @staticmethod
    def _mean_abs_shap(sv_list: list[np.ndarray]) -> np.ndarray:
        """Per-feature mean |SHAP| averaged across classes and samples."""
        mean_abs = np.mean([np.abs(sv) for sv in sv_list], axis=0)
        if mean_abs.ndim > 1:
            return np.mean(mean_abs, axis=0)
        return mean_abs

    def _build_feature_names(
        self, feature_names: dict[str, list[str]] | None,
    ) -> list[str]:
        """Build a flat list of feature names from per-modality names."""
        names: list[str] = []
        for mod in MODALITY_NAMES:
            if mod not in self.modality_dims:
                continue
            dim = self.modality_dims[mod]
            if feature_names and mod in feature_names:
                names.extend(feature_names[mod][:dim])
                if len(feature_names[mod]) < dim:
                    for i in range(len(feature_names[mod]), dim):
                        names.append(f"{mod}_{i}")
            else:
                names.extend(f"{mod}_{i}" for i in range(dim))
        return names

    def compute_global_importance(
        self,
        test_data: dict[str, torch.Tensor],
        n_samples: int = 500,
        feature_names: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Compute global feature importance via SHAP values.

        Args:
            test_data: Batch dict with modality tensors.
            n_samples: Number of test samples to explain.
            feature_names: Optional per-modality feature name lists.

        Returns:
            Dict with ``shap_values``, ``feature_importance``, and
            ``modality_importance``.
        """
        import shap

        flat_data = self._batch_to_flat(test_data)
        n = min(n_samples, flat_data.shape[0])
        subset = flat_data[:n]

        background = flat_data[np.random.choice(flat_data.shape[0], min(100, flat_data.shape[0]), replace=False)]

        def predict(x: np.ndarray) -> np.ndarray:
            return _model_predict_fn(self.model, x, self._modality_slices)

        explainer = shap.KernelExplainer(predict, background)
        shap_values = explainer.shap_values(subset, nsamples=50)

        sv_list = self._normalize_shap_values(shap_values)
        per_feature = self._mean_abs_shap(sv_list)

        names = self._build_feature_names(feature_names)
        feature_importance = dict(zip(names, per_feature.tolist()))

        modality_importance: dict[str, float] = {}
        for mod, (start, end) in self._modality_slices.items():
            modality_importance[mod] = float(np.sum(per_feature[start:end]))

        logger.info("SHAP global importance computed for %d samples", n)
        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "modality_importance": modality_importance,
        }

    def compute_local_explanation(
        self,
        single_sample: dict[str, torch.Tensor],
        feature_names: dict[str, list[str]] | None = None,
        background: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values for one specific variant prediction.

        Args:
            single_sample: Single-sample batch dict (batch dim = 1).
            feature_names: Optional per-modality feature name lists.
            background: Background dataset for the explainer. If ``None``,
                uses the sample itself (zero-centred).

        Returns:
            Dict with ``shap_values`` and ``feature_attributions``.
        """
        import shap

        flat = self._batch_to_flat(single_sample)
        if background is None:
            background = np.zeros_like(flat)

        def predict(x: np.ndarray) -> np.ndarray:
            return _model_predict_fn(self.model, x, self._modality_slices)

        explainer = shap.KernelExplainer(predict, background)
        shap_values = explainer.shap_values(flat, nsamples=50)

        names = self._build_feature_names(feature_names)
        sv_list = self._normalize_shap_values(shap_values)
        if len(sv_list) > 1:
            attributions = {
                name: [float(sv[0, i]) for sv in sv_list]
                for i, name in enumerate(names)
            }
        else:
            attributions = {
                name: float(sv_list[0][0, i]) for i, name in enumerate(names)
            }

        return {
            "shap_values": shap_values,
            "feature_attributions": attributions,
        }

    def generate_shap_plots(
        self,
        shap_values: np.ndarray | list[np.ndarray],
        feature_names: dict[str, list[str]] | None,
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate and save SHAP visualisation plots.

        Args:
            shap_values: SHAP values from :meth:`compute_global_importance`.
            feature_names: Optional per-modality feature name lists.
            output_dir: Directory to save plots.

        Returns:
            List of saved plot file paths.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        names = self._build_feature_names(feature_names)

        sv_list = self._normalize_shap_values(shap_values)
        sv_for_summary = sv_list[0]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            sv_for_summary,
            feature_names=names,
            show=False,
            max_display=30,
        )
        path = output_path / "shap_summary_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)

        per_feature = self._mean_abs_shap(sv_list)
        n_top = min(30, len(per_feature))
        top_idx = np.argsort(per_feature)[-n_top:][::-1]
        top_names = [names[int(i)] for i in top_idx]
        top_vals = [float(per_feature[int(i)]) for i in top_idx]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_names)), top_vals[::-1])
        plt.yticks(range(len(top_names)), top_names[::-1])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 30 Features by SHAP Importance")
        plt.tight_layout()
        path = output_path / "shap_bar_top30.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)

        modality_importance: dict[str, float] = {}
        for mod, (start, end) in self._modality_slices.items():
            modality_importance[mod] = float(np.sum(per_feature[start:end]))
        mods = list(modality_importance.keys())
        vals = [modality_importance[m] for m in mods]

        plt.figure(figsize=(8, 5))
        plt.bar(mods, vals, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"])
        plt.xlabel("Modality")
        plt.ylabel("Sum of Mean |SHAP value|")
        plt.title("Modality Importance (SHAP)")
        plt.tight_layout()
        path = output_path / "shap_modality_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(path)

        logger.info("Saved %d SHAP plots to %s", len(saved), output_path)
        return saved
