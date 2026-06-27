"""Singleton model service for thread-safe inference."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.explainability.shap_explainer import SHAPExplainer
from src.models.full_model import MODALITY_NAMES, PathogenicityPredictor
from src.uncertainty.calibration import TemperatureScaling
from src.uncertainty.mc_dropout import MCDropoutPredictor
from src.utils.config import Config, load_config

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "Pathogenic",
    "Likely Pathogenic",
    "Benign",
    "Likely Benign",
]

HIGH_CONFIDENCE_THRESHOLD: float = 0.7
LOW_CONFIDENCE_THRESHOLD: float = 0.5
UNCERTAINTY_REVIEW_THRESHOLD: float = 0.1


class ModelService:
    """Singleton service wrapping model loading and inference.

    Handles GPU/CPU device management, MC Dropout uncertainty, SHAP
    explanation, and thread-safe inference via asyncio locks.
    """

    _instance: ModelService | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> ModelService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.model: PathogenicityPredictor | None = None
        self.config: Config | None = None
        self.mc_predictor: MCDropoutPredictor | None = None
        self.shap_explainer: SHAPExplainer | None = None
        self.temperature: float | None = None
        self.temperature_scaler: TemperatureScaling | None = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._inference_lock = asyncio.Lock()
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded successfully."""
        return self._model_loaded

    def load_model(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "configs/default.yaml",
    ) -> None:
        """Load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            config_path: Path to the YAML configuration file.
        """
        config_file = Path(config_path)
        if not config_file.is_file():
            logger.warning(
                "Config not found at %s, using minimal defaults", config_file,
            )
            self.config = Config({
                "data": {"test_size": 0.15, "val_size": 0.15},
                "model": {
                    "num_classes": 4,
                    "fusion_dim": 256,
                    "fusion_type": "cross_attention",
                    "dropout": 0.3,
                    "mutation_input_dim": 42,
                    "mutation_embed_dim": 128,
                    "expression_input_dim": 2000,
                    "expression_embed_dim": 256,
                    "methylation_input_dim": 2000,
                    "methylation_embed_dim": 128,
                    "cnv_input_dim": 200,
                    "cnv_embed_dim": 64,
                    "clinical_input_dim": 32,
                    "clinical_embed_dim": 32,
                },
                "training": {"max_epochs": 100, "batch_size": 64},
                "experiment": {"name": "api_inference"},
            })
        else:
            self.config = load_config(config_file)

        self.model = PathogenicityPredictor.from_config(self.config)

        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_file():
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False,
            )
            if "state_dict" in checkpoint:
                state_dict = {
                    k.replace("model.", "", 1): v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("model.")
                }
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            logger.info("Model loaded from checkpoint: %s", ckpt_path)
        else:
            logger.warning(
                "Checkpoint not found at %s, using random weights", ckpt_path,
            )

        self.model.to(self.device)
        self.model.eval()

        self.mc_predictor = MCDropoutPredictor(
            self.model, n_forward_passes=30,
        )

        modality_dims = {
            "mutation": self.config.model.get("mutation_input_dim", 42),
            "expression": self.config.model.get("expression_input_dim", 2000),
            "methylation": self.config.model.get(
                "methylation_input_dim", 2000,
            ),
            "cnv": self.config.model.get("cnv_input_dim", 200),
            "clinical": self.config.model.get("clinical_input_dim", 32),
        }
        self.shap_explainer = SHAPExplainer(self.model, modality_dims)

        self._model_loaded = True
        logger.info("ModelService initialized on device: %s", self.device)

    def load_calibration(self, temperature: float) -> None:
        """Load temperature calibration parameter.

        Args:
            temperature: Fitted temperature value for post-hoc calibration.
        """
        self.temperature = temperature
        self.temperature_scaler = TemperatureScaling(
            initial_temperature=temperature,
        )
        self.temperature_scaler.temperature.data.fill_(temperature)
        logger.info("Calibration loaded with temperature: %.4f", temperature)

    async def predict(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run model forward pass with thread safety.

        Args:
            batch: Dict with modality feature tensors and modality_mask.

        Returns:
            Model output dict with logits, probabilities, etc.
        """
        async with self._inference_lock:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(batch_device)
            return outputs

    async def predict_with_uncertainty(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Run MC Dropout prediction with uncertainty estimation.

        Args:
            batch: Dict with modality feature tensors and modality_mask.

        Returns:
            Dict with mean_probs, epistemic_uncertainty, etc.
        """
        async with self._inference_lock:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            result = self.mc_predictor.predict_with_uncertainty(batch_device)
            return result

    async def explain(
        self, batch: dict[str, torch.Tensor],
        feature_names: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values for a single sample.

        Args:
            batch: Single-sample batch dict.
            feature_names: Optional per-modality feature name lists.

        Returns:
            Dict with shap_values and feature_attributions.
        """
        async with self._inference_lock:
            batch_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            result = self.shap_explainer.compute_local_explanation(
                batch_device, feature_names=feature_names,
            )
            return result

    def get_model_info(self) -> dict[str, Any]:
        """Return model architecture info and parameter counts.

        Returns:
            Dict with architecture details and parameter counts.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        encoder_params: dict[str, int] = {}
        total = 0
        for name in MODALITY_NAMES:
            count = sum(
                p.numel()
                for p in self.model.encoders[name].parameters()
                if p.requires_grad
            )
            encoder_params[name] = count
            total += count

        fusion_count = sum(
            p.numel()
            for p in self.model.fusion.parameters()
            if p.requires_grad
        )
        total += fusion_count

        if self.model.classifier is not None:
            cls_count = sum(
                p.numel()
                for p in self.model.classifier.parameters()
                if p.requires_grad
            )
            total += cls_count

        return {
            "architecture": "PathogenicityPredictor",
            "fusion_type": self.model.fusion_type,
            "total_parameters": total,
            "encoder_parameters": encoder_params,
            "fusion_parameters": fusion_count,
            "num_classes": self.model.num_classes,
            "fusion_dim": self.model.fusion_dim,
            "device": str(self.device),
        }

    def get_recommendation(
        self, confidence: float, uncertainty: float,
    ) -> str:
        """Determine clinical recommendation based on confidence/uncertainty.

        Args:
            confidence: Model prediction confidence (0-1).
            uncertainty: Epistemic uncertainty estimate.

        Returns:
            Recommendation string.
        """
        if (
            confidence >= HIGH_CONFIDENCE_THRESHOLD
            and uncertainty < UNCERTAINTY_REVIEW_THRESHOLD
        ):
            return "High confidence"
        if (
            confidence < LOW_CONFIDENCE_THRESHOLD
            or uncertainty >= UNCERTAINTY_REVIEW_THRESHOLD
        ):
            return "Low confidence"
        return "Review recommended"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (used in testing)."""
        cls._instance = None
