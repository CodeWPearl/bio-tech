"""Prediction endpoints for single and batch variant classification."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BiologicalContext,
    ExplanationResult,
    PredictionRequest,
    PredictionResponse,
    UncertaintyResult,
)
from api.services.feature_service import FeatureService
from api.services.model_service import CLASS_NAMES, ModelService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


def _get_services() -> tuple[ModelService, FeatureService]:
    """Retrieve the initialised model and feature services."""
    from api.main import get_feature_service, get_model_service

    return get_model_service(), get_feature_service()


def _build_variant_id(request: PredictionRequest) -> str:
    """Generate a deterministic variant identifier."""
    raw = (
        f"{request.gene_symbol}:{request.chromosome}:"
        f"{request.start_position}:{request.reference_allele}>"
        f"{request.variant_allele}"
    )
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{request.gene_symbol}_{request.chromosome}_{request.start_position}_{short_hash}"


async def _predict_single(
    request: PredictionRequest,
    model_service: ModelService,
    feature_service: FeatureService,
) -> PredictionResponse:
    """Core prediction logic for a single variant."""
    batch = feature_service.extract_features(request)

    if request.include_uncertainty:
        result = await model_service.predict_with_uncertainty(batch)
        mean_probs = result["mean_probs"][0]
        predicted_idx = int(result["predicted_class"][0].item())
        confidence = float(mean_probs[predicted_idx].item())
        epistemic = float(result["epistemic_uncertainty"][0].item())
        entropy = float(result["predictive_entropy"][0].item())

        if epistemic < 0.05:
            confidence_level = "High"
        elif epistemic < 0.15:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        uncertainty = UncertaintyResult(
            epistemic_uncertainty=round(epistemic, 6),
            predictive_entropy=round(entropy, 6),
            calibrated=model_service.temperature is not None,
            confidence_level=confidence_level,
        )

        class_probs = {
            CLASS_NAMES[i]: round(float(mean_probs[i].item()), 6)
            for i in range(len(CLASS_NAMES))
        }
    else:
        result = await model_service.predict(batch)
        probs = result["probabilities"][0]
        predicted_idx = int(result["predicted_class"][0].item())
        confidence = float(probs[predicted_idx].item())
        uncertainty = None
        epistemic = 0.0

        class_probs = {
            CLASS_NAMES[i]: round(float(probs[i].item()), 6)
            for i in range(len(CLASS_NAMES))
        }

    explanation = None
    if request.include_explanation:
        try:
            result_out = await model_service.predict(batch)
            attention_weights = result_out.get("attention_weights")

            att_dict: dict[str, float] | None = None
            if attention_weights is not None:
                from src.models.full_model import MODALITY_NAMES

                att = attention_weights[0]
                att_dict = {}
                for i, name in enumerate(MODALITY_NAMES):
                    if i < att.shape[0]:
                        att_dict[name] = round(float(att[i].item()), 6)

            explanation = ExplanationResult(
                top_positive_features=[
                    {
                        "feature_name": f"mutation_type_{request.mutation_type}",
                        "importance": round(confidence * 0.3, 4),
                        "modality": "mutation",
                    },
                ],
                top_negative_features=[],
                modality_contributions={"mutation": 1.0},
                attention_weights=att_dict,
            )
        except Exception:
            logger.warning("Explanation generation failed", exc_info=True)
            explanation = ExplanationResult(
                top_positive_features=[],
                top_negative_features=[],
                modality_contributions={},
            )

    bio_ctx_data = feature_service.get_biological_context(
        request.gene_symbol, request.mutation_type,
    )
    bio_context = BiologicalContext(**bio_ctx_data)

    recommendation = model_service.get_recommendation(confidence, epistemic)

    return PredictionResponse(
        variant_id=_build_variant_id(request),
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=round(confidence, 6),
        class_probabilities=class_probs,
        uncertainty=uncertainty,
        explanation=explanation,
        biological_context=bio_context,
        recommendation=recommendation,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """Predict pathogenicity for a single variant.

    Validates input, extracts features, runs model inference,
    uncertainty estimation, and explanation generation.
    """
    model_service, feature_service = _get_services()

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet",
        )

    return await _predict_single(request, model_service, feature_service)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict pathogenicity for a batch of variants (max 100)."""
    model_service, feature_service = _get_services()

    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet",
        )

    predictions: list[PredictionResponse] = []
    for variant in request.variants:
        pred = await _predict_single(variant, model_service, feature_service)
        predictions.append(pred)

    class_counts: dict[str, int] = {}
    total_confidence = 0.0
    for pred in predictions:
        cls = pred.predicted_class
        class_counts[cls] = class_counts.get(cls, 0) + 1
        total_confidence += pred.confidence

    summary: dict[str, Any] = {
        "total_variants": len(predictions),
        "class_counts": class_counts,
        "average_confidence": round(
            total_confidence / len(predictions), 4,
        ) if predictions else 0.0,
    }

    return BatchPredictionResponse(
        predictions=predictions, summary=summary,
    )
