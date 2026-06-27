"""Explanation endpoints for SHAP values and attention weights."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import AttentionRequest, ExplanationResult, SHAPRequest
from api.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explanations"])


@router.post("/explain/shap", response_model=ExplanationResult)
async def explain_shap(request: SHAPRequest) -> ExplanationResult:
    """Compute SHAP explanation for a specific prediction."""
    from api.main import get_feature_service
    from api.schemas import PredictionRequest

    service = ModelService()
    feature_service = get_feature_service()

    if not service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet",
        )

    pred_request = PredictionRequest(
        gene_symbol=request.gene_symbol,
        mutation_type=request.mutation_type,
        chromosome=request.chromosome,
        start_position=request.start_position,
        reference_allele=request.reference_allele,
        variant_allele=request.variant_allele,
        protein_change=request.protein_change,
        cancer_type=request.cancer_type,
        include_explanation=False,
        include_uncertainty=False,
    )

    batch = feature_service.extract_features(pred_request)

    try:
        result = await service.explain(batch)
    except Exception:
        logger.warning("SHAP computation failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="SHAP explanation computation failed",
        )

    attributions = result.get("feature_attributions", {})

    positive_features = []
    negative_features = []

    for feat_name, importance in sorted(
        attributions.items(),
        key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        reverse=True,
    )[:10]:
        val = importance if isinstance(importance, (int, float)) else 0.0
        modality = feat_name.split("_")[0] if "_" in feat_name else "unknown"
        entry = {
            "feature_name": feat_name,
            "importance": round(abs(val), 6),
            "modality": modality,
        }
        if val >= 0:
            positive_features.append(entry)
        else:
            negative_features.append(entry)

    modality_contributions: dict[str, float] = {}
    for feat_name, importance in attributions.items():
        modality = feat_name.split("_")[0] if "_" in feat_name else "unknown"
        val = importance if isinstance(importance, (int, float)) else 0.0
        modality_contributions[modality] = (
            modality_contributions.get(modality, 0.0) + abs(val)
        )

    return ExplanationResult(
        top_positive_features=positive_features[:5],
        top_negative_features=negative_features[:5],
        modality_contributions={
            k: round(v, 6) for k, v in modality_contributions.items()
        },
    )


@router.post("/explain/attention", response_model=dict)
async def explain_attention(request: AttentionRequest) -> dict:
    """Return attention weight visualization data for a prediction."""
    from api.main import get_feature_service
    from api.schemas import PredictionRequest

    service = ModelService()
    feature_service = get_feature_service()

    if not service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet",
        )

    pred_request = PredictionRequest(
        gene_symbol=request.gene_symbol,
        mutation_type=request.mutation_type,
        chromosome=request.chromosome,
        start_position=request.start_position,
        reference_allele=request.reference_allele,
        variant_allele=request.variant_allele,
        protein_change=request.protein_change,
        cancer_type=request.cancer_type,
        include_explanation=False,
        include_uncertainty=False,
    )

    batch = feature_service.extract_features(pred_request)
    result = await service.predict(batch)

    attention_weights = result.get("attention_weights")
    if attention_weights is None:
        return {
            "attention_weights": None,
            "message": "Attention weights not available for this fusion type",
        }

    from src.models.full_model import MODALITY_NAMES

    att = attention_weights[0]
    weights_dict: dict[str, float] = {}
    for i, name in enumerate(MODALITY_NAMES):
        if i < att.shape[0]:
            weights_dict[name] = round(float(att[i].item()), 6)

    return {
        "attention_weights": weights_dict,
        "modalities": list(weights_dict.keys()),
    }


@router.get("/explain/global", response_model=dict)
async def explain_global() -> dict:
    """Return precomputed global feature importance.

    In production, this would load precomputed SHAP values from disk.
    """
    from src.models.full_model import MODALITY_NAMES

    return {
        "modality_importance": {
            name: 1.0 / len(MODALITY_NAMES) for name in MODALITY_NAMES
        },
        "note": "Global importance requires precomputation on the full dataset",
        "top_features": [
            {"feature": "mutation_type", "importance": 0.15, "modality": "mutation"},
            {"feature": "chromosome", "importance": 0.08, "modality": "mutation"},
            {"feature": "gene_driver_status", "importance": 0.12, "modality": "clinical"},
        ],
    }
