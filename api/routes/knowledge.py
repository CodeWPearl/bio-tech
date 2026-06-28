"""Knowledge base endpoints for cancer types, precautions, and treatments."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from webapp.data.cancer_knowledge import (
    CANCER_KNOWLEDGE,
    get_cancer_info,
    list_cancer_types,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/cancer-types")
async def get_cancer_types() -> list[dict]:
    """List all cancer types with abbreviations and key gene summaries."""
    results = []
    for name in list_cancer_types():
        info = CANCER_KNOWLEDGE[name]
        results.append({
            "name": name,
            "abbreviation": info["abbreviation"],
            "overview": info["overview"][:200] + "...",
            "key_genes": info["key_genes"],
            "num_treatments": len(info["treatment_options"]),
            "num_precautions": len(info["precautions"]),
        })
    return results


@router.get("/cancer-types/{cancer_type}")
async def get_cancer_type_detail(cancer_type: str) -> dict:
    """Return full precaution, treatment, and survival data for a cancer type."""
    normalized = cancer_type.replace("-", " ").replace("_", " ")

    info = get_cancer_info(normalized)
    if info is None:
        for key in CANCER_KNOWLEDGE:
            if key.lower() == normalized.lower():
                info = CANCER_KNOWLEDGE[key]
                normalized = key
                break

    if info is None:
        for key, val in CANCER_KNOWLEDGE.items():
            if val["abbreviation"].lower() == normalized.lower():
                info = val
                normalized = key
                break

    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cancer type '{cancer_type}' not found. "
            f"Available types: {', '.join(list_cancer_types())}",
        )

    return {"cancer_type": normalized, **info}
