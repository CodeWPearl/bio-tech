"""Data exploration endpoints for genes, stats, and model info."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import DatasetStats, GeneInfo, ModelInfo
from api.services.feature_service import KNOWN_CANCER_DRIVERS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["exploration"])

GENE_DATA: dict[str, dict] = {
    gene: {
        "variant_count": 0,
        "class_distribution": {
            "Pathogenic": 0,
            "Likely Pathogenic": 0,
            "Benign": 0,
            "Likely Benign": 0,
        },
    }
    for gene in KNOWN_CANCER_DRIVERS
}


@router.get("/genes", response_model=list[dict])
async def list_genes() -> list[dict]:
    """List all genes in the training data with variant counts."""
    genes = []
    for gene, data in sorted(GENE_DATA.items()):
        genes.append({
            "gene_symbol": gene,
            "variant_count": data["variant_count"],
            "is_cancer_driver": gene in KNOWN_CANCER_DRIVERS,
        })
    return genes


@router.get("/genes/{gene_symbol}", response_model=GeneInfo)
async def get_gene_info(gene_symbol: str) -> GeneInfo:
    """Get detailed information about a specific gene."""
    gene_upper = gene_symbol.upper()

    if gene_upper not in GENE_DATA and gene_upper not in KNOWN_CANCER_DRIVERS:
        raise HTTPException(
            status_code=404,
            detail=f"Gene '{gene_symbol}' not found in the dataset",
        )

    data = GENE_DATA.get(gene_upper, {
        "variant_count": 0,
        "class_distribution": {
            "Pathogenic": 0,
            "Likely Pathogenic": 0,
            "Benign": 0,
            "Likely Benign": 0,
        },
    })

    cosmic_info = None
    if gene_upper in KNOWN_CANCER_DRIVERS:
        cosmic_info = (
            f"{gene_upper} is listed in the COSMIC Cancer Gene Census"
        )

    return GeneInfo(
        gene_symbol=gene_upper,
        variant_count=data["variant_count"],
        class_distribution=data["class_distribution"],
        is_known_cancer_driver=gene_upper in KNOWN_CANCER_DRIVERS,
        cosmic_census_info=cosmic_info,
    )


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats() -> DatasetStats:
    """Return dataset-level statistics."""
    total = sum(d["variant_count"] for d in GENE_DATA.values())

    class_dist: dict[str, int] = {
        "Pathogenic": 0,
        "Likely Pathogenic": 0,
        "Benign": 0,
        "Likely Benign": 0,
    }
    for data in GENE_DATA.values():
        for cls, count in data["class_distribution"].items():
            class_dist[cls] = class_dist.get(cls, 0) + count

    sorted_genes = sorted(
        GENE_DATA.items(),
        key=lambda x: x[1]["variant_count"],
        reverse=True,
    )
    top_genes = [
        {g: d["variant_count"]} for g, d in sorted_genes[:20]
    ]

    cancer_types = [
        "Breast Invasive Carcinoma",
        "Lung Adenocarcinoma",
        "Colorectal Adenocarcinoma",
        "Uterine Corpus Endometrial Carcinoma",
        "Ovarian Serous Cystadenocarcinoma",
    ]

    return DatasetStats(
        total_variants=total,
        class_distribution=class_dist,
        top_genes=top_genes,
        cancer_types=cancer_types,
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """Return model architecture summary and training metrics."""
    from api.services.model_service import ModelService

    service = ModelService()
    if not service.is_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet",
        )

    info = service.get_model_info()

    return ModelInfo(
        architecture=info["architecture"],
        fusion_type=info["fusion_type"],
        total_parameters=info["total_parameters"],
        encoder_parameters=info["encoder_parameters"],
        training_metrics={
            "num_classes": info["num_classes"],
            "fusion_dim": info["fusion_dim"],
        },
        validation_auroc=None,
    )
