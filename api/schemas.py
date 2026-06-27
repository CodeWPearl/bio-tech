"""Pydantic request/response models for the pathogenicity prediction API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Single variant prediction request."""

    gene_symbol: str = Field(..., examples=["BRCA1"])
    mutation_type: str = Field(..., examples=["Missense_Mutation"])
    chromosome: str = Field(..., examples=["17"])
    start_position: int = Field(..., examples=[43044295])
    reference_allele: str = Field(..., examples=["A"])
    variant_allele: str = Field(..., examples=["T"])
    protein_change: Optional[str] = Field(None, examples=["p.C61G"])
    cancer_type: Optional[str] = Field(None, examples=["Breast Invasive Carcinoma"])
    include_explanation: bool = True
    include_uncertainty: bool = True

    @field_validator("chromosome")
    @classmethod
    def validate_chromosome(cls, v: str) -> str:
        valid = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}
        cleaned = v.upper().replace("CHR", "")
        if cleaned not in valid:
            raise ValueError(f"Invalid chromosome: {v}")
        return cleaned

    @field_validator("reference_allele", "variant_allele")
    @classmethod
    def validate_allele(cls, v: str) -> str:
        valid_bases = set("ACGTN-")
        if not all(c in valid_bases for c in v.upper()):
            raise ValueError(f"Invalid allele: {v}")
        return v.upper()


class UncertaintyResult(BaseModel):
    """Uncertainty estimation results."""

    epistemic_uncertainty: float
    predictive_entropy: float
    calibrated: bool
    confidence_level: str = Field(..., pattern="^(High|Medium|Low)$")


class ExplanationResult(BaseModel):
    """SHAP-based explanation results."""

    top_positive_features: list[dict] = Field(default_factory=list)
    top_negative_features: list[dict] = Field(default_factory=list)
    modality_contributions: dict[str, float] = Field(default_factory=dict)
    attention_weights: Optional[dict[str, float]] = None


class BiologicalContext(BaseModel):
    """Biological annotation context for a variant."""

    gene_symbol: str
    is_known_cancer_driver: bool
    cosmic_census_info: Optional[str] = None
    clinvar_entries: int = 0
    variant_type_description: str = ""


class PredictionResponse(BaseModel):
    """Single variant prediction response."""

    variant_id: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    uncertainty: Optional[UncertaintyResult] = None
    explanation: Optional[ExplanationResult] = None
    biological_context: BiologicalContext
    recommendation: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (up to 100 variants)."""

    variants: list[PredictionRequest] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with summary statistics."""

    predictions: list[PredictionResponse]
    summary: dict


class GeneInfo(BaseModel):
    """Gene information response."""

    gene_symbol: str
    variant_count: int
    class_distribution: dict[str, int]
    is_known_cancer_driver: bool
    cosmic_census_info: Optional[str] = None


class DatasetStats(BaseModel):
    """Dataset statistics response."""

    total_variants: int
    class_distribution: dict[str, int]
    top_genes: list[dict[str, int]]
    cancer_types: list[str]


class ModelInfo(BaseModel):
    """Model architecture information response."""

    architecture: str
    fusion_type: str
    total_parameters: int
    encoder_parameters: dict[str, int]
    training_metrics: dict[str, float]
    validation_auroc: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


class SHAPRequest(BaseModel):
    """SHAP explanation request for a specific prediction."""

    gene_symbol: str
    mutation_type: str
    chromosome: str
    start_position: int
    reference_allele: str
    variant_allele: str
    protein_change: Optional[str] = None
    cancer_type: Optional[str] = None
    target_class: Optional[int] = None


class AttentionRequest(BaseModel):
    """Attention weight visualization request."""

    gene_symbol: str
    mutation_type: str
    chromosome: str
    start_position: int
    reference_allele: str
    variant_allele: str
    protein_change: Optional[str] = None
    cancer_type: Optional[str] = None
