"""FastAPI application for the Cancer Mutation Pathogenicity Predictor.

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import explain, explore, knowledge, predict
from api.schemas import HealthResponse
from api.services.feature_service import FeatureService
from api.services.model_service import ModelService

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"

_model_service: ModelService | None = None
_feature_service: FeatureService | None = None


def get_model_service() -> ModelService:
    """Return the global ModelService singleton."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_feature_service() -> FeatureService:
    """Return the global FeatureService instance."""
    global _feature_service
    if _feature_service is None:
        raise RuntimeError("FeatureService not initialized")
    return _feature_service


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle for loading model and services."""
    global _model_service, _feature_service

    logger.info("Starting Cancer Mutation Pathogenicity API v%s", API_VERSION)

    _model_service = ModelService()

    checkpoint_path = Path("results/best_model.ckpt")
    config_path = Path("configs/default.yaml")

    _model_service.load_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    calibration_path = Path("results/calibration_temperature.txt")
    if calibration_path.is_file():
        temperature = float(calibration_path.read_text().strip())
        _model_service.load_calibration(temperature)

    _feature_service = FeatureService(
        config=_model_service.config,
        pipeline_path=Path("results/feature_pipeline.pkl"),
    )

    logger.info("API startup complete — model loaded and ready")

    yield

    logger.info("Shutting down API")
    ModelService.reset()
    _model_service = None
    _feature_service = None


app = FastAPI(
    title="Cancer Mutation Pathogenicity Predictor API",
    description=(
        "REST API for predicting pathogenicity of cancer-associated gene "
        "mutations using multi-omics deep learning. Provides variant "
        "classification, uncertainty estimation, and SHAP-based explanations."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(explore.router)
app.include_router(explain.router)
app.include_router(knowledge.router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    service = get_model_service()
    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded,
        version=API_VERSION,
    )


@app.get("/version")
async def get_version() -> dict:
    """Return the API version."""
    return {
        "version": API_VERSION,
        "name": "Cancer Mutation Pathogenicity Predictor API",
    }
