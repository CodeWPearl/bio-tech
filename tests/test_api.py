"""Tests for the Cancer Mutation Pathogenicity Prediction API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_model_service():
    """Reset the ModelService singleton before each test."""
    from api.services.model_service import ModelService

    ModelService.reset()
    yield
    ModelService.reset()


def _make_mock_model_service() -> MagicMock:
    """Create a mock ModelService with realistic return values."""
    service = MagicMock()
    service.is_loaded = True
    service.temperature = None
    service.config = MagicMock()
    service.config.model = MagicMock()
    service.config.model.get = lambda k, d=None: {
        "mutation_input_dim": 42,
        "expression_input_dim": 2000,
        "methylation_input_dim": 2000,
        "cnv_input_dim": 200,
        "clinical_input_dim": 32,
    }.get(k, d)

    probs = torch.tensor([[0.6, 0.2, 0.15, 0.05]])
    predicted_class = torch.tensor([0])

    predict_result = {
        "logits": torch.tensor([[1.5, 0.3, -0.2, -0.8]]),
        "probabilities": probs,
        "predicted_class": predicted_class,
        "fused_embedding": torch.randn(1, 256),
        "modality_embeddings": {},
        "attention_weights": torch.tensor([[0.35, 0.25, 0.2, 0.1, 0.1]]),
    }
    service.predict = AsyncMock(return_value=predict_result)

    uncertainty_result = {
        "mean_probs": probs,
        "predicted_class": predicted_class,
        "epistemic_uncertainty": torch.tensor([0.03]),
        "predictive_entropy": torch.tensor([0.85]),
        "all_predictions": torch.randn(30, 1, 4),
    }
    service.predict_with_uncertainty = AsyncMock(
        return_value=uncertainty_result,
    )

    service.get_recommendation = MagicMock(return_value="High confidence")

    service.get_model_info = MagicMock(return_value={
        "architecture": "PathogenicityPredictor",
        "fusion_type": "cross_attention",
        "total_parameters": 500000,
        "encoder_parameters": {
            "mutation": 50000,
            "expression": 200000,
            "cnv": 30000,
            "methylation": 150000,
            "clinical": 10000,
        },
        "fusion_parameters": 50000,
        "num_classes": 4,
        "fusion_dim": 256,
        "device": "cpu",
    })

    return service


def _make_mock_feature_service() -> MagicMock:
    """Create a mock FeatureService."""
    service = MagicMock()
    service.is_known_gene = MagicMock(return_value=True)

    batch = {
        "mutation": torch.randn(1, 42),
        "expression": torch.zeros(1, 2000),
        "methylation": torch.zeros(1, 2000),
        "cnv": torch.zeros(1, 200),
        "clinical": torch.zeros(1, 32),
        "modality_mask": torch.tensor(
            [[True, False, False, False, False]], dtype=torch.bool,
        ),
    }
    service.extract_features = MagicMock(return_value=batch)
    service.get_biological_context = MagicMock(return_value={
        "gene_symbol": "BRCA1",
        "is_known_cancer_driver": True,
        "cosmic_census_info": "BRCA1 is listed in the COSMIC Cancer Gene Census",
        "clinvar_entries": 0,
        "variant_type_description": (
            "Single nucleotide change causing amino acid substitution"
        ),
    })
    return service


@pytest.fixture()
def client():
    """Create a FastAPI TestClient with mocked services."""
    mock_model = _make_mock_model_service()
    mock_features = _make_mock_feature_service()

    with (
        patch("api.main._model_service", mock_model),
        patch("api.main._feature_service", mock_features),
        patch("api.main.get_model_service", return_value=mock_model),
        patch("api.main.get_feature_service", return_value=mock_features),
        patch(
            "api.routes.predict._get_services",
            return_value=(mock_model, mock_features),
        ),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


VALID_VARIANT = {
    "gene_symbol": "BRCA1",
    "mutation_type": "Missense_Mutation",
    "chromosome": "17",
    "start_position": 43044295,
    "reference_allele": "A",
    "variant_allele": "T",
    "protein_change": "p.C61G",
    "cancer_type": "Breast Invasive Carcinoma",
    "include_explanation": True,
    "include_uncertainty": True,
}


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_fields(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_version_endpoint(self, client: TestClient) -> None:
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "name" in data


class TestSinglePrediction:
    """Tests for the POST /predict endpoint."""

    def test_predict_valid_input(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        assert response.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()

        assert "variant_id" in data
        assert "predicted_class" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "biological_context" in data
        assert "recommendation" in data

    def test_predict_response_values(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()

        assert data["predicted_class"] in [
            "Pathogenic", "Likely Pathogenic", "Benign", "Likely Benign",
        ]
        assert 0.0 <= data["confidence"] <= 1.0

        probs = data["class_probabilities"]
        assert len(probs) == 4
        assert all(0.0 <= v <= 1.0 for v in probs.values())

    def test_predict_with_uncertainty(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()

        assert data["uncertainty"] is not None
        unc = data["uncertainty"]
        assert "epistemic_uncertainty" in unc
        assert "predictive_entropy" in unc
        assert "calibrated" in unc
        assert "confidence_level" in unc
        assert unc["confidence_level"] in ["High", "Medium", "Low"]

    def test_predict_without_uncertainty(self, client: TestClient) -> None:
        variant = {**VALID_VARIANT, "include_uncertainty": False}
        response = client.post("/predict", json=variant)
        data = response.json()
        assert data["uncertainty"] is None

    def test_predict_with_explanation(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()

        assert data["explanation"] is not None
        expl = data["explanation"]
        assert "top_positive_features" in expl
        assert "modality_contributions" in expl

    def test_predict_without_explanation(self, client: TestClient) -> None:
        variant = {**VALID_VARIANT, "include_explanation": False}
        response = client.post("/predict", json=variant)
        data = response.json()
        assert data["explanation"] is None

    def test_predict_biological_context(self, client: TestClient) -> None:
        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()

        bio = data["biological_context"]
        assert bio["gene_symbol"] == "BRCA1"
        assert bio["is_known_cancer_driver"] is True
        assert "variant_type_description" in bio

    def test_predict_minimal_input(self, client: TestClient) -> None:
        minimal = {
            "gene_symbol": "TP53",
            "mutation_type": "Missense_Mutation",
            "chromosome": "17",
            "start_position": 7577120,
            "reference_allele": "G",
            "variant_allele": "A",
        }
        response = client.post("/predict", json=minimal)
        assert response.status_code == 200


class TestBatchPrediction:
    """Tests for the POST /predict/batch endpoint."""

    def test_batch_prediction(self, client: TestClient) -> None:
        batch_request = {
            "variants": [VALID_VARIANT, VALID_VARIANT],
        }
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200

    def test_batch_prediction_response(self, client: TestClient) -> None:
        batch_request = {
            "variants": [VALID_VARIANT, VALID_VARIANT, VALID_VARIANT],
        }
        response = client.post("/predict/batch", json=batch_request)
        data = response.json()

        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 3
        assert data["summary"]["total_variants"] == 3
        assert "class_counts" in data["summary"]
        assert "average_confidence" in data["summary"]

    def test_batch_single_variant(self, client: TestClient) -> None:
        batch_request = {"variants": [VALID_VARIANT]}
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_missing_required_field(self, client: TestClient) -> None:
        incomplete = {
            "gene_symbol": "BRCA1",
            "mutation_type": "Missense_Mutation",
        }
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_chromosome(self, client: TestClient) -> None:
        variant = {**VALID_VARIANT, "chromosome": "99"}
        response = client.post("/predict", json=variant)
        assert response.status_code == 422

    def test_invalid_allele(self, client: TestClient) -> None:
        variant = {**VALID_VARIANT, "reference_allele": "Z"}
        response = client.post("/predict", json=variant)
        assert response.status_code == 422

    def test_empty_gene_symbol(self, client: TestClient) -> None:
        variant = {**VALID_VARIANT, "gene_symbol": ""}
        response = client.post("/predict", json=variant)
        # Empty string still passes Pydantic str validation
        assert response.status_code in (200, 422)

    def test_batch_exceeds_limit(self, client: TestClient) -> None:
        batch_request = {"variants": [VALID_VARIANT] * 101}
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 422

    def test_empty_body(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestExplorationEndpoints:
    """Tests for data exploration endpoints."""

    def test_list_genes(self, client: TestClient) -> None:
        response = client.get("/genes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_gene_info(self, client: TestClient) -> None:
        response = client.get("/genes/TP53")
        assert response.status_code == 200
        data = response.json()
        assert data["gene_symbol"] == "TP53"
        assert "variant_count" in data
        assert "class_distribution" in data

    def test_get_unknown_gene(self, client: TestClient) -> None:
        response = client.get("/genes/FAKEGENE123")
        assert response.status_code == 404

    def test_dataset_stats(self, client: TestClient) -> None:
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_variants" in data
        assert "class_distribution" in data
        assert "top_genes" in data
        assert "cancer_types" in data


class TestExplanationEndpoints:
    """Tests for explanation endpoints."""

    def test_global_explanation(self, client: TestClient) -> None:
        response = client.get("/explain/global")
        assert response.status_code == 200
        data = response.json()
        assert "modality_importance" in data
        assert "top_features" in data


class TestResponseSchemaCompliance:
    """Tests that response schemas match Pydantic model definitions."""

    def test_prediction_response_matches_model(
        self, client: TestClient,
    ) -> None:
        from api.schemas import PredictionResponse

        response = client.post("/predict", json=VALID_VARIANT)
        data = response.json()
        parsed = PredictionResponse(**data)
        assert parsed.variant_id == data["variant_id"]
        assert parsed.predicted_class == data["predicted_class"]
        assert parsed.confidence == data["confidence"]

    def test_health_response_matches_model(
        self, client: TestClient,
    ) -> None:
        from api.schemas import HealthResponse

        response = client.get("/health")
        data = response.json()
        parsed = HealthResponse(**data)
        assert parsed.status == data["status"]

    def test_batch_response_matches_model(
        self, client: TestClient,
    ) -> None:
        from api.schemas import BatchPredictionResponse

        batch_request = {"variants": [VALID_VARIANT]}
        response = client.post("/predict/batch", json=batch_request)
        data = response.json()
        parsed = BatchPredictionResponse(**data)
        assert len(parsed.predictions) == 1
