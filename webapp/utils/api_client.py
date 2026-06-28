"""API client for Streamlit to communicate with the FastAPI backend."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
import streamlit as st

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8001"


class APIClient:
    """HTTP client for the Cancer Mutation Pathogenicity Predictor API."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or os.environ.get("API_URL", DEFAULT_API_URL)

    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: int = 30,
        silent: bool = False,
        **kwargs: Any,
    ) -> dict | list:
        """Make an HTTP request to the API with error handling.

        When silent=True, connection errors are logged but not displayed
        to the user (used for background/optional fetches like gene lists).
        """
        url = f"{self.api_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if not silent:
                st.error(
                    f"Cannot connect to API at {self.api_url}. "
                    "Make sure the FastAPI server is running: "
                    "`uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`"
                )
            return {}
        except requests.exceptions.Timeout:
            if not silent:
                st.error(f"Request to {endpoint} timed out after {timeout}s.")
            return {}
        except requests.exceptions.HTTPError as exc:
            if not silent:
                detail = ""
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"API error: {detail}")
            return {}

    def health_check(self) -> dict:
        """Check API health status (silent — no error popups)."""
        return self._request("GET", "/health", timeout=5, silent=True)

    def predict(self, request_dict: dict) -> dict:
        """Predict pathogenicity for a single variant."""
        return self._request("POST", "/predict", json=request_dict, timeout=30)

    def predict_batch(self, variants: list[dict]) -> dict:
        """Predict pathogenicity for a batch of variants."""
        return self._request(
            "POST",
            "/predict/batch",
            json={"variants": variants},
            timeout=120,
        )

    @st.cache_data(ttl=300, show_spinner=False)
    def get_genes(_self) -> list:
        """Fetch list of genes from the API (cached, silent on failure)."""
        result = _self._request("GET", "/genes", timeout=10, silent=True)
        if isinstance(result, list):
            return result
        return []

    @st.cache_data(ttl=300, show_spinner=False)
    def get_stats(_self) -> dict:
        """Fetch dataset statistics (cached, silent on failure)."""
        result = _self._request("GET", "/stats", timeout=10, silent=True)
        return result if isinstance(result, dict) else {}

    @st.cache_data(ttl=300, show_spinner=False)
    def get_model_info(_self) -> dict:
        """Fetch model architecture and training info (cached, silent on failure)."""
        result = _self._request("GET", "/model/info", timeout=10, silent=True)
        return result if isinstance(result, dict) else {}

    def get_gene_info(self, gene_symbol: str) -> dict:
        """Fetch detailed info for a specific gene."""
        result = self._request("GET", f"/genes/{gene_symbol}", timeout=10)
        return result if isinstance(result, dict) else {}
