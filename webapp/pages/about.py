"""About page with project information and documentation."""

from __future__ import annotations

import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import styled_metric_card


def render(client: APIClient) -> None:
    """Render the About page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>About This Project</h1>
            <p>Cancer Mutation Pathogenicity Predictor — a multi-omics
            deep learning research project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Overview")
    st.markdown(
        """
        This project implements a **research-grade deep learning framework** for
        predicting the pathogenicity of cancer-associated gene mutations. It
        integrates multiple omics data types (genomic, transcriptomic, epigenomic)
        through a cross-attention fusion mechanism to classify variants as:

        - **Pathogenic** — The mutation is disease-causing
        - **Likely Pathogenic** — Strong evidence the mutation is disease-causing
        - **Benign** — The mutation is not disease-causing
        - **Likely Benign** — Strong evidence the mutation is not disease-causing

        The system provides **calibrated uncertainty estimates** using MC Dropout
        and temperature scaling, alongside **SHAP-based explanations** showing
        which features and omics modalities contribute most to each prediction.
        """
    )

    st.markdown("---")

    st.subheader("Technical Architecture")
    arch_cols = st.columns(2)

    with arch_cols[0]:
        st.markdown("**Data Sources (all free, no authentication)**")
        st.markdown(
            """
            | Source | Purpose |
            |--------|---------|
            | **ClinVar** | Pathogenicity labels (gold standard) |
            | **cBioPortal / TCGA** | Multi-omics profiles (mutations, expression, methylation, CNV) |
            | **COSMIC Census** | Known cancer driver gene validation |
            """
        )

    with arch_cols[1]:
        st.markdown("**Model Components**")
        st.markdown(
            """
            | Component | Details |
            |-----------|---------|
            | **Encoders** | Separate encoder per omics modality |
            | **Fusion** | Cross-attention mechanism |
            | **Classifier** | Multi-layer prediction head |
            | **Uncertainty** | MC Dropout + temperature scaling |
            | **Explanations** | SHAP values + attention weights |
            """
        )

    st.markdown("---")

    st.subheader("Technology Stack")
    tech_cols = st.columns(4)
    techs = [
        ("Deep Learning", "PyTorch 2.x\nPyTorch Lightning 2.x"),
        ("ML Baselines", "scikit-learn\nXGBoost\nLightGBM"),
        ("Explainability", "SHAP\nLIME\nCaptum"),
        ("Infrastructure", "FastAPI\nStreamlit\nMLflow\nOptuna"),
    ]
    for col, (title, content) in zip(tech_cols, techs):
        with col:
            st.markdown(f"**{title}**")
            st.code(content, language=None)

    st.markdown("---")

    st.subheader("API Endpoints")
    st.markdown(
        """
        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | `/health` | GET | API health check |
        | `/version` | GET | API version info |
        | `/predict` | POST | Single variant prediction |
        | `/predict/batch` | POST | Batch prediction (up to 100) |
        | `/genes` | GET | List all genes |
        | `/genes/{symbol}` | GET | Gene details |
        | `/stats` | GET | Dataset statistics |
        | `/model/info` | GET | Model architecture info |
        | `/explain/shap` | POST | SHAP explanations |
        | `/explain/attention` | POST | Attention weights |
        | `/explain/global` | GET | Global feature importance |
        """
    )

    st.markdown("---")

    st.subheader("Getting Started")
    st.markdown(
        """
        **1. Start the API backend:**
        ```bash
        cd cancer_mutation_pathogenicity
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ```

        **2. Start this dashboard:**
        ```bash
        streamlit run webapp/app.py
        ```

        **3. Make a prediction:**
        Navigate to **Single Prediction** in the sidebar, enter a variant,
        and click **Predict Pathogenicity**.
        """
    )

    health = client.health_check()
    if health:
        st.markdown("---")
        st.subheader("API Status")
        status_cols = st.columns(3)
        with status_cols[0]:
            status = health.get("status", "unknown")
            color = "#28A745" if status == "healthy" else "#DC3545"
            st.markdown(
                f'<span style="color:{color};font-size:1.2rem;font-weight:700">'
                f'{"&#9679;" if status == "healthy" else "&#9675;"} {status.upper()}</span>',
                unsafe_allow_html=True,
            )
        with status_cols[1]:
            loaded = health.get("model_loaded", False)
            st.metric("Model Loaded", "Yes" if loaded else "No")
        with status_cols[2]:
            st.metric("API Version", health.get("version", "N/A"))

    st.markdown("---")
    st.caption(
        "Cancer Mutation Pathogenicity Predictor | "
        "Research Project | Target: IEEE/Springer/Nature journal submission"
    )
