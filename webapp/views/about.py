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
            <h1>\U0001f4d6 About This Project</h1>
            <p>Cancer Mutation Pathogenicity Predictor — a multi-omics
            deep learning research project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">Project Overview</h3>
            <p style="line-height:1.8">
            This project implements a <strong style="color:#A5B4FC !important">research-grade
            deep learning framework</strong> for predicting the pathogenicity of
            cancer-associated gene mutations. It integrates multiple omics data types
            (genomic, transcriptomic, epigenomic) through a cross-attention fusion
            mechanism to classify variants as:
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0">
                <span style="padding:0.4rem 1rem;background:rgba(239,68,68,0.15);
                       border:1px solid rgba(239,68,68,0.3);border-radius:50px;
                       color:#F87171;font-weight:600;font-size:0.85rem">Pathogenic</span>
                <span style="padding:0.4rem 1rem;background:rgba(249,115,22,0.15);
                       border:1px solid rgba(249,115,22,0.3);border-radius:50px;
                       color:#FB923C;font-weight:600;font-size:0.85rem">Likely Pathogenic</span>
                <span style="padding:0.4rem 1rem;background:rgba(16,185,129,0.15);
                       border:1px solid rgba(16,185,129,0.3);border-radius:50px;
                       color:#34D399;font-weight:600;font-size:0.85rem">Benign</span>
                <span style="padding:0.4rem 1rem;background:rgba(52,211,153,0.15);
                       border:1px solid rgba(52,211,153,0.3);border-radius:50px;
                       color:#6EE7B7;font-weight:600;font-size:0.85rem">Likely Benign</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    arch_cols = st.columns(2)

    with arch_cols[0]:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">\U0001f4be Data Sources</h3>
                <table style="width:100%;border-collapse:separate;border-spacing:0 8px">
                    <tr><td style="color:#A5B4FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(99,102,241,0.1);border-radius:8px 0 0 8px">ClinVar</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">Pathogenicity labels (gold standard)</td></tr>
                    <tr><td style="color:#A5B4FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(99,102,241,0.1);border-radius:8px 0 0 8px">cBioPortal / TCGA</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">Multi-omics profiles</td></tr>
                    <tr><td style="color:#A5B4FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(99,102,241,0.1);border-radius:8px 0 0 8px">COSMIC Census</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">Cancer driver validation</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arch_cols[1]:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">⚙️ Model Components</h3>
                <table style="width:100%;border-collapse:separate;border-spacing:0 8px">
                    <tr><td style="color:#C084FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(139,92,246,0.1);border-radius:8px 0 0 8px">Encoders</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">Per-modality encoders</td></tr>
                    <tr><td style="color:#C084FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(139,92,246,0.1);border-radius:8px 0 0 8px">Fusion</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">Cross-attention mechanism</td></tr>
                    <tr><td style="color:#C084FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(139,92,246,0.1);border-radius:8px 0 0 8px">Uncertainty</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">MC Dropout + Temp. Scaling</td></tr>
                    <tr><td style="color:#C084FC !important;font-weight:600;padding:6px 12px;
                        background:rgba(139,92,246,0.1);border-radius:8px 0 0 8px">Explainability</td>
                        <td style="padding:6px 12px;background:rgba(255,255,255,0.03);
                        border-radius:0 8px 8px 0">SHAP + Attention weights</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech Stack
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">\U0001f6e0️ Technology Stack</h3></div>',
        unsafe_allow_html=True,
    )
    tech_cols = st.columns(4)
    techs = [
        ("\U0001f9e0 Deep Learning", "PyTorch 2.x\nPyTorch Lightning 2.x", "#6366F1"),
        ("\U0001f4ca ML Baselines", "scikit-learn\nXGBoost · LightGBM", "#8B5CF6"),
        ("\U0001f4a1 Explainability", "SHAP · LIME\nCaptum", "#EC4899"),
        ("⚙️ Infrastructure", "FastAPI · Streamlit\nMLflow · Optuna", "#F59E0B"),
    ]
    for col, (title, content, color) in zip(tech_cols, techs):
        with col:
            st.markdown(
                f"""
                <div style="background:rgba(30,27,75,0.5);border:1px solid {color}33;
                     border-radius:16px;padding:1.2rem;text-align:center;
                     border-top:3px solid {color}">
                    <p style="font-weight:700;margin:0 0 0.5rem;color:#F1F5F9 !important;
                       font-size:0.85rem">{title}</p>
                    <pre style="color:#94A3B8;font-size:0.75rem;margin:0;
                         white-space:pre-wrap">{content}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # API Endpoints
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">\U0001f310 API Endpoints</h3></div>',
        unsafe_allow_html=True,
    )
    endpoints = [
        ("GET", "/health", "API health check"),
        ("GET", "/version", "API version info"),
        ("POST", "/predict", "Single variant prediction"),
        ("POST", "/predict/batch", "Batch prediction (up to 100)"),
        ("GET", "/genes", "List all genes"),
        ("GET", "/genes/{symbol}", "Gene details"),
        ("GET", "/stats", "Dataset statistics"),
        ("GET", "/model/info", "Model architecture info"),
        ("POST", "/explain/shap", "SHAP explanations"),
        ("POST", "/explain/attention", "Attention weights"),
        ("GET", "/explain/global", "Global feature importance"),
    ]
    endpoint_html = ""
    for method, path, desc in endpoints:
        method_color = "#10B981" if method == "GET" else "#6366F1"
        endpoint_html += (
            f'<div style="display:flex;align-items:center;gap:12px;padding:0.5rem 0;'
            f'border-bottom:1px solid rgba(99,102,241,0.08)">'
            f'<span style="background:{method_color}22;color:{method_color};'
            f'padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;'
            f'min-width:50px;text-align:center;font-family:monospace">{method}</span>'
            f'<code style="color:#A5B4FC;font-size:0.85rem">{path}</code>'
            f'<span style="color:#64748B;font-size:0.85rem;margin-left:auto">{desc}</span>'
            f'</div>'
        )
    st.markdown(endpoint_html, unsafe_allow_html=True)

    st.markdown("---")

    # Getting Started
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">\U0001f680 Getting Started</h3></div>',
        unsafe_allow_html=True,
    )
    st.code(
        "# 1. Start the API backend\n"
        "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n\n"
        "# 2. Start this dashboard\n"
        "streamlit run webapp/app.py\n\n"
        "# 3. Open in browser\n"
        "# http://localhost:8501",
        language="bash",
    )

    # API Status
    health = client.health_check()
    if health:
        st.markdown("---")
        status_cols = st.columns(3)
        status = health.get("status", "unknown")
        with status_cols[0]:
            color = "#10B981" if status == "healthy" else "#EF4444"
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:10px">
                <div style="width:12px;height:12px;background:{color};border-radius:50%;
                     box-shadow:0 0 10px {color}"></div>
                <span style="color:{color};font-size:1.1rem;font-weight:700">
                {status.upper()}</span></div>""",
                unsafe_allow_html=True,
            )
        with status_cols[1]:
            loaded = health.get("model_loaded", False)
            st.metric("Model Loaded", "Yes" if loaded else "No")
        with status_cols[2]:
            st.metric("API Version", health.get("version", "N/A"))

    st.markdown("---")
    st.caption(
        "Cancer Mutation Pathogenicity Predictor  ·  "
        "Research Project  ·  Target: IEEE/Springer/Nature"
    )
