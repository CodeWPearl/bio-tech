"""Home / landing page for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import styled_metric_card


def render(client: APIClient) -> None:
    """Render the Home landing page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>\U0001f9ec Cancer Mutation Pathogenicity Predictor</h1>
            <p>Multi-omics deep learning for variant classification with
            uncertainty estimation and explainability</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = client.get_stats()
    model_info = client.get_model_info()
    total_variants = stats.get("total_variants", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = f"{total_variants:,}" if isinstance(total_variants, int) and total_variants > 0 else "N/A"
        st.markdown(
            styled_metric_card("Training Variants", val, icon="\U0001f4ca", accent="#6366F1"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            styled_metric_card("Architecture", model_info.get("architecture", "Multi-Omics DL"), icon="\U0001f9e0", accent="#8B5CF6"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            styled_metric_card("Fusion", model_info.get("fusion_type", "Cross-Attention"), icon="\U0001f517", accent="#EC4899"),
            unsafe_allow_html=True,
        )
    with col4:
        params = model_info.get("total_parameters", 0)
        params_str = f"{params / 1000:.1f}K" if params > 0 else "N/A"
        st.markdown(
            styled_metric_card("Parameters", params_str, icon="⚙️", accent="#F59E0B"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">\U0001f680 Quick Start Guide</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **1. \U0001f52c Single Prediction** — Enter a variant (gene, mutation type,
            position, alleles) and get a pathogenicity prediction with
            confidence score, uncertainty estimation, and feature explanations.

            **2. \U0001f4ca Batch Analysis** — Upload a CSV/TSV file with multiple variants
            for high-throughput screening. Download results with predictions
            and confidence scores.

            **3. \U0001f4c8 Model Performance** — Explore ROC curves, PR curves, confusion
            matrices, and ablation studies showing the model's capabilities.

            **4. \U0001f9ea Data Explorer** — Browse genes, cancer types, and class
            distributions in the training dataset.
            """
        )

        health = client.health_check()
        if health and health.get("status") == "healthy":
            st.success(
                "**API is running.** Select **Single Prediction** in the sidebar to get started."
            )
        else:
            st.warning(
                "**API is offline.** Start the backend first:\n\n"
                "```\nuvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n```"
            )

    with right:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">\U0001f3d7️ Architecture</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        arch_path = Path("results/figures/fig01_model_architecture.png")
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)
        else:
            st.markdown(
                """
                <div style="background:rgba(30,27,75,0.5);border:1px solid rgba(99,102,241,0.15);
                     border-radius:16px;padding:2rem;text-align:center">
                <pre style="color:#A5B4FC;font-size:0.75rem;margin:0;line-height:1.5">
  ┌──────────────┐
  │  Genomic     │──┐
  │  Encoder     │  │
  ├──────────────┤  │  ┌──────────────┐
  │  Expression  │──┼──┤  Cross-Attn  │──→ Class
  │  Encoder     │  │  │   Fusion     │
  ├──────────────┤  │  └──────────────┘
  │  Methylation │──┘
  │  Encoder     │
  └──────────────┘
                </pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">\U0001f4dc Research Abstract</h3>
            <p style="font-style:italic;line-height:1.8;color:#94A3B8 !important">
            Accurate classification of cancer-associated gene mutations as pathogenic
            or benign is essential for precision oncology. We present a multi-omics
            deep learning framework that integrates genomic, transcriptomic, and
            epigenomic data through cross-attention fusion to predict variant
            pathogenicity. The model provides calibrated uncertainty estimates via
            MC Dropout and temperature scaling, alongside SHAP-based feature
            explanations showing which omics modalities drive each prediction.
            Evaluated on ClinVar-labeled variants with TCGA multi-omics profiles,
            our approach achieves competitive AUROC with full transparency into
            the prediction rationale.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Cancer Mutation Pathogenicity Predictor  ·  "
        "Research Project  ·  IEEE/Springer/Nature Target"
    )
