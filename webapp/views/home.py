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
            <h1>Cancer Mutation Pathogenicity Predictor</h1>
            <p>Multi-omics deep learning for variant classification with
            uncertainty estimation and explainability</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    stats = client.get_stats()
    total_variants = stats.get("total_variants", "N/A")
    model_info = client.get_model_info()
    training_metrics = model_info.get("training_metrics", {})

    with col1:
        st.markdown(
            styled_metric_card("Training Variants", f"{total_variants:,}" if isinstance(total_variants, int) else str(total_variants)),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            styled_metric_card("Architecture", model_info.get("architecture", "Multi-Omics DL")),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            styled_metric_card("Fusion Strategy", model_info.get("fusion_type", "Cross-Attention")),
            unsafe_allow_html=True,
        )
    with col4:
        params = model_info.get("total_parameters", 0)
        params_str = f"{params / 1000:.1f}K" if params > 0 else "N/A"
        st.markdown(
            styled_metric_card("Parameters", params_str),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Quick Start Guide")
        st.markdown(
            """
            1. **Single Prediction** — Enter a variant (gene, mutation type,
               position, alleles) and get a pathogenicity prediction with
               confidence score, uncertainty estimation, and feature explanations.

            2. **Batch Analysis** — Upload a CSV/TSV file with multiple variants
               for high-throughput screening. Download results with predictions
               and confidence scores.

            3. **Model Performance** — Explore ROC curves, PR curves, confusion
               matrices, and ablation studies showing the model's capabilities.

            4. **Data Explorer** — Browse genes, cancer types, and class
               distributions in the training dataset.

            **To get started**, select **Single Prediction** in the sidebar
            and enter a variant to predict.
            """
        )

        st.info(
            "**Tip:** Make sure the FastAPI backend is running at "
            f"`{client.api_url}` before making predictions. "
            "Start it with: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`"
        )

    with right:
        st.subheader("Model Architecture")
        arch_path = Path("results/figures/fig01_model_architecture.png")
        if arch_path.exists():
            st.image(str(arch_path), use_container_width=True)
        else:
            st.markdown(
                """
                ```
                ┌──────────────┐
                │  Genomic     │──┐
                │  Encoder     │  │
                ├──────────────┤  │  ┌──────────────┐
                │  Expression  │──┼──│   Fusion      │──→ Prediction
                │  Encoder     │  │  │  (Attention)  │
                ├──────────────┤  │  └──────────────┘
                │  Methylation │──┘
                │  Encoder     │
                └──────────────┘
                ```
                """
            )

    st.markdown("---")

    st.subheader("Research Abstract")
    st.markdown(
        """
        > Accurate classification of cancer-associated gene mutations as pathogenic
        > or benign is essential for precision oncology. We present a multi-omics
        > deep learning framework that integrates genomic, transcriptomic, and
        > epigenomic data through cross-attention fusion to predict variant
        > pathogenicity. The model provides calibrated uncertainty estimates via
        > MC Dropout and temperature scaling, alongside SHAP-based feature
        > explanations showing which omics modalities drive each prediction.
        > Evaluated on ClinVar-labeled variants with TCGA multi-omics profiles,
        > our approach achieves competitive AUROC with full transparency into
        > the prediction rationale — addressing a key barrier to clinical
        > adoption of ML-based variant classification tools.
        """
    )

    st.markdown("---")
    st.caption(
        "Cancer Mutation Pathogenicity Predictor | "
        "Research Project | IEEE/Springer/Nature Target"
    )
