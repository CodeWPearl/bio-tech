"""About page with project information and documentation."""

from __future__ import annotations

from pathlib import Path

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

    # --- Project Overview ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">Project Overview</h3>
            <p style="line-height:1.8">
            This project implements a <strong style="color:#A5B4FC !important">
            research-grade deep learning framework</strong> for predicting the
            pathogenicity of cancer-associated gene mutations. It integrates
            multiple omics data types (genomic, transcriptomic, epigenomic)
            through a cross-attention fusion mechanism to classify variants as:
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0">
                <span style="padding:0.4rem 1rem;background:rgba(239,68,68,0.15);
                       border:1px solid rgba(239,68,68,0.3);border-radius:50px;
                       color:#F87171;font-weight:600;font-size:0.85rem">
                       Pathogenic</span>
                <span style="padding:0.4rem 1rem;background:rgba(249,115,22,0.15);
                       border:1px solid rgba(249,115,22,0.3);border-radius:50px;
                       color:#FB923C;font-weight:600;font-size:0.85rem">
                       Likely Pathogenic</span>
                <span style="padding:0.4rem 1rem;background:rgba(16,185,129,0.15);
                       border:1px solid rgba(16,185,129,0.3);border-radius:50px;
                       color:#34D399;font-weight:600;font-size:0.85rem">
                       Benign</span>
                <span style="padding:0.4rem 1rem;background:rgba(52,211,153,0.15);
                       border:1px solid rgba(52,211,153,0.3);border-radius:50px;
                       color:#6EE7B7;font-weight:600;font-size:0.85rem">
                       Likely Benign</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Research Motivation ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">\U0001f52c Research Motivation</h3>
            <p style="line-height:1.8">
            Over half of all variants in ClinVar are classified as
            <strong style="color:#A5B4FC !important">Variants of Uncertain
            Significance (VUS)</strong>, leaving clinicians without actionable
            guidance. Existing computational tools (SIFT, PolyPhen-2, CADD,
            REVEL) rely primarily on sequence-level features and lack
            uncertainty quantification. Our approach addresses three critical
            gaps:
            </p>
            <ol style="line-height:2;color:#CBD5E1 !important">
                <li><strong style="color:#A5B4FC !important">Multi-modal
                integration</strong> — combining genomic, transcriptomic,
                and epigenomic data through cross-attention fusion</li>
                <li><strong style="color:#A5B4FC !important">Calibrated
                uncertainty</strong> — MC Dropout and temperature scaling
                to flag unreliable predictions for expert review</li>
                <li><strong style="color:#A5B4FC !important">Transparent
                explanations</strong> — SHAP values and attention weights
                showing which features and modalities drive each
                prediction</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Architecture Diagram ---
    arch_cols = st.columns([3, 2])
    with arch_cols[0]:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">\U0001f3d7 Architecture</h3>
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
                <div style="background:rgba(30,27,75,0.5);border:1px solid
                     rgba(99,102,241,0.15);border-radius:16px;padding:2rem;
                     text-align:center">
                <pre style="color:#A5B4FC;font-size:0.75rem;margin:0;
                     line-height:1.6">
  ┌──────────────┐
  │  Mutation     │──┐
  │  Encoder (42) │  │
  ├──────────────┤  │
  │  Expression   │  │  ┌──────────────────┐    ┌────────────┐
  │  Encoder(2000)│──┼──┤  Cross-Attention  │───→│ Classifier │──→ 4 Classes
  ├──────────────┤  │  │  Fusion           │    │ + Softmax  │
  │  Methylation  │──┤  └──────────────────┘    └────────────┘
  │  Encoder(2000)│  │         ▲
  ├──────────────┤  │         │
  │  CNV          │──┘    MC Dropout
  │  Encoder (200)│       Uncertainty
  └──────────────┘
                </pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with arch_cols[1]:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">\U0001f4be Data Sources</h3>
                <table style="width:100%;border-collapse:separate;
                       border-spacing:0 8px">
                    <tr>
                        <td style="color:#A5B4FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(99,102,241,0.1);
                            border-radius:8px 0 0 8px">
                            <a href="https://www.ncbi.nlm.nih.gov/clinvar/"
                               style="color:#A5B4FC !important">ClinVar</a>
                        </td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            Pathogenicity labels (gold standard)</td>
                    </tr>
                    <tr>
                        <td style="color:#A5B4FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(99,102,241,0.1);
                            border-radius:8px 0 0 8px">
                            <a href="https://www.cbioportal.org/"
                               style="color:#A5B4FC !important">
                               cBioPortal / TCGA</a>
                        </td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            Multi-omics profiles</td>
                    </tr>
                    <tr>
                        <td style="color:#A5B4FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(99,102,241,0.1);
                            border-radius:8px 0 0 8px">
                            <a href="https://cancer.sanger.ac.uk/census"
                               style="color:#A5B4FC !important">
                               COSMIC Census</a>
                        </td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            Cancer driver validation</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0">Model Components</h3>
                <table style="width:100%;border-collapse:separate;
                       border-spacing:0 8px">
                    <tr>
                        <td style="color:#C084FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(139,92,246,0.1);
                            border-radius:8px 0 0 8px">Encoders</td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            Per-modality encoders</td>
                    </tr>
                    <tr>
                        <td style="color:#C084FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(139,92,246,0.1);
                            border-radius:8px 0 0 8px">Fusion</td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            Cross-attention mechanism</td>
                    </tr>
                    <tr>
                        <td style="color:#C084FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(139,92,246,0.1);
                            border-radius:8px 0 0 8px">Uncertainty</td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            MC Dropout + Temp. Scaling</td>
                    </tr>
                    <tr>
                        <td style="color:#C084FC !important;font-weight:600;
                            padding:6px 12px;background:rgba(139,92,246,0.1);
                            border-radius:8px 0 0 8px">Explainability</td>
                        <td style="padding:6px 12px;
                            background:rgba(255,255,255,0.03);
                            border-radius:0 8px 8px 0">
                            SHAP + Attention weights</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Technology Stack ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f6e0 Technology Stack</h3></div>',
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
                <div style="background:rgba(30,27,75,0.5);
                     border:1px solid {color}33;border-radius:16px;
                     padding:1.2rem;text-align:center;
                     border-top:3px solid {color}">
                    <p style="font-weight:700;margin:0 0 0.5rem;
                       color:#F1F5F9 !important;font-size:0.85rem">
                       {title}</p>
                    <pre style="color:#94A3B8;font-size:0.75rem;margin:0;
                         white-space:pre-wrap">{content}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- Team / Author ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">\U0001f465 Team</h3>
            <div style="display:flex;gap:24px;flex-wrap:wrap;margin:1rem 0">
                <div style="text-align:center;min-width:120px">
                    <div style="width:60px;height:60px;margin:0 auto 8px;
                         background:linear-gradient(135deg,#6366F1,#8B5CF6);
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:1.5rem">
                         \U0001f9d1‍\U0001f4bb</div>
                    <p style="margin:0;font-weight:600;color:#F1F5F9 !important;
                       font-size:0.85rem">Lead Researcher</p>
                    <p style="margin:0;color:#94A3B8 !important;
                       font-size:0.75rem">ML / Bioinformatics</p>
                </div>
                <div style="text-align:center;min-width:120px">
                    <div style="width:60px;height:60px;margin:0 auto 8px;
                         background:linear-gradient(135deg,#EC4899,#F97316);
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:1.5rem">
                         \U0001f9ec</div>
                    <p style="margin:0;font-weight:600;color:#F1F5F9 !important;
                       font-size:0.85rem">Domain Expert</p>
                    <p style="margin:0;color:#94A3B8 !important;
                       font-size:0.75rem">Genomics / Oncology</p>
                </div>
            </div>
            <p style="color:#64748B !important;font-size:0.8rem;
               font-style:italic">
               Author details will be added upon paper submission.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Citation ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4dd Citation</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown("If you use this work, please cite:")
    st.code(
        "@article{cancer_mutation_pathogenicity_2026,\n"
        "  title   = {Multi-Omics Deep Learning Framework with Cross-Attention\n"
        "             Fusion for Predicting Pathogenicity of Cancer-Associated\n"
        "             Gene Mutations},\n"
        "  author  = {Author, A. and Author, B.},\n"
        "  journal = {Under Review},\n"
        "  year    = {2026},\n"
        "  note    = {Target: IEEE/Springer/Nature journal}\n"
        "}",
        language="bibtex",
    )

    st.markdown("---")

    # --- Links ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f517 Links</h3></div>',
        unsafe_allow_html=True,
    )
    link_cols = st.columns(3)
    with link_cols[0]:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem;
                 background:rgba(99,102,241,0.08);border-radius:12px;
                 border:1px solid rgba(99,102,241,0.15)">
                <p style="font-size:1.5rem;margin:0">\U0001f4c4</p>
                <p style="font-weight:600;margin:0.3rem 0 0;
                   color:#F1F5F9 !important;font-size:0.85rem">
                   Paper PDF</p>
                <p style="color:#64748B !important;font-size:0.75rem;
                   margin:0">Available upon publication</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with link_cols[1]:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem;
                 background:rgba(99,102,241,0.08);border-radius:12px;
                 border:1px solid rgba(99,102,241,0.15)">
                <p style="font-size:1.5rem;margin:0">\U0001f4bb</p>
                <p style="font-weight:600;margin:0.3rem 0 0;
                   color:#F1F5F9 !important;font-size:0.85rem">
                   GitHub Repository</p>
                <p style="color:#64748B !important;font-size:0.75rem;
                   margin:0">Will be public after acceptance</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with link_cols[2]:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem;
                 background:rgba(99,102,241,0.08);border-radius:12px;
                 border:1px solid rgba(99,102,241,0.15)">
                <p style="font-size:1.5rem;margin:0">\U0001f4ca</p>
                <p style="font-weight:600;margin:0.3rem 0 0;
                   color:#F1F5F9 !important;font-size:0.85rem">
                   API Docs (Swagger)</p>
                <p style="color:#64748B !important;font-size:0.75rem;
                   margin:0">localhost:8001/docs</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- License ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">\U0001f4dc License</h3>
            <p style="line-height:1.8">
            This project is licensed under the
            <strong style="color:#A5B4FC !important">MIT License</strong>.
            You are free to use, modify, and distribute this software for
            academic and commercial purposes, provided the copyright notice
            and license terms are included.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Acknowledgments ---
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0">\U0001f64f Acknowledgments</h3>
            <ul style="line-height:2;color:#CBD5E1 !important">
                <li><strong style="color:#A5B4FC !important">NCBI ClinVar</strong>
                    — for providing the gold-standard pathogenicity labels</li>
                <li><strong style="color:#A5B4FC !important">cBioPortal &amp;
                    TCGA</strong> — for multi-omics cancer data</li>
                <li><strong style="color:#A5B4FC !important">COSMIC Cancer Gene
                    Census</strong> — for curated cancer driver gene lists</li>
                <li><strong style="color:#A5B4FC !important">PyTorch &amp;
                    PyTorch Lightning</strong> — deep learning framework</li>
                <li><strong style="color:#A5B4FC !important">SHAP</strong>
                    — for model explainability methods</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- API Status ---
    health = client.health_check()
    if health:
        st.markdown("---")
        status_cols = st.columns(3)
        status = health.get("status", "unknown")
        with status_cols[0]:
            color = "#10B981" if status == "healthy" else "#EF4444"
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:10px">
                <div style="width:12px;height:12px;background:{color};
                     border-radius:50%;box-shadow:0 0 10px {color}"></div>
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
