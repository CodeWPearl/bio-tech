"""Cure Options page — cancer precautions, treatments, and clinical resources."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from webapp.data.cancer_knowledge import (
    CANCER_KNOWLEDGE,
    PATHOGENICITY_GUIDANCE,
    get_cancer_info,
    list_cancer_types,
)
from webapp.utils.api_client import APIClient
from webapp.utils.styling import PLOTLY_LIGHT


_CATEGORY_ICONS: dict[str, str] = {
    "Screening": "\U0001f50d",
    "Lifestyle": "\U0001f3c3",
    "Genetic": "\U0001f9ec",
    "Environmental": "\U0001f30d",
}

_STAGE_COLORS: dict[str, str] = {
    "All stages": "#4F46E5",
    "Stage I-II": "#10B981",
    "Stage I-III": "#10B981",
    "Stage II+": "#F59E0B",
    "Stage III+": "#F97316",
    "Stage III-IV": "#EF4444",
    "Stage IV": "#EF4444",
    "Stage I (high-grade) to IV": "#F97316",
    "Stage II-III (rectal)": "#F59E0B",
    "Stage I (high-risk) to III": "#F59E0B",
    "Clinical trials": "#7C3AED",
    "Advanced TNBC": "#EF4444",
    "Advanced/recurrent": "#EF4444",
    "Recurrent low-grade": "#F59E0B",
    "Low-grade, early stage": "#10B981",
    "MSI-H/dMMR tumors": "#7C3AED",
    "Based on molecular profile": "#7C3AED",
    "HR+ tumors, all stages": "#4F46E5",
}


def _render_disclaimer() -> None:
    """Render the research disclaimer banner."""
    st.markdown(
        """
        <div style="background:#FEF3C7;border:1px solid #FCD34D;
             border-radius:10px;padding:1rem 1.5rem;margin-bottom:1.5rem;
             display:flex;align-items:center;gap:12px">
            <span style="font-size:1.5rem">⚠️</span>
            <div>
                <strong style="color:#92400E !important;font-size:0.95rem">
                Research Tool Disclaimer</strong>
                <p style="color:#92400E !important;font-size:0.85rem;margin:0.3rem 0 0">
                This information is for educational and research purposes only.
                It does not constitute medical advice. Always consult qualified
                healthcare professionals for diagnosis and treatment decisions.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(info: dict) -> None:
    """Render the cancer type overview section."""
    abbr = info["abbreviation"]
    overview = info["overview"]
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem">
                <span style="background:#EEF2FF;color:#4F46E5;padding:0.4rem 0.8rem;
                       border-radius:8px;font-weight:700;font-size:0.85rem">{abbr}</span>
                <h3 style="margin:0;color:#1E293B !important">Overview</h3>
            </div>
            <p style="color:#475569 !important;line-height:1.7;margin:0;font-size:0.95rem">
                {overview}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_precautions(precautions: list[dict]) -> None:
    """Render precaution category cards."""
    st.markdown("### \U0001f6e1️ Precautions & Prevention")

    cols = st.columns(2)
    for idx, prec in enumerate(precautions):
        category = prec["category"]
        detail = prec["detail"]
        icon = _CATEGORY_ICONS.get(category, "\U0001f4cb")
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                     border-radius:12px;padding:1.3rem;margin-bottom:1rem;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06);
                     border-left:4px solid #4F46E5">
                    <div style="display:flex;align-items:center;gap:8px;
                         margin-bottom:0.6rem">
                        <span style="font-size:1.3rem">{icon}</span>
                        <strong style="color:#1E293B !important;
                                font-size:1rem">{category}</strong>
                    </div>
                    <p style="color:#475569 !important;font-size:0.88rem;
                       line-height:1.6;margin:0">{detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_treatments(treatments: list[dict]) -> None:
    """Render treatment option cards with stage badges."""
    st.markdown("### \U0001f489 Treatment Options")

    for tx in treatments:
        name = tx["name"]
        description = tx["description"]
        stage = tx["stage"]
        badge_color = _STAGE_COLORS.get(stage, "#6B7280")

        with st.expander(f"• {name}  —  {stage}", expanded=False):
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;
                     margin-bottom:0.8rem">
                    <span style="background:{badge_color};color:white;
                           padding:0.3rem 0.8rem;border-radius:50px;
                           font-size:0.75rem;font-weight:600">{stage}</span>
                </div>
                <p style="color:#475569 !important;font-size:0.9rem;
                   line-height:1.7;margin:0">{description}</p>
                """,
                unsafe_allow_html=True,
            )


def _render_survival_rates(survival_rates: dict[str, str], cancer_name: str) -> None:
    """Render survival rate bar chart."""
    st.markdown("### \U0001f4ca 5-Year Relative Survival Rates")

    stages = list(survival_rates.keys())
    rates = [int(r.replace("%", "")) for r in survival_rates.values()]

    colors = []
    for r in rates:
        if r >= 80:
            colors.append("#10B981")
        elif r >= 50:
            colors.append("#F59E0B")
        else:
            colors.append("#EF4444")

    fig = go.Figure(go.Bar(
        x=stages,
        y=rates,
        marker_color=colors,
        text=[f"{r}%" for r in rates],
        textposition="outside",
        textfont=dict(size=14, color="#1E293B", family="Inter"),
    ))
    fig.update_layout(
        yaxis_title="5-Year Survival Rate (%)",
        yaxis_range=[0, 110],
        height=320,
        margin=dict(l=10, r=10, t=20, b=30),
        xaxis=dict(gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0"),
        **PLOTLY_LIGHT,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Survival rates are approximate 5-year relative survival rates "
        "based on SEER data. Individual outcomes vary significantly based "
        "on tumor biology, treatment, and patient factors."
    )


def _render_key_genes(key_genes: list[str]) -> None:
    """Render key associated genes as clickable badges."""
    st.markdown("### \U0001f9ec Key Associated Genes")

    gene_badges = ""
    for gene in key_genes:
        gene_badges += (
            f'<span style="display:inline-block;background:#EEF2FF;'
            f"color:#4F46E5;padding:0.4rem 1rem;border-radius:50px;"
            f"font-weight:600;font-size:0.85rem;margin:0.3rem 0.4rem "
            f'0.3rem 0;border:1px solid #C7D2FE">{gene}</span>'
        )

    st.markdown(
        f'<div style="margin-bottom:1rem">{gene_badges}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These genes are frequently mutated in this cancer type. "
        "Use the Data Explorer page to browse gene-specific information."
    )


def _render_resources(info: dict, cancer_name: str) -> None:
    """Render external resource links."""
    st.markdown("### \U0001f517 Clinical Resources")

    res_cols = st.columns(3)
    with res_cols[0]:
        st.markdown(
            f"""
            <a href="{info['clinical_trials_url']}" target="_blank"
               style="text-decoration:none">
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                     border-radius:10px;padding:1rem;text-align:center;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06);
                     transition:all 0.2s">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem">\U0001f52c</div>
                    <strong style="color:#4F46E5 !important;font-size:0.85rem">
                    Clinical Trials</strong>
                    <p style="color:#64748B !important;font-size:0.75rem;margin:0.3rem 0 0">
                    ClinicalTrials.gov</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )
    with res_cols[1]:
        st.markdown(
            f"""
            <a href="{info['nccn_url']}" target="_blank"
               style="text-decoration:none">
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                     border-radius:10px;padding:1rem;text-align:center;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem">\U0001f4d6</div>
                    <strong style="color:#4F46E5 !important;font-size:0.85rem">
                    NCCN Guidelines</strong>
                    <p style="color:#64748B !important;font-size:0.75rem;margin:0.3rem 0 0">
                    Patient-friendly version</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )
    with res_cols[2]:
        nci_query = cancer_name.replace(" ", "+")
        st.markdown(
            f"""
            <a href="https://www.cancer.gov/search/results?swKeyword={nci_query}"
               target="_blank" style="text-decoration:none">
                <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                     border-radius:10px;padding:1rem;text-align:center;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06)">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem">\U0001f3e5</div>
                    <strong style="color:#4F46E5 !important;font-size:0.85rem">
                    NCI Information</strong>
                    <p style="color:#64748B !important;font-size:0.75rem;margin:0.3rem 0 0">
                    National Cancer Institute</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )


def _render_pathogenicity_guidance() -> None:
    """Render the pathogenicity guidance reference section."""
    st.markdown("---")
    st.markdown("### \U0001f4cb Pathogenicity Classification Guidance")

    for cls, guidance in PATHOGENICITY_GUIDANCE.items():
        severity = guidance["severity"]
        sev_color = guidance["severity_color"]
        rec = guidance["recommendation"]
        follow_ups = guidance["follow_up"]

        follow_up_html = "".join(
            f'<li style="color:#475569 !important;font-size:0.85rem;'
            f'margin-bottom:0.3rem">{item}</li>'
            for item in follow_ups
        )

        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                 border-radius:12px;padding:1.3rem;margin-bottom:0.8rem;
                 box-shadow:0 1px 3px rgba(0,0,0,0.06);
                 border-left:4px solid {sev_color}">
                <div style="display:flex;align-items:center;gap:10px;
                     margin-bottom:0.6rem">
                    <strong style="color:#1E293B !important;font-size:1rem">
                    {cls}</strong>
                    <span style="background:{sev_color};color:white;
                           padding:0.2rem 0.6rem;border-radius:50px;
                           font-size:0.7rem;font-weight:600">
                    Severity: {severity}</span>
                </div>
                <p style="color:#475569 !important;font-size:0.88rem;
                   line-height:1.6;margin:0 0 0.6rem 0">{rec}</p>
                <p style="color:#1E293B !important;font-size:0.85rem;
                   font-weight:600;margin:0 0 0.3rem 0">
                Recommended Follow-up:</p>
                <ul style="margin:0;padding-left:1.2rem">{follow_up_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render(client: APIClient) -> None:
    """Render the Cure Options page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>\U0001f3e5 Cancer Precautions & Cure Options</h1>
            <p>Evidence-based precautions, treatment options, survival
            statistics, and clinical resources for TCGA cancer types</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_disclaimer()

    cancer_types = list_cancer_types()
    selected_cancer = st.selectbox(
        "Select Cancer Type",
        cancer_types,
        help="Choose a cancer type to view detailed precautions and treatment information.",
    )

    info = get_cancer_info(selected_cancer)
    if info is None:
        st.error("Cancer type information not found.")
        return

    _render_overview(info)

    tab_precautions, tab_treatments, tab_survival, tab_resources = st.tabs(
        [
            "\U0001f6e1️ Precautions",
            "\U0001f489 Treatments",
            "\U0001f4ca Survival Rates",
            "\U0001f517 Resources",
        ]
    )

    with tab_precautions:
        _render_precautions(info["precautions"])

    with tab_treatments:
        _render_treatments(info["treatment_options"])

    with tab_survival:
        _render_survival_rates(info["survival_rates"], selected_cancer)

    with tab_resources:
        _render_key_genes(info["key_genes"])
        _render_resources(info, selected_cancer)

    _render_pathogenicity_guidance()

    _render_disclaimer()
