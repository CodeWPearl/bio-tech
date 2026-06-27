"""Data Explorer page for browsing genes, cancer types, and dataset stats."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import get_class_color, styled_metric_card

PLOTLY_DARK = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#CBD5E1",
    font_family="Inter, sans-serif",
)
GRID = dict(gridcolor="rgba(99,102,241,0.08)")


def render(client: APIClient) -> None:
    """Render the Data Explorer page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>\U0001f9ea Data Explorer</h1>
            <p>Browse genes, cancer types, and class distributions in the
            training dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = client.get_stats()
    if not stats:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;padding:3rem">
                <div style="font-size:3rem;opacity:0.5;margin-bottom:1rem">\U0001f50d</div>
                <p style="color:#94A3B8 !important;font-size:1.1rem">
                    Cannot fetch dataset statistics
                </p>
                <p style="color:#64748B !important;font-size:0.9rem">
                    Start the API server to explore data
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total = stats.get("total_variants", 0)
    class_dist = stats.get("class_distribution", {})
    cancer_types = stats.get("cancer_types", [])
    top_genes = stats.get("top_genes", [])
    genes = client.get_genes()
    gene_symbols = sorted([g["gene_symbol"] for g in genes]) if genes else []

    # --- Overview ---
    ov_cols = st.columns(4)
    with ov_cols[0]:
        st.markdown(styled_metric_card("Total Variants", f"{total:,}" if total else "N/A", icon="\U0001f9ec", accent="#6366F1"), unsafe_allow_html=True)
    with ov_cols[1]:
        st.markdown(styled_metric_card("Gene Count", str(len(gene_symbols)), icon="\U0001f9ec", accent="#8B5CF6"), unsafe_allow_html=True)
    with ov_cols[2]:
        st.markdown(styled_metric_card("Cancer Types", str(len(cancer_types)), icon="\U0001f3e5", accent="#EC4899"), unsafe_allow_html=True)
    with ov_cols[3]:
        st.markdown(styled_metric_card("Classes", str(len(class_dist)), icon="\U0001f3af", accent="#F59E0B"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Class Distribution & Cancer Types ---
    dist_cols = st.columns(2)
    with dist_cols[0]:
        st.markdown(
            '<div class="glass-card"><h3 style="margin-top:0">Class Distribution</h3></div>',
            unsafe_allow_html=True,
        )
        if class_dist:
            fig_class = go.Figure(go.Pie(
                labels=list(class_dist.keys()),
                values=list(class_dist.values()),
                marker_colors=[get_class_color(c) for c in class_dist],
                hole=0.5,
                textinfo="label+value+percent",
                textfont=dict(size=11),
            ))
            fig_class.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_DARK)
            st.plotly_chart(fig_class, use_container_width=True)

    with dist_cols[1]:
        st.markdown(
            '<div class="glass-card"><h3 style="margin-top:0">Cancer Types</h3></div>',
            unsafe_allow_html=True,
        )
        if cancer_types:
            for ct in cancer_types:
                st.markdown(
                    f'<div style="padding:0.5rem 1rem;margin:0.3rem 0;'
                    f'background:rgba(99,102,241,0.08);border-radius:10px;'
                    f'border-left:3px solid #6366F1">'
                    f'<span style="color:#E2E8F0 !important">{ct}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No cancer type data available.")

    st.markdown("---")

    # --- Gene Search ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f50d Gene Search</h3></div>',
        unsafe_allow_html=True,
    )
    gene_query = st.text_input("Search for a gene", placeholder="e.g. BRCA1, TP53, BRAF")

    if gene_query:
        gene_upper = gene_query.strip().upper()
        gene_info = client.get_gene_info(gene_upper)

        if gene_info:
            gi_cols = st.columns(3)
            with gi_cols[0]:
                st.metric("Gene Symbol", gene_info.get("gene_symbol", ""))
            with gi_cols[1]:
                st.metric("Variant Count", gene_info.get("variant_count", 0))
            with gi_cols[2]:
                is_driver = gene_info.get("is_known_cancer_driver", False)
                st.metric("Cancer Driver", "Yes" if is_driver else "No")

            cosmic = gene_info.get("cosmic_census_info")
            if cosmic:
                st.success(cosmic)

            gene_class_dist = gene_info.get("class_distribution", {})
            if gene_class_dist and any(v > 0 for v in gene_class_dist.values()):
                fig_gene_class = go.Figure(go.Bar(
                    x=list(gene_class_dist.keys()),
                    y=list(gene_class_dist.values()),
                    marker_color=[get_class_color(c) for c in gene_class_dist],
                    text=list(gene_class_dist.values()),
                    textposition="auto",
                    textfont=dict(color="white"),
                ))
                fig_gene_class.update_layout(
                    xaxis_title="Class", yaxis_title="Count",
                    height=300, margin=dict(l=10, r=10, t=10, b=30),
                    xaxis=dict(**GRID), yaxis=dict(**GRID),
                    **PLOTLY_DARK,
                )
                st.plotly_chart(fig_gene_class, use_container_width=True)
        else:
            st.warning(f"Gene '{gene_upper}' not found in the dataset.")

    st.markdown("---")

    # --- Top Genes ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4ca Top Genes by Variant Count</h3></div>',
        unsafe_allow_html=True,
    )
    if top_genes:
        gene_names = []
        gene_counts = []
        for entry in top_genes[:20]:
            for name, count in entry.items():
                gene_names.append(name)
                gene_counts.append(count)

        if gene_names:
            fig_top = go.Figure(go.Bar(
                x=gene_names, y=gene_counts,
                marker=dict(
                    color=gene_counts,
                    colorscale="Purples",
                    line=dict(width=0),
                ),
                text=gene_counts,
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig_top.update_layout(
                xaxis_title="Gene", yaxis_title="Variant Count",
                height=400, margin=dict(l=10, r=10, t=10, b=60),
                xaxis=dict(**GRID), yaxis=dict(**GRID),
                **PLOTLY_DARK,
            )
            st.plotly_chart(fig_top, use_container_width=True)

    # --- Gene Browser ---
    if gene_symbols:
        st.markdown("---")
        st.markdown(
            '<div class="glass-card"><h3 style="margin-top:0">'
            f'\U0001f9ec Gene Browser ({len(gene_symbols)} genes)</h3></div>',
            unsafe_allow_html=True,
        )
        display_genes = gene_symbols[:100]
        gene_tags = ""
        for gene in display_genes:
            is_driver = any(
                g.get("gene_symbol") == gene and g.get("is_cancer_driver", False)
                for g in genes
            )
            bg = "rgba(99,102,241,0.15)" if is_driver else "rgba(255,255,255,0.05)"
            border = "#6366F1" if is_driver else "rgba(255,255,255,0.1)"
            gene_tags += (
                f'<span style="display:inline-block;padding:0.3rem 0.8rem;margin:3px;'
                f'background:{bg};border:1px solid {border};border-radius:8px;'
                f'font-size:0.8rem;color:#E2E8F0;font-family:monospace">{gene}</span>'
            )

        st.markdown(
            f'<div style="line-height:2.2">{gene_tags}</div>',
            unsafe_allow_html=True,
        )
        if len(gene_symbols) > 100:
            st.caption(f"Showing first 100 of {len(gene_symbols)} genes. Highlighted = cancer driver")
        else:
            st.caption("Highlighted = known cancer driver gene")
