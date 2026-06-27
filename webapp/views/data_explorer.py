"""Data Explorer page for browsing genes, cancer types, and dataset stats."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import get_class_color, styled_metric_card


def render(client: APIClient) -> None:
    """Render the Data Explorer page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>Data Explorer</h1>
            <p>Browse genes, cancer types, and class distributions in the
            training dataset</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = client.get_stats()
    if not stats:
        st.warning("Cannot fetch dataset statistics. Ensure the API is running.")
        return

    # --- Dataset Overview ---
    st.subheader("Dataset Overview")
    total = stats.get("total_variants", 0)
    class_dist = stats.get("class_distribution", {})
    gene_count = len(stats.get("top_genes", []))
    cancer_types = stats.get("cancer_types", [])

    ov_cols = st.columns(4)
    with ov_cols[0]:
        st.markdown(
            styled_metric_card("Total Variants", f"{total:,}" if total else "N/A"),
            unsafe_allow_html=True,
        )
    with ov_cols[1]:
        st.markdown(
            styled_metric_card("Gene Count", str(gene_count)),
            unsafe_allow_html=True,
        )
    with ov_cols[2]:
        st.markdown(
            styled_metric_card("Cancer Types", str(len(cancer_types))),
            unsafe_allow_html=True,
        )
    with ov_cols[3]:
        st.markdown(
            styled_metric_card("Classes", str(len(class_dist))),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Class Distribution ---
    dist_cols = st.columns(2)
    with dist_cols[0]:
        st.subheader("Class Distribution")
        if class_dist:
            fig_class = go.Figure(go.Pie(
                labels=list(class_dist.keys()),
                values=list(class_dist.values()),
                marker_colors=[get_class_color(c) for c in class_dist],
                hole=0.4,
                textinfo="label+value+percent",
            ))
            fig_class.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_class, use_container_width=True)
        else:
            st.info("No class distribution data available.")

    with dist_cols[1]:
        st.subheader("Cancer Types")
        if cancer_types:
            for ct in cancer_types:
                st.markdown(f"- {ct}")
        else:
            st.info("No cancer type data available.")

    st.markdown("---")

    # --- Gene Search ---
    st.subheader("Gene Search")
    genes = client.get_genes()
    gene_symbols = sorted([g["gene_symbol"] for g in genes]) if genes else []

    gene_query = st.text_input(
        "Search for a gene",
        placeholder="e.g. BRCA1, TP53, BRAF",
    )

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
                ))
                fig_gene_class.update_layout(
                    xaxis_title="Class",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=30),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_gene_class, use_container_width=True)
        else:
            st.warning(f"Gene '{gene_upper}' not found in the dataset.")

    st.markdown("---")

    # --- Top Genes ---
    st.subheader("Top Genes by Variant Count")
    top_genes = stats.get("top_genes", [])
    if top_genes:
        gene_names = []
        gene_counts = []
        for entry in top_genes[:20]:
            for name, count in entry.items():
                gene_names.append(name)
                gene_counts.append(count)

        if gene_names:
            fig_top = go.Figure(go.Bar(
                x=gene_names,
                y=gene_counts,
                marker_color="#1B6EC2",
                text=gene_counts,
                textposition="auto",
            ))
            fig_top.update_layout(
                xaxis_title="Gene",
                yaxis_title="Variant Count",
                height=400,
                margin=dict(l=10, r=10, t=10, b=60),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No gene data available.")

    # --- Gene Browser ---
    if gene_symbols:
        st.markdown("---")
        st.subheader("Gene Browser")
        st.markdown(f"**{len(gene_symbols)} genes** in the dataset:")
        display_genes = gene_symbols[:100]

        gene_cols = st.columns(5)
        for idx, gene in enumerate(display_genes):
            is_driver = any(
                g.get("gene_symbol") == gene and g.get("is_cancer_driver", False)
                for g in genes
            )
            marker = " *" if is_driver else ""
            with gene_cols[idx % 5]:
                st.markdown(f"`{gene}`{marker}")

        if len(gene_symbols) > 100:
            st.caption(f"Showing first 100 of {len(gene_symbols)} genes. * = cancer driver")
        else:
            st.caption("* = known cancer driver gene")
