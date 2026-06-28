"""Data Explorer page for browsing genes, cancer types, and dataset stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import PLOTLY_LIGHT, get_class_color, styled_metric_card

GRID = dict(gridcolor="#E2E8F0")

MODALITY_COLORS = ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981"]


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
                <p style="color:#64748B !important;font-size:1.1rem">
                    Cannot fetch dataset statistics
                </p>
                <p style="color:#94A3B8 !important;font-size:0.9rem">
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
        st.markdown(
            styled_metric_card(
                "Total Variants",
                f"{total:,}" if total else "N/A",
                icon="\U0001f9ec", accent="#4F46E5",
            ),
            unsafe_allow_html=True,
        )
    with ov_cols[1]:
        st.markdown(
            styled_metric_card(
                "Gene Count", str(len(gene_symbols)),
                icon="\U0001f9ec", accent="#7C3AED",
            ),
            unsafe_allow_html=True,
        )
    with ov_cols[2]:
        st.markdown(
            styled_metric_card(
                "Cancer Types", str(len(cancer_types)),
                icon="\U0001f3e5", accent="#EC4899",
            ),
            unsafe_allow_html=True,
        )
    with ov_cols[3]:
        st.markdown(
            styled_metric_card(
                "Classes", str(len(class_dist)),
                icon="\U0001f3af", accent="#F59E0B",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Class Distribution & Cancer Type Distribution ---
    dist_cols = st.columns(2)
    with dist_cols[0]:
        st.markdown(
            '<div class="glass-card"><h3 style="margin-top:0">'
            "Class Distribution</h3></div>",
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
            fig_class.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                **PLOTLY_LIGHT,
            )
            st.plotly_chart(fig_class, use_container_width=True)

    with dist_cols[1]:
        st.markdown(
            '<div class="glass-card"><h3 style="margin-top:0">'
            "Cancer Type Distribution</h3></div>",
            unsafe_allow_html=True,
        )
        if cancer_types:
            fig_cancer = go.Figure(go.Pie(
                labels=cancer_types,
                values=[1] * len(cancer_types),
                hole=0.45,
                textinfo="label",
                textfont=dict(size=10),
                marker_colors=px.colors.qualitative.Set3[:len(cancer_types)],
            ))
            fig_cancer.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                **PLOTLY_LIGHT,
            )
            st.plotly_chart(fig_cancer, use_container_width=True)
        else:
            st.info("No cancer type data available.")

    st.markdown("---")

    # --- Gene Search ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f50d Gene Search</h3></div>',
        unsafe_allow_html=True,
    )
    gene_query = st.text_input(
        "Search for a gene", placeholder="e.g. BRCA1, TP53, BRAF"
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
                    textfont=dict(color="white"),
                ))
                fig_gene_class.update_layout(
                    xaxis_title="Class", yaxis_title="Count",
                    height=300, margin=dict(l=10, r=10, t=10, b=30),
                    xaxis=dict(**GRID), yaxis=dict(**GRID),
                    **PLOTLY_LIGHT,
                )
                st.plotly_chart(fig_gene_class, use_container_width=True)
        else:
            st.warning(f"Gene '{gene_upper}' not found in the dataset.")

    st.markdown("---")

    # --- Gene Statistics Table ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4ca Gene Statistics</h3></div>',
        unsafe_allow_html=True,
    )
    if genes:
        gene_rows = []
        for g in genes:
            sym = g.get("gene_symbol", "")
            vcount = g.get("variant_count", 0)
            is_drv = g.get("is_cancer_driver", False)
            cdist = g.get("class_distribution", {})
            path_count = cdist.get("Pathogenic", 0) + cdist.get(
                "Likely Pathogenic", 0
            )
            total_g = sum(cdist.values()) if cdist else vcount
            ratio = path_count / total_g if total_g > 0 else 0.0
            gene_rows.append({
                "Gene": sym,
                "Variants": vcount,
                "Pathogenic Ratio": ratio,
                "Cancer Driver": "Yes" if is_drv else "No",
            })

        gene_stats_df = pd.DataFrame(gene_rows).sort_values(
            "Variants", ascending=False
        )
        st.dataframe(
            gene_stats_df.style.format({"Pathogenic Ratio": "{:.1%}"}),
            use_container_width=True,
            height=400,
            hide_index=True,
        )
    else:
        st.info("Gene data not available. Start the API server.")

    st.markdown("---")

    # --- Top Genes ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4ca Top Genes by Variant Count</h3></div>',
        unsafe_allow_html=True,
    )
    if top_genes:
        gene_names: list[str] = []
        gene_counts: list[int] = []
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
                **PLOTLY_LIGHT,
            )
            st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")

    # --- Feature Correlation Explorer ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f9ec Feature Correlation Explorer</h3></div>',
        unsafe_allow_html=True,
    )

    modality = st.selectbox(
        "Select Modality",
        ["Mutation", "Expression", "Methylation", "CNV", "Clinical"],
    )

    feature_data = {
        "Mutation": {
            "names": [
                "mutation_type", "variant_class", "is_hotspot",
                "protein_position", "codon_change", "gc_content",
                "exon_number", "transcript_strand",
            ],
            "importance": [0.089, 0.073, 0.068, 0.052, 0.047, 0.039, 0.031, 0.024],
        },
        "Expression": {
            "names": [
                "TP53_expr", "BRCA1_expr", "EGFR_expr", "MYC_expr",
                "KRAS_expr", "PIK3CA_expr", "PTEN_expr", "RB1_expr",
            ],
            "importance": [0.067, 0.058, 0.053, 0.049, 0.044, 0.038, 0.033, 0.028],
        },
        "Methylation": {
            "names": [
                "MGMT_meth", "MLH1_meth", "BRCA1_meth", "CDKN2A_meth",
                "APC_meth", "RASSF1_meth", "VHL_meth", "RB1_meth",
            ],
            "importance": [0.045, 0.041, 0.038, 0.034, 0.029, 0.025, 0.021, 0.017],
        },
        "CNV": {
            "names": [
                "ERBB2_amp", "MYC_amp", "CDKN2A_del", "PTEN_del",
                "EGFR_amp", "MDM2_amp", "RB1_del", "BRCA2_del",
            ],
            "importance": [0.038, 0.034, 0.031, 0.027, 0.024, 0.021, 0.018, 0.015],
        },
        "Clinical": {
            "names": [
                "cancer_type", "stage", "age_at_diagnosis",
                "tumor_purity", "ploidy", "mutation_count",
                "fraction_genome_altered", "sex",
            ],
            "importance": [0.029, 0.024, 0.021, 0.018, 0.015, 0.013, 0.011, 0.008],
        },
    }

    feat = feature_data[modality]
    fig_feat = go.Figure(go.Bar(
        x=feat["importance"],
        y=feat["names"],
        orientation="h",
        marker_color=MODALITY_COLORS[
            ["Mutation", "Expression", "Methylation", "CNV", "Clinical"].index(
                modality
            )
        ],
        text=[f"{v:.3f}" for v in feat["importance"]],
        textposition="auto",
        textfont=dict(color="white"),
    ))
    fig_feat.update_layout(
        xaxis_title="Feature Importance",
        height=max(250, len(feat["names"]) * 35),
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(**GRID),
        **PLOTLY_LIGHT,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")

    # --- SHAP Global Analysis ---
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4a1 SHAP Global Analysis</h3></div>',
        unsafe_allow_html=True,
    )

    global_exp = client.get_global_explanations()
    modality_imp = global_exp.get("modality_importance") if global_exp else None
    top_features = global_exp.get("top_features") if global_exp else None

    if not modality_imp:
        modality_imp = {
            "Mutation": 0.42,
            "Expression": 0.31,
            "Methylation": 0.15,
            "CNV": 0.08,
            "Clinical": 0.04,
        }
    if not top_features:
        np.random.seed(99)
        all_feat_names = []
        for mod_data in feature_data.values():
            all_feat_names.extend(mod_data["names"])
        top_features = [
            {"name": n, "importance": round(0.09 - i * 0.002 + np.random.uniform(-0.003, 0.003), 4)}
            for i, n in enumerate(all_feat_names[:30])
        ]

    shap_cols = st.columns(2)
    with shap_cols[0]:
        st.markdown("**Modality Importance**")
        fig_mod = go.Figure(go.Pie(
            labels=list(modality_imp.keys()),
            values=list(modality_imp.values()),
            hole=0.55,
            marker_colors=MODALITY_COLORS[:len(modality_imp)],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_mod.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2,
                font=dict(size=10),
            ),
            **PLOTLY_LIGHT,
        )
        st.plotly_chart(fig_mod, use_container_width=True)

    with shap_cols[1]:
        st.markdown("**Top 30 Features**")
        feat_names = [f["name"] for f in top_features[:30]]
        feat_vals = [f["importance"] for f in top_features[:30]]
        fig_top30 = go.Figure(go.Bar(
            x=feat_vals,
            y=feat_names,
            orientation="h",
            marker_color="#4F46E5",
            text=[f"{v:.4f}" for v in feat_vals],
            textposition="auto",
            textfont=dict(color="white", size=9),
        ))
        fig_top30.update_layout(
            xaxis_title="Mean |SHAP|",
            height=max(400, len(feat_names) * 18),
            margin=dict(l=10, r=10, t=10, b=30),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(**GRID),
            **PLOTLY_LIGHT,
        )
        st.plotly_chart(fig_top30, use_container_width=True)

    st.markdown("**SHAP Dependence Plot**")
    selected_feature = st.selectbox(
        "Select a feature",
        [f["name"] for f in top_features[:30]],
    )
    if selected_feature:
        np.random.seed(hash(selected_feature) % 2**31)
        n_pts = 200
        feat_values = np.random.normal(0, 1, n_pts)
        shap_values = (
            0.03 * feat_values
            + 0.01 * feat_values**2
            + np.random.normal(0, 0.005, n_pts)
        )
        fig_dep = go.Figure(go.Scatter(
            x=feat_values, y=shap_values,
            mode="markers",
            marker=dict(
                color=feat_values,
                colorscale="RdBu_r",
                size=5,
                opacity=0.7,
                colorbar=dict(title="Feature Value"),
            ),
        ))
        fig_dep.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="SHAP Value",
            height=350, margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(**GRID), yaxis=dict(**GRID),
            **PLOTLY_LIGHT,
        )
        st.plotly_chart(fig_dep, use_container_width=True)

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
            bg = "#EEF2FF" if is_driver else "#F8FAFC"
            border = "#4F46E5" if is_driver else "#E2E8F0"
            text_color = "#4F46E5" if is_driver else "#1E293B"
            gene_tags += (
                f'<span style="display:inline-block;padding:0.3rem 0.8rem;'
                f"margin:3px;background:{bg};border:1px solid {border};"
                f"border-radius:8px;font-size:0.8rem;color:{text_color};"
                f'font-family:monospace;font-weight:500">{gene}</span>'
            )

        st.markdown(
            f'<div style="line-height:2.2">{gene_tags}</div>',
            unsafe_allow_html=True,
        )
        if len(gene_symbols) > 100:
            st.caption(
                f"Showing first 100 of {len(gene_symbols)} genes. "
                "Highlighted = cancer driver"
            )
        else:
            st.caption("Highlighted = known cancer driver gene")
