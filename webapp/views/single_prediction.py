"""Single variant prediction page — the core feature of the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.export import export_to_json
from webapp.utils.report_generator import generate_prediction_report
from webapp.utils.styling import PLOTLY_LIGHT, get_class_color, get_confidence_color

MUTATION_TYPES = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Silent",
]

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]

SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_single_variant.json"


def _load_sample_variant() -> dict | None:
    """Load the sample single variant JSON for the 'Try example' button."""
    if SAMPLE_DATA_PATH.is_file():
        return json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
    return None


def _render_input_form(client: APIClient) -> dict | None:
    """Render the input form and return the request dict when submitted."""
    st.markdown(
        '<div class="glass-card"><h3 style="margin-top:0">'
        '\U0001f4dd Variant Input</h3></div>',
        unsafe_allow_html=True,
    )

    sample = _load_sample_variant()
    if sample:
        if st.button("\U0001f9ea  Try Example (BRAF V600E)", use_container_width=True):
            st.session_state["sample_loaded"] = sample
            st.rerun()

    sample_data = st.session_state.pop("sample_loaded", None)

    genes = client.get_genes()
    gene_symbols = sorted([g["gene_symbol"] for g in genes]) if genes else []

    gene_symbol = st.text_input(
        "Gene Symbol",
        value=sample_data.get("gene_symbol", "") if sample_data else "",
        placeholder="e.g. BRCA1, TP53, BRAF",
        help="Enter a gene symbol. Known cancer driver genes are suggested.",
    )

    if gene_symbols and gene_symbol:
        matches = [g for g in gene_symbols if g.upper().startswith(gene_symbol.upper())]
        if matches and gene_symbol.upper() not in [m.upper() for m in matches]:
            st.caption(f"Suggestions: {', '.join(matches[:8])}")

    mutation_type = st.selectbox("Mutation Type", MUTATION_TYPES)
    chromosome = st.selectbox("Chromosome", CHROMOSOMES)
    start_position = st.number_input(
        "Start Position",
        min_value=1,
        max_value=300_000_000,
        value=43044295,
        step=1,
        help="Genomic position (GRCh38)",
    )

    col_ref, col_alt = st.columns(2)
    with col_ref:
        reference_allele = st.text_input("Reference Allele", value="A", help="Only A/T/C/G")
    with col_alt:
        variant_allele = st.text_input("Variant Allele", value="T", help="Only A/T/C/G")

    protein_change = st.text_input(
        "Protein Change (optional)",
        placeholder="e.g. p.V600E",
    )

    stats = client.get_stats()
    cancer_types = stats.get("cancer_types", []) if stats else []
    cancer_type = st.selectbox(
        "Cancer Type (optional)",
        [""] + cancer_types,
        format_func=lambda x: "— Select —" if x == "" else x,
    )

    st.markdown("---")
    include_explanation = st.checkbox("Include detailed explanation", value=True)
    include_uncertainty = st.checkbox("Include uncertainty estimation", value=True)

    allele_valid = True
    valid_bases = set("ACGTN-")
    for label, val in [("Reference Allele", reference_allele), ("Variant Allele", variant_allele)]:
        if val and not all(c in valid_bases for c in val.upper()):
            st.error(f"{label} must contain only A, T, C, G, N, or -")
            allele_valid = False

    submitted = st.button(
        "\U0001f52c  Predict Pathogenicity",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        if not gene_symbol.strip():
            st.error("Please enter a gene symbol.")
            return None
        if not allele_valid:
            return None

        return {
            "gene_symbol": gene_symbol.strip().upper(),
            "mutation_type": mutation_type,
            "chromosome": chromosome,
            "start_position": start_position,
            "reference_allele": reference_allele.strip().upper(),
            "variant_allele": variant_allele.strip().upper(),
            "protein_change": protein_change.strip() if protein_change.strip() else None,
            "cancer_type": cancer_type if cancer_type else None,
            "include_explanation": include_explanation,
            "include_uncertainty": include_uncertainty,
        }
    return None


def _render_results(response: dict, request_data: dict) -> None:
    """Render the prediction results panel."""
    if not response:
        return

    pred_class = response.get("predicted_class", "Unknown")
    confidence = response.get("confidence", 0.0)
    recommendation = response.get("recommendation", "")
    class_probs = response.get("class_probabilities", {})
    uncertainty = response.get("uncertainty")
    explanation = response.get("explanation")
    bio_context = response.get("biological_context", {})

    # --- a) Prediction Card ---
    color = get_class_color(pred_class)
    conf_color = get_confidence_color(confidence)
    is_high_conf = "High Confidence" in recommendation or confidence >= 0.8

    rec_class = "recommendation-high" if is_high_conf else "recommendation-low"
    rec_icon = "&#10003;" if is_high_conf else "&#9888;"
    rec_text = recommendation if recommendation else ("High Confidence" if is_high_conf else "Manual Review Recommended")

    st.markdown(
        f"""
        <div class="result-card" style="border-left: 5px solid {color}; text-align:center;">
            <span class="prediction-badge" style="background:{color};">
                {pred_class.upper()}
            </span>
            <div style="margin-top:1.2rem;">
                <span style="font-size:2rem;font-weight:800;color:{conf_color};">
                    {confidence * 100:.1f}%
                </span>
                <span style="color:#64748B;font-size:0.9rem;margin-left:6px">confidence</span>
            </div>
            <div style="margin-top:0.3rem;">
                <div style="background:#F1F5F9;border-radius:50px;height:8px;
                     width:80%;margin:0.5rem auto;overflow:hidden">
                    <div style="background:{conf_color};
                         height:100%;width:{confidence * 100}%;border-radius:50px;
                         transition:width 0.5s ease"></div>
                </div>
            </div>
            <div style="margin-top:0.6rem;">
                <span class="recommendation-badge {rec_class}">{rec_icon} {rec_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- b) Class Probabilities ---
    if class_probs:
        st.markdown("#### Class Probabilities")
        prob_names = list(class_probs.keys())
        prob_values = [class_probs[n] * 100 for n in prob_names]
        prob_colors = [get_class_color(n) for n in prob_names]

        fig = go.Figure(go.Bar(
            x=prob_values,
            y=prob_names,
            orientation="h",
            marker_color=prob_colors,
            text=[f"{v:.1f}%" for v in prob_values],
            textposition="auto",
            textfont=dict(color="white", size=12, family="Inter"),
        ))
        fig.update_layout(
            xaxis_title="Probability (%)",
            xaxis_range=[0, 100],
            height=200,
            margin=dict(l=10, r=10, t=10, b=30),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#E2E8F0"),
            **PLOTLY_LIGHT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- c) Uncertainty Panel ---
    if uncertainty:
        with st.expander("\U0001f3af Uncertainty Estimation", expanded=True):
            u_cols = st.columns(3)
            epistemic = uncertainty.get("epistemic_uncertainty", 0)
            entropy = uncertainty.get("predictive_entropy", 0)
            calibrated = uncertainty.get("calibrated", False)
            conf_level = uncertainty.get("confidence_level", "Medium")

            with u_cols[0]:
                st.metric("Epistemic Uncertainty", f"{epistemic:.4f}")
                gauge_pct = min(epistemic * 500, 100)
                gauge_color = "#10B981" if gauge_pct < 30 else ("#F59E0B" if gauge_pct < 60 else "#EF4444")
                st.markdown(
                    f"""<div style="background:#F1F5F9;border-radius:50px;
                    height:8px;width:100%;overflow:hidden">
                    <div style="background:{gauge_color};border-radius:50px;
                    height:100%;width:{gauge_pct}%"></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with u_cols[1]:
                st.metric("Predictive Entropy", f"{entropy:.4f}")
            with u_cols[2]:
                cal_text = "Calibrated" if calibrated else "Uncalibrated"
                st.metric("Calibration", cal_text)
                level_colors = {"High": "#059669", "Medium": "#D97706", "Low": "#DC2626"}
                lc = level_colors.get(conf_level, "#64748B")
                st.markdown(
                    f'<span style="color:{lc};font-weight:700;font-size:0.9rem">'
                    f'{conf_level} Confidence</span>',
                    unsafe_allow_html=True,
                )

            if epistemic < 0.05:
                interp = "This prediction has **low uncertainty** — the model is confident in its assessment."
            elif epistemic < 0.15:
                interp = "This prediction has **moderate uncertainty** — manual review may be warranted."
            else:
                interp = "This prediction has **high uncertainty** — expert review is strongly recommended."
            st.info(interp)

    # --- d) Explanation Panel ---
    if explanation:
        with st.expander("\U0001f4a1 Feature Explanations", expanded=True):
            modality_contrib = explanation.get("modality_contributions", {})
            if modality_contrib:
                st.markdown("**Modality Contributions**")
                labels = list(modality_contrib.keys())
                values = list(modality_contrib.values())
                fig_donut = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker_colors=["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981"][:len(labels)],
                    textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                fig_donut.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=10)),
                    **PLOTLY_LIGHT,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            pos_features = explanation.get("top_positive_features", [])
            neg_features = explanation.get("top_negative_features", [])

            if pos_features or neg_features:
                st.markdown("**Top Contributing Features**")
                feat_names = [f.get("feature_name", "") for f in pos_features[:10]]
                feat_vals = [f.get("importance", 0) for f in pos_features[:10]]
                neg_names = [f.get("feature_name", "") for f in neg_features[:5]]
                neg_vals = [-f.get("importance", 0) for f in neg_features[:5]]

                all_names = feat_names + neg_names
                all_vals = feat_vals + neg_vals
                bar_colors = ["#10B981"] * len(feat_names) + ["#EF4444"] * len(neg_names)

                if all_names:
                    fig_feat = go.Figure(go.Bar(
                        x=all_vals,
                        y=all_names,
                        orientation="h",
                        marker_color=bar_colors,
                        textfont=dict(color="#1E293B"),
                    ))
                    fig_feat.update_layout(
                        xaxis_title="Feature Importance",
                        height=max(180, len(all_names) * 30),
                        margin=dict(l=10, r=10, t=10, b=30),
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(gridcolor="#E2E8F0"),
                        **PLOTLY_LIGHT,
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)

            attention = explanation.get("attention_weights")
            if attention:
                st.markdown("**Cross-Attention Weights**")
                modalities = list(attention.keys())
                weights = [attention[m] for m in modalities]
                fig_att = go.Figure(go.Heatmap(
                    z=[weights],
                    x=modalities,
                    y=["Attention"],
                    colorscale="Purples",
                    text=[[f"{w:.3f}" for w in weights]],
                    texttemplate="%{text}",
                    textfont=dict(size=12, color="white"),
                ))
                fig_att.update_layout(
                    height=120,
                    margin=dict(l=10, r=10, t=10, b=30),
                    **PLOTLY_LIGHT,
                )
                st.plotly_chart(fig_att, use_container_width=True)

    # --- e) Biological Context Panel ---
    if bio_context:
        with st.expander("\U0001f9ec Biological Context", expanded=False):
            gene = bio_context.get("gene_symbol", "")
            is_driver = bio_context.get("is_known_cancer_driver", False)
            cosmic_info = bio_context.get("cosmic_census_info", "")
            clinvar_count = bio_context.get("clinvar_entries", 0)
            var_desc = bio_context.get("variant_type_description", "")

            if is_driver:
                st.success(f"**{gene}** is a known cancer driver gene (COSMIC Census)")
            else:
                st.info(f"**{gene}** is not currently listed in the COSMIC Cancer Gene Census")

            if cosmic_info:
                st.markdown(f"**COSMIC:** {cosmic_info}")
            if var_desc:
                st.markdown(f"**Variant Type:** {var_desc}")

            st.metric("ClinVar Entries for Gene", clinvar_count)

            st.markdown("**External Resources:**")
            link_cols = st.columns(3)
            with link_cols[0]:
                st.markdown(f"[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/?term={gene})")
            with link_cols[1]:
                st.markdown(f"[COSMIC](https://cancer.sanger.ac.uk/cosmic/gene/analysis?ln={gene})")
            with link_cols[2]:
                st.markdown(f"[UniProt](https://www.uniprot.org/uniprotkb?query={gene}+AND+organism_id:9606)")

    # --- f) Export Section ---
    st.markdown("---")
    exp_cols = st.columns(2)
    variant_id = response.get("variant_id", "variant")
    with exp_cols[0]:
        pdf_bytes = generate_prediction_report(response)
        if pdf_bytes:
            st.download_button(
                "\U0001f4e5  Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"pathogenicity_report_{variant_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF export requires `reportlab` package.")
    with exp_cols[1]:
        json_str = export_to_json(response)
        st.download_button(
            "\U0001f4cb  Download as JSON",
            data=json_str,
            file_name=f"prediction_{variant_id}.json",
            mime="application/json",
            use_container_width=True,
        )


def render(client: APIClient) -> None:
    """Render the Single Variant Prediction page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>\U0001f52c Single Variant Prediction</h1>
            <p>Enter variant details to predict pathogenicity with confidence
            scores, uncertainty estimation, and feature explanations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_results = st.columns([2, 3])

    with col_input:
        request_data = _render_input_form(client)

    with col_results:
        if request_data is not None:
            with st.spinner("Running prediction..."):
                response = client.predict(request_data)

            if response:
                st.session_state["last_prediction"] = response
                st.session_state["last_request"] = request_data
                _render_results(response, request_data)
            else:
                st.warning("No response received. Check that the API is running.")
        elif "last_prediction" in st.session_state:
            _render_results(
                st.session_state["last_prediction"],
                st.session_state.get("last_request", {}),
            )
        else:
            st.markdown(
                """
                <div class="result-card" style="text-align:center;padding:4rem 2rem;">
                    <div style="font-size:4rem;margin-bottom:1rem;
                         opacity:0.6;filter:grayscale(30%)">\U0001f9ec</div>
                    <p style="font-size:1.2rem;color:#64748B !important;margin:0">
                        Enter a variant and click <strong style="color:#4F46E5 !important">
                        Predict Pathogenicity</strong> to see results here
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
