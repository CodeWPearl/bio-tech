"""Single variant prediction page — the core feature of the dashboard."""

from __future__ import annotations

import json
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import get_class_color, get_confidence_color


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


def _build_pdf_report(response: dict, request_data: dict) -> bytes:
    """Generate a simple PDF report for the prediction result."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        return b""

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements: list = []

    elements.append(Paragraph("Cancer Mutation Pathogenicity Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Variant Information", styles["Heading2"]))
    variant_data = [
        ["Gene Symbol", request_data.get("gene_symbol", "")],
        ["Mutation Type", request_data.get("mutation_type", "")],
        ["Chromosome", request_data.get("chromosome", "")],
        ["Position", str(request_data.get("start_position", ""))],
        ["Ref / Alt", f"{request_data.get('reference_allele', '')} > {request_data.get('variant_allele', '')}"],
        ["Protein Change", request_data.get("protein_change", "N/A") or "N/A"],
    ]
    table = Table(variant_data, colWidths=[2.5 * inch, 4 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.106, 0.228, 0.361)),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("Prediction Result", styles["Heading2"]))
    pred_class = response.get("predicted_class", "Unknown")
    confidence = response.get("confidence", 0)
    recommendation = response.get("recommendation", "")
    result_data = [
        ["Predicted Class", pred_class],
        ["Confidence", f"{confidence * 100:.1f}%"],
        ["Recommendation", recommendation],
    ]
    table2 = Table(result_data, colWidths=[2.5 * inch, 4 * inch])
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.106, 0.228, 0.361)),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table2)
    elements.append(Spacer(1, 0.3 * inch))

    class_probs = response.get("class_probabilities", {})
    if class_probs:
        elements.append(Paragraph("Class Probabilities", styles["Heading2"]))
        prob_data = [[cls, f"{prob * 100:.2f}%"] for cls, prob in class_probs.items()]
        table3 = Table(prob_data, colWidths=[3 * inch, 3.5 * inch])
        table3.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table3)

    doc.build(elements)
    return buffer.getvalue()


def _render_input_form(client: APIClient) -> dict | None:
    """Render the input form and return the request dict when submitted."""
    st.subheader("Variant Input")

    genes = client.get_genes()
    gene_symbols = sorted([g["gene_symbol"] for g in genes]) if genes else []

    gene_symbol = st.text_input(
        "Gene Symbol",
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

    submitted = st.button("Predict Pathogenicity", type="primary", use_container_width=True)

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
        <div class="result-card" style="border-left: 6px solid {color}; text-align:center;">
            <span class="prediction-badge" style="background:{color};">{pred_class.upper()}</span>
            <div style="margin-top:1rem;">
                <span style="font-size:1.4rem;font-weight:700;color:{conf_color};">
                    {confidence * 100:.1f}% confidence
                </span>
            </div>
            <div style="margin-top:0.5rem;">
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
        ))
        fig.update_layout(
            xaxis_title="Probability (%)",
            xaxis_range=[0, 100],
            height=220,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="white",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- c) Uncertainty Panel ---
    if uncertainty:
        with st.expander("Uncertainty Estimation", expanded=True):
            u_cols = st.columns(3)
            epistemic = uncertainty.get("epistemic_uncertainty", 0)
            entropy = uncertainty.get("predictive_entropy", 0)
            calibrated = uncertainty.get("calibrated", False)
            conf_level = uncertainty.get("confidence_level", "Medium")

            with u_cols[0]:
                st.metric("Epistemic Uncertainty", f"{epistemic:.4f}")
                gauge_pct = min(epistemic * 500, 100)
                gauge_color = "#28A745" if gauge_pct < 30 else ("#FFC107" if gauge_pct < 60 else "#DC3545")
                st.markdown(
                    f"""<div style="background:#e9ecef;border-radius:10px;height:10px;width:100%">
                    <div style="background:{gauge_color};border-radius:10px;height:10px;width:{gauge_pct}%"></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with u_cols[1]:
                st.metric("Predictive Entropy", f"{entropy:.4f}")
            with u_cols[2]:
                cal_icon = "calibrated" if calibrated else "uncalibrated"
                cal_color = "#28A745" if calibrated else "#FFC107"
                st.metric("Calibration", cal_icon)
                st.markdown(
                    f'<span style="color:{cal_color};font-weight:600">{conf_level} Confidence</span>',
                    unsafe_allow_html=True,
                )

            if epistemic < 0.05:
                interp = "This prediction has **low uncertainty**, indicating the model is confident in its assessment."
            elif epistemic < 0.15:
                interp = "This prediction has **moderate uncertainty**. The model is reasonably confident but manual review may be warranted."
            else:
                interp = "This prediction has **high uncertainty**. The model is unsure — manual expert review is strongly recommended."
            st.info(interp)

    # --- d) Explanation Panel ---
    if explanation:
        with st.expander("Feature Explanations", expanded=True):
            modality_contrib = explanation.get("modality_contributions", {})
            if modality_contrib:
                st.markdown("**Modality Contributions**")
                labels = list(modality_contrib.keys())
                values = list(modality_contrib.values())
                fig_donut = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker_colors=px.colors.qualitative.Set2[:len(labels)],
                    textinfo="label+percent",
                ))
                fig_donut.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            pos_features = explanation.get("top_positive_features", [])
            neg_features = explanation.get("top_negative_features", [])
            all_features = pos_features + neg_features

            if all_features:
                st.markdown("**Top Contributing Features**")
                feat_names = [f.get("feature_name", "") for f in pos_features[:10]]
                feat_vals = [f.get("importance", 0) for f in pos_features[:10]]
                neg_names = [f.get("feature_name", "") for f in neg_features[:5]]
                neg_vals = [-f.get("importance", 0) for f in neg_features[:5]]

                all_names = feat_names + neg_names
                all_vals = feat_vals + neg_vals
                bar_colors = ["#28A745"] * len(feat_names) + ["#DC3545"] * len(neg_names)

                if all_names:
                    fig_feat = go.Figure(go.Bar(
                        x=all_vals,
                        y=all_names,
                        orientation="h",
                        marker_color=bar_colors,
                    ))
                    fig_feat.update_layout(
                        xaxis_title="Feature Importance",
                        height=max(200, len(all_names) * 30),
                        margin=dict(l=10, r=10, t=10, b=30),
                        plot_bgcolor="white",
                        yaxis=dict(autorange="reversed"),
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
                    colorscale="Blues",
                    text=[[f"{w:.3f}" for w in weights]],
                    texttemplate="%{text}",
                ))
                fig_att.update_layout(
                    height=150,
                    margin=dict(l=10, r=10, t=10, b=30),
                )
                st.plotly_chart(fig_att, use_container_width=True)

    # --- e) Biological Context Panel ---
    if bio_context:
        with st.expander("Biological Context", expanded=False):
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
    with exp_cols[0]:
        pdf_bytes = _build_pdf_report(response, request_data)
        if pdf_bytes:
            st.download_button(
                "Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"pathogenicity_report_{response.get('variant_id', 'variant')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption("PDF export requires `reportlab` package.")
    with exp_cols[1]:
        json_str = json.dumps(response, indent=2)
        st.download_button(
            "Copy as JSON",
            data=json_str,
            file_name=f"prediction_{response.get('variant_id', 'variant')}.json",
            mime="application/json",
            use_container_width=True,
        )


def render(client: APIClient) -> None:
    """Render the Single Variant Prediction page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>Single Variant Prediction</h1>
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
                <div class="result-card" style="text-align:center;padding:3rem;">
                    <p style="font-size:3rem;margin:0">&#129516;</p>
                    <p style="font-size:1.2rem;color:#6C757D;margin:0.5rem 0 0 0">
                        Enter a variant and click <strong>Predict Pathogenicity</strong>
                        to see results here
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
