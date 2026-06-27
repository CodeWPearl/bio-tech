"""Batch analysis page for processing multiple variants at once."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.styling import get_class_color, styled_metric_card


REQUIRED_COLUMNS = [
    "gene_symbol",
    "mutation_type",
    "chromosome",
    "start_position",
    "reference_allele",
    "variant_allele",
]

SAMPLE_DATA = """gene_symbol,mutation_type,chromosome,start_position,reference_allele,variant_allele,protein_change
BRCA1,Missense_Mutation,17,43044295,A,T,p.C61G
TP53,Nonsense_Mutation,17,7577538,C,T,p.R213*
BRAF,Missense_Mutation,7,140753336,A,T,p.V600E
KRAS,Missense_Mutation,12,25398284,C,A,p.G12V
EGFR,Missense_Mutation,7,55259515,T,G,p.L858R
"""


def _parse_upload(uploaded_file: object) -> pd.DataFrame | None:
    """Parse an uploaded CSV or TSV file into a DataFrame."""
    try:
        name = getattr(uploaded_file, "name", "")
        if name.endswith(".tsv") or name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep="\t")
        else:
            df = pd.read_csv(uploaded_file)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None

        return df
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        return None


def _df_to_requests(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of prediction request dicts."""
    requests = []
    for _, row in df.iterrows():
        req = {
            "gene_symbol": str(row["gene_symbol"]).strip(),
            "mutation_type": str(row["mutation_type"]).strip(),
            "chromosome": str(row["chromosome"]).strip(),
            "start_position": int(row["start_position"]),
            "reference_allele": str(row["reference_allele"]).strip().upper(),
            "variant_allele": str(row["variant_allele"]).strip().upper(),
            "protein_change": str(row.get("protein_change", "")).strip() or None,
            "cancer_type": str(row.get("cancer_type", "")).strip() or None,
            "include_explanation": False,
            "include_uncertainty": True,
        }
        requests.append(req)
    return requests


def _build_results_df(predictions: list[dict]) -> pd.DataFrame:
    """Build a results DataFrame from prediction responses."""
    rows = []
    for pred in predictions:
        row = {
            "variant_id": pred.get("variant_id", ""),
            "predicted_class": pred.get("predicted_class", ""),
            "confidence": pred.get("confidence", 0),
            "recommendation": pred.get("recommendation", ""),
        }
        uncertainty = pred.get("uncertainty")
        if uncertainty:
            row["epistemic_uncertainty"] = uncertainty.get("epistemic_uncertainty", 0)
            row["confidence_level"] = uncertainty.get("confidence_level", "")
        bio = pred.get("biological_context", {})
        row["gene_symbol"] = bio.get("gene_symbol", "")
        row["is_cancer_driver"] = bio.get("is_known_cancer_driver", False)
        rows.append(row)
    return pd.DataFrame(rows)


def render(client: APIClient) -> None:
    """Render the Batch Analysis page."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>Batch Variant Analysis</h1>
            <p>Upload a CSV/TSV file with multiple variants for high-throughput
            pathogenicity screening</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload variant file (CSV or TSV)",
            type=["csv", "tsv", "txt"],
            help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
        )
    with col_sample:
        st.markdown("**Sample Format**")
        st.download_button(
            "Download Sample CSV",
            data=SAMPLE_DATA,
            file_name="sample_variants.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded_file is None:
        st.info(
            "Upload a CSV or TSV file with columns: "
            f"`{', '.join(REQUIRED_COLUMNS)}`. "
            "Optional: `protein_change`, `cancer_type`."
        )
        return

    df = _parse_upload(uploaded_file)
    if df is None:
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Total rows: {len(df)}")

    if len(df) > 100:
        st.warning("Batch limit is 100 variants. Only the first 100 will be processed.")
        df = df.head(100)

    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
        requests = _df_to_requests(df)

        progress_bar = st.progress(0, text="Submitting batch request...")
        progress_bar.progress(10, text="Processing variants...")

        response = client.predict_batch(requests)
        progress_bar.progress(90, text="Building results...")

        if not response or "predictions" not in response:
            progress_bar.empty()
            st.error("Batch prediction failed. Check the API server.")
            return

        predictions = response["predictions"]
        summary = response.get("summary", {})
        progress_bar.progress(100, text="Done!")

        st.session_state["batch_results"] = predictions
        st.session_state["batch_summary"] = summary
        st.session_state["batch_df"] = df

    if "batch_results" not in st.session_state:
        return

    predictions = st.session_state["batch_results"]
    summary = st.session_state["batch_summary"]
    results_df = _build_results_df(predictions)

    st.markdown("---")
    st.subheader("Results Summary")

    s_cols = st.columns(5)
    total = summary.get("total_variants", len(predictions))
    class_counts = summary.get("class_counts", {})
    avg_conf = summary.get("average_confidence", 0)
    pathogenic_count = class_counts.get("Pathogenic", 0) + class_counts.get("Likely Pathogenic", 0)
    benign_count = class_counts.get("Benign", 0) + class_counts.get("Likely Benign", 0)
    low_conf = len([p for p in predictions if p.get("confidence", 0) < 0.6])

    with s_cols[0]:
        st.markdown(styled_metric_card("Total Variants", str(total)), unsafe_allow_html=True)
    with s_cols[1]:
        st.markdown(styled_metric_card("Pathogenic", str(pathogenic_count)), unsafe_allow_html=True)
    with s_cols[2]:
        st.markdown(styled_metric_card("Benign", str(benign_count)), unsafe_allow_html=True)
    with s_cols[3]:
        st.markdown(styled_metric_card("Avg Confidence", f"{avg_conf * 100:.1f}%"), unsafe_allow_html=True)
    with s_cols[4]:
        st.markdown(styled_metric_card("Low Confidence", str(low_conf)), unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Results Table")
    st.dataframe(
        results_df.style.applymap(
            lambda v: f"background-color: {get_class_color(v)}22; color: {get_class_color(v)}"
            if v in get_class_color.__code__.co_consts else "",
            subset=["predicted_class"] if "predicted_class" in results_df.columns else [],
        ),
        use_container_width=True,
        height=400,
    )

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("#### Class Distribution")
        if class_counts:
            fig_pie = go.Figure(go.Pie(
                labels=list(class_counts.keys()),
                values=list(class_counts.values()),
                marker_colors=[get_class_color(c) for c in class_counts],
                hole=0.4,
                textinfo="label+value+percent",
            ))
            fig_pie.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with chart_cols[1]:
        st.markdown("#### Confidence Distribution")
        if "confidence" in results_df.columns:
            fig_hist = px.histogram(
                results_df,
                x="confidence",
                nbins=20,
                color_discrete_sequence=["#1B6EC2"],
                labels={"confidence": "Confidence Score"},
            )
            fig_hist.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=30),
                plot_bgcolor="white",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    if "gene_symbol" in results_df.columns:
        st.markdown("#### Gene-Level Summary")
        gene_summary = results_df.groupby("gene_symbol").agg(
            count=("variant_id", "count"),
            avg_confidence=("confidence", "mean"),
        ).sort_values("count", ascending=False).head(20).reset_index()

        fig_gene = px.bar(
            gene_summary,
            x="gene_symbol",
            y="count",
            color="avg_confidence",
            color_continuous_scale="RdYlGn",
            labels={"gene_symbol": "Gene", "count": "Variant Count", "avg_confidence": "Avg Confidence"},
        )
        fig_gene.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_gene, use_container_width=True)

    st.markdown("---")
    dl_cols = st.columns(2)
    with dl_cols[0]:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            data=csv_data,
            file_name="batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_cols[1]:
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Predictions", index=False)
                if "batch_df" in st.session_state:
                    st.session_state["batch_df"].to_excel(
                        writer, sheet_name="Input Data", index=False,
                    )
                pd.DataFrame([summary]).to_excel(
                    writer, sheet_name="Summary", index=False,
                )
            st.download_button(
                "Download Full Report (Excel)",
                data=excel_buffer.getvalue(),
                file_name="batch_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Excel export requires `openpyxl` package.")
