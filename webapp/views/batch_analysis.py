"""Batch analysis page for processing multiple variants at once."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from webapp.utils.api_client import APIClient
from webapp.utils.export import export_to_csv, export_to_excel
from webapp.utils.styling import PLOTLY_LIGHT, get_class_color, styled_metric_card

GRID = dict(gridcolor="#E2E8F0")

REQUIRED_COLUMNS = [
    "gene_symbol",
    "mutation_type",
    "chromosome",
    "start_position",
    "reference_allele",
    "variant_allele",
]

SAMPLE_CSV_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "sample_batch.csv"


def _load_sample_csv() -> str:
    """Load sample batch CSV from file, fallback to inline data."""
    if SAMPLE_CSV_PATH.is_file():
        return SAMPLE_CSV_PATH.read_text(encoding="utf-8")
    return (
        "gene_symbol,mutation_type,chromosome,start_position,"
        "reference_allele,variant_allele,protein_change\n"
        "BRCA1,Missense_Mutation,17,43044295,A,T,p.C61G\n"
        "TP53,Nonsense_Mutation,17,7577538,C,T,p.R213*\n"
        "BRAF,Missense_Mutation,7,140753336,A,T,p.V600E\n"
        "KRAS,Missense_Mutation,12,25398284,C,A,p.G12V\n"
        "EGFR,Missense_Mutation,7,55259515,T,G,p.L858R\n"
    )


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
            <h1>\U0001f4ca Batch Variant Analysis</h1>
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
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "\U0001f4e5  Sample CSV",
            data=_load_sample_csv(),
            file_name="sample_variants.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded_file is None:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;padding:3rem">
                <div style="font-size:3rem;opacity:0.5;margin-bottom:1rem">\U0001f4c1</div>
                <p style="color:#64748B !important;font-size:1rem">
                    Upload a CSV or TSV file with columns:<br>
                    <code style="color:#4F46E5">{', '.join(REQUIRED_COLUMNS)}</code>
                </p>
                <p style="color:#94A3B8 !important;font-size:0.85rem">
                    Optional: <code style="color:#4F46E5">protein_change</code>,
                    <code style="color:#4F46E5">cancer_type</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
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

    if st.button("\U0001f680  Run Batch Analysis", type="primary", use_container_width=True):
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
        st.markdown(styled_metric_card("Total", str(total), icon="\U0001f4cb", accent="#4F46E5"), unsafe_allow_html=True)
    with s_cols[1]:
        st.markdown(styled_metric_card("Pathogenic", str(pathogenic_count), icon="⚠️", accent="#EF4444"), unsafe_allow_html=True)
    with s_cols[2]:
        st.markdown(styled_metric_card("Benign", str(benign_count), icon="✅", accent="#10B981"), unsafe_allow_html=True)
    with s_cols[3]:
        st.markdown(styled_metric_card("Avg Conf", f"{avg_conf * 100:.1f}%", icon="\U0001f3af", accent="#7C3AED"), unsafe_allow_html=True)
    with s_cols[4]:
        st.markdown(styled_metric_card("Low Conf", str(low_conf), icon="⚡", accent="#F59E0B"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Results Table")

    input_df = st.session_state.get("batch_df")
    if input_df is not None:
        display_df = pd.concat(
            [
                input_df.reset_index(drop=True),
                results_df[["predicted_class", "confidence"]].reset_index(drop=True),
            ],
            axis=1,
        )
        if "epistemic_uncertainty" in results_df.columns:
            display_df["uncertainty"] = results_df["epistemic_uncertainty"].values
    else:
        display_df = results_df

    _CLASS_BG = {
        "Pathogenic": "background-color: rgba(239,68,68,0.1)",
        "Likely Pathogenic": "background-color: rgba(249,115,22,0.1)",
        "Benign": "background-color: rgba(16,185,129,0.1)",
        "Likely Benign": "background-color: rgba(52,211,153,0.1)",
    }

    def _highlight_class(row: pd.Series) -> list[str]:
        style = _CLASS_BG.get(row.get("predicted_class", ""), "")
        return [style] * len(row)

    styled_display = display_df.style.apply(_highlight_class, axis=1)
    st.dataframe(styled_display, use_container_width=True, height=400)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("#### Class Distribution")
        if class_counts:
            fig_pie = go.Figure(go.Pie(
                labels=list(class_counts.keys()),
                values=list(class_counts.values()),
                marker_colors=[get_class_color(c) for c in class_counts],
                hole=0.5,
                textinfo="label+value+percent",
                textfont=dict(size=11),
            ))
            fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), **PLOTLY_LIGHT)
            st.plotly_chart(fig_pie, use_container_width=True)

    with chart_cols[1]:
        st.markdown("#### Confidence Distribution")
        if "confidence" in results_df.columns:
            fig_hist = px.histogram(
                results_df, x="confidence", nbins=20,
                color_discrete_sequence=["#4F46E5"],
                labels={"confidence": "Confidence Score"},
            )
            fig_hist.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=30),
                yaxis_title="Count",
                xaxis=dict(**GRID),
                yaxis=dict(**GRID),
                **PLOTLY_LIGHT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    if "gene_symbol" in results_df.columns:
        st.markdown("#### Gene-Level Summary")
        gene_summary = results_df.groupby("gene_symbol").agg(
            count=("variant_id", "count"),
            avg_confidence=("confidence", "mean"),
        ).sort_values("count", ascending=False).head(20).reset_index()

        fig_gene = px.bar(
            gene_summary, x="gene_symbol", y="count", color="avg_confidence",
            color_continuous_scale="Purples",
            labels={"gene_symbol": "Gene", "count": "Count", "avg_confidence": "Avg Conf"},
        )
        fig_gene.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=30), **PLOTLY_LIGHT)
        st.plotly_chart(fig_gene, use_container_width=True)

    st.markdown("---")
    dl_cols = st.columns(2)
    with dl_cols[0]:
        csv_data = export_to_csv(predictions)
        st.download_button(
            "\U0001f4e5  Download Results (CSV)", data=csv_data,
            file_name="batch_results.csv", mime="text/csv", use_container_width=True,
        )
    with dl_cols[1]:
        excel_bytes = export_to_excel(predictions, summary)
        if excel_bytes:
            st.download_button(
                "\U0001f4e5  Download Report (Excel)", data=excel_bytes,
                file_name="batch_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.caption("Excel export requires `openpyxl` package.")
