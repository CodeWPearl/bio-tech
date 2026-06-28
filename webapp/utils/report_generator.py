"""PDF report generation for Cancer Mutation Pathogenicity predictions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


def generate_prediction_report(prediction_response: dict[str, Any]) -> bytes:
    """Generate a professional PDF report from a prediction response.

    Args:
        prediction_response: Full prediction response dict containing
            variant info, prediction results, class probabilities,
            uncertainty, explanation, and biological context.

    Returns:
        PDF file bytes, or empty bytes if reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        logger.warning("reportlab not installed — PDF export unavailable")
        return b""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    elements: list = []

    brand_color = colors.Color(0.39, 0.40, 0.95)
    light_bg = colors.Color(0.95, 0.95, 0.99)

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=brand_color,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=brand_color,
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    def _make_table(
        data: list[list[str]],
        col_widths: list[float] | None = None,
    ) -> Table:
        if col_widths is None:
            col_widths = [2.5 * inch, 4.2 * inch]
        tbl = Table(data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), brand_color),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BACKGROUND", (1, 0), (1, -1), light_bg),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.92)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return tbl

    # --- Header ---
    elements.append(Paragraph(
        "\U0001f9ec Cancer Mutation Pathogenicity Report", title_style
    ))
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elements.append(Paragraph(
        f"Generated: {timestamp} &nbsp;|&nbsp; Model v{MODEL_VERSION}",
        subtitle_style,
    ))
    elements.append(Spacer(1, 0.15 * inch))

    # --- Variant Information ---
    variant_id = prediction_response.get("variant_id", "N/A")
    bio = prediction_response.get("biological_context", {})
    gene = bio.get("gene_symbol", variant_id.split("_")[0] if "_" in variant_id else "")
    parts = variant_id.split("_") if "_" in variant_id else []

    elements.append(Paragraph("Variant Information", heading_style))
    variant_rows = [
        ["Variant ID", variant_id],
        ["Gene Symbol", gene],
    ]
    if len(parts) >= 5:
        variant_rows.extend([
            ["Chromosome", parts[1]],
            ["Position", parts[2]],
            ["Ref / Alt", f"{parts[3]} > {parts[4]}"],
        ])

    protein_change = prediction_response.get("protein_change", "")
    if protein_change:
        variant_rows.append(["Protein Change", protein_change])

    elements.append(_make_table(variant_rows))
    elements.append(Spacer(1, 0.15 * inch))

    # --- Prediction Result ---
    pred_class = prediction_response.get("predicted_class", "Unknown")
    confidence = prediction_response.get("confidence", 0.0)
    recommendation = prediction_response.get("recommendation", "")

    elements.append(Paragraph("Prediction Result", heading_style))
    result_rows = [
        ["Predicted Class", pred_class],
        ["Confidence", f"{confidence * 100:.1f}%"],
        ["Recommendation", recommendation or "N/A"],
    ]
    elements.append(_make_table(result_rows))
    elements.append(Spacer(1, 0.15 * inch))

    # --- Class Probabilities ---
    class_probs = prediction_response.get("class_probabilities", {})
    if class_probs:
        elements.append(Paragraph("Class Probabilities", heading_style))
        prob_rows = [
            [cls, f"{prob * 100:.2f}%"] for cls, prob in class_probs.items()
        ]
        elements.append(_make_table(prob_rows, [3 * inch, 3.7 * inch]))
        elements.append(Spacer(1, 0.15 * inch))

    # --- Uncertainty Analysis ---
    uncertainty = prediction_response.get("uncertainty")
    if uncertainty:
        elements.append(Paragraph("Uncertainty Analysis", heading_style))
        epistemic = uncertainty.get("epistemic_uncertainty", 0)
        entropy = uncertainty.get("predictive_entropy", 0)
        calibrated = uncertainty.get("calibrated", False)
        conf_level = uncertainty.get("confidence_level", "N/A")

        unc_rows = [
            ["Epistemic Uncertainty", f"{epistemic:.4f}"],
            ["Predictive Entropy", f"{entropy:.4f}"],
            ["Calibrated", "Yes" if calibrated else "No"],
            ["Confidence Level", str(conf_level)],
        ]
        elements.append(_make_table(unc_rows))

        if epistemic < 0.05:
            interp = "Low uncertainty - the model is confident in this prediction."
        elif epistemic < 0.15:
            interp = "Moderate uncertainty - manual review may be warranted."
        else:
            interp = "High uncertainty - expert review is strongly recommended."
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(Paragraph(f"<i>{interp}</i>", body_style))
        elements.append(Spacer(1, 0.15 * inch))

    # --- Top Contributing Features ---
    explanation = prediction_response.get("explanation")
    if explanation:
        pos_features = explanation.get("top_positive_features", [])
        neg_features = explanation.get("top_negative_features", [])

        if pos_features or neg_features:
            elements.append(Paragraph("Top Contributing Features", heading_style))

            feat_header = [["Feature", "Direction", "Importance"]]
            feat_rows = []
            for f in pos_features[:8]:
                feat_rows.append([
                    f.get("feature_name", ""),
                    "Pathogenic (+)",
                    f"{f.get('importance', 0):.4f}",
                ])
            for f in neg_features[:5]:
                feat_rows.append([
                    f.get("feature_name", ""),
                    "Benign (-)",
                    f"{f.get('importance', 0):.4f}",
                ])

            if feat_rows:
                ftbl = Table(
                    feat_header + feat_rows,
                    colWidths=[2.5 * inch, 1.8 * inch, 2.4 * inch],
                )
                ftbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), brand_color),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.92)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_bg]),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                elements.append(ftbl)
            elements.append(Spacer(1, 0.15 * inch))

        # --- Modality Contributions ---
        modality_contrib = explanation.get("modality_contributions", {})
        if modality_contrib:
            elements.append(Paragraph("Modality Contributions", heading_style))
            mod_rows = [
                [mod, f"{val * 100:.1f}%"]
                for mod, val in modality_contrib.items()
            ]
            elements.append(_make_table(mod_rows))
            elements.append(Spacer(1, 0.15 * inch))

    # --- Biological Context ---
    if bio:
        elements.append(Paragraph("Biological Context", heading_style))
        is_driver = bio.get("is_known_cancer_driver", False)
        cosmic = bio.get("cosmic_census_info", "")
        clinvar_count = bio.get("clinvar_entries", 0)
        var_desc = bio.get("variant_type_description", "")

        bio_rows = [
            ["Known Cancer Driver", "Yes" if is_driver else "No"],
        ]
        if cosmic:
            bio_rows.append(["COSMIC Info", cosmic])
        if var_desc:
            bio_rows.append(["Variant Type", var_desc])
        bio_rows.append(["ClinVar Entries", str(clinvar_count)])

        elements.append(_make_table(bio_rows))
        elements.append(Spacer(1, 0.2 * inch))

    # --- Disclaimer ---
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        leading=11,
        spaceBefore=12,
        borderPadding=6,
    )
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This is a research tool, not a clinical diagnostic. "
        "Predictions are generated by a machine learning model and should not "
        "be used as the sole basis for clinical decisions. Always consult with "
        "qualified healthcare professionals and validated clinical tools for "
        "diagnostic or treatment decisions.",
        disclaimer_style,
    ))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(
        f"Report generated on {timestamp} | Model version {MODEL_VERSION} | "
        "Cancer Mutation Pathogenicity Predictor",
        disclaimer_style,
    ))

    doc.build(elements)
    return buffer.getvalue()
