"""Data export utilities for batch predictions (CSV, Excel, JSON)."""

from __future__ import annotations

import json
import logging
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)


def export_to_csv(predictions_list: list[dict[str, Any]]) -> str:
    """Export a list of prediction responses to a CSV string.

    Args:
        predictions_list: List of prediction response dicts.

    Returns:
        CSV-formatted string.
    """
    import csv
    from io import StringIO

    if not predictions_list:
        return ""

    buffer = StringIO()
    fieldnames = [
        "variant_id",
        "predicted_class",
        "confidence",
        "recommendation",
        "epistemic_uncertainty",
        "confidence_level",
        "gene_symbol",
        "is_cancer_driver",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for pred in predictions_list:
        uncertainty = pred.get("uncertainty", {}) or {}
        bio = pred.get("biological_context", {}) or {}
        row = {
            "variant_id": pred.get("variant_id", ""),
            "predicted_class": pred.get("predicted_class", ""),
            "confidence": pred.get("confidence", 0),
            "recommendation": pred.get("recommendation", ""),
            "epistemic_uncertainty": uncertainty.get("epistemic_uncertainty", ""),
            "confidence_level": uncertainty.get("confidence_level", ""),
            "gene_symbol": bio.get("gene_symbol", ""),
            "is_cancer_driver": bio.get("is_known_cancer_driver", ""),
        }
        writer.writerow(row)

    return buffer.getvalue()


def export_to_excel(
    predictions_list: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Export predictions to an Excel workbook with multiple sheets.

    Sheets: Predictions, Summary Statistics, Feature Importance.

    Args:
        predictions_list: List of prediction response dicts.
        metadata: Optional dict with summary/batch metadata.

    Returns:
        Excel file bytes, or empty bytes if openpyxl is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — Excel export unavailable")
        return b""

    if not predictions_list:
        return b""

    buffer = BytesIO()

    pred_rows = []
    feat_rows = []

    for pred in predictions_list:
        uncertainty = pred.get("uncertainty", {}) or {}
        bio = pred.get("biological_context", {}) or {}
        class_probs = pred.get("class_probabilities", {}) or {}

        row: dict[str, Any] = {
            "variant_id": pred.get("variant_id", ""),
            "predicted_class": pred.get("predicted_class", ""),
            "confidence": pred.get("confidence", 0),
            "recommendation": pred.get("recommendation", ""),
        }
        for cls, prob in class_probs.items():
            row[f"prob_{cls}"] = prob

        row["epistemic_uncertainty"] = uncertainty.get("epistemic_uncertainty", "")
        row["confidence_level"] = uncertainty.get("confidence_level", "")
        row["gene_symbol"] = bio.get("gene_symbol", "")
        row["is_cancer_driver"] = bio.get("is_known_cancer_driver", "")

        pred_rows.append(row)

        explanation = pred.get("explanation")
        if explanation:
            for feat in explanation.get("top_positive_features", [])[:5]:
                feat_rows.append({
                    "variant_id": pred.get("variant_id", ""),
                    "feature_name": feat.get("feature_name", ""),
                    "importance": feat.get("importance", 0),
                    "direction": "positive",
                })

    predictions_df = pd.DataFrame(pred_rows)

    summary_data: dict[str, Any] = {}
    if metadata:
        summary_data = dict(metadata)
    else:
        summary_data = {
            "total_variants": len(predictions_list),
            "avg_confidence": predictions_df["confidence"].mean()
            if not predictions_df.empty
            else 0,
        }
        if not predictions_df.empty:
            counts = predictions_df["predicted_class"].value_counts().to_dict()
            for cls, count in counts.items():
                summary_data[f"count_{cls}"] = count

    summary_df = pd.DataFrame([summary_data])
    feat_df = pd.DataFrame(feat_rows) if feat_rows else pd.DataFrame()

    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
            summary_df.to_excel(writer, sheet_name="Summary Statistics", index=False)
            if not feat_df.empty:
                feat_df.to_excel(
                    writer, sheet_name="Feature Importance", index=False
                )
    except ImportError:
        logger.warning("openpyxl not installed — Excel export unavailable")
        return b""

    return buffer.getvalue()


def export_to_json(prediction_response: dict[str, Any]) -> str:
    """Export a prediction response to a formatted JSON string.

    Args:
        prediction_response: A single prediction response dict.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps(prediction_response, indent=2, default=str)
