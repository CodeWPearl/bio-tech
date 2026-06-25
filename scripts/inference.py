"""CLI entry point for running inference on new mutations.

Loads a trained model checkpoint, applies temperature-calibrated predictions,
MC Dropout uncertainty estimation, and outputs per-variant predictions with
confidence, uncertainty, and review recommendations.

Usage::

    python scripts/inference.py --checkpoint best.ckpt --input variants.tsv
    python scripts/inference.py --checkpoint best.ckpt --input variants.tsv --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.full_model import MODALITY_NAMES, PathogenicityPredictor
from src.uncertainty.calibration import TemperatureScaling
from src.uncertainty.mc_dropout import MCDropoutPredictor
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "Pathogenic",
    "Likely Pathogenic",
    "Benign",
    "Likely Benign",
]

HIGH_CONFIDENCE_THRESHOLD: float = 0.7
LOW_CONFIDENCE_THRESHOLD: float = 0.5
UNCERTAINTY_REVIEW_THRESHOLD: float = 0.1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Predict pathogenicity for new variants.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input file of variants to score (TSV/CSV).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save prediction results (JSON). Defaults to stdout.",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=50,
        help="Number of MC Dropout forward passes for uncertainty.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Pre-fitted temperature value for calibration. "
             "If not given, raw probabilities are used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def _load_model(
    checkpoint_path: Path, config: Any,
) -> PathogenicityPredictor:
    """Load model weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the model checkpoint.
        config: Project configuration.

    Returns:
        Model with loaded weights.
    """
    model = PathogenicityPredictor.from_config(config)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if "state_dict" in checkpoint:
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def _prepare_batch(
    row: pd.Series, config: Any,
) -> dict[str, torch.Tensor]:
    """Build a single-sample model batch from a dataframe row.

    Args:
        row: A single row from the input dataframe.
        config: Project configuration.

    Returns:
        Dict of tensors matching the model's expected input format.
    """
    model_cfg = config.model

    dims = {
        "mutation": model_cfg.get("mutation_input_dim", 42),
        "expression": model_cfg.get("expression_input_dim", 2000),
        "methylation": model_cfg.get("methylation_input_dim", 2000),
        "cnv": model_cfg.get("cnv_input_dim", 200),
        "clinical": model_cfg.get("clinical_input_dim", 32),
    }

    batch: dict[str, torch.Tensor] = {}
    mask = []

    for name in MODALITY_NAMES:
        dim = dims[name]
        col_prefix = f"{name}_"
        feature_cols = [c for c in row.index if c.startswith(col_prefix)]

        if feature_cols:
            values = row[feature_cols].values.astype(np.float32)
            if len(values) < dim:
                padded = np.zeros(dim, dtype=np.float32)
                padded[: len(values)] = values
                values = padded
            elif len(values) > dim:
                values = values[:dim]
            batch[name] = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            mask.append(True)
        else:
            batch[name] = torch.zeros(1, dim, dtype=torch.float32)
            mask.append(False)

    batch["modality_mask"] = torch.tensor([mask], dtype=torch.bool)
    return batch


def _get_top_features(
    batch: dict[str, torch.Tensor], probs: torch.Tensor,
) -> str:
    """Generate a simple explanation string from available modality norms.

    Args:
        batch: Input batch tensors.
        probs: Model output probabilities.

    Returns:
        String describing top contributing modalities.
    """
    modality_norms: list[tuple[str, float]] = []
    for name in MODALITY_NAMES:
        if name in batch:
            norm = float(batch[name].norm().item())
            modality_norms.append((name, norm))

    modality_norms.sort(key=lambda x: x[1], reverse=True)
    top = modality_norms[:3]
    parts = [f"{name} (signal={norm:.2f})" for name, norm in top]
    return "Top contributing modalities: " + ", ".join(parts)


def predict_single(
    model: PathogenicityPredictor,
    batch: dict[str, torch.Tensor],
    mc_predictor: MCDropoutPredictor,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Run prediction on a single variant with uncertainty estimation.

    Args:
        model: The trained model.
        batch: Input batch dict.
        mc_predictor: MC Dropout predictor.
        temperature: Optional temperature for calibration.

    Returns:
        Prediction result dict.
    """
    mc_result = mc_predictor.predict_with_uncertainty(batch)

    mean_probs = mc_result["mean_probs"][0]
    predicted_class_idx = int(mc_result["predicted_class"][0].item())
    uncertainty = float(mc_result["epistemic_uncertainty"][0].item())
    confidence = float(mean_probs[predicted_class_idx].item())

    calibrated_probs = mean_probs.tolist()
    if temperature is not None and temperature > 0:
        with torch.no_grad():
            model.eval()
            outputs = model(batch)
            logits = outputs["logits"][0]
            scaler = TemperatureScaling(initial_temperature=temperature)
            scaler.temperature.data.fill_(temperature)
            cal_probs = scaler(logits.unsqueeze(0))
            calibrated_probs = cal_probs[0].tolist()

    explanation = _get_top_features(batch, mean_probs)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD and uncertainty < UNCERTAINTY_REVIEW_THRESHOLD:
        recommendation = "High confidence prediction"
    elif confidence < LOW_CONFIDENCE_THRESHOLD or uncertainty >= UNCERTAINTY_REVIEW_THRESHOLD:
        recommendation = "Low confidence — manual review recommended"
    else:
        recommendation = "Moderate confidence — consider expert review"

    return {
        "predicted_class": CLASS_NAMES[predicted_class_idx],
        "confidence": round(confidence, 4),
        "uncertainty": round(uncertainty, 4),
        "calibrated_probability": [round(p, 4) for p in calibrated_probs],
        "explanation": explanation,
        "recommendation": recommendation,
    }


def main() -> None:
    """Run inference on input variants."""
    args = parse_args()
    log = setup_logging(level="INFO", name=__name__)
    cfg = load_config(args.config)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        log.error("Checkpoint not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    input_path = Path(args.input)
    if not input_path.is_file():
        log.error("Input file not found: %s", input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log.info("Loading model from %s", checkpoint_path)
    model = _load_model(checkpoint_path, cfg)

    mc_predictor = MCDropoutPredictor(model, n_forward_passes=args.mc_passes)

    log.info("Reading input from %s", input_path)
    sep = "\t" if input_path.suffix == ".tsv" else ","
    input_df = pd.read_csv(input_path, sep=sep)

    results: list[dict[str, Any]] = []

    for idx, row in input_df.iterrows():
        batch = _prepare_batch(row, cfg)
        prediction = predict_single(
            model, batch, mc_predictor, temperature=args.temperature,
        )

        meta = {}
        for col in ["gene", "variant", "chromosome", "position", "mutation_type"]:
            if col in row.index and pd.notna(row[col]):
                meta[col] = str(row[col])

        result = {**meta, **prediction}
        results.append(result)

    log.info("Processed %d variants.", len(results))

    output_data = {"predictions": results, "n_variants": len(results)}

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2)
        log.info("Results saved to %s", output_path)
    else:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
