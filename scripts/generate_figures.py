"""Generate explainability figures for the pathogenicity predictor.

Runs all explainability methods (SHAP, Integrated Gradients, attention
visualisation, LIME) on the test set and saves plots to ``results/figures/``.

Usage::

    python scripts/generate_figures.py --checkpoint results/best_model.ckpt \\
        --config configs/default.yaml --output-dir results/figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate explainability figures for publication.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/figures",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--n-shap-samples", type=int, default=500,
        help="Number of test samples for SHAP analysis.",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP analysis (slow).",
    )
    parser.add_argument(
        "--skip-lime", action="store_true",
        help="Skip LIME analysis.",
    )
    parser.add_argument(
        "--skip-ig", action="store_true",
        help="Skip Integrated Gradients analysis.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all explainability methods and save figures.

    .. note::
        This is a stub that will be fleshed out once the full training pipeline
        is operational. The structure and CLI interface are final.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Config: %s", args.config)
    logger.info("Output dir: %s", output_dir)

    # TODO: load config, checkpoint, and DataModule
    # config = load_config(args.config)
    # model = PathogenicityPredictor.from_config(config)
    # model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    # datamodule = PathogenicityDataModule(config)
    # datamodule.setup("test")
    # test_loader = datamodule.test_dataloader()

    # TODO: run SHAP explainer
    # if not args.skip_shap:
    #     shap_explainer = SHAPExplainer(model, modality_dims)
    #     result = shap_explainer.compute_global_importance(test_data, args.n_shap_samples)
    #     shap_explainer.generate_shap_plots(result["shap_values"], feature_names, output_dir / "shap")

    # TODO: run Integrated Gradients
    # if not args.skip_ig:
    #     ig_explainer = IGExplainer(model, modality_dims)
    #     importance = ig_explainer.compute_modality_importance(test_loader)

    # TODO: run attention visualisation
    # if model.fusion_type in ("attention", "cross_attention", "transformer"):
    #     viz = AttentionVisualizer(model)
    #     weights = viz.collect_attention_weights(test_loader)
    #     viz.plot_attention_heatmap(weights, output_dir / "attention_heatmap.png")
    #     viz.plot_attention_distribution(weights, output_dir / "attention_distribution.png")

    # TODO: run LIME on a few representative samples
    # if not args.skip_lime:
    #     lime_explainer = LIMEExplainer(model, modality_dims)
    #     for i, sample in enumerate(test_samples[:5]):
    #         lime_explainer.explain_instance(sample, feature_names)

    logger.info(
        "Figure generation stub complete. Flesh out after training pipeline "
        "is operational."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
