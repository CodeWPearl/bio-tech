"""Tests for the ablation study framework.

Covers modality ablation in PathogenicityPredictor (disabled_modalities
config), ablation config loading, and the comparison table builder.
"""

from __future__ import annotations

import pytest
import torch

from src.models.full_model import MODALITY_NAMES, PathogenicityPredictor
from src.utils.config import Config

# ---------------------------------------------------------------------------
# Shared constants (small dims for fast tests)
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
MUTATION_DIM = 42
MUTATION_EMBED = 64
EXPRESSION_DIM = 100
EXPRESSION_EMBED = 64
METHYLATION_DIM = 100
METHYLATION_EMBED = 64
CNV_DIM = 50
CNV_EMBED = 32
CLINICAL_DIM = 16
CLINICAL_EMBED = 16
FUSION_DIM = 64
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    fusion_type: str = "cross_attention",
    disabled_modalities: list[str] | None = None,
    loss_type: str = "focal",
) -> Config:
    """Build a minimal valid Config for testing."""
    model_cfg: dict = {
        "mutation_input_dim": MUTATION_DIM,
        "mutation_embed_dim": MUTATION_EMBED,
        "expression_input_dim": EXPRESSION_DIM,
        "expression_embed_dim": EXPRESSION_EMBED,
        "methylation_input_dim": METHYLATION_DIM,
        "methylation_embed_dim": METHYLATION_EMBED,
        "cnv_input_dim": CNV_DIM,
        "cnv_embed_dim": CNV_EMBED,
        "clinical_input_dim": CLINICAL_DIM,
        "clinical_embed_dim": CLINICAL_EMBED,
        "fusion_dim": FUSION_DIM,
        "num_classes": NUM_CLASSES,
        "dropout": 0.3,
        "fusion_type": fusion_type,
    }
    if disabled_modalities:
        model_cfg["disabled_modalities"] = disabled_modalities

    return Config({
        "data": {
            "clinvar_url": "https://example.com",
            "cbioportal_url": "https://example.com",
            "studies": [],
            "test_size": 0.15,
            "val_size": 0.15,
            "random_seed": 42,
        },
        "model": model_cfg,
        "training": {
            "max_epochs": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 5,
            "focal_loss_gamma": 2.0,
            "loss_type": loss_type,
            "num_workers": 0,
        },
        "experiment": {
            "name": "test_ablation",
            "tracking_uri": "mlruns",
        },
    })


def _make_batch(batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    """Create a synthetic batch matching the Dataset format."""
    return {
        "mutation": torch.randn(batch_size, MUTATION_DIM),
        "expression": torch.randn(batch_size, EXPRESSION_DIM),
        "methylation": torch.randn(batch_size, METHYLATION_DIM),
        "cnv": torch.randn(batch_size, CNV_DIM),
        "clinical": torch.randn(batch_size, CLINICAL_DIM),
        "modality_mask": torch.ones(batch_size, 5, dtype=torch.bool),
        "label": torch.randint(0, NUM_CLASSES, (batch_size,)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PathogenicityPredictor disabled_modalities
# ═══════════════════════════════════════════════════════════════════════════


class TestDisabledModalities:
    """Tests for the disabled_modalities config in PathogenicityPredictor."""

    def test_no_disabled_modalities_default(self) -> None:
        cfg = _make_config()
        model = PathogenicityPredictor(cfg)
        assert model.disabled_modalities == set()

    def test_disabled_modalities_stored(self) -> None:
        cfg = _make_config(disabled_modalities=["mutation", "cnv"])
        model = PathogenicityPredictor(cfg)
        assert model.disabled_modalities == {"mutation", "cnv"}

    def test_disabled_mutation_zeros_embedding(self) -> None:
        cfg = _make_config(disabled_modalities=["mutation"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert torch.all(embeddings["mutation"] == 0.0)

    def test_disabled_expression_zeros_embedding(self) -> None:
        cfg = _make_config(disabled_modalities=["expression"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert torch.all(embeddings["expression"] == 0.0)

    def test_disabled_methylation_zeros_embedding(self) -> None:
        cfg = _make_config(disabled_modalities=["methylation"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert torch.all(embeddings["methylation"] == 0.0)

    def test_disabled_cnv_zeros_embedding(self) -> None:
        cfg = _make_config(disabled_modalities=["cnv"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert torch.all(embeddings["cnv"] == 0.0)

    def test_enabled_modalities_nonzero(self) -> None:
        cfg = _make_config(disabled_modalities=["mutation"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert not torch.all(embeddings["expression"] == 0.0)
        assert not torch.all(embeddings["cnv"] == 0.0)

    def test_disabled_modality_mask_is_false(self) -> None:
        cfg = _make_config(disabled_modalities=["mutation", "cnv"])
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_with_single_modality(self) -> None:
        disabled = ["expression", "methylation", "cnv", "clinical"]
        cfg = _make_config(disabled_modalities=disabled)
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)
        assert outputs["probabilities"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_forward_with_all_disabled(self) -> None:
        disabled = list(MODALITY_NAMES)
        cfg = _make_config(disabled_modalities=disabled)
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    @pytest.mark.parametrize("fusion_type", [
        "early", "late", "attention", "cross_attention", "transformer",
    ])
    def test_disabled_works_all_fusions(self, fusion_type: str) -> None:
        cfg = _make_config(
            fusion_type=fusion_type,
            disabled_modalities=["mutation", "cnv"],
        )
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_gradients_flow_with_disabled(self) -> None:
        cfg = _make_config(disabled_modalities=["mutation"])
        model = PathogenicityPredictor(cfg)
        model.train()
        batch = _make_batch()
        outputs = model(batch)
        loss = outputs["logits"].sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_multiple_disabled_modalities(self) -> None:
        cfg = _make_config(
            disabled_modalities=["mutation", "expression", "methylation"],
        )
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            embeddings = model._encode_modalities(batch)
        assert torch.all(embeddings["mutation"] == 0.0)
        assert torch.all(embeddings["expression"] == 0.0)
        assert torch.all(embeddings["methylation"] == 0.0)
        assert not torch.all(embeddings["cnv"] == 0.0)
        assert not torch.all(embeddings["clinical"] == 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Ablation config loading
# ═══════════════════════════════════════════════════════════════════════════


class TestAblationConfigs:
    """Tests for loading ablation configs from YAML files."""

    def test_load_no_mutation_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "no_mutation.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        assert "mutation" in cfg.model.disabled_modalities

    def test_load_no_expression_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "no_expression.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        assert "expression" in cfg.model.disabled_modalities

    def test_load_no_attention_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "no_attention.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        assert cfg.model.fusion_type == "early"

    def test_load_no_focal_loss_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "no_focal_loss.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        assert cfg.training.loss_type == "ce"

    def test_load_single_mutation_only_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "single_mutation_only.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        disabled = cfg.model.disabled_modalities
        assert "expression" in disabled
        assert "methylation" in disabled
        assert "cnv" in disabled
        assert "clinical" in disabled
        assert "mutation" not in disabled

    def test_model_from_ablation_config(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "ablation" / "no_mutation.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Ablation config not found")
        cfg = load_config(config_path)
        model = PathogenicityPredictor.from_config(cfg)
        assert "mutation" in model.disabled_modalities

    def test_full_model_has_no_disabled(self) -> None:
        from src.utils.config import load_config
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "default.yaml"
        )
        if not config_path.is_file():
            pytest.skip("Default config not found")
        cfg = load_config(config_path)
        model = PathogenicityPredictor.from_config(cfg)
        assert model.disabled_modalities == set()


# ═══════════════════════════════════════════════════════════════════════════
# Comparison table builder
# ═══════════════════════════════════════════════════════════════════════════


class TestComparisonTable:
    """Tests for the ablation comparison table builder."""

    def test_build_table_structure(self) -> None:
        from scripts.run_ablation import _build_comparison_table

        results = {
            "default": {
                "accuracy": 0.90, "f1_macro": 0.85,
                "roc_auc_macro": 0.95, "pr_auc_macro": 0.88, "mcc": 0.80,
            },
            "no_mutation": {
                "accuracy": 0.85, "f1_macro": 0.80,
                "roc_auc_macro": 0.92, "pr_auc_macro": 0.84, "mcc": 0.75,
            },
        }
        df = _build_comparison_table(results)
        assert len(df) == 2
        assert "Configuration" in df.columns
        assert "Accuracy" in df.columns
        assert "F1-Macro" in df.columns
        assert "AUROC" in df.columns
        assert "PR-AUC" in df.columns
        assert "MCC" in df.columns
        assert "Δ vs Full" in df.columns

    def test_full_model_has_dash_delta(self) -> None:
        from scripts.run_ablation import _build_comparison_table

        results = {
            "default": {
                "accuracy": 0.90, "f1_macro": 0.85,
                "roc_auc_macro": 0.95, "pr_auc_macro": 0.88, "mcc": 0.80,
            },
        }
        df = _build_comparison_table(results)
        assert df.iloc[0]["Δ vs Full"] == "—"

    def test_ablation_has_negative_delta(self) -> None:
        from scripts.run_ablation import _build_comparison_table

        results = {
            "default": {
                "accuracy": 0.90, "f1_macro": 0.85,
                "roc_auc_macro": 0.95, "pr_auc_macro": 0.88, "mcc": 0.80,
            },
            "no_mutation": {
                "accuracy": 0.85, "f1_macro": 0.80,
                "roc_auc_macro": 0.92, "pr_auc_macro": 0.84, "mcc": 0.75,
            },
        }
        df = _build_comparison_table(results)
        delta_str = df.iloc[1]["Δ vs Full"]
        assert delta_str.startswith("-")
        assert delta_str.endswith("%")

    def test_table_preserves_metric_values(self) -> None:
        from scripts.run_ablation import _build_comparison_table

        results = {
            "default": {
                "accuracy": 0.90, "f1_macro": 0.85,
                "roc_auc_macro": 0.95, "pr_auc_macro": 0.88, "mcc": 0.80,
            },
        }
        df = _build_comparison_table(results)
        assert df.iloc[0]["Accuracy"] == pytest.approx(0.90)
        assert df.iloc[0]["F1-Macro"] == pytest.approx(0.85)

    def test_multiple_ablation_rows(self) -> None:
        from scripts.run_ablation import _build_comparison_table

        results = {
            "default": {
                "accuracy": 0.90, "f1_macro": 0.85,
                "roc_auc_macro": 0.95, "pr_auc_macro": 0.88, "mcc": 0.80,
            },
            "no_mutation": {
                "accuracy": 0.85, "f1_macro": 0.80,
                "roc_auc_macro": 0.92, "pr_auc_macro": 0.84, "mcc": 0.75,
            },
            "no_expression": {
                "accuracy": 0.88, "f1_macro": 0.83,
                "roc_auc_macro": 0.94, "pr_auc_macro": 0.87, "mcc": 0.78,
            },
            "no_focal_loss": {
                "accuracy": 0.87, "f1_macro": 0.82,
                "roc_auc_macro": 0.93, "pr_auc_macro": 0.86, "mcc": 0.77,
            },
        }
        df = _build_comparison_table(results)
        assert len(df) == 4
        configs = df["Configuration"].tolist()
        assert "Full Model" in configs
        assert "No Mutation" in configs
        assert "No Expression" in configs

    def test_display_names_mapping(self) -> None:
        from scripts.run_ablation import ABLATION_DISPLAY_NAMES

        assert ABLATION_DISPLAY_NAMES["default"] == "Full Model"
        assert ABLATION_DISPLAY_NAMES["no_mutation"] == "No Mutation"
        assert ABLATION_DISPLAY_NAMES["no_attention"] == "No Attention (Early Fusion)"
        assert ABLATION_DISPLAY_NAMES["no_focal_loss"] == "No Focal Loss (CE)"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ablation config → model → forward pass
# ═══════════════════════════════════════════════════════════════════════════


class TestAblationIntegration:
    """End-to-end tests: load ablation config, build model, run forward."""

    @pytest.mark.parametrize("disabled", [
        ["mutation"],
        ["expression"],
        ["methylation"],
        ["cnv"],
        ["expression", "methylation", "cnv", "clinical"],
        ["mutation", "methylation", "cnv", "clinical"],
    ])
    def test_forward_with_various_disabled(
        self, disabled: list[str],
    ) -> None:
        cfg = _make_config(disabled_modalities=disabled)
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)
        probs = outputs["probabilities"]
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5,
        )

    def test_early_fusion_ablation(self) -> None:
        cfg = _make_config(fusion_type="early")
        model = PathogenicityPredictor(cfg)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["logits"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_ce_loss_ablation(self) -> None:
        from src.training.lightning_module import PathogenicityLightningModule

        cfg = _make_config(loss_type="ce")
        model = PathogenicityPredictor(cfg)
        module = PathogenicityLightningModule(config=cfg, model=model)
        assert isinstance(module.loss_fn, torch.nn.CrossEntropyLoss)

    def test_disabled_modality_produces_different_output(self) -> None:
        torch.manual_seed(42)
        cfg_full = _make_config()
        model_full = PathogenicityPredictor(cfg_full)
        model_full.eval()

        torch.manual_seed(42)
        cfg_ablated = _make_config(disabled_modalities=["mutation"])
        model_ablated = PathogenicityPredictor(cfg_ablated)
        model_ablated.load_state_dict(model_full.state_dict())
        model_ablated.eval()

        torch.manual_seed(0)
        batch = _make_batch()

        with torch.no_grad():
            out_full = model_full(batch)
            out_ablated = model_ablated(batch)

        assert not torch.allclose(
            out_full["logits"], out_ablated["logits"], atol=1e-3,
        )
