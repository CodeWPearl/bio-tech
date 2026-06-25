"""Tests for the hyperparameter optimisation pipeline.

Covers search space suggestion, config application, objective function
construction, best config saving, study creation, and plot generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml

from src.utils.config import Config, load_config

from scripts.run_hpo import (
    apply_hpo_params,
    create_objective,
    generate_plots,
    save_best_config,
    suggest_hyperparameters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sweep_config(**overrides: Any) -> Config:
    """Build a minimal sweep config for testing."""
    data: dict[str, Any] = {
        "hpo": {
            "n_trials": 3,
            "timeout": None,
            "study_name": "test_hpo",
            "storage": "sqlite:///results/test_hpo.db",
            "sampler": "tpe",
            "pruner": "median",
            "pruner_n_startup_trials": 2,
            "pruner_n_warmup_steps": 2,
            "direction": "maximize",
            "metric": "val_auroc",
            "search_space": {
                "learning_rate": {
                    "type": "float",
                    "low": 1e-5,
                    "high": 1e-2,
                    "log": True,
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [32, 64, 128],
                },
                "dropout": {
                    "type": "float",
                    "low": 0.1,
                    "high": 0.5,
                },
                "fusion_type": {
                    "type": "categorical",
                    "choices": ["early", "attention", "cross_attention", "transformer"],
                },
                "mutation_embed_dim": {
                    "type": "categorical",
                    "choices": [64, 128, 256],
                },
                "expression_embed_dim": {
                    "type": "categorical",
                    "choices": [128, 256, 512],
                },
                "focal_loss_gamma": {
                    "type": "float",
                    "low": 0.5,
                    "high": 5.0,
                },
                "weight_decay": {
                    "type": "float",
                    "low": 1e-6,
                    "high": 1e-2,
                    "log": True,
                },
                "num_attention_heads": {
                    "type": "categorical",
                    "choices": [2, 4, 8],
                },
            },
            "training": {
                "max_epochs": 2,
                "patience": 2,
            },
            "retrain": {
                "enabled": False,
                "max_epochs": 5,
                "patience": 3,
            },
        },
        "data": {
            "clinvar_url": "https://example.com",
            "cbioportal_url": "https://example.com",
            "studies": ["test_study"],
            "test_size": 0.15,
            "val_size": 0.15,
            "random_seed": 42,
        },
        "model": {
            "mutation_input_dim": 10,
            "mutation_embed_dim": 16,
            "expression_input_dim": 20,
            "expression_embed_dim": 16,
            "methylation_input_dim": 20,
            "methylation_embed_dim": 16,
            "cnv_input_dim": 10,
            "cnv_embed_dim": 8,
            "clinical_input_dim": 8,
            "clinical_embed_dim": 8,
            "fusion_dim": 32,
            "num_classes": 4,
            "dropout": 0.1,
            "fusion_type": "early",
        },
        "training": {
            "max_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "patience": 5,
            "focal_loss_gamma": 2.0,
            "loss_type": "focal",
            "scheduler_type": "cosine_warm_restarts",
            "cosine_t0": 1,
            "cosine_t_mult": 1,
            "num_workers": 0,
        },
        "experiment": {
            "name": "test_hpo",
            "tracking_uri": "mlruns",
        },
    }
    for key, val in overrides.items():
        parts = key.split(".")
        d = data
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = val
    return Config(data)


def _make_search_space() -> dict[str, Any]:
    """Return the default search space dict for testing."""
    return {
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-2,
            "log": True,
        },
        "batch_size": {
            "type": "categorical",
            "choices": [32, 64, 128],
        },
        "dropout": {
            "type": "float",
            "low": 0.1,
            "high": 0.5,
        },
        "fusion_type": {
            "type": "categorical",
            "choices": ["early", "attention", "cross_attention", "transformer"],
        },
        "mutation_embed_dim": {
            "type": "categorical",
            "choices": [64, 128, 256],
        },
        "expression_embed_dim": {
            "type": "categorical",
            "choices": [128, 256, 512],
        },
        "focal_loss_gamma": {
            "type": "float",
            "low": 0.5,
            "high": 5.0,
        },
        "weight_decay": {
            "type": "float",
            "low": 1e-6,
            "high": 1e-2,
            "log": True,
        },
        "num_attention_heads": {
            "type": "categorical",
            "choices": [2, 4, 8],
        },
    }


# ===========================================================================
# suggest_hyperparameters
# ===========================================================================

class TestSuggestHyperparameters:
    """Tests for hyperparameter suggestion from the search space."""

    def test_returns_all_params(self) -> None:
        """All search space keys appear in the suggested dict."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        search_space = _make_search_space()
        params = suggest_hyperparameters(trial, search_space)
        assert set(params.keys()) == set(search_space.keys())

    def test_learning_rate_in_range(self) -> None:
        """Learning rate is within the log-uniform range."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert 1e-5 <= params["learning_rate"] <= 1e-2

    def test_batch_size_categorical(self) -> None:
        """Batch size is one of the allowed values."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert params["batch_size"] in [32, 64, 128]

    def test_dropout_in_range(self) -> None:
        """Dropout is within the uniform range."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert 0.1 <= params["dropout"] <= 0.5

    def test_fusion_type_categorical(self) -> None:
        """Fusion type is one of the allowed values."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert params["fusion_type"] in [
            "early", "attention", "cross_attention", "transformer",
        ]

    def test_mutation_embed_dim_categorical(self) -> None:
        """Mutation embed dim is one of the allowed values."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert params["mutation_embed_dim"] in [64, 128, 256]

    def test_expression_embed_dim_categorical(self) -> None:
        """Expression embed dim is one of the allowed values."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert params["expression_embed_dim"] in [128, 256, 512]

    def test_focal_loss_gamma_in_range(self) -> None:
        """Focal loss gamma is within the uniform range."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert 0.5 <= params["focal_loss_gamma"] <= 5.0

    def test_weight_decay_in_range(self) -> None:
        """Weight decay is within the log-uniform range."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert 1e-6 <= params["weight_decay"] <= 1e-2

    def test_num_attention_heads_categorical(self) -> None:
        """Attention heads is one of the allowed values."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = suggest_hyperparameters(trial, _make_search_space())
        assert params["num_attention_heads"] in [2, 4, 8]

    def test_unknown_type_raises(self) -> None:
        """Unknown search space type raises ValueError."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        bad_space = {"foo": {"type": "unknown", "low": 0, "high": 1}}
        with pytest.raises(ValueError, match="Unknown search space type"):
            suggest_hyperparameters(trial, bad_space)

    def test_int_type_supported(self) -> None:
        """Integer search space type works correctly."""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        space = {"hidden_dim": {"type": "int", "low": 32, "high": 256}}
        params = suggest_hyperparameters(trial, space)
        assert 32 <= params["hidden_dim"] <= 256
        assert isinstance(params["hidden_dim"], int)


# ===========================================================================
# apply_hpo_params
# ===========================================================================

class TestApplyHpoParams:
    """Tests for applying suggested parameters to a config."""

    def test_learning_rate_applied(self) -> None:
        """Learning rate is written to training section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"learning_rate": 0.005})
        assert result.training.learning_rate == 0.005

    def test_batch_size_applied(self) -> None:
        """Batch size is written to training section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"batch_size": 128})
        assert result.training.batch_size == 128

    def test_dropout_applied(self) -> None:
        """Dropout is written to model section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"dropout": 0.35})
        assert result.model.dropout == 0.35

    def test_fusion_type_applied(self) -> None:
        """Fusion type is written to model section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"fusion_type": "transformer"})
        assert result.model.fusion_type == "transformer"

    def test_mutation_embed_dim_applied(self) -> None:
        """Mutation embed dim is written to model section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"mutation_embed_dim": 256})
        assert result.model.mutation_embed_dim == 256

    def test_expression_embed_dim_applied(self) -> None:
        """Expression embed dim is written to model section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"expression_embed_dim": 512})
        assert result.model.expression_embed_dim == 512

    def test_focal_loss_gamma_applied(self) -> None:
        """Focal loss gamma is written to training section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"focal_loss_gamma": 3.5})
        assert result.training.focal_loss_gamma == 3.5

    def test_weight_decay_applied(self) -> None:
        """Weight decay is written to training section."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"weight_decay": 0.001})
        assert result.training.weight_decay == 0.001

    def test_multiple_params_applied(self) -> None:
        """Multiple parameters are applied simultaneously."""
        cfg = _make_sweep_config()
        params = {
            "learning_rate": 0.003,
            "dropout": 0.4,
            "fusion_type": "attention",
            "batch_size": 32,
        }
        result = apply_hpo_params(cfg, params)
        assert result.training.learning_rate == 0.003
        assert result.model.dropout == 0.4
        assert result.model.fusion_type == "attention"
        assert result.training.batch_size == 32

    def test_original_config_unchanged(self) -> None:
        """Applying params does not mutate the original config."""
        cfg = _make_sweep_config()
        original_lr = cfg.training.learning_rate
        apply_hpo_params(cfg, {"learning_rate": 0.999})
        assert cfg.training.learning_rate == original_lr

    def test_unknown_param_ignored(self) -> None:
        """Unknown parameters are silently ignored."""
        cfg = _make_sweep_config()
        result = apply_hpo_params(cfg, {"unknown_param": 42})
        assert result.training.learning_rate == cfg.training.learning_rate


# ===========================================================================
# save_best_config
# ===========================================================================

class TestSaveBestConfig:
    """Tests for saving the best configuration."""

    def test_saves_yaml_file(self, tmp_path: Path) -> None:
        """Best config is saved as a valid YAML file."""
        cfg = _make_sweep_config()
        params = {"learning_rate": 0.005, "dropout": 0.3}
        output = tmp_path / "best.yaml"

        save_best_config(cfg, params, output)

        assert output.is_file()
        with output.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert isinstance(loaded, dict)

    def test_params_in_saved_config(self, tmp_path: Path) -> None:
        """Saved config reflects the best parameters."""
        cfg = _make_sweep_config()
        params = {"learning_rate": 0.005, "fusion_type": "transformer"}
        output = tmp_path / "best.yaml"

        save_best_config(cfg, params, output)

        with output.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["training"]["learning_rate"] == 0.005
        assert loaded["model"]["fusion_type"] == "transformer"

    def test_hpo_section_removed(self, tmp_path: Path) -> None:
        """The hpo section is removed from the saved config."""
        cfg = _make_sweep_config()
        output = tmp_path / "best.yaml"

        save_best_config(cfg, {}, output)

        with output.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert "hpo" not in loaded

    def test_required_sections_present(self, tmp_path: Path) -> None:
        """Saved config has all required sections."""
        cfg = _make_sweep_config()
        output = tmp_path / "best.yaml"

        save_best_config(cfg, {}, output)

        with output.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        for section in ("data", "model", "training", "experiment"):
            assert section in loaded

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        cfg = _make_sweep_config()
        output = tmp_path / "nested" / "deep" / "best.yaml"

        save_best_config(cfg, {}, output)

        assert output.is_file()


# ===========================================================================
# Optuna study creation
# ===========================================================================

class TestStudyCreation:
    """Tests for Optuna study setup and configuration."""

    def test_tpe_sampler(self) -> None:
        """TPE sampler is created correctly."""
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_median_pruner(self) -> None:
        """MedianPruner is created correctly."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5,
        )
        study = optuna.create_study(pruner=pruner, direction="maximize")
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_maximize_direction(self) -> None:
        """Study direction is set to maximize."""
        study = optuna.create_study(direction="maximize")
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_study_with_sqlite_storage(self, tmp_path: Path) -> None:
        """Study can be created with SQLite storage."""
        db_path = tmp_path / "test_study.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="test",
            storage=storage,
            direction="maximize",
        )
        assert study.study_name == "test"

    def test_study_load_if_exists(self, tmp_path: Path) -> None:
        """Study can be reloaded from existing storage."""
        db_path = tmp_path / "test_study.db"
        storage = f"sqlite:///{db_path}"

        study1 = optuna.create_study(
            study_name="reload_test",
            storage=storage,
            direction="maximize",
        )
        study1.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                distributions={"x": optuna.distributions.FloatDistribution(0, 2)},
                values=[0.8],
            ),
        )

        study2 = optuna.create_study(
            study_name="reload_test",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        assert len(study2.trials) == 1


# ===========================================================================
# generate_plots
# ===========================================================================

class TestGeneratePlots:
    """Tests for Optuna visualisation plot generation."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Output directory is created if absent."""
        study = optuna.create_study(direction="maximize")
        output_dir = tmp_path / "plots"

        generate_plots(study, output_dir)

        assert output_dir.is_dir()

    def test_no_crash_with_empty_study(self, tmp_path: Path) -> None:
        """Handles a study with no trials gracefully."""
        study = optuna.create_study(direction="maximize")
        output_dir = tmp_path / "plots"

        generate_plots(study, output_dir)

    def test_no_crash_with_single_trial(self, tmp_path: Path) -> None:
        """Handles a study with one trial gracefully."""
        study = optuna.create_study(direction="maximize")
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                distributions={"x": optuna.distributions.FloatDistribution(0, 2)},
                values=[0.75],
            ),
        )
        output_dir = tmp_path / "plots"

        generate_plots(study, output_dir)


# ===========================================================================
# create_objective
# ===========================================================================

class TestCreateObjective:
    """Tests for the objective function factory."""

    def test_returns_callable(self) -> None:
        """create_objective returns a callable."""
        cfg = _make_sweep_config()
        obj = create_objective(cfg, gpus=0)
        assert callable(obj)


# ===========================================================================
# sweep.yaml config loading
# ===========================================================================

class TestSweepConfig:
    """Tests for loading the sweep configuration file."""

    def test_sweep_config_loads(self) -> None:
        """configs/sweep.yaml loads without errors."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        assert hasattr(cfg, "hpo")

    def test_sweep_config_has_search_space(self) -> None:
        """configs/sweep.yaml has a search_space section."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        space = cfg.hpo.search_space.to_dict()
        expected_keys = {
            "learning_rate", "batch_size", "dropout", "fusion_type",
            "mutation_embed_dim", "expression_embed_dim",
            "focal_loss_gamma", "weight_decay", "num_attention_heads",
        }
        assert expected_keys == set(space.keys())

    def test_sweep_config_has_training(self) -> None:
        """configs/sweep.yaml has HPO training settings."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        assert hasattr(cfg.hpo.training, "max_epochs")
        assert hasattr(cfg.hpo.training, "patience")

    def test_sweep_config_has_retrain(self) -> None:
        """configs/sweep.yaml has retrain settings."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        assert hasattr(cfg.hpo.retrain, "enabled")
        assert hasattr(cfg.hpo.retrain, "max_epochs")

    def test_sweep_config_has_study_params(self) -> None:
        """configs/sweep.yaml has Optuna study parameters."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        assert hasattr(cfg.hpo, "n_trials")
        assert hasattr(cfg.hpo, "study_name")
        assert hasattr(cfg.hpo, "storage")
        assert hasattr(cfg.hpo, "sampler")
        assert hasattr(cfg.hpo, "pruner")

    def test_all_search_space_types_valid(self) -> None:
        """All search space entries have valid types."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        space = cfg.hpo.search_space.to_dict()
        valid_types = {"float", "int", "categorical"}
        for name, spec in space.items():
            assert spec["type"] in valid_types, f"{name} has invalid type"

    def test_categorical_params_have_choices(self) -> None:
        """Categorical parameters have a choices list."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        space = cfg.hpo.search_space.to_dict()
        for name, spec in space.items():
            if spec["type"] == "categorical":
                assert "choices" in spec, f"{name} missing choices"
                assert len(spec["choices"]) >= 2, f"{name} needs >=2 choices"

    def test_float_params_have_bounds(self) -> None:
        """Float parameters have low and high bounds."""
        config_path = Path("configs/sweep.yaml")
        if not config_path.is_file():
            pytest.skip("configs/sweep.yaml not found")
        cfg = load_config(config_path, validate=True)
        space = cfg.hpo.search_space.to_dict()
        for name, spec in space.items():
            if spec["type"] == "float":
                assert "low" in spec, f"{name} missing low"
                assert "high" in spec, f"{name} missing high"
                assert spec["low"] < spec["high"], f"{name}: low >= high"
