"""Hyperparameter optimisation with Optuna for pathogenicity prediction.

Defines an Optuna objective that suggests hyperparameters from a
configurable search space, trains the model for a limited number of
epochs with early stopping, and returns validation AUROC for
maximisation.  After the sweep, the best configuration is saved and
optionally used to retrain on the full train+val data.

Usage::

    python scripts/run_hpo.py
    python scripts/run_hpo.py --config configs/sweep.yaml --n-trials 100
    python scripts/run_hpo.py --config configs/sweep.yaml --retrain
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
import yaml
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.datamodule import PathogenicityDataModule
from src.models.full_model import PathogenicityPredictor
from src.training.callbacks import EarlyStoppingWithPatience
from src.training.lightning_module import PathogenicityLightningModule
from src.utils.config import Config, load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for HPO.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimisation.",
    )
    parser.add_argument(
        "--config",
        default="configs/sweep.yaml",
        help="Path to the sweep YAML configuration file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the entire study (overrides config).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain final model with best config on train+val data.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files.",
    )
    return parser.parse_args()


def suggest_hyperparameters(
    trial: optuna.Trial, search_space: dict[str, Any],
) -> dict[str, Any]:
    """Suggest hyperparameters from the configured search space.

    Args:
        trial: The current Optuna trial.
        search_space: Search space definition from sweep config.

    Returns:
        Dict of suggested hyperparameter values.
    """
    params: dict[str, Any] = {}

    for name, spec in search_space.items():
        param_type = spec["type"]

        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search space type '{param_type}' for param '{name}'"
            )

    return params


def apply_hpo_params(
    base_cfg: Config, params: dict[str, Any],
) -> Config:
    """Apply suggested hyperparameters to a base configuration.

    Args:
        base_cfg: The base configuration to modify.
        params: Dict of hyperparameter name → suggested value.

    Returns:
        A new Config with the suggested parameters applied.
    """
    data = base_cfg.to_dict()

    param_to_config: dict[str, tuple[str, str]] = {
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "dropout": ("model", "dropout"),
        "fusion_type": ("model", "fusion_type"),
        "mutation_embed_dim": ("model", "mutation_embed_dim"),
        "expression_embed_dim": ("model", "expression_embed_dim"),
        "focal_loss_gamma": ("training", "focal_loss_gamma"),
        "weight_decay": ("training", "weight_decay"),
        "num_attention_heads": ("model", "num_attention_heads"),
    }

    for param_name, value in params.items():
        if param_name in param_to_config:
            section, key = param_to_config[param_name]
            data[section][key] = value

    return Config(data)


def create_objective(
    sweep_cfg: Config,
    gpus: int | None,
) -> Any:
    """Create an Optuna objective function closure.

    Args:
        sweep_cfg: The full sweep configuration.
        gpus: Number of GPUs to use.

    Returns:
        The objective callable for ``study.optimize``.
    """
    search_space = sweep_cfg.hpo.search_space.to_dict()
    hpo_max_epochs = int(sweep_cfg.hpo.training.max_epochs)
    hpo_patience = int(sweep_cfg.hpo.training.patience)

    def objective(trial: optuna.Trial) -> float:
        """Train a model with suggested hyperparameters and return val AUROC.

        Args:
            trial: The current Optuna trial.

        Returns:
            Validation AUROC score to maximise.
        """
        params = suggest_hyperparameters(trial, search_space)
        cfg = apply_hpo_params(sweep_cfg, params)

        seed_everything(int(cfg.data.random_seed))

        datamodule = PathogenicityDataModule(cfg)
        datamodule.setup("fit")

        class_weights = None
        if datamodule._class_weights is not None:
            class_weights = torch.tensor(
                datamodule._class_weights, dtype=torch.float32,
            )

        model = PathogenicityPredictor.from_config(cfg)
        lightning_module = PathogenicityLightningModule(
            config=cfg, model=model, class_weights=class_weights,
        )

        callbacks: list[pl.Callback] = [
            EarlyStoppingWithPatience(patience=hpo_patience),
            PyTorchLightningPruningCallback(trial, monitor="val_auroc"),
        ]

        accelerator = "auto"
        devices: int | str = "auto"
        if gpus is not None:
            accelerator = "gpu" if gpus > 0 else "cpu"
            devices = gpus if gpus > 0 else 1

        trainer = pl.Trainer(
            max_epochs=hpo_max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=False,
            deterministic=True,
            log_every_n_steps=10,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        try:
            import mlflow

            tracking_uri = cfg.experiment.tracking_uri
            experiment_name = f"{cfg.experiment.name}_hpo"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"trial_{trial.number}"):
                mlflow.log_params(params)
                trainer.fit(lightning_module, datamodule=datamodule)

                val_auroc = trainer.callback_metrics.get("val_auroc")
                if val_auroc is not None:
                    val_auroc_value = float(val_auroc.item())
                    mlflow.log_metric("val_auroc", val_auroc_value)
                else:
                    val_auroc_value = 0.0

        except ImportError:
            trainer.fit(lightning_module, datamodule=datamodule)
            val_auroc = trainer.callback_metrics.get("val_auroc")
            val_auroc_value = (
                float(val_auroc.item()) if val_auroc is not None else 0.0
            )

        logger.info(
            "Trial %d finished: val_auroc=%.4f | params=%s",
            trial.number, val_auroc_value, params,
        )

        return val_auroc_value

    return objective


def save_best_config(
    sweep_cfg: Config,
    best_params: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the best hyperparameter configuration as a YAML file.

    Args:
        sweep_cfg: The base sweep configuration.
        best_params: Best parameters from the study.
        output_path: Path to write the YAML config.
    """
    best_cfg = apply_hpo_params(sweep_cfg, best_params)
    data = best_cfg.to_dict()

    for section_name in ("hpo",):
        data.pop(section_name, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Saved best config to %s", output_path)


def generate_plots(study: optuna.Study, output_dir: Path) -> None:
    """Generate Optuna visualisation plots.

    Args:
        study: The completed Optuna study.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) < 2:
            logger.warning(
                "Need at least 2 completed trials for plots; got %d",
                len(completed_trials),
            )
            return

        fig = plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        fig.write_image(str(output_dir / "optimization_history.png"))
        logger.info("Saved optimization history plot")

        fig = plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
        fig.write_image(str(output_dir / "parallel_coordinate.png"))
        logger.info("Saved parallel coordinate plot")

        if len(completed_trials) >= 4:
            fig = plot_param_importances(study)
            fig.write_html(str(output_dir / "param_importances.html"))
            fig.write_image(str(output_dir / "param_importances.png"))
            logger.info("Saved parameter importance plot")
        else:
            logger.warning(
                "Need >= 4 completed trials for importance plot; got %d",
                len(completed_trials),
            )

    except ImportError:
        logger.warning(
            "plotly or kaleido not installed — skipping Optuna plot generation. "
            "Install with: pip install plotly kaleido"
        )
    except Exception:
        logger.warning("Failed to generate Optuna plots", exc_info=True)


def retrain_with_best(
    sweep_cfg: Config,
    best_params: dict[str, Any],
    gpus: int | None,
    output_dir: Path,
) -> None:
    """Retrain the model with best config on full train+val data.

    Args:
        sweep_cfg: The base sweep configuration.
        best_params: Best parameters from the study.
        gpus: Number of GPUs.
        output_dir: Directory for checkpoints and results.
    """
    logger.info("Retraining with best parameters on full train+val data...")

    cfg = apply_hpo_params(sweep_cfg, best_params)
    cfg_data = cfg.to_dict()

    retrain_cfg = sweep_cfg.hpo.retrain
    cfg_data["training"]["max_epochs"] = int(retrain_cfg.max_epochs)
    cfg_data["training"]["patience"] = int(retrain_cfg.patience)

    cfg_data["data"]["val_size"] = 0.0001
    cfg = Config(cfg_data)

    seed_everything(int(cfg.data.random_seed))

    datamodule = PathogenicityDataModule(cfg)
    datamodule.setup("fit")

    class_weights = None
    if datamodule._class_weights is not None:
        class_weights = torch.tensor(
            datamodule._class_weights, dtype=torch.float32,
        )

    model = PathogenicityPredictor.from_config(cfg)
    lightning_module = PathogenicityLightningModule(
        config=cfg, model=model, class_weights=class_weights,
    )

    checkpoint_dir = output_dir / "hpo_best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            filename="hpo_best-{epoch:02d}-{val_auroc:.4f}",
        ),
        EarlyStoppingWithPatience(
            patience=int(retrain_cfg.patience),
        ),
    ]

    accelerator = "auto"
    devices: int | str = "auto"
    if gpus is not None:
        accelerator = "gpu" if gpus > 0 else "cpu"
        devices = gpus if gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=int(retrain_cfg.max_epochs),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=False,
        deterministic=True,
        log_every_n_steps=10,
    )

    trainer.fit(lightning_module, datamodule=datamodule)

    datamodule.setup("test")
    test_results = trainer.test(
        lightning_module, datamodule=datamodule, ckpt_path="best",
    )

    if test_results:
        results_path = output_dir / "tables" / "hpo_test_metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as fh:
            json.dump(test_results[0], fh, indent=2)
        logger.info("Saved retrained model test metrics to %s", results_path)

    logger.info("Retraining complete.")


def main() -> None:
    """Run the full HPO pipeline."""
    args = parse_args()
    log = setup_logging(level="INFO", name=__name__)

    sweep_cfg = load_config(args.config, validate=True)
    hpo_cfg = sweep_cfg.hpo

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trials = args.n_trials or int(hpo_cfg.n_trials)
    timeout = args.timeout
    if timeout is None and hpo_cfg.get("timeout", None) is not None:
        timeout = int(hpo_cfg.timeout)

    study_name = hpo_cfg.study_name
    storage = hpo_cfg.storage

    storage_path = Path(storage.replace("sqlite:///", ""))
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    sampler: optuna.samplers.BaseSampler
    sampler_type = hpo_cfg.get("sampler", "tpe")
    if sampler_type == "tpe":
        sampler = optuna.samplers.TPESampler(seed=int(sweep_cfg.data.random_seed))
    elif sampler_type == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(sweep_cfg.data.random_seed))
    else:
        sampler = optuna.samplers.TPESampler(seed=int(sweep_cfg.data.random_seed))

    pruner: optuna.pruners.BasePruner
    pruner_type = hpo_cfg.get("pruner", "median")
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(hpo_cfg.get("pruner_n_startup_trials", 5)),
            n_warmup_steps=int(hpo_cfg.get("pruner_n_warmup_steps", 5)),
        )
    elif pruner_type == "percentile":
        pruner = optuna.pruners.PercentilePruner(percentile=25.0)
    else:
        pruner = optuna.pruners.MedianPruner()

    direction = hpo_cfg.get("direction", "maximize")

    log.info("Creating Optuna study '%s' with %d trials", study_name, n_trials)
    log.info("Storage: %s", storage)
    log.info("Sampler: %s | Pruner: %s | Direction: %s", sampler_type, pruner_type, direction)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )

    objective = create_objective(sweep_cfg, args.gpus)

    log.info("Starting optimisation...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    log.info("=" * 60)
    log.info("HPO COMPLETE")
    log.info("=" * 60)
    log.info("Best trial: %d", study.best_trial.number)
    log.info("Best val_auroc: %.4f", study.best_value)
    log.info("Best parameters:")
    for key, value in study.best_params.items():
        log.info("  %s: %s", key, value)

    best_config_path = Path("configs") / "best.yaml"
    save_best_config(sweep_cfg, study.best_params, best_config_path)

    study_summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "n_pruned": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]),
    }

    summary_path = output_dir / "tables" / "hpo_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(study_summary, fh, indent=2)
    log.info("Saved study summary to %s", summary_path)

    plots_dir = output_dir / "figures" / "hpo"
    generate_plots(study, plots_dir)

    do_retrain = args.retrain
    if not do_retrain and hasattr(hpo_cfg, "retrain"):
        do_retrain = bool(hpo_cfg.retrain.get("enabled", False))

    if do_retrain:
        retrain_with_best(sweep_cfg, study.best_params, args.gpus, output_dir)

    log.info("HPO pipeline complete.")


if __name__ == "__main__":
    main()
