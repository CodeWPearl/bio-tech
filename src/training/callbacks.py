"""Custom PyTorch Lightning callbacks for training monitoring.

Provides :class:`MetricLogger` for MLflow metric logging,
:class:`GradientMonitor` for gradient norm tracking, and
:class:`EarlyStoppingWithPatience` for val_auroc-based early stopping.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping

logger = logging.getLogger(__name__)


class MetricLogger(Callback):
    """Log all metrics to MLflow at each validation epoch end.

    Reads the metrics already logged by the Lightning module via
    ``self.log()`` and forwards them to the active MLflow run (if any).
    """

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        """Forward logged metrics to MLflow.

        Args:
            trainer: The current trainer instance.
            pl_module: The Lightning module.
        """
        metrics = trainer.callback_metrics
        step = trainer.current_epoch

        try:
            import mlflow

            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                mlflow.log_metric(key, value, step=step)
        except ImportError:
            logger.debug("mlflow not available — skipping metric logging")
        except Exception:
            logger.warning("Failed to log metrics to MLflow", exc_info=True)


class GradientMonitor(Callback):
    """Log gradient norms per layer to detect exploding/vanishing gradients.

    Computes the L2 norm of gradients for each named parameter after each
    training batch and logs them via the Lightning logger.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        """Initialize the gradient monitor.

        Args:
            log_every_n_steps: How often (in training steps) to log norms.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log gradient norms after the backward pass.

        Args:
            trainer: The current trainer instance.
            pl_module: The Lightning module.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                pl_module.log(
                    f"grad_norm/{name}", param_norm,
                    on_step=True, on_epoch=False, logger=True,
                )

        total_norm = total_norm ** 0.5
        pl_module.log(
            "grad_norm/total", total_norm,
            on_step=True, on_epoch=False, logger=True,
        )

        if total_norm > 100.0:
            logger.warning(
                "Exploding gradients detected: total_norm=%.2f at step %d",
                total_norm, trainer.global_step,
            )
        elif total_norm < 1e-7:
            logger.warning(
                "Vanishing gradients detected: total_norm=%.2e at step %d",
                total_norm, trainer.global_step,
            )


class EarlyStoppingWithPatience(EarlyStopping):
    """Early stopping on ``val_auroc`` with logging.

    Wraps Lightning's :class:`EarlyStopping` with additional logging when
    patience is exhausted or a new best score is reached.

    Args:
        patience: Number of validation epochs with no improvement before
            stopping.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001) -> None:
        super().__init__(
            monitor="val_auroc",
            mode="max",
            patience=patience,
            min_delta=min_delta,
            verbose=True,
        )

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> None:
        """Log early stopping state after each validation.

        Args:
            trainer: The current trainer instance.
            pl_module: The Lightning module.
        """
        super().on_validation_end(trainer, pl_module)
        logger.info(
            "EarlyStopping: wait_count=%d/%d, best_score=%.4f",
            self.wait_count, self.patience, self.best_score.item(),
        )
