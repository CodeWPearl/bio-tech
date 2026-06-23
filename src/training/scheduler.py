"""Learning rate scheduler factory.

Provides :func:`get_scheduler` which creates a PyTorch LR scheduler from the
project config, supporting cosine annealing with warm restarts,
reduce-on-plateau, and one-cycle policies.
"""

from __future__ import annotations

import logging
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)

from src.utils.config import Config

logger = logging.getLogger(__name__)


def get_scheduler(
    optimizer: Optimizer,
    config: Config,
) -> dict[str, Any]:
    """Build a learning rate scheduler from the project config.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        config: Project config; reads ``training.scheduler_type`` and related
            keys.

    Returns:
        A dict suitable for PyTorch Lightning's ``configure_optimizers``
        return value, containing ``"scheduler"``, ``"monitor"``, and
        ``"interval"`` keys.

    Raises:
        ValueError: If the scheduler type is unknown.
    """
    training_cfg = config.training
    scheduler_type: str = getattr(training_cfg, "scheduler_type", "cosine_warm_restarts")
    max_epochs: int = int(training_cfg.max_epochs)

    scheduler: LRScheduler

    if scheduler_type == "cosine_warm_restarts":
        t_0: int = int(getattr(training_cfg, "cosine_t0", 10))
        t_mult: int = int(getattr(training_cfg, "cosine_t_mult", 2))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult,
        )
        interval = "epoch"

    elif scheduler_type == "reduce_on_plateau":
        patience: int = int(getattr(training_cfg, "scheduler_patience", 5))
        factor: float = float(getattr(training_cfg, "scheduler_factor", 0.5))
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=factor,
        )
        interval = "epoch"

    elif scheduler_type == "one_cycle":
        max_lr: float = float(training_cfg.learning_rate)
        steps_per_epoch: int = int(getattr(training_cfg, "steps_per_epoch", 100))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        interval = "step"

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    logger.info("Using LR scheduler: %s", scheduler_type)

    return {
        "scheduler": scheduler,
        "monitor": "val_auroc",
        "interval": interval,
    }
