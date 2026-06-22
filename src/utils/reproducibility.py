"""Reproducibility helpers: deterministic seeding across all RNG sources.

Every experiment should call :func:`seed_everything` once at startup so that
runs are bit-for-bit reproducible given the same seed, hardware, and library
versions. The chosen seed is logged alongside the config to MLflow elsewhere.
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: Integer seed applied to every random number generator.
        deterministic: If ``True``, force cuDNN into deterministic mode
            (``cudnn.deterministic = True`` and ``cudnn.benchmark = False``).
            This trades some GPU throughput for exact reproducibility.

    Returns:
        The seed that was applied (useful for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy and PyTorch are imported lazily so this module can be imported
    # before third-party dependencies are installed.
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency at runtime
        logger.warning("numpy not installed; skipping numpy seeding")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency at runtime
        logger.warning("torch not installed; skipping torch seeding")

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
    return seed
