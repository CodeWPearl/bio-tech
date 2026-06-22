"""Centralised logging configuration.

Provides :func:`setup_logging`, which wires up the root logger with a timestamped
console handler and a rotating-friendly file handler under ``results/logs/``. All
modules should obtain loggers via ``logging.getLogger(__name__)`` and never call
``print()`` (see ``CLAUDE.md``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | Path = "results/logs",
    level: str | int = "INFO",
    name: str | None = None,
    log_to_file: bool = True,
) -> logging.Logger:
    """Configure console (and optionally file) logging and return a logger.

    Args:
        log_dir: Directory in which to write the timestamped log file.
        level: Logging level as a name (``"INFO"``) or numeric value.
        name: Name of the logger to return. ``None`` returns the root logger.
        log_to_file: If ``True``, also write logs to a timestamped file in
            ``log_dir``; the directory is created if it does not exist.

    Returns:
        The configured :class:`logging.Logger`.
    """
    numeric_level = (
        logging.getLevelName(level.upper()) if isinstance(level, str) else level
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove pre-existing handlers so repeated calls don't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"run_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logger = logging.getLogger(name)
    logger.info("Logging initialised at level %s", logging.getLevelName(numeric_level))
    return logger
