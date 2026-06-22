"""Configuration loading with dot-access, CLI overrides, and validation.

This module provides a lightweight, dependency-minimal alternative to OmegaConf.
A YAML file is loaded into a :class:`Config` object that supports both attribute
(``cfg.model.fusion_dim``) and item (``cfg["model"]["fusion_dim"]``) access, can
be overridden from the command line, and is validated against the schema expected
by the rest of the project.

Example:
    >>> cfg = load_config("configs/default.yaml", overrides=["model.fusion_dim=512"])
    >>> cfg.model.fusion_dim
    512
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

# Fusion strategies supported by ``src/models/fusion``. Kept here so config
# validation fails fast on a typo rather than deep inside model construction.
VALID_FUSION_TYPES: tuple[str, ...] = (
    "early",
    "late",
    "attention",
    "cross_attention",
    "transformer",
)

# Top-level sections every config must define.
REQUIRED_SECTIONS: tuple[str, ...] = ("data", "model", "training", "experiment")


class Config:
    """Dict-backed configuration object with recursive dot-access.

    Nested mappings are wrapped as :class:`Config` instances on access, so
    ``cfg.model.dropout`` works the same as ``cfg["model"]["dropout"]``.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize from a plain (possibly nested) dictionary.

        Args:
            data: Mapping to wrap. ``None`` yields an empty config.
        """
        object.__setattr__(self, "_data", dict(data or {}))

    # -- access -----------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        """Return ``key`` via attribute access, wrapping nested dicts."""
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(f"No config key '{key}'") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        """Set ``key`` via attribute access."""
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        """Return ``key`` via item access, wrapping nested dicts."""
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set ``key`` via item access."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` is present at this level."""
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``key`` if present, otherwise ``default``."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying plain dictionary."""
        return _deep_copy(self._data)

    def __repr__(self) -> str:
        """Return a readable representation of the config contents."""
        return f"Config({self._data!r})"


def _deep_copy(obj: Any) -> Any:
    """Recursively copy nested dicts/lists of plain values."""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def _coerce(value: str) -> Any:
    """Cast a CLI override string to bool/int/float, falling back to str.

    Args:
        value: Raw right-hand side of a ``key=value`` override.

    Returns:
        The value parsed into the most specific matching Python type.
    """
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c = value`` in a nested dict, creating intermediate dicts.

    Args:
        data: Mutable nested dictionary to modify in place.
        dotted_key: Dot-separated key path, e.g. ``"model.fusion_dim"``.
        value: Already-coerced value to assign.
    """
    keys = dotted_key.split(".")
    cursor = data
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply a list of ``key.path=value`` overrides to a config dict.

    Args:
        data: Nested config dictionary (modified in place and returned).
        overrides: Iterable of ``"key.path=value"`` strings.

    Returns:
        The same ``data`` dictionary with overrides applied.

    Raises:
        ValueError: If an override is not in ``key=value`` form.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'; expected format 'key.path=value'"
            )
        dotted_key, _, raw_value = override.partition("=")
        _apply_override(data, dotted_key.strip(), _coerce(raw_value.strip()))
    return data


def validate_config(data: dict[str, Any]) -> None:
    """Validate the structure and key values of a config dictionary.

    Args:
        data: Nested config dictionary to check.

    Raises:
        ValueError: If a required section/key is missing or a value is invalid.
    """
    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise ValueError(f"Config missing required section '{section}'")

    model = data["model"]
    for key in ("num_classes", "fusion_dim", "fusion_type"):
        if key not in model:
            raise ValueError(f"Config missing required key 'model.{key}'")

    fusion_type = model["fusion_type"]
    if fusion_type not in VALID_FUSION_TYPES:
        raise ValueError(
            f"Invalid model.fusion_type '{fusion_type}'; "
            f"expected one of {VALID_FUSION_TYPES}"
        )

    num_classes = model["num_classes"]
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"model.num_classes must be an int >= 2, got {num_classes!r}")

    data_section = data["data"]
    for key in ("test_size", "val_size"):
        if key in data_section:
            fraction = data_section[key]
            if not (0.0 < float(fraction) < 1.0):
                raise ValueError(f"data.{key} must be in (0, 1), got {fraction!r}")
    if "test_size" in data_section and "val_size" in data_section:
        if float(data_section["test_size"]) + float(data_section["val_size"]) >= 1.0:
            raise ValueError("data.test_size + data.val_size must be < 1.0")


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    validate: bool = True,
) -> Config:
    """Load a YAML config, apply CLI overrides, validate, and wrap for dot-access.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional iterable of ``"key.path=value"`` CLI override strings.
        validate: Whether to run :func:`validate_config` after loading.

    Returns:
        A :class:`Config` exposing the merged configuration via dot-access.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If overrides are malformed or validation fails.
    """
    # Imported lazily so ``from src.utils.config import load_config`` succeeds
    # even before third-party dependencies are installed.
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    if overrides:
        apply_overrides(data, overrides)

    if validate:
        validate_config(data)

    return Config(data)
