"""
Configuration management for KTND-Finance experiments.

Provides utilities for loading, merging, modifying, and saving YAML-based
configuration files with support for nested key access via dotted notation.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping, got {type(config).__name__}"
        )

    return config


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dictionary using dotted key notation.

    Creates intermediate dictionaries as needed. Modifies ``config`` in place.

    Parameters
    ----------
    config : dict
        The configuration dictionary to modify.
    dotted_key : str
        Dot-separated key path, e.g. ``'model.n_modes'``.
    value : Any
        The value to set at the specified path.

    Raises
    ------
    ValueError
        If ``dotted_key`` is empty.
    TypeError
        If an intermediate key maps to a non-dict value.

    Examples
    --------
    >>> d = {}
    >>> set_nested(d, 'model.n_modes', 5)
    >>> d
    {'model': {'n_modes': 5}}
    """
    if not dotted_key:
        raise ValueError("dotted_key must be a non-empty string")

    keys = dotted_key.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot traverse key '{key}' in '{dotted_key}': "
                f"existing value is {type(current[key]).__name__}, not dict"
            )
        current = current[key]

    current[keys[-1]] = value


def merge_configs(
    base: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge *overrides* into *base*, returning a new dictionary.

    - Dictionaries are merged recursively.
    - All other types in *overrides* replace the corresponding value in *base*.
    - Neither *base* nor *overrides* is mutated.

    Parameters
    ----------
    base : dict
        The base configuration dictionary.
    overrides : dict
        The override configuration dictionary whose values take precedence.

    Returns
    -------
    dict
        A new dictionary containing the merged configuration.
    """
    merged = copy.deepcopy(base)

    for key, override_value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(override_value, dict)
        ):
            merged[key] = merge_configs(merged[key], override_value)
        else:
            merged[key] = copy.deepcopy(override_value)

    return merged


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a configuration dictionary to a YAML file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    path : str or Path
        Destination file path (should end in ``.yaml`` or ``.yml``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
