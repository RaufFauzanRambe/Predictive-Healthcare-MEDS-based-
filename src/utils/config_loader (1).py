"""
config_loader.py - YAML configuration file loader.

Provides a centralized interface for loading and caching YAML configuration
files. Supports nested key access, environment variable overrides, and
configuration merging for flexible project setup.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory cache to avoid re-reading files
_config_cache: dict[str, dict] = {}


def load_config(filepath: str, use_cache: bool = True) -> dict:
    """
    Load a YAML configuration file.

    Reads the specified YAML file and returns its contents as a dictionary.
    Results are cached in memory so repeated calls for the same file do not
    incur disk I/O. Environment variable overrides are applied using the
    pattern ${ENV_VAR} in YAML values.

    Args:
        filepath: Path to the YAML configuration file.
        use_cache: Whether to use the in-memory cache.

    Returns:
        Dictionary with the configuration contents.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    # Resolve path relative to project root
    resolved_path = _resolve_path(filepath)

    # Check cache
    cache_key = str(resolved_path)
    if use_cache and cache_key in _config_cache:
        return _config_cache[cache_key]

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Cache the result
    if use_cache:
        _config_cache[cache_key] = config

    logger.debug(f"Loaded config from {resolved_path}")
    return config


def _resolve_path(filepath: str) -> Path:
    """
    Resolve a config file path, checking project root directories.

    Searches for the config file in:
    1. The exact path as given
    2. Relative to the current working directory
    3. Relative to the project root (parent of src/)

    Args:
        filepath: Raw file path string.

    Returns:
        Resolved absolute Path.
    """
    path = Path(filepath)

    if path.exists():
        return path.resolve()

    # Try relative to current working directory
    cwd_path = Path.cwd() / filepath
    if cwd_path.exists():
        return cwd_path.resolve()

    # Try relative to project root
    project_root = Path(__file__).parent.parent.parent
    root_path = project_root / filepath
    if root_path.exists():
        return root_path.resolve()

    # Return the original path (will raise FileNotFoundError later)
    return path.resolve()


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to configuration values.

    Any string value in the config that matches ${VAR_NAME} will be
    replaced with the value of the corresponding environment variable.
    If the environment variable is not set, the original value is kept.

    Args:
        config: Configuration dictionary to process.

    Returns:
        Configuration with environment variables substituted.
    """
    if isinstance(config, dict):
        return {k: _apply_env_overrides(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_apply_env_overrides(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return env_value
        logger.warning(f"Environment variable ${env_var} not set, using config default")
        return config

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge two configuration dictionaries.

    Values in 'override' take precedence over 'base'. Nested dictionaries
    are merged recursively rather than replaced entirely.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def clear_cache() -> None:
    """Clear the configuration cache, forcing re-reads on next load."""
    _config_cache.clear()
    logger.debug("Configuration cache cleared")
