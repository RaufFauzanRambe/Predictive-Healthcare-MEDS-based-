"""
logger.py - Centralized logging utility for the project.

Provides a consistent logging interface across all modules with both
console and file output support, configurable from config.yaml.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Loggers are created with a standardized format that includes timestamp,
    module name, and log level. Supports both console and file output.
    Subsequent calls with the same name return the same logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Optional file path for file logging.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_logging_from_config(config: dict) -> None:
    """
    Configure the root logger based on project config.yaml settings.

    This should be called once at application startup to set up the
    global logging configuration that all module-level loggers inherit.

    Args:
        config: Global configuration dictionary (contents of config.yaml).
    """
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_format = log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    date_format = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console handler
    if log_cfg.get("console_logging", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_cfg.get("file_logging", True):
        log_file = log_cfg.get("log_file", "results/logs/pipeline.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
