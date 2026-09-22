"""Logging configuration utility for the dynamic pricing pipeline."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standardized console logger.

    Parameters
    ----------
    name : Optional[str]
        Logger name (usually __name__). If None, configures the root logger.
    level : int
        Logging severity level (default: logging.INFO).
    log_format : Optional[str]
        Format string for log records.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def configure_app_logging(level: int = logging.WARNING) -> None:
    """Configure logging levels across internal modules and external libraries to silence console spam."""
    # 1. Silence noisy third-party libraries (cmdstanpy, prophet, matplotlib, numba, etc.)
    noisy_libraries = [
        "cmdstanpy",
        "prophet",
        "numba",
        "matplotlib",
        "urllib3",
        "streamlit",
        "tornado",
        "asyncio",
    ]
    for lib in noisy_libraries:
        lgr = logging.getLogger(lib)
        lgr.setLevel(logging.ERROR)
        lgr.propagate = False

    # Specifically silence cmdstanpy internal logger handlers
    try:
        import cmdstanpy
        c_logger = cmdstanpy.utils.get_logger()
        c_logger.setLevel(logging.ERROR)
        c_logger.disabled = True
        for h in c_logger.handlers:
            h.setLevel(logging.ERROR)
    except Exception:
        pass

    # 2. Silence verbose routine INFO logs from internal pipeline modules
    internal_modules = [
        "src",
        "src.data.loader",
        "src.data.validator",
        "src.preprocessing.preprocessing",
        "src.models.model",
        "src.models.baseline",
        "src.models.elasticity",
        "src.optimization.optimizer",
        "src.simulation.simulator",
        "src.evaluation.comparison",
    ]
    for mod in internal_modules:
        lgr = logging.getLogger(mod)
        lgr.setLevel(level)

    # 3. Configure root logger
    logging.getLogger().setLevel(level)
