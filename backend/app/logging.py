"""Application logging configuration."""

import logging
import sys
from pathlib import Path

from .config import DATA_DIR, DEBUG


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    level = logging.DEBUG if DEBUG else logging.INFO

    logger = logging.getLogger("dds")
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logging()
