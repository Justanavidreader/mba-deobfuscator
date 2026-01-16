"""
Logging setup with optional WandB integration.

Provides utilities for setting up Python logging and Weights & Biases
experiment tracking.
"""

import logging
import sys
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up logging with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_wandb(
    project: str,
    config: Dict[str, Any],
    enabled: bool = True,
    name: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None
) -> Optional[Any]:
    """
    Initialize Weights & Biases experiment tracking.

    Args:
        project: WandB project name
        config: Configuration dictionary to log
        enabled: Whether to enable WandB (allows dry runs)
        name: Run name (optional, WandB generates if None)
        tags: List of tags for the run
        notes: Text notes for the run

    Returns:
        WandB run object if enabled and available, else None
    """
    if not enabled:
        logging.getLogger(__name__).info("WandB disabled by configuration")
        return None

    if not WANDB_AVAILABLE:
        logging.getLogger(__name__).warning(
            "WandB not installed. Install with: pip install wandb"
        )
        return None

    run = wandb.init(
        project=project,
        config=config,
        name=name,
        tags=tags,
        notes=notes
    )

    logging.getLogger(__name__).info(f"WandB initialized: {wandb.run.url}")
    return run
