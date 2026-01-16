"""
Configuration loading from YAML files.

Provides a Config class for loading and accessing YAML configurations
with dot notation support.
"""

from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """Load and access YAML config with dot notation."""

    def __init__(self, path: str):
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, 'r') as f:
            self._data = yaml.safe_load(f)

        if self._data is None:
            self._data = {}

    def __getattr__(self, key: str) -> Any:
        """
        Access config values using dot notation.

        Args:
            key: Configuration key

        Returns:
            Configuration value, wrapped in Config if dict

        Raises:
            AttributeError: If key doesn't exist
        """
        if key.startswith('_'):
            raise AttributeError(f"Cannot access private attribute: {key}")

        if key not in self._data:
            raise AttributeError(f"Config has no attribute: {key}")

        value = self._data[key]
        if isinstance(value, dict):
            return Config._wrap_dict(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with default fallback.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        if isinstance(value, dict):
            return Config._wrap_dict(value)
        return value

    @staticmethod
    def _wrap_dict(data: dict) -> 'Config':
        """Wrap dictionary in Config object for dot notation access."""
        config = Config.__new__(Config)
        config._data = data
        return config

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self._data.copy()
