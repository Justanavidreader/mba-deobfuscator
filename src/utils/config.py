"""
Configuration loading from YAML files.

Provides a Config class for loading and accessing YAML configurations
with dot notation support, along with validation utilities.
"""

from pathlib import Path
from typing import Any, Optional, Type

import yaml


def validate_config_value(
    config: dict,
    path: str,
    expected_type: Type,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Extract and validate config value with type checking.

    Args:
        config: Configuration dictionary
        path: Dot-separated path to value (e.g., "model.hidden_dim")
        expected_type: Expected Python type
        default: Default value if not found (ignored if required=True)
        required: If True, raise error when value missing

    Returns:
        Config value cast to expected_type

    Raises:
        ValueError: If required value missing or type mismatch

    Example:
        >>> config = {"model": {"hidden_dim": 256}}
        >>> validate_config_value(config, "model.hidden_dim", int)
        256
    """
    keys = path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
    except (KeyError, TypeError):
        if required:
            raise ValueError(f"Required config key '{path}' not found")
        return default

    if not isinstance(value, expected_type):
        raise ValueError(
            f"Config key '{path}' has wrong type. "
            f"Expected {expected_type.__name__}, got {type(value).__name__}: {value}"
        )

    return value


def create_encoder_from_config(config: dict):
    """
    Create encoder instance from configuration dict.

    Args:
        config: Full config dict with 'model' section

    Returns:
        Encoder instance (HGTEncoder, GGNNEncoder, GATJKNetEncoder, RGCNEncoder)

    Raises:
        ValueError: If config invalid or encoder type unsupported
    """
    from src.models.encoder import HGTEncoder, GGNNEncoder, GATJKNetEncoder, RGCNEncoder

    encoder_type = validate_config_value(
        config, "model.encoder_type", str, required=True
    )

    # Base parameters common to most encoders
    hidden_dim = validate_config_value(config, "model.hidden_dim", int, required=True)
    num_layers = validate_config_value(config, "model.num_encoder_layers", int, required=True)
    dropout = validate_config_value(config, "model.encoder_dropout", float, default=0.1)

    if encoder_type == 'hgt':
        hgt_params = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': validate_config_value(config, "model.num_encoder_heads", int, required=True),
            'dropout': dropout,
            'edge_type_mode': validate_config_value(config, "model.edge_type_mode", str, default="optimized"),
            'use_global_attention': validate_config_value(config, "model.use_global_attention", bool, default=False),
            'global_attn_interval': validate_config_value(config, "model.global_attn_interval", int, default=2),
            'global_attn_heads': validate_config_value(config, "model.global_attn_heads", int, default=8),
            'operation_aware': validate_config_value(config, "model.operation_aware", bool, default=False),
            'operation_aware_strict': validate_config_value(config, "model.operation_aware_strict", bool, default=True),
        }

        # Path encoding params if enabled
        if validate_config_value(config, "model.use_path_encoding", bool, default=False):
            hgt_params['use_path_encoding'] = True
            hgt_params['path_max_length'] = validate_config_value(config, "model.path_max_length", int, default=6)
            hgt_params['path_max_paths'] = validate_config_value(config, "model.path_max_paths", int, default=16)
            hgt_params['path_injection_interval'] = validate_config_value(config, "model.path_injection_interval", int, default=2)

        return HGTEncoder(**hgt_params)

    elif encoder_type == 'ggnn':
        ggnn_params = {
            'hidden_dim': hidden_dim,
            'num_timesteps': validate_config_value(config, "model.num_timesteps", int, default=8),
            'edge_type_mode': validate_config_value(config, "model.edge_type_mode", str, default="legacy"),
        }

        if validate_config_value(config, "model.use_path_encoding", bool, default=False):
            ggnn_params['use_path_encoding'] = True
            ggnn_params['path_max_length'] = validate_config_value(config, "model.path_max_length", int, default=6)

        return GGNNEncoder(**ggnn_params)

    elif encoder_type in ('gat', 'gat_jknet'):
        gat_params = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': validate_config_value(config, "model.num_encoder_heads", int, required=True),
            'dropout': dropout,
        }
        return GATJKNetEncoder(**gat_params)

    elif encoder_type == 'rgcn':
        rgcn_params = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'edge_type_mode': validate_config_value(config, "model.edge_type_mode", str, default="optimized"),
        }
        return RGCNEncoder(**rgcn_params)

    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


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
