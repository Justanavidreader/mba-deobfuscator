"""
Encoder registry for ablation study.

Provides factory function for dynamic encoder instantiation
without hard-coded if/else chains.
"""

from typing import Dict, Type

from src.models.encoder_base import BaseEncoder


# Lazy imports to avoid circular dependencies and optional dependency issues
def _get_encoder_classes() -> Dict[str, Type[BaseEncoder]]:
    """Get encoder classes with lazy imports."""
    from src.models.encoder import (
        GATJKNetEncoder,
        GGNNEncoder,
        HGTEncoder,
        RGCNEncoder,
    )
    from src.models.encoder_ablation import (
        HybridGREATEncoder,
        TransformerOnlyEncoder,
    )

    return {
        "gat_jknet": GATJKNetEncoder,
        "ggnn": GGNNEncoder,
        "hgt": HGTEncoder,
        "rgcn": RGCNEncoder,
        "transformer_only": TransformerOnlyEncoder,
        "hybrid_great": HybridGREATEncoder,
    }


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    """
    Factory function for encoder instantiation.

    Args:
        name: Encoder name (key in registry)
        **kwargs: Encoder-specific hyperparameters

    Returns:
        Instantiated encoder

    Raises:
        ValueError: If encoder name not registered
    """
    registry = _get_encoder_classes()

    if name not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown encoder: '{name}'. Available: {available}")

    encoder_cls = registry[name]
    return encoder_cls(**kwargs)


def list_encoders() -> Dict[str, Dict]:
    """
    List all available encoders with their properties.

    Returns:
        {encoder_name: {'requires_edge_types': bool, 'requires_node_features': bool}}
    """
    registry = _get_encoder_classes()
    result = {}

    for name, cls in registry.items():
        # Instantiate with minimal config to get properties
        try:
            encoder = cls(hidden_dim=32)  # Small for speed
            result[name] = {
                "requires_edge_types": encoder.requires_edge_types,
                "requires_node_features": encoder.requires_node_features,
                "class": cls.__name__,
            }
        except Exception as e:
            # Some encoders may fail without proper dependencies
            result[name] = {
                "requires_edge_types": "unknown",
                "requires_node_features": "unknown",
                "class": cls.__name__,
                "error": str(e),
            }

    return result


# Encoder groups for ablation study organization
HOMOGENEOUS_ENCODERS = ["gat_jknet", "transformer_only", "hybrid_great"]
HETEROGENEOUS_ENCODERS = ["ggnn", "hgt", "rgcn"]
SEQUENCE_ENCODERS = ["transformer_only"]
GRAPH_ENCODERS = ["gat_jknet", "ggnn", "hgt", "rgcn", "hybrid_great"]
