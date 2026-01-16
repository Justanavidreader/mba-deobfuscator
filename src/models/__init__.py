"""Model architecture modules for MBA Deobfuscator."""

from src.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from src.models.heads import TokenHead, ComplexityHead, ValueHead
from src.models.edge_types import EdgeType, NodeType, NODE_TYPE_MAP
from src.models.operation_aware_aggregator import OperationAwareAggregator

# Encoder/decoder imports require torch_geometric/torch_scatter - optional for basic usage
try:
    from src.models.encoder import (
        GATJKNetEncoder,
        GGNNEncoder,
        GraphReadout,
        FingerprintEncoder,
        HGTEncoder,
        RGCNEncoder,
        ScaledGraphReadout,
    )
    from src.models.decoder import TransformerDecoderLayer, TransformerDecoderWithCopy
    from src.models.full_model import MBADeobfuscator, ScaledMBADeobfuscator
    _HAS_FULL_MODELS = True
except ImportError:
    _HAS_FULL_MODELS = False

__all__ = [
    # Positional encodings
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    # Output heads
    'TokenHead',
    'ComplexityHead',
    'ValueHead',
    # Edge/node types
    'EdgeType',
    'NodeType',
    'NODE_TYPE_MAP',
    # Operation-aware aggregation
    'OperationAwareAggregator',
]

if _HAS_FULL_MODELS:
    __all__.extend([
        # Encoders
        'GATJKNetEncoder',
        'GGNNEncoder',
        'GraphReadout',
        'FingerprintEncoder',
        'HGTEncoder',
        'RGCNEncoder',
        'ScaledGraphReadout',
        # Decoder
        'TransformerDecoderLayer',
        'TransformerDecoderWithCopy',
        # Full models
        'MBADeobfuscator',
        'ScaledMBADeobfuscator',
    ])
