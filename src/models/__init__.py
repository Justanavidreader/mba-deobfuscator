"""Model architecture modules for MBA Deobfuscator."""

from src.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from src.models.heads import TokenHead, ComplexityHead, ValueHead
from src.models.edge_types import EdgeType, NodeType, NODE_TYPE_MAP
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
]
