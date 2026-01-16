"""
Graph Matching Networks (GMN) for MBA equivalence detection.

GMN enhances Siamese encoders by introducing cross-graph attention,
enabling explicit node correspondence detection between expression graphs.
"""

from src.models.gmn.cross_attention import (
    CrossGraphAttention,
    MultiHeadCrossGraphAttention,
)
from src.models.gmn.graph_matching import GraphMatchingNetwork
from src.models.gmn.gmn_encoder_wrapper import HGTWithGMN
from src.models.gmn.batch_collator import GMNBatchCollator

__all__ = [
    'CrossGraphAttention',
    'MultiHeadCrossGraphAttention',
    'GraphMatchingNetwork',
    'HGTWithGMN',
    'GMNBatchCollator',
]
