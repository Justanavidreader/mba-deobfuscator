"""
Inference modules for MBA deobfuscation.

Provides:
- GrammarConstraint: Grammar-based token masking
- BeamSearchDecoder: Beam search with copy mechanism
- ThreeTierVerifier: Syntax → Execution → Z3 verification
- MinimalHTPS: HyperTree Proof Search for deep expressions
- CandidateReranker: Rerank by simplicity metrics
- InferencePipeline: End-to-end simplification pipeline
"""

from src.inference.grammar import GrammarConstraint
from src.inference.beam_search import BeamSearchDecoder, BeamHypothesis
from src.inference.verify import ThreeTierVerifier, VerificationResult
from src.inference.htps import MinimalHTPS, Tactic, HyperNode, HyperEdge
from src.inference.rerank import CandidateReranker, RankingFeatures
from src.inference.pipeline import InferencePipeline, SimplificationResult

__all__ = [
    'GrammarConstraint',
    'BeamSearchDecoder',
    'BeamHypothesis',
    'ThreeTierVerifier',
    'VerificationResult',
    'MinimalHTPS',
    'Tactic',
    'HyperNode',
    'HyperEdge',
    'CandidateReranker',
    'RankingFeatures',
    'InferencePipeline',
    'SimplificationResult',
]
