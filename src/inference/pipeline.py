"""
Complete inference pipeline for MBA deobfuscation.

Orchestrates: encode → generate → verify → rerank.
"""

from typing import List, Tuple, Optional, Dict
import torch
import time
from dataclasses import dataclass
import numpy as np

from src.constants import (
    HTPS_DEPTH_THRESHOLD, BEAM_WIDTH, MAX_SEQ_LEN
)
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.ast_parser import expr_to_graph, expr_to_ast_depth
from src.data.fingerprint import SemanticFingerprint
from src.inference.beam_search import BeamSearchDecoder
from src.inference.htps import MinimalHTPS
from src.inference.verify import ThreeTierVerifier, VerificationResult
from src.inference.rerank import CandidateReranker
from src.inference.grammar import GrammarConstraint
import torch_geometric.data as pyg_data


@dataclass
class SimplificationResult:
    """
    Result of simplification pipeline.
    """
    input_expr: str                          # Original expression
    output_expr: str                         # Best simplified expression
    candidates: List[str]                    # All candidates generated
    verification: VerificationResult         # Verification for output
    proof_trace: Optional[List[str]] = None  # HTPS proof trace if used
    elapsed_ms: float = 0.0                  # Total inference time
    method: str = "beam"                     # "beam" or "htps"


class InferencePipeline:
    """
    Complete inference pipeline for MBA deobfuscation.
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        tokenizer: MBATokenizer,
        device: torch.device = None,
        beam_width: int = BEAM_WIDTH,
        use_htps_for_deep: bool = True,
        htps_depth_threshold: int = HTPS_DEPTH_THRESHOLD,
        use_grammar: bool = True,
        verify_output: bool = True
    ):
        """
        Initialize inference pipeline.

        Args:
            model: Trained MBADeobfuscator
            tokenizer: MBATokenizer instance
            device: Device for inference (cpu/cuda)
            beam_width: Beam width for beam search
            use_htps_for_deep: Use HTPS for expressions with depth ≥ threshold
            htps_depth_threshold: Depth threshold for HTPS (default: 10)
            use_grammar: Apply grammar constraints
            verify_output: Run verification on output
        """
        self.device = device or torch.device('cpu')
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

        # CRITICAL FIX: Use SemanticFingerprint class
        self.fingerprint_computer = SemanticFingerprint()

        # Initialize components
        self.grammar = GrammarConstraint(tokenizer) if use_grammar else None
        self.beam_decoder = BeamSearchDecoder(
            model, tokenizer, beam_width=beam_width, use_grammar=use_grammar
        )
        self.verifier = ThreeTierVerifier(tokenizer)
        self.reranker = CandidateReranker(tokenizer)

        # HTPS with circular dependency resolution
        self.htps = MinimalHTPS(model, tokenizer, verifier=None)
        self.htps.verifier = self.verifier  # Resolve circular dependency

        self.use_htps_for_deep = use_htps_for_deep
        self.htps_depth_threshold = htps_depth_threshold
        self.verify_output = verify_output

    def simplify(
        self,
        expr: str,
        return_all_candidates: bool = False
    ) -> SimplificationResult:
        """
        Simplify MBA expression.

        Args:
            expr: Input obfuscated expression
            return_all_candidates: If True, include all candidates in result

        Returns:
            SimplificationResult with best simplification

        Pipeline:
            1. Parse and compute fingerprint
            2. Encode with model.encode()
            3. Choose beam search or HTPS based on depth
            4. Generate candidates
            5. Verify candidates (3-tier)
            6. Rerank by simplicity + confidence
            7. Return top result
        """
        start_time = time.time()

        try:
            # Preprocess
            graph_batch, fingerprint, ast_depth = self._preprocess(expr)

            # Encode
            with torch.no_grad():
                memory = self.model.encode(graph_batch, fingerprint)

            # Choose method based on depth
            if self.use_htps_for_deep and ast_depth >= self.htps_depth_threshold:
                method = "htps"
            else:
                method = "beam"

            # Generate candidates
            candidates, scores, proof_trace = self._generate_candidates(expr, memory, method)

            # If no candidates, return input
            if not candidates:
                elapsed = (time.time() - start_time) * 1000
                return SimplificationResult(
                    input_expr=expr,
                    output_expr=expr,
                    candidates=[expr],
                    verification=VerificationResult(
                        candidate=expr,
                        syntax_valid=True,
                        exec_valid=False,
                        z3_verified=False
                    ),
                    elapsed_ms=elapsed,
                    method=method
                )

            # Verify and rerank
            best_candidate, verification = self._postprocess(expr, candidates, scores)

            elapsed = (time.time() - start_time) * 1000

            return SimplificationResult(
                input_expr=expr,
                output_expr=best_candidate,
                candidates=candidates if return_all_candidates else [best_candidate],
                verification=verification,
                proof_trace=proof_trace,
                elapsed_ms=elapsed,
                method=method
            )

        except Exception as e:
            # Error occurred - return input unchanged
            elapsed = (time.time() - start_time) * 1000
            return SimplificationResult(
                input_expr=expr,
                output_expr=expr,
                candidates=[expr],
                verification=VerificationResult(
                    candidate=expr,
                    syntax_valid=False,
                    exec_valid=False,
                    z3_verified=False
                ),
                elapsed_ms=elapsed,
                method="error"
            )

    def _preprocess(self, expr: str) -> Tuple[pyg_data.Batch, torch.Tensor, int]:
        """
        Preprocess expression for model.

        Args:
            expr: Expression string

        Returns:
            (graph_batch, fingerprint, ast_depth)
            - graph_batch: PyG Batch for encoder
            - fingerprint: [1, 448] tensor
            - ast_depth: AST depth (for choosing beam vs HTPS)
        """
        # CRITICAL FIX: Use expr_to_graph for graphs
        graph = expr_to_graph(expr)
        graph_batch = pyg_data.Batch.from_data_list([graph]).to(self.device)

        # CRITICAL FIX: Use SemanticFingerprint class
        fingerprint_np = self.fingerprint_computer.compute(expr)
        fingerprint = torch.from_numpy(fingerprint_np).float().unsqueeze(0).to(self.device)

        # CRITICAL FIX: Use expr_to_ast_depth for depth
        ast_depth = expr_to_ast_depth(expr)

        return graph_batch, fingerprint, ast_depth

    def _generate_candidates(
        self,
        expr: str,
        memory: torch.Tensor,
        method: str
    ) -> Tuple[List[str], List[float], Optional[List[str]]]:
        """
        Generate candidates with beam search or HTPS.

        Args:
            expr: Input expression
            memory: [1, 1, D_MODEL] encoder output
            method: "beam" or "htps"

        Returns:
            (candidates, scores, proof_trace)
            - candidates: List of expression strings
            - scores: Model log probabilities
            - proof_trace: HTPS proof if method="htps", else None
        """
        if method == "htps":
            # HTPS search
            best_expr, proof_trace = self.htps.search(expr)
            return ([best_expr], [0.0], proof_trace)
        else:
            # Beam search
            src_tokens = self.tokenizer.get_source_tokens(expr)
            beams = self.beam_decoder.decode(memory, src_tokens=src_tokens)

            candidates = []
            scores = []

            for beam in beams:
                # Decode tokens
                candidate = self.tokenizer.decode(beam.tokens, skip_special=True)
                candidates.append(candidate)
                scores.append(beam.normalized_score)

            return (candidates, scores, None)

    def _postprocess(
        self,
        input_expr: str,
        candidates: List[str],
        scores: List[float]
    ) -> Tuple[str, VerificationResult]:
        """
        Verify and rerank candidates.

        Args:
            input_expr: Original expression
            candidates: Generated candidates
            scores: Model scores

        Returns:
            (best_candidate, verification_result)
        """
        if not self.verify_output:
            # No verification - return first candidate
            return (candidates[0], VerificationResult(
                candidate=candidates[0],
                syntax_valid=True,
                exec_valid=False,
                z3_verified=False
            ))

        # Verify all candidates
        verifications = self.verifier.verify_batch(input_expr, candidates)

        # Create verification map
        verification_map = {v.candidate: v for v in verifications}

        # Align verifications with candidates
        aligned_verifications = []
        for candidate in candidates:
            if candidate in verification_map:
                aligned_verifications.append(verification_map[candidate])
            else:
                # Create dummy verification
                aligned_verifications.append(VerificationResult(
                    candidate=candidate,
                    syntax_valid=False,
                    exec_valid=False,
                    z3_verified=False
                ))

        # Rerank
        ranked = self.reranker.rerank(candidates, scores, aligned_verifications)

        if ranked:
            best_candidate = ranked[0][0]
            verification = verification_map.get(best_candidate, aligned_verifications[0])
            return (best_candidate, verification)
        else:
            # No valid candidates - return input
            return (input_expr, VerificationResult(
                candidate=input_expr,
                syntax_valid=True,
                exec_valid=False,
                z3_verified=False
            ))

    def batch_simplify(self, exprs: List[str]) -> List[SimplificationResult]:
        """
        Batch simplification for multiple expressions.

        Args:
            exprs: List of input expressions

        Returns:
            List of SimplificationResults

        Note: Currently processes sequentially. Batching within beam search
              and HTPS could be added for further speedup.
        """
        results = []
        for expr in exprs:
            result = self.simplify(expr)
            results.append(result)
        return results
