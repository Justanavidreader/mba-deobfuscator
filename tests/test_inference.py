"""
Comprehensive tests for inference modules.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.data.tokenizer import MBATokenizer
from src.constants import SOS_IDX, EOS_IDX, PAD_IDX
from src.inference.grammar import GrammarConstraint
from src.inference.beam_search import BeamSearchDecoder, BeamHypothesis
from src.inference.verify import ThreeTierVerifier, VerificationResult
from src.inference.htps import MinimalHTPS, Tactic
from src.inference.rerank import CandidateReranker, RankingFeatures
from src.inference.pipeline import InferencePipeline, SimplificationResult


@pytest.fixture
def tokenizer():
    """Create tokenizer instance."""
    return MBATokenizer()


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock()
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)

    # Mock encode
    def mock_encode(graph_batch, fingerprint):
        batch_size = fingerprint.size(0)
        return torch.randn(batch_size, 1, 512)
    model.encode = Mock(side_effect=mock_encode)

    # Mock decode
    def mock_decode(tgt, memory, tgt_mask=None, memory_mask=None):
        batch_size, seq_len = tgt.shape
        vocab_size = 300
        return {
            'vocab_logits': torch.randn(batch_size, seq_len, vocab_size),
            'copy_attn': torch.rand(batch_size, seq_len, 1),
            'p_gen': torch.rand(batch_size, seq_len, 1)
        }
    model.decode = Mock(side_effect=mock_decode)

    # Mock get_value
    def mock_get_value(graph_batch, fingerprint):
        batch_size = fingerprint.size(0)
        return torch.rand(batch_size, 1)
    model.get_value = Mock(side_effect=mock_get_value)

    # Mock parameters for device detection
    param = torch.nn.Parameter(torch.tensor([1.0]))
    model.parameters = Mock(return_value=[param])

    return model


# ============================================================================
# Grammar Tests
# ============================================================================

class TestGrammarConstraint:
    """Tests for GrammarConstraint."""

    def test_initialization(self, tokenizer):
        """Test grammar constraint initialization."""
        grammar = GrammarConstraint(tokenizer)
        assert grammar.tokenizer == tokenizer
        assert grammar.parser is not None
        assert isinstance(grammar.valid_tokens_cache, dict)

    def test_parse_check_valid(self, tokenizer):
        """Test syntax check for valid expressions."""
        grammar = GrammarConstraint(tokenizer)

        valid_exprs = [
            "x + y",
            "(x & y) + 1",
            "~(x | y)",
            "x * 2 + y"
        ]

        for expr in valid_exprs:
            is_valid, error = grammar.parse_check(expr)
            assert is_valid, f"Expression '{expr}' should be valid, got error: {error}"

    def test_parse_check_invalid(self, tokenizer):
        """Test syntax check for invalid expressions."""
        grammar = GrammarConstraint(tokenizer)

        invalid_exprs = [
            "x &",      # Incomplete
            "& x",      # Invalid start
            "((x + y)", # Unbalanced parens
        ]

        for expr in invalid_exprs:
            is_valid, error = grammar.parse_check(expr)
            assert not is_valid, f"Expression '{expr}' should be invalid"

    def test_get_valid_tokens(self, tokenizer):
        """Test valid token retrieval."""
        grammar = GrammarConstraint(tokenizer)

        # After SOS, should allow variables, constants, parens, unary ops
        valid = grammar.get_valid_tokens([SOS_IDX])
        assert len(valid) > 0

        # After variable, should allow operators, EOS
        var_token = tokenizer.token2id.get('x', tokenizer.token2id.get('x0'))
        valid = grammar.get_valid_tokens([SOS_IDX, var_token])
        assert len(valid) > 0

    def test_mask_logits(self, tokenizer):
        """Test logit masking."""
        grammar = GrammarConstraint(tokenizer)

        logits = torch.randn(300)
        valid_tokens = {1, 2, 3, 10, 20}

        masked = grammar.mask_logits(logits, valid_tokens)

        # Valid tokens should be unchanged or penalized less
        assert not torch.isinf(masked[1])
        assert not torch.isinf(masked[2])

        # Invalid tokens should be -inf
        invalid_idx = 50
        if invalid_idx not in valid_tokens:
            assert torch.isinf(masked[invalid_idx]) and masked[invalid_idx] < 0


# ============================================================================
# Beam Search Tests
# ============================================================================

class TestBeamSearch:
    """Tests for BeamSearchDecoder."""

    def test_beam_hypothesis_normalized_score(self):
        """Test normalized score computation."""
        # Normal case
        hyp = BeamHypothesis(tokens=[1, 2, 3], score=-5.0, length=3)
        norm_score = hyp.normalized_score
        expected = -5.0 / (3 ** 0.6)
        assert abs(norm_score - expected) < 1e-6

        # CRITICAL FIX: Zero length case
        hyp_zero = BeamHypothesis(tokens=[], score=-2.0, length=0)
        assert hyp_zero.normalized_score == -2.0  # Should not divide by zero

    def test_beam_search_initialization(self, mock_model, tokenizer):
        """Test beam search decoder initialization."""
        decoder = BeamSearchDecoder(mock_model, tokenizer, beam_width=10)
        assert decoder.beam_width == 10
        assert decoder.model == mock_model
        assert decoder.tokenizer == tokenizer

    def test_beam_search_decode(self, mock_model, tokenizer):
        """Test beam search decoding."""
        decoder = BeamSearchDecoder(mock_model, tokenizer, beam_width=5, max_length=10)

        # Mock memory
        memory = torch.randn(1, 1, 512)

        # Decode
        beams = decoder.decode(memory)

        assert len(beams) > 0
        assert len(beams) <= 5
        assert all(isinstance(b, BeamHypothesis) for b in beams)

    def test_expand_beams(self, mock_model, tokenizer):
        """Test beam expansion."""
        decoder = BeamSearchDecoder(mock_model, tokenizer, beam_width=5)

        # Create initial beams
        beams = [
            BeamHypothesis(tokens=[SOS_IDX], score=0.0, length=0),
            BeamHypothesis(tokens=[SOS_IDX], score=-1.0, length=0)
        ]

        # Log probs for expansion
        log_probs = torch.randn(2, 300)

        # Expand
        new_beams = decoder._expand_beams(beams, log_probs, k=5)

        assert len(new_beams) == 5
        assert all(isinstance(b, BeamHypothesis) for b in new_beams)


# ============================================================================
# Verification Tests
# ============================================================================

class TestThreeTierVerifier:
    """Tests for ThreeTierVerifier."""

    def test_initialization(self, tokenizer):
        """Test verifier initialization."""
        verifier = ThreeTierVerifier(tokenizer)
        assert verifier.tokenizer == tokenizer
        assert verifier.grammar is not None

    def test_syntax_check(self, tokenizer):
        """Test syntax tier."""
        verifier = ThreeTierVerifier(tokenizer)

        # Valid
        is_valid, error = verifier._syntax_check("x + y")
        assert is_valid

        # Invalid
        is_valid, error = verifier._syntax_check("x &")
        assert not is_valid

    def test_execution_test(self, tokenizer):
        """Test execution tier."""
        verifier = ThreeTierVerifier(tokenizer, exec_samples=10)

        # Equivalent expressions
        passes, ce = verifier._execution_test("x + y", "y + x")
        assert passes
        assert ce is None

        # Non-equivalent
        passes, ce = verifier._execution_test("x + y", "x & y")
        assert not passes

    def test_verify_batch(self, tokenizer):
        """Test batch verification."""
        verifier = ThreeTierVerifier(tokenizer, exec_samples=10, z3_top_k=2)

        candidates = [
            "x + y",      # Should pass syntax and exec
            "x &",        # Should fail syntax
            "x & y",      # Should pass syntax but fail exec
        ]

        results = verifier.verify_batch("x + y", candidates)

        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

        # Check ordering (z3 first, then exec, then syntax)
        assert results[0].syntax_valid

    def test_extract_variables(self, tokenizer):
        """Test variable extraction."""
        verifier = ThreeTierVerifier(tokenizer)

        vars1 = verifier._extract_variables("(x & y) + z")
        assert set(vars1) == {'x', 'y', 'z'}

        vars2 = verifier._extract_variables("x + x + x")
        assert vars2 == ['x']

    def test_generate_test_inputs(self, tokenizer):
        """Test input generation."""
        verifier = ThreeTierVerifier(tokenizer)

        inputs = verifier._generate_test_inputs(['x', 'y'], num_samples=20, width=8)

        assert len(inputs) == 20
        assert all('x' in inp and 'y' in inp for inp in inputs)
        assert all(0 <= inp['x'] <= 255 for inp in inputs)


# ============================================================================
# HTPS Tests
# ============================================================================

class TestMinimalHTPS:
    """Tests for MinimalHTPS."""

    def test_initialization(self, mock_model, tokenizer):
        """Test HTPS initialization."""
        htps = MinimalHTPS(mock_model, tokenizer, budget=100)
        assert htps.model == mock_model
        assert htps.tokenizer == tokenizer
        assert htps.budget == 100

    def test_tactic_enum(self):
        """Test tactic enumeration."""
        assert Tactic.IDENTITY_XOR_SELF.value == "identity_xor_self"
        assert Tactic.MBA_AND_XOR.value == "mba_and_xor"

    @patch('src.inference.htps.parse_to_ast')
    @patch('src.inference.htps.expr_to_graph')
    def test_search_simple(self, mock_expr_to_graph, mock_parse, mock_model, tokenizer):
        """Test HTPS search on simple expression."""
        # Mock parsing
        mock_ast = Mock()
        mock_ast.type = 'VAR'
        mock_ast.value = 'x'
        mock_ast.children = []
        mock_parse.return_value = mock_ast

        # Mock graph
        mock_graph = Mock()
        mock_expr_to_graph.return_value = mock_graph

        htps = MinimalHTPS(mock_model, tokenizer, budget=10)

        result, trace = htps.search("x")

        assert isinstance(result, str)
        assert isinstance(trace, list)


# ============================================================================
# Reranking Tests
# ============================================================================

class TestCandidateReranker:
    """Tests for CandidateReranker."""

    def test_initialization(self, tokenizer):
        """Test reranker initialization."""
        reranker = CandidateReranker(tokenizer)
        assert reranker.tokenizer == tokenizer
        assert reranker.weights is not None

    def test_extract_features(self, tokenizer):
        """Test feature extraction."""
        reranker = CandidateReranker(tokenizer)

        candidate = "x + y"
        model_score = -2.5
        verification = VerificationResult(
            candidate=candidate,
            syntax_valid=True,
            exec_valid=True,
            z3_verified=False
        )

        features = reranker.extract_features(candidate, model_score, verification)

        assert features.candidate == candidate
        assert features.model_confidence == model_score
        assert features.verification_level == 2  # exec valid
        assert features.token_length > 0

    def test_compute_score(self, tokenizer):
        """Test score computation."""
        reranker = CandidateReranker(tokenizer)

        features = RankingFeatures(
            candidate="x",
            token_length=1,
            ast_depth=0,
            num_operators=0,
            num_variables=1,
            num_constants=0,
            model_confidence=-1.0,
            verification_level=3,
            counterexample_found=False
        )

        score = reranker.compute_score(features)
        assert isinstance(score, float)
        assert score != float('-inf')

        # Test counterexample filtering
        features_ce = RankingFeatures(
            candidate="y",
            token_length=1,
            ast_depth=0,
            num_operators=0,
            num_variables=1,
            num_constants=0,
            model_confidence=-1.0,
            verification_level=2,
            counterexample_found=True
        )

        score_ce = reranker.compute_score(features_ce)
        assert score_ce == float('-inf')

    def test_rerank(self, tokenizer):
        """Test candidate reranking."""
        reranker = CandidateReranker(tokenizer)

        candidates = ["x", "x + 0", "x + y"]
        scores = [-1.0, -2.0, -1.5]
        verifications = [
            VerificationResult("x", True, True, True),
            VerificationResult("x + 0", True, True, False),
            VerificationResult("x + y", True, False, False)
        ]

        ranked = reranker.rerank(candidates, scores, verifications)

        assert len(ranked) > 0
        assert ranked[0][0] == "x"  # Simplest and z3-verified should rank first


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestInferencePipeline:
    """Tests for InferencePipeline."""

    @patch('src.inference.pipeline.expr_to_graph')
    @patch('src.inference.pipeline.expr_to_ast_depth')
    def test_initialization(self, mock_depth, mock_graph, mock_model, tokenizer):
        """Test pipeline initialization."""
        pipeline = InferencePipeline(mock_model, tokenizer)

        assert pipeline.model == mock_model
        assert pipeline.tokenizer == tokenizer
        assert pipeline.beam_decoder is not None
        assert pipeline.verifier is not None
        assert pipeline.reranker is not None

    @patch('src.inference.pipeline.expr_to_graph')
    @patch('src.inference.pipeline.expr_to_ast_depth')
    def test_preprocess(self, mock_depth, mock_graph, mock_model, tokenizer):
        """Test preprocessing."""
        # Mock returns
        mock_graph_obj = Mock()
        mock_graph_obj.x = torch.randn(5, 32)
        mock_graph_obj.edge_index = torch.randint(0, 5, (2, 10))
        mock_graph.return_value = mock_graph_obj
        mock_depth.return_value = 3

        pipeline = InferencePipeline(mock_model, tokenizer)

        graph_batch, fingerprint, depth = pipeline._preprocess("x + y")

        assert fingerprint.shape == (1, 448)
        assert depth == 3

    @patch('src.inference.pipeline.expr_to_graph')
    @patch('src.inference.pipeline.expr_to_ast_depth')
    @patch('src.data.fingerprint.SemanticFingerprint.compute')
    def test_simplify_beam(self, mock_fp_compute, mock_depth, mock_graph, mock_model, tokenizer):
        """Test simplification with beam search."""
        # Mock returns
        mock_graph_obj = Mock()
        mock_graph_obj.x = torch.randn(5, 32)
        mock_graph_obj.edge_index = torch.randint(0, 5, (2, 10))
        mock_graph.return_value = mock_graph_obj
        mock_depth.return_value = 3  # Below HTPS threshold
        mock_fp_compute.return_value = np.random.rand(448).astype(np.float32)

        pipeline = InferencePipeline(mock_model, tokenizer, verify_output=False)

        result = pipeline.simplify("x + y")

        assert isinstance(result, SimplificationResult)
        assert result.input_expr == "x + y"
        assert result.method == "beam"
        assert result.elapsed_ms > 0

    @patch('src.inference.pipeline.expr_to_graph')
    @patch('src.inference.pipeline.expr_to_ast_depth')
    def test_batch_simplify(self, mock_depth, mock_graph, mock_model, tokenizer):
        """Test batch simplification."""
        mock_graph_obj = Mock()
        mock_graph_obj.x = torch.randn(5, 32)
        mock_graph_obj.edge_index = torch.randint(0, 5, (2, 10))
        mock_graph.return_value = mock_graph_obj
        mock_depth.return_value = 3

        pipeline = InferencePipeline(mock_model, tokenizer, verify_output=False)

        exprs = ["x + y", "x & y"]
        results = pipeline.batch_simplify(exprs)

        assert len(results) == 2
        assert all(isinstance(r, SimplificationResult) for r in results)


# ============================================================================
# Integration Tests
# ============================================================================

class TestInferenceIntegration:
    """Integration tests for inference pipeline."""

    @patch('src.inference.pipeline.expr_to_graph')
    @patch('src.inference.pipeline.expr_to_ast_depth')
    @patch('src.data.fingerprint.SemanticFingerprint.compute')
    def test_end_to_end(self, mock_fp_compute, mock_depth, mock_graph, mock_model, tokenizer):
        """Test end-to-end pipeline."""
        mock_graph_obj = Mock()
        mock_graph_obj.x = torch.randn(5, 32)
        mock_graph_obj.edge_index = torch.randint(0, 5, (2, 10))
        mock_graph.return_value = mock_graph_obj
        mock_depth.return_value = 3
        mock_fp_compute.return_value = np.random.rand(448).astype(np.float32)

        pipeline = InferencePipeline(
            mock_model,
            tokenizer,
            beam_width=10,
            verify_output=False
        )

        result = pipeline.simplify("x + y")

        assert result.input_expr == "x + y"
        assert result.output_expr is not None
        assert result.elapsed_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
