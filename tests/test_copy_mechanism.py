"""Unit tests for copy mechanism implementation."""

import pytest
import torch
import torch.nn.functional as F

from src.constants import VOCAB_SIZE, PAD_IDX


class TestComputeCopyDistribution:
    """Tests for MBADeobfuscator._compute_copy_distribution method."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        from src.models.full_model import MBADeobfuscator
        return MBADeobfuscator(encoder_type='gat')

    def test_basic_copy_distribution(self, model):
        """Test that copy distribution correctly scatters attention to token IDs."""
        batch_size, src_len, tgt_len = 2, 5, 3
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
        copy_attn = F.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
        p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))

        # Source tokens - use valid token IDs
        source_tokens = torch.tensor([
            [14, 5, 15, 6, 30],  # x0, +, x1, -, 0
            [14, 5, 15, 0, 0],   # x0, +, x1, pad, pad
        ])

        final_logits = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens
        )

        # Shape check
        assert final_logits.shape == (batch_size, tgt_len, vocab_size)
        # Finiteness check
        assert torch.isfinite(final_logits).all(), "Final logits contain NaN/Inf"

    def test_copy_distribution_preserves_probability_mass(self, model):
        """Test that final distribution sums to approximately 1."""
        batch_size, src_len, tgt_len = 2, 4, 3
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
        copy_attn = F.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
        p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))

        source_tokens = torch.tensor([
            [14, 5, 15, 6],
            [14, 5, 15, 6],
        ])

        final_logits = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens
        )

        # Convert to probabilities and check they sum to ~1
        final_probs = F.softmax(final_logits, dim=-1)
        prob_sums = final_probs.sum(dim=-1)

        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            f"Probability sums deviate from 1: {prob_sums}"

    def test_copy_distribution_with_memory_mask(self, model):
        """Test that memory mask properly zeroes out padding positions."""
        batch_size, src_len, tgt_len = 2, 5, 3
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
        # High attention on last 2 positions (which will be masked)
        copy_attn = torch.zeros(batch_size, tgt_len, src_len)
        copy_attn[:, :, -2:] = 0.9  # High attention on padding
        copy_attn[:, :, :3] = 0.1 / 3  # Low attention on real tokens
        copy_attn = F.softmax(copy_attn, dim=-1)

        p_gen = torch.full((batch_size, tgt_len, 1), 0.3)  # 70% copy

        # Last 2 tokens are padding
        source_tokens = torch.tensor([
            [14, 5, 15, PAD_IDX, PAD_IDX],
            [14, 5, 15, PAD_IDX, PAD_IDX],
        ])

        # Memory mask: True = masked position
        memory_mask = torch.tensor([
            [False, False, False, True, True],
            [False, False, False, True, True],
        ])

        # Test WITH mask
        final_logits_masked = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens, memory_mask
        )

        # Test WITHOUT mask
        final_logits_unmasked = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens, memory_mask=None
        )

        # Results should differ because masking changes attention distribution
        assert not torch.allclose(final_logits_masked, final_logits_unmasked), \
            "Memory mask should affect output"

    def test_copy_distribution_invalid_inputs(self, model):
        """Test that invalid inputs raise appropriate errors."""
        batch_size, src_len, tgt_len = 2, 5, 3
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.randn(batch_size, tgt_len, vocab_size)
        copy_attn = F.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
        p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))

        # Wrong src_len for source_tokens
        source_tokens_wrong = torch.tensor([
            [14, 5, 15],  # Only 3 tokens instead of 5
            [14, 5, 15],
        ])

        with pytest.raises(ValueError, match="Shape mismatch"):
            model._compute_copy_distribution(
                vocab_logits, copy_attn, p_gen, source_tokens_wrong
            )

    def test_copy_distribution_handles_repeated_tokens(self, model):
        """Test that repeated source tokens accumulate attention correctly."""
        batch_size, src_len, tgt_len = 1, 4, 1
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.zeros(batch_size, tgt_len, vocab_size)
        # Equal attention across all source positions
        copy_attn = torch.full((batch_size, tgt_len, src_len), 0.25)
        p_gen = torch.zeros(batch_size, tgt_len, 1)  # 100% copy

        # Source has repeated token (token 14 appears twice)
        source_tokens = torch.tensor([[14, 14, 15, 16]])  # x0 appears twice

        final_logits = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens
        )

        final_probs = F.softmax(final_logits, dim=-1)

        # Token 14 should have 0.5 probability (0.25 + 0.25 from two occurrences)
        # Token 15 and 16 should have 0.25 each
        assert abs(final_probs[0, 0, 14].item() - 0.5) < 0.01, \
            f"Repeated token should accumulate: got {final_probs[0, 0, 14].item()}"
        assert abs(final_probs[0, 0, 15].item() - 0.25) < 0.01
        assert abs(final_probs[0, 0, 16].item() - 0.25) < 0.01


class TestModelDecodeWithCopy:
    """Tests for MBADeobfuscator.decode() with copy mechanism."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        from src.models.full_model import MBADeobfuscator
        return MBADeobfuscator(encoder_type='gat')

    def test_decode_returns_final_logits_with_source_tokens(self, model):
        """Test that decode() returns final_logits when source_tokens provided."""
        batch_size = 2
        tgt_len = 5
        src_len = 4

        # Create dummy memory (encoder output)
        memory = torch.randn(batch_size, 1, 512)  # [batch, 1, d_model]

        # Target tokens
        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, tgt_len))

        # Source tokens for copy
        source_tokens = torch.randint(0, VOCAB_SIZE, (batch_size, src_len))

        # Decode with source_tokens
        output = model.decode(tgt, memory, source_tokens=source_tokens)

        # Check that final_logits is present
        assert 'final_logits' in output, "final_logits not in output when source_tokens provided"
        assert output['final_logits'].shape == (batch_size, tgt_len, VOCAB_SIZE)

    def test_decode_no_final_logits_without_source_tokens(self, model):
        """Test that decode() does not return final_logits without source_tokens."""
        batch_size = 2
        tgt_len = 5

        memory = torch.randn(batch_size, 1, 512)
        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, tgt_len))

        # Decode without source_tokens
        output = model.decode(tgt, memory)

        # Check that final_logits is NOT present
        assert 'final_logits' not in output, "final_logits should not be present without source_tokens"
        assert 'vocab_logits' in output


class TestModelForwardWithCopy:
    """Tests for MBADeobfuscator.forward() with copy mechanism."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        from src.models.full_model import MBADeobfuscator
        return MBADeobfuscator(encoder_type='gat')

    @pytest.fixture
    def dummy_graph_batch(self):
        """Create a dummy graph batch for testing."""
        from torch_geometric.data import Data, Batch

        # Simple graph with 3 nodes
        x = torch.randn(3, 32)  # node features
        edge_index = torch.tensor([[0, 1], [1, 2]]).t().contiguous()
        edge_type = torch.tensor([0, 1])

        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        return Batch.from_data_list([data, data])  # batch of 2

    def test_forward_with_source_tokens(self, model, dummy_graph_batch):
        """Test forward() includes final_logits when source_tokens provided."""
        batch_size = 2
        tgt_len = 5
        src_len = 4

        fingerprint = torch.randn(batch_size, 448)
        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, tgt_len))
        source_tokens = torch.randint(0, VOCAB_SIZE, (batch_size, src_len))

        output = model(dummy_graph_batch, fingerprint, tgt, source_tokens=source_tokens)

        assert 'final_logits' in output, "final_logits missing in forward output"
        assert 'vocab_logits' in output
        assert 'copy_attn' in output
        assert 'p_gen' in output

    def test_forward_without_source_tokens(self, model, dummy_graph_batch):
        """Test forward() without source_tokens for backward compatibility."""
        batch_size = 2
        tgt_len = 5

        fingerprint = torch.randn(batch_size, 448)
        tgt = torch.randint(0, VOCAB_SIZE, (batch_size, tgt_len))

        output = model(dummy_graph_batch, fingerprint, tgt)

        assert 'final_logits' not in output, "final_logits should not be present"
        assert 'vocab_logits' in output


class TestGradientFlow:
    """Tests for gradient flow through copy mechanism."""

    @pytest.fixture
    def model(self):
        """Create a minimal model for testing."""
        from src.models.full_model import MBADeobfuscator
        return MBADeobfuscator(encoder_type='gat')

    def test_gradient_flows_through_copy_distribution(self, model):
        """Test that gradients flow through _compute_copy_distribution."""
        batch_size, src_len, tgt_len = 2, 4, 3
        vocab_size = VOCAB_SIZE

        vocab_logits = torch.randn(batch_size, tgt_len, vocab_size, requires_grad=True)
        copy_attn = F.softmax(torch.randn(batch_size, tgt_len, src_len), dim=-1)
        copy_attn.requires_grad_(True)
        p_gen = torch.sigmoid(torch.randn(batch_size, tgt_len, 1))
        p_gen.requires_grad_(True)

        source_tokens = torch.tensor([
            [14, 5, 15, 6],
            [14, 5, 15, 6],
        ])

        final_logits = model._compute_copy_distribution(
            vocab_logits, copy_attn, p_gen, source_tokens
        )

        # Dummy loss
        loss = final_logits.sum()
        loss.backward()

        # Check gradients exist
        assert vocab_logits.grad is not None, "No gradient on vocab_logits"
        assert copy_attn.grad is not None, "No gradient on copy_attn"
        assert p_gen.grad is not None, "No gradient on p_gen"

        # Check gradients are not NaN
        assert torch.isfinite(vocab_logits.grad).all(), "NaN gradient on vocab_logits"
        assert torch.isfinite(copy_attn.grad).all(), "NaN gradient on copy_attn"
        assert torch.isfinite(p_gen.grad).all(), "NaN gradient on p_gen"


class TestEdgeTypeModeParameter:
    """Tests for Phase 3B edge_type_mode parameter."""

    def test_ast_to_graph_legacy_mode(self):
        """Test ast_to_graph with legacy edge type mode (default)."""
        from src.data.ast_parser import expr_to_graph

        graph = expr_to_graph("x & y", edge_type_mode="legacy")

        # Legacy mode should have edge types < 6
        assert graph.edge_type.max() < 6, "Legacy mode should have max edge type < 6"

    def test_ast_to_graph_optimized_mode(self):
        """Test ast_to_graph with optimized edge type mode."""
        from src.data.ast_parser import expr_to_graph

        graph = expr_to_graph("x & y", edge_type_mode="optimized")

        # Optimized mode should have edge types < 8
        assert graph.edge_type.max() < 8, "Optimized mode should have max edge type < 8"

    def test_ast_to_graph_invalid_mode(self):
        """Test ast_to_graph raises error for invalid edge_type_mode."""
        from src.data.ast_parser import expr_to_graph

        with pytest.raises(ValueError, match="edge_type_mode"):
            expr_to_graph("x & y", edge_type_mode="invalid")

    def test_optimized_mode_matches_direct_function(self):
        """Test that edge_type_mode='optimized' produces same result as expr_to_optimized_graph."""
        from src.data.ast_parser import expr_to_graph, expr_to_optimized_graph

        expr = "(x & y) + (x ^ y)"

        g_via_param = expr_to_graph(expr, edge_type_mode="optimized")
        g_direct = expr_to_optimized_graph(expr)

        # Compare shapes
        assert g_via_param.x.shape == g_direct.x.shape
        assert g_via_param.edge_index.shape == g_direct.edge_index.shape

        # Compare edge types
        assert torch.equal(g_via_param.edge_type, g_direct.edge_type)
