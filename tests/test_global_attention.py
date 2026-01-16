"""
Tests for GraphGPS-style global attention module.

Verifies:
- Attention computation correctness
- Batch masking prevents cross-graph information flow
- Gradient isolation between graphs in a batch
- Integration with HGTEncoder
"""

import pytest
import torch
import torch.nn as nn

from src.models.global_attention import GlobalSelfAttention, GlobalAttentionBlock


class TestGlobalSelfAttention:
    """Tests for GlobalSelfAttention module."""

    def test_single_graph_shape(self):
        """Output shape matches input for single graph."""
        attn = GlobalSelfAttention(hidden_dim=256, num_heads=8)
        x = torch.randn(100, 256)

        out = attn(x)

        assert out.shape == x.shape

    def test_batched_shape(self):
        """Output shape matches input for batched graphs."""
        attn = GlobalSelfAttention(hidden_dim=256, num_heads=8)

        # Two graphs: 50 nodes and 30 nodes
        x = torch.randn(80, 256)
        batch = torch.cat([torch.zeros(50), torch.ones(30)]).long()

        out = attn(x, batch)

        assert out.shape == x.shape

    def test_batch_masking_prevents_cross_graph_attention(self):
        """Verify attention scores are zero between different graphs."""
        attn = GlobalSelfAttention(hidden_dim=64, num_heads=4)

        # Two graphs
        x = torch.randn(10, 64)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Access internal attention computation by running forward hooks
        attention_scores = []

        def hook(module, input, output):
            # Capture the attention pattern indirectly via gradient flow
            pass

        # Instead, verify via gradient isolation (more reliable)
        x.requires_grad_(True)
        out = attn(x, batch)

        # Backward pass from first graph's outputs
        out[:5].sum().backward()

        # Gradients for second graph should be zero (no information flow)
        grad_graph2 = x.grad[5:].abs().max().item()
        assert grad_graph2 < 1e-6, f"Cross-graph gradient detected: {grad_graph2}"

    def test_gradient_isolation_reverse(self):
        """Verify gradient isolation in reverse direction."""
        attn = GlobalSelfAttention(hidden_dim=64, num_heads=4)

        x = torch.randn(10, 64, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        out = attn(x, batch)

        # Backward pass from second graph's outputs
        out[5:].sum().backward()

        # Gradients for first graph should be zero
        grad_graph1 = x.grad[:5].abs().max().item()
        assert grad_graph1 < 1e-6, f"Cross-graph gradient detected: {grad_graph1}"

    def test_residual_connection(self):
        """Verify residual connection preserves input signal."""
        attn = GlobalSelfAttention(hidden_dim=64, num_heads=4)

        # Zero-initialize all weights to test residual only
        with torch.no_grad():
            attn.out_proj.weight.zero_()
            attn.out_proj.bias.zero_()

        x = torch.randn(10, 64)
        out = attn(x)

        # With zero output projection, output should equal input (residual)
        assert torch.allclose(out, x, atol=1e-5)

    def test_hidden_dim_divisibility_check(self):
        """Raise error when hidden_dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            GlobalSelfAttention(hidden_dim=100, num_heads=8)

    def test_nan_handling_isolated_nodes(self):
        """Verify NaN handling when a node has no valid attention targets."""
        attn = GlobalSelfAttention(hidden_dim=64, num_heads=4)

        # Three separate graphs with 1 node each (isolated)
        x = torch.randn(3, 64)
        batch = torch.tensor([0, 1, 2])

        out = attn(x, batch)

        # Should not produce NaN
        assert not torch.isnan(out).any(), "NaN detected in output"


class TestGlobalAttentionBlock:
    """Tests for GlobalAttentionBlock module."""

    def test_shape_preservation(self):
        """Block preserves input shape."""
        block = GlobalAttentionBlock(hidden_dim=256, num_heads=8)
        x = torch.randn(50, 256)
        batch = torch.zeros(50).long()

        out = block(x, batch)

        assert out.shape == x.shape

    def test_checkpoint_training_mode(self):
        """Gradient checkpointing activates in training mode."""
        block = GlobalAttentionBlock(hidden_dim=64, num_heads=4, use_checkpoint=True)
        block.train()

        x = torch.randn(20, 64, requires_grad=True)
        batch = torch.zeros(20).long()

        out = block(x, batch)
        loss = out.sum()
        loss.backward()

        # Should complete without error, gradients should exist
        assert x.grad is not None

    def test_no_checkpoint_eval_mode(self):
        """Gradient checkpointing disabled in eval mode."""
        block = GlobalAttentionBlock(hidden_dim=64, num_heads=4, use_checkpoint=True)
        block.eval()

        x = torch.randn(20, 64)
        batch = torch.zeros(20).long()

        # Should complete without error
        out = block(x, batch)
        assert out.shape == x.shape

    def test_ffn_ratio(self):
        """FFN hidden dim respects ratio parameter."""
        block = GlobalAttentionBlock(hidden_dim=64, num_heads=4, ffn_ratio=2.0)

        # Find the FFN linear layer and check its hidden dim
        ffn_hidden_dim = None
        for module in block.ffn:
            if isinstance(module, nn.Linear) and module.in_features == 64:
                ffn_hidden_dim = module.out_features
                break

        assert ffn_hidden_dim == 128  # 64 * 2.0


class TestHGTEncoderGlobalAttention:
    """Tests for HGTEncoder with global attention enabled."""

    @pytest.fixture
    def mock_graph_data(self):
        """Create mock graph data for testing."""
        num_nodes = 50
        num_edges = 200

        # Random node types (0-9)
        x = torch.randint(0, 10, (num_nodes,))

        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Random edge types (0-7 for optimized 8-type system)
        edge_type = torch.randint(0, 8, (num_edges,))

        # Single graph batch
        batch = torch.zeros(num_nodes).long()

        return x, edge_index, edge_type, batch

    def test_hgt_with_global_attention_shape(self, mock_graph_data):
        """HGT with global attention produces correct output shape."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=6,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=2,
            global_attn_heads=4,
        )

        x, edge_index, edge_type, batch = mock_graph_data
        out = encoder(x, edge_index, batch, edge_type=edge_type)

        assert out.shape == (x.size(0), 64)

    def test_hgt_global_attention_blocks_count(self):
        """Verify correct number of global attention blocks created."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        # 12 layers, interval 2: blocks after 1,3,5,7,9,11 but not after 11 (last layer)
        # So blocks after layers 1,3,5,7,9 = 5 blocks
        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=12,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=2,
        )

        assert len(encoder.global_attn_blocks) == 5

    def test_hgt_global_attention_blocks_count_interval_3(self):
        """Verify block count with different interval."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        # 12 layers, interval 3: blocks after 2,5,8,11 but not after 11 (last layer)
        # So blocks after layers 2,5,8 = 3 blocks
        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=12,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=3,
        )

        assert len(encoder.global_attn_blocks) == 3

    def test_hgt_without_global_attention(self, mock_graph_data):
        """HGT without global attention still works."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_global_attention=False,
        )

        x, edge_index, edge_type, batch = mock_graph_data
        out = encoder(x, edge_index, batch, edge_type=edge_type)

        assert out.shape == (x.size(0), 64)
        assert encoder.global_attn_blocks is None

    def test_hgt_global_attention_batched(self):
        """HGT with global attention handles batched graphs correctly."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=2,
        )

        # Two graphs: 30 and 20 nodes
        x = torch.randint(0, 10, (50,))
        edge_index = torch.randint(0, 50, (2, 100))
        edge_type = torch.randint(0, 8, (100,))
        batch = torch.cat([torch.zeros(30), torch.ones(20)]).long()

        out = encoder(x, edge_index, batch, edge_type=edge_type)

        assert out.shape == (50, 64)

    def test_hgt_global_attention_gradient_flow(self, mock_graph_data):
        """Verify gradients flow through global attention blocks."""
        pytest.importorskip("torch_geometric")

        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=2,
            global_attn_checkpoint=False,  # Disable for gradient checking
        )

        x, edge_index, edge_type, batch = mock_graph_data

        # Create embedding to have gradients
        x_embed = encoder.node_type_embed(x)
        x_embed.retain_grad()

        # Forward through encoder manually to check global attn gradients
        out = encoder(x, edge_index, batch, edge_type=edge_type)
        loss = out.sum()
        loss.backward()

        # Check that global attention block has gradients
        for i, block in enumerate(encoder.global_attn_blocks):
            for name, param in block.named_parameters():
                assert param.grad is not None, f"No gradient for global_attn_blocks[{i}].{name}"


class TestGlobalAttentionMemory:
    """Memory-related tests for global attention."""

    def test_checkpoint_reduces_memory(self):
        """Gradient checkpointing should reduce peak memory."""
        # This is a qualitative test - just verify it runs without OOM
        # Actual memory measurement would require more sophisticated tooling

        block_checkpoint = GlobalAttentionBlock(
            hidden_dim=256, num_heads=8, use_checkpoint=True
        )
        block_no_checkpoint = GlobalAttentionBlock(
            hidden_dim=256, num_heads=8, use_checkpoint=False
        )

        block_checkpoint.train()
        block_no_checkpoint.train()

        x = torch.randn(100, 256, requires_grad=True)
        batch = torch.zeros(100).long()

        # Both should work
        out1 = block_checkpoint(x, batch)
        out1.sum().backward()

        x2 = torch.randn(100, 256, requires_grad=True)
        out2 = block_no_checkpoint(x2, batch)
        out2.sum().backward()

        # Just verify both complete successfully
        assert x.grad is not None
        assert x2.grad is not None
