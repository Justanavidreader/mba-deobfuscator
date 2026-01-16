"""
Tests for path encoding integration with GNN encoders.

Tests both GGNNEncoder and HGTEncoder with path encoding enabled,
verifying shape preservation, gradient flow, backward compatibility,
and determinism.
"""

import pytest
import torch
import sys
import os

# Skip all tests if torch_scatter not available (encoder.py requires it)
torch_scatter = pytest.importorskip("torch_scatter", reason="torch_scatter not installed")

# Add src to path to avoid torch_scatter import issues in package __init__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
from path_encoding import PathBasedEdgeEncoder

# Now import encoders normally
from src.models.encoder import GGNNEncoder, HGTEncoder, _infer_node_types


class TestInferNodeTypes:
    """Tests for the _infer_node_types helper function."""

    def test_infer_from_type_ids(self):
        """Node type IDs pass through unchanged."""
        x = torch.tensor([0, 1, 2, 3, 4])
        result = _infer_node_types(x)
        assert torch.equal(result, x)

    def test_infer_from_one_hot(self):
        """One-hot features converted to type IDs via argmax."""
        x = torch.zeros(5, 10)
        x[0, 2] = 1.0  # node 0 is type 2
        x[1, 5] = 1.0  # node 1 is type 5
        x[2, 0] = 1.0  # node 2 is type 0
        x[3, 9] = 1.0  # node 3 is type 9
        x[4, 3] = 1.0  # node 4 is type 3

        result = _infer_node_types(x)
        expected = torch.tensor([2, 5, 0, 9, 3])
        assert torch.equal(result, expected)

    def test_infer_from_soft_features(self):
        """Soft features (not strictly one-hot) use argmax."""
        x = torch.tensor([
            [0.1, 0.8, 0.1],  # type 1
            [0.9, 0.05, 0.05],  # type 0
        ])
        result = _infer_node_types(x)
        expected = torch.tensor([1, 0])
        assert torch.equal(result, expected)


class TestGGNNPathEncoding:
    """Tests for GGNN encoder with path encoding."""

    @pytest.fixture
    def simple_graph(self):
        """Simple graph for testing: 5 nodes, 6 edges."""
        x = torch.randn(5, 32)  # [num_nodes, node_dim]
        edge_index = torch.tensor([
            [0, 1, 2, 0, 3, 4],
            [1, 2, 3, 2, 4, 3]
        ])
        edge_type = torch.tensor([0, 1, 2, 3, 4, 5])
        batch = torch.zeros(5, dtype=torch.long)
        return x, edge_index, edge_type, batch

    def test_ggnn_with_path_encoding_shape(self, simple_graph):
        """GGNN with path encoding produces correct output shape."""
        x, edge_index, edge_type, batch = simple_graph

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=4,
            num_edge_types=6,
            use_path_encoding=True,
            path_max_length=4,
            path_max_paths=8,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

    def test_ggnn_path_encoding_gradient_flow(self, simple_graph):
        """Gradients flow through path encoder."""
        x, edge_index, edge_type, batch = simple_graph

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=2,
            num_edge_types=6,
            use_path_encoding=True,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        loss = output.sum()
        loss.backward()

        # Check path encoder params have gradients
        assert encoder.path_encoder is not None
        for name, param in encoder.path_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_ggnn_backward_compatible(self, simple_graph):
        """GGNN without path encoding produces same shape."""
        x, edge_index, edge_type, batch = simple_graph

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=4,
            num_edge_types=6,
            use_path_encoding=False,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)
        assert encoder.path_encoder is None

    def test_ggnn_path_encoding_determinism(self, simple_graph):
        """Path embeddings are deterministic for same input."""
        x, edge_index, edge_type, batch = simple_graph

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=2,
            num_edge_types=6,
            use_path_encoding=True,
        )
        encoder.eval()

        with torch.no_grad():
            output1 = encoder(x, edge_index, batch, edge_type=edge_type)
            output2 = encoder(x, edge_index, batch, edge_type=edge_type)

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("max_length,max_paths,aggregation", [
        (2, 4, 'mean'),
        (6, 16, 'max'),
        (4, 8, 'attention'),
    ])
    def test_ggnn_path_encoding_parameterized(self, simple_graph, max_length, max_paths, aggregation):
        """GGNN works with various path encoding configurations."""
        x, edge_index, edge_type, batch = simple_graph

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=2,
            num_edge_types=6,
            use_path_encoding=True,
            path_max_length=max_length,
            path_max_paths=max_paths,
            path_aggregation=aggregation,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

        # Verify gradient flow
        loss = output.sum()
        loss.backward()
        assert encoder.path_encoder is not None


class TestHGTPathEncoding:
    """Tests for HGT encoder with path encoding."""

    @pytest.fixture
    def heterogeneous_graph(self):
        """Heterogeneous graph for HGT testing."""
        # Node types: 5 nodes with types 0-4 (VAR, CONST, ADD, SUB, MUL)
        x = torch.tensor([0, 1, 2, 3, 4])

        # Edges matching VALID_TRIPLETS in HGTEncoder
        # (2, 0, 0) = ADD -> VAR as left operand
        # (2, 1, 1) = ADD -> CONST as right operand
        # (3, 0, 2) = SUB -> ADD as left operand
        # (4, 1, 3) = MUL -> SUB as right operand
        edge_index = torch.tensor([
            [2, 2, 3, 4],  # src: ADD, ADD, SUB, MUL
            [0, 1, 2, 3]   # dst: VAR, CONST, ADD, SUB
        ])
        edge_type = torch.tensor([0, 1, 0, 1])  # LEFT_OP, RIGHT_OP, LEFT_OP, RIGHT_OP
        batch = torch.zeros(5, dtype=torch.long)

        return x, edge_index, edge_type, batch

    def test_hgt_with_path_encoding_shape(self, heterogeneous_graph):
        """HGT with path encoding produces correct output shape."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_path_encoding=True,
            path_injection_interval=2,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

    def test_hgt_path_injection_count(self):
        """Verify correct number of path injections."""
        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=8,
            num_heads=4,
            use_path_encoding=True,
            path_injection_interval=2,
        )
        # (8-1) // 2 = 3 injection points (after layers 1, 3, 5)
        assert encoder.path_projectors is not None
        assert len(encoder.path_projectors) == 3

    def test_hgt_path_injection_zero(self):
        """Verify no projectors created when num_injections=0."""
        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            use_path_encoding=True,
            path_injection_interval=2,
        )
        # (2-1) // 2 = 0 injections
        assert encoder.path_projectors is None

    def test_hgt_path_encoding_gradient_flow(self, heterogeneous_graph):
        """Gradients flow through path encoder and projectors."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_path_encoding=True,
            path_injection_interval=2,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        loss = output.sum()
        loss.backward()

        # Check path encoder params have gradients
        assert encoder.path_encoder is not None
        for name, param in encoder.path_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for path_encoder.{name}"

        # Check path projector params have gradients
        assert encoder.path_projectors is not None
        for i, proj in enumerate(encoder.path_projectors):
            assert proj.weight.grad is not None, f"No gradient for projector {i} weight"

    def test_hgt_backward_compatible(self, heterogeneous_graph):
        """HGT without path encoding produces same shape."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_path_encoding=False,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)
        assert encoder.path_encoder is None
        assert encoder.path_projectors is None

    def test_hgt_combined_global_and_path(self, heterogeneous_graph):
        """HGT with both global attention and path encoding."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=6,
            num_heads=4,
            use_global_attention=True,
            global_attn_interval=2,
            use_path_encoding=True,
            path_injection_interval=3,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

        # Both features should be enabled
        assert encoder.global_attn_blocks is not None
        assert encoder.path_encoder is not None

    def test_hgt_path_encoding_determinism(self, heterogeneous_graph):
        """Path embeddings are deterministic for same input."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_path_encoding=True,
        )
        encoder.eval()

        with torch.no_grad():
            output1 = encoder(x, edge_index, batch, edge_type=edge_type)
            output2 = encoder(x, edge_index, batch, edge_type=edge_type)

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("injection_interval,scale", [
        (2, 0.1),
        (3, 0.05),
        (4, 0.2),
    ])
    def test_hgt_path_injection_parameterized(self, heterogeneous_graph, injection_interval, scale):
        """HGT works with various path injection configurations."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=8,
            num_heads=4,
            use_path_encoding=True,
            path_injection_interval=injection_interval,
            path_injection_scale=scale,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

    def test_hgt_path_context_device_consistency(self, heterogeneous_graph):
        """Path context is on same device as node features."""
        x, edge_index, edge_type, batch = heterogeneous_graph

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            use_path_encoding=True,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.device == x.device


class TestPathEncodingIntegrationEdgeCases:
    """Edge case tests for path encoding integration."""

    def test_empty_graph_ggnn(self):
        """GGNN handles empty graph gracefully."""
        x = torch.randn(0, 32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        batch = torch.zeros(0, dtype=torch.long)

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_edge_types=6,
            use_path_encoding=True,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (0, 64)

    def test_single_node_no_edges_ggnn(self):
        """GGNN handles single node with no edges."""
        x = torch.randn(1, 32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_edge_types=6,
            use_path_encoding=True,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (1, 64)

    def test_batched_graphs_ggnn(self):
        """GGNN handles batched graphs correctly."""
        # Two graphs: 3 nodes + 2 nodes = 5 total
        x = torch.randn(5, 32)
        edge_index = torch.tensor([
            [0, 1, 3],  # src
            [1, 2, 4]   # dst
        ])
        edge_type = torch.tensor([0, 1, 2])
        batch = torch.tensor([0, 0, 0, 1, 1])

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_edge_types=6,
            use_path_encoding=True,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (5, 64)

    def test_large_graph_ggnn(self):
        """GGNN handles larger graphs without OOM."""
        num_nodes = 100
        num_edges = 300

        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 6, (num_edges,))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        encoder = GGNNEncoder(
            node_dim=32,
            hidden_dim=64,
            num_edge_types=6,
            use_path_encoding=True,
            path_max_length=4,
            path_max_paths=8,
        )

        output = encoder(x, edge_index, batch, edge_type=edge_type)
        assert output.shape == (num_nodes, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
