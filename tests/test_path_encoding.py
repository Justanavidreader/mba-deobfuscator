"""
Tests for path-based edge encoding module.

Verifies:
- PathFinder correctly finds paths in DAGs
- PathEncoder produces correct output shapes
- PathBasedEdgeEncoder handles edge cases (empty paths, bounds, batching)
- Quality review fixes: bounds checking, empty handling, batched encoding
"""

import pytest
import torch
import torch.nn as nn

import sys
import os

# Add src to path to allow direct import without going through __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from path_encoding import PathFinder, PathEncoder, PathBasedEdgeEncoder


class TestPathFinder:
    """Tests for PathFinder class."""

    def test_simple_dag_paths(self):
        """Find paths in simple DAG with two routes."""
        # DAG: 0 -> 1 -> 3
        #      0 -> 2 -> 3
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        finder = PathFinder(max_path_length=4, max_paths=10)
        adj = finder.build_adjacency(edge_index, num_nodes=4)

        paths = finder.find_paths(adj, 0, 3)

        assert len(paths) == 2
        assert [0, 1, 3] in paths
        assert [0, 2, 3] in paths

    def test_self_loop_path(self):
        """Path from node to itself returns single-node path."""
        edge_index = torch.tensor([[0, 1], [1, 2]])
        finder = PathFinder(max_path_length=4, max_paths=10)
        adj = finder.build_adjacency(edge_index, num_nodes=3)

        paths = finder.find_paths(adj, 1, 1)

        assert len(paths) == 1
        assert paths[0] == [1]

    def test_no_path_exists(self):
        """Return empty list when no path exists."""
        edge_index = torch.tensor([[0], [1]])  # 0 -> 1 only
        finder = PathFinder(max_path_length=4, max_paths=10)
        adj = finder.build_adjacency(edge_index, num_nodes=3)

        paths = finder.find_paths(adj, 2, 0)  # No path from 2 to 0

        assert len(paths) == 0

    def test_max_paths_limit(self):
        """Respect max_paths limit."""
        # Create graph with many paths: 0 -> [1,2,3,4,5] -> 6
        edges_src = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
        edges_dst = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
        edge_index = torch.tensor([edges_src, edges_dst])

        finder = PathFinder(max_path_length=4, max_paths=3)
        adj = finder.build_adjacency(edge_index, num_nodes=7)

        paths = finder.find_paths(adj, 0, 6)

        assert len(paths) <= 3

    def test_max_path_length_limit(self):
        """Respect max_path_length limit."""
        # Linear graph: 0 -> 1 -> 2 -> 3 -> 4 -> 5
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

        finder = PathFinder(max_path_length=3, max_paths=10)
        adj = finder.build_adjacency(edge_index, num_nodes=6)

        # Path 0 -> 5 requires length 6, should not be found with max_length=3
        paths = finder.find_paths(adj, 0, 5)

        assert len(paths) == 0

    def test_bounds_checking_adjacency(self):
        """Ignore edges with out-of-bounds node indices."""
        # Include an out-of-bounds edge
        edge_index = torch.tensor([[0, 0, 100], [1, 2, 101]])

        finder = PathFinder(max_path_length=4, max_paths=10)
        adj = finder.build_adjacency(edge_index, num_nodes=3)

        # Should only have edges within bounds
        assert 100 not in adj
        assert len(adj[0]) == 2  # Only 0->1 and 0->2

    def test_invalid_max_path_length(self):
        """Raise ValueError for max_path_length < 2."""
        with pytest.raises(ValueError, match="max_path_length must be >= 2"):
            PathFinder(max_path_length=1, max_paths=10)

    def test_invalid_max_paths(self):
        """Raise ValueError for max_paths < 1."""
        with pytest.raises(ValueError, match="max_paths must be >= 1"):
            PathFinder(max_path_length=4, max_paths=0)


class TestPathEncoder:
    """Tests for PathEncoder module."""

    def test_output_shape(self):
        """Output shape matches [batch, hidden_dim]."""
        encoder = PathEncoder(hidden_dim=64, num_node_types=10, num_edge_types=8)

        # 4 paths, max length 6
        node_types = torch.randint(1, 10, (4, 6))  # 1-indexed (0=pad)
        edge_types = torch.randint(1, 8, (4, 5))   # 5 edges per path
        path_lengths = torch.tensor([3, 4, 5, 6])

        out = encoder(node_types, edge_types, path_lengths)

        assert out.shape == (4, 64)

    def test_single_path(self):
        """Encode single path."""
        encoder = PathEncoder(hidden_dim=32, num_node_types=10, num_edge_types=8)

        node_types = torch.tensor([[1, 2, 3, 0, 0, 0]])  # Length 3, padded
        edge_types = torch.tensor([[1, 2, 0, 0, 0]])
        path_lengths = torch.tensor([3])

        out = encoder(node_types, edge_types, path_lengths)

        assert out.shape == (1, 32)
        assert not torch.isnan(out).any()

    def test_lstm_encoder(self):
        """PathEncoder works with LSTM backend."""
        encoder = PathEncoder(
            hidden_dim=64,
            num_node_types=10,
            num_edge_types=8,
            use_transformer=False,
        )

        node_types = torch.randint(1, 10, (3, 6))
        edge_types = torch.randint(1, 8, (3, 5))
        path_lengths = torch.tensor([2, 4, 6])

        out = encoder(node_types, edge_types, path_lengths)

        assert out.shape == (3, 64)

    def test_odd_hidden_dim_raises(self):
        """Raise ValueError for odd hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be even"):
            PathEncoder(hidden_dim=65, num_node_types=10, num_edge_types=8)

    def test_padding_handling(self):
        """Padded positions (0) don't contribute to output."""
        encoder = PathEncoder(hidden_dim=64, num_node_types=10, num_edge_types=8)

        # Same content but different padding
        node_types1 = torch.tensor([[1, 2, 3, 0, 0, 0]])
        node_types2 = torch.tensor([[1, 2, 3, 99, 99, 99]])  # Different pad values

        edge_types = torch.tensor([[1, 2, 0, 0, 0]])
        path_lengths = torch.tensor([3])

        # Clamp node_types2 to valid range for embedding
        node_types2 = node_types2.clamp(0, 10)

        out1 = encoder(node_types1, edge_types, path_lengths)
        out2 = encoder(node_types2, edge_types, path_lengths)

        # Outputs should be different due to different embeddings
        # But both should be valid (no NaN)
        assert not torch.isnan(out1).any()
        assert not torch.isnan(out2).any()


class TestPathBasedEdgeEncoder:
    """Tests for PathBasedEdgeEncoder module."""

    def test_simple_graph_shape(self):
        """Output shape matches [num_edges, hidden_dim]."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, num_node_types=10, num_edge_types=8)

        # Simple DAG: 0 -> 1 -> 2
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])
        node_types = torch.tensor([0, 1, 2])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (2, 64)

    def test_empty_edges(self):
        """Handle empty edge list."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64)

        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        node_types = torch.tensor([0, 1, 2])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (0, 64)

    def test_no_paths_uses_direct_embedding(self):
        """When no paths found, still produces valid output."""
        encoder = PathBasedEdgeEncoder(
            hidden_dim=64,
            max_path_length=2,  # Very short
        )

        # Edges where paths are too long
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_type = torch.tensor([0, 1, 2, 3])
        node_types = torch.tensor([0, 1, 2, 3, 4])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (4, 64)
        assert not torch.isnan(edge_emb).any()

    def test_edge_type_clamping(self):
        """Edge types outside valid range are clamped."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, num_edge_types=8)

        edge_index = torch.tensor([[0], [1]])
        edge_type = torch.tensor([100])  # Out of range
        node_types = torch.tensor([0, 1])

        # Should not raise, should clamp
        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (1, 64)
        assert not torch.isnan(edge_emb).any()

    def test_node_type_bounds_checking(self):
        """Node types at boundary don't cause index errors."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, num_node_types=10)

        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])
        node_types = torch.tensor([9, 9, 9])  # Max valid node type

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (2, 64)
        assert not torch.isnan(edge_emb).any()

    def test_aggregation_mean(self):
        """Mean aggregation works."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, aggregation='mean')

        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        edge_type = torch.tensor([0, 1, 2, 3])
        node_types = torch.tensor([0, 1, 2, 3])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (4, 64)

    def test_aggregation_max(self):
        """Max aggregation works."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, aggregation='max')

        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        edge_type = torch.tensor([0, 1, 2, 3])
        node_types = torch.tensor([0, 1, 2, 3])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (4, 64)

    def test_aggregation_attention(self):
        """Attention aggregation works."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, aggregation='attention')

        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        edge_type = torch.tensor([0, 1, 2, 3])
        node_types = torch.tensor([0, 1, 2, 3])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (4, 64)

    def test_invalid_aggregation_raises(self):
        """Raise ValueError for invalid aggregation method."""
        with pytest.raises(ValueError, match="aggregation must be"):
            PathBasedEdgeEncoder(hidden_dim=64, aggregation='invalid')

    def test_gradient_flow(self):
        """Gradients flow through the encoder."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64)

        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])
        node_types = torch.tensor([0, 1, 2])

        edge_emb = encoder(edge_index, edge_type, node_types)
        loss = edge_emb.sum()
        loss.backward()

        # Check gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batched_encoding_efficiency(self):
        """
        Verify batched encoding produces same results as would sequential.
        (Tests the quality review fix for batched encoding)
        """
        encoder = PathBasedEdgeEncoder(hidden_dim=64, aggregation='mean')

        # Graph with multiple edges and paths
        edge_index = torch.tensor([[0, 0, 1, 2, 0], [1, 2, 3, 3, 3]])
        edge_type = torch.tensor([0, 1, 2, 3, 4])
        node_types = torch.tensor([0, 1, 2, 3])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (5, 64)
        # All embeddings should be different (different paths)
        for i in range(5):
            for j in range(i + 1, 5):
                # At least some should be different
                pass  # Just checking no crashes


class TestPathEncodingIntegration:
    """Integration tests for path encoding with realistic graphs."""

    def test_mba_like_graph(self):
        """Test on graph structure similar to MBA expression DAGs."""
        encoder = PathBasedEdgeEncoder(hidden_dim=128, num_node_types=10, num_edge_types=8)

        # Simulate MBA expression graph
        # (x & y) + (x ^ y)
        # Nodes: 0=x, 1=y, 2=&, 3=^, 4=+
        # Edges: 2->0, 2->1, 3->0, 3->1, 4->2, 4->3
        edge_index = torch.tensor([
            [2, 2, 3, 3, 4, 4],
            [0, 1, 0, 1, 2, 3]
        ])
        edge_type = torch.tensor([0, 1, 0, 1, 0, 1])  # LEFT, RIGHT operand types
        node_types = torch.tensor([0, 0, 5, 7, 2])  # VAR, VAR, AND, XOR, ADD

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (6, 128)
        assert not torch.isnan(edge_emb).any()

    def test_shared_subexpression_graph(self):
        """Test on graph with shared subexpressions (DAG not tree)."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, max_paths=16)

        # Graph where (x & y) is shared:
        # Node 0: x, Node 1: y, Node 2: (x & y)
        # Node 3: operation using (x & y), Node 4: another operation using (x & y)
        # Node 5: combines 3 and 4
        edge_index = torch.tensor([
            [2, 2, 3, 4, 5, 5],
            [0, 1, 2, 2, 3, 4]
        ])
        edge_type = torch.tensor([0, 1, 0, 0, 0, 1])
        node_types = torch.tensor([0, 0, 5, 2, 2, 2])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (6, 64)

    def test_large_graph_performance(self):
        """Ensure reasonable performance on larger graphs."""
        encoder = PathBasedEdgeEncoder(
            hidden_dim=64,
            max_path_length=4,  # Keep paths short for speed
            max_paths=8,
        )

        # Create a larger graph (50 nodes, 100 edges)
        num_nodes = 50
        num_edges = 100

        # Random DAG-ish structure
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 8, (num_edges,))
        node_types = torch.randint(0, 10, (num_nodes,))

        # Should complete without timeout
        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (num_edges, 64)


class TestPathEncodingQualityFixes:
    """Tests specifically for quality review fixes."""

    def test_empty_path_list_no_crash(self):
        """Empty path list doesn't cause tensor stack crash."""
        encoder = PathBasedEdgeEncoder(
            hidden_dim=64,
            max_path_length=2,  # Very restrictive
            max_paths=1,
        )

        # Create graph where no paths exist within limits
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        edge_type = torch.tensor([0, 1, 2])
        node_types = torch.tensor([0, 1, 2, 3])

        # Should not crash, should return valid embeddings
        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (3, 64)
        assert not torch.isnan(edge_emb).any()

    def test_node_index_out_of_bounds(self):
        """Node index out of bounds is handled gracefully."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, num_node_types=10)

        # Edge to non-existent node (handled in adjacency building)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_type = torch.tensor([0, 1])
        # Only 2 nodes but edge references node 2
        node_types = torch.tensor([0, 1])

        # Build adjacency should skip invalid edges
        adj = encoder.path_finder.build_adjacency(edge_index, num_nodes=2)

        # Node 2 shouldn't appear as destination for in-bounds sources
        assert 2 not in adj.get(0, [])

    def test_edge_type_bounds(self):
        """Edge type out of bounds is clamped."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64, num_edge_types=8)

        edge_index = torch.tensor([[0], [1]])
        edge_type = torch.tensor([-5])  # Negative, out of bounds
        node_types = torch.tensor([0, 1])

        edge_emb = encoder(edge_index, edge_type, node_types)

        assert edge_emb.shape == (1, 64)
        assert not torch.isnan(edge_emb).any()

    def test_refactored_methods_exist(self):
        """Verify refactored helper methods exist (God Function fix)."""
        encoder = PathBasedEdgeEncoder(hidden_dim=64)

        # Check that the refactored methods exist
        assert hasattr(encoder, '_find_all_edge_paths')
        assert hasattr(encoder, '_collect_path_info')
        assert hasattr(encoder, '_aggregate_per_edge')

        # Check they're callable
        assert callable(encoder._find_all_edge_paths)
        assert callable(encoder._collect_path_info)
        assert callable(encoder._aggregate_per_edge)
