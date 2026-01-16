"""
Unit tests for Graph Matching Network components.

Tests cover:
- CrossGraphAttention forward pass and masking
- MultiHeadCrossGraphAttention forward pass
- GraphMatchingNetwork end-to-end
- HGTWithGMN/GATWithGMN wrappers
- GMNBatchCollator
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.models.gmn.cross_attention import (
    CrossGraphAttention,
    MultiHeadCrossGraphAttention,
)
from src.models.gmn.graph_matching import GraphMatchingNetwork
from src.models.gmn.batch_collator import (
    GMNBatchCollator,
    GMNTripletCollator,
    create_cross_attention_mask,
)


class MockEncoder(nn.Module):
    """Mock encoder for testing GMN without real GNN."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self._hidden_dim = hidden_dim
        self.linear = nn.Linear(32, hidden_dim)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def requires_edge_types(self) -> bool:
        return False

    @property
    def requires_node_features(self) -> bool:
        return True

    def forward(self, x, edge_index, batch, edge_type=None):
        if x.dim() == 1:
            x = torch.zeros(x.size(0), 32, device=x.device)
        elif x.size(-1) != 32:
            x = torch.zeros(x.size(0), 32, device=x.device)
        return self.linear(x)


def create_random_graph(num_nodes: int, num_edges: int = None) -> Data:
    """Create random PyG Data for testing."""
    if num_edges is None:
        num_edges = num_nodes * 2

    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    return Data(x=x, edge_index=edge_index)


class TestCrossGraphAttention:
    """Tests for CrossGraphAttention module."""

    def test_forward_shapes(self):
        """Test that cross-attention produces correct output shapes."""
        hidden_dim = 256
        N1, N2 = 10, 15

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim)

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        match_vector, attn_weights = cross_attn(h1, h2)

        assert match_vector.shape == (N1, hidden_dim)
        assert attn_weights.shape == (N1, N2)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 per row."""
        hidden_dim = 256
        N1, N2 = 10, 15

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim, dropout=0.0)

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        _, attn_weights = cross_attn(h1, h2)

        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N1), atol=1e-5)

    def test_1d_masking(self):
        """Test that 1D masking correctly prevents attention to invalid nodes."""
        hidden_dim = 256
        N1, N2 = 10, 15

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim, dropout=0.0)

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        # Mask out last 5 nodes in h2
        mask2 = torch.ones(N2, dtype=torch.bool)
        mask2[-5:] = False

        _, attn_weights = cross_attn(h1, h2, mask2=mask2)

        # Attention weights should be ~0 for masked nodes
        assert torch.allclose(attn_weights[:, -5:], torch.zeros(N1, 5), atol=1e-5)

    def test_2d_masking(self):
        """Test that 2D masking (batch-aware) works correctly."""
        hidden_dim = 256
        N1, N2 = 10, 15

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim, dropout=0.0)

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        # Create a 2D mask where first half of h1 attends to first half of h2
        mask2 = torch.zeros(N1, N2, dtype=torch.bool)
        mask2[:5, :7] = True
        mask2[5:, 7:] = True

        _, attn_weights = cross_attn(h1, h2, mask2=mask2)

        # Check masked regions have ~0 attention
        assert torch.allclose(attn_weights[:5, 7:], torch.zeros(5, 8), atol=1e-5)
        assert torch.allclose(attn_weights[5:, :7], torch.zeros(5, 7), atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through cross-attention."""
        hidden_dim = 256
        N1, N2 = 10, 15

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim)

        h1 = torch.randn(N1, hidden_dim, requires_grad=True)
        h2 = torch.randn(N2, hidden_dim, requires_grad=True)

        match_vector, _ = cross_attn(h1, h2)
        loss = match_vector.sum()
        loss.backward()

        assert h1.grad is not None
        assert h2.grad is not None
        assert not torch.isnan(h1.grad).any()
        assert not torch.isnan(h2.grad).any()


class TestMultiHeadCrossGraphAttention:
    """Tests for MultiHeadCrossGraphAttention module."""

    def test_forward_shapes(self):
        """Test multi-head cross-attention output shapes."""
        hidden_dim = 256
        num_heads = 8
        N1, N2 = 10, 15

        cross_attn = MultiHeadCrossGraphAttention(
            hidden_dim=hidden_dim, num_heads=num_heads
        )

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        match_vector, attn_weights = cross_attn(h1, h2)

        assert match_vector.shape == (N1, hidden_dim)
        assert attn_weights.shape == (num_heads, N1, N2)

    def test_dimension_validation(self):
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadCrossGraphAttention(hidden_dim=256, num_heads=7)

    def test_multi_head_attention_weights(self):
        """Test that each head has valid attention weights."""
        hidden_dim = 256
        num_heads = 8
        N1, N2 = 10, 15

        cross_attn = MultiHeadCrossGraphAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0
        )

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        _, attn_weights = cross_attn(h1, h2)

        # Each head's attention should sum to 1 per row
        for head in range(num_heads):
            row_sums = attn_weights[head].sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones(N1), atol=1e-5)


class TestGraphMatchingNetwork:
    """Tests for GraphMatchingNetwork class."""

    def test_forward_shapes(self):
        """Test GMN forward pass output shapes."""
        hidden_dim = 256
        encoder = MockEncoder(hidden_dim=hidden_dim)
        gmn = GraphMatchingNetwork(
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_attention_layers=2,
            num_heads=8,
        )

        # Create batched graphs
        graph1 = create_random_graph(10)
        graph2 = create_random_graph(15)

        batch1 = Batch.from_data_list([graph1])
        batch2 = Batch.from_data_list([graph2])

        match_score = gmn(batch1, batch2)

        assert match_score.shape == (1, 1)
        assert 0.0 <= match_score.item() <= 1.0

    def test_batched_forward(self):
        """Test GMN with multiple graph pairs in batch."""
        hidden_dim = 256
        encoder = MockEncoder(hidden_dim=hidden_dim)
        gmn = GraphMatchingNetwork(
            encoder=encoder,
            hidden_dim=hidden_dim,
        )

        # Create batch of 3 graph pairs
        graphs1 = [create_random_graph(8), create_random_graph(10), create_random_graph(12)]
        graphs2 = [create_random_graph(6), create_random_graph(8), create_random_graph(10)]

        batch1 = Batch.from_data_list(graphs1)
        batch2 = Batch.from_data_list(graphs2)

        match_scores = gmn(batch1, batch2)

        assert match_scores.shape == (3, 1)
        assert (match_scores >= 0).all() and (match_scores <= 1).all()

    def test_return_attention(self):
        """Test that attention weights can be returned."""
        hidden_dim = 256
        encoder = MockEncoder(hidden_dim=hidden_dim)
        gmn = GraphMatchingNetwork(
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_attention_layers=2,
        )

        graph1 = create_random_graph(10)
        graph2 = create_random_graph(15)

        batch1 = Batch.from_data_list([graph1])
        batch2 = Batch.from_data_list([graph2])

        match_score, attn_dict = gmn(batch1, batch2, return_attention=True)

        assert 'layer_0' in attn_dict
        assert 'layer_1' in attn_dict
        assert 'h1_to_h2' in attn_dict['layer_0']
        assert 'h2_to_h1' in attn_dict['layer_0']

    @pytest.mark.parametrize("aggregation", ["mean", "max", "mean_max", "attention"])
    def test_aggregation_methods(self, aggregation):
        """Test different aggregation methods."""
        hidden_dim = 256
        encoder = MockEncoder(hidden_dim=hidden_dim)
        gmn = GraphMatchingNetwork(
            encoder=encoder,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
        )

        graph1 = create_random_graph(10)
        graph2 = create_random_graph(15)

        batch1 = Batch.from_data_list([graph1])
        batch2 = Batch.from_data_list([graph2])

        match_score = gmn(batch1, batch2)

        assert match_score.shape == (1, 1)
        assert torch.isfinite(match_score).all()

    def test_gradient_flow(self):
        """Test that gradients flow through entire GMN."""
        hidden_dim = 256
        encoder = MockEncoder(hidden_dim=hidden_dim)
        gmn = GraphMatchingNetwork(
            encoder=encoder,
            hidden_dim=hidden_dim,
        )

        graph1 = create_random_graph(10)
        graph2 = create_random_graph(15)

        batch1 = Batch.from_data_list([graph1])
        batch2 = Batch.from_data_list([graph2])

        match_score = gmn(batch1, batch2)
        loss = match_score.sum()
        loss.backward()

        # Check gradients exist
        for param in gmn.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGMNBatchCollator:
    """Tests for GMNBatchCollator."""

    def test_collation(self):
        """Test basic collation of graph pairs."""
        collator = GMNBatchCollator()

        # Create batch of pairs
        batch_list = [
            (create_random_graph(5), create_random_graph(3), 1),
            (create_random_graph(8), create_random_graph(6), 0),
            (create_random_graph(10), create_random_graph(4), 1),
        ]

        graph1_batch, graph2_batch, labels, pair_indices = collator(batch_list)

        assert graph1_batch.num_graphs == 3
        assert graph2_batch.num_graphs == 3
        assert labels.shape == (3,)
        assert pair_indices.shape == (3, 2)
        assert labels.tolist() == [1.0, 0.0, 1.0]

    def test_variable_sizes(self):
        """Test that variable-size graphs are handled correctly."""
        collator = GMNBatchCollator()

        # Highly variable sizes
        batch_list = [
            (create_random_graph(2), create_random_graph(100), 1),
            (create_random_graph(50), create_random_graph(5), 0),
        ]

        graph1_batch, graph2_batch, labels, _ = collator(batch_list)

        # Check total nodes
        assert graph1_batch.x.size(0) == 2 + 50  # Total nodes in graph1s
        assert graph2_batch.x.size(0) == 100 + 5  # Total nodes in graph2s


class TestGMNTripletCollator:
    """Tests for GMNTripletCollator."""

    def test_triplet_collation(self):
        """Test collation of triplets."""
        collator = GMNTripletCollator()

        batch_list = [
            (create_random_graph(5), create_random_graph(3), create_random_graph(7)),
            (create_random_graph(8), create_random_graph(6), create_random_graph(10)),
        ]

        anchor_batch, positive_batch, negative_batch = collator(batch_list)

        assert anchor_batch.num_graphs == 2
        assert positive_batch.num_graphs == 2
        assert negative_batch.num_graphs == 2


class TestCrossAttentionMask:
    """Tests for create_cross_attention_mask utility."""

    def test_mask_creation(self):
        """Test that mask correctly separates batch pairs."""
        # Batch indices: 3 nodes in graph 0, 2 nodes in graph 1
        batch1 = torch.tensor([0, 0, 0, 1, 1])
        batch2 = torch.tensor([0, 0, 1, 1, 1])

        mask = create_cross_attention_mask(batch1, batch2)

        expected = torch.tensor([
            [True, True, False, False, False],   # Node 0 (batch 0) -> batch 0 nodes
            [True, True, False, False, False],   # Node 1 (batch 0) -> batch 0 nodes
            [True, True, False, False, False],   # Node 2 (batch 0) -> batch 0 nodes
            [False, False, True, True, True],    # Node 3 (batch 1) -> batch 1 nodes
            [False, False, True, True, True],    # Node 4 (batch 1) -> batch 1 nodes
        ])

        assert torch.equal(mask, expected)


class TestNaNHandling:
    """Tests for NaN handling in edge cases."""

    def test_all_masked_row(self):
        """Test handling when all attention targets are masked."""
        hidden_dim = 256
        N1, N2 = 5, 10

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim, dropout=0.0)

        h1 = torch.randn(N1, hidden_dim)
        h2 = torch.randn(N2, hidden_dim)

        # Mask all nodes for first row
        mask2 = torch.ones(N1, N2, dtype=torch.bool)
        mask2[0, :] = False  # First row has no valid targets

        match_vector, attn_weights = cross_attn(h1, h2, mask2=mask2)

        # Should not produce NaN
        assert not torch.isnan(match_vector).any()
        assert not torch.isnan(attn_weights).any()

    def test_empty_graph(self):
        """Test handling of very small graphs."""
        hidden_dim = 256

        cross_attn = CrossGraphAttention(hidden_dim=hidden_dim, dropout=0.0)

        h1 = torch.randn(1, hidden_dim)  # Single node
        h2 = torch.randn(1, hidden_dim)  # Single node

        match_vector, attn_weights = cross_attn(h1, h2)

        assert match_vector.shape == (1, hidden_dim)
        assert attn_weights.shape == (1, 1)
        assert not torch.isnan(match_vector).any()
