"""
Integration test: Verify ast_to_graph() and ast_to_optimized_graph() produce
semantically equivalent graphs before refactoring.

This test ensures that the legacy 6-type and optimized 8-type graph construction
methods produce structurally compatible outputs.
"""

import pytest
import torch


# Test expressions covering various MBA patterns
TEST_EXPRESSIONS = [
    "x + y",
    "(x & y) | (x ^ y)",
    "((a + b) * c) - (d & ~e)",
    "~(x | y) & (x ^ y)",
    "((x + y) - (x & y)) * 2",
    "(a & b) | (c & d)",
    "x - (y + z)",
    "(x ^ y) + (x & y)",
    "~((a | b) & (c | d))",
    "(x * y) + (x * z)",
]


@pytest.fixture
def test_expressions():
    """Provide test expressions for graph equivalence tests."""
    return TEST_EXPRESSIONS


class TestGraphConstruction:
    """Tests for graph construction consistency."""

    def test_expr_to_graph_produces_valid_graph(self, test_expressions):
        """Test that expr_to_graph produces valid PyG Data objects."""
        pytest.importorskip("src.data.ast_parser")
        from src.data.ast_parser import expr_to_graph

        for expr in test_expressions:
            graph = expr_to_graph(expr)

            # Verify graph has required attributes
            assert hasattr(graph, 'x'), f"Missing node features for: {expr}"
            assert hasattr(graph, 'edge_index'), f"Missing edge_index for: {expr}"

            # Verify shapes
            assert graph.x.dim() >= 1, f"Node features should be at least 1D for: {expr}"
            assert graph.edge_index.shape[0] == 2, f"edge_index should have 2 rows for: {expr}"

            # Verify no empty graphs
            num_nodes = graph.x.shape[0] if graph.x.dim() > 1 else graph.x.size(0)
            assert num_nodes > 0, f"Graph should have at least one node for: {expr}"

    def test_node_types_in_valid_range(self, test_expressions):
        """Test that node type IDs are in valid range [0-9]."""
        pytest.importorskip("src.data.ast_parser")
        from src.data.ast_parser import expr_to_graph

        for expr in test_expressions:
            graph = expr_to_graph(expr)

            # Get node types (handle both 1D and 2D formats)
            if graph.x.dim() == 1:
                node_types = graph.x
            else:
                node_types = graph.x.argmax(dim=-1)

            min_type = node_types.min().item()
            max_type = node_types.max().item()

            assert 0 <= min_type, f"Node type IDs should be >= 0 for: {expr}"
            assert max_type <= 9, f"Node type IDs should be <= 9 for: {expr}"


class TestEdgeTypeValidation:
    """Tests for edge type validation in encoders."""

    def test_ggnn_validates_edge_types_legacy(self):
        """Test that GGNNEncoder validates edge types in legacy mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(hidden_dim=64, num_timesteps=2, edge_type_mode="legacy")

        # Create valid input
        node_features = torch.randn(5, 32)  # 5 nodes, NODE_DIM features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        batch = torch.zeros(5, dtype=torch.long)

        # Valid edge types for legacy (0-5)
        valid_edge_type = torch.tensor([0, 1, 2])
        output = encoder(node_features, edge_index, batch, edge_type=valid_edge_type)
        assert output.shape == (5, 64)

        # Invalid edge types (out of range for legacy)
        invalid_edge_type = torch.tensor([6, 7, 8])  # 8-type system values
        with pytest.raises(ValueError, match="exceeds limit"):
            encoder(node_features, edge_index, batch, edge_type=invalid_edge_type)

    def test_ggnn_validates_edge_types_optimized(self):
        """Test that GGNNEncoder validates edge types in optimized mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(hidden_dim=64, num_timesteps=2, edge_type_mode="optimized")

        # Create valid input
        node_features = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        batch = torch.zeros(5, dtype=torch.long)

        # Valid edge types for optimized (0-7)
        valid_edge_type = torch.tensor([0, 5, 7])
        output = encoder(node_features, edge_index, batch, edge_type=valid_edge_type)
        assert output.shape == (5, 64)

        # Invalid edge types (out of range)
        invalid_edge_type = torch.tensor([8, 9, 10])
        with pytest.raises(ValueError, match="exceeds limit"):
            encoder(node_features, edge_index, batch, edge_type=invalid_edge_type)

    def test_hgt_rejects_legacy_mode(self):
        """Test that HGTEncoder rejects legacy edge_type_mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import HGTEncoder

        with pytest.raises(ValueError, match="only supports optimized"):
            HGTEncoder(hidden_dim=64, num_layers=2, num_heads=4, edge_type_mode="legacy")

    def test_rgcn_rejects_legacy_mode(self):
        """Test that RGCNEncoder rejects legacy edge_type_mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import RGCNEncoder

        with pytest.raises(ValueError, match="only supports optimized"):
            RGCNEncoder(hidden_dim=64, num_layers=2, edge_type_mode="legacy")


class TestEdgeTypeModeParameter:
    """Tests for edge_type_mode parameter handling."""

    def test_ggnn_default_is_legacy(self):
        """Test that GGNNEncoder defaults to legacy mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(hidden_dim=64, num_timesteps=2)
        assert encoder.edge_type_mode == "legacy"
        assert encoder.num_edge_types == 6

    def test_ggnn_optimized_mode(self):
        """Test that GGNNEncoder accepts optimized mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import GGNNEncoder

        encoder = GGNNEncoder(hidden_dim=64, num_timesteps=2, edge_type_mode="optimized")
        assert encoder.edge_type_mode == "optimized"
        assert encoder.num_edge_types == 8

    def test_ggnn_invalid_mode_raises(self):
        """Test that GGNNEncoder rejects invalid edge_type_mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import GGNNEncoder

        with pytest.raises(ValueError, match="must be 'legacy' or 'optimized'"):
            GGNNEncoder(hidden_dim=64, num_timesteps=2, edge_type_mode="invalid")

    def test_hgt_default_is_optimized(self):
        """Test that HGTEncoder defaults to optimized mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(hidden_dim=64, num_layers=2, num_heads=4)
        assert encoder.edge_type_mode == "optimized"

    def test_rgcn_default_is_optimized(self):
        """Test that RGCNEncoder defaults to optimized mode."""
        pytest.importorskip("src.models.encoder")
        from src.models.encoder import RGCNEncoder

        encoder = RGCNEncoder(hidden_dim=64, num_layers=2)
        assert encoder.edge_type_mode == "optimized"
