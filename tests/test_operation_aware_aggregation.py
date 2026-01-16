"""
Tests for operation-aware aggregation in HGT encoder.

Covers:
- Commutative operations (ADD, MUL, AND, OR, XOR) use sum aggregation
- Non-commutative operations (SUB) use concat+project aggregation
- Order preservation for non-commutative operations
- Feature flag behavior (operation_aware=True/False)
- Edge cases: unary ops, empty graphs, batched graphs, no-operation-node graphs
- Error handling for malformed graphs
"""

import pytest
import torch
import torch.nn as nn

from src.models.operation_aware_aggregator import OperationAwareAggregator
from src.models.edge_types import EdgeType, NodeType
from src.constants import COMMUTATIVE_OPS, NON_COMMUTATIVE_OPS


class TestOperationAwareAggregator:
    """Tests for OperationAwareAggregator standalone."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator with default settings."""
        return OperationAwareAggregator(hidden_dim=64, dropout=0.0, strict_validation=True)

    @pytest.fixture
    def permissive_aggregator(self):
        """Create aggregator with permissive validation (logs warnings instead of raising)."""
        return OperationAwareAggregator(hidden_dim=64, dropout=0.0, strict_validation=False)

    def test_commutative_add_uses_sum(self, aggregator):
        """Verify ADD operation uses sum aggregation (order-invariant)."""
        # Graph: x + y (3 nodes: VAR, VAR, ADD)
        # Node types: VAR=0, CONST=1, ADD=2
        node_types = torch.tensor([0, 0, 2])  # VAR, VAR, ADD
        node_features = torch.randn(3, 64)
        messages = node_features.clone()

        # Edges: VAR_0 -> ADD (left), VAR_1 -> ADD (right)
        edge_index = torch.tensor([
            [0, 1],  # src
            [2, 2],  # dst
        ])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # ADD node should have sum of children
        expected_sum = node_features[0] + node_features[1]
        assert torch.allclose(output[2], expected_sum, atol=1e-5)

        # VAR nodes should be unchanged
        assert torch.allclose(output[0], messages[0], atol=1e-5)
        assert torch.allclose(output[1], messages[1], atol=1e-5)

    def test_commutative_ops_order_invariant(self, aggregator):
        """Verify all commutative ops produce same result regardless of operand order."""
        for op_name in COMMUTATIVE_OPS:
            # Map op name to node type
            op_type = getattr(NodeType, op_name).value

            node_types = torch.tensor([0, 0, op_type])  # VAR, VAR, OP
            node_features = torch.randn(3, 64)
            messages = node_features.clone()

            # Order 1: VAR_0 left, VAR_1 right
            edge_index_1 = torch.tensor([[0, 1], [2, 2]])
            edge_types = torch.tensor([
                EdgeType.LEFT_OPERAND.value,
                EdgeType.RIGHT_OPERAND.value,
            ])
            output_1 = aggregator(node_features, edge_index_1, edge_types, node_types, messages)

            # Order 2: VAR_1 left, VAR_0 right (swap operands)
            edge_index_2 = torch.tensor([[1, 0], [2, 2]])
            output_2 = aggregator(node_features, edge_index_2, edge_types, node_types, messages)

            # Results should be identical for commutative ops
            assert torch.allclose(output_1[2], output_2[2], atol=1e-5), (
                f"{op_name} should be order-invariant"
            )

    def test_non_commutative_sub_uses_projection(self, aggregator):
        """Verify SUB operation uses concat+project (not simple sum)."""
        # Graph: x - y (3 nodes: VAR, VAR, SUB)
        node_types = torch.tensor([0, 0, 3])  # VAR, VAR, SUB
        node_features = torch.randn(3, 64)
        messages = node_features.clone()

        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # SUB node output should differ from simple sum
        simple_sum = node_features[0] + node_features[1]
        assert not torch.allclose(output[2], simple_sum, atol=1e-3), (
            "SUB should NOT use simple sum"
        )

        # Output shape should be preserved
        assert output[2].shape == (64,)

    def test_non_commutative_order_preservation(self, aggregator):
        """Verify SUB(x, y) != SUB(y, x) - order matters."""
        # Graph 1: x - y
        node_types = torch.tensor([0, 0, 3])  # VAR, VAR, SUB
        node_features = torch.randn(3, 64)
        messages = node_features.clone()

        # x - y: x is left, y is right
        edge_index_xy = torch.tensor([[0, 1], [2, 2]])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])
        output_xy = aggregator(node_features, edge_index_xy, edge_types, node_types, messages)

        # y - x: y is left, x is right
        edge_index_yx = torch.tensor([[1, 0], [2, 2]])
        output_yx = aggregator(node_features, edge_index_yx, edge_types, node_types, messages)

        # SUB results should differ (order-preserving)
        assert not torch.allclose(output_xy[2], output_yx[2], atol=1e-3), (
            "SUB(x,y) should differ from SUB(y,x)"
        )

    def test_unary_operations_pass_through(self, aggregator):
        """Verify unary ops (NOT, NEG) pass through unchanged."""
        # Graph: ~x (2 nodes: VAR, NOT)
        node_types = torch.tensor([0, 8])  # VAR, NOT
        node_features = torch.randn(2, 64)
        messages = node_features.clone()

        # Unary edge: VAR -> NOT
        edge_index = torch.tensor([[0], [1]])
        edge_types = torch.tensor([EdgeType.UNARY_OPERAND.value])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # NOT node should be unchanged (unary, no left/right operands)
        assert torch.allclose(output[1], messages[1], atol=1e-5)

    def test_terminal_nodes_pass_through(self, aggregator):
        """Verify VAR and CONST nodes pass through unchanged."""
        # Graph: just VAR and CONST (no edges)
        node_types = torch.tensor([0, 1])  # VAR, CONST
        node_features = torch.randn(2, 64)
        messages = node_features.clone()

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # All nodes unchanged
        assert torch.allclose(output, messages, atol=1e-5)

    def test_empty_graph(self, aggregator):
        """Verify empty graph returns empty tensor without error."""
        node_types = torch.empty((0,), dtype=torch.long)
        node_features = torch.empty((0, 64))
        messages = node_features.clone()

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        assert output.shape == (0, 64)

    def test_no_operation_nodes(self, aggregator):
        """Verify graph with only VAR/CONST nodes returns messages unchanged."""
        # Graph: just terminals
        node_types = torch.tensor([0, 0, 1, 1])  # VAR, VAR, CONST, CONST
        node_features = torch.randn(4, 64)
        messages = node_features.clone()

        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # All unchanged
        assert torch.allclose(output, messages, atol=1e-5)

    def test_missing_operand_edges_strict_raises(self, aggregator):
        """Verify missing operand edges raises ValueError in strict mode."""
        # Graph: ADD node with only LEFT operand (missing RIGHT)
        node_types = torch.tensor([0, 2])  # VAR, ADD
        node_features = torch.randn(2, 64)
        messages = node_features.clone()

        # Only LEFT edge, no RIGHT
        edge_index = torch.tensor([[0], [1]])
        edge_types = torch.tensor([EdgeType.LEFT_OPERAND.value])

        with pytest.raises(ValueError, match="missing operand edges"):
            aggregator(node_features, edge_index, edge_types, node_types, messages)

    def test_missing_operand_edges_permissive_warns(self, permissive_aggregator, caplog):
        """Verify missing operand edges logs warning in permissive mode."""
        node_types = torch.tensor([0, 2])  # VAR, ADD
        node_features = torch.randn(2, 64)
        messages = node_features.clone()

        edge_index = torch.tensor([[0], [1]])
        edge_types = torch.tensor([EdgeType.LEFT_OPERAND.value])

        import logging
        with caplog.at_level(logging.WARNING):
            output = permissive_aggregator(node_features, edge_index, edge_types, node_types, messages)

        # Should not raise, but log warning
        assert "missing operand edges" in caplog.text
        # ADD node should be unchanged (skipped)
        assert torch.allclose(output[1], messages[1], atol=1e-5)

    def test_duplicate_operand_edges_strict_raises(self, aggregator):
        """Verify duplicate operand edges raises ValueError in strict mode."""
        # Graph: ADD with two LEFT operands AND one RIGHT (invalid - duplicate LEFT)
        node_types = torch.tensor([0, 0, 0, 2])  # VAR, VAR, VAR, ADD
        node_features = torch.randn(4, 64)
        messages = node_features.clone()

        # Two LEFT edges, one RIGHT - invalid because LEFT has duplicates
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.LEFT_OPERAND.value,  # Duplicate LEFT!
            EdgeType.RIGHT_OPERAND.value,
        ])

        with pytest.raises(ValueError, match="invalid operand edge count"):
            aggregator(node_features, edge_index, edge_types, node_types, messages)

    def test_batched_graphs_flat_format(self, aggregator):
        """Verify batched graphs (PyG flattened) work correctly."""
        # Two graphs concatenated:
        # Graph 1: x + y (nodes 0,1,2)
        # Graph 2: a - b (nodes 3,4,5)

        node_types = torch.tensor([0, 0, 2, 0, 0, 3])  # VAR,VAR,ADD, VAR,VAR,SUB
        node_features = torch.randn(6, 64)
        messages = node_features.clone()

        # Edges for both graphs (adjusted indices for graph 2)
        edge_index = torch.tensor([
            [0, 1, 3, 4],  # src
            [2, 2, 5, 5],  # dst
        ])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)

        # Graph 1: ADD should have sum
        expected_add = node_features[0] + node_features[1]
        assert torch.allclose(output[2], expected_add, atol=1e-5)

        # Graph 2: SUB should NOT have simple sum
        simple_sub_sum = node_features[3] + node_features[4]
        assert not torch.allclose(output[5], simple_sub_sum, atol=1e-3)

    def test_shape_validation_3d_input_fails(self, aggregator):
        """Verify 3D input (unbatched) raises assertion error."""
        node_types = torch.tensor([0, 0, 2])
        node_features = torch.randn(1, 3, 64)  # 3D - wrong!
        messages = node_features.clone()

        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_types = torch.tensor([0, 1])

        with pytest.raises(AssertionError, match="Expected 2D node_features"):
            aggregator(node_features, edge_index, edge_types, node_types, messages)


class TestHGTEncoderOperationAware:
    """Tests for HGTEncoder with operation_aware flag."""

    @pytest.fixture(autouse=True)
    def skip_if_no_pyg(self):
        """Skip test if torch_geometric/torch_scatter is not available."""
        pytest.importorskip("torch_geometric")
        pytest.importorskip("torch_scatter")

    def test_operation_aware_false_no_aggregator(self):
        """Verify operation_aware=False does not create aggregator."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            operation_aware=False,
        )

        assert encoder.aggregator is None
        assert encoder.operation_aware is False

    def test_operation_aware_true_creates_aggregator(self):
        """Verify operation_aware=True creates aggregator."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            operation_aware=True,
        )

        assert encoder.aggregator is not None
        assert isinstance(encoder.aggregator, OperationAwareAggregator)
        assert encoder.operation_aware is True

    def test_operation_aware_strict_validation_passthrough(self):
        """Verify operation_aware_strict parameter is passed to aggregator."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            operation_aware=True,
            operation_aware_strict=False,
        )

        assert encoder.aggregator.strict_validation is False

    def test_forward_with_operation_aware(self):
        """Verify forward pass works with operation_aware=True."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            operation_aware=True,
            operation_aware_strict=False,  # Permissive for test flexibility
        )

        # Simple graph: x + y (nodes: VAR_0, VAR_1, ADD_2)
        # Edge directions: LEFT_OPERAND goes parent->child (ADD->VAR)
        node_types = torch.tensor([0, 0, 2])  # VAR, VAR, ADD
        batch = torch.tensor([0, 0, 0])

        # Edges: ADD(2)->VAR(0) left, ADD(2)->VAR(1) right, and inverses
        edge_index = torch.tensor([
            [2, 2, 0, 1],  # src: ADD, ADD, VAR, VAR
            [0, 1, 2, 2],  # dst: VAR, VAR, ADD, ADD
        ])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,      # ADD -> VAR (left child)
            EdgeType.RIGHT_OPERAND.value,     # ADD -> VAR (right child)
            EdgeType.LEFT_OPERAND_INV.value,  # VAR -> ADD (inverse)
            EdgeType.RIGHT_OPERAND_INV.value, # VAR -> ADD (inverse)
        ])

        output = encoder(node_types, edge_index, batch, edge_types)

        # Should produce valid output
        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()

    def test_forward_without_operation_aware(self):
        """Verify forward pass works with operation_aware=False (baseline)."""
        from src.models.encoder import HGTEncoder

        encoder = HGTEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            operation_aware=False,
        )

        # Same graph structure as test_forward_with_operation_aware
        node_types = torch.tensor([0, 0, 2])  # VAR, VAR, ADD
        batch = torch.tensor([0, 0, 0])

        # Edges: ADD(2)->VAR(0) left, ADD(2)->VAR(1) right, and inverses
        edge_index = torch.tensor([
            [2, 2, 0, 1],  # src: ADD, ADD, VAR, VAR
            [0, 1, 2, 2],  # dst: VAR, VAR, ADD, ADD
        ])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,      # ADD -> VAR (left child)
            EdgeType.RIGHT_OPERAND.value,     # ADD -> VAR (right child)
            EdgeType.LEFT_OPERAND_INV.value,  # VAR -> ADD (inverse)
            EdgeType.RIGHT_OPERAND_INV.value, # VAR -> ADD (inverse)
        ])

        output = encoder(node_types, edge_index, batch, edge_types)

        assert output.shape == (3, 64)
        assert not torch.isnan(output).any()

    def test_parameter_count_difference(self):
        """Verify operation_aware=True adds expected parameters (~1.18M for 768d)."""
        from src.models.encoder import HGTEncoder

        encoder_off = HGTEncoder(
            hidden_dim=768,
            num_layers=2,
            num_heads=4,
            operation_aware=False,
        )

        encoder_on = HGTEncoder(
            hidden_dim=768,
            num_layers=2,
            num_heads=4,
            operation_aware=True,
        )

        params_off = sum(p.numel() for p in encoder_off.parameters())
        params_on = sum(p.numel() for p in encoder_on.parameters())

        # non_comm_proj adds: Linear(1536, 768) + LayerNorm(768)
        # = 1536*768 + 768 + 768*2 = 1,179,648 + 768 + 1,536 = ~1.18M
        expected_diff = 2 * 768 * 768 + 768 + 768 * 2  # Linear weights + bias + LayerNorm
        actual_diff = params_on - params_off

        assert abs(actual_diff - expected_diff) < 100, (
            f"Expected ~{expected_diff} additional params, got {actual_diff}"
        )


class TestAggregatorGradientFlow:
    """Tests for gradient flow through aggregator."""

    def test_gradients_flow_through_commutative(self):
        """Verify gradients flow through commutative aggregation."""
        aggregator = OperationAwareAggregator(hidden_dim=64, dropout=0.0)

        node_types = torch.tensor([0, 0, 2])
        node_features = torch.randn(3, 64, requires_grad=True)
        messages = node_features.clone()

        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to input features
        assert node_features.grad is not None
        assert not torch.allclose(node_features.grad, torch.zeros_like(node_features.grad))

    def test_gradients_flow_through_non_commutative(self):
        """Verify gradients flow through non-commutative projection."""
        aggregator = OperationAwareAggregator(hidden_dim=64, dropout=0.0)

        node_types = torch.tensor([0, 0, 3])  # SUB
        node_features = torch.randn(3, 64, requires_grad=True)
        messages = node_features.clone()

        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_types = torch.tensor([
            EdgeType.LEFT_OPERAND.value,
            EdgeType.RIGHT_OPERAND.value,
        ])

        output = aggregator(node_features, edge_index, edge_types, node_types, messages)
        loss = output.sum()
        loss.backward()

        # Gradients should flow to input features AND projection weights
        assert node_features.grad is not None
        assert not torch.allclose(node_features.grad, torch.zeros_like(node_features.grad))

        # Check projection layer has gradients
        for param in aggregator.non_comm_proj.parameters():
            if param.requires_grad:
                assert param.grad is not None
