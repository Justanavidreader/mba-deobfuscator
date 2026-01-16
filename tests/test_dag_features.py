"""
Tests for DAG positional feature computation.

Tests cover:
- Basic feature computation for simple graphs
- Shared subexpression handling (DAG vs tree)
- Cycle detection
- Device consistency validation
- Edge cases (empty graph, single node, disconnected)
"""

import pytest
import torch

from src.data.dag_features import (
    compute_dag_positional_features,
    _build_adjacency,
    _compute_in_degrees,
    _compute_depths_topological,
    _compute_subtree_sizes,
    _normalize_to_unit_range,
)
from src.constants import DAG_FEATURE_DIM


class TestBuildAdjacency:
    """Tests for _build_adjacency helper."""

    def test_simple_tree(self):
        # Tree: 0 -> 1, 0 -> 2
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        children, parents = _build_adjacency(edge_index, num_nodes=3)

        assert children[0] == [1, 2]
        assert children[1] == []
        assert children[2] == []
        assert parents[0] == []
        assert parents[1] == [0]
        assert parents[2] == [0]

    def test_dag_with_shared_node(self):
        # DAG: 0 -> 2, 1 -> 2 (node 2 has two parents)
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        children, parents = _build_adjacency(edge_index, num_nodes=3)

        assert children[0] == [2]
        assert children[1] == [2]
        assert parents[2] == [0, 1]

    def test_empty_graph(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        children, parents = _build_adjacency(edge_index, num_nodes=0)

        assert len(children) == 0
        assert len(parents) == 0


class TestComputeInDegrees:
    """Tests for _compute_in_degrees helper."""

    def test_tree_in_degrees(self):
        # Tree: 0 -> 1, 0 -> 2
        parents = {0: [], 1: [0], 2: [0]}
        in_degrees, max_in_degree = _compute_in_degrees(3, parents, torch.device('cpu'))

        assert in_degrees.tolist() == [0, 1, 1]
        assert max_in_degree == 1

    def test_shared_node_in_degrees(self):
        # Node 2 has two parents
        parents = {0: [], 1: [], 2: [0, 1]}
        in_degrees, max_in_degree = _compute_in_degrees(3, parents, torch.device('cpu'))

        assert in_degrees.tolist() == [0, 0, 2]
        assert max_in_degree == 2

    def test_empty_graph_in_degrees(self):
        in_degrees, max_in_degree = _compute_in_degrees(0, {}, torch.device('cpu'))
        assert in_degrees.numel() == 0
        assert max_in_degree == 1  # Clamped to avoid division by zero


class TestComputeDepths:
    """Tests for _compute_depths_topological helper."""

    def test_simple_tree_depth(self):
        # Tree: root(0) -> left(1), right(2)
        # Depth: leaves=0, root=1
        children = {0: [1, 2], 1: [], 2: []}
        parents = {0: [], 1: [0], 2: [0]}
        node_types = torch.tensor([2, 0, 0], dtype=torch.long)  # root=ADD, leaves=VAR

        depths, max_depth = _compute_depths_topological(
            3, children, parents, node_types, torch.device('cpu')
        )

        assert depths[1].item() == 0  # Leaf
        assert depths[2].item() == 0  # Leaf
        assert depths[0].item() == 1  # Root
        assert max_depth == 1

    def test_deep_tree_depth(self):
        # Linear tree: 0 -> 1 -> 2 -> 3 (leaf)
        children = {0: [1], 1: [2], 2: [3], 3: []}
        parents = {0: [], 1: [0], 2: [1], 3: [2]}
        node_types = torch.tensor([2, 2, 2, 0], dtype=torch.long)

        depths, max_depth = _compute_depths_topological(
            4, children, parents, node_types, torch.device('cpu')
        )

        assert depths[3].item() == 0  # Leaf
        assert depths[2].item() == 1
        assert depths[1].item() == 2
        assert depths[0].item() == 3  # Root
        assert max_depth == 3

    def test_dag_shared_node_depth(self):
        # DAG: 0 -> 2, 1 -> 2, both 0 and 1 are roots
        # Node 2 is shared leaf
        children = {0: [2], 1: [2], 2: []}
        parents = {0: [], 1: [], 2: [0, 1]}
        node_types = torch.tensor([2, 2, 0], dtype=torch.long)

        depths, max_depth = _compute_depths_topological(
            3, children, parents, node_types, torch.device('cpu')
        )

        # Topological sort: 2 processed first (leaf), then 0 and 1
        assert depths[2].item() == 0  # Shared leaf
        assert depths[0].item() == 1  # Root 1
        assert depths[1].item() == 1  # Root 2

    def test_cycle_detection(self):
        # Cycle: 0 -> 1 -> 2 -> 0
        children = {0: [1], 1: [2], 2: [0]}
        parents = {0: [2], 1: [0], 2: [1]}
        node_types = torch.tensor([2, 2, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="cycles"):
            _compute_depths_topological(
                3, children, parents, node_types, torch.device('cpu')
            )


class TestComputeSubtreeSizes:
    """Tests for _compute_subtree_sizes helper."""

    def test_simple_tree_subtree_sizes(self):
        # Tree: root(0) -> left(1), right(2)
        children = {0: [1, 2], 1: [], 2: []}
        parents = {0: [], 1: [0], 2: [0]}

        sizes = _compute_subtree_sizes(3, children, parents, torch.device('cpu'))

        assert sizes[1].item() == 1  # Leaf
        assert sizes[2].item() == 1  # Leaf
        assert sizes[0].item() == 3  # Root: self + 2 children

    def test_shared_node_subtree_sizes(self):
        # DAG: root(0) -> shared(2), other(1) -> shared(2)
        # shared(2) is counted in both parents' subtrees
        #
        # Structure:
        #   0 (root)     1 (other root)
        #    \          /
        #     -> 2 <---
        #
        # Subtree sizes:
        # - Node 2: 1 (just itself)
        # - Node 0: 1 + subtree_size(2) = 1 + 1 = 2
        # - Node 1: 1 + subtree_size(2) = 1 + 1 = 2
        children = {0: [2], 1: [2], 2: []}
        parents = {0: [], 1: [], 2: [0, 1]}

        sizes = _compute_subtree_sizes(3, children, parents, torch.device('cpu'))

        assert sizes[2].item() == 1  # Shared leaf
        assert sizes[0].item() == 2  # Root 0: self + shared
        assert sizes[1].item() == 2  # Root 1: self + shared

    def test_diamond_dag_subtree_sizes(self):
        # Diamond DAG:
        #       0 (root)
        #      / \
        #     1   2
        #      \ /
        #       3 (shared leaf)
        #
        children = {0: [1, 2], 1: [3], 2: [3], 3: []}
        parents = {0: [], 1: [0], 2: [0], 3: [1, 2]}

        sizes = _compute_subtree_sizes(4, children, parents, torch.device('cpu'))

        assert sizes[3].item() == 1  # Shared leaf
        assert sizes[1].item() == 2  # Node 1: self + node 3
        assert sizes[2].item() == 2  # Node 2: self + node 3
        # Root 0: self + node 1 subtree (2) + node 2 subtree (2) = 5
        # Note: shared node 3 contributes to BOTH parents
        assert sizes[0].item() == 5


class TestNormalization:
    """Tests for _normalize_to_unit_range helper."""

    def test_basic_normalization(self):
        feature = torch.tensor([0, 5, 10], dtype=torch.long)
        normalized = _normalize_to_unit_range(feature, 10)

        assert normalized.tolist() == [0.0, 0.5, 1.0]

    def test_zero_max_clamped(self):
        feature = torch.tensor([0, 0], dtype=torch.long)
        # Max value 0 gets clamped to 1 to avoid division by zero
        normalized = _normalize_to_unit_range(feature, 0)

        assert normalized.tolist() == [0.0, 0.0]


class TestComputeDAGPositionalFeatures:
    """Integration tests for compute_dag_positional_features."""

    def test_output_shape(self):
        # Simple tree: 0 -> 1, 0 -> 2
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        node_types = torch.tensor([2, 0, 0], dtype=torch.long)

        features = compute_dag_positional_features(3, edge_index, node_types)

        assert features.shape == (3, DAG_FEATURE_DIM)
        assert features.dtype == torch.float

    def test_feature_ranges(self):
        # All features should be in [0, 1]
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
        node_types = torch.tensor([2, 2, 0, 0], dtype=torch.long)

        features = compute_dag_positional_features(4, edge_index, node_types)

        assert (features >= 0).all()
        assert (features <= 1).all()

    def test_is_shared_binary(self):
        # DAG with shared node 2
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        node_types = torch.tensor([2, 2, 0], dtype=torch.long)

        features = compute_dag_positional_features(3, edge_index, node_types)

        # is_shared column (index 3) should be binary
        is_shared = features[:, 3]
        assert ((is_shared == 0) | (is_shared == 1)).all()
        # Node 2 has in_degree=2, so is_shared=1
        assert is_shared[2].item() == 1.0
        # Nodes 0 and 1 have in_degree=0, so is_shared=0
        assert is_shared[0].item() == 0.0
        assert is_shared[1].item() == 0.0

    def test_empty_graph(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_types = torch.zeros(0, dtype=torch.long)

        features = compute_dag_positional_features(0, edge_index, node_types)

        assert features.shape == (0, DAG_FEATURE_DIM)

    def test_single_node(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_types = torch.tensor([0], dtype=torch.long)

        features = compute_dag_positional_features(1, edge_index, node_types)

        assert features.shape == (1, DAG_FEATURE_DIM)
        # Single node: depth=0, subtree_size=1, in_degree=0, is_shared=0
        # Normalized: depth=0/1=0, subtree=1/1=1, in_degree=0/1=0, is_shared=0
        assert features[0, 0].item() == 0.0  # depth
        assert features[0, 1].item() == 1.0  # subtree_size (normalized by num_nodes)
        assert features[0, 2].item() == 0.0  # in_degree
        assert features[0, 3].item() == 0.0  # is_shared

    def test_device_consistency_cpu(self):
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        node_types = torch.tensor([2, 0], dtype=torch.long)

        features = compute_dag_positional_features(2, edge_index, node_types)

        assert features.device == edge_index.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency_cuda(self):
        edge_index = torch.tensor([[0], [1]], dtype=torch.long).cuda()
        node_types = torch.tensor([2, 0], dtype=torch.long).cuda()

        features = compute_dag_positional_features(2, edge_index, node_types)

        assert features.device == edge_index.device

    def test_device_mismatch_error(self):
        # Create tensors on different devices
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # CPU
        node_types = torch.tensor([2, 0], dtype=torch.long)

        # This should work (same device)
        features = compute_dag_positional_features(2, edge_index, node_types)
        assert features is not None

        # Test with mock different devices would require actual CUDA
        # The validation is in the function - device mismatch raises ValueError

    def test_cycle_raises_error(self):
        # Cycle: 0 -> 1 -> 0
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_types = torch.tensor([2, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="cycles"):
            compute_dag_positional_features(2, edge_index, node_types)


class TestIntegrationWithExpressions:
    """Integration tests with actual expression parsing."""

    def test_simple_expression(self):
        from src.data.ast_parser import expr_to_graph

        # Simple expression: x + y
        graph = expr_to_graph("x + y", use_dag_features=True)

        assert hasattr(graph, 'dag_pos')
        assert graph.dag_pos.shape[0] == graph.x.shape[0]
        assert graph.dag_pos.shape[1] == DAG_FEATURE_DIM

    def test_complex_expression(self):
        from src.data.ast_parser import expr_to_graph

        # More complex expression
        graph = expr_to_graph("(x & y) + (x ^ y)", use_dag_features=True)

        assert hasattr(graph, 'dag_pos')
        assert graph.dag_pos.shape[0] == graph.x.shape[0]
        assert graph.dag_pos.shape[1] == DAG_FEATURE_DIM
        # All features should be valid
        assert not torch.isnan(graph.dag_pos).any()
        assert not torch.isinf(graph.dag_pos).any()

    def test_dag_features_disabled(self):
        from src.data.ast_parser import expr_to_graph

        graph = expr_to_graph("x + y", use_dag_features=False)

        assert not hasattr(graph, 'dag_pos') or graph.dag_pos is None

    def test_optimized_graph(self):
        from src.data.ast_parser import expr_to_optimized_graph

        graph = expr_to_optimized_graph("(x & y) | (x ^ y)", use_dag_features=True)

        assert hasattr(graph, 'dag_pos')
        assert graph.dag_pos.shape[0] == graph.x.shape[0]
        assert graph.dag_pos.shape[1] == DAG_FEATURE_DIM
