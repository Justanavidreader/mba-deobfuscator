"""
Operation-aware aggregation for GNN encoders.

Composable module that handles commutative vs non-commutative operations differently
during message aggregation. Designed for HGT but extensible to other GNN encoders.

Commutative operations (AND, OR, XOR, ADD, MUL): order-invariant sum aggregation.
Non-commutative operations (SUB): order-preserving concatenation + projection.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    COMMUTATIVE_OPS,
    NON_COMMUTATIVE_OPS,
    UNARY_OPS,
    NODE_TYPE_TO_STR,
)
from src.models.edge_types import EdgeType

logger = logging.getLogger(__name__)


class OperationAwareAggregator(nn.Module):
    """
    Composable aggregation layer with operation-aware logic.

    Treats binary operations differently based on commutativity:
    - Commutative (ADD, MUL, AND, OR, XOR): sum of left and right operand messages
    - Non-commutative (SUB): concatenate [left, right] then project to hidden_dim

    Unary operations (NOT, NEG) and terminals (VAR, CONST) pass through unchanged.

    Designed for PyG batching convention: all tensors are flattened across batch dimension.
    Use torch_geometric.data.Batch to prepare batched input.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        strict_validation: bool = True,
    ):
        """
        Initialize operation-aware aggregator.

        Args:
            hidden_dim: Node embedding dimension (must match encoder output).
            dropout: Dropout probability for non-commutative projection.
            strict_validation: If True, raise ValueError on malformed graphs.
                              If False, log warning and skip problematic nodes.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.strict_validation = strict_validation

        # Projects [left_msg || right_msg] -> hidden_dim for non-commutative ops
        self.non_comm_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cache mapping from node type index to operation name
        self._node_type_to_op: Dict[int, str] = NODE_TYPE_TO_STR.copy()

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in self.non_comm_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_operand_edges(
        self,
        node_idx: int,
        op_name: str,
        left_count: int,
        right_count: int,
        incoming_edges: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> bool:
        """
        Validate that binary operation has exactly 1 LEFT and 1 RIGHT operand edge.

        Args:
            node_idx: Index of the operation node.
            op_name: Name of the operation (e.g., 'ADD', 'SUB').
            left_count: Number of LEFT_OPERAND edges.
            right_count: Number of RIGHT_OPERAND edges.
            incoming_edges: Edge index tensor for incoming edges.
            left_mask: Boolean mask for LEFT_OPERAND edges.
            right_mask: Boolean mask for RIGHT_OPERAND edges.

        Returns:
            True if valid, False otherwise.

        Raises:
            ValueError: If strict_validation=True and graph is malformed.
        """
        if left_count == 0 or right_count == 0:
            msg = (
                f"Operation node {node_idx} (type {op_name}) missing operand edges. "
                f"LEFT: {left_count > 0}, RIGHT: {right_count > 0}. "
                f"This indicates a graph construction bug."
            )
            if self.strict_validation:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        if left_count != 1 or right_count != 1:
            left_sources = incoming_edges[0, left_mask].tolist() if left_count > 0 else []
            right_sources = incoming_edges[0, right_mask].tolist() if right_count > 0 else []
            msg = (
                f"Operation node {node_idx} (type {op_name}) has invalid operand edge count. "
                f"Expected 1 LEFT and 1 RIGHT, found {left_count} LEFT and {right_count} RIGHT. "
                f"Edge sources: LEFT={left_sources}, RIGHT={right_sources}. "
                f"This indicates duplicate edges in graph construction."
            )
            if self.strict_validation:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        return True

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        node_types: torch.Tensor,
        messages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate messages with operation-aware logic for commutative/non-commutative operations.

        Commutative operations (AND, OR, XOR, ADD, MUL) use order-invariant sum aggregation.
        Non-commutative operations (SUB) preserve operand order via concatenation and projection.
        Unary operations (NOT, NEG) and terminals (VAR, CONST) pass through unchanged.

        Assumes PyG batching convention: all tensors are flattened across batch dimension.
        Use torch_geometric.data.Batch to prepare batched input.

        Args:
            node_features: Node embeddings from previous layer, shape [num_nodes, hidden_dim].
            edge_index: Edge connectivity, shape [2, num_edges].
            edge_types: Edge type indices, shape [num_edges].
            node_types: Node type indices, shape [num_nodes].
            messages: Attention-weighted messages from HGT/GNN layer, shape [num_nodes, hidden_dim].

        Returns:
            Aggregated messages with operation-aware logic, shape [num_nodes, hidden_dim].

        Raises:
            ValueError: If strict_validation=True and graph structure is malformed.
        """
        # Shape validation (catch batching dimension issues early)
        assert node_features.ndim == 2, (
            f"Expected 2D node_features (flattened batch), got shape {node_features.shape}. "
            f"Use torch_geometric.data.Batch for batching."
        )
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
            f"Expected edge_index shape [2, num_edges], got {edge_index.shape}."
        )
        assert node_features.shape[0] == node_types.shape[0], (
            f"node_features and node_types must have same number of nodes. "
            f"Got {node_features.shape[0]} vs {node_types.shape[0]}."
        )
        assert node_features.shape[1] == self.hidden_dim, (
            f"node_features hidden_dim {node_features.shape[1]} != aggregator hidden_dim {self.hidden_dim}."
        )

        # Start with original messages (non-operation nodes pass through unchanged)
        aggregated = messages.clone()

        # Find binary operation nodes (node_type >= 2 and not unary)
        # Node types: 0=VAR, 1=CONST, 2=ADD, 3=SUB, 4=MUL, 5=AND, 6=OR, 7=XOR, 8=NOT, 9=NEG
        binary_op_mask = (node_types >= 2) & (node_types <= 7)
        binary_op_indices = torch.where(binary_op_mask)[0]

        if binary_op_indices.numel() == 0:
            # No binary operations - return messages unchanged
            return aggregated

        # Precompute edge destinations for efficient lookup
        edge_dst = edge_index[1]

        for node_idx in binary_op_indices:
            node_idx_item = node_idx.item()
            node_type = node_types[node_idx_item].item()

            # Get operation name with error handling for unknown types
            try:
                op_name = self._node_type_to_op[node_type]
            except KeyError:
                msg = (
                    f"Unknown operation node type {node_type} at node {node_idx_item}. "
                    f"Known types: {list(self._node_type_to_op.keys())}. "
                    f"Update NODE_TYPE_TO_STR in constants.py if new node types were added."
                )
                raise ValueError(msg) from None

            # Skip if not a binary operation (shouldn't happen due to mask, but defensive)
            if op_name in UNARY_OPS:
                continue

            # Find incoming edges to this operation node
            incoming_mask = edge_dst == node_idx_item
            if not incoming_mask.any():
                if self.strict_validation:
                    raise ValueError(
                        f"Operation node {node_idx_item} (type {op_name}) has no incoming edges."
                    )
                logger.warning(
                    f"Operation node {node_idx_item} (type {op_name}) has no incoming edges. Skipping."
                )
                continue

            incoming_edges = edge_index[:, incoming_mask]
            incoming_types = edge_types[incoming_mask]

            # Find LEFT and RIGHT operand edges
            left_mask = incoming_types == EdgeType.LEFT_OPERAND.value
            right_mask = incoming_types == EdgeType.RIGHT_OPERAND.value

            left_count = left_mask.sum().item()
            right_count = right_mask.sum().item()

            # Validate edge structure
            if not self._validate_operand_edges(
                node_idx_item, op_name, left_count, right_count,
                incoming_edges, left_mask, right_mask
            ):
                continue

            # Get source node indices for left/right operands
            left_src = incoming_edges[0, left_mask][0].item()
            right_src = incoming_edges[0, right_mask][0].item()

            # Validate source indices are in bounds
            num_nodes = node_features.shape[0]
            if left_src >= num_nodes or right_src >= num_nodes:
                msg = (
                    f"Operation node {node_idx_item} (type {op_name}) has out-of-bounds operand indices. "
                    f"left_src={left_src}, right_src={right_src}, num_nodes={num_nodes}."
                )
                if self.strict_validation:
                    raise ValueError(msg)
                logger.warning(msg)
                continue

            # Get messages from left/right children
            left_msg = node_features[left_src]
            right_msg = node_features[right_src]

            if op_name in COMMUTATIVE_OPS:
                # Order-invariant: element-wise sum
                aggregated[node_idx_item] = left_msg + right_msg

            elif op_name in NON_COMMUTATIVE_OPS:
                # Order-preserving: concatenate [left, right] then project
                combined = torch.cat([left_msg, right_msg], dim=-1)
                aggregated[node_idx_item] = self.non_comm_proj(combined.unsqueeze(0)).squeeze(0)

            # else: unknown binary op, leave aggregated unchanged

        return aggregated

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"hidden_dim={self.hidden_dim}, strict_validation={self.strict_validation}"
