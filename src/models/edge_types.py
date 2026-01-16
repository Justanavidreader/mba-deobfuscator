"""
Edge and node type definitions for MBA expression graphs.
Optimized based on HOList research findings.

Changes from original 6-type schema:
- SIBLING_NEXT, SIBLING_PREV: Removed (redundant with tree structure)
- SAME_VAR: Replaced by subexpression sharing in dataset loader
- PARENT: Split into position-aware inverses (LEFT_OPERAND_INV, etc.)
- DOMAIN_BRIDGE: Split into DOMAIN_BRIDGE_DOWN/UP for bidirectional flow
"""

from enum import IntEnum
from typing import Dict, Set


class EdgeType(IntEnum):
    """
    Optimized edge types for MBA expression graphs.
    Bidirectional edges enable faster information propagation.
    Total: 8 edge types with semantically distinct directions.
    """
    # Structural edges (parent -> child)
    LEFT_OPERAND = 0       # Binary operator to left child
    RIGHT_OPERAND = 1      # Binary operator to right child
    UNARY_OPERAND = 2      # Unary operator (NOT, NEG) to child

    # Structural inverse edges (child -> parent) for bidirectional message passing
    LEFT_OPERAND_INV = 3   # Left child to parent operator
    RIGHT_OPERAND_INV = 4  # Right child to parent operator
    UNARY_OPERAND_INV = 5  # Child to unary parent operator

    # Domain bridge edges (both directions, semantically distinct)
    DOMAIN_BRIDGE_DOWN = 6  # Parent (domain A) -> Child (domain B)
    DOMAIN_BRIDGE_UP = 7    # Child (domain B) -> Parent (domain A)

    @staticmethod
    def get_inverse(edge_type: int) -> int:
        """Get inverse edge type for bidirectional flow."""
        inverse_map = {
            0: 3,  # LEFT_OPERAND -> LEFT_OPERAND_INV
            1: 4,  # RIGHT_OPERAND -> RIGHT_OPERAND_INV
            2: 5,  # UNARY_OPERAND -> UNARY_OPERAND_INV
            3: 0,  # LEFT_OPERAND_INV -> LEFT_OPERAND
            4: 1,  # RIGHT_OPERAND_INV -> RIGHT_OPERAND
            5: 2,  # UNARY_OPERAND_INV -> UNARY_OPERAND
            6: 7,  # DOMAIN_BRIDGE_DOWN -> DOMAIN_BRIDGE_UP
            7: 6,  # DOMAIN_BRIDGE_UP -> DOMAIN_BRIDGE_DOWN
        }
        return inverse_map.get(edge_type, edge_type)

    @staticmethod
    def is_forward_edge(edge_type: int) -> bool:
        """Check if edge is parent->child direction."""
        return edge_type in [0, 1, 2, 6]

    @staticmethod
    def is_inverse_edge(edge_type: int) -> bool:
        """Check if edge is child->parent direction."""
        return edge_type in [3, 4, 5, 7]

    @staticmethod
    def is_domain_bridge(edge_type: int) -> bool:
        """Check if edge is a domain bridge (Boolean<->Arithmetic)."""
        return edge_type in [6, 7]


class NodeType(IntEnum):
    """
    Node types for heterogeneous MBA expression graph.
    10 types covering terminals, arithmetic, and boolean nodes.

    IMPORTANT: This ordering MUST match src/constants.py NODE_TYPES.
    """
    # Terminals (leaves)
    VAR = 0      # Variables: x, y, z, ...
    CONST = 1    # Constants: 0, 1, 2, ...

    # Arithmetic operators
    ADD = 2      # +
    SUB = 3      # -
    MUL = 4      # *

    # Boolean operators
    AND = 5      # &
    OR = 6       # |
    XOR = 7      # ^

    # Unary operators
    NOT = 8      # ~ (boolean)
    NEG = 9      # unary - (arithmetic)

    @staticmethod
    def is_arithmetic(node_type: int) -> bool:
        """Check if node is arithmetic operator (ADD, SUB, MUL, NEG)."""
        return node_type in [2, 3, 4, 9]

    @staticmethod
    def is_boolean(node_type: int) -> bool:
        """Check if node is boolean operator (AND, OR, XOR, NOT)."""
        return node_type in [5, 6, 7, 8]

    @staticmethod
    def is_unary(node_type: int) -> bool:
        """Check if node is unary operator (NOT or NEG)."""
        return node_type in [8, 9]

    @staticmethod
    def is_binary(node_type: int) -> bool:
        """Check if node is binary operator."""
        return node_type in [2, 3, 4, 5, 6, 7]

    @staticmethod
    def is_terminal(node_type: int) -> bool:
        """Check if node is terminal (VAR or CONST)."""
        return node_type in [0, 1]


# Mapping from string names to NodeType
NODE_TYPE_MAP: Dict[str, int] = {
    'ADD': NodeType.ADD,
    'SUB': NodeType.SUB,
    'MUL': NodeType.MUL,
    'NEG': NodeType.NEG,
    'AND': NodeType.AND,
    'OR': NodeType.OR,
    'XOR': NodeType.XOR,
    'NOT': NodeType.NOT,
    'VAR': NodeType.VAR,
    'CONST': NodeType.CONST,
    # Aliases
    '+': NodeType.ADD,
    '-': NodeType.SUB,
    '*': NodeType.MUL,
    '~': NodeType.NOT,
    '&': NodeType.AND,
    '|': NodeType.OR,
    '^': NodeType.XOR,
}

# Legacy edge type mapping for backward compatibility with old datasets
LEGACY_EDGE_MAP: Dict[int, int] = {
    0: EdgeType.LEFT_OPERAND,   # CHILD_LEFT -> LEFT_OPERAND
    1: EdgeType.RIGHT_OPERAND,  # CHILD_RIGHT -> RIGHT_OPERAND
    # Types 2-5 are skipped during conversion (handled in dataset loader)
    # 2: PARENT -> regenerated as inverse edges
    # 3: SIBLING_NEXT -> removed
    # 4: SIBLING_PREV -> removed
    # 5: SAME_VAR -> replaced by subexpression sharing
}

# Edge types that should be skipped when loading legacy datasets
LEGACY_SKIP_TYPES: Set[int] = {2, 3, 4, 5}

# Legacy node type ordering for backward compatibility with datasets
# generated before 2026-01-15 (prior to NodeType enum reordering)
LEGACY_NODE_ORDER = ['ADD', 'SUB', 'MUL', 'NEG', 'AND', 'OR', 'XOR', 'NOT', 'VAR', 'CONST']

# Generated mapping from legacy IDs to current IDs
# Legacy: ADD=0, SUB=1, MUL=2, NEG=3, AND=4, OR=5, XOR=6, NOT=7, VAR=8, CONST=9
# Current: VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, AND=5, OR=6, XOR=7, NOT=8, NEG=9
LEGACY_NODE_MAP: Dict[int, int] = {
    0: 2,   # ADD: 0 -> 2
    1: 3,   # SUB: 1 -> 3
    2: 4,   # MUL: 2 -> 4
    3: 9,   # NEG: 3 -> 9
    4: 5,   # AND: 4 -> 5
    5: 6,   # OR: 5 -> 6
    6: 7,   # XOR: 6 -> 7
    7: 8,   # NOT: 7 -> 8
    8: 0,   # VAR: 8 -> 0
    9: 1,   # CONST: 9 -> 1
}


def convert_legacy_node_types(node_types: 'torch.Tensor') -> 'torch.Tensor':
    """
    Convert legacy node type IDs to current schema.

    Legacy schema (pre-2026-01-15): ADD=0, SUB=1, ..., VAR=8, CONST=9
    Current schema: VAR=0, CONST=1, ADD=2, SUB=3, ...

    Args:
        node_types: [num_nodes] tensor with legacy IDs (values in [0-9])

    Returns:
        [num_nodes] tensor with current IDs

    Raises:
        ValueError: If node_types contains IDs outside [0-9] range
    """
    import torch

    if node_types.numel() == 0:
        return node_types

    min_id = node_types.min().item()
    max_id = node_types.max().item()
    if min_id < 0 or max_id > 9:
        raise ValueError(
            f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}]. "
            f"Dataset may be corrupted or use unsupported schema version."
        )

    # Vectorized lookup table - O(N) time, O(1) extra memory
    lookup = torch.tensor([LEGACY_NODE_MAP[i] for i in range(10)],
                         dtype=node_types.dtype, device=node_types.device)
    return lookup[node_types]
