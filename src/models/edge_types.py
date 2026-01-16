"""
Edge and node type definitions for MBA expression graphs.
Optimized based on HOList research findings.

Changes from original 6-type schema:
- SIBLING_NEXT, SIBLING_PREV: Removed (redundant with tree structure)
- SAME_VAR: Replaced by subexpression sharing in dataset loader
- PARENT: Split into position-aware PARENT_OF_LEFT/RIGHT/UNARY
- DOMAIN_BRIDGE: Added for Boolean<->Arithmetic transitions
"""

from enum import IntEnum
from typing import Dict, Set


class EdgeType(IntEnum):
    """
    Optimized edge types for MBA expression graphs.
    Bidirectional edges enable faster information propagation.
    Total: 7 edge types (vs original 6, but more semantically meaningful)
    """
    # Structural edges (operator -> operand)
    LEFT_OPERAND = 0      # Binary operator to left child
    RIGHT_OPERAND = 1     # Binary operator to right child
    UNARY_OPERAND = 2     # Unary operator (NOT, NEG) to child

    # Inverse edges (operand -> operator) for bidirectional message passing
    PARENT_OF_LEFT = 3    # Left child to parent operator
    PARENT_OF_RIGHT = 4   # Right child to parent operator
    PARENT_OF_UNARY = 5   # Child to unary parent operator

    # Semantic edges
    DOMAIN_BRIDGE = 6     # Boolean <-> Arithmetic transition

    @staticmethod
    def get_inverse(edge_type: int) -> int:
        """Get inverse edge type for bidirectional flow."""
        inverse_map = {
            0: 3,  # LEFT_OPERAND -> PARENT_OF_LEFT
            1: 4,  # RIGHT_OPERAND -> PARENT_OF_RIGHT
            2: 5,  # UNARY_OPERAND -> PARENT_OF_UNARY
            3: 0,  # PARENT_OF_LEFT -> LEFT_OPERAND
            4: 1,  # PARENT_OF_RIGHT -> RIGHT_OPERAND
            5: 2,  # PARENT_OF_UNARY -> UNARY_OPERAND
        }
        return inverse_map.get(edge_type, edge_type)

    @staticmethod
    def is_forward_edge(edge_type: int) -> bool:
        """Check if edge is parent->child direction."""
        return edge_type in [0, 1, 2]

    @staticmethod
    def is_inverse_edge(edge_type: int) -> bool:
        """Check if edge is child->parent direction."""
        return edge_type in [3, 4, 5]


class NodeType(IntEnum):
    """
    Node types for heterogeneous MBA expression graph.
    10 types covering arithmetic, boolean, and terminal nodes.
    """
    # Arithmetic operators
    ADD = 0
    SUB = 1
    MUL = 2
    NEG = 3  # Unary negation

    # Boolean operators
    AND = 4
    OR = 5
    XOR = 6
    NOT = 7  # Unary boolean not

    # Terminals
    VAR = 8
    CONST = 9

    @staticmethod
    def is_arithmetic(node_type: int) -> bool:
        """Check if node is arithmetic operator."""
        return node_type in [0, 1, 2, 3]

    @staticmethod
    def is_boolean(node_type: int) -> bool:
        """Check if node is boolean operator."""
        return node_type in [4, 5, 6, 7]

    @staticmethod
    def is_unary(node_type: int) -> bool:
        """Check if node is unary operator (NEG or NOT)."""
        return node_type in [3, 7]

    @staticmethod
    def is_binary(node_type: int) -> bool:
        """Check if node is binary operator."""
        return node_type in [0, 1, 2, 4, 5, 6]

    @staticmethod
    def is_terminal(node_type: int) -> bool:
        """Check if node is terminal (VAR or CONST)."""
        return node_type in [8, 9]


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
