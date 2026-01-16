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
