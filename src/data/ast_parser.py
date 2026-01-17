"""
AST parser for MBA expressions.

Converts expression strings to Abstract Syntax Trees and PyTorch Geometric graphs.

Edge Type Systems:
- Legacy (6 types): CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR
- Optimized (8 types): LEFT/RIGHT/UNARY_OPERAND, LEFT/RIGHT/UNARY_OPERAND_INV, DOMAIN_BRIDGE_DOWN/UP

The optimized edge types are used by ScaledMBADataset which converts from legacy format
at load time and applies subexpression sharing.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import torch
from torch_geometric.data import Data

from src.constants import NODE_TYPES, EDGE_TYPES, NODE_DIM, USE_DAG_FEATURES
from src.models.edge_types import EdgeType, NodeType, NODE_TYPE_MAP
from src.data.dag_features import compute_dag_positional_features


@dataclass
class ASTNode:
    """Node in an Abstract Syntax Tree."""

    type: str  # From NODE_TYPES
    value: Optional[str] = None  # Variable name or constant value
    children: List['ASTNode'] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """Check if node is a leaf (variable or constant)."""
        return self.type in ('VAR', 'CONST')

    def is_unary(self) -> bool:
        """Check if node is a unary operator."""
        return self.type in ('NOT', 'NEG')

    def is_binary(self) -> bool:
        """Check if node is a binary operator."""
        return self.type in ('ADD', 'SUB', 'MUL', 'AND', 'OR', 'XOR')


def tokenize(expr: str) -> List[str]:
    """
    Tokenize expression string into operators, variables, constants, and parentheses.

    Args:
        expr: Expression string

    Returns:
        List of tokens
    """
    # Normalize whitespace
    expr = ' '.join(expr.split())

    # Add spaces around operators and parentheses
    for op in ['+', '-', '*', '&', '|', '^', '~', '(', ')']:
        expr = expr.replace(op, f' {op} ')

    # Split and filter empty strings
    tokens = [t.strip() for t in expr.split() if t.strip()]

    return tokens


def parse_to_ast(expr: str) -> ASTNode:
    """
    Parse expression string to AST tree structure.

    Uses recursive descent parsing with operator precedence:
    1. | (OR) - lowest precedence
    2. ^ (XOR)
    3. & (AND)
    4. +, - (ADD, SUB)
    5. * (MUL)
    6. ~ (NOT) - unary
    7. Variables and constants - highest precedence

    Args:
        expr: Expression string (e.g., "(x & y) + (x ^ y)")

    Returns:
        Root ASTNode of the parsed tree
    """
    tokens = tokenize(expr)
    result, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected tokens after position {pos}: {tokens[pos:]}")
    return result


def _parse_expr(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse full expression (lowest precedence: OR)."""
    return _parse_or(tokens, pos)


def _parse_or(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse OR expressions."""
    left, pos = _parse_xor(tokens, pos)

    while pos < len(tokens) and tokens[pos] == '|':
        pos += 1  # consume '|'
        right, pos = _parse_xor(tokens, pos)
        left = ASTNode(type='OR', children=[left, right])

    return left, pos


def _parse_xor(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse XOR expressions."""
    left, pos = _parse_and(tokens, pos)

    while pos < len(tokens) and tokens[pos] == '^':
        pos += 1  # consume '^'
        right, pos = _parse_and(tokens, pos)
        left = ASTNode(type='XOR', children=[left, right])

    return left, pos


def _parse_and(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse AND expressions."""
    left, pos = _parse_additive(tokens, pos)

    while pos < len(tokens) and tokens[pos] == '&':
        pos += 1  # consume '&'
        right, pos = _parse_additive(tokens, pos)
        left = ASTNode(type='AND', children=[left, right])

    return left, pos


def _parse_additive(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse addition and subtraction."""
    left, pos = _parse_multiplicative(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ('+', '-'):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_multiplicative(tokens, pos)
        node_type = 'ADD' if op == '+' else 'SUB'
        left = ASTNode(type=node_type, children=[left, right])

    return left, pos


def _parse_multiplicative(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse multiplication."""
    left, pos = _parse_unary(tokens, pos)

    while pos < len(tokens) and tokens[pos] == '*':
        pos += 1  # consume '*'
        right, pos = _parse_unary(tokens, pos)
        left = ASTNode(type='MUL', children=[left, right])

    return left, pos


def _parse_unary(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse unary operators (NOT, NEG)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    if tokens[pos] == '~':
        pos += 1
        operand, pos = _parse_unary(tokens, pos)
        return ASTNode(type='NOT', children=[operand]), pos

    if tokens[pos] == '-':
        # Check if this is unary minus or binary minus
        # It's unary if we're at start or after an operator/open paren
        pos += 1
        operand, pos = _parse_unary(tokens, pos)
        return ASTNode(type='NEG', children=[operand]), pos

    return _parse_primary(tokens, pos)


def _parse_primary(tokens: List[str], pos: int) -> Tuple[ASTNode, int]:
    """Parse primary expressions (variables, constants, parenthesized expressions)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    # Parenthesized expression
    if token == '(':
        pos += 1  # consume '('
        node, pos = _parse_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError(f"Expected ')' at position {pos}")
        pos += 1  # consume ')'
        return node, pos

    # Variable (starts with letter)
    if token.isalpha() or (token[0].isalpha() and token[1:].isdigit()):
        return ASTNode(type='VAR', value=token), pos + 1

    # Constant (numeric)
    if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
        return ASTNode(type='CONST', value=token), pos + 1

    raise ValueError(f"Unexpected token: {token}")


def ast_to_graph(
    ast: ASTNode,
    use_dag_features: bool = USE_DAG_FEATURES,
    edge_type_mode: str = "legacy",
) -> Data:
    """
    Convert AST to PyTorch Geometric Data object.

    Returns Data with:
        x: [num_nodes, NODE_DIM] node features (one-hot type + position encoding)
        edge_index: [2, num_edges] edges
        edge_type: [num_edges] edge type indices
        dag_pos: [num_nodes, 4] DAG positional features (if use_dag_features=True)

    Args:
        ast: Root ASTNode
        use_dag_features: Compute and attach DAG positional features
        edge_type_mode: Edge type system - "legacy" (6-type, default) or "optimized" (8-type)

    Returns:
        PyTorch Geometric Data object

    Edge Types:
        - legacy: CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR
        - optimized: LEFT/RIGHT/UNARY_OPERAND, *_INV, DOMAIN_BRIDGE_DOWN/UP
    """
    if edge_type_mode not in ("legacy", "optimized"):
        raise ValueError(f"edge_type_mode must be 'legacy' or 'optimized', got: {edge_type_mode}")

    if edge_type_mode == "optimized":
        return ast_to_optimized_graph(ast, use_dag_features=use_dag_features)
    # Collect all nodes with DFS
    nodes = []
    node_to_idx = {}

    def collect_nodes(node: ASTNode):
        idx = len(nodes)
        nodes.append(node)
        node_to_idx[id(node)] = idx
        for child in node.children:
            collect_nodes(child)

    collect_nodes(ast)
    num_nodes = len(nodes)

    # Build parent map once for efficient depth computation
    parent_map = _build_parent_map(nodes, node_to_idx)

    # Build node features [num_nodes, NODE_DIM]
    node_features = torch.zeros((num_nodes, NODE_DIM), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        # One-hot encoding for node type
        type_idx = NODE_TYPES[node.type]
        node_features[idx, type_idx] = 1.0

        # Positional encoding (tree depth and position within parent)
        depth = _compute_depth_from_parent_map(idx, parent_map)
        node_features[idx, len(NODE_TYPES)] = depth / 20.0  # Normalize

        # For variables, encode which variable (x0-x7)
        if node.type == 'VAR' and node.value:
            if node.value.startswith('x') and node.value[1:].isdigit():
                var_idx = int(node.value[1:])
                if var_idx < 8:
                    node_features[idx, len(NODE_TYPES) + 1 + var_idx] = 1.0

        # For constants, encode value (normalized)
        if node.type == 'CONST' and node.value:
            try:
                const_val = int(node.value)
                # Normalize constant value to [-1, 1] range
                node_features[idx, len(NODE_TYPES) + 9] = max(-1.0, min(1.0, const_val / 256.0))
            except ValueError:
                pass

    # Build edges
    edge_list = []
    edge_types = []

    # Track variable usage for SAME_VAR edges
    var_nodes: Dict[str, List[int]] = {}

    for idx, node in enumerate(nodes):
        # Track variables
        if node.type == 'VAR' and node.value:
            if node.value not in var_nodes:
                var_nodes[node.value] = []
            var_nodes[node.value].append(idx)

        # CHILD edges (parent -> child) and PARENT edges (child -> parent)
        if len(node.children) >= 1:
            child_idx = node_to_idx[id(node.children[0])]
            edge_list.append([idx, child_idx])
            edge_types.append(EDGE_TYPES['CHILD_LEFT'])
            edge_list.append([child_idx, idx])
            edge_types.append(EDGE_TYPES['PARENT'])

        if len(node.children) >= 2:
            child_idx = node_to_idx[id(node.children[1])]
            edge_list.append([idx, child_idx])
            edge_types.append(EDGE_TYPES['CHILD_RIGHT'])
            edge_list.append([child_idx, idx])
            edge_types.append(EDGE_TYPES['PARENT'])

        # SIBLING edges
        if len(node.children) == 2:
            left_idx = node_to_idx[id(node.children[0])]
            right_idx = node_to_idx[id(node.children[1])]
            edge_list.append([left_idx, right_idx])
            edge_types.append(EDGE_TYPES['SIBLING_NEXT'])
            edge_list.append([right_idx, left_idx])
            edge_types.append(EDGE_TYPES['SIBLING_PREV'])

    # SAME_VAR edges (connect all uses of the same variable)
    for var_name, indices in var_nodes.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_list.append([indices[i], indices[j]])
                edge_types.append(EDGE_TYPES['SAME_VAR'])
                edge_list.append([indices[j], indices[i]])
                edge_types.append(EDGE_TYPES['SAME_VAR'])

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)

    # Compute DAG positional features if enabled
    if use_dag_features:
        node_type_ids = node_features[:, :len(NODE_TYPES)].argmax(dim=1)
        dag_pos = compute_dag_positional_features(num_nodes, edge_index, node_type_ids)
        data.dag_pos = dag_pos

    return data


def _build_parent_map(nodes: List[ASTNode], node_to_idx: Dict[int, int]) -> Dict[int, int]:
    """
    Build mapping from child index to parent index.

    Args:
        nodes: List of all nodes in DFS order
        node_to_idx: Mapping from id(node) to node index

    Returns:
        Dict mapping child_idx -> parent_idx
    """
    parent_map = {}
    for parent_idx, node in enumerate(nodes):
        for child in node.children:
            child_idx = node_to_idx.get(id(child))
            if child_idx is not None:
                parent_map[child_idx] = parent_idx
    return parent_map


def _compute_depth_from_parent_map(idx: int, parent_map: Dict[int, int]) -> int:
    """
    Compute depth of node from parent map.

    Args:
        idx: Index of node
        parent_map: Mapping from child_idx to parent_idx

    Returns:
        Depth of node (root = 0)
    """
    if idx == 0:
        return 0  # Root node

    depth = 0
    current = idx
    while current in parent_map:
        depth += 1
        current = parent_map[current]
        if depth > 50:  # Sanity check for cycles
            break

    return depth


def expr_to_graph(
    expr: str,
    use_dag_features: bool = USE_DAG_FEATURES,
    edge_type_mode: str = "legacy",
) -> Data:
    """
    Convenience function: expression string directly to graph.

    Args:
        expr: Expression string
        use_dag_features: Compute and attach DAG positional features
        edge_type_mode: Edge type system - "legacy" (6-type, default) or "optimized" (8-type)

    Returns:
        PyTorch Geometric Data object

    Edge Types:
        - legacy: CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR
        - optimized: LEFT/RIGHT/UNARY_OPERAND, *_INV, DOMAIN_BRIDGE_DOWN/UP
    """
    ast = parse_to_ast(expr)
    return ast_to_graph(ast, use_dag_features=use_dag_features, edge_type_mode=edge_type_mode)


def expr_to_ast_depth(expr: str) -> int:
    """
    Compute maximum depth of expression AST.

    Args:
        expr: Expression string

    Returns:
        Maximum depth (root = 0)
    """
    ast = parse_to_ast(expr)
    return get_ast_depth(ast)


def get_ast_depth(node: ASTNode) -> int:
    """
    Compute maximum depth of AST from a node.

    Args:
        node: ASTNode root

    Returns:
        Maximum depth from this node (leaf = 0)
    """
    if not node.children:
        return 0
    return 1 + max(get_ast_depth(child) for child in node.children)


def ast_to_optimized_graph(ast: ASTNode, use_dag_features: bool = USE_DAG_FEATURES) -> Data:
    """
    Convert AST to PyTorch Geometric Data with optimized 8-type edge system.

    Uses the new edge type schema:
    - LEFT_OPERAND, RIGHT_OPERAND, UNARY_OPERAND (parent -> child)
    - LEFT_OPERAND_INV, RIGHT_OPERAND_INV, UNARY_OPERAND_INV (child -> parent)
    - DOMAIN_BRIDGE_DOWN, DOMAIN_BRIDGE_UP (boolean <-> arithmetic transitions)

    Does NOT include SIBLING or SAME_VAR edges - these are either redundant
    or handled by subexpression sharing in the dataset loader.

    Args:
        ast: Root ASTNode
        use_dag_features: Compute and attach DAG positional features

    Returns:
        PyTorch Geometric Data object with optimized edge types
    """
    nodes = []
    node_to_idx = {}

    def collect_nodes(node: ASTNode):
        idx = len(nodes)
        nodes.append(node)
        node_to_idx[id(node)] = idx
        for child in node.children:
            collect_nodes(child)

    collect_nodes(ast)
    num_nodes = len(nodes)

    # Build node type features (just type IDs for heterogeneous models)
    node_types = []
    for node in nodes:
        type_id = NODE_TYPE_MAP.get(node.type, 0)
        node_types.append(type_id)

    node_type_tensor = torch.tensor(node_types, dtype=torch.long)

    # Build edges with optimized types
    edge_list = []
    edge_types = []

    for idx, node in enumerate(nodes):
        if node.is_unary() and len(node.children) >= 1:
            # Unary operator edge
            child_idx = node_to_idx[id(node.children[0])]
            edge_list.append([idx, child_idx])
            edge_types.append(int(EdgeType.UNARY_OPERAND))
            edge_list.append([child_idx, idx])
            edge_types.append(int(EdgeType.UNARY_OPERAND_INV))

        elif node.is_binary():
            # Binary operator edges
            if len(node.children) >= 1:
                child_idx = node_to_idx[id(node.children[0])]
                edge_list.append([idx, child_idx])
                edge_types.append(int(EdgeType.LEFT_OPERAND))
                edge_list.append([child_idx, idx])
                edge_types.append(int(EdgeType.LEFT_OPERAND_INV))

            if len(node.children) >= 2:
                child_idx = node_to_idx[id(node.children[1])]
                edge_list.append([idx, child_idx])
                edge_types.append(int(EdgeType.RIGHT_OPERAND))
                edge_list.append([child_idx, idx])
                edge_types.append(int(EdgeType.RIGHT_OPERAND_INV))

            # Domain bridge edges (bool parent -> arith child)
            parent_is_bool = NodeType.is_boolean(NODE_TYPE_MAP.get(node.type, 0))
            for child in node.children:
                child_idx = node_to_idx[id(child)]
                child_type_id = NODE_TYPE_MAP.get(child.type, 0)
                child_is_arith = NodeType.is_arithmetic(child_type_id)

                if parent_is_bool and child_is_arith:
                    edge_list.append([idx, child_idx])
                    edge_types.append(int(EdgeType.DOMAIN_BRIDGE_DOWN))

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    data = Data(x=node_type_tensor, edge_index=edge_index, edge_type=edge_type)

    # Compute DAG positional features if enabled
    if use_dag_features:
        dag_pos = compute_dag_positional_features(num_nodes, edge_index, node_type_tensor)
        data.dag_pos = dag_pos

    return data


def expr_to_optimized_graph(expr: str, use_dag_features: bool = USE_DAG_FEATURES) -> Data:
    """
    Convenience function: expression string to optimized graph.

    Args:
        expr: Expression string
        use_dag_features: Compute and attach DAG positional features

    Returns:
        PyTorch Geometric Data object with optimized edge types
    """
    ast = parse_to_ast(expr)
    return ast_to_optimized_graph(ast, use_dag_features=use_dag_features)


def node_types_to_features(node_types: torch.Tensor, node_dim: int = NODE_DIM) -> torch.Tensor:
    """
    Convert node type IDs to dense feature vectors.

    Bridges optimized graphs (node type IDs) with legacy encoders (dense features).
    Creates one-hot encoding of node type with padding to node_dim.

    Args:
        node_types: [num_nodes] tensor of node type IDs (0-9)
        node_dim: Output feature dimension (default NODE_DIM=32)

    Returns:
        [num_nodes, node_dim] dense feature tensor
    """
    num_nodes = node_types.size(0)
    num_types = len(NODE_TYPES)

    features = torch.zeros((num_nodes, node_dim), dtype=torch.float32, device=node_types.device)

    # Vectorized one-hot encoding in first num_types dimensions
    valid_mask = (node_types >= 0) & (node_types < num_types)
    valid_indices = torch.arange(num_nodes, device=node_types.device)[valid_mask]
    valid_types = node_types[valid_mask].long()
    features[valid_indices, valid_types] = 1.0

    return features


def convert_graph_for_encoder(graph: Data, requires_dense_features: bool) -> Data:
    """
    Convert graph to format expected by encoder.

    Preserves dag_pos attribute if present.

    Args:
        graph: PyG Data object (may have node type IDs or dense features)
        requires_dense_features: If True, convert node type IDs to dense features

    Returns:
        PyG Data object in the required format
    """
    # Check if graph has node type IDs (1D) or dense features (2D)
    is_type_ids = graph.x.dim() == 1

    if requires_dense_features and is_type_ids:
        # Convert type IDs to dense features
        dense_x = node_types_to_features(graph.x)
        result = Data(x=dense_x, edge_index=graph.edge_index, edge_type=graph.edge_type)
    elif not requires_dense_features and not is_type_ids:
        # Convert dense features to type IDs (take argmax of one-hot portion)
        type_ids = graph.x[:, :len(NODE_TYPES)].argmax(dim=1)
        result = Data(x=type_ids, edge_index=graph.edge_index, edge_type=graph.edge_type)
    else:
        # Already in correct format
        return graph

    # Preserve dag_pos attribute if present
    if hasattr(graph, 'dag_pos') and graph.dag_pos is not None:
        result.dag_pos = graph.dag_pos

    return result
