"""
HyperTree Proof Search (HTPS) for deep MBA expressions.

Compositional simplification using fixed tactics with UCB-based search.
"""

from typing import List, Dict, Optional, Set, Tuple
import torch
import math
from dataclasses import dataclass, field
from enum import Enum

from src.constants import (
    HTPS_BUDGET, HTPS_DEPTH_THRESHOLD, HTPS_UCB_CONSTANT, HTPS_TACTICS
)
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.ast_parser import parse_to_ast, ASTNode, expr_to_graph, get_ast_depth
from src.data.fingerprint import SemanticFingerprint
from src.utils.expr_eval import evaluate_expr
from src.inference.verify import ThreeTierVerifier
import torch_geometric.data as pyg_data


class Tactic(Enum):
    """Fixed simplification tactics."""
    IDENTITY_XOR_SELF = "identity_xor_self"    # x ^ x → 0
    IDENTITY_AND_NOT = "identity_and_not"      # x & ~x → 0
    IDENTITY_OR_NOT = "identity_or_not"        # x | ~x → -1
    MBA_AND_XOR = "mba_and_xor"                # (x&y)+(x^y) → x|y
    CONSTANT_FOLD = "constant_fold"            # 3 + 5 → 8
    SIMPLIFY_SUBEXPR = "simplify_subexpr"      # Recurse on subexpression


@dataclass
class HyperNode:
    """
    Node in hypertree search space.
    """
    expr: str                                  # Expression at this node
    ast: ASTNode                               # Parsed AST
    parent: Optional['HyperNode'] = None       # Parent node
    children: List['HyperEdge'] = field(default_factory=list)  # Outgoing hyperedges
    value: float = 0.0                         # Critic value estimate
    visits: int = 0                            # Visit count for UCB
    depth: int = 0                             # Depth in search tree
    is_terminal: bool = False                  # Cannot simplify further
    best_simplification: Optional[str] = None  # Best result found from this node


@dataclass
class HyperEdge:
    """
    Hyperedge representing tactic application.
    Can connect one parent to multiple children (for parallel rewrites).
    """
    tactic: Tactic                             # Applied tactic
    parent: HyperNode                          # Source node
    children: List[HyperNode]                  # Result nodes
    cost: float = 1.0                          # Application cost (for search)
    visits: int = 0                            # Times this edge was selected
    success_count: int = 0                     # Times it led to simplification


class MinimalHTPS:
    """
    HyperTree Proof Search with fixed tactics.
    For expressions with depth >= 10.
    """

    def __init__(
        self,
        model: MBADeobfuscator,
        tokenizer: MBATokenizer,
        verifier: Optional[ThreeTierVerifier] = None,
        budget: int = HTPS_BUDGET,
        ucb_constant: float = HTPS_UCB_CONSTANT,
        tactics: Optional[List[str]] = None
    ):
        """
        Initialize HTPS searcher.

        Args:
            model: Trained model with ValueHead for node evaluation
            tokenizer: MBATokenizer instance
            verifier: ThreeTierVerifier for checking equivalence
            budget: Maximum number of tactic applications
            ucb_constant: UCB exploration constant (√2 typical)
            tactics: List of tactic names to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.budget = budget
        self.ucb_constant = ucb_constant

        # Parse tactics
        tactic_names = tactics or HTPS_TACTICS
        self.tactics = [Tactic(name) for name in tactic_names]

        # CRITICAL FIX: Use SemanticFingerprint class
        self.fingerprint_computer = SemanticFingerprint()

        self.device = next(model.parameters()).device

    def search(self, input_expr: str) -> Tuple[str, List[str]]:
        """
        HTPS search for simplification.

        Args:
            input_expr: Obfuscated expression to simplify

        Returns:
            (best_simplification, proof_trace)
            - best_simplification: Simplest equivalent expression found
            - proof_trace: List of tactic applications showing derivation

        Algorithm:
            1. Create root node with input expression
            2. Evaluate root with model.get_value()
            3. For budget iterations:
                a. Select most promising leaf node (UCB)
                b. Expand node with all applicable tactics
                c. Evaluate new nodes with critic
                d. Backpropagate values
                e. If solution found, return immediately
            4. Extract best partial solution if no complete solution
        """
        # Create root node
        try:
            root_ast = parse_to_ast(input_expr)
        except:
            return (input_expr, [])

        root = HyperNode(expr=input_expr, ast=root_ast, depth=0)

        # Evaluate root
        root.value = self._evaluate_node(root)
        root.visits = 1

        # Search loop
        for iteration in range(self.budget):
            # Select leaf to expand
            leaf = self._select_leaf(root)
            if leaf is None:
                # All nodes are terminal
                break

            # Check if leaf is already simplified enough
            leaf_depth = get_ast_depth(leaf.ast)
            if leaf_depth <= 1:
                # Already very simple - mark terminal
                leaf.is_terminal = True
                continue

            # Expand leaf
            self._expand(leaf)

            # If we found a very simple solution, return immediately
            for edge in leaf.children:
                for child in edge.children:
                    child_depth = get_ast_depth(child.ast)
                    if child_depth == 0:
                        # Found terminal expression (single variable or constant)
                        return self._extract_solution(root)

        # Extract best solution
        return self._extract_solution(root)

    def _select_leaf(self, root: HyperNode) -> Optional[HyperNode]:
        """
        Select most promising leaf using UCB.

        Args:
            root: Root of search tree

        Returns:
            Leaf node to expand, or None if all nodes terminal

        UCB Formula:
            UCB(node) = value(node) + c * sqrt(ln(parent.visits) / node.visits)

            - Balances exploitation (high value) and exploration (low visits)
            - c is UCB constant (typically √2)
        """
        current = root

        while current.children:
            # Select best child edge
            best_edge = None
            best_ucb = float('-inf')

            for edge in current.children:
                for child in edge.children:
                    if child.is_terminal:
                        continue

                    # CRITICAL FIX: Add guards for visits == 0
                    if child.visits == 0:
                        ucb = float('inf')  # Unvisited nodes have infinite exploration value
                    elif current.visits == 0:
                        ucb = child.value  # Fallback to value only
                    else:
                        exploration = self.ucb_constant * math.sqrt(
                            math.log(current.visits) / child.visits
                        )
                        ucb = child.value + exploration

                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_edge = edge
                        current = child

            if best_edge is None:
                # No non-terminal children
                return current if not current.is_terminal else None

        return current if not current.is_terminal else None

    def _expand(self, node: HyperNode) -> None:
        """
        Expand node by applying all tactics.

        Args:
            node: Node to expand

        Process:
            1. For each tactic in self.tactics:
                a. Try to apply tactic to node.expr
                b. If match found, create new child node(s)
                c. Create hyperedge connecting parent to children
                d. Evaluate children with critic (batch inference)
            2. If no tactics apply, mark node as terminal
        """
        applied_any = False

        for tactic in self.tactics:
            results = self._apply_tactic(tactic, node.expr, node.ast)

            if results:
                applied_any = True

                # Create child nodes
                children = []
                for result_expr in results:
                    try:
                        child_ast = parse_to_ast(result_expr)
                        child = HyperNode(
                            expr=result_expr,
                            ast=child_ast,
                            parent=node,
                            depth=node.depth + 1
                        )
                        children.append(child)
                    except:
                        # Parse failed - skip this result
                        continue

                if children:
                    # Evaluate children in batch
                    values = self._evaluate_node_batch(children)
                    for child, value in zip(children, values):
                        child.value = value
                        child.visits = 1

                    # Create hyperedge
                    edge = HyperEdge(
                        tactic=tactic,
                        parent=node,
                        children=children,
                        cost=1.0,
                        visits=1
                    )
                    node.children.append(edge)

                    # Backpropagate
                    for child in children:
                        self._backpropagate(child, child.value)

        if not applied_any:
            node.is_terminal = True

    def _apply_tactic(
        self,
        tactic: Tactic,
        expr: str,
        ast: ASTNode
    ) -> Optional[List[str]]:
        """
        Apply tactic to expression.

        Args:
            tactic: Tactic to apply
            expr: Expression string
            ast: Parsed AST

        Returns:
            List of result expressions if tactic applies, None otherwise

        Implementation per tactic:
            - IDENTITY_XOR_SELF: Find "x^x" subtrees, replace with "0"
            - IDENTITY_AND_NOT: Find "x&(~x)" subtrees, replace with "0"
            - IDENTITY_OR_NOT: Find "x|(~x)" subtrees, replace with "-1"
            - MBA_AND_XOR: Find "(x&y)+(x^y)" patterns, replace with "x|y"
            - CONSTANT_FOLD: Evaluate constant sub-expressions
            - SIMPLIFY_SUBEXPR: Recursively apply all tactics to children
        """
        if tactic == Tactic.IDENTITY_XOR_SELF:
            return self._apply_identity_xor_self(ast)
        elif tactic == Tactic.IDENTITY_AND_NOT:
            return self._apply_identity_and_not(ast)
        elif tactic == Tactic.IDENTITY_OR_NOT:
            return self._apply_identity_or_not(ast)
        elif tactic == Tactic.MBA_AND_XOR:
            return self._apply_mba_and_xor(ast)
        elif tactic == Tactic.CONSTANT_FOLD:
            return self._apply_constant_fold(ast)
        elif tactic == Tactic.SIMPLIFY_SUBEXPR:
            return self._apply_simplify_subexpr(ast)
        else:
            return None

    def _ast_to_str(self, node: ASTNode) -> str:
        """Convert AST back to expression string."""
        if node.type == 'VAR':
            return node.value or 'x'
        elif node.type == 'CONST':
            return node.value or '0'
        elif node.is_unary():
            operand = self._ast_to_str(node.children[0])
            if node.type == 'NOT':
                return f'~{operand}'
            elif node.type == 'NEG':
                return f'-{operand}'
        elif node.is_binary():
            left = self._ast_to_str(node.children[0])
            right = self._ast_to_str(node.children[1])
            op_map = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'AND': '&', 'OR': '|', 'XOR': '^'}
            op = op_map.get(node.type, '?')
            return f'({left} {op} {right})'
        return '0'

    def _nodes_equal(self, a: ASTNode, b: ASTNode) -> bool:
        """Check if two AST nodes are structurally equal."""
        if a.type != b.type:
            return False
        if a.value != b.value:
            return False
        if len(a.children) != len(b.children):
            return False
        return all(self._nodes_equal(ac, bc) for ac, bc in zip(a.children, b.children))

    def _apply_identity_xor_self(self, ast: ASTNode) -> Optional[List[str]]:
        """Find x^x patterns and replace with 0."""
        if ast.type == 'XOR' and len(ast.children) == 2:
            if self._nodes_equal(ast.children[0], ast.children[1]):
                return ['0']
        # Recursively check children
        for i, child in enumerate(ast.children):
            results = self._apply_identity_xor_self(child)
            if results:
                # Replace this child with 0
                new_ast = ASTNode(type=ast.type, value=ast.value)
                new_ast.children = ast.children[:i] + [ASTNode(type='CONST', value='0')] + ast.children[i+1:]
                return [self._ast_to_str(new_ast)]
        return None

    def _apply_identity_and_not(self, ast: ASTNode) -> Optional[List[str]]:
        """Find x&~x patterns and replace with 0."""
        if ast.type == 'AND' and len(ast.children) == 2:
            left, right = ast.children
            # Check if right is ~left
            if right.type == 'NOT' and len(right.children) == 1:
                if self._nodes_equal(left, right.children[0]):
                    return ['0']
            # Check if left is ~right
            if left.type == 'NOT' and len(left.children) == 1:
                if self._nodes_equal(left.children[0], right):
                    return ['0']
        return None

    def _apply_identity_or_not(self, ast: ASTNode) -> Optional[List[str]]:
        """Find x|~x patterns and replace with -1."""
        if ast.type == 'OR' and len(ast.children) == 2:
            left, right = ast.children
            # Check if right is ~left
            if right.type == 'NOT' and len(right.children) == 1:
                if self._nodes_equal(left, right.children[0]):
                    return ['-1']
            # Check if left is ~right
            if left.type == 'NOT' and len(left.children) == 1:
                if self._nodes_equal(left.children[0], right):
                    return ['-1']
        return None

    def _apply_mba_and_xor(self, ast: ASTNode) -> Optional[List[str]]:
        """Find (x&y)+(x^y) patterns and replace with x|y."""
        if ast.type == 'ADD' and len(ast.children) == 2:
            left, right = ast.children
            # Check if pattern matches
            if left.type == 'AND' and right.type == 'XOR':
                if len(left.children) == 2 and len(right.children) == 2:
                    # Check if operands match
                    if (self._nodes_equal(left.children[0], right.children[0]) and
                        self._nodes_equal(left.children[1], right.children[1])):
                        # Build x|y
                        result_ast = ASTNode(type='OR', children=[left.children[0], left.children[1]])
                        return [self._ast_to_str(result_ast)]
        return None

    def _apply_constant_fold(self, ast: ASTNode) -> Optional[List[str]]:
        """Fold constant expressions."""
        if ast.type == 'CONST':
            return None

        # Check if all children are constants
        if ast.is_binary() and len(ast.children) == 2:
            if ast.children[0].type == 'CONST' and ast.children[1].type == 'CONST':
                try:
                    val1 = int(ast.children[0].value)
                    val2 = int(ast.children[1].value)
                    result = None

                    if ast.type == 'ADD':
                        result = val1 + val2
                    elif ast.type == 'SUB':
                        result = val1 - val2
                    elif ast.type == 'MUL':
                        result = val1 * val2
                    elif ast.type == 'AND':
                        result = val1 & val2
                    elif ast.type == 'OR':
                        result = val1 | val2
                    elif ast.type == 'XOR':
                        result = val1 ^ val2

                    if result is not None:
                        return [str(result & 0xFFFFFFFF)]  # 32-bit mask
                except:
                    pass

        return None

    def _apply_simplify_subexpr(self, ast: ASTNode) -> Optional[List[str]]:
        """Recursively simplify subexpressions."""
        results = []
        for i, child in enumerate(ast.children):
            for tactic in self.tactics:
                if tactic == Tactic.SIMPLIFY_SUBEXPR:
                    continue  # Avoid infinite recursion
                child_results = self._apply_tactic(tactic, self._ast_to_str(child), child)
                if child_results:
                    # Build new AST with simplified child
                    for child_result in child_results[:1]:  # Take first result only
                        try:
                            new_child = parse_to_ast(child_result)
                            new_ast = ASTNode(type=ast.type, value=ast.value)
                            new_ast.children = ast.children[:i] + [new_child] + ast.children[i+1:]
                            results.append(self._ast_to_str(new_ast))
                        except:
                            pass
        return results if results else None

    def _backpropagate(self, node: HyperNode, delta_value: float) -> None:
        """
        Backpropagate value update to ancestors.

        Args:
            node: Starting node
            delta_value: Value change to propagate

        Process:
            - Update node.visits += 1
            - Update node.value using incremental mean
            - Recurse to parent until root
        """
        current = node
        while current is not None:
            current.visits += 1
            # Incremental mean
            current.value += (delta_value - current.value) / current.visits
            current = current.parent

    def _evaluate_node(self, node: HyperNode) -> float:
        """Evaluate single node with model critic."""
        return self._evaluate_node_batch([node])[0]

    def _evaluate_node_batch(self, nodes: List[HyperNode]) -> List[float]:
        """
        Batch evaluate nodes with model critic.

        Args:
            nodes: List of nodes to evaluate

        Returns:
            List of value estimates

        Process:
            1. Convert expressions to graphs and fingerprints
            2. Batch through model.get_value()
            3. Return critic scores
        """
        if not nodes:
            return []

        # CRITICAL FIX: Precompute fingerprints for batching
        graphs = []
        fingerprints = []

        for node in nodes:
            try:
                # CRITICAL FIX: Use expr_to_graph for graphs
                graph = expr_to_graph(node.expr)
                graphs.append(graph)

                # CRITICAL FIX: Use SemanticFingerprint class
                fp = self.fingerprint_computer.compute(node.expr)
                fingerprints.append(fp)
            except:
                # Error parsing - assign low value
                return [0.0] * len(nodes)

        # Batch graphs
        graph_batch = pyg_data.Batch.from_data_list(graphs).to(self.device)

        # Stack fingerprints
        import numpy as np
        fp_batch = torch.from_numpy(np.stack(fingerprints)).float().to(self.device)

        # Evaluate
        with torch.no_grad():
            values = self.model.get_value(graph_batch, fp_batch)
            return values.squeeze(-1).cpu().tolist()

    def _extract_solution(self, root: HyperNode) -> Tuple[str, List[str]]:
        """
        Extract best solution and proof trace.

        Args:
            root: Search tree root

        Returns:
            (best_expr, proof_trace)

        Process:
            - Traverse tree following highest-value edges
            - Build proof trace showing tactic applications
            - Return leaf node with highest value and simplest expression
        """
        proof_trace = []
        current = root
        best_expr = root.expr
        best_depth = get_ast_depth(root.ast)

        # Traverse to best leaf
        while current.children:
            best_child = None
            best_value = float('-inf')

            for edge in current.children:
                for child in edge.children:
                    child_depth = get_ast_depth(child.ast)
                    # Prefer simpler expressions
                    adjusted_value = child.value - 0.1 * child_depth

                    if adjusted_value > best_value:
                        best_value = adjusted_value
                        best_child = child
                        best_edge = edge

            if best_child is None:
                break

            # Record tactic
            proof_trace.append(f"{best_edge.tactic.value}: {current.expr} -> {best_child.expr}")

            # Update best if simpler
            child_depth = get_ast_depth(best_child.ast)
            if child_depth < best_depth:
                best_depth = child_depth
                best_expr = best_child.expr

            current = best_child

        return (best_expr, proof_trace)
