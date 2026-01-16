"""
Z3 SMT solver interface for formal verification.

Provides utilities for converting MBA expressions to Z3 bitvector expressions
and verifying semantic equivalence.
"""

import ast
from typing import Dict, Optional

try:
    from z3 import (
        BitVec, BitVecVal, BitVecNumRef, Solver, sat, unsat, BoolRef
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from src.utils.expr_eval import tokenize_expr


def expr_to_z3(expr: str, width: int = 64, var_cache: Optional[Dict[str, 'BitVecNumRef']] = None):
    """
    Convert MBA expression string to Z3 bitvector expression.

    Args:
        expr: Expression string (e.g., "(x&y)+(x^y)")
        width: Bit width for bitvector
        var_cache: Optional dictionary for caching Z3 variables

    Returns:
        Z3 bitvector expression

    Raises:
        ImportError: If z3-solver is not installed
        ValueError: If expression contains unsupported operators
        SyntaxError: If expression is malformed

    Example:
        >>> from z3 import BitVec
        >>> x = BitVec('x', 64)
        >>> y = BitVec('y', 64)
        >>> # expr_to_z3("x&y", 64) produces equivalent Z3 expression
    """
    if not Z3_AVAILABLE:
        raise ImportError("z3-solver not installed. Install with: pip install z3-solver")

    if var_cache is None:
        var_cache = {}

    class Z3Converter(ast.NodeVisitor):
        """AST visitor for converting to Z3 expressions."""

        def visit_Expression(self, node):
            return self.visit(node.body)

        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.BitAnd):
                return left & right
            elif isinstance(node.op, ast.BitOr):
                return left | right
            elif isinstance(node.op, ast.BitXor):
                return left ^ right
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)

            if isinstance(node.op, ast.Invert):
                return ~operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        def visit_Name(self, node):
            var_name = node.id
            if var_name not in var_cache:
                var_cache[var_name] = BitVec(var_name, width)
            return var_cache[var_name]

        def visit_Constant(self, node):
            return BitVecVal(node.value, width)

        def visit_Num(self, node):
            return BitVecVal(node.n, width)

    tree = ast.parse(expr, mode='eval')
    converter = Z3Converter()
    return converter.visit(tree)


def verify_equivalence(
    expr1: str,
    expr2: str,
    width: int = 64,
    timeout_ms: int = 1000
) -> bool:
    """
    Verify two expressions are semantically equivalent using Z3.

    Args:
        expr1: First expression
        expr2: Second expression
        width: Bit width for bitvector operations
        timeout_ms: Timeout in milliseconds (default 1000ms)

    Returns:
        True if expressions are proven equivalent
        False if not equivalent or timeout occurs

    Example:
        >>> verify_equivalence("x|y", "(x&y)+(x^y)")
        True
        >>> verify_equivalence("x&y", "x|y")
        False
    """
    if not Z3_AVAILABLE:
        return False

    try:
        var_cache = {}
        z3_expr1 = expr_to_z3(expr1, width, var_cache)
        z3_expr2 = expr_to_z3(expr2, width, var_cache)

        solver = Solver()
        solver.set('timeout', timeout_ms)
        solver.add(z3_expr1 != z3_expr2)

        result = solver.check()

        if result == unsat:
            return True
        else:
            return False

    except Exception:
        return False


def find_counterexample(
    expr1: str,
    expr2: str,
    width: int = 64,
    timeout_ms: int = 1000
) -> Optional[Dict[str, int]]:
    """
    Find input values where expressions differ, or None if equivalent.

    Args:
        expr1: First expression
        expr2: Second expression
        width: Bit width for bitvector operations
        timeout_ms: Timeout in milliseconds

    Returns:
        Dictionary mapping variable names to counterexample values,
        or None if expressions are equivalent or timeout occurs

    Example:
        >>> result = find_counterexample("x&y", "x|y")
        >>> result is not None
        True
        >>> # result will be something like {'x': 1, 'y': 2}
    """
    if not Z3_AVAILABLE:
        return None

    try:
        var_cache = {}
        z3_expr1 = expr_to_z3(expr1, width, var_cache)
        z3_expr2 = expr_to_z3(expr2, width, var_cache)

        solver = Solver()
        solver.set('timeout', timeout_ms)
        solver.add(z3_expr1 != z3_expr2)

        result = solver.check()

        if result == sat:
            model = solver.model()
            counterexample = {}
            for var_name, var_z3 in var_cache.items():
                val = model.eval(var_z3)
                counterexample[var_name] = val.as_long()
            return counterexample
        else:
            return None

    except Exception:
        return None
