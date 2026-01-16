"""
MBA expression evaluation utilities.

CRITICAL MODULE: Core expression evaluator for MBA expressions with support
for bitwise operations and modular arithmetic.
"""

import ast
import operator
import re
from typing import Dict, List, Optional

from src.constants import OPERATORS


def tokenize_expr(expr: str) -> List[str]:
    """
    Tokenize MBA expression string.

    Args:
        expr: Expression string (e.g., "(x&y)+(x^y)")

    Returns:
        List of tokens including operators, variables, constants, and parentheses

    Example:
        >>> tokenize_expr("(x&y)+2")
        ['(', 'x', '&', 'y', ')', '+', '2']
    """
    # Pattern matches: variables (x, y, x0-x7), numbers, operators, parens
    pattern = r'([a-z]\d?|[0-9]+|[+\-*&|^~()])'
    tokens = re.findall(pattern, expr.replace(' ', ''))
    return tokens


def parse_expr(expr: str) -> ast.AST:
    """
    Parse expression to Python AST.

    Converts MBA notation to Python notation:
    - & -> &
    - | -> |
    - ^ -> ^
    - ~ -> ~

    Args:
        expr: Expression string

    Returns:
        Python AST node

    Raises:
        SyntaxError: If expression is malformed
    """
    return ast.parse(expr, mode='eval')


def evaluate_expr(expr: str, var_values: Dict[str, int], width: int = 64) -> int:
    """
    Evaluate MBA expression with given variable bindings.

    Handles: +, -, *, &, |, ^, ~ operators with modular arithmetic.

    Args:
        expr: Expression string (e.g., "(x&y)+(x^y)")
        var_values: Dictionary mapping variable names to integer values
        width: Bit width for modular arithmetic (8, 16, 32, or 64)

    Returns:
        Integer result, wrapped to [0, 2^width)

    Raises:
        ValueError: If expression contains undefined variables
        SyntaxError: If expression is malformed
        ZeroDivisionError: If division by zero occurs

    Example:
        >>> evaluate_expr("x&y", {"x": 5, "y": 3}, width=8)
        1
        >>> evaluate_expr("(x&y)+(x^y)", {"x": 5, "y": 3}, width=8)
        7
    """
    modulo = 1 << width
    mask = modulo - 1

    class ExprEvaluator(ast.NodeVisitor):
        """AST visitor for expression evaluation."""

        def visit_Expression(self, node):
            return self.visit(node.body)

        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.BitAnd: operator.and_,
                ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
            }

            if type(node.op) in ops:
                result = ops[type(node.op)](left, right)
                return result & mask
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)

            if isinstance(node.op, ast.Invert):
                return (~operand) & mask
            elif isinstance(node.op, ast.USub):
                return ((-operand) & mask)
            elif isinstance(node.op, ast.UAdd):
                return operand & mask
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        def visit_Name(self, node):
            var_name = node.id
            if var_name not in var_values:
                raise ValueError(f"Undefined variable: {var_name}")
            return var_values[var_name] & mask

        def visit_Constant(self, node):
            return node.value & mask

        def visit_Num(self, node):
            return node.n & mask

    try:
        tree = parse_expr(expr)
        evaluator = ExprEvaluator()
        result = evaluator.visit(tree)
        return result & mask
    except Exception as e:
        raise type(e)(f"Error evaluating expression '{expr}': {str(e)}")


def random_inputs(
    num_vars: int,
    width: int = 64,
    count: int = 1,
    var_names: Optional[List[str]] = None
) -> List[Dict[str, int]]:
    """
    Generate random input dictionaries for testing.

    Args:
        num_vars: Number of variables to generate
        width: Bit width (values in range [0, 2^width))
        count: Number of input dictionaries to generate
        var_names: Optional list of variable names. If None, uses ['x', 'y', 'z', ...]

    Returns:
        List of dictionaries mapping variable names to random values

    Example:
        >>> inputs = random_inputs(2, width=8, count=3)
        >>> len(inputs)
        3
        >>> all('x' in inp and 'y' in inp for inp in inputs)
        True
    """
    import random

    if var_names is None:
        if num_vars <= 3:
            var_names = ['x', 'y', 'z'][:num_vars]
        else:
            var_names = [f'x{i}' for i in range(num_vars)]

    max_val = (1 << width) - 1
    result = []

    for _ in range(count):
        inp = {name: random.randint(0, max_val) for name in var_names}
        result.append(inp)

    return result


def expressions_equal(
    expr1: str,
    expr2: str,
    num_samples: int = 100,
    width: int = 64,
    var_names: Optional[List[str]] = None
) -> bool:
    """
    Test if two expressions are equivalent via random sampling.

    Args:
        expr1: First expression
        expr2: Second expression
        num_samples: Number of random test cases
        width: Bit width for evaluation
        var_names: Optional list of variable names. If None, extracts from expressions

    Returns:
        True if all random samples match, False otherwise

    Example:
        >>> expressions_equal("x|y", "(x&y)+(x^y)", num_samples=100)
        True
        >>> expressions_equal("x&y", "x|y", num_samples=10)
        False
    """
    if var_names is None:
        tokens1 = tokenize_expr(expr1)
        tokens2 = tokenize_expr(expr2)
        var_set = set()
        for token in tokens1 + tokens2:
            if token and token[0].isalpha() and token not in OPERATORS:
                var_set.add(token)
        var_names = sorted(var_set)

    if not var_names:
        try:
            val1 = evaluate_expr(expr1, {}, width)
            val2 = evaluate_expr(expr2, {}, width)
            return val1 == val2
        except:
            return False

    num_vars = len(var_names)
    test_inputs = random_inputs(num_vars, width, num_samples, var_names)

    try:
        for inp in test_inputs:
            val1 = evaluate_expr(expr1, inp, width)
            val2 = evaluate_expr(expr2, inp, width)
            if val1 != val2:
                return False
        return True
    except:
        return False
