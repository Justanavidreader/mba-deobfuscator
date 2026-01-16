"""
Evaluation metrics for model performance.

Provides metrics for assessing simplification quality, including exact match,
semantic equivalence, and simplification ratios.
"""

import re
from typing import List

from src.utils.expr_eval import tokenize_expr, parse_expr
from src.utils.z3_interface import verify_equivalence


def exact_match(pred: str, target: str) -> bool:
    """
    Check if prediction exactly matches target (normalized).

    Normalization removes whitespace and standardizes formatting.

    Args:
        pred: Predicted expression
        target: Target expression

    Returns:
        True if expressions match exactly after normalization

    Example:
        >>> exact_match("x + y", "x+y")
        True
        >>> exact_match("x&y", "x|y")
        False
    """
    def normalize(expr: str) -> str:
        """Normalize expression by removing whitespace."""
        return ''.join(expr.split())

    return normalize(pred) == normalize(target)


def z3_accuracy(
    preds: List[str],
    targets: List[str],
    inputs: List[str],
    width: int = 64,
    timeout_ms: int = 1000
) -> float:
    """
    Compute % of predictions that are Z3-verified equivalent to inputs.

    Args:
        preds: List of predicted simplified expressions
        targets: List of target simplified expressions (for reference, not used)
        inputs: List of original obfuscated expressions
        width: Bit width for verification
        timeout_ms: Z3 timeout per verification

    Returns:
        Fraction of predictions verified equivalent to their inputs [0.0, 1.0]

    Example:
        >>> preds = ["x|y", "x&y"]
        >>> inputs = ["(x&y)+(x^y)", "(x|y)-(x^y)"]
        >>> z3_accuracy(preds, [], inputs)
        1.0
    """
    if not preds or not inputs or len(preds) != len(inputs):
        return 0.0

    correct = 0
    for pred, inp in zip(preds, inputs):
        if verify_equivalence(inp, pred, width, timeout_ms):
            correct += 1

    return correct / len(preds)


def simplification_ratio(input_expr: str, output_expr: str) -> float:
    """
    Compute len(output) / len(input) token ratio.

    Lower ratio indicates better simplification.

    Args:
        input_expr: Original expression
        output_expr: Simplified expression

    Returns:
        Ratio of output tokens to input tokens

    Example:
        >>> simplification_ratio("(x&y)+(x^y)", "x|y")
        0.2
        >>> simplification_ratio("x+y", "x+y")
        1.0
    """
    input_tokens = tokenize_expr(input_expr)
    output_tokens = tokenize_expr(output_expr)

    if len(input_tokens) == 0:
        return 1.0

    return len(output_tokens) / len(input_tokens)


def syntax_valid(expr: str) -> bool:
    """
    Check if expression parses without error.

    Args:
        expr: Expression string to validate

    Returns:
        True if expression is syntactically valid

    Example:
        >>> syntax_valid("x+y")
        True
        >>> syntax_valid("x++y")
        False
        >>> syntax_valid("(x&y")
        False
    """
    try:
        parse_expr(expr)
        return True
    except:
        return False


def avg_simplification_ratio(inputs: List[str], outputs: List[str]) -> float:
    """
    Compute average simplification ratio across a dataset.

    Args:
        inputs: List of input expressions
        outputs: List of output expressions

    Returns:
        Average simplification ratio

    Example:
        >>> inputs = ["(x&y)+(x^y)", "x+y+z"]
        >>> outputs = ["x|y", "x+y+z"]
        >>> avg_simplification_ratio(inputs, outputs)
        0.6
    """
    if not inputs or not outputs or len(inputs) != len(outputs):
        return 1.0

    ratios = [simplification_ratio(inp, out) for inp, out in zip(inputs, outputs)]
    return sum(ratios) / len(ratios)


def syntax_accuracy(predictions: List[str]) -> float:
    """
    Compute fraction of syntactically valid predictions.

    Args:
        predictions: List of predicted expressions

    Returns:
        Fraction of valid predictions [0.0, 1.0]

    Example:
        >>> syntax_accuracy(["x+y", "x++y", "x|y"])
        0.667
    """
    if not predictions:
        return 0.0

    valid = sum(1 for pred in predictions if syntax_valid(pred))
    return valid / len(predictions)
