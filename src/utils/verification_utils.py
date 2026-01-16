"""
Shared verification utilities for safe equivalence checking.

Centralizes error handling for verification to avoid duplicate patterns
across trainers and evaluation scripts.
"""

from typing import Tuple, Optional
from src.inference.verify import ThreeTierVerifier
from src.data.tokenizer import MBATokenizer


def safe_verify(
    verifier: ThreeTierVerifier,
    input_expr: str,
    output_expr: str,
    use_z3: bool = False
) -> Tuple[bool, bool]:
    """
    Safely verify expression equivalence with proper error handling.

    Args:
        verifier: ThreeTierVerifier instance
        input_expr: Original obfuscated expression
        output_expr: Simplified output expression
        use_z3: Whether to use Z3 solver for formal verification

    Returns:
        Tuple of (syntax_valid, is_equivalent)
        - syntax_valid: True if output has valid syntax
        - is_equivalent: True if expressions are semantically equivalent
    """
    # Check syntax validity
    try:
        tokens = verifier.tokenizer.encode(output_expr)
        syntax_valid = len(tokens) > 2  # More than just SOS/EOS
    except Exception:
        return False, False

    if not syntax_valid:
        return False, False

    # Verify equivalence
    # verify_batch can return None or empty list on failure
    results = verifier.verify_batch(input_expr, [output_expr])

    if not results or len(results) == 0:
        return True, False

    result = results[0]

    if use_z3:
        equiv = result.z3_verified or result.exec_valid
    else:
        equiv = result.exec_valid

    return True, equiv


def check_syntax(tokenizer: MBATokenizer, expr: str) -> bool:
    """
    Check if expression has valid syntax.

    Args:
        tokenizer: MBATokenizer instance
        expr: Expression to check

    Returns:
        True if expression has valid syntax
    """
    try:
        tokens = tokenizer.encode(expr)
        return len(tokens) > 2
    except Exception:
        return False
