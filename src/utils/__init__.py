"""Utility modules for MBA Deobfuscator."""

from src.utils.config import Config
from src.utils.logging import setup_logging, setup_wandb
from src.utils.expr_eval import (
    tokenize_expr,
    parse_expr,
    evaluate_expr,
    random_inputs,
    expressions_equal
)
from src.utils.z3_interface import (
    expr_to_z3,
    verify_equivalence,
    find_counterexample
)
from src.utils.metrics import (
    exact_match,
    z3_accuracy,
    simplification_ratio,
    syntax_valid,
    avg_simplification_ratio,
    syntax_accuracy
)
# Note: verification_utils imports from src.inference.verify, which creates
# a circular dependency. Import directly when needed instead.
# from src.utils.verification_utils import safe_verify, check_syntax

__all__ = [
    'Config',
    'setup_logging',
    'setup_wandb',
    'tokenize_expr',
    'parse_expr',
    'evaluate_expr',
    'random_inputs',
    'expressions_equal',
    'expr_to_z3',
    'verify_equivalence',
    'find_counterexample',
    'exact_match',
    'z3_accuracy',
    'simplification_ratio',
    'syntax_valid',
    'avg_simplification_ratio',
    'syntax_accuracy',
    # 'safe_verify',  # Import from verification_utils directly
    # 'check_syntax',  # Import from verification_utils directly
]
