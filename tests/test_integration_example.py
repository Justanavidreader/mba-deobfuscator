"""
Integration example showing how to use the utility modules together.

This is not a test file but rather a demonstration of the API.
"""

from src.utils import (
    Config,
    setup_logging,
    tokenize_expr,
    evaluate_expr,
    random_inputs,
    expressions_equal,
    verify_equivalence,
    simplification_ratio,
    syntax_valid,
    exact_match
)


def example_expression_evaluation():
    """Example: Evaluating MBA expressions."""
    print("=== Expression Evaluation ===")

    expr = "(x&y)+(x^y)"
    var_values = {"x": 5, "y": 3}

    result = evaluate_expr(expr, var_values, width=8)
    print(f"Expression: {expr}")
    print(f"Variables: {var_values}")
    print(f"Result: {result}")
    print()


def example_equivalence_checking():
    """Example: Checking if two expressions are equivalent."""
    print("=== Equivalence Checking ===")

    expr1 = "(x&y)+(x^y)"
    expr2 = "x|y"

    # Random sampling
    equal_random = expressions_equal(expr1, expr2, num_samples=100, width=8)
    print(f"Expression 1: {expr1}")
    print(f"Expression 2: {expr2}")
    print(f"Equal (random sampling): {equal_random}")

    # Z3 formal verification
    try:
        equal_z3 = verify_equivalence(expr1, expr2, width=8)
        print(f"Equal (Z3 verification): {equal_z3}")
    except ImportError:
        print("Z3 not available")
    print()


def example_tokenization():
    """Example: Tokenizing expressions."""
    print("=== Tokenization ===")

    expr = "(x&y)+(x^y)"
    tokens = tokenize_expr(expr)

    print(f"Expression: {expr}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print()


def example_metrics():
    """Example: Computing evaluation metrics."""
    print("=== Metrics ===")

    input_expr = "(x&y)+(x^y)"
    output_expr = "x|y"

    print(f"Input: {input_expr}")
    print(f"Output: {output_expr}")
    print(f"Simplification ratio: {simplification_ratio(input_expr, output_expr):.2f}")
    print(f"Syntax valid (input): {syntax_valid(input_expr)}")
    print(f"Syntax valid (output): {syntax_valid(output_expr)}")
    print(f"Exact match: {exact_match(input_expr, output_expr)}")
    print()


def example_config():
    """Example: Loading configuration."""
    print("=== Configuration ===")

    try:
        config = Config("configs/example.yaml")
        print(f"Model hidden dim: {config.model.hidden_dim}")
        print(f"Training batch size: {config.training.batch_size}")
        print(f"Training LR: {config.training.learning_rate}")
        print(f"Missing key with default: {config.get('nonexistent.key', 'default_value')}")
    except FileNotFoundError:
        print("Config file not found")
    print()


def example_random_testing():
    """Example: Random testing of expressions."""
    print("=== Random Testing ===")

    inputs = random_inputs(num_vars=2, width=8, count=5)
    expr = "x&y"

    print(f"Expression: {expr}")
    print(f"Random test inputs:")
    for inp in inputs:
        result = evaluate_expr(expr, inp, width=8)
        print(f"  {inp} -> {result}")
    print()


if __name__ == '__main__':
    # Setup logging
    logger = setup_logging(__name__, level="INFO")
    logger.info("Starting utility examples")

    # Run examples
    example_expression_evaluation()
    example_equivalence_checking()
    example_tokenization()
    example_metrics()
    example_config()
    example_random_testing()

    logger.info("Examples complete")
