"""
Comprehensive tests for utility modules.

Tests cover expression evaluation, Z3 verification, metrics, and configuration.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.config import Config
from src.utils.expr_eval import (
    tokenize_expr,
    parse_expr,
    evaluate_expr,
    random_inputs,
    expressions_equal
)
from src.utils.metrics import (
    exact_match,
    simplification_ratio,
    syntax_valid,
    syntax_accuracy,
    avg_simplification_ratio
)


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for Config class."""

    def test_load_valid_config(self):
        """Test loading valid YAML configuration."""
        yaml_content = """
model:
  hidden_dim: 256
  num_layers: 4
training:
  batch_size: 32
  lr: 0.001
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = Config(config_path)
            assert config.model.hidden_dim == 256
            assert config.model.num_layers == 4
            assert config.training.batch_size == 32
        finally:
            os.unlink(config_path)

    def test_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent.yaml")

    def test_get_with_default(self):
        """Test get method with default values."""
        yaml_content = "model:\n  dim: 128\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = Config(config_path)
            assert config.get('model.dim') == 128
            assert config.get('model.missing', 999) == 999
            assert config.get('nonexistent', 'default') == 'default'
        finally:
            os.unlink(config_path)

    def test_attribute_error(self):
        """Test attribute error for missing keys."""
        yaml_content = "model:\n  dim: 128\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = Config(config_path)
            with pytest.raises(AttributeError):
                _ = config.nonexistent
        finally:
            os.unlink(config_path)


# =============================================================================
# Expression Evaluation Tests
# =============================================================================

class TestExprEval:
    """Tests for expression evaluation utilities."""

    def test_tokenize_simple(self):
        """Test tokenization of simple expressions."""
        tokens = tokenize_expr("x+y")
        assert tokens == ['x', '+', 'y']

    def test_tokenize_complex(self):
        """Test tokenization of complex expressions."""
        tokens = tokenize_expr("(x&y)+(x^y)")
        assert tokens == ['(', 'x', '&', 'y', ')', '+', '(', 'x', '^', 'y', ')']

    def test_tokenize_with_constants(self):
        """Test tokenization with constants."""
        tokens = tokenize_expr("x+123")
        assert tokens == ['x', '+', '123']

    def test_tokenize_with_spaces(self):
        """Test tokenization handles whitespace."""
        tokens = tokenize_expr("x + y")
        assert tokens == ['x', '+', 'y']

    def test_evaluate_simple_addition(self):
        """Test evaluation of simple addition."""
        result = evaluate_expr("x+y", {"x": 5, "y": 3}, width=8)
        assert result == 8

    def test_evaluate_bitwise_and(self):
        """Test evaluation of bitwise AND."""
        result = evaluate_expr("x&y", {"x": 5, "y": 3}, width=8)
        assert result == 1

    def test_evaluate_bitwise_or(self):
        """Test evaluation of bitwise OR."""
        result = evaluate_expr("x|y", {"x": 5, "y": 3}, width=8)
        assert result == 7

    def test_evaluate_bitwise_xor(self):
        """Test evaluation of bitwise XOR."""
        result = evaluate_expr("x^y", {"x": 5, "y": 3}, width=8)
        assert result == 6

    def test_evaluate_bitwise_not(self):
        """Test evaluation of bitwise NOT."""
        result = evaluate_expr("~x", {"x": 0}, width=8)
        assert result == 255

    def test_evaluate_unary_minus(self):
        """Test evaluation of unary minus."""
        result = evaluate_expr("-x", {"x": 10}, width=8)
        assert result == 246

    def test_evaluate_complex_mba(self):
        """Test evaluation of complex MBA expression."""
        result = evaluate_expr("(x&y)+(x^y)", {"x": 5, "y": 3}, width=8)
        expected = evaluate_expr("x|y", {"x": 5, "y": 3}, width=8)
        assert result == expected
        assert result == 7

    def test_evaluate_overflow_8bit(self):
        """Test overflow behavior with 8-bit width."""
        result = evaluate_expr("x+y", {"x": 200, "y": 100}, width=8)
        assert result == 44

    def test_evaluate_overflow_16bit(self):
        """Test overflow behavior with 16-bit width."""
        result = evaluate_expr("x+y", {"x": 60000, "y": 10000}, width=16)
        assert result == 4464

    def test_evaluate_multiplication(self):
        """Test multiplication with overflow."""
        result = evaluate_expr("x*y", {"x": 10, "y": 30}, width=8)
        assert result == 44

    def test_evaluate_subtraction(self):
        """Test subtraction with underflow."""
        result = evaluate_expr("x-y", {"x": 5, "y": 10}, width=8)
        assert result == 251

    def test_evaluate_undefined_variable(self):
        """Test error handling for undefined variables."""
        with pytest.raises(ValueError):
            evaluate_expr("x+z", {"x": 5}, width=8)

    def test_evaluate_syntax_error(self):
        """Test error handling for syntax errors."""
        with pytest.raises(SyntaxError):
            evaluate_expr("x+++", {"x": 5}, width=8)

    def test_random_inputs_count(self):
        """Test random input generation count."""
        inputs = random_inputs(2, width=8, count=5)
        assert len(inputs) == 5

    def test_random_inputs_variables(self):
        """Test random input generation has correct variables."""
        inputs = random_inputs(2, width=8, count=3)
        for inp in inputs:
            assert 'x' in inp
            assert 'y' in inp

    def test_random_inputs_range(self):
        """Test random values are within bit width range."""
        inputs = random_inputs(3, width=8, count=10)
        for inp in inputs:
            for val in inp.values():
                assert 0 <= val < 256

    def test_random_inputs_custom_names(self):
        """Test random inputs with custom variable names."""
        inputs = random_inputs(3, width=8, count=2, var_names=['a', 'b', 'c'])
        for inp in inputs:
            assert set(inp.keys()) == {'a', 'b', 'c'}

    def test_expressions_equal_true(self):
        """Test expressions_equal for equivalent expressions."""
        assert expressions_equal("x|y", "(x&y)+(x^y)", num_samples=50, width=8)

    def test_expressions_equal_false(self):
        """Test expressions_equal for non-equivalent expressions."""
        assert not expressions_equal("x&y", "x|y", num_samples=10, width=8)

    def test_expressions_equal_identity(self):
        """Test expressions_equal for identical expressions."""
        assert expressions_equal("x+y", "x+y", num_samples=10, width=8)

    def test_expressions_equal_constants(self):
        """Test expressions_equal for constant expressions."""
        assert expressions_equal("5", "5", num_samples=1, width=8)
        assert not expressions_equal("5", "6", num_samples=1, width=8)


# =============================================================================
# Z3 Interface Tests
# =============================================================================

class TestZ3Interface:
    """Tests for Z3 verification interface."""

    def test_verify_equivalence_true(self):
        """Test verification of equivalent expressions."""
        try:
            from src.utils.z3_interface import verify_equivalence
            result = verify_equivalence("x|y", "(x&y)+(x^y)", width=8)
            assert result is True
        except ImportError:
            pytest.skip("z3-solver not installed")

    def test_verify_equivalence_false(self):
        """Test verification of non-equivalent expressions."""
        try:
            from src.utils.z3_interface import verify_equivalence
            result = verify_equivalence("x&y", "x|y", width=8)
            assert result is False
        except ImportError:
            pytest.skip("z3-solver not installed")

    def test_verify_equivalence_identity(self):
        """Test verification of identical expressions."""
        try:
            from src.utils.z3_interface import verify_equivalence
            result = verify_equivalence("x+y", "x+y", width=8)
            assert result is True
        except ImportError:
            pytest.skip("z3-solver not installed")

    def test_find_counterexample_exists(self):
        """Test finding counterexample for non-equivalent expressions."""
        try:
            from src.utils.z3_interface import find_counterexample
            from src.utils.expr_eval import evaluate_expr

            result = find_counterexample("x&y", "x|y", width=8)
            assert result is not None
            assert 'x' in result
            assert 'y' in result

            val1 = evaluate_expr("x&y", result, width=8)
            val2 = evaluate_expr("x|y", result, width=8)
            assert val1 != val2
        except ImportError:
            pytest.skip("z3-solver not installed")

    def test_find_counterexample_none(self):
        """Test finding counterexample for equivalent expressions."""
        try:
            from src.utils.z3_interface import find_counterexample
            result = find_counterexample("x|y", "(x&y)+(x^y)", width=8)
            assert result is None
        except ImportError:
            pytest.skip("z3-solver not installed")


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_exact_match_true(self):
        """Test exact match for identical expressions."""
        assert exact_match("x+y", "x+y")

    def test_exact_match_whitespace(self):
        """Test exact match with whitespace differences."""
        assert exact_match("x + y", "x+y")

    def test_exact_match_false(self):
        """Test exact match for different expressions."""
        assert not exact_match("x&y", "x|y")

    def test_simplification_ratio_improved(self):
        """Test simplification ratio for simplified expression."""
        ratio = simplification_ratio("(x&y)+(x^y)", "x|y")
        assert abs(ratio - 3/11) < 0.01

    def test_simplification_ratio_unchanged(self):
        """Test simplification ratio for unchanged expression."""
        ratio = simplification_ratio("x+y", "x+y")
        assert ratio == 1.0

    def test_simplification_ratio_worse(self):
        """Test simplification ratio for expanded expression."""
        ratio = simplification_ratio("x", "x+0")
        assert ratio == 3.0

    def test_syntax_valid_true(self):
        """Test syntax validation for valid expressions."""
        assert syntax_valid("x+y")
        assert syntax_valid("(x&y)|(x^y)")
        assert syntax_valid("~x")

    def test_syntax_valid_false(self):
        """Test syntax validation for invalid expressions."""
        assert not syntax_valid("x+++")
        assert not syntax_valid("(x&y")
        assert not syntax_valid("x y")

    def test_syntax_accuracy(self):
        """Test syntax accuracy computation."""
        preds = ["x+y", "x+++", "x|y", "(x&y"]
        accuracy = syntax_accuracy(preds)
        assert accuracy == 0.5

    def test_syntax_accuracy_all_valid(self):
        """Test syntax accuracy with all valid predictions."""
        preds = ["x+y", "x|y", "x&y"]
        accuracy = syntax_accuracy(preds)
        assert accuracy == 1.0

    def test_syntax_accuracy_empty(self):
        """Test syntax accuracy with empty list."""
        accuracy = syntax_accuracy([])
        assert accuracy == 0.0

    def test_avg_simplification_ratio(self):
        """Test average simplification ratio."""
        inputs = ["(x&y)+(x^y)", "x+y"]
        outputs = ["x|y", "x+y"]
        ratio = avg_simplification_ratio(inputs, outputs)
        expected = ((3/11) + 1.0) / 2
        assert abs(ratio - expected) < 0.01

    def test_avg_simplification_ratio_empty(self):
        """Test average simplification ratio with empty lists."""
        ratio = avg_simplification_ratio([], [])
        assert ratio == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
