"""
Semantic fingerprint computation for MBA expressions.

Computes a 448-dimensional semantic fingerprint consisting of:
- Symbolic features (32 dims)
- Corner evaluations (256 dims)
- Random hash (64 dims)
- Derivatives (32 dims)
- Truth table (64 dims)
"""

import re
import numpy as np
from typing import Dict, List, Optional

from src.constants import (
    FINGERPRINT_DIM,
    SYMBOLIC_DIM,
    CORNER_DIM,
    RANDOM_DIM,
    DERIVATIVE_DIM,
    TRUTH_TABLE_DIM,
    BIT_WIDTHS,
    TRUTH_TABLE_VARS,
    get_corner_values,
)
# Direct import to avoid __init__.py chain that pulls in torch_scatter
import sys
import os
_utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)
from expr_eval import evaluate_expr


class SemanticFingerprint:
    """
    448-dimensional semantic fingerprint for MBA expressions.

    Components:
        - Symbolic features (32): degree, op counts, structure
        - Corner evaluations (256): 4 widths x 64 corner cases
        - Random hash (64): 4 widths x 16 random inputs
        - Derivatives (32): 4 widths x 8 partial derivative approximations
        - Truth table (64): 2^6 boolean outputs for 6 variables
    """

    def __init__(self, seed: int = 42):
        """
        Initialize fingerprint computer.

        Args:
            seed: Random seed for reproducible random inputs
        """
        self.rng = np.random.RandomState(seed)
        self._init_random_inputs()

    def _init_random_inputs(self):
        """Generate random input values for hash computation."""
        # Generate random inputs for each bit width
        self.random_inputs = {}
        for width in BIT_WIDTHS:
            mask = (1 << width) - 1
            # 16 random samples per width
            samples = []
            for _ in range(16):
                sample = {}
                for i in range(8):
                    # Generate random bytes and mask to width
                    if width <= 31:
                        # Use randint for smaller widths
                        sample[f'x{i}'] = int(self.rng.randint(0, mask + 1))
                    else:
                        # For larger widths, generate from uniform distribution
                        # Generate multiple 31-bit values and combine
                        val = 0
                        bits_remaining = width
                        while bits_remaining > 0:
                            bits_to_gen = min(31, bits_remaining)
                            val = (val << bits_to_gen) | int(self.rng.randint(0, 1 << bits_to_gen))
                            bits_remaining -= bits_to_gen
                        sample[f'x{i}'] = val & mask
                samples.append(sample)
            self.random_inputs[width] = samples

    def compute(self, expr: str) -> np.ndarray:
        """
        Compute 448-dim fingerprint for expression.

        Args:
            expr: MBA expression string

        Returns:
            448-dimensional numpy array (float32)
        """
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

        # Extract variables from expression
        variables = self._extract_variables(expr)

        # Compute each component
        offset = 0

        # Symbolic features (32 dims)
        fp[offset:offset + SYMBOLIC_DIM] = self._symbolic_features(expr)
        offset += SYMBOLIC_DIM

        # Corner evaluations (256 dims)
        fp[offset:offset + CORNER_DIM] = self._corner_evals(expr, variables)
        offset += CORNER_DIM

        # Random hash (64 dims)
        fp[offset:offset + RANDOM_DIM] = self._random_hash(expr, variables)
        offset += RANDOM_DIM

        # Derivatives (32 dims)
        fp[offset:offset + DERIVATIVE_DIM] = self._derivatives(expr, variables)
        offset += DERIVATIVE_DIM

        # Truth table (64 dims)
        fp[offset:offset + TRUTH_TABLE_DIM] = self._truth_table(expr, variables)
        offset += TRUTH_TABLE_DIM

        assert offset == FINGERPRINT_DIM

        return fp

    def _extract_variables(self, expr: str) -> List[str]:
        """Extract variable names from expression."""
        # Find all variable names (x, y, z, x0, x1, etc.)
        variables = set()
        for match in re.finditer(r'\b([a-z]\d?)\b', expr, re.IGNORECASE):
            var = match.group(1)
            if var not in ['and', 'or', 'xor', 'not']:  # Exclude keywords
                variables.add(var)
        return sorted(variables)

    def _symbolic_features(self, expr: str) -> np.ndarray:
        """
        Extract structural features from expression.

        Features (32 dims):
        - Expression length (normalized)
        - Number of each operator type (7 types)
        - Number of variables
        - Number of constants
        - Parenthesis depth
        - Total depth estimate
        - Variable usage counts (8 dims for x0-x7)
        - Constant value statistics (mean, std, min, max)
        """
        features = np.zeros(SYMBOLIC_DIM, dtype=np.float32)
        idx = 0

        # Expression length (normalized)
        features[idx] = min(1.0, len(expr) / 200.0)
        idx += 1

        # Operator counts (normalized)
        operators = {'+': 0, '-': 0, '*': 0, '&': 0, '|': 0, '^': 0, '~': 0}
        for op in operators.keys():
            count = expr.count(op)
            operators[op] = count  # Store count back in dict for total_depth
            features[idx] = min(1.0, count / 10.0)
            idx += 1

        # Number of variables (normalized)
        variables = self._extract_variables(expr)
        features[idx] = min(1.0, len(variables) / 8.0)
        idx += 1

        # Number of constants
        constants = re.findall(r'\b\d+\b', expr)
        features[idx] = min(1.0, len(constants) / 10.0)
        idx += 1

        # Parenthesis depth
        max_depth = 0
        current_depth = 0
        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        features[idx] = min(1.0, max_depth / 10.0)
        idx += 1

        # Total depth estimate (operators + parens)
        total_ops = sum(operators.values())
        features[idx] = min(1.0, (total_ops + max_depth) / 20.0)
        idx += 1

        # Variable usage counts (x0-x7)
        for i in range(8):
            var_name = f'x{i}'
            count = len(re.findall(rf'\b{var_name}\b', expr))
            features[idx] = min(1.0, count / 5.0)
            idx += 1

        # Constant statistics
        if constants:
            const_values = [int(c) for c in constants]
            features[idx] = np.mean(const_values) / 256.0
            features[idx + 1] = np.std(const_values) / 256.0
            features[idx + 2] = np.min(const_values) / 256.0
            features[idx + 3] = np.max(const_values) / 256.0
        idx += 4

        return features

    def _corner_evals(self, expr: str, variables: List[str]) -> np.ndarray:
        """
        Evaluate at corner cases across bit widths.

        256 dims = 4 bit widths × 64 corner cases
        """
        features = np.zeros(CORNER_DIM, dtype=np.float32)
        idx = 0

        for width in BIT_WIDTHS:
            mask = (1 << width) - 1
            corners = get_corner_values(width)

            # Use all corners to generate 64 assignments per width (256 total)
            for corner_set in self._generate_corner_assignments(variables, corners, width):
                result = evaluate_expr(expr, corner_set, width)
                if result is not None:
                    # Normalize to [0, 1]
                    features[idx] = result / mask
                else:
                    features[idx] = 0.0
                idx += 1

                if idx >= CORNER_DIM:
                    break

            if idx >= CORNER_DIM:
                break

        return features

    def _generate_corner_assignments(
        self, variables: List[str], corners: List[int], width: int
    ) -> List[Dict[str, int]]:
        """
        Generate variable assignments using corner values.

        Creates exactly 64 assignments per width (256 total across 4 widths)
        to match CORNER_DIM specification.
        """
        mask = (1 << width) - 1
        assignments = []

        if not variables:
            # No variables - single assignment with dummy values
            return [{'x0': 0}]

        # Generate 64 combinations per width using corner values
        num_corners = len(corners)
        num_assignments_per_width = 64

        for i in range(num_assignments_per_width):
            assignment = {}
            # Assign corner values to each variable, cycling through corners
            # Use different offsets per assignment to create diverse combinations
            for j, var in enumerate(variables[:8]):  # Max 8 variables
                # Use (i + j) pattern to create varied combinations
                corner_idx = (i + j * 7) % num_corners  # 7 for better distribution
                assignment[var] = corners[corner_idx] & mask
            assignments.append(assignment)

        return assignments

    def _random_hash(self, expr: str, variables: List[str]) -> np.ndarray:
        """
        Evaluate at random inputs for hash-like signature.

        64 dims = 4 bit widths × 16 random samples
        """
        features = np.zeros(RANDOM_DIM, dtype=np.float32)
        idx = 0

        for width in BIT_WIDTHS:
            mask = (1 << width) - 1
            samples = self.random_inputs[width]

            for sample in samples:
                # Map expression variables to sample values
                var_assignment = {var: sample.get(var, sample['x0']) for var in variables}

                result = evaluate_expr(expr, var_assignment, width)
                if result is not None:
                    # Normalize to [0, 1]
                    features[idx] = result / mask
                else:
                    features[idx] = 0.0
                idx += 1

        return features

    def _derivatives(self, expr: str, variables: List[str]) -> np.ndarray:
        """
        Approximate partial derivatives via finite differences.

        32 dims = 4 bit widths × 8 derivative approximations
        """
        features = np.zeros(DERIVATIVE_DIM, dtype=np.float32)
        idx = 0

        epsilon = 1

        for width in BIT_WIDTHS:
            mask = (1 << width) - 1
            # Base point (all variables = midpoint)
            midpoint = 1 << (width - 1)
            base_point = {var: midpoint for var in (variables or ['x0'])}

            # Compute partial derivatives for each variable
            for i in range(8):
                if i < len(variables):
                    var = variables[i]
                    # f(x + ε) - f(x)
                    perturbed = base_point.copy()
                    perturbed[var] = (base_point[var] + epsilon) & mask

                    f_base = evaluate_expr(expr, base_point, width)
                    f_perturbed = evaluate_expr(expr, perturbed, width)

                    if f_base is not None and f_perturbed is not None:
                        derivative = ((f_perturbed - f_base) & mask) / epsilon
                        features[idx] = min(1.0, derivative / mask)
                    else:
                        features[idx] = 0.0
                else:
                    features[idx] = 0.0

                idx += 1
                if idx >= DERIVATIVE_DIM:
                    break

            if idx >= DERIVATIVE_DIM:
                break

        return features

    def _truth_table(self, expr: str, variables: List[str]) -> np.ndarray:
        """
        Compute 64-entry truth table (LSB of output for 6-var inputs).

        Truth table: evaluate expression for all 2^6 = 64 combinations
        of first 6 variables, recording LSB (bit 0) of output.

        64 dims = 64 truth table entries (one per input combination)
        """
        features = np.zeros(TRUTH_TABLE_DIM, dtype=np.float32)

        # Use first TRUTH_TABLE_VARS (6) variables
        vars_to_use = variables[:TRUTH_TABLE_VARS] if variables else []

        # Pad with dummy variables if needed
        while len(vars_to_use) < TRUTH_TABLE_VARS:
            vars_to_use.append(f'_dummy{len(vars_to_use)}')

        # Enumerate all 2^6 = 64 input combinations
        for i in range(64):
            # Generate variable assignment from bit pattern
            assignment = {}
            for j in range(TRUTH_TABLE_VARS):
                var = vars_to_use[j]
                # Extract bit j from i
                assignment[var] = (i >> j) & 1

            # Evaluate expression (use 64-bit width)
            result = evaluate_expr(expr, assignment, 64)

            if result is not None:
                # Extract LSB
                features[i] = float(result & 1)
            else:
                features[i] = 0.0

        return features
