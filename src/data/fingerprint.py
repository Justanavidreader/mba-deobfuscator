"""
Semantic fingerprint computation for MBA expressions.

Computes a 448-dimensional semantic fingerprint consisting of:
- Symbolic features (32 dims)
- Corner evaluations (256 dims)
- Random hash (64 dims)
- Derivatives (32 dims)
- Truth table (64 dims)

Optional C++ acceleration via pybind11:
- Falls back to pure Python if C++ module not available
"""

import re
import numpy as np
from typing import Dict, List, Optional
import warnings

from src.constants import (
    FINGERPRINT_DIM, FINGERPRINT_DIM_FULL,
    SYMBOLIC_DIM,
    CORNER_DIM,
    RANDOM_DIM,
    DERIVATIVE_DIM,
    TRUTH_TABLE_DIM,
    BIT_WIDTHS,
    TRUTH_TABLE_VARS,
    FINGERPRINT_MODE,
    get_corner_values,
)
# Direct import to avoid __init__.py chain that pulls in torch_scatter
import sys
import os
_utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)
from expr_eval import evaluate_expr

# Try to import C++ accelerated fingerprint module (optional)
try:
    import mba_fingerprint_cpp
    HAS_CPP_FINGERPRINT = True
except ImportError:
    HAS_CPP_FINGERPRINT = False
    # Only warn on first import, not every time
    if not hasattr(sys.modules[__name__], '_cpp_import_warned'):
        warnings.warn(
            "C++ fingerprint module not available. Using pure Python implementation. "
            "For better performance, build the C++ module with: "
            "cd ../MBA_Generator && cmake -B build && cmake --build build --config Release",
            ImportWarning
        )
        sys.modules[__name__]._cpp_import_warned = True


def has_cpp_acceleration() -> bool:
    """
    Check if C++ accelerated fingerprint computation is available.

    Returns:
        True if mba_fingerprint_cpp module is available, False otherwise
    """
    return HAS_CPP_FINGERPRINT


def get_implementation_info() -> dict:
    """
    Get information about the current fingerprint implementation.

    Returns:
        Dictionary with implementation details:
        - 'cpp_available': bool - C++ module available
        - 'cpp_version': str - C++ module version (if available)
        - 'fingerprint_dim': int - Fingerprint dimensions
    """
    info = {
        'cpp_available': HAS_CPP_FINGERPRINT,
        'cpp_version': None,
        'fingerprint_dim_full': FINGERPRINT_DIM_FULL,  # 448 (with derivatives)
        'fingerprint_dim_ml': FINGERPRINT_DIM,  # 416 (without derivatives)
    }

    if HAS_CPP_FINGERPRINT:
        try:
            info['cpp_version'] = getattr(mba_fingerprint_cpp, '__version__', 'unknown')
        except AttributeError:
            pass

    return info


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

    def __init__(self, seed: int = 42, use_cpp: bool = True):
        """
        Initialize fingerprint computer.

        Args:
            seed: Unused, kept for backward compatibility (now deterministic)
            use_cpp: If True and C++ module available, use C++ implementation for better performance
        """
        self.use_cpp = use_cpp and HAS_CPP_FINGERPRINT
        self._init_deterministic_inputs()

        if use_cpp and not HAS_CPP_FINGERPRINT:
            warnings.warn(
                "C++ fingerprint requested but not available. Using Python implementation.",
                RuntimeWarning
            )

    def _init_deterministic_inputs(self):
        """Generate deterministic test patterns for hash computation (no RNG)."""
        # Deterministic test patterns replace random inputs for C++/Python consistency
        self.random_inputs = {}

        for width in BIT_WIDTHS:
            mask = (1 << width) - 1
            samples = []

            # 16 deterministic test patterns per width
            # Pattern 0-3: all same value (0, 1, mid, max)
            samples.append({f'x{i}': 0 for i in range(8)})
            samples.append({f'x{i}': 1 for i in range(8)})
            samples.append({f'x{i}': (1 << (width - 1)) & mask for i in range(8)})  # Midpoint
            samples.append({f'x{i}': mask for i in range(8)})  # Max

            # Pattern 4-7: specific bit patterns
            samples.append({f'x{i}': (0xAA & mask) for i in range(8)})  # 10101010
            samples.append({f'x{i}': (0x55 & mask) for i in range(8)})  # 01010101
            samples.append({f'x{i}': (0xF0 & mask) for i in range(8)})  # 11110000
            samples.append({f'x{i}': (0x0F & mask) for i in range(8)})  # 00001111

            # Pattern 8-11: variable-specific patterns (each var gets different value)
            for pattern_idx in range(4):
                sample = {}
                for var_idx in range(8):
                    # Rotate bit pattern based on variable index
                    val = ((pattern_idx << var_idx) | (pattern_idx >> (8 - var_idx))) & 0xFF
                    sample[f'x{var_idx}'] = val & mask
                samples.append(sample)

            # Pattern 12-15: powers of 2 patterns
            for pattern_idx in range(4):
                sample = {}
                for var_idx in range(8):
                    bit_pos = (pattern_idx * 8 + var_idx) % width
                    sample[f'x{var_idx}'] = (1 << bit_pos) & mask
                samples.append(sample)

            self.random_inputs[width] = samples

    def compute(self, expr: str) -> np.ndarray:
        """
        Compute 448-dim fingerprint for expression.

        Based on FINGERPRINT_MODE constant:
        - "full": All components (symbolic + corner + random + derivative + truth_table)
        - "truth_table_only": Only truth table (zeros for other components)

        Args:
            expr: MBA expression string

        Returns:
            448-dimensional numpy array (float64/double)
        """
        # Try C++ implementation first if available and enabled
        if self.use_cpp:
            try:
                # NOTE: C++ implementation not yet complete
                # When ready, this will call: mba_fingerprint_cpp.compute_fingerprint(expr)
                # For now, fall through to Python implementation
                pass
            except (AttributeError, NotImplementedError) as e:
                # C++ function not implemented yet, fall back to Python
                pass

        # Python implementation (always available)
        # Always compute full 448-dim fingerprint (derivatives stripped at dataset layer)
        fp = np.zeros(FINGERPRINT_DIM_FULL, dtype=np.float64)

        # Extract variables from expression
        variables = self._extract_variables(expr)

        # Compute components based on mode
        offset = 0

        if FINGERPRINT_MODE == "truth_table_only":
            # Skip to truth table offset (symbolic + corner + random + derivative)
            offset = SYMBOLIC_DIM + CORNER_DIM + RANDOM_DIM + DERIVATIVE_DIM

            # Truth table (64 dims) - only component computed
            fp[offset:offset + TRUTH_TABLE_DIM] = self._truth_table(expr, variables)
            offset += TRUTH_TABLE_DIM

        else:  # "full" mode
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

        assert offset == FINGERPRINT_DIM_FULL, f"Computed {offset} dims, expected {FINGERPRINT_DIM_FULL}"

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
        [0]     Expression length (normalized by 200)
        [1-7]   Operator counts (normalized by 10):
                  [1] '+' (ADD)
                  [2] '-' (SUB)
                  [3] '*' (MUL)
                  [4] '&' (AND)
                  [5] '|' (OR)
                  [6] '^' (XOR)
                  [7] '~' (NOT)
        [8]     Number of variables (normalized by 8)
        [9]     Number of constants (normalized by 10)
        [10]    Parenthesis depth (normalized)
        [11]    Total depth estimate (normalized)
        [12-19] Variable usage counts for x0-x7 (normalized by 5)
        [20-23] Constant value statistics (mean, std, min, max, normalized)
        [24-31] Reserved/padding
        """
        features = np.zeros(SYMBOLIC_DIM, dtype=np.float64)
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

        Uses explicit float64 precision to match C++ implementation.
        """
        features = np.zeros(CORNER_DIM, dtype=np.float64)
        idx = 0

        for width in BIT_WIDTHS:
            # Keep mask as int for bitwise operations, use float64 for normalization
            mask_int = (1 << width) - 1
            mask_float = np.float64(mask_int)
            corners = get_corner_values(width)

            # Use all corners to generate 64 assignments per width (256 total)
            for corner_set in self._generate_corner_assignments(variables, corners, width):
                result = evaluate_expr(expr, corner_set, width)
                if result is not None:
                    # Normalize to [0, 1] with float64 precision
                    features[idx] = np.float64(result) / mask_float
                else:
                    features[idx] = np.float64(0.0)
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
        Evaluate at deterministic test patterns for hash-like signature.

        64 dims = 4 bit widths × 16 deterministic patterns

        Uses explicit float64 precision to match C++ implementation.
        """
        features = np.zeros(RANDOM_DIM, dtype=np.float64)
        idx = 0

        for width in BIT_WIDTHS:
            # Keep mask as int for bitwise operations, use float64 for normalization
            mask_int = (1 << width) - 1
            mask_float = np.float64(mask_int)
            samples = self.random_inputs[width]

            for sample in samples:
                # Map expression variables to sample values
                var_assignment = {var: sample.get(var, sample['x0']) for var in variables}

                result = evaluate_expr(expr, var_assignment, width)
                if result is not None:
                    # Normalize to [0, 1] with float64 precision
                    features[idx] = np.float64(result) / mask_float
                else:
                    features[idx] = np.float64(0.0)
                idx += 1

        return features

    def _derivatives(self, expr: str, variables: List[str]) -> np.ndarray:
        """
        Approximate partial derivatives via finite differences.

        32 dims = 4 bit widths × 8 derivative approximations

        Uses explicit float64 (double) precision to match C++ implementation.
        """
        features = np.zeros(DERIVATIVE_DIM, dtype=np.float64)
        idx = 0

        epsilon = 1

        for width in BIT_WIDTHS:
            # Keep mask as int for bitwise operations, use float64 for normalization
            mask_int = (1 << width) - 1
            mask_double = np.float64(mask_int)
            # Base point (all variables = midpoint)
            midpoint = 1 << (width - 1)
            base_point = {var: midpoint for var in (variables or ['x0'])}

            # Compute partial derivatives for each variable
            for i in range(8):
                if i < len(variables):
                    var = variables[i]
                    # f(x + ε) - f(x)
                    perturbed = base_point.copy()
                    perturbed[var] = (base_point[var] + epsilon) & mask_int

                    f_base = evaluate_expr(expr, base_point, width)
                    f_perturbed = evaluate_expr(expr, perturbed, width)

                    if f_base is not None and f_perturbed is not None:
                        # Use explicit float64 (double) arithmetic (signed difference)
                        # For width=64, values can exceed signed int64 range, so use Python int
                        # then compute difference and cast to float64
                        f_base_py = int(f_base)
                        f_perturbed_py = int(f_perturbed)
                        diff = np.float64(f_perturbed_py - f_base_py)

                        # Normalize to [-1, 1] range with float64 precision
                        normalized = np.clip(diff / mask_double, -1.0, 1.0).astype(np.float64)
                        features[idx] = normalized
                    else:
                        features[idx] = np.float64(0.0)
                else:
                    features[idx] = np.float64(0.0)

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
        features = np.zeros(TRUTH_TABLE_DIM, dtype=np.float64)

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
