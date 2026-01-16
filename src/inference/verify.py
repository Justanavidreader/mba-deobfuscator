"""
Three-tier verification cascade: Syntax → Execution → Z3.
"""

from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass

from src.constants import (
    EXEC_TEST_SAMPLES, Z3_TIMEOUT_MS, Z3_TOP_K, BIT_WIDTHS
)
from src.utils.z3_interface import verify_equivalence, find_counterexample
from src.utils.expr_eval import evaluate_expr, tokenize_expr
from src.data.tokenizer import MBATokenizer
from src.inference.grammar import GrammarConstraint


@dataclass
class VerificationResult:
    """
    Result of verification process.
    """
    candidate: str                          # Candidate expression
    syntax_valid: bool                      # Passed syntax check
    exec_valid: bool                        # Passed execution test
    z3_verified: bool                       # Passed Z3 proof
    z3_timeout: bool = False                # Z3 timed out (distinguishes from inequivalent)
    counterexample: Optional[Dict[str, int]] = None  # If exec/z3 failed
    elapsed_ms: float = 0.0                 # Verification time


class ThreeTierVerifier:
    """
    Cascaded verification: Syntax (instant) → Execution (1ms) → Z3 (1s).
    """

    def __init__(
        self,
        tokenizer: MBATokenizer,
        exec_samples: int = EXEC_TEST_SAMPLES,
        z3_timeout_ms: int = Z3_TIMEOUT_MS,
        z3_top_k: int = Z3_TOP_K,
        bit_widths: List[int] = None
    ):
        """
        Initialize three-tier verifier.

        Args:
            tokenizer: MBATokenizer for decoding candidates
            exec_samples: Number of random test cases for execution tier
            z3_timeout_ms: Z3 solver timeout in milliseconds
            z3_top_k: Maximum candidates to verify with Z3
            bit_widths: Bit widths to test (default: [8, 16, 32, 64])
        """
        self.tokenizer = tokenizer
        self.grammar = GrammarConstraint(tokenizer)
        self.exec_samples = exec_samples
        self.z3_timeout_ms = z3_timeout_ms
        self.z3_top_k = z3_top_k
        self.bit_widths = bit_widths or BIT_WIDTHS

    def verify_batch(
        self,
        input_expr: str,
        candidates: List[str]
    ) -> List[VerificationResult]:
        """
        Verify batch of candidates through three tiers.

        Args:
            input_expr: Original obfuscated expression
            candidates: List of candidate simplified expressions

        Returns:
            List of VerificationResult, ordered by verification level reached
            (Z3-verified first, then exec-verified, then syntax-valid)

        Algorithm:
            1. Tier 1 (Syntax): Parse all candidates, filter invalid
            2. Tier 2 (Execution): Random test 100 samples at 4 widths
            3. Tier 3 (Z3): Formal verification for top-k exec-passing candidates

        Performance:
            - Syntax: ~10µs per candidate
            - Execution: ~1ms per candidate (100 samples × 4 widths)
            - Z3: ~100-1000ms per candidate (timeout at 1000ms)
        """
        import time
        start_time = time.time()

        results = []

        # Tier 1: Syntax check all candidates
        syntax_valid_candidates = []
        for candidate in candidates:
            is_valid, error_msg = self._syntax_check(candidate)
            if is_valid:
                syntax_valid_candidates.append(candidate)
                results.append(VerificationResult(
                    candidate=candidate,
                    syntax_valid=True,
                    exec_valid=False,
                    z3_verified=False
                ))
            else:
                results.append(VerificationResult(
                    candidate=candidate,
                    syntax_valid=False,
                    exec_valid=False,
                    z3_verified=False
                ))

        # Tier 2: Execution testing for syntax-valid candidates
        exec_valid_candidates = []
        for i, candidate in enumerate(syntax_valid_candidates):
            passes, counterexample = self._execution_test(input_expr, candidate)

            # Find corresponding result
            result_idx = next(j for j, r in enumerate(results) if r.candidate == candidate)
            results[result_idx].exec_valid = passes
            results[result_idx].counterexample = counterexample

            if passes:
                exec_valid_candidates.append(candidate)

        # Tier 3: Z3 verification for top-k exec-valid candidates
        z3_candidates = exec_valid_candidates[:self.z3_top_k]
        for candidate in z3_candidates:
            is_equiv, counterexample, timed_out = self._z3_verify(input_expr, candidate)

            # Find corresponding result
            result_idx = next(j for j, r in enumerate(results) if r.candidate == candidate)
            results[result_idx].z3_verified = is_equiv if is_equiv is not None else False
            results[result_idx].z3_timeout = timed_out
            if counterexample is not None:
                results[result_idx].counterexample = counterexample

        # Add timing
        elapsed = (time.time() - start_time) * 1000
        for result in results:
            result.elapsed_ms = elapsed / len(results)

        # Sort by verification level
        def sort_key(r: VerificationResult) -> Tuple[int, int, int]:
            return (
                int(r.z3_verified),
                int(r.exec_valid),
                int(r.syntax_valid)
            )

        results.sort(key=sort_key, reverse=True)
        return results

    def _syntax_check(self, expr: str) -> Tuple[bool, Optional[str]]:
        """
        Tier 1: Syntax validation.

        Args:
            expr: Expression string

        Returns:
            (is_valid, error_message)

        Checks:
            - Grammar parsing succeeds
            - Balanced parentheses
            - No consecutive operators/variables
        """
        return self.grammar.parse_check(expr)

    def _execution_test(
        self,
        input_expr: str,
        candidate: str
    ) -> Tuple[bool, Optional[Dict[str, int]]]:
        """
        Tier 2: Random execution sampling.

        Args:
            input_expr: Original expression
            candidate: Candidate expression

        Returns:
            (passes_test, counterexample)
            - (True, None) if all samples match
            - (False, counterexample) if mismatch found

        Test Strategy:
            - Extract variables from both expressions
            - Generate exec_samples random inputs
            - Test at all bit_widths (8, 16, 32, 64)
            - Include corner cases: 0, 1, max_val, mid_val
        """
        # Extract variables
        input_vars = self._extract_variables(input_expr)
        candidate_vars = self._extract_variables(candidate)

        # Verify candidate uses same or subset of variables
        if not set(candidate_vars).issubset(set(input_vars)):
            # Candidate has extra variables - automatic fail
            return (False, None)

        variables = sorted(set(input_vars + candidate_vars))

        # Test at each bit width
        for width in self.bit_widths:
            # Generate test inputs
            test_inputs = self._generate_test_inputs(variables, self.exec_samples, width)

            for var_values in test_inputs:
                # CRITICAL FIX: Wrap evaluate_expr in try-except
                try:
                    result_input = evaluate_expr(input_expr, var_values, width)
                except Exception:
                    # Treat evaluation failure as mismatch
                    return (False, None)

                try:
                    result_candidate = evaluate_expr(candidate, var_values, width)
                except Exception:
                    # Treat evaluation failure as mismatch
                    return (False, None)

                if result_input != result_candidate:
                    return (False, var_values)

        return (True, None)

    def _z3_verify(
        self,
        input_expr: str,
        candidate: str
    ) -> Tuple[Optional[bool], Optional[Dict[str, int]], bool]:
        """
        Tier 3: Z3 SMT formal verification.

        Args:
            input_expr: Original expression
            candidate: Candidate expression

        Returns:
            (is_equivalent, counterexample, timed_out)
            - (True, None, False) if formally equivalent
            - (False, counterexample, False) if Z3 finds counterexample
            - (None, None, True) if timeout

        Implementation:
            Uses src.utils.z3_interface.verify_equivalence()
            Tests at 64-bit width with timeout
        """
        # CRITICAL FIX: Distinguish timeout from inequivalence
        try:
            # Check equivalence
            is_equiv = verify_equivalence(input_expr, candidate, 64, self.z3_timeout_ms)

            if is_equiv:
                return (True, None, False)

            # Not equivalent - try to find counterexample
            counterexample = find_counterexample(input_expr, candidate, 64, self.z3_timeout_ms)

            if counterexample is not None:
                # Found concrete counterexample
                return (False, counterexample, False)
            else:
                # No counterexample but not equivalent - likely timeout
                return (None, None, True)

        except Exception:
            # Error occurred - treat as timeout
            return (None, None, True)

    def _extract_variables(self, expr: str) -> List[str]:
        """
        Extract sorted list of variables from expression.

        Args:
            expr: Expression string

        Returns:
            Sorted list of unique variable names

        Example:
            >>> _extract_variables("(x & y) + x")
            ['x', 'y']
        """
        tokens = tokenize_expr(expr)
        variables = set()

        for token in tokens:
            # Check if it's a variable (starts with letter, not an operator)
            if token and token[0].isalpha() and token not in ['+', '-', '*', '&', '|', '^', '~']:
                variables.add(token)

        return sorted(variables)

    def _generate_test_inputs(
        self,
        variables: List[str],
        num_samples: int,
        width: int
    ) -> List[Dict[str, int]]:
        """
        Generate test inputs for execution testing.

        Args:
            variables: List of variable names
            num_samples: Number of random samples
            width: Bit width

        Returns:
            List of variable bindings (dicts)

        Strategy:
            - 50% random values
            - 25% corner cases (0, 1, max, mid, powers of 2)
            - 25% structured patterns (all same, alternating bits, etc.)
        """
        max_val = (1 << width) - 1
        mid_val = 1 << (width - 1)
        inputs = []

        # Corner cases
        corner_values = [0, 1, 2, max_val, max_val - 1, mid_val, mid_val - 1]
        num_corners = max(num_samples // 4, 10)

        for _ in range(num_corners):
            inp = {}
            for var in variables:
                inp[var] = random.choice(corner_values)
            inputs.append(inp)

        # Structured patterns
        num_structured = max(num_samples // 4, 10)

        # All same value
        for val in [0, 1, max_val]:
            inp = {var: val for var in variables}
            inputs.append(inp)

        # Alternating bits
        inp = {var: 0xAAAAAAAA & max_val for var in variables}
        inputs.append(inp)
        inp = {var: 0x55555555 & max_val for var in variables}
        inputs.append(inp)

        # Powers of 2
        for i in range(min(width, 5)):
            val = (1 << i) & max_val
            inp = {var: val for var in variables}
            inputs.append(inp)

        # Random values
        num_random = num_samples - len(inputs)
        for _ in range(max(num_random, num_samples // 2)):
            inp = {}
            for var in variables:
                inp[var] = random.randint(0, max_val)
            inputs.append(inp)

        # Return exactly num_samples
        return inputs[:num_samples]
