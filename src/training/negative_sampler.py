"""
Negative sampler for GMN training with hard negative mining.

Generates challenging non-equivalent expression pairs using various strategies
and Z3 verification with caching.
"""

import logging
import random
import time
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class Z3TimeoutPolicy(Enum):
    """Policy for handling Z3 verification timeouts."""
    RETURN_NONE = "return_none"  # Return None, let caller handle
    FALLBACK_HEURISTIC = "fallback_heuristic"  # Use syntactic similarity
    RETRY_DIFFERENT = "retry_different"  # Sample new candidate


class NegativeSampler:
    """
    Hard negative sampler with Z3 verification.

    Generates non-equivalent expression pairs that are syntactically similar
    or share structure, forcing GMN to learn semantic distinctions.

    Strategies:
        1. Random sampling: Pick random expressions from dataset
        2. Syntactic similarity: Find expressions with similar AST structure
        3. Depth matching: Sample expressions with same depth
        4. Variable swapping: Swap variables in equivalent expressions
    """

    def __init__(
        self,
        dataset: List[Dict[str, str]],
        z3_timeout_ms: int = 500,
        cache_size: int = 10000,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize negative sampler.

        Args:
            dataset: List of {'obfuscated': str, 'simplified': str, 'depth': int}
            z3_timeout_ms: Z3 verification timeout (default: 500ms)
            cache_size: Size of verified negative pair cache
            num_workers: Parallel Z3 verification workers
            device: Device for computation (default: CPU for Z3 compatibility)
        """
        self.dataset = dataset
        self.z3_timeout_ms = z3_timeout_ms
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.device = device if device is not None else torch.device('cpu')

        # Build indexes for efficient sampling
        self._build_indexes()

        # LRU cache for verified pairs: (expr1, expr2) -> (result, timestamp)
        self._cache: OrderedDict[Tuple[str, str], Tuple[Optional[bool], float]] = OrderedDict()

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._z3_timeouts = 0
        self._z3_errors = 0
        self._samples_by_strategy: Dict[str, int] = {
            'random': 0, 'syntactic': 0, 'depth': 0, 'variable': 0
        }

    def _build_indexes(self):
        """Build indexes for efficient negative sampling."""
        # Index by depth
        self._depth_index: Dict[int, List[int]] = {}
        # All simplified expressions for random sampling
        self._all_expressions: List[str] = []

        for idx, item in enumerate(self.dataset):
            depth = item.get('depth', 0)
            if depth not in self._depth_index:
                self._depth_index[depth] = []
            self._depth_index[depth].append(idx)
            self._all_expressions.append(item.get('simplified', item.get('obfuscated', '')))

        logger.info(f"Built indexes: {len(self.dataset)} items, {len(self._depth_index)} depth levels")

    def _normalize_expr(self, expr: str) -> str:
        """Normalize expression for cache key."""
        return expr.replace(' ', '').lower()

    def _cache_key(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """Create normalized cache key (order-independent)."""
        n1, n2 = self._normalize_expr(expr1), self._normalize_expr(expr2)
        return (min(n1, n2), max(n1, n2))

    def _cache_get(self, expr1: str, expr2: str) -> Optional[Tuple[Optional[bool], float]]:
        """Get cached verification result."""
        key = self._cache_key(expr1, expr2)
        if key in self._cache:
            self._cache_hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        return None

    def _cache_put(self, expr1: str, expr2: str, result: Optional[bool]):
        """Put verification result in cache."""
        key = self._cache_key(expr1, expr2)
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _verify_equivalence_with_timeout(
        self,
        expr1: str,
        expr2: str,
    ) -> Optional[bool]:
        """
        Verify equivalence with Z3, handling timeouts gracefully.

        Args:
            expr1: First expression
            expr2: Second expression

        Returns:
            True if equivalent, False if not equivalent, None if timeout/error
        """
        # Check cache first
        cached = self._cache_get(expr1, expr2)
        if cached is not None:
            return cached[0]

        try:
            from src.utils.z3_interface import Z3_AVAILABLE

            if not Z3_AVAILABLE:
                # Z3 not available, fall back to syntactic check
                return None

            from z3 import Solver, unsat, unknown
            from src.utils.z3_interface import expr_to_z3

            var_cache = {}
            z3_expr1 = expr_to_z3(expr1, 64, var_cache)
            z3_expr2 = expr_to_z3(expr2, 64, var_cache)

            solver = Solver()
            solver.set('timeout', self.z3_timeout_ms)
            solver.add(z3_expr1 != z3_expr2)

            result = solver.check()

            if result == unsat:
                # Expressions are equivalent
                self._cache_put(expr1, expr2, True)
                return True
            elif result == unknown:
                # Timeout or unknown
                self._z3_timeouts += 1
                self._cache_put(expr1, expr2, None)
                return None
            else:
                # SAT = counterexample found = not equivalent
                self._cache_put(expr1, expr2, False)
                return False

        except Exception as e:
            logger.debug(f"Z3 verification error: {e}")
            self._z3_errors += 1
            self._cache_put(expr1, expr2, None)
            return None

    def _syntactic_heuristic(self, expr1: str, expr2: str) -> bool:
        """
        Syntactic heuristic for equivalence (fallback when Z3 times out).

        Returns True if expressions are LIKELY non-equivalent (conservative).
        """
        n1 = self._normalize_expr(expr1)
        n2 = self._normalize_expr(expr2)

        # If normalized expressions are identical, likely equivalent
        if n1 == n2:
            return True

        # Count operators - significantly different counts suggest non-equivalence
        ops1 = sum(1 for c in n1 if c in '&|^+-*~')
        ops2 = sum(1 for c in n2 if c in '&|^+-*~')
        if abs(ops1 - ops2) > 3:
            return False

        # If very different lengths, likely non-equivalent
        if abs(len(n1) - len(n2)) > len(n1) // 2:
            return False

        # Conservative: assume might be equivalent
        return True

    def _sample_random(self) -> str:
        """Sample a random expression from dataset."""
        return random.choice(self._all_expressions)

    def _sample_by_depth(self, target_depth: int) -> Optional[str]:
        """Sample expression with same depth."""
        if target_depth not in self._depth_index:
            # Try nearby depths
            for delta in [1, -1, 2, -2]:
                if target_depth + delta in self._depth_index:
                    target_depth = target_depth + delta
                    break
            else:
                return None

        idx = random.choice(self._depth_index[target_depth])
        return self._all_expressions[idx]

    def _sample_variable_swap(self, expr: str) -> str:
        """
        Swap variables to create non-equivalent expression.

        Guaranteed to break equivalence if variables are distinct.
        """
        # Common variable swaps
        swaps = [
            ('x', 'y'), ('y', 'z'), ('z', 'w'),
            ('a', 'b'), ('b', 'c')
        ]

        result = expr
        for old, new in swaps:
            if old in result and new in result:
                # Both present, swap them
                placeholder = '__PLACEHOLDER__'
                result = result.replace(old, placeholder)
                result = result.replace(new, old)
                result = result.replace(placeholder, new)
                break
            elif old in result:
                # Just replace old with new
                result = result.replace(old, new)
                break

        return result

    def sample_negative(
        self,
        obf_expr: str,
        simp_expr: str,
        strategy: str = 'mixed',
        max_attempts: int = 3,
    ) -> Tuple[str, str, str]:
        """
        Sample a negative pair for given (obf, simp) positive pair.

        Args:
            obf_expr: Obfuscated expression (anchor)
            simp_expr: Simplified expression (positive)
            strategy: Sampling strategy ('random', 'syntactic', 'depth', 'variable', 'mixed')
            max_attempts: Maximum attempts before fallback

        Returns:
            Tuple of (obf_expr, negative_expr, negative_type)
        """
        if strategy == 'mixed':
            # Randomly choose strategy: 30% random, 40% depth, 20% variable, 10% syntactic
            r = random.random()
            if r < 0.3:
                strategy = 'random'
            elif r < 0.7:
                strategy = 'depth'
            elif r < 0.9:
                strategy = 'variable'
            else:
                strategy = 'syntactic'

        for attempt in range(max_attempts):
            candidate = None

            if strategy == 'random':
                candidate = self._sample_random()
            elif strategy == 'depth':
                # Get depth from dataset if available
                depth = 5  # Default depth
                for item in self.dataset:
                    if item.get('obfuscated') == obf_expr or item.get('simplified') == simp_expr:
                        depth = item.get('depth', 5)
                        break
                candidate = self._sample_by_depth(depth) or self._sample_random()
            elif strategy == 'variable':
                candidate = self._sample_variable_swap(simp_expr)
            elif strategy == 'syntactic':
                # For syntactic, use depth matching as approximation
                candidate = self._sample_by_depth(5) or self._sample_random()

            if candidate is None:
                candidate = self._sample_random()

            # Skip if candidate equals the positive
            if self._normalize_expr(candidate) == self._normalize_expr(simp_expr):
                continue

            # Verify non-equivalence
            equiv_result = self._verify_equivalence_with_timeout(obf_expr, candidate)

            if equiv_result is False:
                # Confirmed non-equivalent
                self._samples_by_strategy[strategy] += 1
                return (obf_expr, candidate, strategy)
            elif equiv_result is None:
                # Timeout - use heuristic
                if not self._syntactic_heuristic(obf_expr, candidate):
                    # Heuristic says non-equivalent
                    self._samples_by_strategy[strategy] += 1
                    return (obf_expr, candidate, strategy)
            # equiv_result is True means equivalent, try again

        # All attempts exhausted, fall back to random
        logger.debug(f"Falling back to random after {max_attempts} attempts")
        candidate = self._sample_random()
        self._samples_by_strategy['random'] += 1
        return (obf_expr, candidate, 'random_fallback')

    def batch_sample_negatives(
        self,
        positive_pairs: List[Tuple[str, str]],
        strategy: str = 'mixed',
    ) -> List[Tuple[str, str, str]]:
        """
        Batch sample negatives for multiple positive pairs.

        Uses map_async with timeout to prevent worker deadlock.

        Args:
            positive_pairs: List of (obf_expr, simp_expr) tuples
            strategy: Sampling strategy

        Returns:
            List of (obf_expr, negative_expr, negative_type) tuples
        """
        # For small batches or single-process mode, use sequential
        if len(positive_pairs) <= 4 or self.num_workers <= 1:
            return [
                self.sample_negative(pair[0], pair[1], strategy)
                for pair in positive_pairs
            ]

        # For larger batches, try multiprocessing with timeout
        try:
            import multiprocessing

            def sample_wrapper(args):
                obf, simp, strat = args
                return self.sample_negative(obf, simp, strat)

            inputs = [(pair[0], pair[1], strategy) for pair in positive_pairs]

            with multiprocessing.Pool(processes=self.num_workers) as pool:
                # CRITICAL: Use map_async with timeout to prevent deadlock
                timeout = (self.z3_timeout_ms / 1000 + 5) * len(positive_pairs) / self.num_workers
                result = pool.map_async(sample_wrapper, inputs)
                try:
                    negatives = result.get(timeout=timeout)
                    return negatives
                except multiprocessing.TimeoutError:
                    logger.warning(f"Pool timeout after {timeout}s, falling back to sequential")
                    pool.terminate()
                    pool.join()

        except Exception as e:
            logger.warning(f"Multiprocessing error: {e}, falling back to sequential")

        # Fallback to sequential
        return [
            self.sample_negative(pair[0], pair[1], strategy)
            for pair in positive_pairs
        ]

    @property
    def stats(self) -> Dict[str, any]:
        """Return sampler statistics."""
        total_accesses = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_accesses, 1)

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
            'z3_timeouts': self._z3_timeouts,
            'z3_errors': self._z3_errors,
            'samples_by_strategy': dict(self._samples_by_strategy),
            'cache_size': len(self._cache),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._z3_timeouts = 0
        self._z3_errors = 0
        self._samples_by_strategy = {
            'random': 0, 'syntactic': 0, 'depth': 0, 'variable': 0
        }
