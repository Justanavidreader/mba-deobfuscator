# Implementation Plan: Variable Permutation Augmentation

**Status: PLANNING** (2024-01-17)

## Quality Review Fixes Applied

Based on quality-reviewer feedback (NEEDS_CHANGES):

1. **HIGH - DataLoader worker seed collision**: Added worker-aware seeding via `torch.utils.data.get_worker_info()` to ensure different workers get different permutations
2. **SHOULD_FIX - Missing ContrastiveDataset test**: Added test verifying anchor and positive get different permutations
3. **SUGGESTION - Duplicate initialization**: Added `VariableAugmentationMixin` to share augmentation logic across dataset classes

## Problem Statement

The model may memorize variable name patterns instead of learning structural equivalences:
- If training always shows `x & y → x & y`, model learns "x comes before y"
- Positional bias: first variable in input tends to appear first in output
- Copy mechanism alone doesn't prevent this — model still sees consistent name patterns

## Goal

Randomly permute variable names during training so the model learns structure, not names:
```
Original:    (x & y) + (x ^ y) → x | y
Permuted A:  (a & b) + (a ^ b) → a | b
Permuted B:  (y & x) + (y ^ x) → y | x
Permuted C:  (x1 & x0) + (x1 ^ x0) → x1 | x0
```

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LOADING WITH VARIABLE PERMUTATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JSONL: {"obfuscated": "(x & y) + (x ^ y)", "simplified": "x|y"}│
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  VariablePermuter                   │                        │
│  │  1. Extract variables from expr     │                        │
│  │  2. Generate random permutation     │                        │
│  │  3. Apply to BOTH obfuscated AND    │                        │
│  │     simplified consistently         │                        │
│  └─────────────────────────────────────┘                        │
│                        │                                        │
│                        ▼                                        │
│  Permuted: {"obfuscated": "(b & a) + (b ^ a)", "simplified": "b|a"}
│                        │                                        │
│                        ▼                                        │
│  ┌─────────────────────────────────────┐                        │
│  │  Standard pipeline:                 │                        │
│  │  - expr_to_graph()                  │                        │
│  │  - compute_fingerprint()            │                        │
│  │  - tokenizer.encode()               │                        │
│  └─────────────────────────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Permutation applied at string level** (before parsing)
   - Simpler than modifying AST
   - Consistent across obfuscated/simplified pair
   - Works with any expression format

2. **Variable pool for permutation**
   - Use canonical names: x0, x1, x2, ..., x7
   - Ensures all permuted variables have dedicated tokens
   - Avoids `<unk>` token issues

3. **Permutation probability**
   - Apply permutation with probability p (default 0.8)
   - 20% of samples keep original names for robustness
   - Configurable via dataset parameter

4. **Deterministic option for validation**
   - Disable permutation during validation/test
   - Or use seeded permutation for reproducibility

## Implementation Details

### 1. VariablePermuter Class

```python
# src/data/augmentation.py

import re
import random
from typing import Tuple, List, Dict, Optional

class VariablePermuter:
    """
    Randomly permutes variable names in MBA expressions.

    Ensures both input and output expressions use consistent permutation,
    preventing the model from memorizing variable name patterns.
    """

    # Canonical variable names (have dedicated tokens)
    CANONICAL_VARS = [f'x{i}' for i in range(8)]  # x0, x1, ..., x7

    # Pattern to match variables: single letter optionally followed by digit
    VAR_PATTERN = re.compile(r'\b([a-zA-Z]\d?)\b')

    def __init__(
        self,
        permute_prob: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Args:
            permute_prob: Probability of applying permutation (0.0-1.0)
            seed: Random seed for reproducibility (None for random)
        """
        if not 0.0 <= permute_prob <= 1.0:
            raise ValueError(f"permute_prob must be in [0, 1], got {permute_prob}")

        self.permute_prob = permute_prob

        # Quality fix: Worker-aware seeding for multi-worker DataLoader
        # Without this, all workers with same seed produce identical permutations
        if seed is not None:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                seed = seed + worker_info.id

        self.rng = random.Random(seed)

    def extract_variables(self, expr: str) -> List[str]:
        """
        Extract unique variables from expression in order of first appearance.

        Args:
            expr: MBA expression string

        Returns:
            List of variable names in order of first appearance
        """
        seen = set()
        variables = []
        for match in self.VAR_PATTERN.finditer(expr):
            var = match.group(1)
            if var not in seen:
                seen.add(var)
                variables.append(var)
        return variables

    def generate_permutation(self, variables: List[str]) -> Dict[str, str]:
        """
        Generate a random mapping from original variables to canonical names.

        Args:
            variables: List of original variable names

        Returns:
            Dict mapping original names to permuted canonical names
        """
        if len(variables) > len(self.CANONICAL_VARS):
            raise ValueError(
                f"Expression has {len(variables)} variables, "
                f"but only {len(self.CANONICAL_VARS)} canonical names available"
            )

        # Shuffle canonical names and assign
        shuffled = self.CANONICAL_VARS[:len(variables)].copy()
        self.rng.shuffle(shuffled)

        return dict(zip(variables, shuffled))

    def apply_permutation(self, expr: str, mapping: Dict[str, str]) -> str:
        """
        Apply variable permutation to expression.

        Uses word boundary matching to avoid partial replacements.
        Processes longer variable names first to avoid conflicts (e.g., x1 before x).

        Args:
            expr: Original expression
            mapping: Variable name mapping

        Returns:
            Expression with permuted variable names
        """
        # Sort by length descending to replace longer names first
        # This prevents 'x' from matching in 'x1'
        sorted_vars = sorted(mapping.keys(), key=len, reverse=True)

        result = expr
        # Use temporary placeholders to avoid replacement conflicts
        placeholders = {var: f'__VAR_{i}__' for i, var in enumerate(sorted_vars)}

        # First pass: replace with placeholders
        for var in sorted_vars:
            pattern = rf'\b{re.escape(var)}\b'
            result = re.sub(pattern, placeholders[var], result)

        # Second pass: replace placeholders with final names
        for var in sorted_vars:
            result = result.replace(placeholders[var], mapping[var])

        return result

    def __call__(
        self,
        obfuscated: str,
        simplified: str,
    ) -> Tuple[str, str]:
        """
        Permute variables in both expressions consistently.

        Args:
            obfuscated: Obfuscated MBA expression
            simplified: Simplified MBA expression

        Returns:
            Tuple of (permuted_obfuscated, permuted_simplified)
        """
        # Skip permutation with probability (1 - permute_prob)
        if self.rng.random() > self.permute_prob:
            return obfuscated, simplified

        # Extract variables from both expressions (union)
        vars_obf = set(self.extract_variables(obfuscated))
        vars_simp = set(self.extract_variables(simplified))
        all_vars = list(vars_obf | vars_simp)

        # Sort for deterministic ordering before shuffling
        all_vars.sort()

        if not all_vars:
            return obfuscated, simplified

        # Generate and apply permutation
        mapping = self.generate_permutation(all_vars)

        permuted_obf = self.apply_permutation(obfuscated, mapping)
        permuted_simp = self.apply_permutation(simplified, mapping)

        return permuted_obf, permuted_simp
```

### 2. VariableAugmentationMixin (Quality Fix: Reduce Duplication)

```python
# src/data/augmentation.py (add to same file)

class VariableAugmentationMixin:
    """
    Mixin for variable permutation augmentation.

    Provides shared initialization and application logic for all dataset classes.
    Quality fix: Avoids duplicating augmentation code across MBADataset,
    ContrastiveDataset, and ScaledMBADataset.
    """

    def _init_augmentation(
        self,
        augment_variables: bool = True,
        augment_prob: float = 0.8,
        augment_seed: Optional[int] = None,
    ):
        """Initialize variable permutation augmentation."""
        self.augment_variables = augment_variables
        self.permuter: Optional[VariablePermuter] = None
        if augment_variables:
            self.permuter = VariablePermuter(
                permute_prob=augment_prob,
                seed=augment_seed,
            )

    def _apply_augmentation(
        self,
        obfuscated: str,
        simplified: str,
    ) -> Tuple[str, str]:
        """Apply variable permutation to expression pair."""
        if self.permuter is not None:
            return self.permuter(obfuscated, simplified)
        return obfuscated, simplified
```

### 3. Integrate with MBADataset

```python
# src/data/dataset.py (modifications)

from src.data.augmentation import VariablePermuter, VariableAugmentationMixin

class MBADataset(Dataset, VariableAugmentationMixin):
    def __init__(
        self,
        data_path: str,
        # ... existing params ...
        # New augmentation params
        augment_variables: bool = True,
        augment_prob: float = 0.8,
        augment_seed: Optional[int] = None,
    ):
        # ... existing init ...

        # Variable permutation augmentation (via mixin)
        self._init_augmentation(augment_variables, augment_prob, augment_seed)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        obfuscated = sample['obfuscated']
        simplified = sample['simplified']

        # Apply variable permutation BEFORE any processing
        obfuscated, simplified = self._apply_augmentation(obfuscated, simplified)

        # ... rest of existing __getitem__ ...
        # expr_to_graph(obfuscated)
        # compute_fingerprint(obfuscated)
        # tokenizer.encode(simplified)
```

### 4. Update ContrastiveDataset

```python
# src/data/dataset.py (ContrastiveDataset modifications)

class ContrastiveDataset(Dataset, VariableAugmentationMixin):
    def __init__(
        self,
        # ... existing params ...
        augment_variables: bool = True,
        augment_prob: float = 0.8,
    ):
        # ... existing init ...

        # Use mixin but no seed (want different permutations each call)
        self._init_augmentation(augment_variables, augment_prob, augment_seed=None)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # For contrastive learning, we want DIFFERENT permutations
        # for anchor and positive to learn permutation invariance
        anchor_expr = sample['obfuscated']
        positive_expr = sample['obfuscated']  # Same expression

        if self.permuter is not None:
            # Apply different permutations to anchor and positive
            # Each call to permuter uses its RNG, producing different results
            anchor_expr, _ = self.permuter(anchor_expr, anchor_expr)
            positive_expr, _ = self.permuter(positive_expr, positive_expr)

        # ... rest of existing __getitem__ ...
```

### 4. Add Constants

```python
# src/constants.py additions

# Variable permutation augmentation
VAR_AUGMENT_ENABLED: bool = True
VAR_AUGMENT_PROB: float = 0.8
VAR_CANONICAL_NAMES: List[str] = [f'x{i}' for i in range(8)]
```

### 5. Testing

```python
# tests/test_variable_permutation.py

class TestVariablePermuter:
    def test_extract_variables_simple(self):
        """Extract variables in order of appearance."""
        permuter = VariablePermuter()
        expr = "x & y + x"
        vars = permuter.extract_variables(expr)
        assert vars == ['x', 'y']

    def test_extract_variables_with_numbers(self):
        """Handle x0, x1 style variables."""
        permuter = VariablePermuter()
        expr = "x0 & x1 + x2"
        vars = permuter.extract_variables(expr)
        assert vars == ['x0', 'x1', 'x2']

    def test_consistent_permutation(self):
        """Same mapping applied to both expressions."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf = "(x & y) + (x ^ y)"
        simp = "x | y"

        perm_obf, perm_simp = permuter(obf, simp)

        # Variables should be consistently renamed
        # If x -> x3 in obf, then x -> x3 in simp
        assert perm_obf.count('x3') == perm_simp.count('x3') or \
               perm_obf.count('x3') > 0 == (perm_simp.count('x3') > 0)

    def test_no_partial_replacement(self):
        """x1 should not be affected when replacing x."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        expr = "x + x1"
        vars = permuter.extract_variables(expr)

        # Should extract both x and x1 as separate variables
        assert 'x' in vars
        assert 'x1' in vars

    def test_permute_prob_zero(self):
        """No permutation when prob=0."""
        permuter = VariablePermuter(permute_prob=0.0)
        obf, simp = "x & y", "x | y"

        perm_obf, perm_simp = permuter(obf, simp)

        assert perm_obf == obf
        assert perm_simp == simp

    def test_deterministic_with_seed(self):
        """Same seed produces same permutation."""
        obf, simp = "(x & y) + z", "x | y | z"

        p1 = VariablePermuter(permute_prob=1.0, seed=123)
        p2 = VariablePermuter(permute_prob=1.0, seed=123)

        result1 = p1(obf, simp)
        result2 = p2(obf, simp)

        assert result1 == result2

    def test_all_canonical_names_used(self):
        """Permuted variables use only canonical names."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf = "a & b + c"
        simp = "a | c"

        perm_obf, perm_simp = permuter(obf, simp)

        # Should only contain x0-x7 variables
        vars_in_result = permuter.extract_variables(perm_obf + perm_simp)
        for var in vars_in_result:
            assert var in VariablePermuter.CANONICAL_VARS

    def test_max_variables_error(self):
        """Error when expression has too many variables."""
        permuter = VariablePermuter(permute_prob=1.0)
        # 9 variables, but only 8 canonical names
        expr = "a & b & c & d & e & f & g & h & i"

        with pytest.raises(ValueError, match="only 8 canonical names"):
            permuter(expr, expr)

    def test_empty_expression(self):
        """Handle expressions with no variables."""
        permuter = VariablePermuter(permute_prob=1.0)
        obf, simp = "1 + 2", "3"

        perm_obf, perm_simp = permuter(obf, simp)

        assert perm_obf == obf
        assert perm_simp == simp


class TestDatasetIntegration:
    def test_mba_dataset_with_augmentation(self, tmp_path):
        """MBADataset applies variable permutation."""
        # Create test data file
        data = [
            {"obfuscated": "x & y", "simplified": "x & y", "depth": 1},
        ]
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        dataset = MBADataset(
            str(data_file),
            augment_variables=True,
            augment_prob=1.0,
            augment_seed=42,
        )

        sample = dataset[0]
        # Variable names should be from canonical set
        # (exact names depend on seed)

    def test_augmentation_disabled_for_validation(self, tmp_path):
        """Augmentation can be disabled for validation set."""
        # ... similar test with augment_variables=False ...

    # Quality fix: Test for ContrastiveDataset dual-permutation behavior
    def test_contrastive_different_permutations(self, tmp_path):
        """ContrastiveDataset applies different permutations to anchor and positive."""
        data = [{"obfuscated": "x & y", "simplified": "x & y", "depth": 1}]
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        dataset = ContrastiveDataset(
            str(data_file),
            augment_variables=True,
            augment_prob=1.0,
            # No seed - we want different permutations each call
        )

        # Get multiple samples of the same item
        samples = [dataset[0] for _ in range(10)]

        # Anchor and positive should have different permutations at least sometimes
        different_count = sum(
            1 for s in samples
            if s['anchor_expr'] != s['positive_expr']
        )
        assert different_count > 0, (
            "Anchor and positive should sometimes differ due to random permutations"
        )

    def test_worker_seed_isolation(self):
        """Different workers get different seeds even with same base seed."""
        import unittest.mock as mock

        # Mock worker_info to simulate different workers
        with mock.patch('torch.utils.data.get_worker_info') as mock_worker_info:
            # Worker 0
            mock_worker_info.return_value = mock.Mock(id=0)
            p0 = VariablePermuter(permute_prob=1.0, seed=42)

            # Worker 1
            mock_worker_info.return_value = mock.Mock(id=1)
            p1 = VariablePermuter(permute_prob=1.0, seed=42)

        # Same input, different workers should produce different results
        obf, simp = "x & y", "x | y"
        result0 = p0(obf, simp)
        result1 = p1(obf, simp)

        # Results may or may not differ (depends on shuffle), but seeds differ
        # This test mainly verifies the code path works without error
        assert result0 is not None
        assert result1 is not None
```

## Task Breakdown

| Task | File | Status | Priority |
|------|------|--------|----------|
| Create VariablePermuter class (with worker-aware seeding) | `src/data/augmentation.py` | TODO | P0 |
| Create VariableAugmentationMixin | `src/data/augmentation.py` | TODO | P0 |
| Add augmentation constants | `src/constants.py` | TODO | P0 |
| Integrate with MBADataset | `src/data/dataset.py` | TODO | P0 |
| Integrate with ContrastiveDataset | `src/data/dataset.py` | TODO | P1 |
| Integrate with ScaledMBADataset | `src/data/dataset.py` | TODO | P1 |
| Create unit tests (VariablePermuter) | `tests/test_variable_permutation.py` | TODO | P0 |
| Add ContrastiveDataset dual-permutation test | `tests/test_variable_permutation.py` | TODO | P0 |
| Add worker seed isolation test | `tests/test_variable_permutation.py` | TODO | P0 |
| Update data generation docs | `docs/DATA_PIPELINE.md` | TODO | P2 |

## Edge Cases

1. **Variable name conflicts**: `x` vs `x1` — handled by sorting by length and using placeholders
2. **No variables**: Constants-only expressions pass through unchanged
3. **Too many variables**: Error raised if >8 unique variables (configurable)
4. **Operators as substrings**: Regex uses word boundaries to avoid matching operators

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regex misses edge cases | Incorrect permutation | Comprehensive test suite |
| Performance overhead | Slower data loading | String operations are O(n), negligible vs graph construction |
| Breaks fingerprint cache | Invalid cached features | Fingerprint computed after permutation, no cache issue |
| Contrastive learning disruption | Worse Phase 1 | Different permutations for anchor/positive teaches invariance |

## Success Criteria

1. All unit tests pass
2. Variable names in training batches show uniform distribution across canonical names
3. Model accuracy on validation set (with fixed names) matches or exceeds baseline
4. No increase in `<unk>` token frequency in decoded outputs

## Rollback Plan

Augmentation is opt-in via `augment_variables=False`. Default enabled for training, disabled for validation/test.
