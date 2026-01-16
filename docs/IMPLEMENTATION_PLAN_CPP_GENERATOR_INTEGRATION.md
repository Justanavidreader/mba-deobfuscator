# Implementation Plan: C++ Generator Integration

**Status**: Draft
**Priority**: P0 (blocking training data pipeline)
**Estimated LOC**: ~200

## Overview

Unify dataset loading to support both legacy Python-only format and C++ generator v2 format. Training uses pre-computed fingerprints from C++; inference computes fingerprints in Python.

**CRITICAL INVARIANT**: Pre-computed fingerprints can ONLY be used when variable augmentation is DISABLED. Variable permutation changes expression semantics (variable names change), invalidating pre-computed fingerprints computed on the original expression. All pre-computed loading must check `not self.augment_variables` before using cached data.

## Current State

**ScaledMBADataset** (lines 509-761 in `dataset.py`):
- ✅ Field name mapping: `obfuscated_expr`→`obfuscated`, `ground_truth_expr`→`simplified`
- ✅ Pre-computed `fingerprint.flat` loading
- ⚠️ Uses `ast` field, C++ generator outputs `ast_v2`

**MBADataset** (lines 30-240):
- ❌ Requires `obfuscated`/`simplified` fields exactly
- ❌ Always computes fingerprints via Python

**ContrastiveDataset** (lines 243-422):
- ❌ Same limitations as MBADataset

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dataset Classes                              │
├─────────────────────────────────────────────────────────────────┤
│  _load_data():                                                   │
│    - Accept both field name conventions                         │
│    - Normalize to internal format                               │
│                                                                  │
│  __getitem__():                                                  │
│    can_use_precomputed = (                                       │
│        'fingerprint.flat' in item                               │
│        AND not self.augment_variables  # CRITICAL CHECK         │
│    )                                                             │
│                                                                  │
│    if can_use_precomputed:                                       │
│        fp = load_precomputed()    # Training (C++ data, no aug) │
│    else:                                                         │
│        fp = self.fingerprint.compute()  # Augmentation/inference│
└─────────────────────────────────────────────────────────────────┘
```

**Why the augmentation check matters**: Variable permutation (e.g., `x → x0`, `y → x1`) changes the expression semantics from the fingerprint's perspective. Corner evaluations use specific variable values. If the fingerprint was computed for `x & y` but the dataset returns a graph built from `x0 & x1`, the fingerprint's corner evaluations (which tested x=0, y=255, etc.) no longer correspond to the actual variables in the graph. This corrupts the training signal.

## Section 1: Field Name Normalization

**File**: `src/data/dataset.py`

### 1.1 Extract Common Loader Function

Create `_normalize_item()` at module level to handle both field name conventions:

```python
def _normalize_item(item: Dict) -> Dict:
    """
    Normalize field names from C++ generator format to internal format.

    C++ format: obfuscated_expr, ground_truth_expr, ast_v2
    Internal:   obfuscated, simplified, ast
    """
    normalized = item.copy()

    # Expression fields - use falsy check to handle empty strings
    # Matches ScaledMBADataset's existing logic: `item.get('obfuscated') or item.get('obfuscated_expr')`
    if not normalized.get('obfuscated'):
        normalized['obfuscated'] = normalized.get('obfuscated_expr')
    if not normalized.get('simplified'):
        normalized['simplified'] = normalized.get('ground_truth_expr')

    # AST field (C++ uses ast_v2)
    if 'ast' not in normalized and 'ast_v2' in normalized:
        normalized['ast'] = normalized['ast_v2']

    return normalized
```

### 1.2 Update MBADataset._load_data()

**Current** (lines 89-119):
```python
# Validate required fields
if 'obfuscated' not in item or 'simplified' not in item:
    continue
```

**Updated**:
```python
# Normalize field names (supports C++ generator format)
item = _normalize_item(item)

# Validate required fields
if not item.get('obfuscated') or not item.get('simplified'):
    continue
```

### 1.3 Update ContrastiveDataset._load_data()

Same change as 1.2, at lines 306-336.

### 1.4 Consolidate ScaledMBADataset._load_data()

Remove inline normalization (lines 509-518) and use `_normalize_item()`.

## Section 2: Pre-computed Fingerprint Loading

### 2.0 Extract Validation Helper

Create module-level validation function to avoid duplicating logic across dataset classes:

```python
import numpy as np

def _validate_precomputed_fingerprint(fp: List[float]) -> np.ndarray:
    """
    Validate pre-computed fingerprint dimension and numeric values.

    Args:
        fp: Pre-computed fingerprint from C++ generator

    Returns:
        Validated fingerprint as numpy array

    Raises:
        ValueError: If dimension mismatch or NaN/inf detected
    """
    if len(fp) != FINGERPRINT_DIM:
        raise ValueError(
            f"Pre-computed fingerprint has {len(fp)} dims, expected {FINGERPRINT_DIM}"
        )

    fp_array = np.array(fp, dtype=np.float32)
    if not np.all(np.isfinite(fp_array)):
        raise ValueError(
            f"Pre-computed fingerprint contains NaN or inf values"
        )

    return fp_array
```

### 2.1 Update MBADataset.__getitem__()

**Current** (lines 220-222):
```python
# Compute semantic fingerprint
fp = self.fingerprint.compute(obfuscated)
fp_tensor = torch.from_numpy(fp)
```

**Updated**:
```python
# Load pre-computed fingerprint if available AND augmentation disabled
# CRITICAL: Pre-computed fingerprints were computed on original expressions.
# Variable augmentation changes expression semantics, invalidating the fingerprint.
# Pattern matches ScaledMBADataset's AST protection (line 710).
can_use_precomputed = (
    'fingerprint' in item
    and 'flat' in item['fingerprint']
    and not self.augment_variables  # Augmentation invalidates pre-computed
)

if can_use_precomputed:
    # Validate dimension and numeric values (NaN/inf)
    fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
    fp_tensor = torch.from_numpy(fp_array)
else:
    # Compute fingerprint for (potentially augmented) expression
    fp = self.fingerprint.compute(obfuscated)
    fp_tensor = torch.from_numpy(fp).float()
```

Note: Item dict from `self.data` must be accessed in `__getitem__`, not just `obfuscated`/`simplified` strings. Current code extracts strings early; need to preserve dict reference.

### 2.2 Preserve Item Dict in MBADataset.__getitem__()

**Current** (lines 188-192):
```python
item = self.data[idx]
obfuscated = item['obfuscated']
simplified = item['simplified']
depth = item.get('depth', 0)
```

No change needed—`item` is already available. Just use it for fingerprint loading.

### 2.3 Update ContrastiveDataset.__getitem__()

**Current** (lines 410-412):
```python
# Compute fingerprints for both
obf_fp = torch.from_numpy(self.fingerprint.compute(obfuscated))
simp_fp = torch.from_numpy(self.fingerprint.compute(simplified))
```

**Updated**:
```python
# Load pre-computed fingerprint if available AND augmentation disabled
# CRITICAL: Pre-computed fingerprints were computed on original expressions.
# ContrastiveDataset applies DIFFERENT permutations to anchor/positive (lines 392-394),
# so pre-computed fingerprints are ONLY valid when augmentation is disabled.
can_use_precomputed = (
    'fingerprint' in item
    and 'flat' in item['fingerprint']
    and not self.augment_variables  # Augmentation invalidates pre-computed
)

if can_use_precomputed:
    # Validate dimension and numeric values (NaN/inf)
    fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
    obf_fp = torch.from_numpy(fp_array)
else:
    # Compute fingerprint for (potentially augmented) expression
    obf_fp = torch.from_numpy(self.fingerprint.compute(obfuscated)).float()

# Simplified always computed (not pre-stored in C++ format)
simp_fp = torch.from_numpy(self.fingerprint.compute(simplified)).float()
```

**CRITICAL**: C++ generator computes fingerprint for `obfuscated_expr` only. Pre-computed fingerprints can ONLY be used when augmentation is disabled. When augmentation is enabled, fingerprints must be computed AFTER permutation to match the returned expression.

### 2.4 Fix ScaledMBADataset Fingerprint Protection

**Current** (lines 754-761):
```python
# Use pre-flattened fingerprint if available
if 'fingerprint' in item and 'flat' in item['fingerprint']:
    fp = item['fingerprint']['flat']
    fp_tensor = torch.tensor(fp, dtype=torch.float32)
```

**Issue**: ScaledMBADataset already protects pre-computed AST from augmentation (line 710: `if 'ast' in item and not self.augment_variables`) but does NOT protect fingerprints. When augmentation is enabled (line 701), fingerprint represents unpermuted expression but graph/returned expression is permuted.

**Updated**:
```python
# Load pre-computed fingerprint if available AND augmentation disabled
# CRITICAL: Pattern matches AST protection at line 710.
can_use_precomputed = (
    'fingerprint' in item
    and 'flat' in item['fingerprint']
    and not self.augment_variables  # Augmentation invalidates pre-computed
)

if can_use_precomputed:
    # Validate dimension and numeric values (NaN/inf)
    fp_array = _validate_precomputed_fingerprint(item['fingerprint']['flat'])
    fp_tensor = torch.from_numpy(fp_array)
elif self.fingerprint is not None:
    fp = self.fingerprint.compute(obfuscated)
    fp_tensor = torch.from_numpy(fp).float()
else:
    raise ValueError("No fingerprint available and no fingerprint computer provided")
```

## Section 3: AST Loading from C++ Generator

### 3.1 Update ScaledMBADataset._build_optimized_graph()

The C++ generator outputs `ast_v2` with different node/edge type names. Add compatibility:

**C++ generator node types** (from validate_v2_format.py):
```python
V2_NODE_TYPES = ["ADD", "SUB", "MUL", "NEG", "AND", "OR", "XOR", "NOT", "VAR", "CONST"]
```

**C++ generator edge types**:
```python
V2_EDGE_TYPES = [
    "LEFT_OPERAND", "RIGHT_OPERAND", "UNARY_OPERAND",
    "PARENT_OF_LEFT", "PARENT_OF_RIGHT", "PARENT_OF_UNARY",
    "DOMAIN_BRIDGE"
]
```

The node type IDs already match the current schema (0-9 mapping).

Edge types need mapping in `_build_optimized_graph()`:
```python
# C++ v2 edge type indices
CPP_LEFT_OPERAND = 0
CPP_RIGHT_OPERAND = 1
CPP_UNARY_OPERAND = 2
# Parent edges (inverse) already included
CPP_PARENT_OF_LEFT = 3
CPP_PARENT_OF_RIGHT = 4
CPP_PARENT_OF_UNARY = 5
CPP_DOMAIN_BRIDGE = 6
```

Since C++ already outputs inverse edges, skip the inverse edge generation when loading from `ast_v2`.

### 3.2 Add AST v2 Detection

```python
def _is_v2_ast(self, ast_data: Dict) -> bool:
    """Check if AST uses C++ generator v2 format.

    C++ generator MUST set explicit 'version': 2 field.
    Do NOT use uses_subexpr_sharing check - that field indicates whether
    sharing was enabled, not whether the AST uses v2 format.
    """
    return ast_data.get('version') == 2
```

**NOTE**: C++ generator must include `"version": 2` in AST output. This explicit version field is the canonical way to detect format.

### 3.3 Update _build_optimized_graph() for V2

When `_is_v2_ast()` returns True, load edges directly without regenerating inverses. Track invalid edge references to catch C++ generator bugs:

```python
if self._is_v2_ast(ast_data):
    # V2: Edges already include inverses, load directly
    # Track invalid references to catch data corruption
    total_edges = len(edges)
    skipped_edges = 0

    for edge in edges:
        src = node_mapping.get(edge['src'])
        dst = node_mapping.get(edge['dst'])
        if src is None or dst is None:
            skipped_edges += 1
            continue
        edge_type = edge.get('type', 0)
        new_edges.append((src, dst, edge_type))

    # Validate edge integrity - catch C++ generator bugs
    if total_edges > 0:
        skip_rate = skipped_edges / total_edges
        if skip_rate > 0.1:  # More than 10% invalid
            raise ValueError(
                f"AST v2 has {skip_rate*100:.1f}% invalid edge references "
                f"({skipped_edges}/{total_edges}). Data corruption detected."
            )
        elif skipped_edges > 0:
            import warnings
            warnings.warn(
                f"AST v2 skipped {skipped_edges}/{total_edges} edges with invalid references"
            )
else:
    # Legacy: Generate inverse edges
    # ... existing code ...
```

## Section 4: Fingerprint Consistency Validation

### 4.1 Create Validation Script

**File**: `scripts/validate_fingerprint_consistency.py`

```python
#!/usr/bin/env python3
"""
Validate that Python fingerprint computation matches C++ generator output.

Usage:
    python scripts/validate_fingerprint_consistency.py dataset.json --samples 100
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.data.fingerprint import SemanticFingerprint
from src.constants import (
    FINGERPRINT_DIM, SYMBOLIC_DIM, CORNER_DIM,
    RANDOM_DIM, DERIVATIVE_DIM, TRUTH_TABLE_DIM
)


def get_fingerprint_components() -> List[Tuple[str, int, int]]:
    """
    Compute fingerprint component boundaries from constants.

    Returns:
        List of (name, start_idx, end_idx) tuples
    """
    sym_end = SYMBOLIC_DIM
    corner_end = sym_end + CORNER_DIM
    random_end = corner_end + RANDOM_DIM
    deriv_end = random_end + DERIVATIVE_DIM
    truth_end = deriv_end + TRUTH_TABLE_DIM

    assert truth_end == FINGERPRINT_DIM, f"Component dims don't sum to {FINGERPRINT_DIM}"

    return [
        ('symbolic', 0, sym_end),
        ('corner', sym_end, corner_end),
        ('random', corner_end, random_end),
        ('derivative', random_end, deriv_end),
        ('truth_table', deriv_end, truth_end),
    ]


def validate_sample(
    expr: str,
    cpp_fp: np.ndarray,
    fingerprint: SemanticFingerprint,
    tolerance: float,
    components: List[Tuple[str, int, int]]
) -> Optional[Dict]:
    """
    Validate a single fingerprint sample.

    Args:
        expr: Expression to compute fingerprint for
        cpp_fp: Pre-computed fingerprint from C++ generator
        fingerprint: Python fingerprint computer
        tolerance: Numerical tolerance for comparison
        components: List of (name, start, end) component boundaries

    Returns:
        None if match, or dict with mismatch details if different
    """
    try:
        py_fp = fingerprint.compute(expr)
    except Exception as e:
        return {'error': str(e)}

    if not np.allclose(cpp_fp, py_fp, atol=tolerance):
        diff = np.abs(cpp_fp - py_fp)
        component_diffs = {}
        for name, start, end in components:
            comp_diff = diff[start:end].max()
            if comp_diff > tolerance:
                component_diffs[name] = comp_diff

        return {
            'max_diff': diff.max(),
            'components': component_diffs
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Validate Python fingerprint consistency with C++ generator'
    )
    parser.add_argument('filepath', type=Path, help='JSONL file to validate')
    parser.add_argument('--samples', '-n', type=int, default=100, help='Max samples')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Numerical tolerance')
    args = parser.parse_args()

    fingerprint = SemanticFingerprint()
    components = get_fingerprint_components()

    mismatches = 0
    total = 0
    skipped = 0

    with open(args.filepath) as f:
        for i, line in enumerate(f):
            if i >= args.samples:
                break

            # Parse JSON
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Sample {i}: JSON parse error: {e}")
                skipped += 1
                continue

            # Check required fields
            if 'fingerprint' not in item or 'flat' not in item['fingerprint']:
                skipped += 1
                continue

            expr = item.get('obfuscated_expr') or item.get('obfuscated')
            if not expr:
                print(f"Sample {i}: missing expression")
                skipped += 1
                continue

            cpp_fp = np.array(item['fingerprint']['flat'], dtype=np.float32)

            # Validate dimension
            if len(cpp_fp) != FINGERPRINT_DIM:
                print(f"Sample {i}: wrong dimension {len(cpp_fp)}, expected {FINGERPRINT_DIM}")
                skipped += 1
                continue

            # Validate sample
            result = validate_sample(expr, cpp_fp, fingerprint, args.tolerance, components)

            if result:
                if 'error' in result:
                    print(f"Sample {i}: Python compute error: {result['error']}")
                    skipped += 1
                else:
                    mismatches += 1
                    print(f"Sample {i}: max diff = {result['max_diff']:.6f}")
                    print(f"  Expr: {expr[:60]}...")
                    for name, diff in result['components'].items():
                        print(f"    {name}: max diff = {diff:.6f}")
            else:
                total += 1

    print(f"\nResults: {mismatches}/{total} mismatches, {skipped} skipped")
    return 1 if mismatches > 0 else 0


if __name__ == '__main__':
    exit(main())
```

### 4.2 Expected Differences

The Python and C++ implementations may differ in:

1. **Random hash component**: Uses RNG with seed. Must use same seed.
2. **Floating point precision**: Python uses float64 internally, C++ may use float32.
3. **Truth table variable ordering**: Python uses first-occurrence indexing (fixed in truth table plan). C++ must match.

If mismatches occur, document them and decide:
- If random_hash only: Ensure seed is passed through JSON
- If precision only: Accept within tolerance
- If truth_table: Ensure both use same variable indexing

## Section 5: Add Imports

Import constants and numpy for validation:

```python
import numpy as np
from src.constants import FINGERPRINT_DIM
```

## Section 6: Tests

### 6.1 Test Field Name Normalization

**File**: `tests/test_dataset_cpp_compat.py`

```python
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import torch

from src.data.dataset import MBADataset, _normalize_item


class TestFieldNormalization:
    """Test C++ generator field name support."""

    def test_normalize_item_cpp_format(self):
        """C++ field names normalized to internal format."""
        item = {
            'obfuscated_expr': 'x & y',
            'ground_truth_expr': 'x & y',
            'ast_v2': {'version': 2, 'nodes': [], 'edges': []},
            'fingerprint': {'flat': [0.0] * 448},
        }

        normalized = _normalize_item(item)

        assert normalized['obfuscated'] == 'x & y'
        assert normalized['simplified'] == 'x & y'
        assert normalized['ast'] == item['ast_v2']

    def test_normalize_item_legacy_format(self):
        """Legacy field names preserved."""
        item = {
            'obfuscated': 'x | y',
            'simplified': 'x | y',
            'depth': 1,
        }

        normalized = _normalize_item(item)

        assert normalized['obfuscated'] == 'x | y'
        assert normalized['simplified'] == 'x | y'

    def test_normalize_item_mixed_format(self):
        """Mixed format uses first available."""
        item = {
            'obfuscated': 'a',  # Internal name present
            'obfuscated_expr': 'b',  # C++ name also present
            'simplified': 'a',
        }

        normalized = _normalize_item(item)

        # Internal name takes precedence
        assert normalized['obfuscated'] == 'a'

    def test_normalize_item_empty_string_fallback(self):
        """Empty string falls back to C++ field name."""
        item = {
            'obfuscated': '',  # Empty string
            'obfuscated_expr': 'x & y',  # C++ name has value
            'simplified': 'x & y',
        }

        normalized = _normalize_item(item)

        # Should use C++ value when internal is empty
        assert normalized['obfuscated'] == 'x & y'


class TestPrecomputedFingerprint:
    """Test pre-computed fingerprint loading."""

    @pytest.fixture
    def cpp_data_file(self, tmp_path):
        """Create test data in C++ generator format."""
        data = [
            {
                'obfuscated_expr': 'x & y',
                'ground_truth_expr': 'x & y',
                'depth': 1,
                'fingerprint': {'flat': [0.5] * 448},
            },
        ]
        path = tmp_path / 'cpp_data.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(path)

    @pytest.fixture
    def mock_tokenizer(self):
        mock = MagicMock()
        mock.encode.return_value = [1, 2, 3]
        mock.get_source_tokens.return_value = [4, 5]
        return mock

    @pytest.fixture
    def mock_fingerprint(self):
        import numpy as np
        mock = MagicMock()
        # Should NOT be called when pre-computed available
        mock.compute.return_value = np.zeros(448, dtype=np.float32)
        return mock

    def test_load_precomputed_fingerprint(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """Pre-computed fingerprint loaded instead of computed."""
        dataset = MBADataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        sample = dataset[0]

        # Fingerprint should be 0.5 (from JSON), not 0.0 (from mock)
        assert sample['fingerprint'][0].item() == pytest.approx(0.5)

        # compute() should NOT have been called
        mock_fingerprint.compute.assert_not_called()

    def test_fallback_to_computed(self, tmp_path, mock_tokenizer, mock_fingerprint):
        """Fall back to computed fingerprint when not in JSON."""
        import numpy as np

        # Data without pre-computed fingerprint
        data = [{'obfuscated': 'x', 'simplified': 'x', 'depth': 1}]
        path = tmp_path / 'legacy.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        # Mock returns ones to distinguish from pre-computed
        mock_fingerprint.compute.return_value = np.ones(448, dtype=np.float32)

        dataset = MBADataset(
            str(path),
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        sample = dataset[0]

        # Should be 1.0 from computed
        assert sample['fingerprint'][0].item() == pytest.approx(1.0)
        mock_fingerprint.compute.assert_called_once()

    def test_augmentation_blocks_precomputed(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """Variable augmentation forces fingerprint recomputation."""
        import numpy as np

        # Mock returns different value than pre-computed (0.5 in JSON, 0.9 from mock)
        mock_fingerprint.compute.return_value = np.full(448, 0.9, dtype=np.float32)

        dataset = MBADataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=True,  # Augmentation enabled
            augment_prob=0.0,  # But don't actually permute (for determinism)
        )

        sample = dataset[0]

        # Even with pre-computed available, augmentation forces compute
        # Should be 0.9 (from mock), not 0.5 (from JSON)
        assert sample['fingerprint'][0].item() == pytest.approx(0.9)
        mock_fingerprint.compute.assert_called_once()
```

### 6.2 Test Invalid Fingerprint Dimension

```python
def test_invalid_fingerprint_dimension_raises(self, tmp_path, mock_tokenizer, mock_fingerprint):
    """Wrong fingerprint dimension raises ValueError."""
    data = [{
        'obfuscated_expr': 'x',
        'ground_truth_expr': 'x',
        'fingerprint': {'flat': [0.0] * 100},  # Wrong dimension
    }]
    path = tmp_path / 'bad_fp.jsonl'
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    dataset = MBADataset(
        str(path),
        mock_tokenizer,
        mock_fingerprint,
        node_type_schema='current',
        augment_variables=False,
    )

    with pytest.raises(ValueError, match="expected 448"):
        _ = dataset[0]
```

### 6.3 Test NaN/Inf Fingerprint Validation

```python
def test_nan_fingerprint_raises(self, tmp_path, mock_tokenizer, mock_fingerprint):
    """NaN in fingerprint raises ValueError."""
    fp_with_nan = [0.0] * 448
    fp_with_nan[100] = float('nan')

    data = [{
        'obfuscated_expr': 'x',
        'ground_truth_expr': 'x',
        'fingerprint': {'flat': fp_with_nan},
    }]
    path = tmp_path / 'nan_fp.jsonl'
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    dataset = MBADataset(
        str(path),
        mock_tokenizer,
        mock_fingerprint,
        node_type_schema='current',
        augment_variables=False,
    )

    with pytest.raises(ValueError, match="NaN or inf"):
        _ = dataset[0]


def test_inf_fingerprint_raises(self, tmp_path, mock_tokenizer, mock_fingerprint):
    """Inf in fingerprint raises ValueError."""
    fp_with_inf = [0.0] * 448
    fp_with_inf[50] = float('inf')

    data = [{
        'obfuscated_expr': 'x',
        'ground_truth_expr': 'x',
        'fingerprint': {'flat': fp_with_inf},
    }]
    path = tmp_path / 'inf_fp.jsonl'
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    dataset = MBADataset(
        str(path),
        mock_tokenizer,
        mock_fingerprint,
        node_type_schema='current',
        augment_variables=False,
    )

    with pytest.raises(ValueError, match="NaN or inf"):
        _ = dataset[0]
```

## Implementation Order

1. **Section 1**: Field name normalization (low risk, unlocks C++ data loading)
2. **Section 5**: Import FINGERPRINT_DIM
3. **Section 2**: Pre-computed fingerprint loading (unlocks training speedup)
4. **Section 3**: AST v2 loading (unlocks graph pre-computation)
5. **Section 4**: Validation script (verify consistency)
6. **Section 6**: Tests

## Rollback Plan

If C++ fingerprints don't match Python:
1. Log warning and compute in Python (slower but correct)
2. Add config flag `use_precomputed_fingerprint: bool = True`
3. Investigate component-by-component to fix C++ generator

## Dependencies

- `src/constants.py`: FINGERPRINT_DIM = 448
- `src/data/fingerprint.py`: SemanticFingerprint.compute()
- C++ generator must output `fingerprint.flat` with seed for reproducibility
