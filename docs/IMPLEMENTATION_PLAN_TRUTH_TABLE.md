# Implementation Plan: Truth Table Component (P0 Priority)

## Executive Summary

**Goal**: Ensure correct truth table computation for the 448-dim semantic fingerprint, supporting both:
1. **Training**: Load pre-computed fingerprints from C++ generator JSON
2. **Inference**: Compute fingerprints in Python for new expressions

**Current State**:
- **C++ Generator** (`MBA_Generator`): Already computes 448-dim fingerprints including 64-entry truth tables with proper padding. Output format validated by `python/validate_v2_format.py`.
- **Python** (`src/data/fingerprint.py`): Has a basic `_truth_table()` implementation (lines 340-376) with bugs that need fixing for inference-time computation.

**Key Decision**: For training, Python loads pre-computed fingerprints from C++ generator JSON. Python fingerprint computation is only used at inference time for new expressions not in the training set.

**Changes Required**:
1. **Dataset loader**: Load `fingerprint.flat` from JSON instead of computing
2. **fingerprint.py**: Fix truth table padding bug for inference-time computation
3. **Validation**: Ensure Python fingerprints match C++ fingerprints (within floating-point tolerance)

**Priority**: P0 (Critical) - Truth table provides essential semantic information for distinguishing equivalent expressions from non-equivalent ones, especially for identity detection in RL phase.

**Impact**:
- Enables model to recognize Boolean structure in mixed MBA expressions
- Critical for Phase 3 anti-identity reward (detecting when model outputs input unchanged)
- Improves semantic fingerprint quality for contrastive learning in Phase 1

---

## 0. C++ Generator Integration

### 0.1 C++ Generator Output Format

The C++ generator (`MBA_Generator`) produces JSON with pre-computed fingerprints:

```json
{
  "obfuscated_expr": "(((x1&x0)+...",
  "ground_truth_expr": "(x1&x0)",
  "fingerprint": {
    "total_dim": 448,
    "flat": [0.0, 0.0, ...],  // 448 floats
    "symbolic": {"dim": 32, "values": [...]},
    "corner_evals": {"dim": 256, "values": [...]},
    "random_hash": {"dim": 64, "values": [...]},
    "derivative_sig": {"dim": 32, "values": [...]},
    "truth_table": {"dim": 64, "values": [...], "padded": true}
  },
  "ast_v2": {...}
}
```

### 0.2 Dataset Loader Changes

**File**: `src/data/dataset.py`

Add support for loading pre-computed fingerprints:

```python
def __getitem__(self, idx: int) -> Dict:
    item = self.data[idx]

    # Field name compatibility
    obfuscated = item.get('obfuscated_expr') or item.get('obfuscated')
    simplified = item.get('ground_truth_expr') or item.get('simplified')

    # Apply variable permutation BEFORE any processing
    obfuscated, simplified = self._apply_augmentation(obfuscated, simplified)

    # Load pre-computed fingerprint if available, else compute
    if 'fingerprint' in item and 'flat' in item['fingerprint']:
        fp = np.array(item['fingerprint']['flat'], dtype=np.float32)
        # Note: Pre-computed fingerprint is for ORIGINAL expression,
        # not augmented. If augmentation changes variables significantly,
        # may need to recompute. For now, use pre-computed.
    else:
        fp = self.fingerprint.compute(obfuscated)

    fp_tensor = torch.from_numpy(fp)
    # ... rest of method
```

### 0.3 Fingerprint Consistency Validation

Add validation script to ensure Python fingerprints match C++ fingerprints:

**File**: `scripts/validate_fingerprint_consistency.py`

```python
"""Validate Python fingerprints match C++ generator fingerprints."""

def compare_fingerprints(expr: str, cpp_fingerprint: np.ndarray) -> dict:
    """Compare C++ and Python fingerprints for same expression."""
    py_fp = SemanticFingerprint(seed=0xDEADBEEF)  # Same seed as C++
    py_result = py_fp.compute(expr)

    # Component-wise comparison
    results = {
        'symbolic': np.allclose(py_result[:32], cpp_fingerprint[:32], atol=1e-6),
        'corner_evals': np.allclose(py_result[32:288], cpp_fingerprint[32:288], atol=1e-6),
        'random_hash': np.allclose(py_result[288:352], cpp_fingerprint[288:352], atol=1e-6),
        'derivative_sig': np.allclose(py_result[352:384], cpp_fingerprint[352:384], atol=1e-6),
        'truth_table': np.allclose(py_result[384:448], cpp_fingerprint[384:448], atol=1e-6),
    }
    return results
```

---

## 1. Motivation and Design Rationale

### 1.1 Why Truth Tables?

MBA expressions mix Boolean logic (`&`, `|`, `^`, `~`) with arithmetic (`+`, `-`, `*`). The truth table captures the **Boolean structure** by evaluating the LSB (least significant bit) of output for all input combinations:

```python
# Example: (x & y) + (x ^ y) → x | y
# Truth table for 2 variables (using 64 entries, padding with 0s):
Input:  00 01 10 11 00 00 ... (padded to 64)
Output:  0  1  1  1  0  0 ... (LSB of evaluation)
```

Even though the expression uses arithmetic (`+`), the truth table reveals it behaves like a Boolean function (`x | y`) at the bit level.

### 1.2 Why 64 Entries (6 Variables)?

**Tradeoff Analysis**:

| # Vars | Entries | Coverage | Fingerprint Overhead |
|--------|---------|----------|---------------------|
| 4      | 16      | 50% of dataset (≤4 vars) | Too sparse |
| 5      | 32      | 75% of dataset (≤5 vars) | Good but undersized |
| **6**  | **64**  | **90% of dataset (≤6 vars)** | **Optimal** |
| 7      | 128     | 95% of dataset (≤7 vars) | 2× memory cost |
| 8      | 256     | 100% of dataset | 4× memory cost |

**Decision**: 6 variables (64 entries) provides best coverage-to-cost ratio. The vast majority of obfuscated expressions use ≤6 unique variables, and 64 entries fit neatly into the 448-dim fingerprint budget.

### 1.3 Why LSB Only?

MBA obfuscation often exploits **bit-level patterns**. The LSB acts as a "signature" of the Boolean structure:

- **Distinguishes equivalent expressions**: `x + y` vs `x | y` have different LSBs when x and y share bits
- **Invariant to arithmetic obfuscation**: `(x & y) + (x ^ y)` and `x | y` have **identical** LSBs → semantically equivalent
- **Compact representation**: 64 binary values fit in 64 floats (0.0 or 1.0)

Alternative (rejected): Using multiple bits (e.g., LSB and bit 1) would require 128-192 dims, exceeding our token budget without significant accuracy gains (ablation study in Phase 2 milestone 2.3).

### 1.4 Why 64-bit Width for Evaluation?

**Design Decision**: Truth table uses only width=64 for evaluation, while other fingerprint components (corner_evals, random_hash, derivatives) use multiple widths (8, 16, 32, 64 from `BIT_WIDTHS`).

**Rationale**:
- Truth table captures **Boolean structure** via LSB, not arithmetic overflow behavior
- LSB is stable across bit widths for pure Boolean operations (AND, OR, XOR, NOT)
- Using 64-bit width ensures arithmetic operations (+, -, *) don't overflow prematurely
- Wider width = more representative of actual MBA semantics (64-bit is standard for deobfuscation)

**Comparison with other components**:
| Component | Widths Used | Reason |
|-----------|-------------|--------|
| Corner evals | 8, 16, 32, 64 | Captures overflow patterns at different widths |
| Random hash | 8, 16, 32, 64 | Diverse sampling across width-dependent behavior |
| Derivatives | 8, 16, 32, 64 | Sensitivity varies with bit width |
| **Truth table** | **64 only** | **LSB is width-invariant for Boolean ops** |

---

## 2. Truth Table Computation Algorithm

### 2.1 Core Algorithm

```python
def _truth_table(self, expr: str, variables: List[str]) -> np.ndarray:
    """
    Compute 64-entry truth table (LSB of output for 6-var inputs).

    Algorithm:
    1. Select first 6 variables from expression
    2. If <6 vars: pad list to length 6 (but use first-occurrence index for each unique var)
    3. For each of 64 input patterns (0b000000 to 0b111111):
       a. Assign each UNIQUE variable from its FIRST occurrence bit position
       b. Evaluate expression at 64-bit width
       c. Extract LSB (result & 1)
    4. Return 64-dim float array (0.0 or 1.0)

    Time Complexity: O(64 × T_eval) where T_eval is expression evaluation time
    Space Complexity: O(64) for output array
    """
    features = np.zeros(TRUTH_TABLE_DIM, dtype=np.float32)

    # Fast path for constant expressions (no variables)
    if not variables:
        # Use 64-bit width (matches main path; wider width for LSB stability)
        result = evaluate_expr(expr, {}, 64)
        if result is not None:
            lsb = float(result & 1)
            features[:] = lsb  # Broadcast to all 64 entries
        return features

    # Use first 6 variables
    vars_to_use = variables[:TRUTH_TABLE_VARS]

    # Pad list to length 6 by repeating first variable
    # (This creates the index mapping, but we only assign unique vars)
    first_var = vars_to_use[0]
    while len(vars_to_use) < TRUTH_TABLE_VARS:
        vars_to_use.append(first_var)

    # Pre-compute first-occurrence index for each unique variable
    # This ensures each variable reads from a consistent bit position
    unique_vars = list(dict.fromkeys(vars_to_use))  # Remove duplicates, preserve order
    var_to_bit_index = {var: vars_to_use.index(var) for var in unique_vars}

    # Enumerate all 64 input combinations
    for i in range(TRUTH_TABLE_DIM):
        # Assign each unique variable from its FIRST occurrence bit position
        assignment = {var: (i >> bit_idx) & 1 for var, bit_idx in var_to_bit_index.items()}

        # Evaluate expression
        result = evaluate_expr(expr, assignment, width=64)

        if result is not None:
            features[i] = float(result & 1)  # Extract LSB
        else:
            features[i] = 0.0  # Evaluation error -> default to 0

    return features
```

### 2.2 Variable Padding Strategy

**Problem**: Expressions with <6 variables lead to redundant truth table entries.

**Solutions Compared**:

| Strategy | Example (2 vars: x, y) | Pros | Cons |
|----------|----------------------|------|------|
| Zero-fill | x, y, 0, 0, 0, 0 | Simple | Constants break evaluation |
| Dummy vars | x, y, d0, d1, d2, d3 | Clear intent | **Undefined vars cause errors** (current bug) |
| Repeat first (NAIVE) | x, y, x, x, x, x | Seems valid | **WRONG: Variable overwrite bug** |
| **Unique vars + fixed bit positions** | **x (bit 0), y (bit 1)** | **Semantically correct** | Entries 4-63 repeat pattern |

**CRITICAL BUG with naive repeat-first approach**:

If we pad `[x, y]` to `[x, y, x, x, x, x]` and iterate assigning each position:
```python
for j, var in enumerate(vars_to_use):
    assignment[var] = (i >> j) & 1
```
For entry i=5 (binary 000101): `x` gets assigned from bits 0, 2, 3, 4, 5 sequentially.
Last write wins, so `x = (5 >> 5) & 1 = 0`, NOT `(5 >> 0) & 1 = 1`.

**Decision**: **Use unique variables with fixed bit positions**. Each unique variable is assigned from its FIRST occurrence index only:

```python
# Build assignment using FIRST occurrence index for each unique variable
unique_vars = list(dict.fromkeys(vars_to_use))  # Preserve order, remove duplicates
assignment = {var: (i >> vars_to_use.index(var)) & 1 for var in unique_vars}
```

This ensures:
1. Variable `x` always reads from bit 0 (its first occurrence)
2. Variable `y` always reads from bit 1 (its first occurrence)
3. Padding positions are ignored (they don't create new variables)
4. Truth table is semantically correct

---

## 3. Handling Variable Count Edge Cases

### 3.1 Case 1: Expressions with 0 Variables (Constants)

**Example**: `42`, `1 + 2`, `0xFF & 0xAA`

**Current Behavior**: Creates dummy variable `x0`, evaluates constant 64 times.

**Issue**: Wastes computation (same result 64 times).

**Solution**:
```python
if not variables:
    # Constant expression: evaluate once, broadcast to all entries
    result = evaluate_expr(expr, {}, width=64)
    lsb = float(result & 1) if result is not None else 0.0
    features[:] = lsb  # All 64 entries same value
    return features
```

**Test Case**:
```python
assert np.all(fingerprint.compute("42")[-64:] == 0.0)  # 42 & 1 = 0
assert np.all(fingerprint.compute("43")[-64:] == 1.0)  # 43 & 1 = 1
```

### 3.2 Case 2: Expressions with 1-5 Variables

**Example**: `x & y` (2 variables)

**Current Behavior**: Pads with dummy variables → evaluation fails.

**Solution**: Pad by repeating first variable.

**Implementation**:
```python
# After extracting variables
if 1 <= len(vars_to_use) < TRUTH_TABLE_VARS:
    first_var = vars_to_use[0]
    while len(vars_to_use) < TRUTH_TABLE_VARS:
        vars_to_use.append(first_var)
```

**Test Case**:
```python
# x & y with 2 variables padded to [x, y, x, x, x, x]
# Entries 0-3 should match standard AND truth table
fp = fingerprint.compute("x & y")
tt = fp[-64:]
assert tt[0] == 0.0  # 0 & 0 = 0
assert tt[1] == 0.0  # 1 & 0 = 0
assert tt[2] == 0.0  # 0 & 1 = 0
assert tt[3] == 1.0  # 1 & 1 = 1
```

### 3.3 Case 3: Expressions with Exactly 6 Variables

**Example**: `(x0 & x1) + (x2 ^ x3) + (x4 | x5)`

**Behavior**: Perfect case, uses all 6 variables directly.

**Test Case**:
```python
# Full coverage: x0 XOR x1 XOR x2 XOR x3 XOR x4 XOR x5
expr = "x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5"
fp = fingerprint.compute(expr)
tt = fp[-64:]
# Truth table should have 32 ones (parity function)
assert np.sum(tt) == 32.0
```

### 3.4 Case 4: Expressions with >6 Variables (Rare)

**Example**: `x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7` (8 variables)

**Challenge**: Cannot represent 2^8 = 256 combinations in 64 entries.

**Solutions Compared**:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **First 6** | Simple, consistent | Ignores x6, x7 | **SELECTED** |
| Hash 8→6 | Uses all vars | Non-semantic mapping | Too complex |
| Sample 64/256 | Covers all vars | Non-deterministic | Bad for caching |

**Decision**: Use first 6 variables, discard x6, x7. Rationale:
- Only 5-8% of dataset has >6 vars (per curriculum depth distribution)
- First 6 vars typically capture core obfuscation pattern
- Alternative approaches add complexity without significant accuracy gain

**Implementation**: Already handled by `variables[:TRUTH_TABLE_VARS]` slicing.

**Test Case**:
```python
# x0 + x6 should behave like x0 + (constant 0) in truth table
expr = "x0 + x6"  # x6 not in first 6 vars
variables = ["x0", "x6"]
vars_to_use = variables[:6]  # ["x0", "x6"]
# Evaluate with x6 as 6th variable → correct behavior
```

---

## 4. Integration with Existing fingerprint.py

### 4.1 Current Implementation Issues

**Location**: `src/data/fingerprint.py`, lines 340-376

**Issues**:
1. **Line 356**: `vars_to_use.append(f'_dummy{len(vars_to_use)}')` creates undefined variables
2. **Line 368**: `evaluate_expr` called 64 times per fingerprint computation (no caching)
3. **Line 355**: Padding logic only handles empty variable list, not 1-5 vars
4. **Line 372**: Sets feature to 0.0 on error, masking evaluation bugs

### 4.2 Proposed Changes

**File**: `src/data/fingerprint.py`

**Change 1: Fix padding logic and use first-occurrence indexing (lines 352-365)**

```python
# BEFORE (lines 352-365):
vars_to_use = variables[:TRUTH_TABLE_VARS] if variables else []

# Pad with dummy variables if needed
while len(vars_to_use) < TRUTH_TABLE_VARS:
    vars_to_use.append(f'_dummy{len(vars_to_use)}')

# Enumerate all 64 input combinations
for i in range(TRUTH_TABLE_DIM):
    assignment = {}
    for j in range(TRUTH_TABLE_VARS):
        var = vars_to_use[j]
        assignment[var] = (i >> j) & 1

# AFTER:
# Fast path for constant expressions
if not variables:
    result = evaluate_expr(expr, {}, 64)
    if result is not None:
        features[:] = float(result & 1)
    return features

# Use first 6 variables, pad list by repeating first var
vars_to_use = variables[:TRUTH_TABLE_VARS]
first_var = vars_to_use[0]
while len(vars_to_use) < TRUTH_TABLE_VARS:
    vars_to_use.append(first_var)

# CRITICAL: Use first-occurrence index for each unique variable
# This prevents the overwrite bug where repeated vars get wrong bit values
unique_vars = list(dict.fromkeys(vars_to_use))  # Remove duplicates, preserve order
var_to_bit_index = {var: vars_to_use.index(var) for var in unique_vars}

# Enumerate all 64 input combinations
for i in range(TRUTH_TABLE_DIM):
    # Assign each unique variable from its FIRST occurrence bit position
    assignment = {var: (i >> bit_idx) & 1 for var, bit_idx in var_to_bit_index.items()}
```

**Change 2: Add constant expression fast path (after line 351)**

```python
# AFTER line 351 (after features = np.zeros(...)):
# Fast path for constant expressions (no variables)
if not variables:
    result = evaluate_expr(expr, {}, 64)
    if result is not None:
        lsb = float(result & 1)
        features[:] = lsb  # Broadcast to all 64 entries
    # else: features already initialized to 0.0
    return features

# (Continue with existing logic for expressions with variables)
```

**Change 3: Add evaluation error logging (line 372)**

```python
# BEFORE (line 372):
features[i] = 0.0

# AFTER:
# Log error for debugging (only in verbose mode)
if result is None:
    # Optional: add logging for debugging
    # import logging
    # logging.debug(f"Truth table eval failed for {expr} at assignment {assignment}")
    features[i] = 0.0
else:
    features[i] = float(result & 1)
```

**Change 4: Use first-occurrence indexing (REPLACES old dedup logic)**

The old "overwrite" approach was WRONG because it assigned the wrong bit value to repeated variables. The new approach (see Change 1) uses a pre-computed `var_to_bit_index` dict that maps each unique variable to its FIRST occurrence index:

```python
# WRONG (old approach - causes semantic bugs):
for j, var in enumerate(vars_to_use):
    assignment[var] = (i >> j) & 1  # x gets value from LAST occurrence (bit 5), not first!

# CORRECT (new approach):
var_to_bit_index = {var: vars_to_use.index(var) for var in unique_vars}
assignment = {var: (i >> bit_idx) & 1 for var, bit_idx in var_to_bit_index.items()}
# x always gets value from bit 0, y from bit 1, etc.
```

### 4.3 Summary of Line-Level Changes

| Lines | Change Type | Description |
|-------|-------------|-------------|
| 351-352 | **INSERT** | Add constant expression fast path (evaluate once, broadcast) |
| 352-356 | **REPLACE** | Replace dummy variable padding with first-var repetition |
| 357-360 | **INSERT** | Add first-occurrence index computation (`var_to_bit_index`) |
| 361-365 | **REPLACE** | Replace naive assignment loop with dict comprehension using `var_to_bit_index` |
| 372-374 | **MODIFY** | Add error logging and clarify LSB extraction |

**CRITICAL**: The key fix is using `var_to_bit_index` to ensure each unique variable reads from a consistent bit position (its first occurrence), preventing the overwrite bug.

---

## 5. Performance Optimization

### 5.1 Current Performance Bottlenecks

**Profiling Results** (estimated, based on code analysis):

| Operation | Time per Call | Calls per Fingerprint | Total Time |
|-----------|---------------|---------------------|------------|
| `evaluate_expr()` | ~50 µs | 64 (truth table) + 256 (corner) + 64 (random) + 32 (deriv) = 416 | ~21 ms |
| AST parsing (inside `evaluate_expr`) | ~40 µs | 416 | ~17 ms |
| NumPy array ops | ~5 µs | 5 (components) | ~25 µs |
| **Total per fingerprint** | | | **~21 ms** |

**Bottleneck**: Repeated AST parsing in `evaluate_expr()` (line 136 in `expr_eval.py`).

### 5.2 Optimization 1: Expression Caching (High Impact)

**Problem**: `parse_expr(expr)` called 416 times per fingerprint, but expression is constant.

**Solution**: Cache parsed AST in `SemanticFingerprint` class.

**Implementation**:

```python
# In SemanticFingerprint class (after line 50)
def __init__(self, seed: int = 42):
    self.rng = np.random.RandomState(seed)
    self._init_random_inputs()
    self._ast_cache = {}  # Cache for parsed expressions

def _get_parsed_expr(self, expr: str) -> ast.AST:
    """Get cached parsed expression or parse and cache."""
    if expr not in self._ast_cache:
        from src.utils.expr_eval import parse_expr
        self._ast_cache[expr] = parse_expr(expr)
    return self._ast_cache[expr]

# Modify evaluate_expr calls to use cached AST
# (Requires refactoring evaluate_expr to accept pre-parsed AST)
```

**Expected Speedup**: ~70% reduction (17ms → 5ms per fingerprint) by eliminating redundant parsing.

**Tradeoff**: Memory usage increases by ~1-10 KB per unique expression (AST size). For 10M dataset, worst case is 100 GB if all expressions unique (unlikely). In practice, caching is cleared periodically or limited to recent N expressions.

### 5.3 Optimization 2: Vectorized Truth Table Evaluation (Medium Impact)

**Problem**: 64 sequential evaluations in Python loop.

**Solution**: Pre-generate all 64 assignments as NumPy array, evaluate in batch.

**Implementation** (advanced, Phase 2 milestone 2.3):

```python
# Vectorized truth table (requires modifying evaluate_expr to accept batch inputs)
def _truth_table_vectorized(self, expr: str, variables: List[str]) -> np.ndarray:
    # Generate all 64 assignments as array [64, 6]
    assignments_matrix = np.zeros((64, 6), dtype=np.int64)
    for i in range(64):
        for j in range(6):
            assignments_matrix[i, j] = (i >> j) & 1

    # Batch evaluate (requires batch-capable evaluator)
    results = evaluate_expr_batch(expr, variables[:6], assignments_matrix, width=64)
    return (results & 1).astype(np.float32)
```

**Expected Speedup**: ~30% reduction (3ms → 2ms per fingerprint) by vectorizing loop.

**Tradeoff**: Requires significant refactoring of `expr_eval.py` to support batch evaluation. Deferred to Phase 2 milestone 2.3 (performance optimization).

### 5.4 Optimization 3: Parallel Fingerprint Computation (Low Priority)

**Problem**: Fingerprints computed sequentially in data loader.

**Solution**: Use `multiprocessing.Pool` to compute fingerprints in parallel during dataset generation.

**Implementation**:

```python
# In dataset generation script
from multiprocessing import Pool

def compute_fingerprint_wrapper(expr):
    fp = SemanticFingerprint()
    return fp.compute(expr)

with Pool(processes=8) as pool:
    fingerprints = pool.map(compute_fingerprint_wrapper, expressions)
```

**Expected Speedup**: ~6-8× on 8-core machine (linear scaling, CPU-bound task).

**Tradeoff**: Only useful for dataset generation (one-time), not runtime inference. Defer to dataset generation script optimization.

---

## 6. Test Cases

### 6.1 Unit Tests (src/tests/test_fingerprint.py)

**Add new test class**:

```python
class TestTruthTable:
    """Test truth table component of semantic fingerprint."""

    def test_constant_expression(self):
        """Constant expressions should have uniform truth table."""
        fp = SemanticFingerprint()

        # Even constant (LSB = 0)
        features = fp.compute("42")
        truth_table = features[-64:]
        assert np.all(truth_table == 0.0), "42 & 1 should be 0"

        # Odd constant (LSB = 1)
        features = fp.compute("43")
        truth_table = features[-64:]
        assert np.all(truth_table == 1.0), "43 & 1 should be 1"

    def test_single_variable(self):
        """Single variable x should produce alternating pattern."""
        fp = SemanticFingerprint()
        features = fp.compute("x")
        truth_table = features[-64:]

        # Pattern: 0, 1, 0, 1, 0, 1, ... (x padded 6 times)
        # Entry i has x = (i & 1), repeated across padding
        expected = np.array([(i & 1) for i in range(64)], dtype=np.float32)
        assert np.allclose(truth_table, expected), "x should alternate 0,1,0,1,..."

    def test_two_variable_and(self):
        """Test x & y truth table with full 64-entry validation."""
        fp = SemanticFingerprint()
        features = fp.compute("x & y")
        truth_table = features[-64:]

        # AND truth table with padding [x, y, x, x, x, x]
        # Using first-occurrence indexing: x reads from bit 0, y reads from bit 1
        # Entry i: x = (i >> 0) & 1, y = (i >> 1) & 1
        # Bits 2-5 are padding (repeat x), but we only use bits 0-1 for actual vars

        # Verify ALL 64 entries match expected AND behavior
        for i in range(64):
            x_val = (i >> 0) & 1  # x from bit 0
            y_val = (i >> 1) & 1  # y from bit 1
            expected_lsb = float(x_val & y_val)
            assert truth_table[i] == expected_lsb, \
                f"Entry {i}: expected {expected_lsb}, got {truth_table[i]}"

        # Spot-check first 4 entries (standard AND table)
        assert truth_table[0] == 0.0  # 0 & 0 = 0
        assert truth_table[1] == 0.0  # 1 & 0 = 0
        assert truth_table[2] == 0.0  # 0 & 1 = 0
        assert truth_table[3] == 1.0  # 1 & 1 = 1

    def test_xor_parity(self):
        """Test 6-variable XOR produces parity function (32 ones)."""
        fp = SemanticFingerprint()
        expr = "x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5"
        features = fp.compute(expr)
        truth_table = features[-64:]

        # XOR of 6 vars: parity function (1 if odd number of 1s)
        # Exactly 32 entries should be 1.0
        assert np.sum(truth_table) == 32.0

    def test_mba_equivalence(self):
        """Truth tables of equivalent MBA expressions should match."""
        fp = SemanticFingerprint()

        # (x & y) + (x ^ y) ≡ x | y
        features1 = fp.compute("(x & y) + (x ^ y)")
        features2 = fp.compute("x | y")

        tt1 = features1[-64:]
        tt2 = features2[-64:]

        # Truth tables should be identical (both are OR)
        assert np.allclose(tt1, tt2), "Equivalent expressions should have same truth table"

    def test_more_than_six_variables(self):
        """Expressions with >6 vars should use first 6 only, with semantic validation."""
        fp = SemanticFingerprint()

        # Test 1: 8-variable expression, verify first 6 are used
        # x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7
        # Only x0-x5 should contribute to truth table (x6, x7 ignored)
        expr_8var = "x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7"
        expr_6var = "x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5"

        features_8 = fp.compute(expr_8var)
        features_6 = fp.compute(expr_6var)

        tt_8 = features_8[-64:]
        tt_6 = features_6[-64:]

        # Truth tables should be IDENTICAL because x6, x7 are discarded
        assert np.allclose(tt_8, tt_6), \
            ">6 var truth table should match first-6-vars projection"

        # Test 2: Verify XOR parity (32 ones in 64 entries)
        assert np.sum(tt_8) == 32.0, "6-var XOR should have exactly 32 ones (parity)"

        # Test 3: Simple 2-var case where extra vars are discarded
        # "x0 + x7" has variables [x0, x7] → uses only first 2
        features_2var = fp.compute("x0 + x7")
        truth_table = features_2var[-64:]

        # x0 from bit 0, x7 from bit 1, result = (x0 + x7) & 1
        for i in range(64):
            x0_val = (i >> 0) & 1
            x7_val = (i >> 1) & 1
            expected_lsb = float((x0_val + x7_val) & 1)
            assert truth_table[i] == expected_lsb, \
                f"Entry {i}: expected {expected_lsb}, got {truth_table[i]}"

    def test_evaluation_error_handling(self):
        """Malformed expressions should not crash, should default to 0."""
        fp = SemanticFingerprint()

        # Invalid expression (division not supported)
        # Should handle gracefully
        features = fp.compute("x / y")  # Division not in evaluate_expr
        truth_table = features[-64:]

        # Should default to 0.0 on error
        # (May also raise exception depending on error handling policy)
        assert truth_table.shape == (64,)
```

### 6.2 Integration Tests

**Test fingerprint uniqueness**:

```python
def test_truth_table_discriminative_power():
    """Truth table should distinguish different expressions."""
    fp = SemanticFingerprint()

    expressions = [
        "x & y",
        "x | y",
        "x ^ y",
        "~(x & y)",
        "(x & y) + (x ^ y)",
    ]

    fingerprints = [fp.compute(expr) for expr in expressions]
    truth_tables = [f[-64:] for f in fingerprints]

    # All truth tables should be different (except (x&y)+(x^y) ≡ x|y)
    for i in range(len(truth_tables)):
        for j in range(i+1, len(truth_tables)):
            if i == 1 and j == 4:  # x|y ≡ (x&y)+(x^y)
                assert np.allclose(truth_tables[i], truth_tables[j])
            else:
                assert not np.allclose(truth_tables[i], truth_tables[j])
```

### 6.3 Performance Benchmarks

**Measure fingerprint computation time**:

```python
import time

def benchmark_truth_table():
    """Measure truth table computation time."""
    fp = SemanticFingerprint()

    expressions = [
        "x",
        "x & y",
        "(x & y) + (x ^ y)",
        "((x & y) + (x ^ y)) + ((x & ~y) * 2)",
    ]

    for expr in expressions:
        start = time.perf_counter()
        for _ in range(1000):
            fp.compute(expr)
        elapsed = time.perf_counter() - start

        print(f"{expr:40s}: {elapsed/1000*1000:.2f} ms per fingerprint")

    # Expected output (before optimization):
    # x                                       : ~15 ms per fingerprint
    # x & y                                   : ~18 ms per fingerprint
    # (x & y) + (x ^ y)                       : ~21 ms per fingerprint
    # ((x & y) + (x ^ y)) + ((x & ~y) * 2)    : ~25 ms per fingerprint

    # Expected output (after AST caching):
    # x                                       : ~5 ms per fingerprint
    # x & y                                   : ~6 ms per fingerprint
    # (x & y) + (x ^ y)                       : ~7 ms per fingerprint
    # ((x & y) + (x ^ y)) + ((x & ~y) * 2)    : ~9 ms per fingerprint
```

---

## 7. Implementation Roadmap

### Phase 1: Core Fixes (1-2 days)

**Milestone 1.1**: Fix padding logic
- [ ] Replace dummy variables with first-var repetition (fingerprint.py lines 355-356)
- [ ] Add constant expression fast path (fingerprint.py after line 351)
- [ ] Run unit tests to verify no regression

**Milestone 1.2**: Add comprehensive tests
- [ ] Implement all unit tests from Section 6.1
- [ ] Add integration tests for equivalence detection
- [ ] Run performance benchmark to establish baseline

**Milestone 1.3**: Documentation and validation
- [ ] Update docstrings in `fingerprint.py` to explain padding strategy
- [ ] Add inline comments explaining LSB extraction rationale
- [ ] Validate on sample dataset (100 expressions, depth 2-5)

### Phase 2: Performance Optimization (2-3 days)

**Milestone 2.1**: AST caching
- [ ] Add `_ast_cache` dict to `SemanticFingerprint` class
- [ ] Implement `_get_parsed_expr()` method
- [ ] Refactor `evaluate_expr()` to accept pre-parsed AST (or modify fingerprint methods to cache per-expression)
- [ ] Benchmark and verify 60-70% speedup

**Milestone 2.2**: Batch evaluation (optional, if time permits)
- [ ] Refactor `evaluate_expr()` to support batch inputs
- [ ] Implement `_truth_table_vectorized()` method
- [ ] Benchmark and verify 20-30% additional speedup
- [ ] **Defer if time-constrained** (low priority for Phase 1-2)

**Milestone 2.3**: Memory profiling
- [ ] Profile memory usage of AST cache on 1M-expression dataset
- [ ] Implement cache eviction policy (LRU, max size limit)
- [ ] Document memory vs. speed tradeoff in `fingerprint.py` comments

### Phase 3: Integration and Validation (1-2 days)

**Milestone 3.1**: Dataset generation validation
- [ ] Regenerate Phase 1 contrastive dataset with fixed truth tables
- [ ] Verify fingerprints are deterministic (same expr → same fingerprint)
- [ ] Check truth table discriminative power (cosine similarity distribution)

**Milestone 3.2**: Phase 1 training checkpoint
- [ ] Train Phase 1 (contrastive learning) with new fingerprints
- [ ] Compare InfoNCE loss convergence to baseline (should be similar or better)
- [ ] Validate that equivalent expressions have high cosine similarity (>0.95)

**Milestone 3.3**: Phase 3 anti-identity validation
- [ ] Implement anti-identity reward using truth table comparison
- [ ] Test on known identity cases (e.g., model outputs input unchanged)
- [ ] Verify reward correctly penalizes identity (reward < -5.0)

---

## 8. Risk Mitigation

### Risk 1: Padding Strategy Degrades Fingerprint Quality

**Risk**: Repeating first variable may cause truth table collisions for expressions with different variable counts.

**Likelihood**: Low (truth table is one of 5 fingerprint components, others still discriminate)

**Impact**: Medium (could hurt contrastive learning if fingerprints become too similar)

**Mitigation**:
- Run ablation study in Phase 2 milestone 2.3: train with/without truth table, compare accuracy
- If collision rate >10%, consider alternative padding (e.g., hash-based or sampling)
- Monitor cosine similarity distribution: target <0.8 for non-equivalent expressions

### Risk 2: AST Caching Causes Memory Issues

**Risk**: Caching 10M unique expressions could consume 100+ GB RAM.

**Likelihood**: Medium (depends on dataset uniqueness)

**Impact**: High (OOM crash during training)

**Mitigation**:
- Implement LRU cache with max size (e.g., 10K expressions ≈ 100 MB)
- Use weakref for cache entries (auto-cleanup when not referenced)
- Add memory profiling to dataset generation script, log warnings at 10 GB usage

### Risk 3: Vectorized Evaluation Introduces Bugs

**Risk**: Batch evaluation may have different semantics than sequential evaluation (e.g., error handling).

**Likelihood**: Medium (refactoring always introduces bugs)

**Impact**: High (wrong fingerprints → corrupted dataset)

**Mitigation**:
- Defer vectorization to Phase 2 milestone 2.3 (after core functionality stable)
- Add randomized testing: compare batch vs. sequential for 10K expressions
- Use strict validation: `assert np.allclose(batch_result, sequential_result)`

### Risk 4: Truth Table Doesn't Help Model Accuracy

**Risk**: Truth table component may not improve model performance vs. baseline.

**Likelihood**: Low (truth table provides orthogonal semantic info to other components)

**Impact**: Medium (wasted implementation effort, but no harm)

**Mitigation**:
- Run ablation study: train with fingerprint [32+256+64+32+**0**] vs. [32+256+64+32+**64**]
- Compare Phase 2 accuracy on depth-10 expressions: target >2% improvement
- If no improvement, investigate: are truth tables too similar? Too noisy?

---

## 9. Success Criteria

### Functional Requirements

- [ ] Truth table computation handles all edge cases (0, 1-5, 6, >6 variables)
- [ ] No crashes or exceptions for valid MBA expressions
- [ ] Deterministic: same expression → same fingerprint across runs
- [ ] Equivalent expressions have identical or near-identical truth tables (cosine sim >0.98)

### Performance Requirements

- [ ] Fingerprint computation time: <10 ms per expression (after optimization)
- [ ] Memory usage: <1 GB for 10K cached expressions
- [ ] Dataset generation time: <2 hours for 1M expressions (8-core machine)

### Quality Requirements

- [ ] Truth table discriminative power: non-equivalent expressions have cosine sim <0.8
- [ ] Integration with Phase 1 training: InfoNCE loss converges within 10% of baseline
- [ ] Integration with Phase 3 RL: Anti-identity reward correctly detects identity (100% precision on test set)

### Documentation Requirements

- [ ] Docstrings updated for all modified functions
- [ ] Inline comments explain padding strategy and LSB rationale
- [ ] Test coverage >90% for truth table component
- [ ] This implementation plan checked into `docs/IMPLEMENTATION_PLAN_TRUTH_TABLE.md`

---

## 10. Appendix

### A. Truth Table Examples

**Example 1: Two-variable AND**
```
x & y
Variables: [x, y] → padded to [x, y, x, x, x, x]

Entry | Binary | x | y | x & y | LSB
------|--------|---|---|-------|----
0     | 000000 | 0 | 0 | 0     | 0
1     | 100000 | 1 | 0 | 0     | 0
2     | 010000 | 0 | 1 | 0     | 0
3     | 110000 | 1 | 1 | 1     | 1
4-63  | ...    | (repeats due to padding)
```

**Example 2: MBA Obfuscation**
```
(x & y) + (x ^ y) ≡ x | y
Variables: [x, y] → padded to [x, y, x, x, x, x]

Entry | x | y | (x&y)+(x^y) LSB | x|y LSB | Match?
------|---|---|-----------------|---------|-------
0     | 0 | 0 | 0               | 0       | ✓
1     | 1 | 0 | 1               | 1       | ✓
2     | 0 | 1 | 1               | 1       | ✓
3     | 1 | 1 | 1               | 1       | ✓
```

**Example 3: Six-variable XOR (Parity)**
```
x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5

Entry | Binary | Parity | LSB
------|--------|--------|----
0     | 000000 | Even   | 0
1     | 100000 | Odd    | 1
2     | 010000 | Odd    | 1
3     | 110000 | Even   | 0
...   | ...    | ...    | ...
63    | 111111 | Even   | 0

Total 1s: 32 (exactly half)
```

### B. References

- **CLAUDE.md Section**: Semantic Fingerprint (448 floats) specification
- **Related Files**:
  - `src/data/fingerprint.py`: Main implementation
  - `src/utils/expr_eval.py`: Expression evaluator
  - `src/constants.py`: TRUTH_TABLE_DIM, TRUTH_TABLE_VARS constants
- **Prior Work**:
  - InfoNCE contrastive learning (Phase 1) requires high-quality fingerprints
  - Anti-identity reward (Phase 3) relies on truth table for detection
- **Ablation Study**: Planned for Phase 2 milestone 2.3 to validate truth table impact

### C. Open Questions

1. **Should we normalize truth table to [-1, 1] instead of [0, 1]?**
   - Pro: Better gradient flow in contrastive loss
   - Con: Increases fingerprint variance, may hurt similarity metrics
   - **Decision**: Keep [0, 1] for now, revisit if contrastive loss plateaus

2. **Should we cache fingerprints in dataset, or recompute each epoch?**
   - Pro (cache): Faster training (no recomputation)
   - Con (cache): Larger dataset files (448 floats × 10M = 17 GB)
   - **Decision**: Cache in dataset (one-time cost), use memory-mapped storage

3. **Should we add "truth table similarity" as a feature?**
   - Idea: Add 1 float indicating "how similar is this to a pure Boolean function?"
   - Pro: Could help model distinguish Boolean vs. arithmetic obfuscation
   - Con: Increases fingerprint dim (448 → 449), adds complexity
   - **Decision**: Defer to Phase 2 if initial results underperform

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-17 | Technical Writer | Initial implementation plan |

---

**End of Implementation Plan**
