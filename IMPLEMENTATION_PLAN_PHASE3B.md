# Implementation Plan: Phase 3B - AST Parser Edge Type Mode

## Overview

Add `edge_type_mode` parameter to `ast_to_graph()` and `expr_to_graph()` functions in `src/data/ast_parser.py` to enable config-driven edge type selection.

**Priority**: P0 (blocks config-driven edge type consolidation)
**Effort**: ~45 minutes
**Risk**: Low (additive change with backward-compatible defaults)

---

## Current State

| Function | Edge Type Mode Support | Status |
|----------|----------------------|--------|
| `ast_to_graph()` | None (always legacy 6-type) | MISSING |
| `ast_to_optimized_graph()` | Always optimized 8-type | EXISTS |
| `expr_to_graph()` | None (always legacy 6-type) | MISSING |
| `expr_to_optimized_graph()` | Always optimized 8-type | EXISTS |

**Problem**: Config has `edge_type_mode` but callers must choose between separate functions. No unified API.

---

## Callers Analysis

| Caller | File:Line | Current Call | Needs Update |
|--------|-----------|--------------|--------------|
| simplify.py | scripts/simplify.py:81 | `expr_to_graph(expr)` | Yes |
| simplify.py | scripts/simplify.py:128 | `expr_to_graph(expr)` | Yes |
| pipeline.py | src/inference/pipeline.py:200 | `expr_to_graph(expr)` | Yes |
| htps.py | src/inference/htps.py:513 | `expr_to_graph(node.expr)` | Yes |
| test_data.py | tests/test_data.py:85,95,106,114,398,460,461 | `ast_to_graph(ast)` / `expr_to_graph(expr)` | No (tests legacy) |
| test_edge_type_equivalence.py | tests/test_edge_type_equivalence.py:43,63 | `expr_to_graph(expr)` | No (tests legacy) |
| test_dag_features.py | tests/test_dag_features.py:326,336,348 | `expr_to_graph(...)` | No (tests legacy) |

---

## Implementation

### Step 1: Modify `ast_to_graph()` (src/data/ast_parser.py:209)

**Current signature**:
```python
def ast_to_graph(ast: ASTNode, use_dag_features: bool = USE_DAG_FEATURES) -> Data:
```

**New signature**:
```python
def ast_to_graph(
    ast: ASTNode,
    use_dag_features: bool = USE_DAG_FEATURES,
    edge_type_mode: str = "legacy",
) -> Data:
```

**Changes**:
1. Add `edge_type_mode: str = "legacy"` parameter after `use_dag_features`
2. Add validation at function start:
   ```python
   if edge_type_mode not in ("legacy", "optimized"):
       raise ValueError(f"edge_type_mode must be 'legacy' or 'optimized', got: {edge_type_mode}")
   ```
3. Add delegation at function start (after validation):
   ```python
   if edge_type_mode == "optimized":
       return ast_to_optimized_graph(ast, use_dag_features=use_dag_features)
   ```
4. Update docstring to document the new parameter:
   ```python
   """
   Convert AST to PyTorch Geometric Data object.

   Args:
       ast: Root ASTNode
       use_dag_features: Compute and attach DAG positional features
       edge_type_mode: Edge type system - "legacy" (6-type, default) or "optimized" (8-type)

   Returns:
       PyTorch Geometric Data object

   Edge Types:
       - legacy: CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR
       - optimized: LEFT/RIGHT/UNARY_OPERAND, *_INV, DOMAIN_BRIDGE_DOWN/UP
   """
   ```

### Step 2: Modify `expr_to_graph()` (src/data/ast_parser.py:349)

**Current signature**:
```python
def expr_to_graph(expr: str, use_dag_features: bool = USE_DAG_FEATURES) -> Data:
```

**New signature**:
```python
def expr_to_graph(
    expr: str,
    use_dag_features: bool = USE_DAG_FEATURES,
    edge_type_mode: str = "legacy",
) -> Data:
```

**Changes**:
1. Add `edge_type_mode: str = "legacy"` parameter
2. Pass through to `ast_to_graph()`:
   ```python
   return ast_to_graph(ast, use_dag_features=use_dag_features, edge_type_mode=edge_type_mode)
   ```
3. Update docstring to document the new parameter

### Step 3: Update Callers (Optional - can defer)

These callers should eventually pass `edge_type_mode` from config, but since the default is "legacy" (backward-compatible), this can be deferred:

| File | Change |
|------|--------|
| scripts/simplify.py | Add `--edge-type-mode` CLI arg, pass to `expr_to_graph()` |
| src/inference/pipeline.py | Accept `edge_type_mode` in constructor, pass to `expr_to_graph()` |
| src/inference/htps.py | Accept `edge_type_mode` in constructor, pass to `expr_to_graph()` |

**Note**: Caller updates are P1, not P0. The default "legacy" maintains backward compatibility.

---

## Verification

### Unit Tests

```bash
# Run existing edge type equivalence tests
pytest tests/test_edge_type_equivalence.py -v

# Run existing data tests (should pass with default "legacy")
pytest tests/test_data.py -v

# Run DAG feature tests
pytest tests/test_dag_features.py -v
```

### Manual Verification

```python
from src.data.ast_parser import expr_to_graph

# Test 1: Default (legacy) - backward compatibility
g1 = expr_to_graph("x & y")
assert g1.edge_type.max() < 6, "Legacy should have max edge type < 6"

# Test 2: Explicit legacy
g2 = expr_to_graph("x & y", edge_type_mode="legacy")
assert g2.edge_type.max() < 6

# Test 3: Optimized mode
g3 = expr_to_graph("x & y", edge_type_mode="optimized")
assert g3.edge_type.max() < 8, "Optimized should have max edge type < 8"

# Test 4: Invalid mode raises
try:
    expr_to_graph("x & y", edge_type_mode="invalid")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "edge_type_mode" in str(e)

print("All manual tests passed!")
```

### Integration Test

```python
# Verify optimized mode produces same structure as ast_to_optimized_graph()
from src.data.ast_parser import expr_to_graph, expr_to_optimized_graph

expr = "(x & y) + (x ^ y)"

g_via_param = expr_to_graph(expr, edge_type_mode="optimized")
g_direct = expr_to_optimized_graph(expr)

assert g_via_param.x.shape == g_direct.x.shape
assert g_via_param.edge_index.shape == g_direct.edge_index.shape
assert torch.equal(g_via_param.edge_type, g_direct.edge_type)

print("Integration test passed!")
```

---

## Rollback Plan

If issues arise:
1. The `ast_to_optimized_graph()` and `expr_to_optimized_graph()` functions remain available as direct alternatives
2. Default "legacy" mode ensures no existing code breaks
3. Can revert the parameter addition without affecting any caller that doesn't use it

---

## Files Changed

| File | Type | Lines Modified |
|------|------|----------------|
| src/data/ast_parser.py | MODIFY | ~20 lines (2 functions) |

---

## Dependencies

**Upstream**: None (this is leaf implementation)

**Downstream**:
- scripts/simplify.py (optional update)
- src/inference/pipeline.py (optional update)
- src/inference/htps.py (optional update)

---

## Acceptance Criteria

1. [ ] `ast_to_graph(ast, edge_type_mode="legacy")` produces 6-type edges
2. [ ] `ast_to_graph(ast, edge_type_mode="optimized")` produces 8-type edges (delegates to `ast_to_optimized_graph`)
3. [ ] `expr_to_graph(expr, edge_type_mode="optimized")` produces identical output to `expr_to_optimized_graph(expr)`
4. [ ] Invalid `edge_type_mode` raises `ValueError`
5. [ ] All existing tests pass (backward compatibility)
6. [ ] Docstrings updated with new parameter documentation
