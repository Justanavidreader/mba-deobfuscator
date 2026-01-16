# Implementation Plan: Critical Issues Resolution

## Overview
Resolves P0/P1/P2 issues from ACTION_PLAN_CRITICAL_ISSUES.md in dependency order.

**WINDOWS USERS**: Run all verification commands in Git Bash, not PowerShell or CMD.

---

## Completeness Matrix: Action Plan → Implementation Plan

| Action Plan Item | Priority | Implementation Phase | Status |
|------------------|----------|---------------------|--------|
| 1.1: Consolidate edge type usage | P0 | Phase 3A, 3B | Verify in Phase 0 |
| 1.2: Add config exposure | P1 | Phase 3C | Verify in Phase 0 |
| 1.3: Documentation update | P2 | Phase 5C | Pending |
| 2.1: Dataset schema detection | P0 | Phase 2A, 2B, 2C | Verify in Phase 0 |
| 2.2: Schema version docs + migration | P1 | Phase 2D, 5C | Verify in Phase 0 |
| 3.1: Expose feature flags | P1 | Phase 4A | Verify in Phase 0 |
| 3.2: Config loading logic | P1 | Phase 1A, 4B | Verify in Phase 0 |
| 3.3: Document feature flags | P2 | Phase 5C | Pending |

**P0 items**: 1.1, 2.1 (BLOCKING - must complete)
**P1 items**: 1.2, 2.2, 3.1, 3.2 (HIGH IMPACT - should complete)
**P2 items**: 1.3, 3.3 (NICE TO HAVE)

---

## Dependencies

```
Phase 0: Current State Assessment (RUN FIRST)
└── Determines which phases to skip

Phase 1: Foundation (if needed)
├── 1A: Create config utility (src/utils/config.py)
├── 1B: Add node type conversion (src/models/edge_types.py)
└── 1C: Create test fixtures

Phase 2: Dataset Schema (depends on 1B)
├── 2A: Add node_type_schema to MBADataset
├── 2B: Add node_type_schema to ContrastiveDataset
├── 2C: Add node_type_schema to ScaledMBADataset
└── 2D: Create migration script

Phase 3: Edge Type Consolidation (depends on 1A)
├── 3A: Add edge_type_mode to encoders
├── 3B: Add edge_type_mode to ast_parser
└── 3C: Update configs

Phase 4: Feature Flag Exposure (depends on 3C)
├── 4A: Expose feature flags in configs
└── 4B: Wire config loading to encoder creation

Phase 5: Tests & Documentation (depends on all above)
├── 5A: Verify/extend test_dataset_schema_version.py
├── 5B: Verify/extend test_edge_type_equivalence.py
└── 5C: Documentation updates

Phase 6: Rollback Testing (before production deploy)
└── Verify all rollback mechanisms work
```

---

## Phase 0: Current State Assessment (RUN FIRST)

**Purpose**: Determine what's already implemented to avoid redundant work.

Run each check and mark result:

### Check 1: Config Utility
```bash
# Does src/utils/config.py exist with validate_config_value()?
ls src/utils/config.py 2>/dev/null && grep -q "def validate_config_value" src/utils/config.py && echo "EXISTS" || echo "MISSING"

# Does it have create_encoder_from_config()?
grep -q "def create_encoder_from_config" src/utils/config.py 2>/dev/null && echo "EXISTS" || echo "MISSING"
```
- [ ] If BOTH exist: Skip Phase 1A
- [ ] If MISSING: Implement Phase 1A

### Check 2: Node Type Conversion
```bash
# Does edge_types.py have convert_legacy_node_types()?
grep -q "def convert_legacy_node_types" src/models/edge_types.py && echo "EXISTS" || echo "MISSING"

# Does it have LEGACY_NODE_MAP?
grep -q "LEGACY_NODE_MAP" src/models/edge_types.py && echo "EXISTS" || echo "MISSING"
```
- [ ] If BOTH exist: Skip Phase 1B
- [ ] If MISSING: Implement Phase 1B

### Check 3: MBADataset Schema Support
```bash
# Does MBADataset have node_type_schema parameter?
grep -n "node_type_schema" src/data/dataset.py | head -5

# Does it validate as REQUIRED (no default)?
grep -A5 "def __init__" src/data/dataset.py | grep -q "node_type_schema.*=.*None" && echo "HAS DEFAULT (BAD)" || echo "NO DEFAULT (GOOD)"
```
- [ ] If has `node_type_schema` as REQUIRED: Skip Phase 2A
- [ ] If MISSING or has default: Implement Phase 2A

### Check 4: ContrastiveDataset Schema Support
```bash
# Find ContrastiveDataset class
grep -n "class ContrastiveDataset" src/data/dataset.py

# Check if it has node_type_schema
grep -A30 "class ContrastiveDataset" src/data/dataset.py | grep -q "node_type_schema" && echo "EXISTS" || echo "MISSING"
```
- [ ] If exists: Skip Phase 2B
- [ ] If MISSING: Implement Phase 2B

### Check 5: ScaledMBADataset Schema Support
```bash
# Find ScaledMBADataset class
grep -n "class ScaledMBADataset" src/data/dataset.py

# Check if it has node_type_schema
grep -A30 "class ScaledMBADataset" src/data/dataset.py | grep -q "node_type_schema" && echo "EXISTS" || echo "MISSING"
```
- [ ] If exists: Skip Phase 2C
- [ ] If MISSING: Implement Phase 2C

### Check 6: Edge Type Mode in Encoders
```bash
# Does GGNNEncoder have edge_type_mode?
grep -A20 "class GGNNEncoder" src/models/encoder.py | grep -q "edge_type_mode" && echo "EXISTS" || echo "MISSING"

# Does HGTEncoder validate optimized-only?
grep -A30 "class HGTEncoder" src/models/encoder.py | grep -q "edge_type_mode" && echo "EXISTS" || echo "MISSING"
```
- [ ] If BOTH exist: Skip Phase 3A
- [ ] If MISSING: Implement Phase 3A

### Check 7: Edge Type Mode in ast_parser
```bash
# Does ast_to_graph have edge_type_mode parameter?
grep -A5 "def ast_to_graph" src/data/ast_parser.py | grep -q "edge_type_mode" && echo "EXISTS" || echo "MISSING"
```
- [ ] If exists: Skip Phase 3B
- [ ] If MISSING: Implement Phase 3B

### Check 8: Config Edge Type Mode
```bash
# Do configs have edge_type_mode?
grep -q "edge_type_mode" configs/phase1.yaml && echo "phase1: EXISTS" || echo "phase1: MISSING"
grep -q "edge_type_mode" configs/phase2.yaml && echo "phase2: EXISTS" || echo "phase2: MISSING"
grep -q "edge_type_mode" configs/scaled_model.yaml && echo "scaled: EXISTS" || echo "scaled: MISSING"
```
- [ ] If ALL exist: Skip Phase 3C
- [ ] If ANY missing: Implement Phase 3C

### Check 9: Feature Flags in Configs
```bash
# Check for advanced feature flags
grep -q "use_global_attention" configs/phase1.yaml && echo "global_attn: EXISTS" || echo "global_attn: MISSING"
grep -q "use_path_encoding" configs/phase1.yaml && echo "path_enc: EXISTS" || echo "path_enc: MISSING"
grep -q "operation_aware" configs/phase1.yaml && echo "op_aware: EXISTS" || echo "op_aware: MISSING"
```
- [ ] If ALL exist: Skip Phase 4A
- [ ] If ANY missing: Implement Phase 4A

### Check 10: Test Files
```bash
# Do test files exist?
ls tests/test_dataset_schema_version.py 2>/dev/null && echo "schema tests: EXISTS" || echo "schema tests: MISSING"
ls tests/test_edge_type_equivalence.py 2>/dev/null && echo "edge tests: EXISTS" || echo "edge tests: MISSING"
```
- [ ] If exist: Phase 5A/5B = Verify/extend (not create)
- [ ] If MISSING: Phase 5A/5B = Create

### Check 11: Migration Script
```bash
ls scripts/migrate_legacy_datasets.py 2>/dev/null && echo "EXISTS" || echo "MISSING"
```
- [ ] If exists: Skip Phase 2D
- [ ] If MISSING: Implement Phase 2D

---

## Phase 1: Foundation

### 1A: Create Config Utility
**File**: `src/utils/config.py` (new)
**Skip if**: Phase 0 Check 1 shows EXISTS
**Effort**: 30 min

```python
# Core functions:
# - validate_config_value(config, path, expected_type, default, required)
# - create_encoder_from_config(config) -> Encoder
```

**Implementation**:
1. Create `src/utils/` directory if missing
2. Create `src/utils/__init__.py` with exports
3. Create `src/utils/config.py` with:
   - `validate_config_value()` - type-safe config extraction
   - `create_encoder_from_config()` - encoder factory with validation

### 1B: Add Node Type Conversion
**File**: `src/models/edge_types.py`
**Skip if**: Phase 0 Check 2 shows EXISTS
**Effort**: 20 min

**Implementation** (add after line 149):
```python
LEGACY_NODE_ORDER = ['ADD', 'SUB', 'MUL', 'NEG', 'AND', 'OR', 'XOR', 'NOT', 'VAR', 'CONST']

LEGACY_NODE_MAP: Dict[int, int] = {
    0: 2,   # ADD: 0 -> 2
    1: 3,   # SUB: 1 -> 3
    2: 4,   # MUL: 2 -> 4
    3: 9,   # NEG: 3 -> 9
    4: 5,   # AND: 4 -> 5
    5: 6,   # OR: 5 -> 6
    6: 7,   # XOR: 6 -> 7
    7: 8,   # NOT: 7 -> 8
    8: 0,   # VAR: 8 -> 0
    9: 1,   # CONST: 9 -> 1
}

def convert_legacy_node_types(node_types: torch.Tensor) -> torch.Tensor:
    """Convert legacy node type IDs to current schema."""
    if node_types.numel() == 0:
        return node_types

    min_id, max_id = node_types.min().item(), node_types.max().item()
    if min_id < 0 or max_id > 9:
        raise ValueError(f"Node type IDs must be in [0-9], got [{min_id}, {max_id}]")

    lookup = torch.tensor([LEGACY_NODE_MAP[i] for i in range(10)],
                         dtype=node_types.dtype, device=node_types.device)
    return lookup[node_types]
```

### 1C: Create Test Fixtures
**File**: `tests/fixtures/mock_datasets.py` (new)
**Effort**: 15 min

Create helper functions for generating mock JSONL data for tests.

---

## Phase 2: Dataset Schema Detection (P0)

### 2A: MBADataset Schema Support
**File**: `src/data/dataset.py`
**Skip if**: Phase 0 Check 3 shows REQUIRED parameter exists
**Effort**: 45 min

**Changes to `MBADataset.__init__()`**:
1. Add `node_type_schema: str` parameter (REQUIRED, no default)
2. Add validation:
   ```python
   if node_type_schema is None:
       raise ValueError(
           "node_type_schema is REQUIRED. Specify 'legacy' or 'current'.\n"
           "  - Use 'legacy' for datasets generated before 2026-01-15\n"
           "  - Use 'current' for datasets with schema_version >= 2"
       )
   if node_type_schema not in ["legacy", "current"]:
       raise ValueError(f"node_type_schema must be 'legacy' or 'current', got: {node_type_schema}")
   ```
3. Store `self.node_type_schema` and `self._schema_validated = False`

**Changes to `MBADataset.__getitem__()`**:
1. After graph construction, call `self._validate_node_type_schema(graph.x)` on first batch
2. If `node_type_schema == "legacy"`, call `convert_legacy_node_types(graph.x)`

**Add method `_validate_node_type_schema(node_types)`** (CRITICAL - includes distribution heuristic):
```python
def _validate_node_type_schema(self, node_types: torch.Tensor) -> None:
    """Validate node type IDs with distribution heuristic."""
    if node_types.numel() == 0:
        return

    min_id, max_id = node_types.min().item(), node_types.max().item()
    if min_id < 0 or max_id > 9:
        raise ValueError(f"Node type IDs must be in [0-9], got [{min_id}, {max_id}]")

    # CRITICAL: Distribution heuristic to detect schema mismatch
    type_counts = torch.bincount(node_types, minlength=10)

    if self.node_type_schema == "legacy":
        # Legacy: operators (0-7) should exist, terminals (8-9) should exist
        operator_count = type_counts[0:8].sum().item()
        terminal_count = type_counts[8:10].sum().item()
        if operator_count == 0 or terminal_count == 0:
            raise ValueError(
                f"Unusual node type distribution for legacy schema. "
                f"Operators (0-7): {operator_count}, Terminals (8-9): {terminal_count}. "
                f"Verify dataset was generated with legacy schema."
            )
    elif self.node_type_schema == "current":
        # Current: terminals (0-1) should exist, operators (2-9) should exist
        terminal_count = type_counts[0:2].sum().item()
        operator_count = type_counts[2:10].sum().item()
        if terminal_count == 0 or operator_count == 0:
            raise ValueError(
                f"Unusual node type distribution for current schema. "
                f"Terminals (0-1): {terminal_count}, Operators (2-9): {operator_count}. "
                f"Verify dataset has schema_version >= 2."
            )

    self._schema_validated = True
```

### 2B: ContrastiveDataset Schema Support
**File**: `src/data/dataset.py`
**Skip if**: Phase 0 Check 4 shows EXISTS
**Effort**: 20 min

**Pre-flight**: Locate class first:
```bash
grep -n "class ContrastiveDataset" src/data/dataset.py
```

**If inherits from MBADataset**: Verify `super().__init__()` passes `node_type_schema`
**If standalone**: Apply same pattern as 2A (add parameter, validation, conversion)

### 2C: ScaledMBADataset Schema Support
**File**: `src/data/dataset.py`
**Skip if**: Phase 0 Check 5 shows EXISTS
**Effort**: 20 min

Apply same pattern as 2A. Note: `_build_optimized_graph()` is separate from schema conversion.

### 2D: Migration Script
**File**: `scripts/migrate_legacy_datasets.py` (new)
**Skip if**: Phase 0 Check 11 shows EXISTS
**Effort**: 15 min

**IMPLEMENTATION**: Copy from ACTION_PLAN_CRITICAL_ISSUES.md lines 676-715:
```python
"""
Migrate legacy datasets to current schema.

Converts node type IDs from legacy (pre-2026-01-15) to current format
and adds schema_version field.
"""
import argparse
import json
from pathlib import Path
from src.models.edge_types import LEGACY_NODE_MAP

def migrate_dataset(input_path: str, output_path: str):
    """Convert legacy dataset to current schema."""
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            item = json.loads(line)

            # Update AST if present
            if 'ast' in item and 'nodes' in item['ast']:
                for node in item['ast']['nodes']:
                    old_id = node.get('type_id', node.get('type'))
                    if old_id is not None and isinstance(old_id, int):
                        node['type_id'] = LEGACY_NODE_MAP[old_id]

                # Add schema version
                item['ast']['schema_version'] = 2

            f_out.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    migrate_dataset(args.input, args.output)
    print(f"Migrated {args.input} → {args.output}")
```

---

## Phase 3: Edge Type Consolidation (P0)

### 3A: Encoder Edge Type Mode
**File**: `src/models/encoder.py`
**Skip if**: Phase 0 Check 6 shows EXISTS
**Effort**: 45 min

**GGNNEncoder changes**:
1. Add `edge_type_mode: str = "legacy"` to `__init__()`
2. Store `self.edge_type_mode` and set `num_edge_types` accordingly (6 for legacy, 8 for optimized)
3. Add validation in `forward()`:
   ```python
   if self.edge_type_mode == "legacy":
       assert edge_type.max() < 6, f"Edge type {edge_type.max()} exceeds legacy limit 5"
   else:
       assert edge_type.max() < 8, f"Edge type {edge_type.max()} exceeds optimized limit 7"
   ```

**HGTEncoder changes**:
1. Add `edge_type_mode: str = "optimized"` to `__init__()`
2. Raise ValueError if `edge_type_mode == "legacy"`:
   ```python
   if edge_type_mode == "legacy":
       raise ValueError("HGTEncoder only supports optimized 8-type edges. Use GGNNEncoder for legacy datasets.")
   ```

**RGCNEncoder changes**: Same as HGTEncoder.

### 3B: AST Parser Edge Type Mode
**File**: `src/data/ast_parser.py`
**Skip if**: Phase 0 Check 7 shows EXISTS
**Effort**: 30 min

**Pre-flight check**:
```bash
grep -A5 "def ast_to_graph" src/data/ast_parser.py
grep "edge_type_mode" src/data/ast_parser.py
```

**If edge_type_mode already exists**: Verify it delegates to `ast_to_optimized_graph()` when mode="optimized"

**If MISSING**:
1. Add `edge_type_mode: str = "legacy"` parameter to `ast_to_graph()`
2. If `mode == "optimized"`, delegate to `ast_to_optimized_graph()`
3. Update docstring clarifying legacy vs optimized output

**Changes to `expr_to_graph()`**:
1. Add `edge_type_mode` parameter
2. Pass through to `ast_to_graph()`

### 3C: Config Edge Type Mode
**Files**: `configs/phase1.yaml`, `configs/phase2.yaml`, `configs/scaled_model.yaml`
**Skip if**: Phase 0 Check 8 shows ALL exist
**Effort**: 15 min

**Add to each config** (under `model:` section):
```yaml
edge_type_mode: legacy  # or "optimized" for scaled_model
```

**Do NOT add to `configs/phase3.yaml`** - inherits from Phase 2 checkpoint.

---

## Phase 4: Feature Flag Exposure (P1)

### 4A: Expose Feature Flags in Configs
**Files**: `configs/phase1.yaml`, `configs/phase2.yaml`, `configs/scaled_model.yaml`
**Skip if**: Phase 0 Check 9 shows ALL exist
**Effort**: 20 min

**Add to each config** (under `model:` section):
```yaml
# Advanced encoder features (experimental)
use_global_attention: false
global_attn_interval: 2
global_attn_heads: 8

use_path_encoding: false
path_max_length: 6
path_max_paths: 16
path_injection_interval: 2

operation_aware: false
operation_aware_strict: true
```

**Add comment to `configs/phase3.yaml`**:
```yaml
# Encoder features frozen from Phase 2 checkpoint - do not modify here
```

### 4B: Wire Config Loading
**File**: `scripts/train.py`
**Effort**: 30 min

**Changes**:
1. Import `create_encoder_from_config` from `src.utils.config`
2. Replace hardcoded encoder creation with config-driven factory
3. Add `--validate-schema` flag for dry-run validation
4. Pass `node_type_schema` from config to dataset classes

---

## Phase 5: Tests & Documentation

### 5A: Dataset Schema Tests
**File**: `tests/test_dataset_schema_version.py`
**Action**: VERIFY existing tests, EXTEND if incomplete
**Effort**: 30 min

**Step 1**: Run existing tests:
```bash
pytest tests/test_dataset_schema_version.py -v
```

**Step 2**: Verify coverage against action plan (lines 467-609). Required tests:
1. `test_missing_schema_version_field` - ValueError for None
2. `test_legacy_schema_conversion` - VAR=8 → VAR=0
3. `test_current_schema_passthrough` - no conversion
4. `test_out_of_range_ids` - ValueError for ID > 9
5. `test_schema_validation_on_first_batch` - validation runs once
6. `test_distribution_heuristic_legacy` - detects degenerate distribution
7. `test_distribution_heuristic_current` - detects degenerate distribution

**Step 3**: If any test missing, ADD it (don't rewrite entire file)

### 5B: Edge Type Equivalence Tests
**File**: `tests/test_edge_type_equivalence.py`
**Action**: VERIFY existing tests, EXTEND if incomplete
**Effort**: 20 min

**Step 1**: Run existing tests:
```bash
pytest tests/test_edge_type_equivalence.py -v
```

**Step 2**: Verify coverage. Required tests:
1. `test_graph_equivalence` - compare ast_to_graph() vs ast_to_optimized_graph()
2. `test_edge_type_range` - legacy [0-5], optimized [0-7]

### 5C: Documentation Updates
**Files**: `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/DATASET_SCHEMA.md` (new)
**Effort**: 30 min

**CLAUDE.md updates**:
- Add edge type systems summary (legacy 6-type vs optimized 8-type)
- Add node type schema migration note
- Update Novel Approaches table with config flags

**docs/ARCHITECTURE.md updates**:
- Add Advanced Encoder Features section (global attention, path encoding, operation-aware)
- Add Edge Type Systems section with migration guide

**docs/DATASET_SCHEMA.md** (new):
- Document Version 1 (legacy) and Version 2 (current) schemas
- Include node type ID mappings
- Include migration instructions

---

## Phase 6: Rollback Testing (Before Production Deploy)

**Purpose**: Verify all rollback mechanisms work WITHOUT code changes.

### Test 1: Auto Schema Detection (if implemented)
```python
# If node_type_schema="auto" is supported as fallback
from src.data.dataset import MBADataset

# Create dataset without schema_version field
# Load with auto detection
# Verify correct schema selected
# Verify warning logged
```

### Test 2: Fallback Config Loading
```python
# Temporarily mock create_encoder_from_config to fail
# Verify training script falls back to direct kwargs
# Verify same encoder architecture created
```

### Test 3: Dual Graph Constructors
```bash
# Both functions must produce equivalent graphs
pytest tests/test_edge_type_equivalence.py -v --tb=short

# Run with 1000 samples if test supports it
pytest tests/test_edge_type_equivalence.py -v --samples=1000
```

**Pass/Fail Criteria**: All rollback paths must work WITHOUT code changes.

---

## Verification Checklist

**WINDOWS USERS**: Run all commands in Git Bash, not PowerShell or CMD.

After each phase, run:

```bash
# Syntax check
python -m py_compile src/utils/config.py
python -m py_compile src/models/edge_types.py
python -m py_compile src/data/dataset.py
python -m py_compile src/models/encoder.py
python -m py_compile src/data/ast_parser.py

# YAML validation
for f in configs/phase1.yaml configs/phase2.yaml configs/scaled_model.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f'))"
done

# Unit tests
pytest tests/test_dataset_schema_version.py -v
pytest tests/test_edge_type_equivalence.py -v

# Integration test: Missing schema parameter
python -c "
from src.data.dataset import MBADataset
try:
    MBADataset('x', None, None)  # Should fail: missing node_type_schema
except (ValueError, TypeError) as e:
    print('PASS: Missing schema detected')
"

# Integration test: Config type validation
python -c "
from src.utils.config import validate_config_value

config = {'model': {'hidden_dim': '256'}}  # Wrong type (string not int)
try:
    validate_config_value(config, 'model.hidden_dim', int, required=True)
    assert False
except ValueError as e:
    assert 'wrong type' in str(e).lower() or 'expected' in str(e).lower()
    print('PASS: Config type validation works')
"
```

---

## Execution Order

```
Day 1 (2-4h depending on Phase 0 results):
[ ] Phase 0: Run ALL checks, mark what exists
[ ] Phase 1A: If needed
[ ] Phase 1B: If needed
[ ] Phase 1C: If needed
[ ] Phase 2A: If needed (verify distribution heuristic!)

Day 2 (2-4h):
[ ] Phase 2B: If needed
[ ] Phase 2C: If needed
[ ] Phase 2D: If needed
[ ] Phase 5A: Verify/extend tests

Day 3 (2-3h):
[ ] Phase 3A: If needed
[ ] Phase 3B: If needed
[ ] Phase 3C: If needed
[ ] Phase 5B: Verify/extend tests

Day 4 (2h):
[ ] Phase 4A: If needed
[ ] Phase 4B: If needed
[ ] Phase 5C: Documentation
[ ] Phase 6: Rollback testing
[ ] Final integration tests
```

---

## Rollback Plan

If issues arise:

1. **Dataset schema**: Add `node_type_schema="auto"` fallback that detects from data (emit deprecation warning)
2. **Edge types**: Keep both `ast_to_graph()` and `ast_to_optimized_graph()` as separate functions indefinitely
3. **Config loading**: Fall back to direct kwargs if `create_encoder_from_config()` fails

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing training scripts | Add deprecation warnings first, not errors |
| Dataset loading failures | Clear error messages with schema detection hints |
| Performance regression | Feature flags default to False (existing behavior) |
| Import cycles | Config utility imports encoders lazily inside function |
| Silent schema mismatch | Distribution heuristic catches degenerate cases |
