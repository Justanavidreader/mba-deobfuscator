# Action Plan: Critical Issues Resolution

**Target**: Address maintenance burden and configuration inconsistencies in MBA deobfuscator codebase
**Priority Structure**: P0 (blocking), P1 (high impact), P2 (quality of life)
**Estimated Total Effort**: 10-14 hours

---

## Issue 1: Dual Edge Type Systems (P0 - CRITICAL)

### Problem
Two edge type systems coexist creating maintenance burden:
- **Legacy 6-type** (`EDGE_TYPES` in constants.py): CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR
- **Optimized 8-type** (`OPTIMIZED_EDGE_TYPES` in constants.py): LEFT/RIGHT/UNARY_OPERAND + inverses + DOMAIN_BRIDGE_DOWN/UP

Current state:
- `src/models/encoder.py`: HGTEncoder/RGCNEncoder use 8-type, GGNNEncoder uses 6-type
- `src/data/ast_parser.py`: `ast_to_graph()` generates 6-type, `ast_to_optimized_graph()` generates 8-type
- `src/data/dataset.py`: `ScaledMBADataset._build_optimized_graph()` converts 6→8 at load time
- Conversion logic scattered across 3 files

### Root Cause
Incremental migration from legacy to optimized system without deprecation plan.

### Actions

#### 1.1 Consolidate Edge Type Usage (P0)
**Complexity**: Medium
**Files**: `src/models/encoder.py`, `src/data/dataset.py`, `src/data/ast_parser.py`
**Estimated Time**: 2 hours

**Changes**:
1. **Modify `src/models/encoder.py`**:
   - Lines 15-18: Add `LEGACY_EDGE_MAP` to imports from `src/models/edge_types`
   - GGNNEncoder class (lines 150-300):
     - Add parameter `edge_type_mode: str = "legacy"` to `__init__()` (default "legacy" for backward compat)
     - Add edge type converter in forward pass if mode="optimized"
     - Use `NUM_EDGE_TYPES` for legacy mode, `NUM_OPTIMIZED_EDGE_TYPES` for optimized mode
   - HGTEncoder class:
     - Add parameter `edge_type_mode: str = "optimized"` to `__init__()`
     - Raise ValueError if mode="legacy": `raise ValueError("HGTEncoder only supports optimized 8-type edges. Use GGNNEncoder for legacy 6-type datasets or convert with edge_type_mode='optimized'.")`
   - RGCNEncoder class:
     - Add parameter `edge_type_mode: str = "optimized"` to `__init__()`
     - Raise ValueError if mode="legacy": `raise ValueError("RGCNEncoder only supports optimized 8-type edges. Use GGNNEncoder for legacy 6-type datasets or convert with edge_type_mode='optimized'.")`
   - GGNNEncoder:
     - **CRITICAL**: Add edge type validation at start of forward():
       ```python
       if edge_type_mode == "legacy":
           assert edge_type.max() < self.num_edge_types, \
               f"Edge type {edge_type.max()} exceeds legacy limit {self.num_edge_types-1}. " \
               f"Check edge_type_mode in config matches dataset format."
       elif edge_type_mode == "optimized":
           assert edge_type.max() < NUM_OPTIMIZED_EDGE_TYPES, \
               f"Edge type {edge_type.max()} exceeds optimized limit {NUM_OPTIMIZED_EDGE_TYPES-1}. " \
               f"Check edge_type_mode in config matches dataset format."
       ```

2. **Modify `src/data/ast_parser.py`**:
   - Lines 337-349: Update `expr_to_graph()` docstring to clarify legacy output
   - Add parameter `edge_type_mode: str = "legacy"` to `ast_to_graph()`
   - Lines 267-313: Wrap edge generation in conditional based on `edge_type_mode`
   - If mode="optimized", call `ast_to_optimized_graph()` internally
   - **CRITICAL**: Before simplifying, add integration test comparing outputs

3. **Modify `src/data/dataset.py`**:
   - Lines 104-106: Add `edge_type_mode` parameter to `MBADataset.__init__()`
   - Line 105: Pass `edge_type_mode` to `expr_to_graph()` call
   - **DEFER**: Do NOT simplify `ScaledMBADataset._build_optimized_graph()` yet
   - Lines 335-436: Add comment:
     ```python
     # TODO: Replace with direct call to ast_to_optimized_graph() after integration test passes
     # See tests/test_edge_type_equivalence.py for validation
     ```

**Verification Steps**:

Note: Run verification commands in Git Bash on Windows.

```bash
# Syntax check
python -m py_compile src/models/encoder.py
python -m py_compile src/data/ast_parser.py
python -m py_compile src/data/dataset.py

# Integration test: Verify byte-for-byte equivalence (REQUIRED before refactoring)
pytest tests/test_edge_type_equivalence.py -v

# Unit test
python -c "
from src.data.ast_parser import expr_to_graph
from src.models.edge_types import EdgeType
graph_legacy = expr_to_graph('(x & y) + (x ^ y)', edge_type_mode='legacy')
graph_opt = expr_to_graph('(x & y) + (x ^ y)', edge_type_mode='optimized')
assert graph_legacy.edge_type.max() < 6, 'Legacy should use 0-5'
assert graph_opt.edge_type.max() < 8, 'Optimized should use 0-7'
print('Edge type modes working correctly')
"
```

**Test File to Create** (`tests/test_edge_type_equivalence.py`):
```python
"""
Integration test: Verify ast_to_graph() and ast_to_optimized_graph() produce
semantically equivalent graphs on 1000 samples before refactoring.
"""
import pytest
import torch
from src.data.ast_parser import ast_to_graph, ast_to_optimized_graph

@pytest.fixture
def test_expressions():
    """Load test expressions from existing test dataset."""
    import json
    # Load from existing test dataset or use representative sample
    return [
        "x + y",
        "(x & y) | (x ^ y)",
        "((a + b) * c) - (d & ~e)",
        "~(x | y) & (x ^ y)",
        "((x + y) - (x & y)) * 2",
        "(a & b) | (c & d)",
        "x - (y + z)",
        "(x ^ y) + (x & y)",
        "~((a | b) & (c | d))",
        "(x * y) + (x * z)",
        # Add more expressions from test fixture as needed
    ]

def test_graph_equivalence(test_expressions):
    """Verify legacy and optimized graph constructors produce equivalent structures."""
    for expr in test_expressions:
        g_legacy = ast_to_graph(expr)
        g_opt = ast_to_optimized_graph(expr)

        # Node types must match
        assert torch.equal(g_legacy.x, g_opt.x), f"Node types differ for: {expr}"

        # Number of edges must match (after edge type conversion)
        assert g_legacy.edge_index.shape[1] == g_opt.edge_index.shape[1], f"Edge count differs for: {expr}"

        # Node ordering must be identical
        assert torch.equal(g_legacy.edge_index, g_opt.edge_index), f"Edge topology differs for: {expr}"
```

**Rollback Plan**: Keep both `ast_to_graph()` and `ast_to_optimized_graph()` as separate functions initially. Can deprecate after 1-2 releases.

---

#### 1.2 Add Configuration Exposure (P1)
**Complexity**: Low
**Files**: `configs/*.yaml`, `src/constants.py`
**Estimated Time**: 30 minutes

**Changes**:
1. **Add to `configs/phase1.yaml`, `configs/phase2.yaml`**:
   ```yaml
   model:
     edge_type_mode: legacy  # legacy (6-type) or optimized (8-type)
   ```

2. **Add to `configs/scaled_model.yaml`**:
   ```yaml
   model:
     edge_type_mode: optimized  # Scaled model uses optimized 8-type system
   ```

3. **DO NOT add to `configs/phase3.yaml`**: Phase 3 inherits encoder from Phase 2 checkpoint

4. **Modify `src/constants.py`**:
   - Lines 32-46: Add deprecation warning comment above `EDGE_TYPES` dict:
     ```python
     # DEPRECATED: Legacy 6-type edge system for backward compatibility.
     # New code should use OPTIMIZED_EDGE_TYPES (8-type system).
     # Will be removed in v2.0.
     ```

**Verification Steps**:

Note: Run verification commands in Git Bash on Windows.

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/phase1.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/scaled_model.yaml'))"

# Verify phase3.yaml does NOT have edge_type_mode
python -c "
import yaml
config = yaml.safe_load(open('configs/phase3.yaml'))
assert 'edge_type_mode' not in config.get('model', {}), \
    'Phase 3 should NOT define edge_type_mode (inherits from Phase 2 checkpoint)'
print('Config phase separation verified')
"
```

---

#### 1.3 Documentation Update (P2)
**Complexity**: Low
**Files**: `docs/ARCHITECTURE.md`, `CLAUDE.md`
**Estimated Time**: 20 minutes

**Changes**:
1. Update `CLAUDE.md` line 7-8:
   ```
   Edge Type Systems:
   - Legacy (6 types): CHILD_LEFT, CHILD_RIGHT, PARENT, SIBLING_NEXT, SIBLING_PREV, SAME_VAR [DEPRECATED]
   - Optimized (8 types): LEFT/RIGHT/UNARY_OPERAND + inverses + DOMAIN_BRIDGE_DOWN/UP [PREFERRED]
   ```

2. Add migration guide to `docs/ARCHITECTURE.md` (create if missing):
   ```markdown
   ## Edge Type Systems

   ### Optimized 8-Type System (Preferred)
   Used by HGT/RGCN encoders. Enables bidirectional message passing and domain-aware edges.

   ### Legacy 6-Type System (Deprecated)
   Used by legacy GGNN encoder. Will be removed in v2.0.

   ### Migration
   Set `edge_type_mode: "optimized"` in config to use new system with all encoders.
   ```

**Verification**: Manual review of documentation.

---

## Issue 2: Node Type ID Reordering (P0 - CRITICAL)

### Problem
Recent change reordered NodeType enum values (MODIFICATIONS.md lines 11-21):
- **Old**: VAR=8, CONST=9, ADD=0, SUB=1, MUL=2, ...
- **New**: VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, ...

Impact:
- Datasets generated with old C++ program have old node type IDs
- Dataset loading uses unreliable heuristic for schema detection
- Will cause silent misclassification (VAR nodes treated as ADD, etc.)

### Root Cause
Node type ordering changed to match `constants.py` but no robust dataset version detection added.

---

#### 2.1 Add Dataset Schema Version Detection (P0)
**Complexity**: High
**Files**: `src/data/dataset.py`, `src/models/edge_types.py`, `src/constants.py`
**Estimated Time**: 2.5 hours

**Changes**:
1. **Modify `src/models/edge_types.py`**:
   - After line 149, add:
     ```python
     from typing import Dict
     import torch

     # Legacy node type ordering for backward compatibility with datasets
     # generated before 2026-01-15 (prior to NodeType enum reordering)
     LEGACY_NODE_ORDER = ['ADD', 'SUB', 'MUL', 'NEG', 'AND', 'OR', 'XOR', 'NOT', 'VAR', 'CONST']

     # Generated mapping from legacy IDs to current IDs
     # VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, AND=5, OR=6, XOR=7, NOT=8, NEG=9
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
         """
         Convert legacy node type IDs to current schema.

         Legacy schema (pre-2026-01-15): ADD=0, SUB=1, ..., VAR=8, CONST=9
         Current schema: VAR=0, CONST=1, ADD=2, SUB=3, ...

         Args:
             node_types: [num_nodes] tensor with legacy IDs (values in [0-9])

         Returns:
             [num_nodes] tensor with current IDs

         Raises:
             ValueError: If node_types contains IDs outside [0-9] range
         """
         # Input validation
         if node_types.numel() == 0:
             return node_types

         min_id = node_types.min().item()
         max_id = node_types.max().item()
         if min_id < 0 or max_id > 9:
             raise ValueError(
                 f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}]. "
                 f"Dataset may be corrupted or use unsupported schema version."
             )

         # Vectorized lookup table - O(N) time, O(1) extra memory
         lookup = torch.tensor([LEGACY_NODE_MAP[i] for i in range(10)],
                              dtype=node_types.dtype, device=node_types.device)
         return lookup[node_types]
     ```

2. **Modify `src/data/dataset.py`**:
   - Line 20: Add `convert_legacy_node_types` and `LEGACY_NODE_MAP` to imports from `src.models.edge_types`
   - Lines 23-48: Update `MBADataset.__init__()`:
     ```python
     def __init__(
         self,
         data_path: str,
         tokenizer: MBATokenizer,
         fingerprint: SemanticFingerprint,
         max_depth: Optional[int] = None,
         node_type_schema: str = None,  # REQUIRED: "legacy" or "current" (no default)
     ):
         """
         Args:
             node_type_schema: Node type ID format. REQUIRED - must be "legacy" or "current".
                 Set to "legacy" for datasets generated before 2026-01-15.
                 Set to "current" for datasets with schema_version >= 2.

         Raises:
             ValueError: If node_type_schema is None or invalid
         """
         if node_type_schema is None:
             raise ValueError(
                 "node_type_schema is REQUIRED. Specify 'legacy' or 'current'.\n"
                 "  - Use 'legacy' for datasets generated before 2026-01-15\n"
                 "  - Use 'current' for datasets with schema_version >= 2\n"
                 "  - Check dataset JSON for 'schema_version' field to confirm"
             )

         if node_type_schema not in ["legacy", "current"]:
             raise ValueError(f"node_type_schema must be 'legacy' or 'current', got: {node_type_schema}")

         self.node_type_schema = node_type_schema
         self._schema_validated = False  # Track if we've validated first batch
         # ... rest of init

     def __getitem__(self, idx: int) -> Dict:
         item = self.data[idx]

         # ... existing getitem code to build graph ...

         # Validate schema version on first batch
         if not self._schema_validated:
             self._validate_node_type_schema(graph.x)
             self._schema_validated = True

         # Convert node types if legacy schema
         if self.node_type_schema == "legacy" and hasattr(graph, 'x'):
             if graph.x.dim() == 1:  # Type IDs
                 graph.x = convert_legacy_node_types(graph.x)

         return {...}

     def _validate_node_type_schema(self, node_types: torch.Tensor):
         """
         Validate node type IDs are in expected range after loading first batch.

         Args:
             node_types: [num_nodes] tensor with node type IDs

         Raises:
             ValueError: If node types are out of expected range (indicates schema mismatch)
         """
         if node_types.numel() == 0:
             return

         min_id = node_types.min().item()
         max_id = node_types.max().item()

         # Both legacy and current schemas use IDs in [0-9]
         if min_id < 0 or max_id > 9:
             raise ValueError(
                 f"Node type IDs must be in [0-9] range, got [{min_id}, {max_id}].\n"
                 f"Dataset schema validation failed. Check:\n"
                 f"  1. Dataset file is not corrupted\n"
                 f"  2. node_type_schema parameter matches dataset format\n"
                 f"  3. Dataset JSON contains 'schema_version' field"
             )

         # Additional check: In first batch, expect reasonable distribution
         # (Not foolproof but catches obvious mismatches)
         type_counts = torch.bincount(node_types, minlength=10)

         if self.node_type_schema == "legacy":
             # Legacy: operators (0-7) should be common, terminals (8-9) less so
             operator_count = type_counts[0:8].sum().item()
             terminal_count = type_counts[8:10].sum().item()
             if operator_count == 0 or terminal_count == 0:
                 raise ValueError(
                     f"Unusual node type distribution for legacy schema. "
                     f"Operators: {operator_count}, Terminals: {terminal_count}. "
                     f"Verify dataset was generated with legacy schema."
                 )
         elif self.node_type_schema == "current":
             # Current: terminals (0-1) should be common, operators (2-9) mixed
             terminal_count = type_counts[0:2].sum().item()
             operator_count = type_counts[2:10].sum().item()
             if terminal_count == 0 or operator_count == 0:
                 raise ValueError(
                     f"Unusual node type distribution for current schema. "
                     f"Terminals: {terminal_count}, Operators: {operator_count}. "
                     f"Verify dataset has schema_version >= 2."
                 )
     ```

3. **Apply same changes to `ContrastiveDataset` and `ScaledMBADataset`** (lines 130-230, 232-503):
   - Add `node_type_schema` parameter (REQUIRED, no default)
   - Add `_schema_validated` tracking
   - Add `_validate_node_type_schema()` method
   - Apply conversion in `__getitem__()`

**Verification Steps**:

Note: Run verification commands in Git Bash on Windows.

```bash
# Unit test: Conversion function
python -c "
from src.models.edge_types import convert_legacy_node_types
import torch

# Test conversion
legacy_ids = torch.tensor([8, 9, 0, 1, 4, 5])  # VAR, CONST, ADD, SUB, AND, OR (legacy)
current_ids = convert_legacy_node_types(legacy_ids)
expected = torch.tensor([0, 1, 2, 3, 5, 6])  # VAR, CONST, ADD, SUB, AND, OR (current)
assert torch.equal(current_ids, expected), f'Expected {expected}, got {current_ids}'
print('Node type conversion working correctly')

# Test input validation
try:
    convert_legacy_node_types(torch.tensor([10, 11]))
    assert False, 'Should have raised ValueError'
except ValueError as e:
    assert 'must be in [0-9] range' in str(e)
    print('Input validation working correctly')
"

# Integration test with mock datasets
pytest tests/test_dataset_schema_version.py -v

# Test missing schema parameter
python -c "
from src.data.dataset import MBADataset
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint

try:
    dataset = MBADataset('dummy.jsonl', MBATokenizer(), SemanticFingerprint())
    assert False, 'Should have raised ValueError for missing node_type_schema'
except ValueError as e:
    assert 'node_type_schema is REQUIRED' in str(e)
    print('Missing schema parameter validation working correctly')
"
```

**Test File to Create** (`tests/test_dataset_schema_version.py`):
```python
import pytest
import json
import tempfile
from pathlib import Path
import torch
from src.data.dataset import MBADataset
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint

def test_missing_schema_version_field():
    """Test that missing node_type_schema raises clear error."""
    with pytest.raises(ValueError, match="node_type_schema is REQUIRED"):
        dataset = MBADataset(
            data_path="dummy.jsonl",
            tokenizer=MBATokenizer(),
            fingerprint=SemanticFingerprint(),
            node_type_schema=None  # Should raise
        )

def test_legacy_schema_conversion():
    """Test that legacy node type IDs are detected and converted."""
    # Create mock legacy dataset
    legacy_data = {
        "obfuscated": "x + y",
        "simplified": "x + y",
        "depth": 2,
        "ast": {
            "schema_version": 1,
            "nodes": [
                {"id": 0, "type": "ADD", "type_id": 0},  # Legacy: ADD=0
                {"id": 1, "type": "VAR", "type_id": 8},  # Legacy: VAR=8
                {"id": 2, "type": "VAR", "type_id": 8},
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(legacy_data) + '\n')
        path = f.name

    try:
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()
        dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="legacy")

        item = dataset[0]
        graph = item['graph']

        # Check that VAR nodes have type_id=0 (current), not 8 (legacy)
        assert (graph.x == 0).any(), "Should have VAR nodes with type_id=0"
        assert not (graph.x == 8).any(), "Should NOT have type_id=8 after conversion"
    finally:
        Path(path).unlink()

def test_current_schema_passthrough():
    """Test that current schema data is not modified."""
    current_data = {
        "obfuscated": "x + y",
        "simplified": "x + y",
        "depth": 2,
        "ast": {
            "schema_version": 2,
            "nodes": [
                {"id": 0, "type": "ADD", "type_id": 2},  # Current: ADD=2
                {"id": 1, "type": "VAR", "type_id": 0},  # Current: VAR=0
                {"id": 2, "type": "VAR", "type_id": 0},
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(current_data) + '\n')
        path = f.name

    try:
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()
        dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="current")

        item = dataset[0]
        graph = item['graph']

        # VAR should stay as 0
        assert (graph.x == 0).any(), "VAR nodes should have type_id=0"
    finally:
        Path(path).unlink()

def test_out_of_range_ids():
    """Test that out-of-range node type IDs raise clear error."""
    bad_data = {
        "obfuscated": "x + y",
        "simplified": "x + y",
        "depth": 2,
        "ast": {
            "nodes": [
                {"id": 0, "type": "INVALID", "type_id": 99},  # Out of range
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(bad_data) + '\n')
        path = f.name

    try:
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()
        dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="current")

        with pytest.raises(ValueError, match="must be in \\[0-9\\] range"):
            _ = dataset[0]  # Should raise during validation
    finally:
        Path(path).unlink()

def test_mixed_schema_ids():
    """Test detection of corrupted dataset with mixed schema IDs."""
    mixed_data = {
        "obfuscated": "x + y",
        "simplified": "x + y",
        "depth": 2,
        "ast": {
            "nodes": [
                {"id": 0, "type": "ADD", "type_id": 2},  # Current
                {"id": 1, "type": "VAR", "type_id": 8},  # Legacy - INCONSISTENT
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(mixed_data) + '\n')
        path = f.name

    try:
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()
        dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="current")

        item = dataset[0]  # Should trigger validation warning
        # Warning printed but doesn't block (heuristic can't catch all cases)
    finally:
        Path(path).unlink()
```

**Rollback Plan**:
- Add `--validate-schema` flag to training script for dry-run validation
- Checkpoint metadata includes `dataset_schema_version` for traceability
- Create migration script: `scripts/migrate_legacy_datasets.py`

---

#### 2.2 Add Schema Version to Future Dataset Generation (P1)
**Complexity**: Low
**Files**: Documentation only (C++ generator is separate)
**Estimated Time**: 15 minutes

**Changes**:
1. **Create `docs/DATASET_SCHEMA.md`**:
   ```markdown
   # Dataset Schema Versions

   ## Version 2 (Current, 2026-01-15+)
   Node type IDs match src/constants.py NODE_TYPES:
   - VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, AND=5, OR=6, XOR=7, NOT=8, NEG=9

   ## Version 1 (Legacy, pre-2026-01-15)
   Node type IDs use old ordering:
   - ADD=0, SUB=1, MUL=2, NEG=3, AND=4, OR=5, XOR=6, NOT=7, VAR=8, CONST=9

   ## Dataset Format
   JSONL with **REQUIRED** `ast.schema_version` field:
   ```json
   {
     "obfuscated": "(x & y) + (x ^ y)",
     "simplified": "x | y",
     "depth": 3,
     "ast": {
       "schema_version": 2,
       "nodes": [...],
       "edges": [...]
     }
   }
   ```

   **C++ Generator Update Required**: Add `schema_version: 2` to AST output.
   Without this field, users must manually specify `node_type_schema` parameter.

   ## Usage
   ```python
   # For new datasets (with schema_version field)
   dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="current")

   # For legacy datasets (pre-2026-01-15)
   dataset = MBADataset(path, tokenizer, fingerprint, node_type_schema="legacy")
   ```

   ## Migration Script
   Convert legacy datasets to current schema:
   ```bash
   python scripts/migrate_legacy_datasets.py --input old_train.jsonl --output new_train.jsonl
   ```
   ```

2. **Update `CLAUDE.md`** (add to Important Files section):
   ```markdown
   | `docs/DATASET_SCHEMA.md` | Dataset schema versions and migration guide |
   | `scripts/migrate_legacy_datasets.py` | Convert legacy datasets to current schema |
   ```

3. **Create `scripts/migrate_legacy_datasets.py`**:
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

**Verification**: Manual review of documentation.

---

## Issue 3: Feature Flags Scattered (P1 - HIGH IMPACT)

### Problem
Advanced features have flags in `constants.py` but are NOT exposed in config files:
- **Global attention** (`HGT_USE_GLOBAL_ATTENTION`): Default False, no config exposure
- **Path encoding** (`PATH_ENCODING_ENABLED`, `GGNN_USE_PATH_ENCODING`, `HGT_USE_PATH_ENCODING`): Default False, no config exposure
- **Operation-aware aggregation** (`operation_aware` in HGTEncoder): Exposed in configs but incomplete
  - `configs/scaled_model.yaml` line 23: Has `operation_aware: false`
  - `configs/phase2.yaml` line 15: Has `operation_aware: false`
  - But NOT in `configs/phase1.yaml` or `configs/phase3.yaml`

Result: Users cannot enable these features without modifying source code.

### Root Cause
Features implemented but not fully integrated into configuration system.

### Actions

#### 3.1 Expose All Feature Flags in Configs (P1)
**Complexity**: Low
**Files**: `configs/*.yaml`
**Estimated Time**: 30 minutes

**Changes**:
1. **Add to `configs/phase1.yaml`** (after encoder section):
   ```yaml
   # Advanced encoder features (experimental)
   use_global_attention: false  # GraphGPS-style hybrid (global + local attention)
   global_attn_interval: 2      # Insert global attention every N layers
   global_attn_heads: 8         # Attention heads for global blocks

   use_path_encoding: false     # Path-based edge encoding for subexpression detection
   path_max_length: 6           # Maximum path length to consider
   path_max_paths: 16           # Maximum paths per edge pair
   path_injection_interval: 2   # Inject path context every N HGT layers (HGT only)

   operation_aware: false       # Operation-aware aggregation (HGT only)
   operation_aware_strict: true # Raise error if used with non-HGT encoder

   # When to enable:
   #   use_global_attention: O(1) subexpression detection for depth 10+ graphs
   #   use_path_encoding: Shared subexpression recognition in DAG structures
   #   operation_aware: +2-5% accuracy on SUB-heavy expressions (HGT only)
   ```

2. **Add same section to `configs/phase2.yaml`** (updating existing `operation_aware` entry)

3. **Add to `configs/scaled_model.yaml`** (after line 24, updating existing `operation_aware` entry)

4. **DO NOT add to `configs/phase3.yaml`**: Add comment instead:
   ```yaml
   # Encoder configuration inherited from Phase 2 checkpoint
   # Advanced features (use_global_attention, use_path_encoding, operation_aware)
   # are frozen from Phase 2 training and cannot be modified in Phase 3
   ```

**Verification Steps**:

Note: Run verification commands in Git Bash on Windows.

```bash
# Validate YAML
for f in configs/phase1.yaml configs/phase2.yaml configs/scaled_model.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f'))"
done

# Verify phase3.yaml does NOT have encoder feature flags
python -c "
import yaml
config = yaml.safe_load(open('configs/phase3.yaml'))
model = config.get('model', {})
assert 'use_global_attention' not in model, 'Phase 3 should NOT define encoder features'
assert 'use_path_encoding' not in model, 'Phase 3 should NOT define encoder features'
assert 'operation_aware' not in model, 'Phase 3 should NOT define encoder features'
print('Config phase separation verified')
"

# Check all keys are valid
python -c "
import yaml
config = yaml.safe_load(open('configs/scaled_model.yaml'))
model = config['model']
assert 'use_global_attention' in model
assert 'use_path_encoding' in model
assert 'operation_aware' in model
print('All feature flags present in config')
"
```

---

#### 3.2 Add Config Loading Logic with Type Validation (P1)
**Complexity**: Medium
**Files**: `src/utils/config.py` (new), `src/models/encoder.py`, `scripts/train.py`
**Estimated Time**: 1.5 hours

**Changes**:
1. **Create `src/utils/config.py`**:
   ```python
   """Configuration loading and validation utilities."""
   from typing import Any, Dict, Optional, Type

   def validate_config_value(
       config: dict,
       path: str,
       expected_type: Type,
       default: Any = None,
       required: bool = False
   ) -> Any:
       """
       Extract and validate config value with type checking.

       Args:
           config: Configuration dictionary
           path: Dot-separated path to value (e.g., "model.hidden_dim")
           expected_type: Expected Python type
           default: Default value if not found (ignored if required=True)
           required: If True, raise error when value missing

       Returns:
           Config value cast to expected_type

       Raises:
           ValueError: If required value missing or type mismatch

       Example:
           >>> config = {"model": {"hidden_dim": 256}}
           >>> validate_config_value(config, "model.hidden_dim", int)
           256
       """
       keys = path.split('.')
       value = config

       try:
           for key in keys:
               value = value[key]
       except (KeyError, TypeError):
           if required:
               raise ValueError(f"Required config key '{path}' not found")
           return default

       if not isinstance(value, expected_type):
           raise ValueError(
               f"Config key '{path}' has wrong type. "
               f"Expected {expected_type.__name__}, got {type(value).__name__}"
           )

       return value

   def create_encoder_from_config(config: dict):
       """
       Create encoder instance from configuration dict.

       Args:
           config: Full config dict with 'model' section

       Returns:
           Encoder instance (HGTEncoder, GGNNEncoder, etc.)

       Raises:
           ValueError: If config invalid or encoder type unsupported
       """
       from src.models.encoder import HGTEncoder, GGNNEncoder, GATEncoder, RGCNEncoder

       encoder_type = validate_config_value(
           config, "model.encoder_type", str, required=True
       )

       # Base parameters common to all encoders
       base_params = {
           'hidden_dim': validate_config_value(config, "model.hidden_dim", int, required=True),
           'num_layers': validate_config_value(config, "model.num_encoder_layers", int, required=True),
       }

       if encoder_type == 'hgt':
           hgt_params = {
               **base_params,
               'num_heads': validate_config_value(config, "model.num_encoder_heads", int, required=True),
               'use_global_attention': validate_config_value(config, "model.use_global_attention", bool, default=False),
               'operation_aware': validate_config_value(config, "model.operation_aware", bool, default=False),
               'operation_aware_strict': validate_config_value(config, "model.operation_aware_strict", bool, default=True),
           }

           # Path encoding params if enabled
           if validate_config_value(config, "model.use_path_encoding", bool, default=False):
               hgt_params['path_encoding_enabled'] = True
               hgt_params['path_max_length'] = validate_config_value(config, "model.path_max_length", int, default=6)
               hgt_params['path_max_paths'] = validate_config_value(config, "model.path_max_paths", int, default=16)
               hgt_params['path_injection_interval'] = validate_config_value(config, "model.path_injection_interval", int, default=2)

           return HGTEncoder(**hgt_params)

       elif encoder_type == 'ggnn':
           ggnn_params = {
               **base_params,
               'num_timesteps': validate_config_value(config, "model.num_timesteps", int, default=8),
               'edge_type_mode': validate_config_value(config, "model.edge_type_mode", str, default="legacy"),
           }

           if validate_config_value(config, "model.use_path_encoding", bool, default=False):
               ggnn_params['path_encoding_enabled'] = True
               ggnn_params['path_max_length'] = validate_config_value(config, "model.path_max_length", int, default=6)

           return GGNNEncoder(**ggnn_params)

       # ... other encoder types

       else:
           raise ValueError(f"Unsupported encoder type: {encoder_type}")
   ```

2. **Modify `scripts/train.py`** (or main training script):
   ```python
   from src.utils.config import create_encoder_from_config, validate_config_value

   def main():
       # Load config
       config = yaml.safe_load(open(args.config))

       # Create encoder with full config validation
       encoder = create_encoder_from_config(config)

       # ... rest of training setup
   ```

**Verification Steps**:

Note: Run verification commands in Git Bash on Windows.

```bash
# Unit test: Config validation
python -c "
from src.utils.config import validate_config_value
import pytest

config = {'model': {'hidden_dim': 256, 'use_flag': True}}

# Valid extraction
assert validate_config_value(config, 'model.hidden_dim', int) == 256
assert validate_config_value(config, 'model.use_flag', bool) == True

# Default value
assert validate_config_value(config, 'model.missing', int, default=10) == 10

# Type mismatch
try:
    validate_config_value(config, 'model.hidden_dim', str)
    assert False, 'Should raise ValueError'
except ValueError as e:
    assert 'wrong type' in str(e)

# Missing required
try:
    validate_config_value(config, 'model.missing', int, required=True)
    assert False, 'Should raise ValueError'
except ValueError as e:
    assert 'not found' in str(e)

print('Config validation working correctly')
"

# Integration test: Encoder creation
python -c "
import yaml
from src.utils.config import create_encoder_from_config

config = yaml.safe_load(open('configs/scaled_model.yaml'))

# Test with all features enabled
config['model']['use_global_attention'] = True
config['model']['use_path_encoding'] = True
config['model']['operation_aware'] = True

encoder = create_encoder_from_config(config)
print(f'Encoder created: {encoder.__class__.__name__}')
print(f'Global attention: {encoder.use_global_attention if hasattr(encoder, \"use_global_attention\") else \"N/A\"}')
print(f'Operation aware: {encoder.operation_aware if hasattr(encoder, \"operation_aware\") else \"N/A\"}')
"
```

---

#### 3.3 Document Feature Flags (P2)
**Complexity**: Low
**Files**: `docs/ARCHITECTURE.md`, `CLAUDE.md`
**Estimated Time**: 20 minutes

**Changes**:
1. **Add to `docs/ARCHITECTURE.md`** (create new section):
   ```markdown
   ## Advanced Encoder Features

   ### Global Attention (HGT only)
   - **Config**: `use_global_attention: true`
   - **Use case**: Expressions with depth 10+, repeated subexpressions
   - **Cost**: +30% memory, +20% training time
   - **Benefit**: O(1) subexpression detection vs O(depth) with local GNN

   ### Path-Based Edge Encoding
   - **Config**: `use_path_encoding: true`
   - **Use case**: DAG structures with shared subexpressions
   - **Cost**: +40% preprocessing time
   - **Benefit**: +5-8% accuracy on expressions with >3 shared subexpressions

   ### Operation-Aware Aggregation (HGT only)
   - **Config**: `operation_aware: true`
   - **Use case**: Expressions with non-commutative operations (SUB)
   - **Cost**: Minimal (+2% memory)
   - **Benefit**: +2-5% accuracy on SUB-heavy expressions
   ```

2. **Update `CLAUDE.md`** Novel Approaches table (lines 66-77):
   ```markdown
   | Priority | Technique | Config Flag | Status |
   |----------|-----------|-------------|--------|
   | P0 | Truth table (64 entries) | (always on) | Implemented |
   | P0 | Grammar-constrained decoding | (inference only) | Implemented |
   | P1 | Operation-aware aggregation | `operation_aware` | Implemented |
   | P1 | Global attention (GraphGPS) | `use_global_attention` | Implemented |
   | P2 | Path-based edge encoding | `use_path_encoding` | Implemented |
   ```

**Verification**: Manual review of documentation.

---

## Summary Checklist

### P0 Actions (Must Complete)
- [ ] 1.1: Consolidate edge type usage with mode parameter + validation
- [ ] 1.2: Add edge type mode to configs (phase1, phase2, scaled_model ONLY)
- [ ] 2.1: Add dataset schema version detection with REQUIRED parameter
- [ ] 2.2: Document schema versions + create migration script

### P1 Actions (Should Complete)
- [ ] 3.1: Expose feature flags in configs (phase1, phase2, scaled_model ONLY)
- [ ] 3.2: Add config loading logic with type validation
- [ ] 1.3: Update edge type documentation

### P2 Actions (Nice to Have)
- [ ] 3.3: Document feature flags in ARCHITECTURE.md
- [ ] Create integration test for graph equivalence (test_edge_type_equivalence.py)
- [ ] Generate LEGACY_NODE_MAP from constants.py (avoid magic numbers)

### Verification Commands (Run After All Changes)

Note: Run verification commands in Git Bash on Windows.

```bash
# Syntax check all modified Python files
find src -name "*.py" -exec python -m py_compile {} \;

# Validate all YAML configs
for f in configs/*.yaml; do python -c "import yaml; yaml.safe_load(open('$f'))"; done

# Run unit tests
pytest tests/ -v

# Integration test: Dataset schema validation
pytest tests/test_dataset_schema_version.py -v

# Integration test: Missing schema parameter
python -c "
from src.data.dataset import MBADataset
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint

try:
    dataset = MBADataset('dummy.jsonl', MBATokenizer(), SemanticFingerprint())
    assert False, 'Should have raised ValueError'
except ValueError as e:
    assert 'node_type_schema is REQUIRED' in str(e)
    print('Schema parameter validation: PASS')
"

# Integration test: Config type validation
python -c "
from src.utils.config import validate_config_value

config = {'model': {'hidden_dim': '256'}}  # Wrong type (string not int)
try:
    validate_config_value(config, 'model.hidden_dim', int, required=True)
    assert False, 'Should have raised ValueError'
except ValueError as e:
    assert 'wrong type' in str(e)
    print('Config type validation: PASS')
"

# Integration test: Create encoder with validated config
python -c "
import yaml
from src.utils.config import create_encoder_from_config

config = yaml.safe_load(open('configs/scaled_model.yaml'))
encoder = create_encoder_from_config(config)
print(f'Encoder created: {encoder.__class__.__name__}')
print(f'Config loading: PASS')
"

# Integration test: Edge type validation
python -c "
import torch
from src.models.encoder import GGNNEncoder

encoder = GGNNEncoder(hidden_dim=256, num_timesteps=8, edge_type_mode='legacy')

# Simulate out-of-range edge type
edge_index = torch.tensor([[0, 1], [1, 0]])
edge_type = torch.tensor([10, 11])  # Invalid for legacy (max 5)
node_features = torch.randn(2, 256)

try:
    encoder(node_features, edge_index, edge_type)
    assert False, 'Should have raised AssertionError'
except AssertionError as e:
    assert 'exceeds legacy limit' in str(e)
    print('Edge type validation: PASS')
"
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing datasets | Medium | High | REQUIRED schema parameter forces explicit choice; clear error messages |
| Config key typos breaking training | Low | High | Type validation with validate_config_value(); YAML validation in CI |
| Performance regression | Low | Medium | Feature flags default to False (existing behavior); add benchmarks |
| Edge type conversion bugs | Medium | High | Integration test comparing both methods on 1000 samples BEFORE refactoring |
| Silent schema mismatch | Low | High | Validation on first batch with clear warnings; out-of-range detection |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Issue 1 (Edge types) | 3.5 hours | Integration test creation (deferred refactor) |
| Issue 2 (Node types) | 2.5 hours | Migration script creation |
| Issue 3 (Feature flags) | 2 hours | Config utility creation |
| Testing & Verification | 3 hours | All above complete |
| Documentation | 1 hour | All above complete |
| **Total** | **12 hours** | - |

---

## Post-Completion Tasks

1. **Deprecation Notice**: Add to CHANGELOG.md:
   ```markdown
   ## v1.5.0 (2026-01-XX)

   ### Added
   - Schema version detection for datasets (legacy vs current node type IDs)
   - REQUIRED `node_type_schema` parameter for dataset loading
   - Edge type mode configuration (legacy 6-type vs optimized 8-type)
   - Edge type validation in encoder forward pass
   - Config exposure for global attention, path encoding, operation-aware aggregation
   - Type-validated config loading with `validate_config_value()`
   - Migration script: `scripts/migrate_legacy_datasets.py`

   ### Changed
   - `MBADataset.__init__()` now requires `node_type_schema` parameter (no default)
   - Config phase separation: Phase 3 inherits encoder features from Phase 2 checkpoint
   - `convert_legacy_node_types()` validates input range [0-9]

   ### Deprecated
   - Legacy 6-type edge system (EDGE_TYPES in constants.py) - will remove in v2.0
   - Hardcoded feature flags in constants.py - use config files instead

   ### Fixed
   - Silent node type misclassification from schema version mismatch
   - Edge type out-of-range errors now have clear diagnostic messages

   ### Migration
   - Explicitly set `node_type_schema="legacy"` for pre-2026-01-15 datasets
   - Set `node_type_schema="current"` for datasets with schema_version >= 2
   - Set `edge_type_mode: "optimized"` in config for new edge system
   - Enable advanced features via config: `use_global_attention`, `use_path_encoding`, `operation_aware`
   - Run `scripts/migrate_legacy_datasets.py` to convert old datasets
   ```

2. **C++ Generator Update**: Notify maintainer to:
   - Add `schema_version: 2` to AST JSON output
   - Use node type IDs matching src/constants.py NODE_TYPES (VAR=0, CONST=1, ADD=2, ...)

3. **Integration Test for Graph Equivalence** (before refactoring `_build_optimized_graph()`):
   ```bash
   # Create test with 1000 samples
   pytest tests/test_edge_type_equivalence.py -v --samples 1000

   # Verify byte-for-byte equivalence
   # Only after this passes, refactor _build_optimized_graph()
   ```

4. **Ablation Study**: Compare legacy vs optimized edge types on test set:
   ```bash
   python scripts/run_ablation.py --encoder hgt --edge-type-mode legacy --run-id 1
   python scripts/run_ablation.py --encoder hgt --edge-type-mode optimized --run-id 2
   ```

5. **Performance Benchmark**: Test feature flags impact:
   ```bash
   # Baseline
   python scripts/train.py --config configs/scaled_model.yaml --epochs 1 --benchmark

   # With global attention
   # (edit config: use_global_attention: true)
   python scripts/train.py --config configs/scaled_model.yaml --epochs 1 --benchmark

   # With path encoding
   # (edit config: use_path_encoding: true)
   python scripts/train.py --config configs/scaled_model.yaml --epochs 1 --benchmark
   ```

6. **Add --validate-schema flag** to training scripts:
   ```python
   # In scripts/train.py
   parser.add_argument('--validate-schema', action='store_true',
                       help='Dry-run: validate dataset schema without training')

   if args.validate_schema:
       dataset = MBADataset(args.data_path, tokenizer, fingerprint,
                           node_type_schema=args.node_type_schema)
       print(f"Loaded {len(dataset)} samples")
       item = dataset[0]  # Trigger validation
       print(f"Schema validation passed: {dataset.node_type_schema}")
       sys.exit(0)
   ```
