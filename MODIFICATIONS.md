# Modifications Summary for Quality Review

## Overview
Fixes for 16 inconsistencies across 11 categories identified in codebase analysis.
All changes ensure consistency between constants.py (authoritative source) and dependent modules.

---

## 1. src/models/edge_types.py

### Changes Made:
- **NodeType enum reordered** (lines 67-112) to match constants.py NODE_TYPES:
  - Old: ADD=0, SUB=1, MUL=2, NEG=3, AND=4, OR=5, XOR=6, NOT=7, VAR=8, CONST=9
  - New: VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, AND=5, OR=6, XOR=7, NOT=8, NEG=9
- **Updated helper methods** with new IDs:
  - `is_arithmetic()`: [0,1,2,3] → [2,3,4,9]
  - `is_boolean()`: [4,5,6,7] → [5,6,7,8]
  - `is_unary()`: [3,7] → [8,9]
  - `is_binary()`: [0,1,2,4,5,6] → [2,3,4,5,6,7]
  - `is_terminal()`: [8,9] → [0,1]
- **Added docstring note** that ordering MUST match constants.py

### Rationale:
constants.py is imported more widely and encoder.py VALID_TRIPLETS already use this ordering.

---

## 2. src/data/ast_parser.py

### Changes Made:
- **Line 8**: Fixed comment "7 types" → "8 types"
- **Lines 383-392**: Fixed docstring to use correct edge type names:
  - "7-type" → "8-type"
  - "PARENT_OF_LEFT/RIGHT/UNARY" → "LEFT/RIGHT/UNARY_OPERAND_INV"
  - "DOMAIN_BRIDGE" → "DOMAIN_BRIDGE_DOWN, DOMAIN_BRIDGE_UP"
- **Line 431**: `EdgeType.PARENT_OF_UNARY` → `EdgeType.UNARY_OPERAND_INV`
- **Line 440**: `EdgeType.PARENT_OF_LEFT` → `EdgeType.LEFT_OPERAND_INV`
- **Line 446**: `EdgeType.PARENT_OF_RIGHT` → `EdgeType.RIGHT_OPERAND_INV`
- **Line 458**: `EdgeType.DOMAIN_BRIDGE` → `EdgeType.DOMAIN_BRIDGE_DOWN`
- **Added new functions** (lines 485-535):
  - `node_types_to_features()`: Converts node type IDs to dense feature vectors
  - `convert_graph_for_encoder()`: Adapts graph format based on encoder requirements

### Rationale:
EdgeType enum uses `*_INV` naming, not `PARENT_OF_*`. DOMAIN_BRIDGE doesn't exist (split into DOWN/UP).
New adapter functions bridge optimized graphs with legacy encoders.

---

## 3. src/data/dataset.py

### Changes Made:
- **Line 17**: Added import `from src.constants import EDGE_TYPES`
- **Line 320**: Magic numbers → named constants:
  - `if edge_type in [0, 1]` → `if edge_type in [EDGE_TYPES['CHILD_LEFT'], EDGE_TYPES['CHILD_RIGHT']]`
- **Line 404**: `if edge_type_idx == 0` → `if edge_type_idx == EDGE_TYPES['CHILD_LEFT']`
- **Line 406**: `EdgeType.PARENT_OF_LEFT` → `EdgeType.LEFT_OPERAND_INV`
- **Line 407**: `elif edge_type_idx == 1` → `elif edge_type_idx == EDGE_TYPES['CHILD_RIGHT']`
- **Line 409**: `EdgeType.PARENT_OF_RIGHT` → `EdgeType.RIGHT_OPERAND_INV`
- **Line 430**: `EdgeType.DOMAIN_BRIDGE` → `EdgeType.DOMAIN_BRIDGE_DOWN`

### Rationale:
Replaced magic numbers with named constants for maintainability.
Fixed non-existent enum values that would cause AttributeError at runtime.

---

## 4. src/constants.py

### Changes Made:
- **Lines 32-34**: Added deprecation comment on legacy EDGE_TYPES section
- **Lines 48-60**: Added new `OPTIMIZED_EDGE_TYPES` dict with 8 edge types:
  ```python
  OPTIMIZED_EDGE_TYPES: Dict[str, int] = {
      'LEFT_OPERAND': 0,
      'RIGHT_OPERAND': 1,
      'UNARY_OPERAND': 2,
      'LEFT_OPERAND_INV': 3,
      'RIGHT_OPERAND_INV': 4,
      'UNARY_OPERAND_INV': 5,
      'DOMAIN_BRIDGE_DOWN': 6,
      'DOMAIN_BRIDGE_UP': 7,
  }
  ```

### Rationale:
Consolidates both edge type systems in one authoritative location.
Legacy EDGE_TYPES kept for backward compatibility with existing datasets.

---

## 5. src/models/encoder.py

### Changes Made:
- **Line 18**: Added `NUM_OPTIMIZED_EDGE_TYPES` to imports
- **Line 406**: HGTEncoder default `num_edge_types: int = 8` → `num_edge_types: int = NUM_OPTIMIZED_EDGE_TYPES`
- **Line 553**: RGCNEncoder default `num_edge_types: int = 8` → `num_edge_types: int = NUM_OPTIMIZED_EDGE_TYPES`
- **VALID_TRIPLETS** (lines 345-397): Already fixed in prior session - 312 triplets with correct node type IDs matching constants.py

### Rationale:
HGT/RGCN encoders need 8 edge types. Using constant instead of magic number.
GGNN correctly uses NUM_EDGE_TYPES (6) for legacy datasets.

---

## Files Modified Summary

| File | Changes | Severity |
|------|---------|----------|
| src/models/edge_types.py | NodeType enum reordered, helpers updated | CRITICAL |
| src/data/ast_parser.py | Fixed 4 non-existent EdgeType refs, added adapters | CRITICAL |
| src/data/dataset.py | Fixed 3 non-existent EdgeType refs, named constants | CRITICAL |
| src/constants.py | Added OPTIMIZED_EDGE_TYPES, deprecation note | HIGH |
| src/models/encoder.py | Import + use NUM_OPTIMIZED_EDGE_TYPES | MEDIUM |

## Files NOT Modified (no changes needed)
- src/models/full_model.py - Already uses NUM_OPTIMIZED_EDGE_TYPES correctly
- src/models/decoder.py - No edge type dependencies
- src/data/tokenizer.py - No edge/node type dependencies
- src/data/fingerprint.py - No edge/node type dependencies

---

## Verification Commands

```bash
# Syntax check all modified files
python -m py_compile src/models/edge_types.py
python -m py_compile src/data/ast_parser.py
python -m py_compile src/data/dataset.py
python -m py_compile src/constants.py
python -m py_compile src/models/encoder.py

# Run tests if available
pytest tests/ -v
```

## Breaking Changes

1. **NodeType enum values changed** - Any code using hardcoded NodeType integer values will break
2. **NODE_TYPE_MAP values changed** - Values now match constants.py NODE_TYPES

## Migration Notes

Code using `NodeType.VAR` (was 8, now 0) or similar must be updated.
Code importing from edge_types.py should work correctly as long as it uses enum names, not values.
