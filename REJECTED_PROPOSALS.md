# Rejected Optimization Proposals

This document records optimization proposals that were considered and rejected, along with rationale. Purpose: prevent re-proposing the same ideas.

---

## Redundant Edge Pruning (Rejected: 2025-01-17)

### Proposal Summary
Remove "redundant" edges from MBA expression graphs:
- Remove all SIBLING edges (SIBLING_NEXT, SIBLING_PREV)
- Reduce SAME_VAR edges from complete graphs to spanning trees

**Claimed benefits**:
- 20-40% edge reduction
- 10-20% training speedup
- Zero information loss

### Rejection Rationale

**Critical flaw**: Proposal assumes both edge type systems have redundant edges, but this is false.

#### Edge Type System Reality

| Edge System | SIBLING Edges | SAME_VAR Edges | Pruning Benefit |
|-------------|---------------|----------------|-----------------|
| **Legacy (6-type)** | ✓ Present | ✓ Present | 20-40% reduction possible |
| **Optimized (8-type)** | ✗ Never created | ✗ Never created | **0% reduction** (nothing to prune) |

**Key findings from quality review**:

1. **Optimized system already excludes redundant edges** (`ast_to_optimized_graph` in `ast_parser.py:449-544` never creates SIBLING or SAME_VAR edges)

2. **ScaledMBADataset uses optimized system** - Now default for all training (Phase 1, 2, 3, GMN, Semantic HGT)

3. **Legacy mode is deprecated** - Only kept for backward compatibility with old datasets

4. **Production configuration gets zero benefit**:
   - Base model (15M): Uses ScaledMBADataset → optimized edges → no pruning possible
   - Scaled model (360M): Uses ScaledMBADataset → optimized edges → no pruning possible
   - All configs updated to `dataset_type: scaled` → optimized edges → no pruning possible

#### Additional Issues Found

Even if legacy mode were important, implementation has problems:

1. **Pre-computed AST conflict**: ScaledMBADataset loads C++-generated ASTs; Python pruning post-load creates metadata inconsistency

2. **Non-determinism**: BFS spanning tree construction uses set iteration (non-deterministic) → non-reproducible training

3. **Missing validation**: Doesn't validate edge type IDs against expected ranges (violates project pattern)

4. **Testing gaps**: No encoder integration tests (critical since this affects GNN message passing)

#### Cost-Benefit Analysis

**If implemented for legacy mode only**:
- **Benefit**: 20-40% edge reduction for deprecated edge system used by ~0% of production training
- **Cost**:
  - New module (~400 lines)
  - Dataset integration complexity
  - C++/Python coordination for pre-computed ASTs
  - Non-determinism fixes
  - Encoder integration tests
  - Maintenance burden

**Verdict**: Cost >> Benefit. Not worth implementing.

### Lessons Learned

1. **Verify optimization applies to production config** - Don't optimize deprecated code paths
2. **Check if optimization already exists** - Optimized edge system was designed to eliminate redundancy
3. **Understand dual-system architecture** - Legacy vs optimized edge types have different characteristics
4. **ScaledMBADataset is now baseline** - All optimizations must benefit the scaled dataset path

### Alternative Approaches (If Edge Count Matters)

If reducing edge count is critical:

1. **Migrate remaining legacy datasets to optimized** - Already has minimal edges
2. **Improve ScaledMBADataset subexpression sharing** - Already provides 3× edge reduction
3. **Graph sampling for very large expressions** - Sample neighborhoods instead of full graph

**Bottom line**: The codebase already has the optimal edge representation (optimized 8-type system). Further pruning is not beneficial.

---

## Multi-Layer Graph (MLG) Encoder (Rejected: 2025-01-17)

### Proposal Summary

Add MLG architecture as 9th encoder variant extending HGTEncoder with:
- 3 semantic layers (AST=all nodes, Boolean=bool ops+shared, Arithmetic=arith ops+shared)
- Gated cross-layer attention every 3 HGT layers
- Layer-specific normalization (per semantic layer per GNN layer)
- Learnable residual scaling per semantic layer

**Claimed benefits**:
- +3-5% accuracy improvement at depth 10-14
- Addresses over-smoothing in deep GNNs
- Better Boolean↔Arithmetic pattern recognition

**Implementation scope**: ~520 lines across 7 files

### Rejection Rationale

**Four critical production-blocking defects with default configuration**:

#### 1. Index Out of Bounds Crash (CRITICAL)
**Location**: `mlg_encoder.py:311, 382`

Formula creates 3 cross-layer blocks: `num_cross_blocks = max(1, (num_layers - 1) // cross_layer_interval)`

For defaults (12 layers, interval 3): `(12-1)//3 = 3` blocks created.

Trigger condition fires at layers 2, 5, 8, 11 (4 times) → accesses `self.cross_layer_blocks[3]` → IndexError.

**Result**: Training crashes on first forward pass before any learning.

#### 2. Silent Data Corruption (CRITICAL)
**Location**: `mlg_encoder.py:411-416 (_dict_to_flat)`

Dict-to-flat conversion doesn't validate embedding counts match mask sizes. If `x_dict[type]` has N embeddings but mask sums to M ≠ N:
- N > M: Runtime crash "shape mismatch in assignment"
- N < M: Silent zero propagation → model learns to ignore cross-layer attention

No validation before tensor assignment → production reliability violation.

#### 3. Removes Existing Over-Smoothing Mitigation (CRITICAL)
**Location**: `mlg_encoder.py:331-397 (forward method)`

MLGHGTEncoder inherits from HGTEncoder but **discards GCNII initial residual logic**:
- HGTEncoder stores `h_0_dict` and mixes with `gcnii_alpha=0.15` (proven effective)
- MLG encoder never stores `h_0_dict` → no initial residual → GCNII mitigation lost

**Paradox**: Plan claims to address over-smoothing while removing existing mitigation. For depth 12+ expressions, expect accuracy **degradation** not "+3-5% improvement".

#### 4. Readout API Signature Mismatch (CRITICAL)
**Location**: `mlg_components.py:212-220`

`LayerAwareReadout.forward` expects 3 args: `(node_embeddings, node_types, batch)`

`MBADeobfuscator.forward` calls: `self.graph_readout(graph_emb, batch)` (2 args)

**Result**: TypeError at first model forward pass. Training fails before first gradient update.

---

### Conformance Violations

#### Missing Encoder Registry Integration (RULE 1)
Plan modifies `encoder.py` and `full_model.py` but **not `encoder_registry.py`**.

CLAUDE.md documents: "Encoder registry pattern via `encoder_registry.py`"

**Result**:
- `get_encoder('mlg_hgt')` → KeyError or fallback
- Config files with `encoder_type: mlg_hgt` fail to load
- Ablation studies can't discover MLG encoder

**Impact**: Core integration pattern violated, encoder unusable in production workflow.

---

### Structural Debt

1. **God function** (67 lines, 4 nesting levels): `MLGHGTEncoder.forward` mixes orchestration, data transformation, scheduling. Exceeds RULE 2 threshold (50 lines, 3 levels).

2. **Duplicate logic**: Dict-flat conversions reimplemented from HGTEncoder instead of reusing parent methods. Inheritance benefit lost.

3. **Inconsistent semantic layer mapping**: Node type → layer mapping defined in two places (`compute_layer_masks` + `_get_semantic_layer`). Violates DRY principle.

4. **Missing ablation integration**: Plan claims "+3-5% accuracy at depth 10-14" but provides no integration with `ablation_trainer.py` or depth-bucketed metrics. Performance claims unverifiable.

---

### Architectural Concerns

#### Redundancy with Existing GCNII
Codebase already has GCNII over-smoothing mitigation (α=0.15, λ=1.0) documented as production-ready. Plan **removes GCNII** while adding untested cross-layer attention.

**Risk**: Trading proven technique for unverified alternative in 85-90% complete production codebase.

#### Over-Engineering
Four new hyperparameters + three new components (gated attention, layer-specific norms, layer-aware readout) with **no justification** for why simpler alternatives were rejected:
- Tune existing GCNII α/λ parameters
- Add single global attention block (existing `global_attention.py` infrastructure)
- Adjust existing layer normalization strategy

#### Test Coverage Gaps
Unit tests present but **no systematic comparison to baseline HGT**. Project has established ablation infrastructure but plan provides no integration. The "+3-5% accuracy" claim (line 689 in plan) is unverifiable.

---

### Cost-Benefit Analysis

**Costs**:
- 520 lines of new code
- Four critical defects to fix before first successful run
- Removes existing mitigation (GCNII) creating regression risk
- Adds technical debt (god function, duplicated logic, inconsistent mapping)
- No ablation infrastructure to validate claims
- Four new hyperparameters to tune
- Violates project patterns (encoder registry)

**Benefits**:
- Claimed "+3-5% at depth 10-14" - **unverified and contradicted by GCNII removal**
- Claimed over-smoothing mitigation - **already exists via GCNII**
- Cross-layer pattern recognition - **speculative, no baseline comparison**

**Verdict**: Costs >> Benefits. High risk for unproven gains in production codebase.

---

### Lessons Learned

1. **Don't remove existing mitigations**: GCNII is proven and documented. Removing it while claiming to improve over-smoothing is contradictory.

2. **Validate claims with ablations**: "+3-5% accuracy" requires systematic comparison via `ablation_trainer.py`, not assertion in implementation plan.

3. **Follow project patterns**: Encoder registry is documented standard. Skipping it indicates incomplete codebase understanding.

4. **Test with default configs**: All four critical defects occur with plan's own default configuration (12 layers, interval 3). Should be caught before submission.

5. **Justify complexity**: Four hyperparameters + three components requires proving simpler alternatives insufficient. Plan provides no justification.

6. **Production codebase standards**: At 85-90% completion, new features need higher evidence bar. Unverified architectural changes create regression risk.

---

### Alternative Approaches

If over-smoothing at depth 12+ is actually a problem (contradicts production status claim):

1. **Tune existing GCNII parameters**: α=0.15, λ=1.0 are defaults. Literature shows α∈[0.05, 0.3] effective. Try α=0.2 or λ=0.8 first.

2. **Enable existing global attention**: `HGT_USE_GLOBAL_ATTENTION = True` flag already implemented in `global_attention.py`. Test before adding new architecture.

3. **Increase GCNII initial residual weight**: Modify HGT to use higher α for deeper layers (α_layer = α + β × layer_idx).

4. **Use Semantic HGT**: Already has property detection and Walsh-Hadamard features. May address pattern recognition without new architecture.

5. **Systematic ablation first**: Run `ablation_trainer.py` on all 8 existing encoders at depth 10-14. Identify if over-smoothing exists and which encoder handles it best. Then optimize best-performing encoder.

**Bottom line**: The codebase already has over-smoothing mitigation (GCNII) and global attention infrastructure. Adding MLG without proving existing tools insufficient is premature optimization.

---

## Template for Future Rejections

```markdown
## [Optimization Name] (Rejected: YYYY-MM-DD)

### Proposal Summary
[Brief description]

### Rejection Rationale
[Why it's not worth implementing]

### Lessons Learned
[Key takeaways to prevent re-proposal]
```
