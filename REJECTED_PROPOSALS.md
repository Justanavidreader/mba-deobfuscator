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
