# Inference System

Complete inference pipeline for MBA deobfuscation. Orchestrates encoding, candidate generation, verification, and reranking.

## Overview

```
Input MBA Expression
    │
    ├─→ Encode (GNN → 256d embedding)
    │
    ├─→ Route by Depth
    │   ├─→ depth < 10:  Beam Search (beam_width=50, 4 diversity groups)
    │   └─→ depth ≥ 10:  HTPS (budget=500, UCB search, 6 tactics)
    │
    ├─→ Verify (3-tier cascade)
    │   ├─→ Tier 1: Syntax (grammar parse, ~10µs)
    │   ├─→ Tier 2: Execution (100 samples × 4 widths, ~1ms)
    │   └─→ Tier 3: Z3 (top-10 only, timeout 1000ms)
    │
    ├─→ Rerank (simplicity + confidence + verification level)
    │
    └─→ Output (best verified candidate)
```

## Pipeline Flow

### 1. Preprocessing (`pipeline._preprocess`)

```python
expr: str → (graph_batch, fingerprint, ast_depth)
```

**Process:**
1. Parse expression to graph: `expr_to_graph(expr)` → PyG Data object
2. Compute 448-dim semantic fingerprint:
   - Symbolic features (32): AST depth, operator counts, variable usage
   - Corner evaluations (256): 4 widths × 64 special values
   - Random hash (64): 4 widths × 16 random inputs
   - Derivatives (32): 4 widths × 8 derivative orders
   - Truth table (64): Complete table for 6 variables
3. Compute AST depth for routing decision: `expr_to_ast_depth(expr)`

**Output:**
- `graph_batch`: `[1, num_nodes]` PyG Batch for encoder
- `fingerprint`: `[1, 448]` float tensor
- `ast_depth`: scalar int

### 2. Encoding (`model.encode`)

```python
(graph_batch, fingerprint) → memory: [1, 1, 512]
```

Encoder produces fixed-size context vector combining:
- Graph structure (via GAT/GGNN)
- Semantic fingerprint (via projection layer)

### 3. Routing (Depth-Based)

| Condition | Method | Rationale |
|-----------|--------|-----------|
| depth < 10 | Beam Search | Fast, high-quality for shallow expressions |
| depth ≥ 10 | HTPS | Compositional tactics handle deep nesting better |

Threshold `HTPS_DEPTH_THRESHOLD = 10` tuned empirically. Beam search degrades at depth 12+.

### 4. Candidate Generation

#### Beam Search (`beam_search.py`)

**Parameters:**
- `beam_width = 50`: Number of hypotheses maintained
- `diversity_groups = 4`: Parallel beam groups with diversity penalty
- `diversity_penalty = 0.5`: Score reduction for tokens selected by other groups
- `temperature = 0.7`: Softmax temperature for sampling
- `max_length = 64`: Maximum output sequence

**Algorithm:**
```
1. Initialize: beams = [<sos>]
2. For each step (up to max_length):
   a. Batch decode current beams → logits [num_beams, vocab_size]
   b. Apply copy mechanism (if src_tokens provided):
      - Combine vocab logits and copy attention
      - P(w) = p_gen * P_vocab(w) + (1 - p_gen) * Σ(attn_i | src_i == w)
   c. Apply grammar constraints:
      - Get valid tokens from GrammarConstraint.get_valid_tokens()
      - Mask invalid tokens with -inf
   d. For each diversity group:
      - Apply diversity penalty for tokens selected by previous groups
      - Select top-k hypotheses
      - Track selected tokens
   e. Prune finished beams (hit EOS or max_length)
3. Rank by normalized score: score / length^0.6 (Wu et al. 2016)
4. Return top beam_width candidates
```

**Copy Mechanism:**
- Enables direct copying of source variables
- Critical for preserving variable names (x, y, z → x, y, z)
- Formula: `P(w) = p_gen * softmax(vocab_logits) + (1 - p_gen) * Σ(copy_attn_i | src_i == w)`

**Grammar Constraints:**
- Precomputed cache: parser_state → valid token IDs
- Lark LALR parser with MBA expression grammar
- Ensures syntactically valid sequences at every step
- Cache limited to 5000 states, max depth 20

#### HTPS (`htps.py`)

**Parameters:**
- `budget = 500`: Maximum tactic applications
- `ucb_constant = 1.414` (√2): UCB exploration constant
- `tactics = 6`: Identity laws, MBA patterns, constant folding

**Algorithm:**
```
1. Create root node with input expression
2. Evaluate root: value ← model.get_value(graph, fingerprint)
3. For budget iterations:
   a. Select leaf with highest UCB:
      UCB(node) = value + c * √(ln(parent.visits) / node.visits)
   b. Expand leaf by applying all tactics:
      - Try each tactic (identity_xor_self, mba_and_xor, etc.)
      - For each match, create child node
      - Batch evaluate children with model critic
   c. Backpropagate values to ancestors (incremental mean)
   d. If terminal expression found (depth ≤ 1), return immediately
4. Extract solution: follow highest-value path to leaf
5. Return (best_expr, proof_trace)
```

**Tactics (in order):**

| Tactic | Pattern | Replacement | Complexity |
|--------|---------|-------------|------------|
| `identity_xor_self` | `x ^ x` | `0` | O(n) |
| `identity_and_not` | `x & ~x` | `0` | O(n) |
| `identity_or_not` | `x \| ~x` | `-1` | O(n) |
| `mba_and_xor` | `(x&y) + (x^y)` | `x\|y` | O(n²) |
| `constant_fold` | `3 + 5` | `8` | O(n) |
| `simplify_subexpr` | Recursive | All tactics on children | O(n³) |

**UCB Selection:**
- Balances exploitation (high value) vs exploration (low visits)
- Unvisited nodes get infinite exploration value (always tried first)
- Guards against division by zero: `visits == 0 → ucb = inf`

**Value Estimation:**
- Critic head predicts simplification potential: `model.get_value(graph, fingerprint) → [0, 1]`
- Higher values indicate more promising nodes
- Batch evaluation for efficiency

### 5. Verification (3-Tier Cascade)

#### Tier 1: Syntax Check (~10µs/candidate)

```python
GrammarConstraint.parse_check(expr) → (is_valid, error_msg)
```

**Checks:**
- Grammar parsing (Lark LALR parser)
- Balanced parentheses
- Valid operator/operand alternation

**Purpose:** Fast rejection of malformed candidates before expensive tests.

#### Tier 2: Execution Test (~1ms/candidate)

```python
_execution_test(input_expr, candidate) → (passes, counterexample)
```

**Strategy:**
```
For each bit width in [8, 16, 32, 64]:
    Generate 100 test inputs:
        - 50% random values
        - 25% corner cases (0, 1, max, mid, powers of 2)
        - 25% structured patterns (all same, alternating bits)

    For each test input:
        result_input ← evaluate_expr(input_expr, vars, width)
        result_candidate ← evaluate_expr(candidate, vars, width)

        If result_input ≠ result_candidate:
            Return (False, counterexample)

Return (True, None)
```

**Corner Values (per width):**
- Small: 0, 1, 2, 3
- Near max: max, max-1, max-2
- Midpoint: 2^(width-1), mid±1
- Alternating bits: 0xAA, 0x55
- Powers of 2: 2^0, 2^1, ..., 2^(width-1)

**Purpose:** Catches most inequivalences in ~1ms. Avoids expensive Z3 calls for obviously wrong candidates.

#### Tier 3: Z3 Formal Verification (~100-1000ms/candidate)

```python
_z3_verify(input_expr, candidate) → (is_equiv, counterexample, timed_out)
```

**Process:**
1. Convert expressions to Z3 BitVec formulas (64-bit)
2. Check equivalence: `solver.check(expr1 != expr2) == unsat`
3. If not equivalent, find counterexample: `solver.model()`
4. Timeout after 1000ms

**Returns:**
- `(True, None, False)`: Formally equivalent
- `(False, counterexample, False)`: Inequivalent with concrete counterexample
- `(None, None, True)`: Timeout (inconclusive)

**Selective Application:**
- Only top-k exec-passing candidates verified (`z3_top_k = 10`)
- Saves ~90% of Z3 time compared to verifying all candidates

### 6. Reranking (`rerank.py`)

#### Feature Extraction

```python
RankingFeatures:
    token_length: int        # From tokenizer
    ast_depth: int           # Maximum tree depth
    num_operators: int       # Count of +, -, &, |, ^, ~, *
    num_variables: int       # Count of x, y, z, ...
    num_constants: int       # Count of numeric literals
    model_confidence: float  # Beam normalized score
    verification_level: int  # 0=none, 1=syntax, 2=exec, 3=z3
```

#### Scoring Formula

```python
score = Σ(weight_i * normalize(feature_i))
```

**Weights (default):**

| Feature | Weight | Rationale |
|---------|--------|-----------|
| `token_length` | -1.0 | Prefer shorter expressions |
| `ast_depth` | -0.5 | Prefer shallower trees |
| `num_operators` | -0.3 | Prefer fewer operations |
| `num_variables` | 0.0 | Neutral (should match input) |
| `num_constants` | -0.2 | Slightly prefer fewer constants |
| `model_confidence` | 2.0 | Trust model's probabilities |
| `verification_level` | 5.0 | Strongly prefer verified candidates |

**Normalization:**
- `token_length / 64` (MAX_SEQ_LEN)
- `ast_depth / 16` (MAX_OUTPUT_DEPTH)
- `num_operators / 20` (typical max)
- `model_confidence`: already normalized (log prob)
- `verification_level`: {0, 1, 2, 3} direct

**Filtering:**
- Candidates with counterexamples get score = -∞ (excluded)

#### Output

```python
List[(candidate, score)] sorted by descending score
```

## Performance Characteristics

### Beam Search

| Depth | Candidates | Time | Accuracy |
|-------|------------|------|----------|
| 2-4   | 50         | ~50ms | >95% |
| 5-7   | 50         | ~150ms | >90% |
| 8-10  | 50         | ~300ms | >80% |
| 11+   | 50         | ~500ms | ~70% (degrades) |

**Bottlenecks:**
- Decoder forward pass: ~10ms per step × 20 steps avg = 200ms
- Grammar constraint lookup: ~1ms per step (precomputed cache)
- Copy mechanism: +2ms per step (attention computation)

**Tuning:**
- Increase `beam_width` for better coverage (linear time increase)
- Increase `diversity_groups` for more varied outputs (same total beam_width)
- Decrease `temperature` for more focused search (0.5-0.9 range)

### HTPS

| Depth | Budget | Time | Accuracy |
|-------|--------|------|----------|
| 10-12 | 500    | ~2s  | >85% |
| 13-14 | 500    | ~3s  | >75% |
| 15+   | 500    | ~5s  | ~60% |

**Bottlenecks:**
- Critic evaluation: ~5ms per node × ~500 nodes = 2500ms
- Tactic matching: ~0.5ms per application
- Z3 verification: ~100-1000ms per top candidate

**Tuning:**
- Increase `budget` for deeper search (linear time increase)
- Adjust `ucb_constant`: higher = more exploration (√2 is standard)
- Add/remove tactics: each adds ~10% overhead

### Verification

| Tier | Time/Candidate | Rejection Rate | Cumulative Pass |
|------|----------------|----------------|-----------------|
| Syntax | 10µs | ~5% | 95% |
| Execution | 1ms | ~50% | 48% |
| Z3 (top-10) | 100-1000ms | ~20% | 38% |

**Overall:**
- 50 candidates → ~1 second for verification cascade
- Z3 only called on ~10 candidates (top exec-passing)

## Tuning Parameters

### For Speed

```python
pipeline = InferencePipeline(
    beam_width=20,              # Reduce from 50
    use_htps_for_deep=False,    # Disable HTPS (accept lower accuracy for depth ≥10)
    use_grammar=False,          # Disable grammar (accept 5% syntax errors)
    verify_output=False         # Disable verification (accept 50% wrong answers)
)
```

**Effect:** ~50ms total (100× speedup), accuracy drops to ~50%.

### For Accuracy

```python
pipeline = InferencePipeline(
    beam_width=100,             # Increase from 50
    htps_depth_threshold=8,     # Use HTPS earlier
    use_grammar=True,
    verify_output=True
)

# Increase HTPS budget
htps.budget = 1000             # From 500
```

**Effect:** ~5s total (5× slower), accuracy improves to >90% for depth 14.

### For Balanced

```python
# Default settings already balanced for depth ≤12
beam_width=50
htps_depth_threshold=10
use_grammar=True
verify_output=True
```

**Effect:** ~300ms for depth ≤10, ~2s for depth ≥10.

## Adding New Search Strategies

### Custom Beam Search Variant

```python
class MyBeamDecoder(BeamSearchDecoder):
    def decode(self, memory, src_tokens):
        # Override with custom beam expansion logic
        ...
```

Register in pipeline:
```python
pipeline.beam_decoder = MyBeamDecoder(model, tokenizer)
```

### Custom HTPS Tactic

Add to `htps.py`:
```python
def _apply_my_tactic(self, ast: ASTNode) -> Optional[List[str]]:
    """
    Find custom pattern and replace.

    Returns:
        List of result expressions if tactic applies, None otherwise
    """
    if ast.type == 'PATTERN' and ...:
        # Build replacement AST
        result_ast = ...
        return [self._ast_to_str(result_ast)]
    return None
```

Add to tactic enum and `_apply_tactic()`:
```python
class Tactic(Enum):
    MY_TACTIC = "my_tactic"

def _apply_tactic(self, tactic, expr, ast):
    ...
    elif tactic == Tactic.MY_TACTIC:
        return self._apply_my_tactic(ast)
```

Add to constants:
```python
HTPS_TACTICS = [..., 'my_tactic']
```

### Custom Verifier Tier

```python
class MyVerifier(ThreeTierVerifier):
    def verify_batch(self, input_expr, candidates):
        results = super().verify_batch(input_expr, candidates)

        # Add custom tier 4
        for result in results:
            if result.z3_verified:
                # Apply additional check
                if my_custom_check(input_expr, result.candidate):
                    result.custom_verified = True

        return results
```

## File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `pipeline.py` | Main orchestration | `simplify()`, `_preprocess()`, `_generate_candidates()`, `_postprocess()` |
| `beam_search.py` | Beam search decoder | `decode()`, `_expand_beams()`, `_apply_copy_mechanism()` |
| `htps.py` | HyperTree proof search | `search()`, `_select_leaf()`, `_expand()`, `_apply_tactic()` |
| `grammar.py` | Grammar constraints | `get_valid_tokens()`, `mask_logits()`, `parse_check()` |
| `verify.py` | 3-tier verification | `verify_batch()`, `_syntax_check()`, `_execution_test()`, `_z3_verify()` |
| `rerank.py` | Candidate reranking | `rerank()`, `extract_features()`, `compute_score()` |

## Common Issues

### Issue: Beam search produces invalid syntax

**Cause:** Grammar cache miss or disabled grammar constraints.

**Fix:**
```python
# Rebuild grammar cache
grammar = GrammarConstraint(tokenizer)
grammar._build_token_cache()

# Ensure grammar enabled
beam_decoder = BeamSearchDecoder(model, tokenizer, use_grammar=True)
```

### Issue: HTPS takes too long

**Cause:** Budget too high or too many tactics.

**Fix:**
```python
# Reduce budget
htps = MinimalHTPS(model, tokenizer, budget=200)  # From 500

# Disable expensive tactics
htps.tactics = [
    Tactic.IDENTITY_XOR_SELF,
    Tactic.IDENTITY_AND_NOT,
    Tactic.MBA_AND_XOR,
    # Remove SIMPLIFY_SUBEXPR (O(n³) complexity)
]
```

### Issue: Z3 times out frequently

**Cause:** Expressions too complex or timeout too low.

**Fix:**
```python
# Increase timeout
verifier = ThreeTierVerifier(tokenizer, z3_timeout_ms=5000)  # From 1000

# Or reduce Z3 verification count
verifier.z3_top_k = 5  # From 10
```

### Issue: Wrong results passing verification

**Cause:** Insufficient execution test coverage.

**Fix:**
```python
# Increase test samples
verifier = ThreeTierVerifier(tokenizer, exec_samples=500)  # From 100

# Or add custom corner cases in _generate_test_inputs()
```

## References

**Beam Search:**
- Wu et al. (2016): Length normalization formula `score / length^0.6`
- Vijayakumar et al. (2018): Diverse beam search with groups

**HTPS:**
- Upper Confidence Bound (UCB): Kocsis & Szepesvári (2006)
- Compositional tactics: Inspired by Lean/Coq theorem provers

**Verification:**
- Z3 SMT solver: de Moura & Bjørner (2008)
- Three-tier cascade: Custom design for MBA expressions
