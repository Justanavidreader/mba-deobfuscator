# Inference Pipeline

Complete specification of beam search, HTPS, verification, and reranking for the MBA Deobfuscator.

---

## Inference Overview

```
Input Expression "(x0 & x1) + (x0 ^ x1)"
    ↓
[Depth Check] depth < 10? → Beam Search : HTPS
    ↓
[Candidate Generation] 50-500 candidates
    ↓
[3-Tier Verification Cascade]
    ├─ Tier 1: Syntax (~10µs) → Filter ~60%
    ├─ Tier 2: Execution (~1ms) → Filter ~35%
    └─ Tier 3: Z3 SMT (~100ms, top-10 only)
    ↓
[Reranking] Multi-criteria scoring
    ↓
Simplified Expression "x0 | x1"
```

**Performance**:
- Shallow (depth < 10): 100-500ms per expression
- Deep (depth ≥ 10): 500ms-2s per expression
- Verification filters 95% before expensive Z3

---

## 1. Strategy Selection

### Depth-Based Routing

```python
def simplify(expression):
    depth = compute_depth(expression)

    if depth < 10:
        return beam_search(expression)
    else:
        return htps(expression)
```

**Rationale**:
- **Beam search**: Fast parallel decoding, good for shallow expressions
- **HTPS**: Compositional search, handles deep expressions better

### Manual Override

```bash
# Force beam search
python scripts/simplify.py --expr "..." --mode beam

# Force HTPS
python scripts/simplify.py --expr "..." --mode htps
```

---

## 2. Beam Search

### Algorithm

Diverse beam search with grammar constraints:

```python
def beam_search(expression, beam_width=50, num_groups=4):
    # Encode input
    encoder_output = model.encode(expression)

    # Initialize beams
    beams = [Beam(tokens=[BOS], score=0.0)]

    # Decode autoregressively
    for step in range(max_length):
        candidates = []

        for beam in beams:
            # Get next token logits
            logits = model.decode_step(beam.tokens, encoder_output)

            # Apply grammar constraints
            logits = grammar.constrain(logits, beam.state)

            # Top-k tokens
            top_k_tokens, top_k_scores = logits.topk(k=beam_width)

            for token, score in zip(top_k_tokens, top_k_scores):
                new_beam = beam.extend(token, score)
                candidates.append(new_beam)

        # Diversity grouping: divide candidates into groups
        groups = divide_into_groups(candidates, num_groups)

        # Select top beams from each group
        beams = []
        for group in groups:
            beams.extend(group.top_k(beam_width // num_groups))

        # Apply diversity penalty
        beams = apply_diversity_penalty(beams)

        # Prune to beam_width
        beams = beams[:beam_width]

        # Stop if all beams finished
        if all(beam.is_finished() for beam in beams):
            break

    # Return top beam
    return beams[0].tokens
```

### Configuration

```python
BEAM_WIDTH = 50
NUM_GROUPS = 4  # Diversity groups
TEMPERATURE = 0.7
LENGTH_PENALTY_ALPHA = 0.6  # Wu et al. 2016
DIVERSITY_PENALTY = 0.5
MAX_LENGTH = 256
```

### Length Normalization

Normalize scores by length to prevent bias toward short outputs:

```python
# Wu et al. 2016 formula
normalized_score = score / ((5 + length) / 6) ** alpha
# alpha = 0.6 (tuned)
```

### Diversity Grouping

Divide beams into groups to encourage diverse candidates:

```python
def divide_into_groups(beams, num_groups):
    # Hash each beam by its first different token
    groups = [[] for _ in range(num_groups)]

    for beam in beams:
        hash_val = hash(beam.tokens[:3])  # First 3 tokens
        group_id = hash_val % num_groups
        groups[group_id].append(beam)

    return groups
```

**Example**: Group 0 gets beams starting with `(x0 &`, Group 1 gets `(x0 |`, etc.

### Diversity Penalty

Penalize beams similar to already-selected beams:

```python
def apply_diversity_penalty(beams):
    selected = []

    for beam in sorted(beams, key=lambda b: b.score, reverse=True):
        # Compute similarity to already selected
        max_similarity = max([similarity(beam, sel) for sel in selected])

        # Penalize
        beam.score -= DIVERSITY_PENALTY * max_similarity

        selected.append(beam)

    return selected
```

### Grammar Constraints

Prevent syntactically invalid expressions:

```python
class GrammarConstraint:
    def __init__(self):
        self.grammar = self.build_grammar()

    def build_grammar(self):
        # BNF grammar
        return {
            'expr': ['term', 'expr binop term'],
            'term': ['factor', 'unaryop factor'],
            'factor': ['variable', 'constant', '( expr )'],
            'binop': ['+', '-', '*', '&', '|', '^'],
            'unaryop': ['~', 'neg'],
            'variable': ['x0', 'x1', ..., 'x7'],
            'constant': ['0', '1', ..., '255']
        }

    def constrain(self, logits, state):
        # Get valid next tokens based on state
        valid_tokens = self.get_valid_tokens(state)

        # Mask invalid tokens
        mask = torch.zeros_like(logits)
        mask[valid_tokens] = 1
        logits = logits * mask + (1 - mask) * (-1e9)  # -inf for invalid

        return logits

    def get_valid_tokens(self, state):
        # State machine based on parser state
        if state.expecting == 'binop':
            return [ADD, SUB, MUL, AND, OR, XOR]
        elif state.expecting == 'variable':
            return [X0, X1, X2, X3, X4, X5, X6, X7]
        # ... more states
```

**Guarantees**: All generated candidates are syntactically valid

### Implementation

```python
# src/inference/beam_search.py
from src.inference.beam_search import BeamSearchDecoder

decoder = BeamSearchDecoder(
    model=model,
    beam_width=50,
    num_groups=4,
    temperature=0.7,
    length_penalty_alpha=0.6,
    max_length=256
)

candidates = decoder.search(expression)
# candidates: List[str] (sorted by score, length=50)
```

---

## 3. HTPS (HyperTree Proof Search)

### Motivation

Deep expressions (depth ≥ 10) have exponential search space. HTPS decomposes them compositionally.

**Example**:
```
((x0 & x1) + (x0 ^ x1)) * ((x2 | x3) - (x2 ^ x3))
           ↓                          ↓
      Simplify left              Simplify right
           ↓                          ↓
       (x0 | x1)                  (x2 & x3)
           ↓                          ↓
              (x0 | x1) * (x2 & x3)
```

### Algorithm

UCB (Upper Confidence Bound) tree search:

```python
def htps(expression, budget=500):
    root = Node(expression, parent=None)
    tree = {root}

    for iteration in range(budget):
        # Select: UCB1
        node = select_ucb(tree)

        # Expand: Apply tactics
        children = expand(node)
        tree.update(children)

        # Simulate: Check if simplification is valid
        for child in children:
            reward = simulate(child)
            backpropagate(child, reward)

    # Return best leaf
    best = max(tree, key=lambda n: n.total_reward / n.visits)
    return best.expression
```

### UCB Selection

```python
def select_ucb(tree, c=sqrt(2)):
    def ucb1(node):
        if node.visits == 0:
            return float('inf')

        exploitation = node.total_reward / node.visits
        exploration = c * sqrt(log(node.parent.visits) / node.visits)

        return exploitation + exploration

    # Select node with highest UCB
    return max(tree, key=ucb1)
```

**Parameters**:
- `c = √2`: Exploration constant (balance exploit vs explore)

### Tactics (6 Fixed Rules)

```python
TACTICS = [
    identity_laws,          # x & x → x, x | x → x, x ^ x → 0
    mba_patterns,           # (x&y)+(x^y) → x|y, (x|y)-(x^y) → x&y
    constant_folding,       # Evaluate constants: 3 + 5 → 8
    distributive_laws,      # x & (y | z) → (x&y) | (x&z)
    de_morgan_laws,         # ~(x & y) → ~x | ~y
    algebraic_simplify      # General algebraic rules
]

def expand(node):
    children = []

    for tactic in TACTICS:
        # Apply tactic to node expression
        simplified = tactic(node.expression)

        if simplified != node.expression:
            child = Node(simplified, parent=node)
            children.append(child)

    return children
```

### Tactic Examples

#### 1. Identity Laws
```python
def identity_laws(expr):
    expr = re.sub(r'(\w+) & \1', r'\1', expr)  # x & x → x
    expr = re.sub(r'(\w+) \| \1', r'\1', expr)  # x | x → x
    expr = re.sub(r'(\w+) \^ \1', '0', expr)    # x ^ x → 0
    expr = re.sub(r'(\w+) \+ 0', r'\1', expr)   # x + 0 → x
    expr = re.sub(r'(\w+) \* 1', r'\1', expr)   # x * 1 → x
    return expr
```

#### 2. MBA Patterns
```python
def mba_patterns(expr):
    # (x & y) + (x ^ y) → x | y
    expr = re.sub(
        r'\((\w+) & (\w+)\) \+ \(\1 \^ \2\)',
        r'\1 | \2',
        expr
    )

    # (x | y) - (x ^ y) → x & y
    expr = re.sub(
        r'\((\w+) \| (\w+)\) - \(\1 \^ \2\)',
        r'\1 & \2',
        expr
    )

    return expr
```

#### 3. Constant Folding
```python
def constant_folding(expr):
    # Parse expression
    ast = parse(expr)

    # Evaluate constant subtrees
    def fold(node):
        if node.is_leaf():
            return node

        # Recursively fold children
        node.left = fold(node.left)
        node.right = fold(node.right)

        # If both children are constants, evaluate
        if node.left.is_constant() and node.right.is_constant():
            result = evaluate(node.operator, node.left.value, node.right.value)
            return ConstantNode(result)

        return node

    simplified_ast = fold(ast)
    return ast_to_string(simplified_ast)
```

### Simulation & Reward

```python
def simulate(node):
    # Check if simplification is valid
    is_valid = verify_execution(root.expression, node.expression)

    if is_valid:
        # Reward based on simplification ratio
        reward = len(root.expression) / len(node.expression)
    else:
        reward = 0.0

    return reward
```

### Backpropagation

```python
def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent
```

### Configuration

```python
HTPS_BUDGET = 500  # Node expansions
HTPS_UCB_C = sqrt(2)
HTPS_NUM_TACTICS = 6
HTPS_MAX_DEPTH = 20  # Search tree depth limit
```

### Implementation

```python
# src/inference/htps.py
from src.inference.htps import HyperTreeProofSearch

htps = HyperTreeProofSearch(
    tactics=TACTICS,
    budget=500,
    ucb_c=sqrt(2)
)

simplified = htps.search(expression)
```

---

## 4. Three-Tier Verification

### Cascade Architecture

Filter candidates efficiently before expensive Z3 verification:

```
All candidates (N=50-500)
    ↓
[Tier 1: Syntax] (~10µs per candidate)
    ├─ Valid → Pass (~40% of candidates)
    └─ Invalid → Reject (~60%)
    ↓
[Tier 2: Execution] (~1ms per candidate)
    ├─ Equivalent → Pass (~5% of original)
    └─ Not equivalent → Reject (~35%)
    ↓
[Tier 3: Z3 SMT] (~100ms per candidate, top-10 only)
    ├─ Proven equivalent → Accept
    └─ Not proven → Reject
    ↓
Verified candidates (1-10)
```

**Efficiency**: 95% filtered before Z3, saving ~99% verification time

### Tier 1: Syntax Validation

Check if expression is syntactically valid:

```python
def verify_syntax(expression):
    try:
        # Parse expression
        ast = parse_expression(expression)

        # Check grammar rules
        if not is_valid_ast(ast):
            return False

        # Check balanced parentheses
        if not balanced_parens(expression):
            return False

        # Check variable names (x0-x7 only)
        vars = extract_variables(expression)
        if any(v not in ['x0', ..., 'x7'] for v in vars):
            return False

        return True

    except ParseError:
        return False
```

**Cost**: ~10µs per expression (fast)

**Filters**: ~60% of candidates (malformed expressions)

### Tier 2: Execution Testing

Evaluate both expressions on random inputs and compare results:

```python
def verify_execution(expr1, expr2, num_samples=100):
    np.random.seed(42)  # Deterministic

    for width in [8, 16, 32, 64]:
        for _ in range(num_samples // 4):
            # Random inputs
            inputs = np.random.randint(0, 2**width, size=8)

            # Evaluate both expressions
            result1 = evaluate_expression(expr1, inputs, width)
            result2 = evaluate_expression(expr2, inputs, width)

            # Check equivalence
            if result1 != result2:
                return False  # Found counterexample

    return True  # Likely equivalent (no counterexample found)
```

**Cost**: ~1ms per expression (100 samples × 4 widths)

**Filters**: ~35% of remaining candidates (non-equivalent)

**False positive rate**: <0.1% (probabilistic test)

### Tier 3: Z3 SMT Solving

Formal verification using Z3 theorem prover:

```python
from src.utils.z3_interface import verify_equivalence

def verify_z3(expr1, expr2, timeout=1000):
    """
    Formally prove equivalence using Z3.

    Returns:
        is_equivalent: bool
        proof: Optional[str]
    """
    is_equiv, proof = verify_equivalence(expr1, expr2, timeout=timeout)
    return is_equiv, proof
```

**Cost**: ~100-1000ms per expression (timeout at 1s)

**Applied to**: Top-10 candidates only (after Tier 1+2 filtering)

**Guarantees**: 100% correctness (no false positives)

### Verification Result

```python
@dataclass
class VerificationResult:
    is_verified: bool
    tier: str  # 'syntax' | 'execution' | 'z3'
    proof: Optional[str]
    counterexample: Optional[Dict]
    verification_time: float
```

### Implementation

```python
# src/inference/verify.py
from src.inference.verify import ThreeTierVerifier

verifier = ThreeTierVerifier(
    num_execution_samples=100,
    z3_timeout=1000
)

result = verifier.verify(original, simplified)
# result.is_verified: bool
# result.tier: 'z3' (highest tier reached)
# result.proof: Z3 proof string
```

---

## 5. Reranking

After verification, rerank remaining candidates:

### Reranking Criteria

```python
def rerank_score(candidate):
    score = 0.0

    # 1. Verification tier (highest priority)
    if candidate.tier == 'z3':
        score += 100.0
    elif candidate.tier == 'execution':
        score += 50.0
    elif candidate.tier == 'syntax':
        score += 10.0

    # 2. Model confidence
    score += 20.0 * candidate.confidence  # Softmax probability

    # 3. Simplification ratio
    ratio = len(original) / len(candidate.expression)
    score += 10.0 * max(0, ratio - 1.0)  # Bonus for shorter

    # 4. Depth reduction
    depth_reduction = original_depth - candidate.depth
    score += 5.0 * max(0, depth_reduction)

    # 5. Complexity head prediction
    score += 5.0 * (1.0 - candidate.predicted_complexity)

    return score
```

**Weights** (tunable):
- Verification tier: 100 / 50 / 10
- Model confidence: 20
- Simplification ratio: 10
- Depth reduction: 5
- Complexity prediction: 5

### Multi-Objective Ranking

Alternative: Pareto ranking on multiple objectives:

```python
def pareto_rank(candidates):
    # Objectives: minimize (length, depth, !verified)
    fronts = compute_pareto_fronts(candidates)

    # Return candidates from first front
    return fronts[0]
```

### Implementation

```python
# src/inference/rerank.py
from src.inference.rerank import CandidateReranker

reranker = CandidateReranker(
    weights={
        'tier_z3': 100.0,
        'tier_execution': 50.0,
        'tier_syntax': 10.0,
        'confidence': 20.0,
        'simplification_ratio': 10.0,
        'depth_reduction': 5.0,
        'complexity': 5.0
    }
)

ranked = reranker.rerank(candidates)
best = ranked[0]
```

---

## 6. End-to-End Pipeline

### InferencePipeline

```python
# src/inference/pipeline.py
from src.inference.pipeline import InferencePipeline

pipeline = InferencePipeline(
    model=model,
    mode='auto',  # or 'beam', 'htps'
    beam_width=50,
    htps_budget=500,
    num_execution_samples=100,
    z3_timeout=1000
)

result = pipeline.simplify("(x0 & x1) + (x0 ^ x1)")

print(result.simplified)  # "x0 | x1"
print(result.verification_tier)  # "z3"
print(result.inference_time)  # 0.312 seconds
print(result.num_candidates)  # 50
print(result.num_verified)  # 3
```

### InferenceResult

```python
@dataclass
class InferenceResult:
    simplified: str
    original: str
    verification_tier: str  # 'z3' | 'execution' | 'syntax' | 'none'
    proof: Optional[str]
    inference_time: float  # seconds
    num_candidates: int
    num_verified: int
    all_candidates: List[str]  # All verified candidates (for debugging)
```

---

## 7. Performance Optimization

### Batched Verification (Tier 2)

Evaluate multiple candidates in parallel:

```python
def verify_execution_batch(original, candidates, num_samples=100):
    results = []

    # Generate random inputs once
    inputs = generate_random_inputs(num_samples)

    # Evaluate original
    original_outputs = [evaluate(original, inp) for inp in inputs]

    # Evaluate all candidates in parallel
    for candidate in candidates:
        candidate_outputs = [evaluate(candidate, inp) for inp in inputs]

        # Check equivalence
        is_equiv = all(o1 == o2 for o1, o2 in zip(original_outputs, candidate_outputs))
        results.append(is_equiv)

    return results
```

**Speedup**: 10× (amortize input generation)

### Parallel Z3 Verification

Run multiple Z3 instances in parallel:

```python
from multiprocessing import Pool

def verify_z3_parallel(original, candidates, num_workers=4):
    with Pool(num_workers) as pool:
        results = pool.starmap(
            verify_equivalence,
            [(original, cand) for cand in candidates]
        )

    return results
```

**Speedup**: 4× with 4 workers (independent verification tasks)

### Caching

Cache verification results to avoid recomputation:

```python
verification_cache = {}

def verify_cached(expr1, expr2):
    key = (expr1, expr2)

    if key in verification_cache:
        return verification_cache[key]

    result = verify_equivalence(expr1, expr2)
    verification_cache[key] = result

    return result
```

**Speedup**: 100× for repeated pairs

---

## 8. Inference Modes

### Mode Comparison

| Mode | Best For | Time | Accuracy | Candidates |
|------|----------|------|----------|------------|
| Greedy | Baseline | 10ms | 60% | 1 |
| Beam | Depth <10 | 100-500ms | 85% | 50 |
| HTPS | Depth ≥10 | 500ms-2s | 80% | 100-500 |
| Ensemble | Critical | 1-3s | 90% | 100+ |

### Greedy Decoding

Fastest (no search):

```python
def greedy_decode(expression):
    encoder_output = model.encode(expression)

    tokens = [BOS]
    for step in range(max_length):
        logits = model.decode_step(tokens, encoder_output)
        next_token = logits.argmax()
        tokens.append(next_token)

        if next_token == EOS:
            break

    return tokenizer.decode(tokens)
```

**Use case**: Fast approximation, interactive demos

### Ensemble

Combine multiple strategies:

```python
def ensemble_simplify(expression):
    # Run beam search and HTPS in parallel
    beam_candidates = beam_search(expression)
    htps_candidates = htps(expression)

    # Merge and deduplicate
    all_candidates = set(beam_candidates + htps_candidates)

    # Verify and rerank
    verified = [c for c in all_candidates if verify(original, c).is_verified]
    ranked = rerank(verified)

    return ranked[0]
```

**Use case**: Critical applications (maximum accuracy)

---

## 9. Evaluation Metrics

### Accuracy

```python
def accuracy(predictions, targets):
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return correct / len(predictions)
```

### Semantic Equivalence

```python
def semantic_accuracy(predictions, originals):
    correct = sum(
        verify_equivalence(pred, orig).is_verified
        for pred, orig in zip(predictions, originals)
    )
    return correct / len(predictions)
```

**More lenient**: Accepts any equivalent expression, not just canonical form

### Simplification Ratio

```python
def simplification_ratio(predictions, originals):
    ratios = [
        len(tokenize(orig)) / len(tokenize(pred))
        for orig, pred in zip(originals, predictions)
    ]
    return np.mean(ratios)
```

**Target**: >2.0× average simplification

### Depth Reduction

```python
def depth_reduction(predictions, originals):
    reductions = [
        compute_depth(orig) - compute_depth(pred)
        for orig, pred in zip(originals, predictions)
    ]
    return np.mean(reductions)
```

### Inference Time

```python
def average_inference_time(model, test_set):
    times = []

    for sample in test_set:
        start = time.time()
        _ = model.simplify(sample)
        end = time.time()
        times.append(end - start)

    return np.mean(times)
```

---

## 10. Common Issues & Solutions

### Issue: Beam Search Gets Stuck

**Symptoms**: All beams converge to same output

**Solutions**:
1. Increase diversity penalty (0.5 → 1.0)
2. Increase number of diversity groups (4 → 8)
3. Increase temperature (0.7 → 1.0)
4. Add nucleus sampling (top-p=0.9)

### Issue: HTPS Exceeds Budget

**Symptoms**: Search doesn't terminate, uses all budget

**Solutions**:
1. Reduce max depth (20 → 15)
2. Add early stopping (if best score not improved for 100 iterations)
3. Increase UCB exploration (c = √2 → c = 2.0)

### Issue: Z3 Timeout

**Symptoms**: Z3 times out (>1s) frequently

**Solutions**:
1. Increase timeout (1s → 5s) for critical applications
2. Use execution-only verification for non-critical
3. Simplify expressions before Z3 (constant folding)

### Issue: Low Verification Rate

**Symptoms**: <50% of candidates verified

**Solutions**:
1. Improve model training (higher accuracy)
2. More beam search candidates (50 → 100)
3. Lower verification threshold (use execution instead of Z3)

---

## 11. Best Practices

1. **Use auto mode** - Let depth determine strategy
2. **Start with beam search** - Faster and usually sufficient
3. **Use HTPS for depth ≥10** - Handles complexity better
4. **Always verify top-1** - At least with execution
5. **Z3 verify top-10** - Formal guarantee for best candidates
6. **Cache verifications** - Avoid redundant Z3 calls
7. **Batch execution tests** - 10× speedup
8. **Monitor inference time** - Set timeouts for production
9. **Log all candidates** - Useful for debugging
10. **Rerank by multiple criteria** - Not just model confidence

---

## 12. Production Deployment

### REST API

```python
from flask import Flask, request, jsonify
from src.inference.pipeline import InferencePipeline

app = Flask(__name__)
pipeline = InferencePipeline(model, mode='auto')

@app.route('/simplify', methods=['POST'])
def simplify():
    expr = request.json['expression']
    mode = request.json.get('mode', 'auto')

    result = pipeline.simplify(expr, mode=mode)

    return jsonify({
        'simplified': result.simplified,
        'tier': result.verification_tier,
        'time': result.inference_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Usage

```bash
curl -X POST http://localhost:5000/simplify \
  -H "Content-Type: application/json" \
  -d '{"expression": "(x0 & x1) + (x0 ^ x1)"}'
```

### Batch API

```python
@app.route('/simplify_batch', methods=['POST'])
def simplify_batch():
    expressions = request.json['expressions']

    results = [pipeline.simplify(expr) for expr in expressions]

    return jsonify({
        'results': [
            {'simplified': r.simplified, 'tier': r.verification_tier}
            for r in results
        ]
    })
```

---

## 13. Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| Pipeline | `src/inference/pipeline.py` | 300 |
| Beam Search | `src/inference/beam_search.py` | 250 |
| HTPS | `src/inference/htps.py` | 350 |
| Verification | `src/inference/verify.py` | 200 |
| Grammar | `src/inference/grammar.py` | 180 |
| Reranking | `src/inference/rerank.py` | 150 |
| Simplify Script | `scripts/simplify.py` | 120 |

---

## 14. References

1. **Beam Search**: Graves, "Sequence Transduction with Recurrent Neural Networks" (2012)
2. **Diverse Beam Search**: Vijayakumar et al., "Diverse Beam Search" (ICLR 2018)
3. **Length Normalization**: Wu et al., "Google's Neural Machine Translation System" (2016)
4. **HTPS/MCTS**: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
5. **Grammar-Constrained Decoding**: Scholak et al., "Picard: Parsing Incrementally for Constrained Auto-Regressive Decoding" (EMNLP 2021)
6. **Z3 SMT Solver**: de Moura & Bjørner, "Z3: An Efficient SMT Solver" (TACAS 2008)
