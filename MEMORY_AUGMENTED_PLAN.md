# MEMORY-AUGMENTED GNN IMPLEMENTATION PLAN

## Executive Summary

This plan adds memory-augmented GNNs with algebraic rule memory to the MBA deobfuscation system. The implementation introduces a new encoder architecture (`MemoryAugmentedEncoder`) that combines:

1. **AlgebraicRuleMemory**: Differentiable external memory bank storing learned transformation patterns
2. **Hybrid GNN**: RGCN (local patterns) + HGT (global attention) with memory-guided message passing
3. **Stack-augmented networks**: PUSH/POP operations for tree traversal
4. **Multi-task training**: Rule application prediction, contrastive equivalence, axiom satisfaction

**Integration Strategy**: Extends existing encoder registry pattern, maintains backward compatibility, integrates with Phase 2 curriculum training.

**Estimated Complexity**: ~15M parameters for memory module, 4-6 weeks implementation, medium risk.

---

## 1. Architecture Analysis & Integration Points

### 1.1 Current System Architecture

**Existing Encoders** (from `src/models/encoder.py`):
- `GATJKNetEncoder`: Homogeneous, 4 layers, 256d, ~2.8M params
- `GGNNEncoder`: Heterogeneous, 8 timesteps, 256d, ~3.2M params
- `HGTEncoder`: Heterogeneous, 12 layers, 768d, ~60M params (with GCNII over-smoothing mitigation)
- `RGCNEncoder`: Relational GCN, 12 layers, 768d, ~60M params

**Existing Infrastructure**:
- `BaseEncoder` abstract class with `forward()`, `requires_edge_types`, `requires_node_features` interface
- `MBADeobfuscator` full model orchestrates encoder → readout → decoder
- Phase 2 trainer with 4-stage curriculum (depth 2→5→10→14)
- Semantic fingerprint (416d after removing derivatives): symbolic + corner + random + truth table

### 1.2 Integration Points

| Component | Integration Method | Backward Compatibility |
|-----------|-------------------|------------------------|
| **Encoder** | New `MemoryAugmentedEncoder(BaseEncoder)` | ✓ Via encoder registry |
| **Training** | Extend `Phase2Trainer` with multi-task losses | ✓ Via config flag `use_memory_augmented` |
| **Config** | New `configs/memory_augmented.yaml` | ✓ Optional config |
| **Full Model** | `MBADeobfuscator(encoder_type='memory_augmented')` | ✓ Existing pattern |
| **Ablation** | Add to `encoder_registry.py` | ✓ Via `get_encoder()` |

**Key Insight**: The system already has multi-head output (`equivalence_head`, `simplify_head`) via `TokenHead`, `ComplexityHead`, `ValueHead`. We extend this pattern with new auxiliary heads for rule prediction.

---

## 2. Detailed Component Design

### 2.1 AlgebraicRuleMemory Module

**Purpose**: Differentiable memory bank storing learned algebraic transformation patterns (e.g., `x^x→0`, `(x&y)+(x^y)→x|y`).

**Architecture**:
```python
class AlgebraicRuleMemory(nn.Module):
    """
    Differentiable memory bank for algebraic transformation patterns.

    Memory slots store rule embeddings; node features query via attention
    to retrieve relevant transformation context.
    """
    def __init__(self, num_slots: int = 256, slot_dim: int = 256):
        # Memory: [num_slots, slot_dim] learnable embeddings
        # Query projection: node features → query vectors
        # Attention: multi-head attention over memory slots
        # Output projection: retrieved context → node update
```

**Dimensions**:
- `num_slots`: 256 (configurable via `MEMORY_NUM_SLOTS`)
- `slot_dim`: Matches `hidden_dim` (256 for base, 768 for scaled)
- Attention heads: 4 (configurable via `MEMORY_ATTN_HEADS`)

**Memory Initialization**: Xavier uniform (random algebraic patterns, refined during training).

**Parameters**: `256 * 256 = 65,536` base parameters (0.065M params).

---

### 2.2 Hybrid Encoder Architecture

**Design**: RGCN (2 layers, local) → HGT (3 layers, global) → Memory-guided aggregation

**Rationale**:
- RGCN captures local structural patterns (operator precedence, tree structure)
- HGT captures global dependencies (shared subexpressions)
- Memory provides learned rule context for message passing

**Architecture**:
```
Input Node Embeddings [N, node_dim]
    ↓
RGCN Layer 1 (local pattern extraction)
    ↓ [N, hidden_dim]
RGCN Layer 2 (local pattern refinement)
    ↓ [N, hidden_dim]
Memory Query & Retrieval (attention over rule slots)
    ↓ [N, hidden_dim] + [N, hidden_dim] = [N, 2*hidden_dim]
HGT Layer 1 (global attention with memory context)
    ↓ [N, hidden_dim]
HGT Layer 2 (global attention refinement)
    ↓ [N, hidden_dim]
HGT Layer 3 (global attention finalization)
    ↓ [N, hidden_dim]
Final Node Embeddings
```

**Memory Integration**: After RGCN, query memory with node embeddings, concatenate retrieved context before HGT layers.

**Parameters**:
- RGCN (2 layers): ~10M params (2 * 768 * 768 * 8 edge types)
- HGT (3 layers): ~18M params (3 * 768 * 768 * 16 heads)
- Memory module: ~0.065M params
- **Total: ~28M params** (between GGNN 3.2M and HGT 60M)

---

### 2.3 Stack-Augmented Recursive Networks

**Purpose**: Maintain explicit stack for tree traversal, enabling model to track parent-child relationships and backtrack.

**Design**:
```python
class StackAugmentedGNN(nn.Module):
    """
    Stack-augmented GNN with PUSH/POP operations for tree traversal.

    During message passing, model predicts PUSH (descend) or POP (ascend)
    operations to maintain stack state reflecting current tree path.
    """
    def __init__(self, hidden_dim: int, max_stack_depth: int = 20):
        # Stack: [batch_size, max_depth, hidden_dim] differentiable stack
        # PUSH gate: MLP predicting whether to push current node
        # POP gate: MLP predicting whether to pop stack
        # Stack pointer: Soft pointer to current stack top
```

**Operations**:
- **PUSH**: When entering child node, push parent state
- **POP**: When leaving subtree, pop to return to parent
- **READ**: Current node reads stack top for parent context

**Soft Stack Implementation** (differentiable):
- Stack pointer: Continuous value in [0, max_depth]
- PUSH: Increment pointer, write to position
- POP: Decrement pointer, read from position
- Use sigmoid gates for push/pop decisions

**Parameters**: ~0.5M params (gates + stack embeddings).

---

### 2.4 Positional Encodings

**Required Encodings**:

1. **Depth-based encoding**: Position in tree hierarchy (already available as `dag_pos` in current system)
   - Dimension: 4 ([depth, subtree_size, in_degree, is_shared])
   - Integration: Concatenate to node features before encoder

2. **Path-based encoding**: Root-to-node path representation
   - **Method**: LSTM over path sequence (node types from root to current)
   - Dimension: `hidden_dim`
   - Integration: Add to node embeddings as residual

3. **PEARL method** (2025): Learnable positional features with linear complexity
   - **Method**: Learned node-type-specific positional embeddings
   - Dimension: `hidden_dim`
   - Integration: Add to node embeddings as residual

**Implementation Strategy**: All three encodings are **additive residuals** to node embeddings before GNN layers.

**Configuration**: Enable/disable via config flags:
- `USE_DEPTH_ENCODING`: bool (default True)
- `USE_PATH_ENCODING`: bool (default True)
- `USE_PEARL_ENCODING`: bool (default False, experimental)

---

### 2.5 Multi-Head Output Architecture

**Heads** (extending existing `heads.py`):

1. **RuleApplicationHead**: Predicts which algebraic rule applies at each node
   - Input: `[N, hidden_dim]` node embeddings
   - Output: `[N, num_rule_classes]` logits
   - Loss: Cross-entropy with rule labels

2. **EquivalenceHead**: Binary classification for expression equivalence (already exists as `ValueHead`, repurpose)
   - Input: `[B, hidden_dim]` graph embeddings
   - Output: `[B, 1]` equivalence probability
   - Loss: Contrastive loss (equivalent pairs → similar embeddings)

3. **AxiomHead**: Soft constraint satisfaction for algebraic axioms
   - Input: `[N, hidden_dim]` node embeddings
   - Output: `[N, num_axioms]` satisfaction scores
   - Loss: MSE against axiom satisfaction targets

**Existing Heads** (remain unchanged):
- `TokenHead`: Vocabulary logits for generation
- `ComplexityHead`: Length/depth prediction
- `ValueHead`: HTPS guidance (repurposed for equivalence in multi-task variant)

---

## 3. Implementation Phases

### Phase 1: Core Memory Module (Week 1-2)

**New Files**:
```
src/models/memory_augmented/
├── __init__.py
├── algebraic_rule_memory.py    # AlgebraicRuleMemory class
├── stack_augmented_gnn.py      # StackAugmentedGNN class
├── hybrid_encoder.py           # MemoryAugmentedEncoder class
└── positional_pearl.py         # PEARL positional encoding
```

**Implementation Steps**:

1. **AlgebraicRuleMemory** (`algebraic_rule_memory.py`):
   ```python
   class AlgebraicRuleMemory(nn.Module):
       def __init__(self, num_slots=256, slot_dim=256, num_heads=4):
           self.memory_slots = nn.Parameter(torch.randn(num_slots, slot_dim))
           self.query_proj = nn.Linear(slot_dim, slot_dim)
           self.attention = nn.MultiheadAttention(slot_dim, num_heads)
           self.output_proj = nn.Linear(slot_dim, slot_dim)

       def forward(self, node_features, batch):
           # Query memory with node features
           # Return: [N, slot_dim] retrieved rule context
   ```

2. **StackAugmentedGNN** (`stack_augmented_gnn.py`):
   ```python
   class StackAugmentedGNN(nn.Module):
       def __init__(self, hidden_dim=256, max_depth=20):
           self.stack = nn.Parameter(torch.zeros(1, max_depth, hidden_dim))
           self.push_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
           self.pop_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

       def forward(self, x, edge_index, edge_type):
           # Perform GNN message passing with stack operations
           # Return: [N, hidden_dim] with stack-augmented context
   ```

3. **MemoryAugmentedEncoder** (`hybrid_encoder.py`):
   ```python
   class MemoryAugmentedEncoder(BaseEncoder):
       def __init__(self, hidden_dim=256, num_rgcn_layers=2, num_hgt_layers=3):
           self.rgcn = RGCNEncoder(hidden_dim, num_layers=num_rgcn_layers)
           self.memory = AlgebraicRuleMemory(num_slots=256, slot_dim=hidden_dim)
           self.hgt = HGTEncoder(hidden_dim, num_layers=num_hgt_layers)
           self.stack = StackAugmentedGNN(hidden_dim)

       def _forward_impl(self, x, edge_index, batch, edge_type, dag_pos=None):
           # RGCN local encoding
           h = self.rgcn._forward_impl(x, edge_index, batch, edge_type, dag_pos)
           # Memory retrieval
           rule_context = self.memory(h, batch)
           h_memory = torch.cat([h, rule_context], dim=-1)
           # HGT global attention
           h = self.hgt._forward_impl(h_memory, edge_index, batch, edge_type, dag_pos)
           # Stack augmentation
           h = self.stack(h, edge_index, edge_type)
           return h
   ```

4. **PEARL Encoding** (`positional_pearl.py`):
   ```python
   class PEARLEncoding(nn.Module):
       def __init__(self, num_node_types=10, hidden_dim=256):
           self.embeddings = nn.Embedding(num_node_types, hidden_dim)

       def forward(self, node_types):
           return self.embeddings(node_types)
   ```

**Testing**: Unit tests for each module in `tests/test_memory_augmented.py`.

---

### Phase 2: Auxiliary Heads & Multi-Task Losses (Week 3)

**Modified Files**:
- `src/models/heads.py` (add new heads)
- `src/training/losses.py` (add multi-task losses)

**New Heads**:
```python
class RuleApplicationHead(nn.Module):
    """Predicts algebraic rule class at each node."""
    def __init__(self, d_model=256, num_rule_classes=20):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_rule_classes)
        )

class AxiomHead(nn.Module):
    """Predicts axiom satisfaction scores."""
    def __init__(self, d_model=256, num_axioms=10):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_axioms),
            nn.Sigmoid()  # Satisfaction scores in [0, 1]
        )
```

**Multi-Task Losses**:
```python
def memory_augmented_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for memory-augmented training.

    Losses:
    1. Standard Phase 2 losses (CE, complexity, copy)
    2. Rule application loss (CE at node level)
    3. Contrastive equivalence loss (InfoNCE)
    4. Axiom satisfaction loss (MSE)
    """
    # Standard losses (from phase2_loss)
    ce_loss, complexity_loss, copy_loss = ...

    # Rule application loss
    rule_logits = outputs['rule_logits']  # [N, num_classes]
    rule_targets = targets['rule_labels']  # [N]
    rule_loss = F.cross_entropy(rule_logits, rule_targets)

    # Contrastive equivalence loss
    equiv_embeddings = outputs['graph_embeddings']  # [B, hidden_dim]
    pos_pairs, neg_pairs = targets['equiv_pairs']
    contrastive_loss = contrastive_equivalence_loss(
        equiv_embeddings, pos_pairs, neg_pairs, temperature=0.07
    )

    # Axiom satisfaction loss
    axiom_scores = outputs['axiom_scores']  # [N, num_axioms]
    axiom_targets = targets['axiom_targets']  # [N, num_axioms]
    axiom_loss = F.mse_loss(axiom_scores, axiom_targets)

    # Weighted sum
    total_loss = (
        config['ce_weight'] * ce_loss +
        config['complexity_weight'] * complexity_loss +
        config['copy_weight'] * copy_loss +
        config['rule_weight'] * rule_loss +
        config['contrastive_weight'] * contrastive_loss +
        config['axiom_weight'] * axiom_loss
    )

    return total_loss, {
        'ce': ce_loss.item(),
        'complexity': complexity_loss.item(),
        'copy': copy_loss.item(),
        'rule': rule_loss.item(),
        'contrastive': contrastive_loss.item(),
        'axiom': axiom_loss.item(),
    }
```

**Configuration** (add to `src/constants.py`):
```python
# Memory-Augmented GNN Constants
MEMORY_NUM_SLOTS: int = 256
MEMORY_ATTN_HEADS: int = 4
MEMORY_SLOT_DIM: int = 256  # Matches hidden_dim

# Stack-Augmented GNN
STACK_MAX_DEPTH: int = 20

# Multi-Task Loss Weights
RULE_LOSS_WEIGHT: float = 0.2
CONTRASTIVE_LOSS_WEIGHT: float = 0.1
AXIOM_LOSS_WEIGHT: float = 0.05

# Rule Classes (algebraic transformations)
NUM_RULE_CLASSES: int = 20  # Identity, absorption, distribution, etc.
NUM_AXIOMS: int = 10  # Commutativity, associativity, distributivity, etc.
```

---

### Phase 3: Data Augmentation & Label Generation (Week 4)

**Challenge**: Multi-task training requires auxiliary labels:
- Rule application labels (which rule applies at each node)
- Equivalence pairs (positive/negative samples)
- Axiom satisfaction targets (does subexpression satisfy axiom?)

**Solution**: Extend data generation pipeline.

**Modified Files**:
- `src/data/dataset.py` (extend `MBADataset` to include auxiliary labels)
- `scripts/generate_auxiliary_labels.py` (NEW: generate rule/axiom labels from expression trees)

**Label Generation Strategy**:

1. **Rule Labels**: Pattern matching on AST subtrees
   ```python
   def label_rules(ast_node):
       if matches_pattern(ast_node, "x ^ x"):
           return RULE_IDENTITY_XOR
       elif matches_pattern(ast_node, "(x & y) + (x ^ y)"):
           return RULE_MBA_TO_OR
       # ... 20 rule patterns
       else:
           return RULE_NONE
   ```

2. **Equivalence Pairs**: Sample from dataset
   - **Positive pairs**: Different obfuscations of same simplified form
   - **Negative pairs**: Different simplified forms
   - Generate at dataloader time via `collate_fn`

3. **Axiom Targets**: Property checking
   ```python
   def check_axiom_satisfaction(node, axiom):
       if axiom == AXIOM_COMMUTATIVE:
           return is_commutative_op(node.op)
       elif axiom == AXIOM_ASSOCIATIVE:
           return is_associative_op(node.op)
       # ... 10 axioms
   ```

**Data Format** (extend JSONL):
```json
{
  "obfuscated": "(x & y) + (x ^ y)",
  "simplified": "x | y",
  "depth": 3,
  "rule_labels": [2, 2, 0, 1, 0, 1, 5],  # Per-node rule IDs
  "axiom_targets": [[1,0,1,...], ...],   # [N, 10] axiom satisfaction
}
```

**Backward Compatibility**: Existing datasets without auxiliary labels → default to `RULE_NONE` and zeros for axioms.

---

### Phase 4: Training Infrastructure (Week 5)

**New Trainer**:
```python
class MemoryAugmentedTrainer(Phase2Trainer):
    """
    Extends Phase 2 supervised trainer with multi-task losses.

    Maintains curriculum learning (depth 2→5→10→14) and self-paced learning.
    Adds auxiliary loss computation and logging.
    """
    def train_step(self, batch):
        # Standard forward pass
        outputs = self.model(...)

        # Compute rule predictions
        rule_logits = self.rule_head(outputs['node_embeddings'])
        axiom_scores = self.axiom_head(outputs['node_embeddings'])

        # Prepare targets
        targets = {
            'target_ids': batch['target_ids'],
            'length': batch['length'],
            'depth': batch['depth'],
            'rule_labels': batch['rule_labels'],
            'equiv_pairs': self._sample_equiv_pairs(batch),
            'axiom_targets': batch['axiom_targets'],
        }

        # Multi-task loss
        loss, loss_dict = memory_augmented_loss(outputs, targets, self.config)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss_dict
```

**Configuration** (`configs/memory_augmented.yaml`):
```yaml
model:
  encoder_type: memory_augmented
  hidden_dim: 256
  num_rgcn_layers: 2
  num_hgt_layers: 3
  memory_num_slots: 256
  memory_attn_heads: 4
  use_stack_augmented: true
  use_pearl_encoding: false

training:
  phase: 2
  use_memory_augmented: true
  batch_size: 32
  num_epochs: 50

  # Standard loss weights
  ce_weight: 1.0
  complexity_weight: 0.1
  copy_weight: 0.1

  # Memory-augmented loss weights
  rule_weight: 0.2
  contrastive_weight: 0.1
  axiom_weight: 0.05

  # Curriculum stages (same as Phase 2)
  curriculum_stages:
    - {max_depth: 2, epochs: 10, target: 0.95}
    - {max_depth: 5, epochs: 15, target: 0.90}
    - {max_depth: 10, epochs: 15, target: 0.80}
    - {max_depth: 14, epochs: 10, target: 0.70}
```

**Training Script** (`scripts/train_memory_augmented.py`):
```python
if __name__ == "__main__":
    # Load config
    config = load_yaml("configs/memory_augmented.yaml")

    # Initialize model
    model = MBADeobfuscator(encoder_type='memory_augmented', **config['model'])

    # Initialize trainer
    trainer = MemoryAugmentedTrainer(
        model=model,
        train_path="data/train.jsonl",
        val_path="data/val.jsonl",
        config=config['training'],
    )

    # Train
    trainer.train()
```

---

### Phase 5: Ablation Studies & Evaluation (Week 6)

**Ablation Dimensions**:
1. Memory module (with/without)
2. Stack augmentation (with/without)
3. Hybrid architecture (RGCN+HGT vs. HGT only vs. RGCN only)
4. Positional encodings (depth-only vs. depth+path vs. depth+path+PEARL)
5. Multi-task losses (standard vs. +rule vs. +rule+contrastive vs. +rule+contrastive+axiom)

**Evaluation Metrics**:
- **Accuracy**: Exact match on depth buckets [2-4, 5-7, 8-10, 11-14]
- **Equivalence verification**: Z3 verification rate
- **Rule prediction accuracy**: Per-node rule classification
- **Extrapolation**: Train on depth ≤10, test on depth 11-14

**Ablation Script** (`scripts/ablate_memory_augmented.py`):
```python
ablation_configs = [
    {'memory': True, 'stack': True, 'encoding': 'depth+path+pearl'},
    {'memory': True, 'stack': False, 'encoding': 'depth+path+pearl'},
    {'memory': False, 'stack': True, 'encoding': 'depth+path+pearl'},
    # ... 16 configurations
]

for config in ablation_configs:
    model = build_model(config)
    trainer = train(model)
    results = evaluate(model)
    log_results(config, results)
```

**Expected Results**:
- **Memory module**: +5-8% accuracy on depth 10-14 (better rule application)
- **Stack augmentation**: +3-5% accuracy on depth 11-14 (better extrapolation)
- **PEARL encoding**: +2-3% accuracy on depth 11-14 (better positional awareness)
- **Multi-task losses**: +4-6% accuracy overall (joint optimization improves feature learning)

---

## 4. File Modification Matrix

| File | Modification Type | Description |
|------|------------------|-------------|
| **NEW: `src/models/memory_augmented/__init__.py`** | Create | Package initialization |
| **NEW: `src/models/memory_augmented/algebraic_rule_memory.py`** | Create | AlgebraicRuleMemory class |
| **NEW: `src/models/memory_augmented/stack_augmented_gnn.py`** | Create | StackAugmentedGNN class |
| **NEW: `src/models/memory_augmented/hybrid_encoder.py`** | Create | MemoryAugmentedEncoder class |
| **NEW: `src/models/memory_augmented/positional_pearl.py`** | Create | PEARL positional encoding |
| **MODIFY: `src/models/heads.py`** | Add classes | RuleApplicationHead, AxiomHead |
| **MODIFY: `src/models/encoder.py`** | Import | Add import for MemoryAugmentedEncoder |
| **MODIFY: `src/models/full_model.py`** | Add branch | Handle `encoder_type='memory_augmented'` |
| **MODIFY: `src/constants.py`** | Add constants | Memory/stack/axiom constants |
| **MODIFY: `src/training/losses.py`** | Add function | `memory_augmented_loss()` |
| **NEW: `src/training/memory_augmented_trainer.py`** | Create | MemoryAugmentedTrainer class |
| **MODIFY: `src/data/dataset.py`** | Extend | Load auxiliary labels from JSONL |
| **NEW: `scripts/generate_auxiliary_labels.py`** | Create | Generate rule/axiom labels |
| **NEW: `scripts/train_memory_augmented.py`** | Create | Training entry point |
| **NEW: `scripts/ablate_memory_augmented.py`** | Create | Ablation study script |
| **NEW: `configs/memory_augmented.yaml`** | Create | Configuration file |
| **NEW: `tests/test_memory_augmented.py`** | Create | Unit tests |
| **NEW: `docs/MEMORY_AUGMENTED.md`** | Create | Architecture documentation |

**Backward Compatibility Strategy**:
- All new functionality gated by config flags
- Existing encoders (GAT, GGNN, HGT, RGCN) unchanged
- Existing datasets work without auxiliary labels (default to zeros)
- Existing training scripts unaffected

---

## 5. Risk Assessment & Mitigation

### 5.1 High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Memory module overfitting** | High | High | Regularization (dropout 0.3 in memory attention), curriculum learning starting from depth 2 |
| **Stack implementation complexity** | Medium | High | Start with simplified soft stack, defer hard stack to Phase 2 |
| **Auxiliary label generation errors** | Medium | Medium | Extensive unit testing, manual validation on 100 samples |
| **Training instability (multi-task)** | Medium | Medium | Gradual loss weight annealing (start with standard losses, add auxiliary losses after 5 epochs) |
| **Increased training time** | High | Low | Acceptable (multi-task training ~1.3× slower, still completes in ~2 days on single GPU) |

### 5.2 Parameter Budget

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| RGCN (2 layers) | ~10M | 35% |
| HGT (3 layers) | ~18M | 63% |
| Memory module | ~0.065M | 0.2% |
| Stack module | ~0.5M | 1.8% |
| **Total Encoder** | **~28.5M** | **100%** |
| Decoder (unchanged) | ~12M | - |
| **Full Model** | **~40.5M** | - |

**Comparison**:
- Standard model (GAT): ~15M params
- This model: ~40.5M params (2.7× larger)
- Scaled model (HGT): ~360M params (8.9× larger than this)

**Conclusion**: Fits between standard and scaled models. Trainable on single GPU (RTX 3090) with batch size 16.

### 5.3 Compatibility Matrix

| System Component | Compatible? | Notes |
|-----------------|------------|-------|
| Phase 1 (contrastive) | ✓ | Can use memory encoder for contrastive pretraining |
| Phase 2 (supervised) | ✓ | Primary training phase with multi-task losses |
| Phase 3 (RL/PPO) | ✓ | Can use memory encoder with existing RL trainer |
| Beam search inference | ✓ | Encoder output compatible with existing decoder |
| HTPS inference | ✓ | ValueHead provides critic for UCB search |
| Verification pipeline | ✓ | No changes to verification (3-tier: syntax→exec→Z3) |
| Existing datasets | ✓ | Backward compatible (defaults for missing labels) |
| Ablation framework | ✓ | Extends encoder registry pattern |

---

## 6. Testing & Validation Strategy

### 6.1 Unit Tests

**Test Coverage** (target: >90%):
```python
# tests/test_memory_augmented.py

def test_algebraic_rule_memory_forward():
    """Memory retrieval produces correct shapes."""
    memory = AlgebraicRuleMemory(num_slots=256, slot_dim=256)
    node_features = torch.randn(100, 256)
    batch = torch.zeros(100, dtype=torch.long)
    output = memory(node_features, batch)
    assert output.shape == (100, 256)

def test_stack_augmented_gnn_push_pop():
    """Stack operations maintain differentiability."""
    stack = StackAugmentedGNN(hidden_dim=256)
    x = torch.randn(50, 256)
    edge_index = torch.randint(0, 50, (2, 100))
    edge_type = torch.randint(0, 8, (100,))
    output = stack(x, edge_index, edge_type)
    assert output.shape == (50, 256)
    assert output.requires_grad

def test_memory_augmented_encoder_interface():
    """Encoder satisfies BaseEncoder interface."""
    encoder = MemoryAugmentedEncoder(hidden_dim=256)
    assert encoder.requires_edge_types == True
    assert encoder.requires_node_features == False

def test_rule_application_head():
    """RuleApplicationHead produces correct logits."""
    head = RuleApplicationHead(d_model=256, num_rule_classes=20)
    x = torch.randn(100, 256)
    logits = head(x)
    assert logits.shape == (100, 20)
```

### 6.2 Integration Tests

**Test Scenarios**:
1. **End-to-end forward pass**: Input graph → memory encoder → decoder → output tokens
2. **Multi-task loss computation**: All loss terms computed without errors
3. **Backward pass**: Gradients flow through memory module
4. **Training step**: One epoch completes without crashes
5. **Inference**: Generate simplified expression from obfuscated input

### 6.3 Ablation Validation

**Baseline Comparison**:
- **Baseline 1**: HGTEncoder (12 layers) - current best encoder
- **Baseline 2**: RGCNEncoder (12 layers) - alternative scaled encoder
- **Baseline 3**: GATJKNetEncoder (4 layers) - standard encoder

**Statistical Significance**: Run 5 independent trials per configuration, report mean ± std, Welch's t-test (p < 0.05).

---

## 7. Implementation Timeline

| Week | Milestone | Deliverables | Risks |
|------|-----------|--------------|-------|
| **1** | Core memory module | `algebraic_rule_memory.py`, `hybrid_encoder.py`, unit tests | Memory initialization, attention mechanism |
| **2** | Stack & positional encodings | `stack_augmented_gnn.py`, `positional_pearl.py`, integration tests | Soft stack differentiability |
| **3** | Auxiliary heads & losses | `heads.py` modifications, `losses.py` additions, multi-task trainer | Loss balancing, gradient conflicts |
| **4** | Data augmentation | `generate_auxiliary_labels.py`, extended dataset, validation | Label generation accuracy |
| **5** | Training infrastructure | `memory_augmented_trainer.py`, config files, training script | Training stability |
| **6** | Ablation & evaluation | Ablation script, results analysis, documentation | Time constraints |

**Critical Path**: Weeks 1-2 (core modules) → Week 3 (multi-task losses) → Week 5 (training).

**Buffer**: Week 6 can be compressed if needed (ablation can run in parallel with documentation).

---

## 8. Success Criteria

### 8.1 Functional Requirements

- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass (end-to-end forward/backward)
- [ ] Model trains without NaN/Inf losses
- [ ] Inference produces valid expressions (pass syntax verification)

### 8.2 Performance Requirements

| Metric | Target | Baseline (HGT) |
|--------|--------|----------------|
| **Accuracy (depth 2-4)** | >95% | 95% |
| **Accuracy (depth 5-7)** | >90% | 88% |
| **Accuracy (depth 8-10)** | >85% | 80% |
| **Accuracy (depth 11-14)** | >75% | 65% (extrapolation) |
| **Rule prediction accuracy** | >80% | N/A (new) |
| **Training time (50 epochs)** | <48 hours | 36 hours |
| **Inference latency** | <200ms | 150ms |

**Key Improvement**: +10% accuracy on depth 11-14 (extrapolation via memory and stack).

### 8.3 Ablation Requirements

- [ ] Memory module provides statistically significant improvement (p < 0.05)
- [ ] Stack augmentation improves extrapolation (depth 11-14)
- [ ] Multi-task losses improve overall accuracy
- [ ] PEARL encoding provides marginal benefit (≥2%)

---

## 9. Documentation & Knowledge Transfer

**Documentation Deliverables**:
1. **Architecture doc**: `docs/MEMORY_AUGMENTED.md` (architecture diagrams, design rationale)
2. **Training guide**: `docs/TRAINING_MEMORY_AUGMENTED.md` (hyperparameters, curriculum, troubleshooting)
3. **API reference**: Docstrings for all new classes/functions
4. **README update**: Add memory-augmented section to main README

**Code Review Checkpoints**:
- End of Week 2: Core modules
- End of Week 4: Data pipeline
- End of Week 5: Full training system

**Knowledge Transfer**:
- Weekly progress reports with code samples
- Final presentation with ablation results
- Jupyter notebook with inference examples

---

## 10. Future Extensions (Out of Scope)

**Deferred to Future Work**:
1. **Process Reward Model** (PRM): Train separate model to predict reward at each step (requires large-scale RL training)
2. **HTPS Online Learning**: Update memory module during HTPS search (requires inference-time optimization)
3. **Hard Stack Implementation**: Replace soft stack with discrete operations (requires REINFORCE or Gumbel-Softmax)
4. **Cross-Domain Transfer**: Train on MBA, test on polynomial simplification (requires new dataset)
5. **Symbolic Memory Initialization**: Initialize memory with known rules (requires symbolic rule extraction)

---

## Appendix A: Key Design Decisions

### A.1 Why Hybrid (RGCN+HGT) Instead of HGT-Only?

**Rationale**:
- RGCN (2 layers) provides strong local pattern recognition at lower computational cost
- HGT (3 layers) captures global dependencies without over-smoothing
- Total 5 layers vs. HGT 12 layers → 2.4× faster training, 50% fewer parameters

**Tradeoff**: Slightly lower maximum capacity, but mitigated by memory module providing learned rule context.

### A.2 Why Soft Stack Instead of Hard Stack?

**Rationale**:
- Hard stack requires discrete operations (not differentiable)
- REINFORCE/Gumbel-Softmax adds training complexity
- Soft stack with continuous pointer is differentiable, easier to train

**Tradeoff**: Less interpretable than hard stack, but sufficient for learning implicit tree traversal.

### A.3 Why 256 Memory Slots?

**Rationale**:
- Algebraic rules for MBAs: ~20-50 common patterns
- Each slot can represent multiple similar rules via clustering
- 256 slots provides 5-10× redundancy for robustness

**Tradeoff**: More slots = more parameters, but 256 slots is only 0.065M params (negligible).

### A.4 Why Multi-Task Training Instead of Sequential?

**Rationale**:
- Multi-task learning improves feature learning via shared representations
- Rule prediction provides supervision at node level (denser signal than sequence-level)
- Contrastive loss encourages invariance to obfuscation style

**Tradeoff**: More complex training (need to balance loss weights), but expected +5% accuracy gain justifies complexity.

---

## Appendix B: Hyperparameter Recommendations

| Hyperparameter | Recommended Value | Range | Notes |
|----------------|------------------|-------|-------|
| `memory_num_slots` | 256 | [128, 512] | 256 sufficient for MBAs |
| `memory_attn_heads` | 4 | [2, 8] | 4 heads balance capacity/speed |
| `stack_max_depth` | 20 | [15, 30] | Max tree depth = 14, add buffer |
| `rule_loss_weight` | 0.2 | [0.1, 0.5] | Rule prediction important |
| `contrastive_weight` | 0.1 | [0.05, 0.2] | Contrastive less critical |
| `axiom_weight` | 0.05 | [0.01, 0.1] | Axiom satisfaction is regularization |
| `learning_rate` | 1e-4 | [5e-5, 5e-4] | Lower than Phase 2 (3e-4) |
| `batch_size` | 16 | [8, 32] | Constrained by GPU memory |
| `warmup_epochs` | 5 | [3, 10] | Gradual loss weight annealing |

**Loss Weight Annealing Schedule**:
- Epochs 1-5: Standard losses only (ce, complexity, copy)
- Epochs 6-10: Add rule loss (weight 0.2)
- Epochs 11-15: Add contrastive loss (weight 0.1)
- Epochs 16+: Add axiom loss (weight 0.05)

**Rationale**: Gradual introduction prevents auxiliary losses from destabilizing primary task.

---

This plan provides a comprehensive, actionable roadmap for implementing memory-augmented GNNs with algebraic rule memory in the MBA deobfuscation system. The phased approach with clear milestones, risk mitigation, and backward compatibility ensures successful integration into the existing codebase.
