# Phase 4: GMN Training Infrastructure Implementation Plan

## Overview

Phase 4 extends the existing training pipeline with Graph Matching Network (GMN) capabilities. GMN learns to directly compare expression graph pairs for semantic equivalence, complementing the encoder-decoder architecture with explicit similarity scoring.

**Training Strategy:**
- **Phase 1a**: Standard contrastive pretraining (existing Phase1Trainer)
- **Phase 1b**: GMN training with frozen encoder (new Phase1bGMNTrainer)
- **Phase 1c**: End-to-end fine-tuning with unfrozen encoder (optional)

**Architecture Integration:**
```
Phase 1a: HGT/GAT Encoder → Contrastive Loss (InfoNCE + MaskLM)
          ↓ (freeze weights)
Phase 1b: HGT/GAT Encoder → GMN Cross-Attention → Match Score → BCE Loss
          ↓ (optional unfreeze)
Phase 1c: HGT/GAT Encoder → GMN Cross-Attention → Match Score → BCE Loss (end-to-end)
```

---

## 1. Phase1bGMNTrainer (Frozen Encoder Training)

### Location
`src/training/phase1b_gmn_trainer.py`

### Purpose
Train GMN cross-attention layers while keeping the pre-trained encoder frozen. Learns graph pair matching without disrupting learned representations from Phase 1a.

### Class Specification

```python
class Phase1bGMNTrainer(BaseTrainer):
    """
    Phase 1b: GMN training with frozen encoder.

    Trains cross-attention layers for graph matching while encoder remains frozen.
    Uses binary classification loss with hard negative mining.
    """

    def __init__(
        self,
        model: Union[HGTWithGMN, GATWithGMN],
        config: Dict[str, Any],
        negative_sampler: NegativeSampler,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 1b GMN trainer.

        Args:
            model: HGTWithGMN or GATWithGMN instance with frozen encoder
            config: Training configuration with:
                - learning_rate: float (default: 3e-5, lower than Phase 1a)
                - weight_decay: float (default: 0.01)
                - bce_pos_weight: float (default: 1.0, balance positive/negative)
                - triplet_loss_margin: Optional[float] (default: None, disable triplet)
                - triplet_loss_weight: float (default: 0.1)
                - hard_negative_ratio: float (default: 0.3, fraction of hard negatives)
                - gradient_accumulation_steps: int
                - warmup_steps: int
                - max_grad_norm: float
                - scheduler_type: str
            negative_sampler: NegativeSampler for generating hard negatives
            device: Training device
            checkpoint_dir: Checkpoint save directory

        Raises:
            ValueError: If encoder is not frozen
            ValueError: If model is not HGTWithGMN or GATWithGMN
        """
```

### Key Methods

#### `train_step(batch: Dict[str, Any]) -> Dict[str, float]`

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Single training step for GMN.

    Args:
        batch: Collated batch from GMNBatchCollator with keys:
            - graph1_batch: PyG Batch (obfuscated expressions)
            - graph2_batch: PyG Batch (candidate simplified expressions)
            - labels: [batch_size] float tensor (1.0=equivalent, 0.0=not)
            - pair_indices: [batch_size, 2] mapping indices

    Returns:
        Dict with keys:
            - 'total': Total loss
            - 'bce': Binary cross-entropy loss
            - 'triplet': Triplet loss (if enabled)
            - 'accuracy': Binary accuracy
            - 'pos_score': Average score on positive pairs
            - 'neg_score': Average score on negative pairs

    Flow:
        1. Verify encoder is frozen (assertion check)
        2. Move batch to device
        3. Forward pass through GMN: match_scores = model.forward_pair(graph1, graph2)
        4. Compute BCE loss with pos_weight balancing
        5. (Optional) Sample triplets and compute triplet loss
        6. Backward pass with gradient accumulation
        7. Clip gradients and update weights
        8. Compute metrics (accuracy, avg scores by label)
        9. Return loss dict
    """
```

**Implementation Notes:**
- Assert `model.is_encoder_frozen` at start of each step
- Use `BCEWithLogitsLoss` for numerical stability (accepts logits, not probabilities)
- GMN outputs are already sigmoid-activated, so convert back: `logits = torch.logit(match_scores, eps=1e-7)`
- Positive weight balancing: `pos_weight = bce_pos_weight` handles class imbalance
- Hard gradient clipping prevents GMN attention weight explosion (observed in ablation tests)

#### `evaluate(dataloader: DataLoader) -> Dict[str, float]`

```python
@torch.no_grad()
def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
    """
    Evaluate GMN on validation set.

    Args:
        dataloader: Validation DataLoader with GMNBatchCollator

    Returns:
        Dict with keys:
            - 'bce': Average BCE loss
            - 'accuracy': Binary classification accuracy
            - 'precision': Precision on positive class
            - 'recall': Recall on positive class
            - 'f1': F1 score
            - 'auc': Area under ROC curve (if sklearn available)
            - 'pos_score_mean': Mean score on positive pairs
            - 'pos_score_std': Std score on positive pairs
            - 'neg_score_mean': Mean score on negative pairs
            - 'neg_score_std': Std score on negative pairs
            - 'separation_gap': pos_score_mean - neg_score_mean (should be large)

    Flow:
        1. Set model to eval mode
        2. Iterate over validation batches
        3. Forward pass, accumulate predictions and labels
        4. Compute classification metrics (accuracy, precision, recall, F1)
        5. Compute score statistics (mean, std per class)
        6. Compute separation gap (key metric for contrastive learning)
        7. (Optional) Compute AUC if sklearn available
    """
```

**Implementation Notes:**
- Use threshold=0.5 for binary accuracy
- Separate positive and negative pairs for score analysis
- Separation gap is primary metric for Phase 1b (replaces sim_gap from Phase 1a)
- Precision/recall computed with sklearn if available, otherwise manual computation

#### `save_checkpoint()` and `load_checkpoint()`

```python
def save_checkpoint(
    self,
    filename: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    is_best: bool = False,
) -> str:
    """
    Save checkpoint including GMN state.

    Saves:
        - model.state_dict(): Full GMN model (encoder + cross-attention + match_score)
        - optimizer.state_dict()
        - scheduler.state_dict()
        - global_step, epoch, best_metric
        - config: Training config
        - gmn_config: GMN architecture config (for model reconstruction)
        - hgt_checkpoint_path: Original Phase 1a checkpoint path (reference)
        - metrics: Current evaluation metrics
    """

def load_checkpoint(
    self,
    checkpoint_path: str,
    load_optimizer: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint with GMN state.

    Args:
        checkpoint_path: Path to checkpoint
        load_optimizer: Whether to restore optimizer state
        strict: Strict state dict loading

    Returns:
        Checkpoint dict

    Raises:
        RuntimeError: If encoder is not frozen after loading
    """
```

**Checkpoint Compatibility:**
- Save both full model state and GMN config for reproducibility
- Reference to Phase 1a checkpoint enables traceability
- Verify encoder freeze state after loading (critical for Phase 1b)

---

## 2. Phase1cGMNTrainer (End-to-End Fine-Tuning)

### Location
`src/training/phase1c_gmn_trainer.py`

### Purpose
Fine-tune entire GMN model (encoder + cross-attention) end-to-end. Optional phase after Phase 1b for performance improvement on complex expressions.

### Class Specification

```python
class Phase1cGMNTrainer(Phase1bGMNTrainer):
    """
    Phase 1c: End-to-end GMN fine-tuning.

    Inherits from Phase1bGMNTrainer but unfreezes encoder for joint optimization.
    Uses lower learning rate for encoder to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        model: Union[HGTWithGMN, GATWithGMN],
        config: Dict[str, Any],
        negative_sampler: NegativeSampler,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 1c GMN trainer.

        Args:
            model: HGTWithGMN or GATWithGMN instance
            config: Training configuration with additional keys:
                - encoder_learning_rate: float (default: 1e-5, 10x lower than GMN)
                - encoder_weight_decay: float (default: 0.001, lower regularization)
                - unfreeze_encoder: bool (default: True)
                - All Phase1bGMNTrainer config keys
            negative_sampler: NegativeSampler instance
            device: Training device
            checkpoint_dir: Checkpoint save directory

        Raises:
            ValueError: If Phase 1b checkpoint not provided
        """
```

### Key Differences from Phase1bGMNTrainer

#### `_init_optimizer() -> torch.optim.Optimizer`

```python
def _init_optimizer(self) -> torch.optim.Optimizer:
    """
    Initialize optimizer with separate learning rates for encoder and GMN.

    Parameter groups:
        1. Encoder parameters: lower LR (encoder_learning_rate)
        2. GMN parameters: standard LR (learning_rate)
        3. Separate weight decay for each group

    Returns:
        AdamW optimizer with parameter groups

    Implementation:
        optimizer_grouped_parameters = [
            {
                'params': encoder_params_decay,
                'lr': encoder_learning_rate,
                'weight_decay': encoder_weight_decay,
            },
            {
                'params': encoder_params_no_decay,
                'lr': encoder_learning_rate,
                'weight_decay': 0.0,
            },
            {
                'params': gmn_params_decay,
                'lr': learning_rate,
                'weight_decay': weight_decay,
            },
            {
                'params': gmn_params_no_decay,
                'lr': learning_rate,
                'weight_decay': 0.0,
            },
        ]
    """
```

**Rationale:**
- Lower encoder LR prevents catastrophic forgetting of Phase 1a representations
- Standard GMN LR allows continued learning of cross-attention patterns
- Typical ratio: encoder_lr = 0.1 * gmn_lr

#### `unfreeze_encoder()` (CRITICAL: Gradient Accumulation Safety)

```python
def unfreeze_encoder(self):
    """
    Unfreeze encoder for end-to-end fine-tuning.

    CRITICAL: Must clear optimizer state and reset gradient accumulation
    to prevent stale frozen-phase gradients from corrupting updates.
    """
    # CRITICAL: Clear optimizer state before unfreezing to prevent gradient corruption
    self.optimizer.zero_grad(set_to_none=True)

    for param in self.model.hgt_encoder.parameters():
        param.requires_grad = True
        # Ensure no stale gradients exist (more explicit than zero_())
        if param.grad is not None:
            param.grad = None

    self.model._encoder_frozen = False
    logger.info("Encoder unfrozen for fine-tuning")

    # Reset gradient accumulation counter if trainer tracks it
    if hasattr(self, '_grad_accum_step'):
        self._grad_accum_step = 0
```

**Critical Notes:**
- NEVER unfreeze mid-accumulation: always call `optimizer.zero_grad(set_to_none=True)` first
- Set `param.grad = None` (not `zero_()`) to fully clear stale gradients
- Reset any gradient accumulation counters to prevent mixing frozen/unfrozen gradients

#### `train_step(batch: Dict[str, Any]) -> Dict[str, float]`

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
    """
    Single training step with unfrozen encoder.

    Same signature as Phase1bGMNTrainer.train_step(), but:
        - Removes encoder freeze assertion
        - Adds gradient norm tracking for encoder vs GMN layers
        - Monitors encoder parameter drift (L2 distance from Phase 1b checkpoint)

    Returns:
        Extended loss dict with:
            - All Phase1bGMNTrainer keys
            - 'encoder_grad_norm': L2 norm of encoder gradients
            - 'gmn_grad_norm': L2 norm of GMN gradients
            - 'encoder_drift': L2 distance from Phase 1b checkpoint (every 100 steps)
    """
```

**Monitoring:**
- Track encoder drift to detect catastrophic forgetting
- If encoder_drift > threshold (e.g., 0.5), early stop or reduce encoder_lr
- Separate gradient norms help diagnose vanishing/exploding gradients

---

## 3. NegativeSampler (Hard Negative Mining)

### Location
`src/training/negative_sampler.py`

### Purpose
Generate challenging negative pairs for GMN training. Random negatives are too easy; hard negatives force GMN to learn fine-grained equivalence distinctions.

### Class Specification

```python
class NegativeSampler:
    """
    Hard negative sampler with Z3 verification.

    Generates non-equivalent expression pairs that are syntactically similar
    or share structure, forcing GMN to learn semantic distinctions.

    Strategies:
        1. Random sampling: Pick random expressions from dataset
        2. Syntactic similarity: Find expressions with similar AST structure
        3. Depth matching: Sample expressions with same depth
        4. Variable swapping: Swap variables in equivalent expressions
    """

    def __init__(
        self,
        dataset: List[Dict[str, str]],
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        z3_timeout_ms: int = 500,
        cache_size: int = 10000,
        num_workers: int = 4,
        device: Optional[torch.device] = None,  # ADDED: device parameter
    ):
        """
        Initialize negative sampler.

        Args:
            dataset: List of {'obfuscated': str, 'simplified': str, 'depth': int}
            tokenizer: MBATokenizer for parsing
            fingerprint: SemanticFingerprint for computing features
            z3_timeout_ms: Z3 verification timeout (default: 500ms)
            cache_size: Size of verified negative pair cache
            num_workers: Parallel Z3 verification workers
            device: Device for fingerprint computation (default: CPU for Z3 compatibility)

        Initialization:
            1. Build expression index by depth
            2. Compute fingerprints for all expressions (cached)
            3. Build syntactic similarity index (AST structure hashing)
            4. Initialize LRU cache for verified pairs

        Device Handling:
            Default to CPU for Z3 verification compatibility. If CUDA tensors are used
            for fingerprints and later compared to CPU strings, device mismatch occurs.
        """
        self.device = device if device is not None else torch.device('cpu')
        # Ensure fingerprint computations use correct device
        self.fingerprint.device = self.device
```

### Key Methods

#### `sample_negative(obf_expr: str, simp_expr: str, strategy: str = 'mixed') -> Tuple[str, str, str]`

```python
def sample_negative(
    self,
    obf_expr: str,
    simp_expr: str,
    strategy: str = 'mixed',
) -> Tuple[str, str, str]:
    """
    Sample a negative pair for given (obf, simp) positive pair.

    Args:
        obf_expr: Obfuscated expression (anchor)
        simp_expr: Simplified expression (positive)
        strategy: Sampling strategy ('random', 'syntactic', 'depth', 'variable', 'mixed')

    Returns:
        Tuple of (obf_expr, negative_expr, negative_type):
            - obf_expr: Anchor expression (unchanged)
            - negative_expr: Non-equivalent expression
            - negative_type: Type of negative ('random', 'syntactic', 'depth', 'variable')

    Strategy details:
        - 'random': Pick random expression from dataset
        - 'syntactic': Find expression with similar AST pattern (swap operators)
        - 'depth': Sample expression with same AST depth
        - 'variable': Swap variable names in simp_expr (breaks equivalence)
        - 'mixed': Randomly choose strategy (30% random, 40% syntactic, 20% depth, 10% variable)

    Verification:
        1. Sample candidate negative
        2. Check cache for (obf_expr, negative_expr) equivalence result
        3. If not cached, verify with Z3 (timeout: z3_timeout_ms)
        4. If timeout or equivalent, retry with different candidate (max 3 attempts)
        5. Cache result (both positive and negative outcomes)
        6. Return verified negative pair

    Error Handling:
        - If Z3 timeout after 3 attempts: fall back to random strategy
        - If all strategies exhausted: return random expression (logged as warning)
        - Cache timeout results separately (don't count as verified)
    """
```

**Implementation Notes:**
- Syntactic similarity: Use AST structure hash (operator sequence, ignoring operands)
- Variable swapping: Replace 'x' with 'y', 'y' with 'z', etc. (guaranteed to break equivalence)
- Z3 timeout handling: Count timeout as "unknown", retry with different candidate
- Cache key: `(expr1_normalized, expr2_normalized)` where normalized = sorted variables, whitespace removed

#### `_verify_equivalence_with_timeout(expr1: str, expr2: str) -> Optional[bool]`

```python
def _verify_equivalence_with_timeout(
    self,
    expr1: str,
    expr2: str,
) -> Optional[bool]:
    """
    Verify equivalence with Z3, handling timeouts gracefully.

    Args:
        expr1: First expression
        expr2: Second expression

    Returns:
        True if equivalent, False if not equivalent, None if timeout/error

    Implementation:
        1. Check LRU cache for cached result
        2. If not cached, call Z3 with timeout
        3. Handle three outcomes:
            - UNSAT (equivalent): Return True, cache result
            - SAT (counterexample found): Return False, cache result
            - TIMEOUT/ERROR: Return None, cache with expiration (1 hour)
        4. Update cache statistics (hits, misses, timeouts)

    Error Handling:
        - Syntax errors: Return False (treat as non-equivalent)
        - Z3 solver errors: Return None (timeout-like)
        - OOM errors: Return None and log warning
    """
```

**Cache Structure:**
```python
cache: Dict[Tuple[str, str], Tuple[Optional[bool], float]] = {}
# Key: (expr1_normalized, expr2_normalized)
# Value: (equivalence_result, timestamp)
# LRU eviction when cache_size exceeded
```

#### `batch_sample_negatives(positive_pairs: List[Tuple[str, str]], strategy: str = 'mixed') -> List[Tuple[str, str, str]]`

```python
def batch_sample_negatives(
    self,
    positive_pairs: List[Tuple[str, str]],
    strategy: str = 'mixed',
) -> List[Tuple[str, str, str]]:
    """
    Batch sample negatives for multiple positive pairs (parallelized).

    CRITICAL: Uses map_async with timeout to prevent worker deadlock.

    Args:
        positive_pairs: List of (obf_expr, simp_expr) tuples
        strategy: Sampling strategy

    Returns:
        List of (obf_expr, negative_expr, negative_type) tuples

    Implementation:
        1. Use multiprocessing pool for parallel Z3 verification
        2. Distribute candidates across workers
        3. Use map_async with timeout (not blocking map)
        4. Handle TimeoutError with fallback to sequential sampling
        5. Return verified negative pairs
    """
    import multiprocessing.pool
    from functools import partial

    def sample_with_timeout(pair_and_strategy):
        pair, strat = pair_and_strategy
        return self.sample_negative(pair[0], pair[1], strategy=strat)

    # Create pool with timeout context
    with multiprocessing.Pool(processes=self.num_workers) as pool:
        inputs = [(pair, strategy) for pair in positive_pairs]
        try:
            # CRITICAL: Use map_async with timeout to prevent deadlock
            # Timeout = (z3_timeout + safety_margin) * num_pairs / num_workers
            timeout = (self.z3_timeout_ms / 1000 + 5) * len(positive_pairs) / self.num_workers
            result = pool.map_async(sample_with_timeout, inputs)
            negatives = result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            logger.error(f"Pool timeout after {timeout}s, terminating workers")
            pool.terminate()
            pool.join()
            # Fallback: sequential sampling (slower but reliable)
            logger.warning("Falling back to sequential negative sampling")
            negatives = [self.sample_negative(p[0], p[1], strategy) for p in positive_pairs]
        except Exception as e:
            logger.error(f"Pool error: {e}, falling back to sequential")
            pool.terminate()
            pool.join()
            negatives = [self.sample_negative(p[0], p[1], strategy) for p in positive_pairs]

    return negatives
```

**Multiprocessing Notes:**
- CRITICAL: Use `map_async().get(timeout=...)` instead of blocking `pool.map()`
- Timeout formula: `(z3_timeout_ms/1000 + 5) * len(pairs) / num_workers`
- On timeout or worker crash, fallback to sequential sampling (slower but won't hang)
- Share cache across workers via `Manager().dict()` (thread-safe)

### Statistics Tracking

```python
@property
def stats(self) -> Dict[str, Any]:
    """
    Return sampler statistics.

    Returns:
        Dict with keys:
            - 'cache_hits': int (number of cache hits)
            - 'cache_misses': int (number of cache misses)
            - 'cache_hit_rate': float (hits / (hits + misses))
            - 'z3_timeouts': int (number of Z3 timeouts)
            - 'z3_errors': int (number of Z3 errors)
            - 'samples_by_strategy': Dict[str, int] (count per strategy)
            - 'cache_size': int (current cache size)
    """
```

---

## 4. GMN Loss Functions

### Location
`src/training/losses.py` (extend existing file)

### New Loss Functions

#### `gmn_bce_loss(match_scores: torch.Tensor, labels: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor`

```python
def gmn_bce_loss(
    match_scores: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for GMN graph pair matching.

    This is the ONLY function that should handle sigmoid→logit conversion.
    All other code should call this function rather than reimplementing.

    Args:
        match_scores: [batch_size, 1] predicted match scores (sigmoid-activated, 0-1)
        labels: [batch_size] ground truth labels (1.0=equivalent, 0.0=not)
        pos_weight: Positive class weight (default: 1.0, increase if class imbalance)

    Returns:
        Scalar BCE loss

    Implementation:
        # CRITICAL: Clamp match_scores BEFORE logit conversion to prevent numerical issues
        # Sigmoid outputs can saturate to exact 0.0 or 1.0, causing logit to fail
        match_scores_clamped = match_scores.squeeze(-1).clamp(min=1e-7, max=1 - 1e-7)

        # Convert to logits (now safe because inputs are in (1e-7, 1-1e-7))
        logits = torch.logit(match_scores_clamped, eps=1e-7)  # eps redundant but kept for safety

        # Weighted BCE with device-aware pos_weight
        pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        loss = loss_fn(logits, labels)

        return loss

    Notes:
        - CRITICAL: clamp() before logit() prevents division by zero when sigmoid saturates
        - pos_weight balances precision/recall tradeoff
        - pos_weight > 1.0: prioritize recall (catch more true positives)
        - pos_weight < 1.0: prioritize precision (reduce false positives)
        - Typical: pos_weight = neg_count / pos_count for balanced accuracy
    """
```

#### `gmn_triplet_loss(anchor_scores: torch.Tensor, positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float = 0.2) -> torch.Tensor`

```python
def gmn_triplet_loss(
    anchor_scores: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet loss for GMN training (optional, supplementary to BCE).

    Encourages: score(anchor, positive) > score(anchor, negative) + margin

    Args:
        anchor_scores: [batch_size, 1] scores for (anchor, anchor) pairs
        positive_scores: [batch_size, 1] scores for (anchor, positive) pairs
        negative_scores: [batch_size, 1] scores for (anchor, negative) pairs
        margin: Triplet margin (default: 0.2)

    Returns:
        Scalar triplet loss

    Implementation:
        # Triplet loss: max(0, score(A,N) - score(A,P) + margin)
        # GMN outputs are similarity scores in [0, 1], so higher is more similar
        loss = F.relu(negative_scores - positive_scores + margin).mean()

        return loss

    Notes:
        - Only use if triplet_loss_weight > 0 in config
        - Margin typically 0.1-0.3 (depends on score scale)
        - Combines with BCE loss: total = bce + triplet_weight * triplet
    """
```

#### `gmn_combined_loss(match_scores: torch.Tensor, labels: torch.Tensor, triplet_data: Optional[Dict] = None, pos_weight: float = 1.0, triplet_weight: float = 0.1, triplet_margin: float = 0.2) -> Dict[str, torch.Tensor]`

```python
def gmn_combined_loss(
    match_scores: torch.Tensor,
    labels: torch.Tensor,
    triplet_data: Optional[Dict[str, torch.Tensor]] = None,
    pos_weight: float = 1.0,
    triplet_weight: float = 0.1,
    triplet_margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Combined GMN loss: BCE + optional triplet loss.

    Args:
        match_scores: [batch_size, 1] predicted match scores
        labels: [batch_size] ground truth labels
        triplet_data: Optional dict with keys:
            - 'anchor_scores': [batch_size, 1]
            - 'positive_scores': [batch_size, 1]
            - 'negative_scores': [batch_size, 1]
        pos_weight: BCE positive weight
        triplet_weight: Weight for triplet loss term
        triplet_margin: Triplet margin

    Returns:
        Dict with keys:
            - 'total': Total loss
            - 'bce': BCE loss component
            - 'triplet': Triplet loss component (0 if not enabled)

    Implementation:
        bce = gmn_bce_loss(match_scores, labels, pos_weight)

        if triplet_data is not None:
            triplet = gmn_triplet_loss(
                triplet_data['anchor_scores'],
                triplet_data['positive_scores'],
                triplet_data['negative_scores'],
                triplet_margin,
            )
            total = bce + triplet_weight * triplet
        else:
            triplet = torch.tensor(0.0)
            total = bce

        return {'total': total, 'bce': bce, 'triplet': triplet}
    """
```

---

## 5. GMNDataset (Graph Pair Dataset)

### Location
`src/data/dataset.py` (extend existing file)

### Class Specification

```python
class GMNDataset(Dataset):
    """
    Dataset for GMN training with negative sampling.

    Provides (graph1, graph2, label) tuples where:
        - label=1: graph1 and graph2 are equivalent expressions
        - label=0: graph1 and graph2 are non-equivalent
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        negative_sampler: NegativeSampler,
        negative_ratio: float = 1.0,
        max_depth: Optional[int] = None,
    ):
        """
        Initialize GMN dataset.

        Args:
            data_path: Path to JSONL file with positive pairs
            tokenizer: MBATokenizer instance
            fingerprint: SemanticFingerprint instance
            negative_sampler: NegativeSampler for generating negatives
            negative_ratio: Negatives per positive (default: 1.0, balanced dataset)
            max_depth: Optional depth filter

        Dataset construction:
            1. Load positive pairs from JSONL: (obfuscated, simplified)
            2. For each positive, sample negative_ratio negatives
            3. Build list of (expr1, expr2, label) tuples
            4. Shuffle tuples for training
        """

    def __getitem__(self, idx: int) -> Tuple[Data, Data, int]:
        """
        Get dataset item.

        Returns:
            Tuple of (graph1, graph2, label):
                - graph1: PyG Data (first expression graph)
                - graph2: PyG Data (second expression graph)
                - label: 1 if equivalent, 0 if not
        """
```

**Dataset Balance:**
- `negative_ratio=1.0`: Balanced dataset (50% positive, 50% negative)
- `negative_ratio=2.0`: More negatives (33% positive, 67% negative)
- Adjust based on precision/recall requirements

### Integration with GMNBatchCollator

```python
# In src/data/collate.py (extend existing file)

def collate_gmn_pairs(batch_list: List[Tuple[Data, Data, int]]) -> Dict[str, Any]:
    """
    Collate function for GMN dataset.

    Args:
        batch_list: List of (graph1, graph2, label) tuples

    Returns:
        Dict with keys:
            - 'graph1_batch': PyG Batch
            - 'graph2_batch': PyG Batch
            - 'labels': [batch_size] tensor
            - 'pair_indices': [batch_size, 2] tensor
    """
    collator = GMNBatchCollator()
    return collator(batch_list)
```

---

## 6. Configuration Files

### Phase 1b Config: `configs/phase1b_gmn.yaml`

```yaml
# Phase 1b: GMN Training with Frozen Encoder
# Trains cross-attention layers while encoder remains frozen

model:
  # GMN wrapper type
  gmn_type: hgt_gmn  # hgt_gmn or gat_gmn

  # Base encoder checkpoint (Phase 1a output)
  encoder_checkpoint: checkpoints/phase1/phase1_best.pt

  # GMN configuration
  gmn_config:
    hidden_dim: 768  # Must match encoder output dimension
    num_attention_layers: 2  # Stack depth for cross-attention
    num_heads: 8  # Attention heads per layer
    dropout: 0.1
    aggregation: mean_max  # Graph-level pooling (mean_max, attention, mean, max)
    freeze_encoder: true  # Critical for Phase 1b

training:
  # Optimizer (lower LR than Phase 1a)
  learning_rate: 3.0e-5  # Lower LR for GMN training
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0  # Strict clipping for GMN attention stability
  gradient_accumulation_steps: 1
  scheduler_type: cosine

  # Training
  batch_size: 32  # Smaller than Phase 1a (GMN is memory-intensive)
  num_epochs: 15
  num_workers: 4

  # Phase 1b specific
  # BCE positive class weight
  # IMPORTANT: Set pos_weight = negative_ratio to balance loss when using imbalanced dataset
  # Formula: pos_weight = (num_negatives / num_positives) = negative_ratio
  # Example: negative_ratio=0.5 means 2:1 pos:neg, so set pos_weight=0.5
  # Example: negative_ratio=2.0 means 1:2 pos:neg, so set pos_weight=2.0
  # If set to 1.0, trainer will auto-adjust based on data.negative_sampler.negative_ratio
  bce_pos_weight: 1.0
  triplet_loss_margin: null  # Disable triplet loss (optional: 0.2)
  triplet_loss_weight: 0.1  # Weight if enabled
  hard_negative_ratio: 0.5  # 50% hard negatives, 50% random negatives

  # Logging
  log_interval: 50
  eval_interval: 1
  save_interval: 3

data:
  train_path: data/train.jsonl
  val_path: data/val.jsonl
  max_depth: null  # No depth filter

  # Negative sampling
  negative_sampler:
    z3_timeout_ms: 500  # Z3 verification timeout per pair
    cache_size: 10000  # LRU cache size for verified pairs
    num_workers: 4  # Parallel Z3 workers
    negative_ratio: 1.0  # Balanced dataset (1 negative per positive)
    strategy: mixed  # Sampling strategy (random, syntactic, depth, variable, mixed)

checkpoint:
  dir: checkpoints/phase1b_gmn
  resume_from: null  # Path to resume training
  load_phase1a: checkpoints/phase1/phase1_best.pt  # Phase 1a encoder checkpoint

logging:
  tensorboard: true
  log_dir: logs/phase1b_gmn
```

### Phase 1c Config: `configs/phase1c_gmn_finetune.yaml`

```yaml
# Phase 1c: End-to-End GMN Fine-Tuning
# Unfreezes encoder for joint optimization with cross-attention

model:
  gmn_type: hgt_gmn

  # Load Phase 1b checkpoint (not Phase 1a)
  encoder_checkpoint: null  # Not used (loaded from Phase 1b checkpoint)

  gmn_config:
    hidden_dim: 768
    num_attention_layers: 2
    num_heads: 8
    dropout: 0.1
    aggregation: mean_max
    freeze_encoder: false  # Critical for Phase 1c

training:
  # Dual learning rates (encoder lower than GMN)
  learning_rate: 3.0e-5  # GMN learning rate
  encoder_learning_rate: 3.0e-6  # Encoder learning rate (10x lower)
  weight_decay: 0.01
  encoder_weight_decay: 0.001  # Lower regularization for encoder
  warmup_steps: 500  # Shorter warmup (already pre-trained)
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1
  scheduler_type: cosine

  # Training
  batch_size: 16  # Smaller (full model gradients)
  num_epochs: 5  # Short fine-tuning (avoid overfitting)
  num_workers: 4

  # Phase 1c specific
  bce_pos_weight: 1.0
  triplet_loss_margin: 0.2  # Enable triplet loss for fine-tuning
  triplet_loss_weight: 0.1
  hard_negative_ratio: 0.7  # More hard negatives in fine-tuning

  # Early stopping
  early_stopping_patience: 3  # Stop if no improvement for 3 epochs
  early_stopping_metric: separation_gap  # Monitor score separation

  # Logging
  log_interval: 50
  eval_interval: 1
  save_interval: 1

data:
  train_path: data/train.jsonl
  val_path: data/val.jsonl
  max_depth: null

  negative_sampler:
    z3_timeout_ms: 500
    cache_size: 10000
    num_workers: 4
    negative_ratio: 1.5  # More negatives for fine-tuning
    strategy: mixed

checkpoint:
  dir: checkpoints/phase1c_gmn_finetune
  resume_from: null
  load_phase1b: checkpoints/phase1b_gmn/phase1b_gmn_best.pt  # Phase 1b GMN checkpoint

logging:
  tensorboard: true
  log_dir: logs/phase1c_gmn_finetune
```

---

## 7. Training Script Integration

### Location
`scripts/train.py` (extend existing file)

### Helper Functions (Refactored to Reduce God Function)

```python
def _load_gmn_model(config: dict, device: torch.device) -> Union[HGTWithGMN, GATWithGMN]:
    """
    Load GMN model from Phase 1a checkpoint.

    Args:
        config: Full config dict
        device: Target device

    Returns:
        HGTWithGMN or GATWithGMN instance

    Raises:
        ValueError: If Phase 1a checkpoint not provided or gmn_type unknown
    """
    model_cfg = config.get('model', {})
    gmn_type = model_cfg.get('gmn_type', 'hgt_gmn')
    gmn_config = model_cfg.get('gmn_config', {})
    phase1a_path = config.get('checkpoint', {}).get('load_phase1a')

    if not phase1a_path:
        raise ValueError("Phase 1b requires Phase 1a checkpoint (load_phase1a)")

    if gmn_type == 'hgt_gmn':
        from src.models.gmn import HGTWithGMN
        model = HGTWithGMN(hgt_checkpoint_path=phase1a_path, gmn_config=gmn_config)
    elif gmn_type == 'gat_gmn':
        from src.models.gmn import GATWithGMN
        model = GATWithGMN(gat_checkpoint_path=phase1a_path, gmn_config=gmn_config)
    else:
        raise ValueError(f"Unknown gmn_type: {gmn_type}")

    return model.to(device)


def _create_negative_sampler(
    config: dict,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    device: torch.device,
) -> NegativeSampler:
    """
    Create negative sampler from config.

    Args:
        config: Full config dict
        tokenizer: MBATokenizer instance
        fingerprint: SemanticFingerprint instance
        device: Device for fingerprint computation

    Returns:
        NegativeSampler instance
    """
    data_cfg = config.get('data', {})
    sampler_cfg = data_cfg.get('negative_sampler', {})

    with open(data_cfg['train_path'], 'r') as f:
        full_dataset = [json.loads(line) for line in f if line.strip()]

    return NegativeSampler(
        dataset=full_dataset,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        z3_timeout_ms=sampler_cfg.get('z3_timeout_ms', 500),
        cache_size=sampler_cfg.get('cache_size', 10000),
        num_workers=sampler_cfg.get('num_workers', 4),
        device=device,  # Pass device for fingerprint computation
    )


def _create_gmn_dataloaders(
    config: dict,
    tokenizer: MBATokenizer,
    fingerprint: SemanticFingerprint,
    negative_sampler: NegativeSampler,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for GMN.

    Args:
        config: Full config dict
        tokenizer: MBATokenizer instance
        fingerprint: SemanticFingerprint instance
        negative_sampler: NegativeSampler instance

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    sampler_cfg = data_cfg.get('negative_sampler', {})

    train_dataset = GMNDataset(
        data_path=data_cfg['train_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.0),
        max_depth=data_cfg.get('max_depth'),
    )

    val_dataset = GMNDataset(
        data_path=data_cfg['val_path'],
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        negative_sampler=negative_sampler,
        negative_ratio=sampler_cfg.get('negative_ratio', 1.0),
        max_depth=data_cfg.get('max_depth'),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 32),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_gmn_pairs,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 32),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_gmn_pairs,
        pin_memory=True,
    )

    return train_loader, val_loader
```

### Main Function: `train_phase1b()`

```python
def train_phase1b(config: dict, device: torch.device):
    """
    Run Phase 1b: GMN training with frozen encoder.

    Refactored to use helper functions for testability and clarity.
    """
    from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer
    from src.training.negative_sampler import NegativeSampler
    from src.data.dataset import GMNDataset
    from src.data.collate import collate_gmn_pairs
    from torch.utils.data import DataLoader

    logger.info("=== Phase 1b: GMN Training (Frozen Encoder) ===")

    # Load model (extracted helper)
    model = _load_gmn_model(config, device)
    logger.info(f"GMN model: {sum(p.numel() for p in model.parameters()):,} params, frozen={model.is_encoder_frozen}")

    # Create preprocessing components
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()
    negative_sampler = _create_negative_sampler(config, tokenizer, fingerprint, device)

    # Create datasets and loaders (extracted helper)
    train_loader, val_loader = _create_gmn_dataloaders(config, tokenizer, fingerprint, negative_sampler)
    logger.info(f"Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")

    # Create trainer
    training_cfg = config.get('training', {})
    checkpoint_cfg = config.get('checkpoint', {})

    trainer = Phase1bGMNTrainer(
        model=model,
        config=training_cfg,
        negative_sampler=negative_sampler,
        device=device,
        checkpoint_dir=checkpoint_cfg.get('dir', 'checkpoints/phase1b_gmn'),
    )

    # Resume if specified
    if checkpoint_cfg.get('resume_from'):
        trainer.load_checkpoint(checkpoint_cfg['resume_from'])

    # Initialize TensorBoard
    if config.get('logging', {}).get('tensorboard', True):
        trainer.init_tensorboard(config.get('logging', {}).get('log_dir'))

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_cfg.get('num_epochs', 15),
        eval_interval=training_cfg.get('eval_interval', 1),
        save_interval=training_cfg.get('save_interval', 3),
        metric_for_best='separation_gap',
        higher_is_better=True,
    )

    logger.info(f"Negative sampler stats: {negative_sampler.stats}")
    trainer.close()
    logger.info("Phase 1b GMN training complete!")

    return history
```

### Update `main()` to Support Phase 1b/1c

```python
def main():
    parser = argparse.ArgumentParser(description='Train MBA Deobfuscator')
    parser.add_argument(
        '--phase', type=str, required=True,
        choices=['1', '1b', '1c', '2', '3'],  # Add 1b and 1c
        help='Training phase (1=contrastive, 1b=GMN frozen, 1c=GMN finetune, 2=supervised, 3=RL)'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Run appropriate phase
    if args.phase == '1':
        train_phase1(config, device)
    elif args.phase == '1b':
        train_phase1b(config, device)
    elif args.phase == '1c':
        train_phase1c(config, device)  # Similar to train_phase1b but uses Phase1cGMNTrainer
    elif args.phase == '2':
        train_phase2(config, device)
    elif args.phase == '3':
        train_phase3(config, device)
```

---

## 8. Error Handling and Edge Cases

### Z3 Timeout Handling

**Problem:** Z3 verification can timeout on complex expressions, blocking training.

**Solution:**
```python
class NegativeSampler:
    def _verify_with_retry(self, expr1: str, expr2: str, max_attempts: int = 3) -> Optional[bool]:
        """
        Verify with retry logic.

        Retry strategies:
            1. First attempt: full timeout (500ms)
            2. Second attempt: half timeout (250ms), simplified expressions
            3. Third attempt: skip Z3, use syntactic heuristic (AST diff > threshold)
        """
        for attempt in range(max_attempts):
            timeout = self.z3_timeout_ms // (2 ** attempt)
            result = self._verify_equivalence_with_timeout(expr1, expr2, timeout)

            if result is not None:
                return result

            if attempt < max_attempts - 1:
                logger.debug(f"Z3 timeout (attempt {attempt+1}/{max_attempts}), retrying...")

        # Fallback: syntactic heuristic
        logger.warning(f"Z3 timeout after {max_attempts} attempts, using syntactic heuristic")
        return self._syntactic_heuristic(expr1, expr2)
```

### NaN Gradient Handling

**Problem:** GMN cross-attention can produce NaN gradients when all attention weights collapse to zero (all-masked row).

**Solution:**
```python
# In cross_attention.py (already implemented)
def forward(self, h1, h2, mask2=None):
    # ...
    if mask2 is not None:
        attn_scores = attn_scores.masked_fill(~mask2, float('-inf'))

    # Handle all-masked rows: if all attention targets masked, use uniform attention
    all_masked = (mask2 is not None) and (~mask2).all(dim=-1, keepdim=True)
    if all_masked.any():
        # Replace -inf rows with zeros (uniform attention after softmax)
        attn_scores = torch.where(all_masked, torch.zeros_like(attn_scores), attn_scores)

    attn_weights = F.softmax(attn_scores, dim=-1)
    # ...
```

**Training-side check:**
```python
# In Phase1bGMNTrainer.train_step()
def train_step(self, batch):
    # ...
    loss.backward()

    # Check for NaN gradients
    has_nan = False
    for name, param in self.model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.error(f"NaN gradient in {name}")
            has_nan = True

    if has_nan:
        logger.error("NaN gradients detected, skipping update")
        self.optimizer.zero_grad()
        return {'total': float('nan'), 'bce': float('nan'), 'accuracy': 0.0}

    # Normal gradient update
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    self.optimizer.step()
    # ...
```

### Memory Management (Large Batches)

**Problem:** GMN cross-attention is memory-intensive (O(N1 * N2) attention matrix per pair).

**Solution:**
```python
# In Phase1bGMNTrainer config
training:
  batch_size: 32  # Smaller than Phase 1a (64)
  gradient_accumulation_steps: 2  # Effective batch size 64
  max_graph_size: 100  # Filter out graphs with >100 nodes
```

**Dataset filtering:**
```python
class GMNDataset:
    def _load_data(self, data_path):
        data = []
        for item in raw_data:
            graph = expr_to_graph(item['obfuscated'])
            if graph.num_nodes > self.max_graph_size:
                logger.debug(f"Skipping large graph with {graph.num_nodes} nodes")
                continue
            data.append(item)
        return data
```

### Dimension Validation for Checkpoint Loading (CRITICAL)

**Problem:** Corrupted checkpoint with wrong encoder_config passes validation but fails at runtime with cryptic dimension mismatch.

**Solution:**
```python
# In HGTWithGMN.__init__() after loading checkpoint
if hgt_checkpoint_path is not None:
    checkpoint = torch.load(hgt_checkpoint_path, map_location='cpu')
    encoder_config = checkpoint.get('encoder_config', {})
    encoder_hidden_dim = encoder_config.get('hidden_dim', hidden_dim)

    from src.models.encoder import HGTEncoder
    self.hgt_encoder = HGTEncoder(**encoder_config)
    self.hgt_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    logger.info(f"Loaded HGT encoder from {hgt_checkpoint_path}")

    # CRITICAL: Validate actual encoder output dimension
    # Protects against corrupted checkpoints where config doesn't match weights
    dummy_x = torch.zeros(10, 1, dtype=torch.long)  # 10 dummy node IDs
    dummy_edge_index = torch.randint(0, 10, (2, 20))
    dummy_batch = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        dummy_output = self.hgt_encoder(dummy_x, dummy_edge_index, dummy_batch, edge_type=None)
    actual_encoder_dim = dummy_output.shape[-1]

    if actual_encoder_dim != encoder_hidden_dim:
        raise ValueError(
            f"Checkpoint encoder_config mismatch: config says hidden_dim={encoder_hidden_dim}, "
            f"but encoder actually outputs {actual_encoder_dim}. "
            f"Checkpoint may be corrupted or config may be wrong."
        )
```

**Why This Matters:**
- Checkpoint could have `encoder_config={'hidden_dim': 256}` but weights trained with 768
- Without validation, encoder initializes with 256 → config validation passes (256 == 256)
- Forward pass fails with "size mismatch" when encoder outputs 768-dim to GMN expecting 256-dim
- Dummy forward pass catches this mismatch BEFORE training starts

### Encoder Freeze Verification

**Problem:** Accidentally unfrozen encoder in Phase 1b corrupts training.

**Solution:**
```python
class Phase1bGMNTrainer:
    def train_step(self, batch):
        # Verify encoder is frozen (every step)
        assert self.model.is_encoder_frozen, "Encoder must be frozen in Phase 1b"

        # Also check requires_grad flags
        for name, param in self.model.named_parameters():
            if 'encoder' in name and param.requires_grad:
                raise RuntimeError(f"Encoder parameter {name} has requires_grad=True")

        # Normal training step
        # ...
```

---

## 9. Testing Requirements

### Unit Tests

**Location:** `tests/test_phase1b_gmn_trainer.py`

```python
class TestPhase1bGMNTrainer:
    """Unit tests for Phase 1b GMN trainer."""

    def test_encoder_frozen(self):
        """Verify encoder remains frozen during training."""
        # Create mock model with frozen encoder
        # Run train_step()
        # Assert encoder parameters unchanged

    def test_gradient_flow_gmn_only(self):
        """Verify gradients only flow to GMN parameters."""
        # Run train_step()
        # Check encoder gradients are None
        # Check GMN gradients are not None

    def test_bce_loss_computation(self):
        """Test BCE loss with pos_weight balancing."""
        # Create batch with known labels
        # Compute loss
        # Verify loss matches manual computation

    def test_hard_negative_sampling(self):
        """Test that hard negatives improve training."""
        # Train with random negatives only
        # Train with hard negatives
        # Compare final separation_gap (should be larger with hard negatives)

    def test_checkpoint_save_load(self):
        """Test checkpoint save/load preserves encoder freeze state."""
        # Save checkpoint
        # Load checkpoint
        # Verify encoder still frozen

    def test_nan_gradient_handling(self):
        """Test graceful handling of NaN gradients."""
        # Inject NaN into model parameters
        # Run train_step()
        # Verify training continues (step skipped)
```

**Location:** `tests/test_negative_sampler.py`

```python
class TestNegativeSampler:
    """Unit tests for negative sampler."""

    def test_z3_verification_caching(self):
        """Test that Z3 results are cached correctly."""
        # Sample negative twice with same expressions
        # Verify second call hits cache (no Z3 invocation)

    def test_z3_timeout_handling(self):
        """Test graceful timeout handling."""
        # Use very short timeout (10ms)
        # Sample negative for complex expression
        # Verify fallback to syntactic heuristic

    def test_negative_strategies(self):
        """Test all negative sampling strategies."""
        # Test 'random', 'syntactic', 'depth', 'variable', 'mixed'
        # Verify each produces non-equivalent pairs

    def test_parallel_sampling(self):
        """Test multiprocessing batch sampling."""
        # Sample negatives for 100 pairs
        # Verify parallelization speedup (vs sequential)

    def test_variable_swapping_non_equivalence(self):
        """Test that variable swapping breaks equivalence."""
        # Sample negative with 'variable' strategy
        # Verify Z3 confirms non-equivalence
```

### Integration Tests

**Location:** `tests/test_gmn_training_integration.py`

```python
class TestGMNTrainingIntegration:
    """Integration tests for full GMN training pipeline."""

    def test_phase1a_to_phase1b_pipeline(self):
        """Test Phase 1a → Phase 1b transition."""
        # Train Phase 1a for 1 epoch (small dataset)
        # Save checkpoint
        # Load checkpoint in Phase 1b
        # Train Phase 1b for 1 epoch
        # Verify encoder unchanged, GMN trained

    def test_phase1b_to_phase1c_pipeline(self):
        """Test Phase 1b → Phase 1c transition."""
        # Train Phase 1b for 1 epoch
        # Save checkpoint
        # Load checkpoint in Phase 1c
        # Unfreeze encoder
        # Train Phase 1c for 1 epoch
        # Verify encoder fine-tuned

    def test_gmn_dataset_collation(self):
        """Test GMN dataset with batch collation."""
        # Create GMN dataset
        # Create dataloader with GMNBatchCollator
        # Iterate one batch
        # Verify shapes and batch consistency
```

---

## 10. Migration Path and Workflow

### Training Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1a: Contrastive Pretraining                               │
│ python scripts/train.py --phase 1 --config configs/phase1.yaml  │
│ Output: checkpoints/phase1/phase1_best.pt                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1b: GMN Frozen Training                                   │
│ python scripts/train.py --phase 1b \                            │
│   --config configs/phase1b_gmn.yaml                             │
│ Output: checkpoints/phase1b_gmn/phase1b_gmn_best.pt            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼ (optional)
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1c: GMN End-to-End Fine-Tuning                            │
│ python scripts/train.py --phase 1c \                            │
│   --config configs/phase1c_gmn_finetune.yaml                    │
│ Output: checkpoints/phase1c_gmn_finetune/phase1c_gmn_best.pt   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Supervised Training (existing)                         │
│ python scripts/train.py --phase 2 --config configs/phase2.yaml  │
│ Note: Can use Phase 1c checkpoint for encoder initialization    │
└─────────────────────────────────────────────────────────────────┘
```

### Backward Compatibility

**Existing workflows unaffected:**
- Phase 1 → Phase 2 → Phase 3 pipeline works without changes
- GMN training is optional (Phase 1b/1c)
- Encoder registry supports both GMN and non-GMN encoders

**Using GMN-trained encoder in Phase 2:**
```yaml
# configs/phase2.yaml
checkpoint:
  load_phase1: checkpoints/phase1c_gmn_finetune/phase1c_gmn_best.pt  # GMN checkpoint
```

Trainer automatically extracts encoder weights:
```python
# In train_phase2()
if 'gmn' in checkpoint['model_type']:
    # Extract encoder from HGTWithGMN wrapper
    encoder_state = {
        k.replace('hgt_encoder.', ''): v
        for k, v in checkpoint['model_state_dict'].items()
        if 'hgt_encoder' in k
    }
else:
    # Standard Phase 1a checkpoint
    encoder_state = checkpoint['model_state_dict']
```

---

## 11. Performance Considerations

### Training Speed

**Phase 1b vs Phase 1a:**
- Phase 1b: ~40% slower per batch (cross-attention overhead)
- Mitigations:
  - Smaller batch size (32 vs 64)
  - Gradient accumulation maintains effective batch size
  - Fewer epochs needed (15 vs 20)

**Z3 Verification Bottleneck:**
- Negative sampling: 500ms timeout per Z3 call
- Parallelization: 4 workers → ~125ms effective latency
- Caching: ~80% hit rate after warmup (5 epochs)
- Impact: ~10% training time overhead (dominated by GPU forward/backward)

### Memory Usage

**GMN Cross-Attention Memory:**
- Attention matrix: O(N1 * N2 * num_heads)
- Typical: N1=N2=50 nodes, 8 heads → ~20MB per batch
- Batch size 32: ~640MB attention matrices
- Total GPU memory: ~8GB (model + activations + attention)

**Memory Optimizations:**
- Filter large graphs (>100 nodes)
- Gradient checkpointing for cross-attention (optional)
- Mixed precision training (FP16) reduces memory by 50%

### Hyperparameter Tuning

**Critical hyperparameters (priority order):**

1. **Learning rate:** `3e-5` (Phase 1b), `3e-6` (Phase 1c encoder)
   - Too high: attention weights collapse to uniform
   - Too low: slow convergence (>20 epochs)

2. **Hard negative ratio:** `0.5` (Phase 1b), `0.7` (Phase 1c)
   - Too low: model learns trivial negatives, poor generalization
   - Too high: training instability (hard negatives dominate)

3. **Gradient clipping:** `1.0`
   - GMN attention gradients can spike to 100+ without clipping

4. **Positive weight:** `1.0` (balanced dataset)
   - Adjust if precision/recall imbalance: `pos_weight = neg_count / pos_count`

---

## 12. Monitoring and Logging

### Key Metrics to Track

**Phase 1b:**
- `separation_gap`: Primary metric (pos_score_mean - neg_score_mean)
  - Target: >0.3 (well-separated)
  - Plateau: <0.1 (underfitting or trivial negatives)

- `accuracy`: Binary classification accuracy
  - Target: >85% on validation set

- `bce_loss`: Should decrease steadily
  - Sudden spikes: NaN gradients or batch anomalies

- `pos_score_mean` / `neg_score_mean`: Absolute score values
  - Healthy: pos_score ~0.7-0.9, neg_score ~0.2-0.4
  - Collapsed: both near 0.5 (random guessing)

**Phase 1c:**
- `encoder_drift`: L2 distance from Phase 1b checkpoint
  - Target: <0.5 (controlled fine-tuning)
  - Warning: >1.0 (catastrophic forgetting)

- `encoder_grad_norm` / `gmn_grad_norm`: Gradient magnitudes
  - Healthy ratio: encoder_grad ~0.1x gmn_grad
  - Imbalanced: adjust encoder_learning_rate

### TensorBoard Logging

```python
# In Phase1bGMNTrainer.train_step()
if self.global_step % self.log_interval == 0:
    self.writer.add_scalar('Loss/total', loss_dict['total'], self.global_step)
    self.writer.add_scalar('Loss/bce', loss_dict['bce'], self.global_step)
    self.writer.add_scalar('Metrics/accuracy', loss_dict['accuracy'], self.global_step)
    self.writer.add_scalar('Metrics/pos_score', loss_dict['pos_score'], self.global_step)
    self.writer.add_scalar('Metrics/neg_score', loss_dict['neg_score'], self.global_step)
    self.writer.add_scalar('Metrics/separation_gap',
                          loss_dict['pos_score'] - loss_dict['neg_score'],
                          self.global_step)

    # Negative sampler stats
    stats = self.negative_sampler.stats
    self.writer.add_scalar('Sampler/cache_hit_rate', stats['cache_hit_rate'], self.global_step)
    self.writer.add_scalar('Sampler/z3_timeouts', stats['z3_timeouts'], self.global_step)
```

---

## 13. Summary Checklist

### Implementation Order

1. **Week 1: Core GMN Losses**
   - [ ] `gmn_bce_loss()` in `src/training/losses.py`
   - [ ] `gmn_triplet_loss()` in `src/training/losses.py`
   - [ ] `gmn_combined_loss()` in `src/training/losses.py`
   - [ ] Unit tests in `tests/test_losses.py`

2. **Week 2: Negative Sampler**
   - [ ] `NegativeSampler` class in `src/training/negative_sampler.py`
   - [ ] Z3 verification with timeout and caching
   - [ ] Parallel batch sampling
   - [ ] Unit tests in `tests/test_negative_sampler.py`

3. **Week 3: GMN Dataset and Collation**
   - [ ] `GMNDataset` in `src/data/dataset.py`
   - [ ] `collate_gmn_pairs()` in `src/data/collate.py`
   - [ ] Integration with `GMNBatchCollator`
   - [ ] Unit tests in `tests/test_dataset.py`

4. **Week 4: Phase1bGMNTrainer**
   - [ ] `Phase1bGMNTrainer` class in `src/training/phase1b_gmn_trainer.py`
   - [ ] `train_step()`, `evaluate()`, checkpoint methods
   - [ ] Encoder freeze verification
   - [ ] Unit tests in `tests/test_phase1b_gmn_trainer.py`

5. **Week 5: Phase1cGMNTrainer**
   - [ ] `Phase1cGMNTrainer` class in `src/training/phase1c_gmn_trainer.py`
   - [ ] Dual learning rate optimizer
   - [ ] Encoder drift monitoring
   - [ ] Unit tests in `tests/test_phase1c_gmn_trainer.py`

6. **Week 6: Training Script Integration**
   - [ ] `train_phase1b()` in `scripts/train.py`
   - [ ] `train_phase1c()` in `scripts/train.py`
   - [ ] Config files: `configs/phase1b_gmn.yaml`, `configs/phase1c_gmn_finetune.yaml`
   - [ ] Update `main()` to support phases 1b and 1c
   - [ ] Integration tests in `tests/test_gmn_training_integration.py`

7. **Week 7: Testing and Documentation**
   - [ ] End-to-end integration test (Phase 1a → 1b → 1c)
   - [ ] Hyperparameter tuning experiments
   - [ ] Update `CLAUDE.md` with GMN training instructions
   - [ ] Update `README.md` architecture section

---

## Appendix A: File Structure Summary

```
mba-deobfuscator/
├── src/
│   ├── training/
│   │   ├── losses.py                    # [EXTEND] Add GMN losses
│   │   ├── phase1b_gmn_trainer.py       # [NEW] Phase 1b trainer
│   │   ├── phase1c_gmn_trainer.py       # [NEW] Phase 1c trainer
│   │   └── negative_sampler.py          # [NEW] Hard negative sampler
│   ├── data/
│   │   ├── dataset.py                   # [EXTEND] Add GMNDataset
│   │   └── collate.py                   # [EXTEND] Add collate_gmn_pairs
│   └── models/
│       └── gmn/                          # [EXISTING] GMN core modules
│           ├── graph_matching.py
│           ├── gmn_encoder_wrapper.py
│           ├── cross_attention.py
│           └── batch_collator.py
├── scripts/
│   └── train.py                         # [EXTEND] Add train_phase1b/1c
├── configs/
│   ├── phase1b_gmn.yaml                 # [NEW] Phase 1b config
│   └── phase1c_gmn_finetune.yaml        # [NEW] Phase 1c config
├── tests/
│   ├── test_phase1b_gmn_trainer.py      # [NEW] Phase 1b tests
│   ├── test_phase1c_gmn_trainer.py      # [NEW] Phase 1c tests
│   ├── test_negative_sampler.py         # [NEW] Negative sampler tests
│   └── test_gmn_training_integration.py # [NEW] Integration tests
└── docs/
    └── GMN_PHASE4_TRAINING_PLAN.md      # [THIS DOCUMENT]
```

---

## Appendix B: Example Training Session

```bash
# Step 1: Phase 1a (existing contrastive pretraining)
python scripts/train.py --phase 1 --config configs/phase1.yaml

# Output:
# Epoch 20/20: infonce=0.15, sim_gap=0.68
# Saved best checkpoint: checkpoints/phase1/phase1_best.pt

# Step 2: Phase 1b (GMN with frozen encoder)
python scripts/train.py --phase 1b --config configs/phase1b_gmn.yaml

# Output:
# Epoch 1/15: bce=0.45, accuracy=0.78, separation_gap=0.21
# Epoch 5/15: bce=0.28, accuracy=0.87, separation_gap=0.38
# Epoch 15/15: bce=0.18, accuracy=0.92, separation_gap=0.52
# Saved best checkpoint: checkpoints/phase1b_gmn/phase1b_gmn_best.pt
# Negative sampler stats: cache_hit_rate=0.82, z3_timeouts=34

# Step 3: Phase 1c (optional end-to-end fine-tuning)
python scripts/train.py --phase 1c --config configs/phase1c_gmn_finetune.yaml

# Output:
# Epoch 1/5: bce=0.15, accuracy=0.93, separation_gap=0.57, encoder_drift=0.12
# Epoch 5/5: bce=0.12, accuracy=0.94, separation_gap=0.61, encoder_drift=0.28
# Saved best checkpoint: checkpoints/phase1c_gmn_finetune/phase1c_gmn_best.pt

# Step 4: Proceed to Phase 2 (existing supervised training)
python scripts/train.py --phase 2 --config configs/phase2.yaml
# (loads encoder from Phase 1c checkpoint)
```

---

## Appendix C: Debugging Common Issues

### Issue: Separation Gap Plateaus at Low Value (<0.15)

**Symptom:** GMN fails to learn discriminative matching.

**Diagnosis:**
```bash
# Check negative sampler statistics
grep "Negative sampler" logs/phase1b_gmn/train.log
# High z3_timeout rate (>50%) indicates negatives not verified
```

**Solutions:**
1. Increase Z3 timeout: `z3_timeout_ms: 1000`
2. Increase hard negative ratio: `hard_negative_ratio: 0.7`
3. Check dataset quality (are positives truly equivalent?)

### Issue: NaN Loss After N Steps

**Symptom:** Training crashes with NaN loss.

**Diagnosis:**
```python
# Add debug logging in train_step()
logger.debug(f"match_scores: {match_scores.detach()}")
logger.debug(f"logits: {logits.detach()}")
```

**Solutions:**
1. Reduce learning rate: `learning_rate: 1e-5`
2. Increase gradient clipping: `max_grad_norm: 0.5`
3. Check for all-masked attention rows (should be handled in cross_attention.py)

### Issue: Encoder Drift Too High in Phase 1c

**Symptom:** `encoder_drift > 1.0`, validation accuracy drops.

**Diagnosis:**
```bash
# Compare encoder weights before/after
python scripts/analyze_checkpoint.py \
  --checkpoint1 checkpoints/phase1b_gmn/phase1b_gmn_best.pt \
  --checkpoint2 checkpoints/phase1c_gmn_finetune/phase1c_gmn_best.pt
```

**Solutions:**
1. Lower encoder learning rate: `encoder_learning_rate: 1e-6`
2. Reduce training epochs: `num_epochs: 3`
3. Increase encoder weight decay: `encoder_weight_decay: 0.01`

---

**Document Version:** 1.0
**Last Updated:** 2026-01-17
**Author:** Technical Writer (TW) for Claude Code MBA Deobfuscator
