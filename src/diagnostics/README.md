# Diagnostic Tools for MBA Deobfuscator

Comprehensive diagnostic tools for detecting and visualizing GNN/Transformer pathologies during training.

## Features

✅ **Layer Activation Monitoring** - Gradient vanishing/explosion detection
✅ **Representation Collapse Detection** - Over-smoothing in deep GNNs
✅ **Attention Pattern Visualization** - Head specialization analysis
✅ **Node Embedding Diversity Metrics** - Over-squashing detection

**Key improvements**:
- Memory budgeting for O(N²) operations (prevents OOM on depth 10-14 expressions)
- Deterministic sampling for reproducible ablation studies
- Robust hook cleanup with atexit handlers
- TensorBoard integration with graceful degradation

---

## Quick Start

### Enable Diagnostics in Training

```python
from src.training.phase2_trainer import Phase2Trainer

trainer = Phase2Trainer(model, tokenizer, fingerprint, ...)

# Enable diagnostics (optional, disabled by default)
trainer.enable_activation_monitor(track_gradients=True)
trainer.enable_collapse_detector(sample_size=500, max_nodes_for_full_distance=5000)
trainer.enable_diversity_metrics(sample_size=500, max_nodes_for_full_distance=5000)

# Initialize TensorBoard
trainer.init_tensorboard('logs/my_experiment')

# Training loop
for epoch in range(num_epochs):
    trainer.train_epoch(dataloader)

    # Log diagnostics every 100 steps
    trainer.log_diagnostics(log_interval=100)

# CRITICAL: Clean up hooks at end
trainer.cleanup_diagnostics()
```

### Standalone Diagnostic Collection

```python
from src.diagnostics import ActivationMonitor, CollapseDetector

# Create monitors
activation_monitor = ActivationMonitor(track_gradients=True)
collapse_detector = CollapseDetector(sample_size=500)

# Register hooks
activation_monitor.register_hooks(model)
collapse_detector.register_hooks(model)

# Forward pass
output = model(batch)
loss.backward()

# Get statistics
act_stats = activation_monitor.get_statistics()
collapse_stats = collapse_detector.get_statistics()

# Detect pathologies
pathologies = activation_monitor.detect_pathologies()
if pathologies['vanishing_gradients']:
    print(f"Vanishing gradients detected in {len(pathologies['vanishing_gradients'])} layers")

# Print summary
print(activation_monitor.summary())
print(collapse_detector.summary())

# CRITICAL: Clean up
activation_monitor.remove_hooks()
collapse_detector.remove_hooks()
```

### Visualization

```bash
# Generate plots from TensorBoard logs
python scripts/visualize_diagnostics.py --tensorboard logs/experiment --output plots/

# Generate trend plots
python scripts/visualize_diagnostics.py --tensorboard logs/experiment --trends \
    --category collapse --metric diversity

# From saved JSON data
python scripts/visualize_diagnostics.py --json diagnostics_data.json --output plots/
```

---

## Module Overview

### ActivationMonitor

**Purpose**: Track layer-wise activation and gradient norms.

**Detects**:
- Vanishing gradients (norm < 1e-4)
- Exploding gradients (norm > 100)
- Dead neurons (std < 1e-6)

**API**:
```python
monitor = ActivationMonitor(track_gradients=True, dead_neuron_threshold=1e-6)
monitor.register_hooks(model)
stats = monitor.get_statistics()  # {layer_name: {activation_norm, gradient_norm, dead_neurons_pct}}
pathologies = monitor.detect_pathologies()
```

### CollapseDetector

**Purpose**: Detect representation collapse via inter-layer similarity.

**Metrics**:
- Inter-layer cosine similarity (>0.95 = collapse)
- Pairwise embedding diversity
- Effective rank of representation matrix

**API**:
```python
detector = CollapseDetector(sample_size=500, max_nodes_for_full_distance=5000)
detector.register_hooks(model)
stats = detector.get_statistics(global_step)  # {layer_name: {diversity, effective_rank, mean_similarity}}
collapse = detector.detect_collapse(similarity_threshold=0.95)
```

**Memory budgeting**: Automatically skips full O(N²) computation if total nodes exceed `max_nodes_for_full_distance`.

### AttentionAnalyzer

**Purpose**: Analyze attention patterns in Transformer decoder and HGT encoder.

**Metrics**:
- Attention entropy (uniform vs focused)
- Mean attention distance (local vs global)
- Head specialization variance

**API**:
```python
analyzer = AttentionAnalyzer(max_seq_length_to_store=128)
analyzer.register_hooks(model, module_types=['MultiheadAttention', 'HGTConv'])
stats = analyzer.get_statistics()  # {layer_name: {entropy, mean_distance, head_specialization}}
```

**Note**: Requires `MultiheadAttention` to be called with `need_weights=True`.

### DiversityMetrics

**Purpose**: Measure node embedding diversity to detect over-squashing.

**Metrics**:
- Mean pairwise distance
- Variance of pairwise distances
- Per-dimension variance

**API**:
```python
metrics = DiversityMetrics(sample_size=500, max_nodes_for_full_distance=5000)
metrics.register_hooks(model)
stats = metrics.get_statistics(global_step)  # {layer_name: {mean_distance, variance_distance, dimension_variance}}
squashing = metrics.detect_over_squashing(distance_threshold=0.5)
```

---

## Configuration

See `configs/diagnostics.yaml` for full configuration options.

### Memory Budgeting (CRITICAL)

For production training with ScaledMBADataset at depth 10-14:

```yaml
collapse_detector:
  sample_size: 500  # MANDATORY - sample nodes for O(N²) computation
  max_nodes_for_full_distance: 5000  # Skip full computation if exceeded
  chunk_size: 500  # Chunk size for memory-efficient estimation

diversity_metrics:
  sample_size: 500
  max_nodes_for_full_distance: 5000
  chunk_size: 500
```

**Why**: At depth 12-14, batch graphs can contain 2000+ nodes. Full `torch.cdist()` would allocate 2000×2000×768 = **23GB** per forward pass. Memory budgeting prevents OOM crashes.

---

## Performance Overhead

| Diagnostic | Overhead (depth ≤10) | Overhead (depth 10-14) | Memory (extra) |
|------------|---------------------|----------------------|----------------|
| ActivationMonitor | ~2-3% | ~2-3% | ~50MB |
| CollapseDetector | ~3-5% | ~5-8% | ~100MB |
| AttentionAnalyzer | ~2-4% | ~2-4% | ~50MB |
| DiversityMetrics | ~3-5% | ~5-8% | ~100MB |
| **All combined** | ~10-15% | ~15-20% | ~300MB |

Measured on Phase 2 training with batch_size=32, ScaledMBADataset.

---

## Best Practices

### 1. Memory Management

```python
# Use memory budgeting for production training
trainer.enable_collapse_detector(
    sample_size=500,  # Required
    max_nodes_for_full_distance=5000  # Prevents OOM
)
```

### 2. Hook Cleanup

```python
# Always clean up in finally block
try:
    trainer.fit(train_loader, val_loader)
finally:
    trainer.cleanup_diagnostics()
```

Or use context manager pattern:

```python
from contextlib import contextmanager

@contextmanager
def diagnostic_context(trainer):
    trainer.enable_activation_monitor()
    try:
        yield trainer
    finally:
        trainer.cleanup_diagnostics()

with diagnostic_context(trainer) as t:
    t.fit(train_loader, val_loader)
```

### 3. Logging Frequency

```python
# Adjust log_interval based on overhead tolerance
trainer.log_diagnostics(log_interval=100)  # Default: every 100 steps

# For expensive diagnostics (collapse, diversity), increase interval
trainer.log_diagnostics(log_interval=500)  # Every 500 steps
```

### 4. Deterministic Sampling

All sampling is deterministic within a training step for reproducibility. No manual seed management required.

---

## Troubleshooting

### "CUDA out of memory" during collapse/diversity detection

**Cause**: Batch has >5000 total nodes, exceeding memory budget.

**Fix**: Increase `max_nodes_for_full_distance` or reduce `sample_size`:

```python
trainer.enable_collapse_detector(
    sample_size=300,  # Reduced from 500
    max_nodes_for_full_distance=3000  # More aggressive budget
)
```

### "Attention weights not normalized" warning

**Cause**: `AttentionAnalyzer` captured pre-softmax logits instead of normalized weights.

**Fix**: Ensure `MultiheadAttention` is called with `need_weights=True` and hook captures post-softmax output.

### Hooks persist after training crashes

**Cause**: Abnormal termination prevented atexit handlers from running.

**Fix**: Call `trainer.cleanup_diagnostics()` in finally block (see Best Practices above).

---

## Implementation Details

### CRITICAL FIXES Applied

1. **Atexit cleanup instead of __del__**: Reliable hook cleanup during abnormal termination
2. **Memory budgeting**: Prevents O(N²) OOM crashes on depth 10-14 expressions
3. **Deterministic sampling**: Reproducible ablation studies via feature-based sorting
4. **TensorBoard null checks**: Graceful degradation when TensorBoard not initialized
5. **Weight normalization validation**: Prevents garbage entropy values from unnormalized attention
6. **Refactored log_diagnostics**: Reduced from 71 lines (4 nesting) to 30 lines (2 nesting)

### Testing

```bash
# Run diagnostic tests
pytest tests/test_diagnostics.py -v

# Performance overhead benchmark
python scripts/benchmark_diagnostics.py --depth 12 --batch-size 32
```

---

## Contributing

When adding new diagnostic tools:

1. Use atexit handlers for cleanup (not `__del__`)
2. Validate memory budgets for O(N²) operations
3. Support deterministic sampling (no random operations)
4. Handle None TensorBoard writer gracefully
5. Add unit tests with mock models

---

## References

- Over-smoothing: [Li et al. 2018 - Deeper Insights into GNNs](https://arxiv.org/abs/1810.00826)
- Over-squashing: [Alon & Yahav 2021 - On the Bottleneck of GNNs](https://arxiv.org/abs/2006.05205)
- GCNII: [Chen et al. 2020 - Simple and Deep GCNs](https://arxiv.org/abs/2007.02133)
