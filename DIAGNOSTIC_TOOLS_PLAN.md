# MBA Deobfuscator Diagnostic Tools Implementation Plan

> **Status**: Ready for implementation
> **Priority**: P2 (Developer tooling for model debugging)
> **Estimated Scope**: 800-1000 LOC across 5 new files + 300 LOC integration
> **Target**: Hook-based diagnostics with minimal overhead (<5% training slowdown)

---

## Executive Summary

Add comprehensive diagnostic tools to detect and visualize common GNN/Transformer pathologies:

1. **Layer activation monitoring** - Gradient vanishing/explosion detection
2. **Representation collapse detection** - Over-smoothing in deep GNNs
3. **Attention pattern visualization** - Head specialization analysis
4. **Node embedding diversity metrics** - Over-squashing detection

**Design principles**:
- Optional hooks (disabled by default, zero overhead when off)
- Integration with existing BaseTrainer/AblationTrainer
- TensorBoard logging + standalone visualization utilities
- Minimal invasiveness (no core model changes)

---

## Architecture Overview

```
src/diagnostics/
├── __init__.py                    # Public API exports
├── activation_monitor.py          # Layer-wise activation/gradient norms (180 LOC)
├── collapse_detector.py           # Representation collapse via cosine similarity (150 LOC)
├── attention_analyzer.py          # Attention pattern extraction/visualization (220 LOC)
├── diversity_metrics.py           # Embedding diversity/over-squashing metrics (160 LOC)
└── visualizers.py                 # Matplotlib plotting utilities (250 LOC)

scripts/
└── visualize_diagnostics.py      # Standalone visualization script (200 LOC)

Integration points:
- src/training/base_trainer.py     # Add hook management methods (+100 LOC)
- src/training/ablation_trainer.py # Add diagnostic collection (+50 LOC)
- src/models/encoder.py            # Register hook points (minimal, +30 LOC)
- src/models/decoder.py            # Register hook points (minimal, +20 LOC)
```

---

## Module 1: Activation Monitor

**File**: `src/diagnostics/activation_monitor.py`

**Purpose**: Track embedding norms and gradient magnitudes per layer to detect vanishing/exploding gradients.

### Implementation

```python
"""
Layer activation and gradient monitoring for GNN/Transformer models.

Tracks per-layer statistics to detect gradient pathologies:
- Vanishing gradients: norm < 1e-4
- Exploding gradients: norm > 100
- Dead neurons: activation std < 1e-6
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class ActivationMonitor:
    """
    Monitor layer activations and gradients during training/inference.

    Tracks:
    - Forward activation norms (L2 norm of layer outputs)
    - Backward gradient norms (L2 norm of gradients)
    - Dead neuron detection (channels with near-zero variance)

    Usage:
        monitor = ActivationMonitor()
        monitor.register_hooks(model)

        # Training loop
        output = model(batch)
        loss.backward()

        stats = monitor.get_statistics()
        monitor.log_to_tensorboard(writer, global_step)
        monitor.reset()
    """

    def __init__(self, track_gradients: bool = True):
        """
        Initialize activation monitor.

        Args:
            track_gradients: Whether to track gradient norms (requires backward pass)
        """
        self.track_gradients = track_gradients
        self.activation_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.activation_stds: Dict[str, List[float]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: nn.Module, layer_filter: Optional[callable] = None) -> None:
        """
        Register forward/backward hooks on model layers.

        Args:
            model: PyTorch model to monitor
            layer_filter: Optional filter function(name, module) -> bool
                         Default: Monitor GATConv, HGTConv, MultiheadAttention, Linear
        """
        if layer_filter is None:
            # Default: monitor key layer types
            def default_filter(name: str, module: nn.Module) -> bool:
                monitor_types = (
                    nn.Linear,
                    nn.MultiheadAttention,
                    nn.LayerNorm,
                )
                # Check for PyG layers by name (avoid import)
                if any(x in type(module).__name__ for x in ['GATConv', 'HGTConv', 'RGCNConv', 'GatedGraphConv']):
                    return True
                return isinstance(module, monitor_types)
            layer_filter = default_filter

        for name, module in model.named_modules():
            if layer_filter(name, module):
                # Forward hook: capture activations
                hook = module.register_forward_hook(
                    self._make_forward_hook(name)
                )
                self.hooks.append(hook)

                # Backward hook: capture gradients
                if self.track_gradients:
                    hook = module.register_full_backward_hook(
                        self._make_backward_hook(name)
                    )
                    self.hooks.append(hook)

    def _make_forward_hook(self, layer_name: str):
        """Create forward hook closure for a specific layer."""
        def hook(module, input, output):
            # Handle tuple outputs (e.g., attention returns (output, weights))
            if isinstance(output, tuple):
                output = output[0]

            # Compute L2 norm
            if isinstance(output, torch.Tensor):
                norm = output.detach().norm(p=2).item()
                std = output.detach().std().item()

                self.activation_norms[layer_name].append(norm)
                self.activation_stds[layer_name].append(std)

        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create backward hook closure for a specific layer."""
        def hook(module, grad_input, grad_output):
            # grad_output is tuple of gradients w.r.t. outputs
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
                norm = grad.norm(p=2).item()
                self.gradient_norms[layer_name].append(norm)

        return hook

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across tracked batches.

        Returns:
            Dictionary mapping layer_name -> {
                'activation_norm_mean': float,
                'activation_norm_std': float,
                'gradient_norm_mean': float (if tracked),
                'gradient_norm_std': float (if tracked),
                'dead_neurons': bool (std < 1e-6),
                'vanishing_gradient': bool (grad_norm < 1e-4),
                'exploding_gradient': bool (grad_norm > 100),
            }
        """
        stats = {}

        for layer_name in self.activation_norms.keys():
            layer_stats = {}

            # Activation statistics
            act_norms = self.activation_norms[layer_name]
            if act_norms:
                layer_stats['activation_norm_mean'] = float(np.mean(act_norms))
                layer_stats['activation_norm_std'] = float(np.std(act_norms))

            # Dead neuron detection
            act_stds = self.activation_stds[layer_name]
            if act_stds:
                layer_stats['activation_std_mean'] = float(np.mean(act_stds))
                layer_stats['dead_neurons'] = np.mean(act_stds) < 1e-6

            # Gradient statistics
            if self.track_gradients and layer_name in self.gradient_norms:
                grad_norms = self.gradient_norms[layer_name]
                if grad_norms:
                    grad_mean = float(np.mean(grad_norms))
                    layer_stats['gradient_norm_mean'] = grad_mean
                    layer_stats['gradient_norm_std'] = float(np.std(grad_norms))
                    layer_stats['vanishing_gradient'] = grad_mean < 1e-4
                    layer_stats['exploding_gradient'] = grad_mean > 100

            stats[layer_name] = layer_stats

        return stats

    def log_to_tensorboard(self, writer, global_step: int, prefix: str = 'diagnostics/activation') -> None:
        """Log statistics to TensorBoard."""
        stats = self.get_statistics()

        for layer_name, layer_stats in stats.items():
            # Sanitize layer name for TensorBoard
            clean_name = layer_name.replace('.', '/')

            for metric_name, value in layer_stats.items():
                if isinstance(value, bool):
                    value = float(value)

                writer.add_scalar(
                    f'{prefix}/{clean_name}/{metric_name}',
                    value,
                    global_step
                )

    def detect_pathologies(self) -> Dict[str, List[str]]:
        """
        Detect common training pathologies.

        Returns:
            Dictionary mapping pathology_type -> [affected_layer_names]
        """
        stats = self.get_statistics()
        pathologies = {
            'vanishing_gradients': [],
            'exploding_gradients': [],
            'dead_neurons': [],
        }

        for layer_name, layer_stats in stats.items():
            if layer_stats.get('vanishing_gradient', False):
                pathologies['vanishing_gradients'].append(layer_name)
            if layer_stats.get('exploding_gradient', False):
                pathologies['exploding_gradients'].append(layer_name)
            if layer_stats.get('dead_neurons', False):
                pathologies['dead_neurons'].append(layer_name)

        return pathologies

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self.activation_norms.clear()
        self.gradient_norms.clear()
        self.activation_stds.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
```

**Integration with BaseTrainer**:

```python
# In src/training/base_trainer.py, add:

def enable_activation_monitoring(self, track_gradients: bool = True) -> None:
    """Enable activation monitoring for gradient pathology detection."""
    from src.diagnostics.activation_monitor import ActivationMonitor

    if not hasattr(self, '_activation_monitor'):
        self._activation_monitor = ActivationMonitor(track_gradients=track_gradients)
        self._activation_monitor.register_hooks(self.model)
        logger.info("Activation monitoring enabled")

def log_activation_diagnostics(self) -> None:
    """Log activation statistics to TensorBoard (call after backward)."""
    if hasattr(self, '_activation_monitor') and self.tensorboard_writer is not None:
        self._activation_monitor.log_to_tensorboard(
            self.tensorboard_writer,
            self.global_step,
            prefix='diagnostics/activation'
        )

        # Log pathologies to console
        pathologies = self._activation_monitor.detect_pathologies()
        if any(pathologies.values()):
            logger.warning(f"Detected pathologies at step {self.global_step}: {pathologies}")

        self._activation_monitor.reset()
```

---

## Module 2: Collapse Detector

**File**: `src/diagnostics/collapse_detector.py`

**Purpose**: Detect representation collapse via pairwise cosine similarity between layer outputs.

### Implementation

```python
"""
Representation collapse detection for GNN encoders.

Over-smoothing in GNNs causes node embeddings to converge to similar values,
losing discriminative power. This module detects collapse by measuring:

1. Inter-layer similarity: cosine(layer_i, layer_{i+1}) → should decrease
2. Intra-layer diversity: pairwise distances within layer → should stay high
3. Rank collapse: effective rank of embedding matrix → should stay high
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


class CollapseDetector:
    """
    Detect representation collapse in GNN encoders.

    Tracks:
    - Pairwise cosine similarity between consecutive layers
    - Embedding diversity (mean pairwise L2 distance)
    - Effective rank of embedding matrix

    Collapse indicators:
    - Cosine similarity > 0.95 between consecutive layers
    - Embedding diversity < 0.1
    - Effective rank < 10% of embedding dimension
    """

    def __init__(self, sample_size: int = 1000):
        """
        Initialize collapse detector.

        Args:
            sample_size: Number of nodes to sample for diversity computation
                        (full computation O(N^2) is expensive for large graphs)
        """
        self.sample_size = sample_size
        self.layer_embeddings: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Statistics
        self.interlayer_similarities: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.diversity_scores: Dict[str, List[float]] = defaultdict(list)
        self.effective_ranks: Dict[str, List[float]] = defaultdict(list)

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Register hooks on GNN encoder layers.

        Args:
            model: Model to monitor
            layer_names: Specific layer names to monitor (default: auto-detect GNN layers)
        """
        if layer_names is None:
            # Auto-detect GNN layers
            layer_names = []
            for name, module in model.named_modules():
                if any(x in type(module).__name__ for x in ['GATConv', 'HGTConv', 'RGCNConv', 'GatedGraphConv', 'GCN']):
                    layer_names.append(name)

        self.layer_names = sorted(layer_names)  # Ensure consistent ordering

        for name in self.layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Create hook to capture layer output."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]

            # Store detached copy
            self.layer_embeddings[layer_name] = output.detach().clone()

        return hook

    def compute_statistics(self) -> None:
        """
        Compute collapse statistics from captured embeddings.

        Call this after forward pass completes.
        """
        if len(self.layer_embeddings) < 2:
            return

        # Compute inter-layer similarities
        for i in range(len(self.layer_names) - 1):
            layer1 = self.layer_names[i]
            layer2 = self.layer_names[i + 1]

            if layer1 in self.layer_embeddings and layer2 in self.layer_embeddings:
                emb1 = self.layer_embeddings[layer1]
                emb2 = self.layer_embeddings[layer2]

                # Cosine similarity between layer means
                sim = self._cosine_similarity(emb1.mean(dim=0), emb2.mean(dim=0))
                self.interlayer_similarities[(layer1, layer2)].append(sim)

        # Compute diversity and rank for each layer
        for layer_name, embeddings in self.layer_embeddings.items():
            diversity = self._compute_diversity(embeddings)
            self.diversity_scores[layer_name].append(diversity)

            rank = self._compute_effective_rank(embeddings)
            self.effective_ranks[layer_name].append(rank)

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Cosine similarity between two vectors."""
        return float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())

    def _compute_diversity(self, embeddings: torch.Tensor) -> float:
        """
        Mean pairwise L2 distance (sampled for efficiency).

        Args:
            embeddings: [num_nodes, hidden_dim]

        Returns:
            Mean pairwise distance
        """
        num_nodes = embeddings.size(0)
        if num_nodes <= 1:
            return 0.0

        # Sample nodes if too many
        if num_nodes > self.sample_size:
            indices = torch.randperm(num_nodes)[:self.sample_size]
            embeddings = embeddings[indices]
            num_nodes = self.sample_size

        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)

        # Upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        mean_dist = dists[mask].mean().item()

        return float(mean_dist)

    def _compute_effective_rank(self, embeddings: torch.Tensor) -> float:
        """
        Effective rank via singular value entropy.

        Effective rank = exp(entropy(normalized_singular_values))

        High rank → diverse representations
        Low rank → collapsed representations
        """
        # SVD on embedding matrix
        try:
            _, S, _ = torch.svd(embeddings)

            # Normalize singular values
            S_normalized = S / S.sum()

            # Entropy
            entropy = -(S_normalized * torch.log(S_normalized + 1e-12)).sum()

            # Effective rank
            return float(torch.exp(entropy).item())
        except:
            return 0.0

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics.

        Returns:
            {
                'interlayer_similarity': {(layer1, layer2): mean_similarity},
                'diversity': {layer_name: mean_diversity},
                'effective_rank': {layer_name: mean_rank},
            }
        """
        stats = {}

        # Inter-layer similarities
        stats['interlayer_similarity'] = {
            f"{l1}_to_{l2}": float(np.mean(sims))
            for (l1, l2), sims in self.interlayer_similarities.items()
            if sims
        }

        # Diversity scores
        stats['diversity'] = {
            layer: float(np.mean(scores))
            for layer, scores in self.diversity_scores.items()
            if scores
        }

        # Effective ranks
        stats['effective_rank'] = {
            layer: float(np.mean(ranks))
            for layer, ranks in self.effective_ranks.items()
            if ranks
        }

        return stats

    def detect_collapse(self, similarity_threshold: float = 0.95,
                       diversity_threshold: float = 0.1,
                       rank_threshold_ratio: float = 0.1) -> Dict[str, bool]:
        """
        Detect collapse conditions.

        Args:
            similarity_threshold: Cosine similarity threshold for collapse
            diversity_threshold: Minimum diversity threshold
            rank_threshold_ratio: Minimum effective rank as fraction of embedding dim

        Returns:
            {
                'high_interlayer_similarity': bool,
                'low_diversity': bool,
                'rank_collapse': bool,
            }
        """
        stats = self.get_statistics()

        # Check inter-layer similarity
        high_sim = any(
            sim > similarity_threshold
            for sim in stats.get('interlayer_similarity', {}).values()
        )

        # Check diversity
        low_div = any(
            div < diversity_threshold
            for div in stats.get('diversity', {}).values()
        )

        # Check rank (assume hidden_dim from first layer)
        rank_collapsed = False
        if self.layer_embeddings:
            first_layer = list(self.layer_embeddings.values())[0]
            hidden_dim = first_layer.size(-1)
            rank_threshold = hidden_dim * rank_threshold_ratio

            rank_collapsed = any(
                rank < rank_threshold
                for rank in stats.get('effective_rank', {}).values()
            )

        return {
            'high_interlayer_similarity': high_sim,
            'low_diversity': low_div,
            'rank_collapse': rank_collapsed,
        }

    def log_to_tensorboard(self, writer, global_step: int, prefix: str = 'diagnostics/collapse') -> None:
        """Log collapse metrics to TensorBoard."""
        stats = self.get_statistics()

        for metric_type, metric_dict in stats.items():
            for name, value in metric_dict.items():
                writer.add_scalar(
                    f'{prefix}/{metric_type}/{name}',
                    value,
                    global_step
                )

        # Log collapse flags
        collapse_flags = self.detect_collapse()
        for flag_name, flag_value in collapse_flags.items():
            writer.add_scalar(
                f'{prefix}/flags/{flag_name}',
                float(flag_value),
                global_step
            )

    def reset(self) -> None:
        """Clear captured embeddings (not accumulated statistics)."""
        self.layer_embeddings.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()
```

---

## Module 3: Attention Analyzer

**File**: `src/diagnostics/attention_analyzer.py`

**Purpose**: Extract and analyze attention patterns from Transformer decoder and HGT encoder.

### Implementation

```python
"""
Attention pattern analysis for Transformer and HGT models.

Extracts attention weights to diagnose:
1. Head specialization (do different heads learn different patterns?)
2. Attention entropy (uniform vs peaked distributions)
3. Positional biases (local vs global attention)
4. Copy mechanism usage (decoder copy vs generate decisions)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class AttentionAnalyzer:
    """
    Analyze attention patterns from multi-head attention layers.

    Tracks:
    - Per-head attention entropy
    - Attention distance statistics (local vs global)
    - Head specialization (variance across heads)
    - Copy probability distribution (for decoder)
    """

    def __init__(self, max_samples: int = 100):
        """
        Initialize attention analyzer.

        Args:
            max_samples: Maximum attention matrices to store (memory limit)
        """
        self.max_samples = max_samples
        self.attention_weights: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: nn.Module, layer_filter: Optional[callable] = None) -> None:
        """
        Register hooks on attention layers.

        Args:
            model: Model containing attention layers
            layer_filter: Optional filter for layer names
        """
        if layer_filter is None:
            def default_filter(name: str, module: nn.Module) -> bool:
                # MultiheadAttention or HGT attention layers
                return isinstance(module, nn.MultiheadAttention) or \
                       'Attention' in type(module).__name__
            layer_filter = default_filter

        for name, module in model.named_modules():
            if layer_filter(name, module):
                if isinstance(module, nn.MultiheadAttention):
                    # PyTorch MultiheadAttention
                    hook = module.register_forward_hook(
                        self._make_mha_hook(name)
                    )
                    self.hooks.append(hook)
                else:
                    # Custom attention (HGT, etc.) - may need custom extraction
                    hook = module.register_forward_hook(
                        self._make_generic_hook(name)
                    )
                    self.hooks.append(hook)

    def _make_mha_hook(self, layer_name: str):
        """Hook for PyTorch MultiheadAttention."""
        def hook(module, input, output):
            # MultiheadAttention returns (output, attention_weights)
            if isinstance(output, tuple) and len(output) == 2:
                attn_weights = output[1]  # [batch, tgt_len, src_len]
                if attn_weights is not None:
                    self.attention_weights[layer_name].append(attn_weights.detach().cpu())

                    # Limit memory
                    if len(self.attention_weights[layer_name]) > self.max_samples:
                        self.attention_weights[layer_name].pop(0)

        return hook

    def _make_generic_hook(self, layer_name: str):
        """Hook for custom attention layers (extract from output if available)."""
        def hook(module, input, output):
            # Try to extract attention weights from output
            # Common patterns: (output, attn_weights) or dict with 'attn'
            attn = None

            if isinstance(output, tuple):
                # Look for tensor with attention-like shape
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() in [2, 3]:
                        # Heuristic: attention is usually [batch, tgt, src] or [tgt, src]
                        attn = item
                        break
            elif isinstance(output, dict) and 'attn' in output:
                attn = output['attn']

            if attn is not None:
                self.attention_weights[layer_name].append(attn.detach().cpu())

                if len(self.attention_weights[layer_name]) > self.max_samples:
                    self.attention_weights[layer_name].pop(0)

        return hook

    def compute_entropy(self, attn: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution.

        High entropy → uniform attention (less focused)
        Low entropy → peaked attention (highly selective)

        Args:
            attn: [batch, tgt_len, src_len] or [tgt_len, src_len]

        Returns:
            Mean entropy across batch/queries
        """
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        # Add small epsilon for numerical stability
        attn = attn + 1e-12

        # Entropy: -sum(p * log(p))
        entropy = -(attn * torch.log(attn)).sum(dim=-1)

        return float(entropy.mean().item())

    def compute_attention_distance(self, attn: torch.Tensor) -> Tuple[float, float]:
        """
        Compute mean attention distance (local vs global).

        Distance = weighted sum of |query_pos - key_pos|

        Args:
            attn: [batch, tgt_len, src_len]

        Returns:
            (mean_distance, std_distance)
        """
        if attn.dim() == 2:
            attn = attn.unsqueeze(0)

        batch, tgt_len, src_len = attn.shape

        # Position indices
        tgt_pos = torch.arange(tgt_len).unsqueeze(1).float()  # [tgt, 1]
        src_pos = torch.arange(src_len).unsqueeze(0).float()  # [1, src]

        # Distance matrix [tgt, src]
        distances = torch.abs(tgt_pos - src_pos)

        # Weighted distance
        weighted_dists = (attn * distances.unsqueeze(0)).sum(dim=-1)  # [batch, tgt]

        return float(weighted_dists.mean().item()), float(weighted_dists.std().item())

    def compute_head_specialization(self, attn: torch.Tensor, num_heads: int) -> float:
        """
        Measure variance in attention patterns across heads.

        High variance → heads specialize in different patterns
        Low variance → heads learn similar patterns (redundant)

        Args:
            attn: [batch * num_heads, tgt_len, src_len] (concatenated heads)
            num_heads: Number of attention heads

        Returns:
            Variance of per-head attention statistics
        """
        batch_heads = attn.size(0)
        batch_size = batch_heads // num_heads

        if batch_size == 0:
            return 0.0

        # Reshape to [batch, num_heads, tgt, src]
        attn = attn.view(batch_size, num_heads, attn.size(1), attn.size(2))

        # Compute per-head statistics (e.g., mean attention per position)
        head_stats = attn.mean(dim=(0, 2, 3))  # [num_heads]

        return float(head_stats.std().item())

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute attention statistics.

        Returns:
            {
                layer_name: {
                    'entropy_mean': float,
                    'entropy_std': float,
                    'distance_mean': float,
                    'distance_std': float,
                }
            }
        """
        stats = {}

        for layer_name, attn_list in self.attention_weights.items():
            if not attn_list:
                continue

            entropies = []
            distances_mean = []
            distances_std = []

            for attn in attn_list:
                entropies.append(self.compute_entropy(attn))
                dist_mean, dist_std = self.compute_attention_distance(attn)
                distances_mean.append(dist_mean)
                distances_std.append(dist_std)

            stats[layer_name] = {
                'entropy_mean': float(np.mean(entropies)),
                'entropy_std': float(np.std(entropies)),
                'distance_mean': float(np.mean(distances_mean)),
                'distance_std': float(np.std(distances_std)),
            }

        return stats

    def log_to_tensorboard(self, writer, global_step: int, prefix: str = 'diagnostics/attention') -> None:
        """Log attention statistics to TensorBoard."""
        stats = self.get_statistics()

        for layer_name, layer_stats in stats.items():
            clean_name = layer_name.replace('.', '/')
            for metric_name, value in layer_stats.items():
                writer.add_scalar(
                    f'{prefix}/{clean_name}/{metric_name}',
                    value,
                    global_step
                )

    def reset(self) -> None:
        """Clear stored attention weights."""
        self.attention_weights.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()
```

---

## Module 4: Diversity Metrics

**File**: `src/diagnostics/diversity_metrics.py`

**Purpose**: Measure node embedding diversity to detect over-squashing in message-passing GNNs.

### Implementation

```python
"""
Node embedding diversity metrics for over-squashing detection.

Over-squashing occurs when GNNs compress long-range dependencies into fixed-size
vectors, losing information. Symptoms:
- Low embedding variance across nodes
- High correlation between distant nodes
- Bottleneck nodes dominate message passing

Metrics:
1. Embedding variance (per dimension)
2. Neighbor correlation (same vs different subgraphs)
3. Bottleneck score (nodes with high degree centrality but low diversity)
"""

import torch
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


class DiversityMetrics:
    """
    Measure embedding diversity to detect over-squashing.

    Tracks:
    - Per-dimension variance
    - Mean pairwise distance
    - Correlation between neighbor embeddings
    """

    def __init__(self):
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.batch_assignments: Optional[torch.Tensor] = None
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """Register hooks on GNN layers."""
        if layer_names is None:
            layer_names = []
            for name, module in model.named_modules():
                if any(x in type(module).__name__ for x in ['GATConv', 'HGTConv', 'RGCNConv', 'GatedGraphConv']):
                    layer_names.append(name)

        self.layer_names = layer_names

        for name in layer_names:
            module = dict(model.named_modules())[name]
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Capture layer embeddings."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.embeddings[layer_name] = output.detach().clone()
        return hook

    def compute_variance(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute embedding variance statistics.

        Args:
            embeddings: [num_nodes, hidden_dim]

        Returns:
            {
                'var_mean': Mean variance across dimensions,
                'var_std': Std of variance across dimensions,
                'var_min': Minimum variance (dead dimensions),
                'var_max': Maximum variance,
            }
        """
        # Per-dimension variance
        var_per_dim = embeddings.var(dim=0)  # [hidden_dim]

        return {
            'var_mean': float(var_per_dim.mean().item()),
            'var_std': float(var_per_dim.std().item()),
            'var_min': float(var_per_dim.min().item()),
            'var_max': float(var_per_dim.max().item()),
        }

    def compute_pairwise_distance(self, embeddings: torch.Tensor, sample_size: int = 500) -> float:
        """
        Mean pairwise L2 distance (sampled).

        Low distance → embeddings collapsed
        High distance → diverse embeddings
        """
        num_nodes = embeddings.size(0)
        if num_nodes <= 1:
            return 0.0

        # Sample for efficiency
        if num_nodes > sample_size:
            indices = torch.randperm(num_nodes)[:sample_size]
            embeddings = embeddings[indices]
            num_nodes = sample_size

        # Pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)

        return float(dists[mask].mean().item())

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute diversity statistics for all layers.

        Returns:
            {layer_name: {variance_metrics, pairwise_distance}}
        """
        stats = {}

        for layer_name, embeddings in self.embeddings.items():
            layer_stats = {}

            # Variance
            var_stats = self.compute_variance(embeddings)
            layer_stats.update(var_stats)

            # Pairwise distance
            layer_stats['pairwise_distance'] = self.compute_pairwise_distance(embeddings)

            stats[layer_name] = layer_stats

        return stats

    def detect_over_squashing(self, var_threshold: float = 0.01,
                             distance_threshold: float = 0.1) -> Dict[str, List[str]]:
        """
        Detect over-squashing indicators.

        Args:
            var_threshold: Minimum acceptable variance
            distance_threshold: Minimum acceptable pairwise distance

        Returns:
            {
                'low_variance': [layer_names],
                'low_distance': [layer_names],
            }
        """
        stats = self.get_statistics()

        low_var_layers = []
        low_dist_layers = []

        for layer_name, layer_stats in stats.items():
            if layer_stats.get('var_mean', 1.0) < var_threshold:
                low_var_layers.append(layer_name)
            if layer_stats.get('pairwise_distance', 1.0) < distance_threshold:
                low_dist_layers.append(layer_name)

        return {
            'low_variance': low_var_layers,
            'low_distance': low_dist_layers,
        }

    def log_to_tensorboard(self, writer, global_step: int, prefix: str = 'diagnostics/diversity') -> None:
        """Log diversity metrics to TensorBoard."""
        stats = self.get_statistics()

        for layer_name, layer_stats in stats.items():
            clean_name = layer_name.replace('.', '/')
            for metric_name, value in layer_stats.items():
                writer.add_scalar(
                    f'{prefix}/{clean_name}/{metric_name}',
                    value,
                    global_step
                )

    def reset(self) -> None:
        """Clear embeddings."""
        self.embeddings.clear()

    def remove_hooks(self) -> None:
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()
```

---

## Module 5: Visualizers

**File**: `src/diagnostics/visualizers.py`

**Purpose**: Matplotlib-based plotting utilities for diagnostic data.

### Implementation

```python
"""
Visualization utilities for diagnostic data.

Creates publication-quality plots for:
- Layer-wise activation/gradient norms
- Representation collapse progression
- Attention heatmaps
- Embedding diversity across layers
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def plot_activation_norms(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot layer-wise activation and gradient norms.

    Args:
        stats: Output from ActivationMonitor.get_statistics()
        output_path: Path to save figure (None = show)
        figsize: Figure size
    """
    layers = sorted(stats.keys())

    act_means = [stats[l].get('activation_norm_mean', 0) for l in layers]
    grad_means = [stats[l].get('gradient_norm_mean', 0) for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Activation norms
    ax1.bar(range(len(layers)), act_means, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Activation Norm (L2)', fontsize=11)
    ax1.set_title('Layer-wise Activation Norms', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Gradient norms
    if any(g > 0 for g in grad_means):
        ax2.bar(range(len(layers)), grad_means, alpha=0.7, color='coral')
        ax2.axhline(y=1e-4, color='red', linestyle='--', label='Vanishing threshold')
        ax2.axhline(y=100, color='orange', linestyle='--', label='Exploding threshold')
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.set_ylabel('Gradient Norm (L2)', fontsize=11)
        ax2.set_title('Layer-wise Gradient Norms', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_collapse_metrics(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot representation collapse metrics.

    Args:
        stats: Output from CollapseDetector.get_statistics()
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Inter-layer similarity
    if 'interlayer_similarity' in stats:
        similarities = stats['interlayer_similarity']
        layer_pairs = list(similarities.keys())
        values = list(similarities.values())

        axes[0].bar(range(len(layer_pairs)), values, alpha=0.7, color='steelblue')
        axes[0].axhline(y=0.95, color='red', linestyle='--', label='Collapse threshold')
        axes[0].set_xlabel('Layer Pair', fontsize=10)
        axes[0].set_ylabel('Cosine Similarity', fontsize=10)
        axes[0].set_title('Inter-Layer Similarity', fontsize=11, fontweight='bold')
        axes[0].set_xticks(range(len(layer_pairs)))
        axes[0].set_xticklabels(layer_pairs, rotation=45, ha='right', fontsize=8)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

    # Diversity
    if 'diversity' in stats:
        diversity = stats['diversity']
        layers = list(diversity.keys())
        values = list(diversity.values())

        axes[1].bar(range(len(layers)), values, alpha=0.7, color='orange')
        axes[1].axhline(y=0.1, color='red', linestyle='--', label='Low diversity threshold')
        axes[1].set_xlabel('Layer', fontsize=10)
        axes[1].set_ylabel('Mean Pairwise Distance', fontsize=10)
        axes[1].set_title('Embedding Diversity', fontsize=11, fontweight='bold')
        axes[1].set_xticks(range(len(layers)))
        axes[1].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right', fontsize=8)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

    # Effective rank
    if 'effective_rank' in stats:
        ranks = stats['effective_rank']
        layers = list(ranks.keys())
        values = list(ranks.values())

        axes[2].bar(range(len(layers)), values, alpha=0.7, color='green')
        axes[2].set_xlabel('Layer', fontsize=10)
        axes[2].set_ylabel('Effective Rank', fontsize=10)
        axes[2].set_title('Representation Rank', fontsize=11, fontweight='bold')
        axes[2].set_xticks(range(len(layers)))
        axes[2].set_xticklabels([l.split('.')[-1] for l in layers], rotation=45, ha='right', fontsize=8)
        axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Attention Heatmap",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot attention heatmap.

    Args:
        attention_weights: [tgt_len, src_len] attention matrix
        output_path: Path to save
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar=True,
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )

    ax.set_xlabel('Source Position', fontsize=11)
    ax.set_ylabel('Target Position', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_diversity_progression(
    stats_history: List[Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot diversity metrics over training steps.

    Args:
        stats_history: List of DiversityMetrics.get_statistics() outputs
        output_path: Path to save
        figsize: Figure size
    """
    # Extract layer-wise diversity over time
    layer_names = list(stats_history[0].keys()) if stats_history else []

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for layer in layer_names:
        variances = [stats[layer].get('var_mean', 0) for stats in stats_history]
        distances = [stats[layer].get('pairwise_distance', 0) for stats in stats_history]

        axes[0].plot(variances, label=layer.split('.')[-1], alpha=0.7)
        axes[1].plot(distances, label=layer.split('.')[-1], alpha=0.7)

    axes[0].set_xlabel('Training Step', fontsize=11)
    axes[0].set_ylabel('Embedding Variance', fontsize=11)
    axes[0].set_title('Variance Progression', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Training Step', fontsize=11)
    axes[1].set_ylabel('Pairwise Distance', fontsize=11)
    axes[1].set_title('Distance Progression', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()
```

---

## Integration with BaseTrainer

**File**: `src/training/base_trainer.py` (additions)

```python
# Add to BaseTrainer class:

def enable_diagnostics(
    self,
    activation_monitoring: bool = False,
    collapse_detection: bool = False,
    attention_analysis: bool = False,
    diversity_tracking: bool = False,
    log_interval: int = 100,
) -> None:
    """
    Enable diagnostic tools for model debugging.

    Args:
        activation_monitoring: Track activation/gradient norms
        collapse_detection: Track representation collapse
        attention_analysis: Analyze attention patterns
        diversity_tracking: Track embedding diversity
        log_interval: Steps between diagnostic logging
    """
    self._diagnostic_log_interval = log_interval
    self._diagnostic_step_counter = 0

    if activation_monitoring:
        from src.diagnostics.activation_monitor import ActivationMonitor
        self._activation_monitor = ActivationMonitor(track_gradients=True)
        self._activation_monitor.register_hooks(self.model)
        logger.info("✓ Activation monitoring enabled")

    if collapse_detection:
        from src.diagnostics.collapse_detector import CollapseDetector
        self._collapse_detector = CollapseDetector()
        self._collapse_detector.register_hooks(self.model)
        logger.info("✓ Collapse detection enabled")

    if attention_analysis:
        from src.diagnostics.attention_analyzer import AttentionAnalyzer
        self._attention_analyzer = AttentionAnalyzer()
        self._attention_analyzer.register_hooks(self.model)
        logger.info("✓ Attention analysis enabled")

    if diversity_tracking:
        from src.diagnostics.diversity_metrics import DiversityMetrics
        self._diversity_tracker = DiversityMetrics()
        self._diversity_tracker.register_hooks(self.model)
        logger.info("✓ Diversity tracking enabled")

def log_diagnostics(self, force: bool = False) -> None:
    """
    Log diagnostic metrics to TensorBoard.

    Call after backward pass in training loop.

    Args:
        force: Force logging even if not at interval
    """
    if not force:
        self._diagnostic_step_counter += 1
        if self._diagnostic_step_counter % self._diagnostic_log_interval != 0:
            return

    if self.tensorboard_writer is None:
        return

    # Activation monitoring
    if hasattr(self, '_activation_monitor'):
        self._activation_monitor.log_to_tensorboard(
            self.tensorboard_writer,
            self.global_step
        )

        # Check for pathologies
        pathologies = self._activation_monitor.detect_pathologies()
        if any(pathologies.values()):
            logger.warning(
                f"[Step {self.global_step}] Detected pathologies: {pathologies}"
            )

        self._activation_monitor.reset()

    # Collapse detection
    if hasattr(self, '_collapse_detector'):
        self._collapse_detector.compute_statistics()
        self._collapse_detector.log_to_tensorboard(
            self.tensorboard_writer,
            self.global_step
        )

        collapse_flags = self._collapse_detector.detect_collapse()
        if any(collapse_flags.values()):
            logger.warning(
                f"[Step {self.global_step}] Collapse detected: {collapse_flags}"
            )

        self._collapse_detector.reset()

    # Attention analysis
    if hasattr(self, '_attention_analyzer'):
        self._attention_analyzer.log_to_tensorboard(
            self.tensorboard_writer,
            self.global_step
        )
        self._attention_analyzer.reset()

    # Diversity tracking
    if hasattr(self, '_diversity_tracker'):
        self._diversity_tracker.log_to_tensorboard(
            self.tensorboard_writer,
            self.global_step
        )

        over_squashing = self._diversity_tracker.detect_over_squashing()
        if any(over_squashing.values()):
            logger.warning(
                f"[Step {self.global_step}] Over-squashing detected: {over_squashing}"
            )

        self._diversity_tracker.reset()

def cleanup_diagnostics(self) -> None:
    """Remove all diagnostic hooks."""
    for attr in ['_activation_monitor', '_collapse_detector', '_attention_analyzer', '_diversity_tracker']:
        if hasattr(self, attr):
            getattr(self, attr).remove_hooks()
            delattr(self, attr)
```

---

## Integration with AblationTrainer

**File**: `src/training/ablation_trainer.py` (additions)

```python
# Add to AblationTrainer class:

def enable_diagnostics(self, **kwargs) -> None:
    """
    Enable diagnostics for ablation study.

    Wrapper around diagnostic enablement for encoder comparison.
    """
    # Use activation monitoring and collapse detection for encoder comparison
    kwargs.setdefault('activation_monitoring', True)
    kwargs.setdefault('collapse_detection', True)
    kwargs.setdefault('log_interval', 50)

    from src.diagnostics.activation_monitor import ActivationMonitor
    from src.diagnostics.collapse_detector import CollapseDetector

    if kwargs.get('activation_monitoring'):
        self._activation_monitor = ActivationMonitor()
        self._activation_monitor.register_hooks(self.encoder)

    if kwargs.get('collapse_detection'):
        self._collapse_detector = CollapseDetector()
        self._collapse_detector.register_hooks(self.encoder)

    self._diagnostic_log_interval = kwargs.get('log_interval', 50)
    self._diagnostic_counter = 0

    logger.info(f"Diagnostics enabled for encoder: {self.encoder_name}")
```

---

## Standalone Visualization Script

**File**: `scripts/visualize_diagnostics.py`

```python
#!/usr/bin/env python3
"""
Visualize diagnostic metrics from TensorBoard logs or saved checkpoints.

Usage:
    # From TensorBoard logs
    python scripts/visualize_diagnostics.py --logdir runs/experiment_1 --output plots/

    # From saved diagnostic data
    python scripts/visualize_diagnostics.py --data diagnostics.pkl --output plots/
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from src.diagnostics.visualizers import (
    plot_activation_norms,
    plot_collapse_metrics,
    plot_attention_heatmap,
    plot_diversity_progression,
)


def parse_tensorboard_logs(logdir: str) -> Dict:
    """
    Parse TensorBoard logs to extract diagnostic metrics.

    Args:
        logdir: Path to TensorBoard log directory

    Returns:
        Dictionary with diagnostic data
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError("tensorboard required for log parsing. Install: pip install tensorboard")

    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    # Extract scalar tags
    tags = ea.Tags()['scalars']

    data = {
        'activation': {},
        'collapse': {},
        'attention': {},
        'diversity': {},
    }

    # Parse tags by prefix
    for tag in tags:
        if 'diagnostics/activation' in tag:
            values = [(e.step, e.value) for e in ea.Scalars(tag)]
            data['activation'][tag] = values
        elif 'diagnostics/collapse' in tag:
            values = [(e.step, e.value) for e in ea.Scalars(tag)]
            data['collapse'][tag] = values
        elif 'diagnostics/attention' in tag:
            values = [(e.step, e.value) for e in ea.Scalars(tag)]
            data['attention'][tag] = values
        elif 'diagnostics/diversity' in tag:
            values = [(e.step, e.value) for e in ea.Scalars(tag)]
            data['diversity'][tag] = values

    return data


def main():
    parser = argparse.ArgumentParser(description='Visualize diagnostic metrics')
    parser.add_argument('--logdir', type=str, help='TensorBoard log directory')
    parser.add_argument('--data', type=str, help='Pickled diagnostic data')
    parser.add_argument('--output', type=str, default='plots/', help='Output directory')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'])

    args = parser.parse_args()

    if not args.logdir and not args.data:
        parser.error("Must specify --logdir or --data")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.logdir:
        print(f"Parsing TensorBoard logs: {args.logdir}")
        data = parse_tensorboard_logs(args.logdir)
    else:
        print(f"Loading diagnostic data: {args.data}")
        with open(args.data, 'rb') as f:
            data = pickle.load(f)

    # Generate plots
    if 'activation' in data and data['activation']:
        print("Generating activation plots...")
        # Convert TensorBoard format to statistics format
        # (simplified - actual implementation needs proper aggregation)
        plot_activation_norms(
            data['activation'],
            output_path=str(output_dir / f'activation_norms.{args.format}')
        )

    if 'collapse' in data and data['collapse']:
        print("Generating collapse plots...")
        plot_collapse_metrics(
            data['collapse'],
            output_path=str(output_dir / f'collapse_metrics.{args.format}')
        )

    print(f"\n✓ Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
```

---

## Configuration Integration

**File**: `configs/diagnostics.yaml` (new)

```yaml
# Diagnostic configuration for model debugging

diagnostics:
  enabled: false  # Set to true to enable diagnostics

  # Activation monitoring
  activation_monitoring:
    enabled: true
    track_gradients: true
    log_interval: 100

  # Collapse detection
  collapse_detection:
    enabled: true
    sample_size: 1000
    similarity_threshold: 0.95
    diversity_threshold: 0.1
    rank_threshold_ratio: 0.1

  # Attention analysis
  attention_analysis:
    enabled: true
    max_samples: 100

  # Diversity tracking
  diversity_tracking:
    enabled: true
    sample_size: 500
    var_threshold: 0.01
    distance_threshold: 0.1

  # Logging
  log_interval: 100
  save_checkpoints: true
  checkpoint_dir: "diagnostics/checkpoints"
```

---

## Usage Examples

### Example 1: Enable Diagnostics in Training

```python
# In scripts/train.py or training script

from src.training.base_trainer import BaseTrainer
from src.models.full_model import MBADeobfuscator

# Initialize model and trainer
model = MBADeobfuscator(encoder_type='hgt')
trainer = BaseTrainer(model, config, checkpoint_dir='checkpoints/')

# Initialize TensorBoard
trainer.init_tensorboard(log_dir='runs/hgt_diagnostics')

# Enable diagnostics
trainer.enable_diagnostics(
    activation_monitoring=True,
    collapse_detection=True,
    attention_analysis=True,
    diversity_tracking=True,
    log_interval=100,
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        output = model(batch)
        loss = compute_loss(output, batch)

        # Backward
        trainer.backward(loss)

        # Log diagnostics (every 100 steps)
        trainer.log_diagnostics()

# Cleanup
trainer.cleanup_diagnostics()
trainer.close()
```

### Example 2: Ablation Study with Diagnostics

```python
# In scripts/run_ablation.py

from src.training.ablation_trainer import AblationTrainer

# Initialize ablation trainer
config = load_config('configs/ablation.yaml')
trainer = AblationTrainer(config)

# Enable diagnostics for encoder comparison
trainer.enable_diagnostics(
    activation_monitoring=True,
    collapse_detection=True,
)

# Train
for epoch in range(epochs):
    trainer.train_epoch(train_loader)
```

### Example 3: Inference-Time Diagnostics

```python
# For debugging inference issues

from src.models.full_model import MBADeobfuscator
from src.diagnostics.attention_analyzer import AttentionAnalyzer
from src.diagnostics.activation_monitor import ActivationMonitor

model = MBADeobfuscator.load_checkpoint('best.pt')
model.eval()

# Enable inference diagnostics
attn_analyzer = AttentionAnalyzer()
attn_analyzer.register_hooks(model.decoder)

act_monitor = ActivationMonitor(track_gradients=False)
act_monitor.register_hooks(model)

# Run inference
with torch.no_grad():
    output = model(test_batch)

# Analyze
attn_stats = attn_analyzer.get_statistics()
act_stats = act_monitor.get_statistics()

print("Attention entropy:", attn_stats)
print("Activation norms:", act_stats)

# Visualize
from src.diagnostics.visualizers import plot_attention_heatmap
import numpy as np

attn_weights = attn_analyzer.attention_weights['decoder.layers.0'][0].numpy()
plot_attention_heatmap(attn_weights, output_path='attention_layer0.png')
```

---

## Testing Strategy

**File**: `tests/test_diagnostics.py` (new)

```python
"""
Unit tests for diagnostic tools.
"""

import pytest
import torch
import torch.nn as nn
from src.diagnostics.activation_monitor import ActivationMonitor
from src.diagnostics.collapse_detector import CollapseDetector
from src.diagnostics.attention_analyzer import AttentionAnalyzer
from src.diagnostics.diversity_metrics import DiversityMetrics


class SimpleMLP(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        return self.layers(x)


def test_activation_monitor():
    """Test activation monitoring."""
    model = SimpleMLP()
    monitor = ActivationMonitor()
    monitor.register_hooks(model)

    # Forward pass
    x = torch.randn(4, 10)
    output = model(x)

    # Check statistics
    stats = monitor.get_statistics()
    assert len(stats) > 0
    assert 'activation_norm_mean' in stats['layers.0']

    monitor.remove_hooks()


def test_collapse_detector():
    """Test collapse detection."""
    model = SimpleMLP()
    detector = CollapseDetector()
    detector.register_hooks(model, layer_names=['layers.0', 'layers.2'])

    # Forward pass
    x = torch.randn(4, 10)
    output = model(x)

    # Compute statistics
    detector.compute_statistics()
    stats = detector.get_statistics()

    assert 'diversity' in stats
    assert len(stats['diversity']) > 0

    detector.remove_hooks()


def test_attention_analyzer():
    """Test attention analysis."""
    attn_layer = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    analyzer = AttentionAnalyzer()
    analyzer.register_hooks(attn_layer)

    # Forward pass
    query = torch.randn(2, 10, 16)
    key = torch.randn(2, 10, 16)
    value = torch.randn(2, 10, 16)

    output, attn_weights = attn_layer(query, key, value, need_weights=True)

    # Check statistics
    stats = analyzer.get_statistics()
    assert len(stats) > 0

    analyzer.remove_hooks()


def test_diversity_metrics():
    """Test diversity tracking."""
    model = SimpleMLP()
    tracker = DiversityMetrics()
    tracker.register_hooks(model, layer_names=['layers.0'])

    # Forward pass
    x = torch.randn(4, 10)
    output = model(x)

    # Check statistics
    stats = tracker.get_statistics()
    assert 'layers.0' in stats
    assert 'var_mean' in stats['layers.0']

    tracker.remove_hooks()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Performance Considerations

### Overhead Analysis

| Diagnostic Tool | Overhead (Training) | Memory Overhead | When to Use |
|-----------------|---------------------|-----------------|-------------|
| Activation Monitor | ~2-3% | Minimal (~10MB) | Always safe for debugging |
| Collapse Detector | ~5-8% | Moderate (~50MB per layer) | Encoder debugging, depth >8 |
| Attention Analyzer | ~3-5% | Moderate (~100MB) | Decoder debugging, attention issues |
| Diversity Metrics | ~5-7% | Moderate (~50MB) | Over-squashing diagnosis |
| **All combined** | ~12-15% | High (~200MB) | Detailed debugging only |

**Recommendations**:
- Enable selectively based on suspected issue
- Use higher `log_interval` (500-1000) for minimal overhead
- Disable after identifying issue
- Use only on dev set, not full training

### Memory Management

```python
# For long training runs, periodically save and clear diagnostic data

# Every 1000 steps
if step % 1000 == 0 and hasattr(trainer, '_collapse_detector'):
    stats = trainer._collapse_detector.get_statistics()

    # Save to disk
    import pickle
    with open(f'diagnostics/collapse_step_{step}.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # Clear accumulated data
    trainer._collapse_detector.reset()
```

---

## Documentation Updates

### README.md Addition

Add to **Quick Commands Reference**:

```markdown
# Diagnostics
python scripts/train.py --phase 2 --enable-diagnostics --diagnostic-interval 100
python scripts/visualize_diagnostics.py --logdir runs/experiment --output plots/
```

### ARCHITECTURE.md Addition

Add section:

```markdown
## Diagnostic Tools

Monitor model health during training:

- **Activation Monitor**: Gradient vanishing/explosion detection
- **Collapse Detector**: Over-smoothing in deep GNNs
- **Attention Analyzer**: Head specialization, attention entropy
- **Diversity Metrics**: Embedding diversity, over-squashing

See `src/diagnostics/` for implementation.
```

---

## Milestone Summary

### Milestone 1: Core Diagnostic Modules (Priority: P0)
- [ ] Implement `src/diagnostics/activation_monitor.py` (180 LOC)
- [ ] Implement `src/diagnostics/collapse_detector.py` (150 LOC)
- [ ] Implement `src/diagnostics/attention_analyzer.py` (220 LOC)
- [ ] Implement `src/diagnostics/diversity_metrics.py` (160 LOC)
- [ ] Create `tests/test_diagnostics.py` (150 LOC)

**Deliverable**: Functional diagnostic tools with unit tests

### Milestone 2: Visualization Utilities (Priority: P1)
- [ ] Implement `src/diagnostics/visualizers.py` (250 LOC)
- [ ] Create `scripts/visualize_diagnostics.py` (200 LOC)
- [ ] Add visualization tests

**Deliverable**: Standalone plotting utilities

### Milestone 3: Trainer Integration (Priority: P0)
- [ ] Extend `src/training/base_trainer.py` (+100 LOC)
- [ ] Extend `src/training/ablation_trainer.py` (+50 LOC)
- [ ] Create `configs/diagnostics.yaml`
- [ ] Integration tests

**Deliverable**: Hook-based diagnostics in training loop

### Milestone 4: Documentation & Polish (Priority: P2)
- [ ] Update `ARCHITECTURE.md`
- [ ] Update `README.md`
- [ ] Create usage examples in `docs/DIAGNOSTICS.md`
- [ ] Add to API reference

**Deliverable**: Complete documentation

---

## Acceptance Criteria

1. ✅ All diagnostic tools run without errors on sample model
2. ✅ Overhead < 15% with all diagnostics enabled
3. ✅ TensorBoard logging works correctly
4. ✅ Visualizations render correctly (no matplotlib errors)
5. ✅ Integration with BaseTrainer is non-invasive (opt-in)
6. ✅ Unit test coverage > 80% for diagnostic modules
7. ✅ Documentation includes usage examples
8. ✅ Ablation study can compare encoder health metrics

---

## Future Enhancements (Out of Scope)

1. **Real-time dashboard**: Web-based diagnostic viewer
2. **Automatic pathology detection**: ML-based anomaly detection
3. **Comparative analysis**: Multi-run diagnostic comparison
4. **Profiling integration**: CUDA kernel profiling
5. **Distributed diagnostics**: Multi-GPU metric aggregation

---

**Implementation Priority**: P2 (Developer tooling)
**Estimated Timeline**: 3-4 days full-time
**Dependencies**: None (optional TensorBoard for logging)
**Risk**: Low (isolated from core training logic)
