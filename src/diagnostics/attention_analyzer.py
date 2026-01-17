"""
Attention pattern extraction and analysis for Transformer/HGT models.

Extracts and analyzes attention weights to understand:
- Attention entropy (uniform vs focused)
- Attention distance (local vs global patterns)
- Head specialization (diversity across heads)

CRITICAL FIXES:
- Validates attention weight normalization before entropy computation
- Atexit cleanup instead of __del__
- Graceful handling of None TensorBoard writer
"""

import atexit
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze attention patterns in Transformer decoder and HGT encoder.

    Tracks:
    - Attention entropy per head (uniform vs peaked distributions)
    - Attention distance statistics (how far attention spans)
    - Head specialization (variance across heads)

    Usage:
        analyzer = AttentionAnalyzer()
        analyzer.register_hooks(model)

        # Forward pass (attention weights captured automatically)
        output = model(batch)

        stats = analyzer.get_statistics()
        analyzer.log_to_tensorboard(writer, global_step)
        analyzer.reset()

        # Cleanup (REQUIRED)
        analyzer.remove_hooks()
    """

    def __init__(self, max_seq_length_to_store: int = 128):
        """
        Initialize attention analyzer.

        Args:
            max_seq_length_to_store: Skip storing attention for sequences
                                      longer than this (memory efficiency)
        """
        self.max_seq_length_to_store = max_seq_length_to_store

        # Storage for attention weights
        # Format: {layer_name: [(attn_weights, query_positions, key_positions)]}
        self.attention_weights: Dict[str, List[Tuple[torch.Tensor, ...]]] = defaultdict(list)

        # Hook handles
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register atexit handler (CRITICAL FIX)
        atexit.register(self.remove_hooks)

    def register_hooks(
        self,
        model: nn.Module,
        module_types: Optional[List[str]] = None,
    ) -> None:
        """
        Register hooks on attention modules.

        CRITICAL FIX: Clears stale hooks before registering.

        Args:
            model: PyTorch model
            module_types: Attention module class names to hook
                         (default: ['MultiheadAttention', 'HGTConv'])
        """
        # Clear stale hooks (CRITICAL FIX)
        self.remove_hooks()

        if module_types is None:
            module_types = ['MultiheadAttention', 'HGTConv']

        hooked_count = 0
        for name, module in model.named_modules():
            module_type = type(module).__name__

            # Hook attention modules
            if any(att_type in module_type for att_type in module_types):
                # For MultiheadAttention, we need to modify forward to return weights
                # This is tricky - we'll use a wrapper approach
                if 'MultiheadAttention' in module_type:
                    hook_handle = module.register_forward_hook(
                        self._make_mha_hook(name)
                    )
                    self.hooks.append(hook_handle)
                    hooked_count += 1
                # For HGT, attention weights might be in edge_attr or separate
                elif 'HGT' in module_type:
                    hook_handle = module.register_forward_hook(
                        self._make_hgt_hook(name)
                    )
                    self.hooks.append(hook_handle)
                    hooked_count += 1

        logger.info(f"Registered {hooked_count} hooks for attention analysis")

    def _make_mha_hook(self, layer_name: str):
        """
        Create hook for MultiheadAttention.

        NOTE: This requires MultiheadAttention to be called with need_weights=True.
        """
        def hook(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights) when need_weights=True
            if isinstance(output, tuple) and len(output) == 2:
                _, attn_weights = output

                if attn_weights is not None:
                    # attn_weights: [batch_size, tgt_len, src_len] or
                    #               [batch_size * num_heads, tgt_len, src_len]

                    # Memory check
                    if attn_weights.size(-1) > self.max_seq_length_to_store:
                        return  # Skip long sequences

                    self.attention_weights[layer_name].append(
                        (attn_weights.detach(), None, None)
                    )
        return hook

    def _make_hgt_hook(self, layer_name: str):
        """Create hook for HGT attention (if available)."""
        def hook(module, input, output):
            # HGT attention extraction is model-specific
            # Placeholder for now - would need to inspect HGT internals
            pass
        return hook

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.

        CRITICAL FIX: Robust cleanup.
        """
        removed_count = 0
        for hook in self.hooks:
            try:
                hook.remove()
                removed_count += 1
            except Exception as e:
                logger.debug(f"Hook removal warning: {e}")

        self.hooks.clear()

        if removed_count > 0:
            logger.info(f"Removed {removed_count} attention analyzer hooks")

    def reset(self) -> None:
        """Clear collected attention weights."""
        self.attention_weights.clear()

    def _compute_entropy(self, attn_weights: torch.Tensor) -> float:
        """
        Compute Shannon entropy of attention distribution.

        CRITICAL FIX: Validates attention weights are normalized before computing entropy.

        Args:
            attn_weights: [batch, tgt_len, src_len] or [batch * heads, tgt_len, src_len]

        Returns:
            Mean entropy across all distributions (higher = more uniform attention)
        """
        # CRITICAL FIX: Validate normalization
        # Sum along source dimension (last dim) should be ~1.0
        row_sums = attn_weights.sum(dim=-1)
        expected_sum = torch.ones_like(row_sums)

        if not torch.allclose(row_sums, expected_sum, atol=1e-3):
            logger.warning(
                f"Attention weights not normalized (sums: {row_sums.min():.4f} to {row_sums.max():.4f}). "
                "Normalizing before entropy computation. "
                "Ensure MultiheadAttention is called with need_weights=True and attention is after softmax."
            )
            # CRITICAL FIX: Normalize if not normalized
            attn_weights = attn_weights / (row_sums.unsqueeze(-1) + 1e-12)

        # Add small epsilon to prevent log(0)
        attn_safe = attn_weights + 1e-12

        # Entropy: -sum(p * log(p))
        entropy = -(attn_weights * torch.log(attn_safe)).sum(dim=-1)

        return entropy.mean().item()

    def _compute_mean_distance(self, attn_weights: torch.Tensor) -> float:
        """
        Compute mean attention distance (how far attention spans).

        For position-based attention, computes weighted distance between
        query and key positions.

        Args:
            attn_weights: [batch, tgt_len, src_len]

        Returns:
            Mean attention distance
        """
        batch_size, tgt_len, src_len = attn_weights.shape

        # Create position indices
        query_pos = torch.arange(tgt_len, device=attn_weights.device).unsqueeze(1)  # [tgt, 1]
        key_pos = torch.arange(src_len, device=attn_weights.device).unsqueeze(0)  # [1, src]

        # Distance matrix [tgt, src]
        distances = torch.abs(query_pos - key_pos).float()

        # Weighted mean distance per query position
        # attn_weights: [batch, tgt, src], distances: [tgt, src]
        weighted_distances = (attn_weights * distances.unsqueeze(0)).sum(dim=-1)  # [batch, tgt]

        return weighted_distances.mean().item()

    def _compute_head_specialization(self, attn_weights: torch.Tensor, num_heads: int) -> float:
        """
        Compute variance across attention heads (head specialization).

        High variance = heads specialize differently.
        Low variance = heads attend to similar positions.

        Args:
            attn_weights: [batch * num_heads, tgt_len, src_len]
            num_heads: Number of attention heads

        Returns:
            Standard deviation of attention patterns across heads
        """
        batch_heads, tgt_len, src_len = attn_weights.shape
        batch_size = batch_heads // num_heads

        if batch_size == 0:
            return 0.0

        # Reshape to [batch, num_heads, tgt_len, src_len]
        reshaped = attn_weights.reshape(batch_size, num_heads, tgt_len, src_len)

        # Compute variance across heads
        # Mean per head: [batch, tgt, src]
        head_variance = reshaped.var(dim=1)  # Variance across head dimension

        return head_variance.mean().item()

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute attention statistics from collected weights.

        Returns:
            {layer_name: {
                'entropy': float,  # Shannon entropy (higher = more uniform)
                'mean_distance': float,  # Average attention span
                'head_specialization': float,  # Variance across heads
            }}
        """
        stats = {}

        for layer_name, weight_list in self.attention_weights.items():
            if not weight_list:
                continue

            # Collect statistics across all batches
            entropies = []
            distances = []
            specializations = []

            for attn_weights, query_pos, key_pos in weight_list:
                # Compute entropy
                entropies.append(self._compute_entropy(attn_weights))

                # Compute mean distance (for sequential attention)
                if attn_weights.ndim == 3:
                    distances.append(self._compute_mean_distance(attn_weights))

                # Compute head specialization (if multi-head)
                # Heuristic: if first dimension is divisible by common head counts
                batch_size = attn_weights.size(0)
                for num_heads in [8, 16, 24, 32]:  # Common head counts
                    if batch_size % num_heads == 0:
                        specializations.append(
                            self._compute_head_specialization(attn_weights, num_heads)
                        )
                        break

            layer_stats = {}
            if entropies:
                layer_stats['entropy'] = float(np.mean(entropies))
            if distances:
                layer_stats['mean_distance'] = float(np.mean(distances))
            if specializations:
                layer_stats['head_specialization'] = float(np.mean(specializations))

            stats[layer_name] = layer_stats

        return stats

    def log_to_tensorboard(
        self,
        writer,
        global_step: int,
        prefix: str = "diagnostics/attention",
    ) -> None:
        """
        Log statistics to TensorBoard.

        CRITICAL FIX: Handles None writer gracefully.

        Args:
            writer: TensorBoard SummaryWriter (can be None)
            global_step: Current training step
            prefix: Scalar name prefix
        """
        # CRITICAL FIX: Return early if no writer
        if writer is None:
            return

        stats = self.get_statistics()

        for layer_name, layer_stats in stats.items():
            for metric_name, value in layer_stats.items():
                writer.add_scalar(
                    f"{prefix}/{layer_name}/{metric_name}",
                    value,
                    global_step,
                )

    def get_diagnostics_dict(self) -> Dict:
        """
        Get diagnostics as dictionary (no TensorBoard dependency).

        For console logging during evaluation before TensorBoard init.

        Returns:
            {'statistics': {...}}
        """
        return {
            'statistics': self.get_statistics(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        stats = self.get_statistics()

        lines = ["=" * 60, "Attention Analyzer Summary", "=" * 60, ""]

        lines.append(f"{'Layer':<40} {'Entropy':<12} {'Mean Dist':<12}")
        lines.append("-" * 60)

        for layer_name, layer_stats in sorted(stats.items()):
            entropy = layer_stats.get('entropy', 0.0)
            mean_dist = layer_stats.get('mean_distance', 0.0)
            lines.append(f"{layer_name:<40} {entropy:<12.4f} {mean_dist:<12.4f}")

        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Entropy: Higher = more uniform attention (diffuse)")
        lines.append("  Mean Distance: Average span of attention window")
        lines.append("=" * 60)

        return "\n".join(lines)
