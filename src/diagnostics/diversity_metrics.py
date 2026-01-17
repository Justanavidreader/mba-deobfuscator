"""
Node embedding diversity metrics for over-squashing detection.

Tracks how node embedding diversity changes across GNN layers.
Over-squashing causes embeddings to become similar (low diversity).

CRITICAL FIXES:
- Memory budgeting for O(N²) pairwise distance computation
- Deterministic sampling for reproducibility
- Atexit cleanup instead of __del__
- Missing nn import (CRITICAL FIX)
"""

import atexit
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn  # CRITICAL FIX: Missing import
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiversityMetrics:
    """
    Measure node embedding diversity to detect over-squashing.

    Over-squashing occurs when information from distant nodes is compressed
    into fixed-size embeddings, causing embeddings to become similar.

    Tracks:
    - Pairwise embedding distances (mean and variance)
    - Per-dimension variance across nodes
    - Diversity degradation across layers

    Usage:
        metrics = DiversityMetrics(sample_size=500, max_nodes_for_full_distance=5000)
        metrics.register_hooks(model)

        # Forward pass
        output = model(batch)

        stats = metrics.get_statistics()
        squashing_detected = metrics.detect_over_squashing()
        metrics.log_to_tensorboard(writer, global_step)
        metrics.reset()

        # Cleanup (REQUIRED)
        metrics.remove_hooks()
    """

    def __init__(
        self,
        sample_size: int = 500,
        max_nodes_for_full_distance: int = 5000,
        chunk_size: int = 500,
    ):
        """
        Initialize diversity metrics.

        CRITICAL FIX: Memory budgeting parameters to prevent OOM.

        Args:
            sample_size: Number of nodes to sample for pairwise distance.
                        REQUIRED for memory budgeting.
            max_nodes_for_full_distance: Skip full O(N²) computation if
                        total nodes exceeds this. Prevents OOM.
            chunk_size: Chunk size for distance computation
        """
        self.sample_size = sample_size
        self.max_nodes_for_full_distance = max_nodes_for_full_distance
        self.chunk_size = chunk_size

        # Storage for embeddings
        self.embeddings: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # Hook handles
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register atexit handler (CRITICAL FIX)
        atexit.register(self.remove_hooks)

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Register hooks on GNN layers.

        CRITICAL FIX: Clears stale hooks before registering.

        Args:
            model: PyTorch model
            layer_names: Specific layers to hook (if None, hooks all)
        """
        # Clear stale hooks (CRITICAL FIX)
        self.remove_hooks()

        target_modules = []
        if layer_names:
            for name, module in model.named_modules():
                if name in layer_names:
                    target_modules.append((name, module))
        else:
            # Hook all leaf modules in encoder
            for name, module in model.named_modules():
                # Look for GNN-like modules
                module_type = type(module).__name__
                if any(gnn_type in module_type for gnn_type in
                       ['Conv', 'GAT', 'GCN', 'GGNN', 'HGT', 'RGCN']):
                    target_modules.append((name, module))

        for name, module in target_modules:
            hook_handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self.hooks.append(hook_handle)

        logger.info(f"Registered {len(self.hooks)} hooks for diversity metrics")

    def _make_forward_hook(self, layer_name: str):
        """Create forward hook closure."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.embeddings[layer_name].append(output.detach())
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
            logger.info(f"Removed {removed_count} diversity metrics hooks")

    def reset(self) -> None:
        """Clear collected embeddings."""
        self.embeddings.clear()

    def _sample_embeddings_deterministic(
        self,
        embeddings: torch.Tensor,
        sample_size: int,
        global_step: int = 0,
    ) -> torch.Tensor:
        """
        Deterministically sample embeddings.

        CRITICAL FIX: Uses feature-based sorting for determinism.

        Args:
            embeddings: [N, D] tensor
            sample_size: Number of samples
            global_step: Training step (for seed)

        Returns:
            [min(sample_size, N), D] sampled embeddings
        """
        num_nodes = embeddings.size(0)

        if num_nodes <= sample_size:
            return embeddings

        # CRITICAL FIX: Deterministic sampling via feature hash
        node_signatures = embeddings.sum(dim=1)
        sorted_indices = node_signatures.argsort(descending=True)[:sample_size]

        return embeddings[sorted_indices]

    def compute_pairwise_distance(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute pairwise distance statistics.

        CRITICAL FIX: Memory budgeting with chunked computation.

        Args:
            embeddings: [N, D] tensor

        Returns:
            (mean_distance, variance_distance)
        """
        num_nodes = embeddings.size(0)

        if num_nodes <= 1:
            return 0.0, 0.0

        # CRITICAL FIX: Memory budgeting check
        if num_nodes > self.max_nodes_for_full_distance:
            logger.warning(
                f"Skipping full pairwise distance: {num_nodes} nodes "
                f"exceeds budget {self.max_nodes_for_full_distance}. "
                f"Using chunked estimation."
            )

            # Chunked estimation
            chunk_means = []
            chunk_vars = []

            for i in range(0, num_nodes, self.chunk_size):
                chunk_end = min(i + self.chunk_size, num_nodes)
                chunk = embeddings[i:chunk_end]

                # Euclidean distance from chunk to all nodes
                # chunk: [chunk_size, D], embeddings: [N, D]
                # Compute ||chunk - embeddings||^2
                dists = torch.cdist(chunk, embeddings, p=2)  # [chunk_size, N]

                chunk_means.append(dists.mean().item())
                chunk_vars.append(dists.var().item())

            return float(np.mean(chunk_means)), float(np.mean(chunk_vars))

        # Small enough for full computation
        # Compute pairwise Euclidean distances
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)  # [N, N]

        # Exclude diagonal (distance to self = 0)
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=embeddings.device)
        distances = distance_matrix[mask]

        return distances.mean().item(), distances.var().item()

    def compute_dimension_variance(self, embeddings: torch.Tensor) -> float:
        """
        Compute mean variance across embedding dimensions.

        Low variance indicates embeddings collapse to similar values.

        Args:
            embeddings: [N, D] tensor

        Returns:
            Mean variance across dimensions
        """
        if embeddings.size(0) <= 1:
            return 0.0

        # Variance per dimension [D]
        dim_variance = embeddings.var(dim=0)

        return dim_variance.mean().item()

    def get_statistics(self, global_step: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Compute diversity statistics from collected embeddings.

        Args:
            global_step: Current training step (for deterministic sampling)

        Returns:
            {layer_name: {
                'mean_distance': float,  # Mean pairwise distance
                'variance_distance': float,  # Variance of pairwise distances
                'dimension_variance': float,  # Variance per dimension
            }}
        """
        stats = {}

        for layer_name, emb_list in self.embeddings.items():
            if not emb_list:
                continue

            # Concatenate all batches
            embeddings = torch.cat(emb_list, dim=0)  # [total_nodes, D]

            # Sample if too large (CRITICAL FIX: deterministic)
            sampled_emb = self._sample_embeddings_deterministic(
                embeddings, self.sample_size, global_step
            )

            # Compute diversity metrics
            mean_dist, var_dist = self.compute_pairwise_distance(sampled_emb)
            dim_var = self.compute_dimension_variance(sampled_emb)

            stats[layer_name] = {
                'mean_distance': mean_dist,
                'variance_distance': var_dist,
                'dimension_variance': dim_var,
            }

        return stats

    def detect_over_squashing(
        self,
        distance_threshold: float = 0.5,
        variance_threshold: float = 0.01,
    ) -> Dict[str, List[str]]:
        """
        Detect over-squashing indicators.

        Args:
            distance_threshold: Mean distance below this = squashing
            variance_threshold: Dimension variance below this = squashing

        Returns:
            {
                'low_distance': [layer_names],
                'low_variance': [layer_names],
            }
        """
        stats = self.get_statistics()
        indicators = {
            'low_distance': [],
            'low_variance': [],
        }

        for layer_name, layer_stats in stats.items():
            if layer_stats['mean_distance'] < distance_threshold:
                indicators['low_distance'].append(layer_name)

            if layer_stats['dimension_variance'] < variance_threshold:
                indicators['low_variance'].append(layer_name)

        return indicators

    def log_to_tensorboard(
        self,
        writer,
        global_step: int,
        prefix: str = "diagnostics/diversity",
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

        stats = self.get_statistics(global_step)

        for layer_name, layer_stats in stats.items():
            for metric_name, value in layer_stats.items():
                writer.add_scalar(
                    f"{prefix}/{layer_name}/{metric_name}",
                    value,
                    global_step,
                )

    def get_diagnostics_dict(self, global_step: int = 0) -> Dict:
        """
        Get diagnostics as dictionary (no TensorBoard dependency).

        For console logging during evaluation before TensorBoard init.

        Returns:
            {
                'statistics': {...},
                'over_squashing_detected': bool,
                'indicators': {...},
            }
        """
        stats = self.get_statistics(global_step)
        indicators = self.detect_over_squashing()

        return {
            'statistics': stats,
            'over_squashing_detected': any(indicators.values()),
            'indicators': indicators,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        stats = self.get_statistics()
        indicators = self.detect_over_squashing()

        lines = ["=" * 60, "Diversity Metrics Summary", "=" * 60, ""]

        lines.append(f"{'Layer':<40} {'Mean Dist':<12} {'Dim Var':<12}")
        lines.append("-" * 60)

        for layer_name, layer_stats in sorted(stats.items()):
            mean_dist = layer_stats.get('mean_distance', 0.0)
            dim_var = layer_stats.get('dimension_variance', 0.0)
            lines.append(f"{layer_name:<40} {mean_dist:<12.4f} {dim_var:<12.4f}")

        lines.append("")

        # Over-squashing indicators
        if any(indicators.values()):
            lines.append("Over-Squashing Indicators Detected:")
            for indicator_type, layers in indicators.items():
                if layers:
                    lines.append(f"  {indicator_type}: {len(layers)} layers")
                    for layer in layers[:3]:
                        lines.append(f"    - {layer}")
                    if len(layers) > 3:
                        lines.append(f"    ... and {len(layers) - 3} more")
        else:
            lines.append("No over-squashing detected.")

        lines.append("=" * 60)
        return "\n".join(lines)
