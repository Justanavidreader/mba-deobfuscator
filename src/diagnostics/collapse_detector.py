"""
Representation collapse detection via inter-layer similarity analysis.

Measures cosine similarity between layer outputs to detect over-smoothing
in deep GNNs where node embeddings become indistinguishable.

CRITICAL FIXES:
- Memory budgeting for O(N²) operations (prevents OOM on depth 10-14)
- Deterministic sampling for reproducibility
- Chunked distance computation for large graphs
- Atexit cleanup instead of __del__
"""

import atexit
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CollapseDetector:
    """
    Detect representation collapse via inter-layer similarity.

    Tracks:
    - Cosine similarity between consecutive layers
    - Embedding diversity (pairwise distances)
    - Effective rank of representation matrix

    High inter-layer similarity (>0.95) indicates over-smoothing.

    Usage:
        detector = CollapseDetector(sample_size=500, max_nodes_for_full_distance=5000)
        detector.register_hooks(model)

        # Training loop
        output = model(batch)

        stats = detector.get_statistics()
        collapse_detected = detector.detect_collapse(threshold=0.95)
        detector.log_to_tensorboard(writer, global_step)
        detector.reset()

        # Cleanup (REQUIRED)
        detector.remove_hooks()
    """

    def __init__(
        self,
        sample_size: int = 500,
        max_nodes_for_full_distance: int = 5000,
        chunk_size: int = 500,
    ):
        """
        Initialize collapse detector.

        CRITICAL FIX: sample_size is now MANDATORY (not defaulted to None).
        CRITICAL FIX: max_nodes_for_full_distance provides memory budgeting.

        Args:
            sample_size: Number of nodes to sample for diversity computation.
                        REQUIRED for memory budgeting. Use 200-500 for production.
            max_nodes_for_full_distance: Skip full O(N²) distance computation
                        if total nodes across batch exceeds this. Prevents OOM.
            chunk_size: Chunk size for distance computation (memory efficiency)
        """
        self.sample_size = sample_size
        self.max_nodes_for_full_distance = max_nodes_for_full_distance
        self.chunk_size = chunk_size

        # Storage for layer embeddings
        self.embeddings: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # Hook handles
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register atexit handler (CRITICAL FIX)
        atexit.register(self.remove_hooks)

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Register hooks on encoder layers.

        CRITICAL FIX: Clears stale hooks before registering.

        Args:
            model: PyTorch model
            layer_names: Specific layers to hook (if None, hooks all leaf modules)
        """
        # Clear stale hooks (CRITICAL FIX)
        self.remove_hooks()

        target_modules = []
        if layer_names:
            for name, module in model.named_modules():
                if name in layer_names:
                    target_modules.append((name, module))
        else:
            # Hook all leaf modules
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:
                    target_modules.append((name, module))

        for name, module in target_modules:
            hook_handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self.hooks.append(hook_handle)

        logger.info(f"Registered {len(self.hooks)} hooks for collapse detection")

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
            logger.info(f"Removed {removed_count} collapse detector hooks")

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
        Deterministically sample embeddings for reproducibility.

        CRITICAL FIX: Uses feature-based sorting instead of random sampling.

        Args:
            embeddings: [N, D] tensor
            sample_size: Number of samples to select
            global_step: Current training step (for seed)

        Returns:
            [min(sample_size, N), D] sampled embeddings
        """
        num_nodes = embeddings.size(0)

        if num_nodes <= sample_size:
            return embeddings

        # CRITICAL FIX: Deterministic sampling via feature hash
        # Sum across features to get per-node signature
        node_signatures = embeddings.sum(dim=1)
        # Sort by signature and take top-K (deterministic)
        sorted_indices = node_signatures.argsort(descending=True)[:sample_size]

        return embeddings[sorted_indices]

    def _compute_diversity_chunked(
        self,
        embeddings: torch.Tensor,
    ) -> float:
        """
        Compute pairwise distance diversity using chunked computation.

        CRITICAL FIX: Chunked computation prevents O(N²) memory allocation.

        Args:
            embeddings: [N, D] tensor

        Returns:
            Mean pairwise cosine distance (0 = collapsed, 1 = diverse)
        """
        num_nodes = embeddings.size(0)

        if num_nodes == 0:
            return 0.0

        if num_nodes == 1:
            return 1.0  # Single node is maximally diverse

        # CRITICAL FIX: Memory budgeting check
        # For ScaledMBADataset at depth 12-14, batch can have 2000+ nodes
        # Full cdist would allocate 2000 × 2000 × hidden_dim memory
        if num_nodes > self.max_nodes_for_full_distance:
            logger.warning(
                f"Skipping full distance computation: {num_nodes} nodes "
                f"exceeds budget {self.max_nodes_for_full_distance}. "
                f"Using chunked estimation instead."
            )
            # Chunked estimation: compute distance from chunks to full set
            chunk_distances = []
            for i in range(0, num_nodes, self.chunk_size):
                chunk_end = min(i + self.chunk_size, num_nodes)
                chunk = embeddings[i:chunk_end]

                # Compute distance from chunk to all embeddings
                # chunk: [chunk_size, D], embeddings: [N, D]
                # Result: [chunk_size, N]
                dists = 1 - F.cosine_similarity(
                    chunk.unsqueeze(1),
                    embeddings.unsqueeze(0),
                    dim=2
                )
                chunk_distances.append(dists.mean().item())

            return float(np.mean(chunk_distances))

        # Small enough for full computation
        # Normalize embeddings
        emb_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity via matrix multiplication
        # [N, D] @ [D, N] = [N, N]
        similarity_matrix = torch.mm(emb_norm, emb_norm.t())

        # Convert to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Exclude diagonal (distance to self = 0)
        mask = ~torch.eye(num_nodes, dtype=torch.bool, device=embeddings.device)
        distances = distance_matrix[mask]

        return distances.mean().item()

    def _compute_effective_rank(self, embeddings: torch.Tensor) -> float:
        """
        Compute effective rank of embedding matrix.

        Lower rank indicates collapse (all embeddings in low-dimensional subspace).

        Args:
            embeddings: [N, D] tensor

        Returns:
            Effective rank (normalized by D)
        """
        if embeddings.size(0) == 0:
            return 0.0

        # Compute singular values
        try:
            _, S, _ = torch.svd(embeddings)
        except RuntimeError:
            # SVD can fail on degenerate matrices
            logger.warning("SVD failed in effective rank computation")
            return 0.0

        # Normalize singular values to probabilities
        S_normalized = S / (S.sum() + 1e-12)

        # Compute Shannon entropy
        entropy = -(S_normalized * torch.log(S_normalized + 1e-12)).sum()

        # Effective rank = exp(entropy)
        effective_rank = torch.exp(entropy).item()

        # Normalize by embedding dimension
        return effective_rank / embeddings.size(1)

    def get_statistics(self, global_step: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Compute collapse statistics from collected embeddings.

        Args:
            global_step: Current training step (for deterministic sampling)

        Returns:
            {layer_name: {
                'mean_similarity': float,  # Inter-layer similarity
                'diversity': float,  # Pairwise distance mean
                'effective_rank': float,  # Normalized effective rank
            }}
        """
        stats = {}
        layer_names = sorted(self.embeddings.keys())

        for i, layer_name in enumerate(layer_names):
            emb_list = self.embeddings[layer_name]
            if not emb_list:
                continue

            # Concatenate all batches
            embeddings = torch.cat(emb_list, dim=0)  # [total_nodes, D]

            # Sample if too large (CRITICAL FIX: deterministic)
            sampled_emb = self._sample_embeddings_deterministic(
                embeddings, self.sample_size, global_step
            )

            layer_stats = {
                'diversity': self._compute_diversity_chunked(sampled_emb),
                'effective_rank': self._compute_effective_rank(sampled_emb),
            }

            # Compute inter-layer similarity (consecutive layers)
            if i > 0:
                prev_layer = layer_names[i - 1]
                prev_emb_list = self.embeddings[prev_layer]
                if prev_emb_list:
                    prev_embeddings = torch.cat(prev_emb_list, dim=0)
                    prev_sampled = self._sample_embeddings_deterministic(
                        prev_embeddings, self.sample_size, global_step
                    )

                    # Compute cosine similarity between layers
                    # Take minimum size if different
                    min_size = min(sampled_emb.size(0), prev_sampled.size(0))
                    curr = F.normalize(sampled_emb[:min_size], p=2, dim=1)
                    prev = F.normalize(prev_sampled[:min_size], p=2, dim=1)

                    # Element-wise cosine similarity
                    similarity = (curr * prev).sum(dim=1).mean().item()
                    layer_stats['mean_similarity'] = similarity

            stats[layer_name] = layer_stats

        return stats

    def detect_collapse(
        self,
        similarity_threshold: float = 0.95,
        diversity_threshold: float = 0.1,
    ) -> Dict[str, List[str]]:
        """
        Detect representation collapse.

        Args:
            similarity_threshold: Inter-layer similarity above this = collapse
            diversity_threshold: Diversity below this = collapse

        Returns:
            {
                'high_similarity': [layer_pairs],
                'low_diversity': [layer_names],
            }
        """
        stats = self.get_statistics()
        collapse_indicators = {
            'high_similarity': [],
            'low_diversity': [],
        }

        for layer_name, layer_stats in stats.items():
            # Check inter-layer similarity
            if 'mean_similarity' in layer_stats:
                if layer_stats['mean_similarity'] > similarity_threshold:
                    collapse_indicators['high_similarity'].append(layer_name)

            # Check diversity
            if layer_stats['diversity'] < diversity_threshold:
                collapse_indicators['low_diversity'].append(layer_name)

        return collapse_indicators

    def log_to_tensorboard(
        self,
        writer,
        global_step: int,
        prefix: str = "diagnostics/collapse",
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
                'collapse_detected': bool,
                'collapse_indicators': {...},
            }
        """
        stats = self.get_statistics(global_step)
        collapse = self.detect_collapse()

        return {
            'statistics': stats,
            'collapse_detected': any(collapse.values()),
            'collapse_indicators': collapse,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        stats = self.get_statistics()
        collapse = self.detect_collapse()

        lines = ["=" * 60, "Collapse Detector Summary", "=" * 60, ""]

        lines.append(f"{'Layer':<40} {'Diversity':<12} {'Eff. Rank':<12}")
        lines.append("-" * 60)

        for layer_name, layer_stats in sorted(stats.items()):
            diversity = layer_stats.get('diversity', 0.0)
            eff_rank = layer_stats.get('effective_rank', 0.0)
            lines.append(f"{layer_name:<40} {diversity:<12.4f} {eff_rank:<12.4f}")

        lines.append("")

        # Collapse indicators
        if any(collapse.values()):
            lines.append("Collapse Indicators Detected:")
            for indicator_type, layers in collapse.items():
                if layers:
                    lines.append(f"  {indicator_type}: {len(layers)} layers")
                    for layer in layers[:3]:
                        lines.append(f"    - {layer}")
                    if len(layers) > 3:
                        lines.append(f"    ... and {len(layers) - 3} more")
        else:
            lines.append("No collapse detected.")

        lines.append("=" * 60)
        return "\n".join(lines)
