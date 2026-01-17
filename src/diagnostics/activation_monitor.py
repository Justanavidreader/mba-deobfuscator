"""
Layer activation and gradient monitoring for GNN/Transformer models.

Tracks per-layer statistics to detect gradient pathologies:
- Vanishing gradients: norm < 1e-4
- Exploding gradients: norm > 100
- Dead neurons: activation std < 1e-6

CRITICAL FIX: Uses atexit handlers instead of __del__ for reliable cleanup.
"""

import atexit
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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

        # Cleanup (REQUIRED at end of training)
        monitor.remove_hooks()
    """

    def __init__(self, track_gradients: bool = True, dead_neuron_threshold: float = 1e-6):
        """
        Initialize activation monitor.

        Args:
            track_gradients: If True, track gradient norms (requires backward pass)
            dead_neuron_threshold: Std threshold below which neurons considered dead
        """
        self.track_gradients = track_gradients
        self.dead_neuron_threshold = dead_neuron_threshold

        # Storage for collected data
        self.activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.gradients: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # Hook handles for cleanup
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register atexit handler for cleanup (CRITICAL FIX)
        atexit.register(self.remove_hooks)

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward and backward hooks on model layers.

        CRITICAL FIX: Calls remove_hooks() first to clear any stale registrations.

        Args:
            model: PyTorch model to monitor
        """
        # Clear stale hooks before registering new ones (CRITICAL FIX)
        self.remove_hooks()

        for name, module in model.named_modules():
            # Skip containers (Sequential, ModuleList, etc.)
            if len(list(module.children())) > 0:
                continue

            # Register on leaf modules only
            hook_handle = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self.hooks.append(hook_handle)

            if self.track_gradients:
                grad_hook_handle = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self.hooks.append(grad_hook_handle)

        logger.info(f"Registered {len(self.hooks)} hooks on {model.__class__.__name__}")

    def _make_forward_hook(self, layer_name: str):
        """Create forward hook closure for layer."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store detached copy to avoid interfering with autograd
                self.activations[layer_name].append(output.detach())
        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create backward hook closure for layer."""
        def hook(module, grad_input, grad_output):
            # grad_output is tuple of gradients w.r.t. outputs
            if grad_output[0] is not None:
                self.gradients[layer_name].append(grad_output[0].detach())
        return hook

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.

        CRITICAL FIX: Robust cleanup that handles already-removed hooks gracefully.
        """
        removed_count = 0
        for hook in self.hooks:
            try:
                hook.remove()
                removed_count += 1
            except Exception as e:
                # Hook might already be removed, log but don't fail
                logger.debug(f"Hook removal warning: {e}")

        self.hooks.clear()

        if removed_count > 0:
            logger.info(f"Removed {removed_count} activation monitor hooks")

    def reset(self) -> None:
        """Clear collected data."""
        self.activations.clear()
        self.gradients.clear()

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics from collected activations/gradients.

        Returns:
            {layer_name: {
                'activation_mean': float,
                'activation_std': float,
                'activation_norm': float,
                'gradient_norm': float,  # if track_gradients=True
                'dead_neurons_pct': float,
            }}
        """
        stats = {}

        for layer_name, act_list in self.activations.items():
            if not act_list:
                continue

            # Stack activations from all batches
            activations = torch.cat(act_list, dim=0)  # [total_samples, ...]

            layer_stats = {
                'activation_mean': activations.mean().item(),
                'activation_std': activations.std().item(),
                'activation_norm': activations.norm().item(),
            }

            # Dead neuron detection (channels with near-zero std)
            if activations.ndim >= 2:
                # Reshape to [samples, channels, ...]
                flat_act = activations.reshape(activations.size(0), activations.size(1), -1)
                channel_std = flat_act.std(dim=(0, 2))  # Std per channel
                dead_mask = channel_std < self.dead_neuron_threshold
                layer_stats['dead_neurons_pct'] = dead_mask.float().mean().item() * 100

            stats[layer_name] = layer_stats

        # Add gradient norms if tracked
        if self.track_gradients:
            for layer_name, grad_list in self.gradients.items():
                if not grad_list:
                    continue

                gradients = torch.cat(grad_list, dim=0)
                if layer_name not in stats:
                    stats[layer_name] = {}

                stats[layer_name]['gradient_norm'] = gradients.norm().item()

        return stats

    def detect_pathologies(
        self,
        vanishing_threshold: float = 1e-4,
        exploding_threshold: float = 100.0,
    ) -> Dict[str, List[str]]:
        """
        Detect gradient pathologies from statistics.

        Args:
            vanishing_threshold: Gradient norm below this indicates vanishing
            exploding_threshold: Gradient norm above this indicates exploding

        Returns:
            {
                'vanishing_gradients': [layer_name1, ...],
                'exploding_gradients': [layer_name2, ...],
                'dead_neurons': [layer_name3, ...],
            }
        """
        stats = self.get_statistics()
        pathologies = {
            'vanishing_gradients': [],
            'exploding_gradients': [],
            'dead_neurons': [],
        }

        for layer_name, layer_stats in stats.items():
            # Check gradient norms
            if 'gradient_norm' in layer_stats:
                grad_norm = layer_stats['gradient_norm']
                if grad_norm < vanishing_threshold:
                    pathologies['vanishing_gradients'].append(layer_name)
                elif grad_norm > exploding_threshold:
                    pathologies['exploding_gradients'].append(layer_name)

            # Check dead neurons
            if 'dead_neurons_pct' in layer_stats:
                if layer_stats['dead_neurons_pct'] > 10.0:  # >10% dead
                    pathologies['dead_neurons'].append(layer_name)

        return pathologies

    def log_to_tensorboard(
        self,
        writer,
        global_step: int,
        prefix: str = "diagnostics/activations",
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

    def summary(self) -> str:
        """Generate human-readable summary of statistics."""
        stats = self.get_statistics()
        pathologies = self.detect_pathologies()

        lines = ["=" * 60, "Activation Monitor Summary", "=" * 60, ""]

        # Layer-wise statistics
        lines.append(f"{'Layer':<40} {'Act Norm':<12} {'Grad Norm':<12}")
        lines.append("-" * 60)

        for layer_name, layer_stats in sorted(stats.items()):
            act_norm = layer_stats.get('activation_norm', 0.0)
            grad_norm = layer_stats.get('gradient_norm', 0.0)
            lines.append(f"{layer_name:<40} {act_norm:<12.4f} {grad_norm:<12.4f}")

        lines.append("")

        # Pathologies
        if any(pathologies.values()):
            lines.append("Detected Pathologies:")
            for pathology_type, layers in pathologies.items():
                if layers:
                    lines.append(f"  {pathology_type}: {len(layers)} layers")
                    for layer in layers[:3]:  # Show first 3
                        lines.append(f"    - {layer}")
                    if len(layers) > 3:
                        lines.append(f"    ... and {len(layers) - 3} more")
        else:
            lines.append("No pathologies detected.")

        lines.append("=" * 60)
        return "\n".join(lines)
