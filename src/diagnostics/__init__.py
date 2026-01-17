"""
Diagnostic tools for model debugging and analysis.

Provides tools to detect and visualize common GNN/Transformer pathologies:
- Layer activation monitoring (gradient vanishing/explosion)
- Representation collapse detection (over-smoothing)
- Attention pattern visualization
- Node embedding diversity metrics (over-squashing)

Usage:
    from src.diagnostics import ActivationMonitor, CollapseDetector

    monitor = ActivationMonitor()
    monitor.register_hooks(model)

    # Training loop
    output = model(batch)
    loss.backward()

    stats = monitor.get_statistics()
    monitor.log_to_tensorboard(writer, global_step)
    monitor.reset()
"""

from src.diagnostics.activation_monitor import ActivationMonitor
from src.diagnostics.collapse_detector import CollapseDetector
from src.diagnostics.attention_analyzer import AttentionAnalyzer
from src.diagnostics.diversity_metrics import DiversityMetrics

__all__ = [
    "ActivationMonitor",
    "CollapseDetector",
    "AttentionAnalyzer",
    "DiversityMetrics",
]
