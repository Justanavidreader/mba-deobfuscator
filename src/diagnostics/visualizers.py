"""
Visualization utilities for diagnostic data.

Provides matplotlib-based plotting for:
- Activation/gradient norms across layers
- Representation collapse heatmaps
- Attention pattern visualizations
- Diversity metrics trends

Dependencies: matplotlib, seaborn (optional for better styling)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Optional seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.info("Seaborn not available. Using matplotlib defaults.")


def plot_activation_norms(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Layer Activation Norms",
) -> None:
    """
    Plot activation and gradient norms across layers.

    Args:
        stats: Output from ActivationMonitor.get_statistics()
        output_path: Save path (if None, displays plot)
        title: Plot title
    """
    layers = sorted(stats.keys())
    activation_norms = [stats[layer].get('activation_norm', 0.0) for layer in layers]
    gradient_norms = [stats[layer].get('gradient_norm', 0.0) for layer in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Activation norms
    ax1.bar(range(len(layers)), activation_norms, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Activation Norm')
    ax1.set_title(f'{title} - Activations')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Gradient norms
    if any(g > 0 for g in gradient_norms):
        ax2.bar(range(len(layers)), gradient_norms, color='orange', alpha=0.7)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title(f'{title} - Gradients')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # Add thresholds
        ax2.axhline(y=1e-4, color='red', linestyle='--', label='Vanishing threshold')
        ax2.axhline(y=100, color='red', linestyle='--', label='Exploding threshold')
        ax2.legend()
        ax2.set_yscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_collapse_metrics(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Representation Collapse Metrics",
) -> None:
    """
    Plot collapse detection metrics.

    Args:
        stats: Output from CollapseDetector.get_statistics()
        output_path: Save path (if None, displays plot)
        title: Plot title
    """
    layers = sorted(stats.keys())
    diversity = [stats[layer].get('diversity', 0.0) for layer in layers]
    effective_rank = [stats[layer].get('effective_rank', 0.0) for layer in layers]
    similarity = [stats[layer].get('mean_similarity', 0.0) for layer in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Diversity and effective rank
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(range(len(layers)), diversity, 'o-', color='steelblue',
                     label='Diversity', linewidth=2, markersize=6)
    line2 = ax1_twin.plot(range(len(layers)), effective_rank, 's-', color='orange',
                          label='Effective Rank', linewidth=2, markersize=6)

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Diversity (pairwise distance)', color='steelblue')
    ax1_twin.set_ylabel('Effective Rank (normalized)', color='orange')
    ax1.set_title(f'{title} - Diversity Metrics')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Add threshold for diversity
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ['Collapse threshold']
    ax1.legend(lines + [ax1.lines[-1]], labels, loc='upper left')

    # Inter-layer similarity
    if similarity and any(s > 0 for s in similarity):
        ax2.plot(range(len(layers)), similarity, 'o-', color='crimson',
                linewidth=2, markersize=6)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Inter-Layer Cosine Similarity')
        ax2.set_title(f'{title} - Layer Similarity')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='red', linestyle='--', label='Collapse threshold')
        ax2.legend()
        ax2.set_ylim([0, 1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_diversity_metrics(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Node Embedding Diversity",
) -> None:
    """
    Plot diversity metrics (over-squashing detection).

    Args:
        stats: Output from DiversityMetrics.get_statistics()
        output_path: Save path (if None, displays plot)
        title: Plot title
    """
    layers = sorted(stats.keys())
    mean_distance = [stats[layer].get('mean_distance', 0.0) for layer in layers]
    dimension_variance = [stats[layer].get('dimension_variance', 0.0) for layer in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Mean pairwise distance
    ax1.plot(range(len(layers)), mean_distance, 'o-', color='steelblue',
            linewidth=2, markersize=6)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Pairwise Distance')
    ax1.set_title(f'{title} - Pairwise Distance')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Squashing threshold')
    ax1.legend()

    # Dimension variance
    ax2.plot(range(len(layers)), dimension_variance, 'o-', color='orange',
            linewidth=2, markersize=6)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean Dimension Variance')
    ax2.set_title(f'{title} - Dimension Variance')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Squashing threshold')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_patterns(
    stats: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Attention Pattern Analysis",
) -> None:
    """
    Plot attention statistics.

    Args:
        stats: Output from AttentionAnalyzer.get_statistics()
        output_path: Save path (if None, displays plot)
        title: Plot title
    """
    layers = sorted(stats.keys())
    entropy = [stats[layer].get('entropy', 0.0) for layer in layers]
    mean_distance = [stats[layer].get('mean_distance', 0.0) for layer in layers]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax_twin = ax.twinx()

    # Entropy
    line1 = ax.plot(range(len(layers)), entropy, 'o-', color='steelblue',
                   label='Entropy', linewidth=2, markersize=6)

    # Mean distance
    line2 = ax_twin.plot(range(len(layers)), mean_distance, 's-', color='orange',
                        label='Mean Distance', linewidth=2, markersize=6)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Attention Entropy', color='steelblue')
    ax_twin.set_ylabel('Mean Attention Distance', color='orange')
    ax.set_title(title)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_diagnostics(
    activation_stats: Optional[Dict] = None,
    collapse_stats: Optional[Dict] = None,
    diversity_stats: Optional[Dict] = None,
    attention_stats: Optional[Dict] = None,
    output_dir: str = "diagnostics_plots",
    prefix: str = "",
) -> None:
    """
    Generate all diagnostic plots and save to directory.

    Args:
        activation_stats: ActivationMonitor statistics
        collapse_stats: CollapseDetector statistics
        diversity_stats: DiversityMetrics statistics
        attention_stats: AttentionAnalyzer statistics
        output_dir: Output directory for plots
        prefix: Filename prefix (e.g., "step_1000_")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if activation_stats:
        plot_activation_norms(
            activation_stats,
            output_path=str(output_path / f"{prefix}activation_norms.png")
        )

    if collapse_stats:
        plot_collapse_metrics(
            collapse_stats,
            output_path=str(output_path / f"{prefix}collapse_metrics.png")
        )

    if diversity_stats:
        plot_diversity_metrics(
            diversity_stats,
            output_path=str(output_path / f"{prefix}diversity_metrics.png")
        )

    if attention_stats:
        plot_attention_patterns(
            attention_stats,
            output_path=str(output_path / f"{prefix}attention_patterns.png")
        )

    logger.info(f"All diagnostic plots saved to {output_dir}")


def plot_training_trends(
    step_stats: List[Tuple[int, Dict]],
    metric_name: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot metric trends across training steps.

    Args:
        step_stats: List of (global_step, statistics_dict) tuples
        metric_name: Metric to plot (e.g., 'diversity', 'gradient_norm')
        output_path: Save path (if None, displays plot)
        title: Plot title
    """
    if not step_stats:
        logger.warning("No statistics provided for trend plotting")
        return

    # Extract steps and layer-wise metrics
    steps = [step for step, _ in step_stats]
    layer_names = sorted(step_stats[0][1].keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each layer's trend
    for layer_name in layer_names:
        values = []
        for step, stats in step_stats:
            if layer_name in stats and metric_name in stats[layer_name]:
                values.append(stats[layer_name][metric_name])
            else:
                values.append(np.nan)

        ax.plot(steps, values, 'o-', label=layer_name, alpha=0.7)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title or f'{metric_name.replace("_", " ").title()} Across Training')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trend plot to {output_path}")
    else:
        plt.show()

    plt.close()
