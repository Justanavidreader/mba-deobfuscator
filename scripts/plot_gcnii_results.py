#!/usr/bin/env python3
"""
Plot GCNII ablation study results.

Creates visualizations comparing baseline HGT vs GCNII-HGT:
1. Accuracy by depth bucket (bar chart)
2. Training curves from TensorBoard logs (line plot)
3. Statistical comparison with error bars (bar chart)

Usage:
    python scripts/plot_gcnii_results.py --results results/gcnii_ablation.json
    python scripts/plot_gcnii_results.py --tensorboard logs/gcnii_ablation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_depth_bucket_comparison(
    baseline_results: Dict,
    gcnii_results: Dict,
    output_path: Optional[str] = None,
):
    """Plot accuracy comparison across depth buckets."""
    buckets = sorted(baseline_results.keys(), key=lambda x: int(x.split('-')[0]))

    baseline_acc = [baseline_results[b]['exact_match'] for b in buckets]
    gcnii_acc = [gcnii_results[b]['exact_match'] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline HGT', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, gcnii_acc, width, label='GCNII-HGT', color='orange', alpha=0.8)

    # Add improvement annotations
    for i, (b_acc, g_acc) in enumerate(zip(baseline_acc, gcnii_acc)):
        improvement = g_acc - b_acc
        y_pos = max(b_acc, g_acc) + 0.02
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, y_pos, f'+{improvement:.3f}', ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Depth Bucket', fontsize=12)
    ax.set_ylabel('Exact Match Accuracy', fontsize=12)
    ax.set_title('GCNII Over-Smoothing Mitigation: Accuracy by Depth', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_aggregate_comparison(
    baseline_agg: Dict,
    gcnii_agg: Dict,
    output_path: Optional[str] = None,
):
    """Plot aggregate results with error bars (mean ± std)."""
    buckets = sorted(baseline_agg.keys(), key=lambda x: int(x.split('-')[0]))

    baseline_means = [baseline_agg[b]['exact_match']['mean'] for b in buckets]
    baseline_stds = [baseline_agg[b]['exact_match']['std'] for b in buckets]
    gcnii_means = [gcnii_agg[b]['exact_match']['mean'] for b in buckets]
    gcnii_stds = [gcnii_agg[b]['exact_match']['std'] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline HGT', color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, gcnii_means, width, yerr=gcnii_stds,
                   label='GCNII-HGT', color='orange', alpha=0.8, capsize=5)

    # Add improvement annotations
    for i, (b_mean, g_mean) in enumerate(zip(baseline_means, gcnii_means)):
        improvement = g_mean - b_mean
        y_pos = max(b_mean, g_mean) + max(baseline_stds[i], gcnii_stds[i]) + 0.02
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, y_pos, f'+{improvement:.3f}', ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Depth Bucket', fontsize=12)
    ax.set_ylabel('Exact Match Accuracy (mean ± std)', fontsize=12)
    ax.set_title('GCNII Ablation Study: Aggregate Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_improvement_heatmap(
    baseline_results: Dict,
    gcnii_results: Dict,
    output_path: Optional[str] = None,
):
    """Plot improvement as a heatmap-style bar chart."""
    buckets = sorted(baseline_results.keys(), key=lambda x: int(x.split('-')[0]))
    improvements = [
        gcnii_results[b]['exact_match'] - baseline_results[b]['exact_match']
        for b in buckets
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(buckets, improvements, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.4f}', ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Depth Bucket', fontsize=12)
    ax.set_ylabel('Accuracy Improvement (GCNII - Baseline)', fontsize=12)
    ax.set_title('GCNII Impact: Per-Depth Accuracy Gain', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves_from_results(
    baseline_history: Dict,
    gcnii_history: Dict,
    output_path: Optional[str] = None,
):
    """Plot training curves from history dict."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract training loss
    baseline_train = baseline_history.get('train', [])
    gcnii_train = gcnii_history.get('train', [])

    if baseline_train and gcnii_train:
        baseline_epochs = list(range(1, len(baseline_train) + 1))
        gcnii_epochs = list(range(1, len(gcnii_train) + 1))

        baseline_loss = [t.get('total', 0) for t in baseline_train]
        gcnii_loss = [t.get('total', 0) for t in gcnii_train]

        ax1.plot(baseline_epochs, baseline_loss, label='Baseline HGT', color='steelblue', linewidth=2)
        ax1.plot(gcnii_epochs, gcnii_loss, label='GCNII-HGT', color='orange', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Training Loss', fontsize=11)
        ax1.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

    # Extract validation accuracy
    baseline_val = baseline_history.get('val', [])
    gcnii_val = gcnii_history.get('val', [])

    if baseline_val and gcnii_val:
        baseline_val_epochs = list(range(1, len(baseline_val) + 1))
        gcnii_val_epochs = list(range(1, len(gcnii_val) + 1))

        baseline_acc = [v.get('exact_match', 0) for v in baseline_val]
        gcnii_acc = [v.get('exact_match', 0) for v in gcnii_val]

        ax2.plot(baseline_val_epochs, baseline_acc, label='Baseline HGT', color='steelblue', linewidth=2, marker='o')
        ax2.plot(gcnii_val_epochs, gcnii_acc, label='GCNII-HGT', color='orange', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Accuracy', fontsize=11)
        ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot GCNII ablation results')
    parser.add_argument(
        '--results', type=str, required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show plots instead of saving'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    with open(args.results, 'r') as f:
        results = json.load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check result format
    if 'aggregate' in results:
        # Multi-trial results
        print("Detected multi-trial results")
        baseline_agg = results['aggregate']['baseline']
        gcnii_agg = results['aggregate']['gcnii']

        # Plot aggregate comparison
        output_path = None if args.show else str(output_dir / 'aggregate_comparison.png')
        plot_aggregate_comparison(baseline_agg, gcnii_agg, output_path)

        # Use first trial for detailed plots
        baseline_results = results['all_results']['baseline'][0]['test_results']
        gcnii_results = results['all_results']['gcnii'][0]['test_results']
        baseline_history = results['all_results']['baseline'][0]['metrics'].get('history', {})
        gcnii_history = results['all_results']['gcnii'][0]['metrics'].get('history', {})

    elif 'baseline_results' in results and 'gcnii_results' in results:
        # Single comparison
        print("Detected single comparison results")
        baseline_results = results['baseline_results']
        gcnii_results = results['gcnii_results']
        baseline_history = {}
        gcnii_history = {}

    else:
        # Single model results
        print("Detected single model results")
        if 'mode' in results:
            mode = results['mode']
            test_results = results['test_results']
            print(f"Mode: {mode}")
            print("Cannot create comparison plots with single model.")
            print("Run both baseline and GCNII, then use --mode evaluate")
            return

    # Plot depth bucket comparison
    output_path = None if args.show else str(output_dir / 'depth_comparison.png')
    plot_depth_bucket_comparison(baseline_results, gcnii_results, output_path)

    # Plot improvement heatmap
    output_path = None if args.show else str(output_dir / 'improvement_heatmap.png')
    plot_improvement_heatmap(baseline_results, gcnii_results, output_path)

    # Plot training curves if available
    if baseline_history and gcnii_history:
        output_path = None if args.show else str(output_dir / 'training_curves.png')
        plot_training_curves_from_results(baseline_history, gcnii_history, output_path)

    print(f"\n✓ Plots generated successfully")
    if not args.show:
        print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
