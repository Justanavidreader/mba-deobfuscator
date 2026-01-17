#!/usr/bin/env python3
"""
Standalone diagnostic visualization script.

Reads TensorBoard event files or saved diagnostic data and generates plots.

Usage:
    # From TensorBoard logs
    python scripts/visualize_diagnostics.py --tensorboard logs/experiment

    # From saved JSON data
    python scripts/visualize_diagnostics.py --json diagnostics_data.json

    # Generate all plots
    python scripts/visualize_diagnostics.py --tensorboard logs/experiment --output plots/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.visualizers import (
    plot_activation_norms,
    plot_attention_patterns,
    plot_collapse_metrics,
    plot_diversity_metrics,
    plot_training_trends,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_tensorboard_logs(log_dir: str, tag_prefix: str = "diagnostics") -> Dict:
    """
    Parse TensorBoard event files to extract diagnostic data.

    Args:
        log_dir: Path to TensorBoard log directory
        tag_prefix: Prefix for diagnostic tags

    Returns:
        Dictionary of parsed statistics by step
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.error("TensorBoard not installed. Install with: pip install tensorboard")
        return {}

    logger.info(f"Parsing TensorBoard logs from {log_dir}")

    ea = EventAccumulator(log_dir)
    ea.Reload()

    # Get all scalar tags
    tags = ea.Tags()['scalars']
    diagnostic_tags = [tag for tag in tags if tag.startswith(tag_prefix)]

    logger.info(f"Found {len(diagnostic_tags)} diagnostic tags")

    # Parse data by step
    data_by_step: Dict[int, Dict] = {}

    for tag in diagnostic_tags:
        events = ea.Scalars(tag)

        for event in events:
            step = event.step
            value = event.value

            # Parse tag: diagnostics/activations/layer_name/metric_name
            parts = tag.split('/')
            if len(parts) < 4:
                continue

            category = parts[1]  # activations, collapse, attention, diversity
            layer_name = parts[2]
            metric_name = parts[3]

            if step not in data_by_step:
                data_by_step[step] = {}

            if category not in data_by_step[step]:
                data_by_step[step][category] = {}

            if layer_name not in data_by_step[step][category]:
                data_by_step[step][category][layer_name] = {}

            data_by_step[step][category][layer_name][metric_name] = value

    return data_by_step


def load_json_data(json_path: str) -> Dict:
    """
    Load diagnostic data from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary of statistics
    """
    logger.info(f"Loading data from {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def generate_all_plots(
    data: Dict,
    output_dir: str,
    prefix: str = "",
) -> None:
    """
    Generate all diagnostic plots from data.

    Args:
        data: Diagnostic data dictionary
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating plots to {output_dir}")

    # If data is by-step, get latest step
    if all(isinstance(k, (int, str)) and str(k).isdigit() for k in data.keys()):
        # Data is {step: {category: {layer: {metric: value}}}}
        latest_step = max(int(k) for k in data.keys())
        latest_data = data[str(latest_step)]
        logger.info(f"Using data from step {latest_step}")
    else:
        # Data is already {category: {layer: {metric: value}}}
        latest_data = data

    # Plot each category
    if 'activations' in latest_data:
        plot_activation_norms(
            latest_data['activations'],
            output_path=str(output_path / f"{prefix}activation_norms.png"),
        )

    if 'collapse' in latest_data:
        plot_collapse_metrics(
            latest_data['collapse'],
            output_path=str(output_path / f"{prefix}collapse_metrics.png"),
        )

    if 'diversity' in latest_data:
        plot_diversity_metrics(
            latest_data['diversity'],
            output_path=str(output_path / f"{prefix}diversity_metrics.png"),
        )

    if 'attention' in latest_data:
        plot_attention_patterns(
            latest_data['attention'],
            output_path=str(output_path / f"{prefix}attention_patterns.png"),
        )

    logger.info(f"Plots saved to {output_dir}")


def generate_trend_plots(
    data_by_step: Dict[int, Dict],
    output_dir: str,
    metric_name: str,
    category: str,
) -> None:
    """
    Generate trend plots showing metric evolution across training.

    Args:
        data_by_step: Dictionary mapping step to statistics
        output_dir: Output directory
        metric_name: Metric to plot
        category: Category (activations, collapse, diversity, attention)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to list of (step, stats) tuples
    step_stats = []
    for step in sorted(data_by_step.keys()):
        if category in data_by_step[step]:
            step_stats.append((step, data_by_step[step][category]))

    if not step_stats:
        logger.warning(f"No data found for {category}/{metric_name}")
        return

    plot_training_trends(
        step_stats,
        metric_name=metric_name,
        output_path=str(output_path / f"{category}_{metric_name}_trend.png"),
        title=f"{category.title()} {metric_name.replace('_', ' ').title()} Across Training",
    )

    logger.info(f"Trend plot saved for {category}/{metric_name}")


def main():
    parser = argparse.ArgumentParser(description='Visualize diagnostic data')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--tensorboard',
        type=str,
        help='Path to TensorBoard log directory',
    )
    input_group.add_argument(
        '--json',
        type=str,
        help='Path to JSON diagnostic data file',
    )

    parser.add_argument(
        '--output',
        type=str,
        default='diagnostics_plots',
        help='Output directory for plots (default: diagnostics_plots)',
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Filename prefix for plots',
    )

    parser.add_argument(
        '--trends',
        action='store_true',
        help='Generate trend plots (requires TensorBoard data)',
    )

    parser.add_argument(
        '--category',
        type=str,
        choices=['activations', 'collapse', 'diversity', 'attention'],
        help='Specific category to plot (default: all)',
    )

    parser.add_argument(
        '--metric',
        type=str,
        help='Specific metric for trend plotting (e.g., diversity, gradient_norm)',
    )

    args = parser.parse_args()

    # Load data
    if args.tensorboard:
        data = parse_tensorboard_logs(args.tensorboard)
        if not data:
            logger.error("No data found in TensorBoard logs")
            return 1
    else:
        data = load_json_data(args.json)
        if not data:
            logger.error("No data found in JSON file")
            return 1

    # Generate plots
    if args.trends and args.tensorboard:
        # Generate trend plots
        if args.category and args.metric:
            generate_trend_plots(data, args.output, args.metric, args.category)
        else:
            logger.error("--trends requires --category and --metric")
            return 1
    else:
        # Generate snapshot plots
        generate_all_plots(data, args.output, args.prefix)

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
