"""
Statistical significance testing for ablation study.

Provides paired t-tests, effect size computation, and pairwise comparisons
for encoder architecture evaluation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# scipy is optional - provide fallback implementations
try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def paired_t_test(
    encoder_a_results: List[float],
    encoder_b_results: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Paired t-test between two encoder results.

    Args:
        encoder_a_results: Metric values from encoder A (5 runs)
        encoder_b_results: Metric values from encoder B (5 runs)
        alpha: Significance level

    Returns:
        t_statistic, p_value, is_significant

    Raises:
        ValueError: If result lists have different lengths or too few samples
    """
    if len(encoder_a_results) != len(encoder_b_results):
        raise ValueError(
            f"Result lists must have same length: "
            f"{len(encoder_a_results)} vs {len(encoder_b_results)}"
        )

    n = len(encoder_a_results)
    if n < 2:
        raise ValueError(f"Need at least 2 samples for t-test, got {n}")

    if HAS_SCIPY:
        t_stat, p_value = stats.ttest_rel(encoder_a_results, encoder_b_results)
    else:
        # Fallback implementation
        t_stat, p_value = _paired_t_test_fallback(encoder_a_results, encoder_b_results)

    is_significant = p_value < alpha
    return float(t_stat), float(p_value), is_significant


def _paired_t_test_fallback(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Paired t-test without scipy.

    Uses the formula: t = mean(d) / (std(d) / sqrt(n))
    where d = a - b
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    d = a_arr - b_arr
    n = len(d)

    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)  # Sample std

    if std_d == 0:
        return 0.0, 1.0  # No difference

    t_stat = mean_d / (std_d / np.sqrt(n))

    # Two-tailed p-value approximation using normal distribution
    # This is less accurate than t-distribution but works without scipy
    z = abs(t_stat)
    p_value = 2 * (1 - _normal_cdf(z))

    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (error < 7.5e-8)."""
    # Abramowitz and Stegun approximation 7.1.26
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def compute_effect_size(
    encoder_a_results: List[float],
    encoder_b_results: List[float],
) -> float:
    """
    Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        encoder_a_results: Metric values from encoder A
        encoder_b_results: Metric values from encoder B

    Returns:
        Cohen's d (positive means A > B)
    """
    a_arr = np.array(encoder_a_results)
    b_arr = np.array(encoder_b_results)

    mean_a = np.mean(a_arr)
    mean_b = np.mean(b_arr)
    std_a = np.std(a_arr, ddof=1)
    std_b = np.std(b_arr, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    if pooled_std == 0:
        return 0.0

    cohen_d = (mean_a - mean_b) / pooled_std
    return float(cohen_d)


def interpret_effect_size(cohen_d: float) -> str:
    """Interpret Cohen's d value."""
    d = abs(cohen_d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def pairwise_comparison(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence",
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], Dict]:
    """
    Perform pairwise statistical tests across all encoder pairs.

    Args:
        aggregated_results: Output from AblationMetricsCollector.aggregate_by_encoder()
        metric: Metric to compare (default: z3_equivalence)
        alpha: Significance level

    Returns:
        {(encoder_a, encoder_b): {
            't_statistic': float,
            'p_value': float,
            'significant': bool,
            'effect_size': float,
            'effect_interpretation': str,
            'winner': str or None
        }}
    """
    encoders = list(aggregated_results.keys())
    comparisons: Dict[Tuple[str, str], Dict] = {}

    for i, enc_a in enumerate(encoders):
        for enc_b in encoders[i + 1 :]:
            # Get run results for the metric
            results_a = aggregated_results.get(enc_a, {}).get(metric, {}).get("runs", [])
            results_b = aggregated_results.get(enc_b, {}).get(metric, {}).get("runs", [])

            if not results_a or not results_b:
                comparisons[(enc_a, enc_b)] = {
                    "t_statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "effect_size": float("nan"),
                    "effect_interpretation": "insufficient data",
                    "winner": None,
                }
                continue

            if len(results_a) != len(results_b):
                # Handle mismatched run counts
                min_len = min(len(results_a), len(results_b))
                results_a = results_a[:min_len]
                results_b = results_b[:min_len]

            try:
                t_stat, p_value, significant = paired_t_test(results_a, results_b, alpha)
                effect_size = compute_effect_size(results_a, results_b)

                # Determine winner (higher is better for accuracy metrics)
                winner = None
                if significant:
                    mean_a = np.mean(results_a)
                    mean_b = np.mean(results_b)
                    winner = enc_a if mean_a > mean_b else enc_b

                comparisons[(enc_a, enc_b)] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": significant,
                    "effect_size": effect_size,
                    "effect_interpretation": interpret_effect_size(effect_size),
                    "winner": winner,
                }
            except ValueError as e:
                comparisons[(enc_a, enc_b)] = {
                    "t_statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "effect_size": float("nan"),
                    "effect_interpretation": f"error: {str(e)}",
                    "winner": None,
                }

    return comparisons


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from pairwise tests
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating significance after correction
    """
    n_comparisons = len(p_values)
    adjusted_alpha = alpha / n_comparisons

    return [p < adjusted_alpha for p in p_values]


def rank_encoders(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence",
) -> List[Tuple[str, float, float]]:
    """
    Rank encoders by mean performance on a metric.

    Args:
        aggregated_results: Output from AblationMetricsCollector.aggregate_by_encoder()
        metric: Metric to rank by

    Returns:
        List of (encoder_name, mean, std) sorted by mean (descending)
    """
    rankings: List[Tuple[str, float, float]] = []

    for encoder, metrics in aggregated_results.items():
        if metric in metrics:
            mean = metrics[metric].get("mean", 0.0)
            std = metrics[metric].get("std", 0.0)
            rankings.append((encoder, mean, std))

    # Sort by mean, descending
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def generate_comparison_report(
    aggregated_results: Dict[str, Dict],
    metric: str = "z3_equivalence",
    alpha: float = 0.05,
) -> str:
    """
    Generate a text report of pairwise comparisons.

    Args:
        aggregated_results: Output from AblationMetricsCollector.aggregate_by_encoder()
        metric: Metric to compare
        alpha: Significance level

    Returns:
        Formatted text report
    """
    lines = [
        f"Pairwise Comparison Report: {metric}",
        f"Significance level: {alpha}",
        "=" * 70,
        "",
    ]

    # Rankings
    rankings = rank_encoders(aggregated_results, metric)
    lines.append("Rankings (by mean):")
    lines.append("-" * 40)
    for rank, (encoder, mean, std) in enumerate(rankings, 1):
        lines.append(f"  {rank}. {encoder}: {mean:.4f} +/- {std:.4f}")
    lines.append("")

    # Pairwise comparisons
    comparisons = pairwise_comparison(aggregated_results, metric, alpha)

    lines.append("Pairwise Comparisons:")
    lines.append("-" * 70)
    lines.append(f"{'Pair':<40} {'p-value':<10} {'Effect':<10} {'Winner':<15}")
    lines.append("-" * 70)

    for (enc_a, enc_b), result in sorted(comparisons.items()):
        pair = f"{enc_a} vs {enc_b}"
        p_val = result["p_value"]
        effect = result["effect_interpretation"]
        winner = result["winner"] or "n.s."

        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        sig_marker = "*" if result["significant"] else ""

        lines.append(f"  {pair:<38} {p_str:<10}{sig_marker} {effect:<10} {winner:<15}")

    lines.append("")
    lines.append("* = statistically significant at alpha={:.2f}".format(alpha))

    return "\n".join(lines)
