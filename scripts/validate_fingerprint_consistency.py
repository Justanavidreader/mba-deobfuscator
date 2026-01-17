#!/usr/bin/env python3
"""
Validate that Python fingerprint computation matches C++ generator output.

Usage:
    python scripts/validate_fingerprint_consistency.py dataset.json --samples 100
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Direct imports to avoid __init__.py chains that pull in torch_scatter
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, '..')
sys.path.insert(0, repo_root)

from src.constants import (
    FINGERPRINT_DIM, SYMBOLIC_DIM, CORNER_DIM,
    RANDOM_DIM, DERIVATIVE_DIM, TRUTH_TABLE_DIM
)
from src.data.fingerprint import SemanticFingerprint


def get_fingerprint_components() -> List[Tuple[str, int, int]]:
    """
    Compute fingerprint component boundaries from constants.

    Returns:
        List of (name, start_idx, end_idx) tuples
    """
    sym_end = SYMBOLIC_DIM
    corner_end = sym_end + CORNER_DIM
    random_end = corner_end + RANDOM_DIM
    deriv_end = random_end + DERIVATIVE_DIM
    truth_end = deriv_end + TRUTH_TABLE_DIM

    assert truth_end == FINGERPRINT_DIM, f"Component dims don't sum to {FINGERPRINT_DIM}"

    return [
        ('symbolic', 0, sym_end),
        ('corner', sym_end, corner_end),
        ('random', corner_end, random_end),
        ('derivative', random_end, deriv_end),
        ('truth_table', deriv_end, truth_end),
    ]


def validate_sample(
    expr: str,
    cpp_fp: np.ndarray,
    fingerprint: SemanticFingerprint,
    tolerance: float,
    components: List[Tuple[str, int, int]]
) -> Optional[Dict]:
    """
    Validate a single fingerprint sample.

    Args:
        expr: Expression to compute fingerprint for
        cpp_fp: Pre-computed fingerprint from C++ generator
        fingerprint: Python fingerprint computer
        tolerance: Numerical tolerance for comparison
        components: List of (name, start, end) component boundaries

    Returns:
        None if match, or dict with mismatch details if different
    """
    try:
        py_fp = fingerprint.compute(expr)
    except Exception as e:
        return {'error': str(e)}

    if not np.allclose(cpp_fp, py_fp, atol=tolerance):
        diff = np.abs(cpp_fp - py_fp)
        component_diffs = {}
        component_matches = {}

        for name, start, end in components:
            comp_cpp = cpp_fp[start:end]
            comp_py = py_fp[start:end]
            comp_diff = diff[start:end]

            max_diff = comp_diff.max()
            matches = np.sum(comp_diff <= tolerance)
            total = len(comp_diff)

            if max_diff > tolerance:
                component_diffs[name] = {
                    'max_diff': float(max_diff),
                    'matches': int(matches),
                    'total': int(total),
                    'match_rate': float(matches / total) if total > 0 else 0.0
                }
            component_matches[name] = int(matches)

        return {
            'max_diff': diff.max(),
            'components': component_diffs,
            'matches': component_matches
        }

    return None


def load_samples(filepath: Path) -> List[Dict]:
    """Load samples from JSON or JSONL file."""
    with open(filepath) as f:
        content = f.read().strip()

    # Try JSON array format first (C++ generator output)
    if content.startswith('{'):
        data = json.loads(content)
        if 'samples' in data:
            return data['samples']
        return [data]

    # JSONL format
    samples = []
    for line in content.split('\n'):
        if line.strip():
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Validate Python fingerprint consistency with C++ generator'
    )
    parser.add_argument('filepath', type=Path, help='JSON or JSONL file to validate')
    parser.add_argument('--samples', '-n', type=int, default=100, help='Max samples')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Numerical tolerance')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-sample details')
    args = parser.parse_args()

    fingerprint = SemanticFingerprint()
    components = get_fingerprint_components()

    print(f"Loading {args.filepath}...")
    samples = load_samples(args.filepath)
    print(f"Found {len(samples)} samples, validating up to {args.samples}")
    print(f"Tolerance: {args.tolerance}\n")

    mismatches = 0
    total = 0
    skipped = 0

    # Track component-level statistics
    component_names = [name for name, _, _ in components]
    component_match_counts = {name: 0 for name in component_names}
    component_total_counts = {name: 0 for name in component_names}
    component_max_diffs = {name: [] for name in component_names}

    for i, item in enumerate(samples[:args.samples]):
        # Check required fields
        if 'fingerprint' not in item or 'flat' not in item['fingerprint']:
            skipped += 1
            continue

        # C++ fingerprint computed from ground truth (semantic signature)
        expr = item.get('ground_truth_expr') or item.get('simplified')
        if not expr:
            if args.verbose:
                print(f"Sample {i}: missing ground truth expression")
            skipped += 1
            continue

        cpp_fp = np.array(item['fingerprint']['flat'], dtype=np.float32)

        # Validate dimension
        if len(cpp_fp) != FINGERPRINT_DIM:
            print(f"Sample {i}: wrong dimension {len(cpp_fp)}, expected {FINGERPRINT_DIM}")
            skipped += 1
            continue

        # Validate sample
        result = validate_sample(expr, cpp_fp, fingerprint, args.tolerance, components)

        if result:
            if 'error' in result:
                print(f"Sample {i}: Python compute error: {result['error']}")
                skipped += 1
            else:
                mismatches += 1
                if args.verbose:
                    print(f"Sample {i}: max diff = {result['max_diff']:.6f}")
                    print(f"  Expr: {expr[:60]}...")
                    for name, info in result['components'].items():
                        match_rate = info['match_rate'] * 100
                        print(f"    {name}: max diff = {info['max_diff']:.6f}, "
                              f"matches = {info['matches']}/{info['total']} ({match_rate:.1f}%)")

                # Accumulate component statistics
                for name, info in result['components'].items():
                    component_max_diffs[name].append(info['max_diff'])
                for name, matches in result.get('matches', {}).items():
                    component_match_counts[name] += matches
                    _, start, end = [(n, s, e) for n, s, e in components if n == name][0]
                    component_total_counts[name] += (end - start)
        else:
            total += 1
            # Perfect match - all components match
            for name, start, end in components:
                component_match_counts[name] += (end - start)
                component_total_counts[name] += (end - start)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    for name in component_names:
        matches = component_match_counts[name]
        total_dims = component_total_counts[name]
        if total_dims > 0:
            match_rate = matches / total_dims * 100
            max_diffs = component_max_diffs[name]
            max_diff_val = max(max_diffs) if max_diffs else 0.0
            status = "✓" if match_rate == 100.0 else "✗"
            print(f"{status} {name:15s}: {matches:5d}/{total_dims:5d} matches ({match_rate:6.2f}%) | "
                  f"Max diff: {max_diff_val:.6f}")

    print("\n" + "=" * 70)
    print(f"Overall: {total}/{total+mismatches} samples fully matched")
    print(f"Skipped: {skipped} samples")
    if total + mismatches > 0:
        overall_rate = total / (total + mismatches) * 100
        print(f"Match rate: {overall_rate:.1f}%")
    print("=" * 70)

    return 1 if mismatches > 0 else 0


if __name__ == '__main__':
    exit(main())
