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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))

from constants import (
    FINGERPRINT_DIM, SYMBOLIC_DIM, CORNER_DIM,
    RANDOM_DIM, DERIVATIVE_DIM, TRUTH_TABLE_DIM
)
from fingerprint import SemanticFingerprint


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
        for name, start, end in components:
            comp_diff = diff[start:end].max()
            if comp_diff > tolerance:
                component_diffs[name] = comp_diff

        return {
            'max_diff': diff.max(),
            'components': component_diffs
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
    args = parser.parse_args()

    fingerprint = SemanticFingerprint()
    components = get_fingerprint_components()

    print(f"Loading {args.filepath}...")
    samples = load_samples(args.filepath)
    print(f"Found {len(samples)} samples, validating up to {args.samples}")

    mismatches = 0
    total = 0
    skipped = 0

    for i, item in enumerate(samples[:args.samples]):
        # Check required fields
        if 'fingerprint' not in item or 'flat' not in item['fingerprint']:
            skipped += 1
            continue

        expr = item.get('obfuscated_expr') or item.get('obfuscated')
        if not expr:
            print(f"Sample {i}: missing expression")
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
                print(f"Sample {i}: max diff = {result['max_diff']:.6f}")
                print(f"  Expr: {expr[:60]}...")
                for name, diff in result['components'].items():
                    print(f"    {name}: max diff = {diff:.6f}")
        else:
            total += 1

    print(f"\nResults: {mismatches}/{total+mismatches} mismatches, {skipped} skipped")
    if total + mismatches > 0:
        print(f"Match rate: {total/(total+mismatches)*100:.1f}%")
    return 1 if mismatches > 0 else 0


if __name__ == '__main__':
    exit(main())
