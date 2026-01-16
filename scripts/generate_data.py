#!/usr/bin/env python3
"""
Dataset generation script for MBA Deobfuscator.

Generates (obfuscated, simplified) expression pairs with varying depths.

Usage:
    python scripts/generate_data.py --output data/train.jsonl --samples 100000
    python scripts/generate_data.py --output data/train.jsonl --samples 1000000 --min-depth 1 --max-depth 14

Output format (JSONL):
    {"obfuscated": "(x & y) + (x ^ y)", "simplified": "x | y", "depth": 3}
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ExprNode:
    """AST node for expression generation."""
    op: str
    left: Optional['ExprNode'] = None
    right: Optional['ExprNode'] = None
    value: Optional[str] = None  # For terminals

    def __str__(self) -> str:
        if self.value is not None:
            return self.value
        if self.op in ('~', 'NEG'):
            return f"(~{self.left})" if self.op == '~' else f"(-{self.left})"
        return f"({self.left} {self.op} {self.right})"

    @property
    def depth(self) -> int:
        if self.value is not None:
            return 1
        if self.right is None:
            return 1 + self.left.depth
        return 1 + max(self.left.depth, self.right.depth)


# MBA identities for obfuscation and simplification
MBA_IDENTITIES = [
    # (simplified, obfuscated_template)
    ("x | y", "(x & y) + (x ^ y)"),
    ("x & y", "((x | y) - (x ^ y)) // 2"),  # Integer division
    ("x ^ y", "(x | y) - (x & y)"),
    ("x + y", "(x ^ y) + 2 * (x & y)"),
    ("x - y", "(x ^ (~y)) + 2 * (x & (~y)) + 1"),
    ("~x", "-(x + 1)"),
    ("-x", "(~x) + 1"),
    ("x", "x ^ 0"),
    ("x", "x | 0"),
    ("x", "x & (~0)"),
    ("0", "x ^ x"),
    ("0", "x & (~x)"),
    ("-1", "x | (~x)"),
]

# Simple rewrite rules for simplification
SIMPLIFY_RULES = [
    # Tautologies
    ("x ^ x", "0"),
    ("x & ~x", "0"),
    ("x | ~x", "-1"),
    ("x ^ 0", "x"),
    ("x | 0", "x"),
    ("x & 0", "0"),
    ("x & -1", "x"),
    ("x | -1", "-1"),
    ("~~x", "x"),
    # MBA simplifications
    ("(x & y) + (x ^ y)", "x | y"),
    ("(x ^ y) + (x & y)", "x | y"),
    ("(x | y) - (x ^ y)", "x & y"),  # Simplified, ignoring division by 2
]

VARIABLES = ['x', 'y', 'z', 'a', 'b', 'c']
BINARY_OPS = ['+', '-', '*', '&', '|', '^']
UNARY_OPS = ['~']
CONSTANTS = list(range(256))


def compute_expr_depth(expr: str) -> int:
    """
    Compute actual AST depth of expression by tracking parenthesis nesting.

    More accurate than simple paren count since depth measures max nesting level.
    """
    max_depth = 0
    current_depth = 0

    for char in expr:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    # Account for base expression (variables/constants have depth 1)
    return max(max_depth + 1, 1)


def generate_random_expr(depth: int, variables: Optional[List[str]] = None) -> ExprNode:
    """Generate random expression tree."""
    if variables is None:
        variables = VARIABLES[:4]

    if depth <= 1:
        # Terminal: variable or constant
        if random.random() < 0.7:
            return ExprNode(op='VAR', value=random.choice(variables))
        else:
            return ExprNode(op='CONST', value=str(random.choice(CONSTANTS[:16])))

    # Choose operator
    if random.random() < 0.1:
        # Unary
        op = random.choice(UNARY_OPS)
        child = generate_random_expr(depth - 1, variables)
        return ExprNode(op=op, left=child)
    else:
        # Binary
        op = random.choice(BINARY_OPS)
        # Split remaining depth between children
        left_depth = random.randint(1, depth - 1)
        right_depth = depth - 1
        left = generate_random_expr(left_depth, variables)
        right = generate_random_expr(right_depth, variables)
        return ExprNode(op=op, left=left, right=right)


def apply_obfuscation(expr_str: str, depth: int = 1) -> str:
    """Apply MBA obfuscation to expression."""
    result = expr_str

    for _ in range(depth):
        # Try each identity
        for simp, obf in MBA_IDENTITIES:
            # Simple variable substitution (not a full parser)
            if 'x' in simp and 'y' not in simp:
                # Single variable identity
                for var in VARIABLES:
                    pattern = simp.replace('x', var)
                    replacement = obf.replace('x', var)
                    if pattern in result:
                        result = result.replace(pattern, f"({replacement})", 1)
                        break

    return result


def generate_mba_pair(target_depth: int) -> Tuple[str, str, int]:
    """
    Generate an (obfuscated, simplified) pair.

    Returns:
        (obfuscated, simplified, actual_depth)
    """
    # Generate simplified expression
    base_depth = max(1, target_depth - random.randint(1, 3))
    simplified_tree = generate_random_expr(base_depth)
    simplified = str(simplified_tree)

    # Apply obfuscation
    obfuscation_rounds = random.randint(1, min(3, target_depth // 2 + 1))
    obfuscated = simplified

    for _ in range(obfuscation_rounds):
        # Apply random MBA identity
        identity_idx = random.randint(0, len(MBA_IDENTITIES) - 1)
        simp_pattern, obf_pattern = MBA_IDENTITIES[identity_idx]

        # Try to apply (simple string replacement)
        for var in VARIABLES:
            old = simp_pattern.replace('x', var).replace('y', var if 'y' not in simp_pattern else random.choice(VARIABLES))
            new = obf_pattern.replace('x', var).replace('y', var if 'y' not in obf_pattern else random.choice(VARIABLES))

            if random.random() < 0.3:
                # Wrap entire expression
                obfuscated = new.replace('x', f"({obfuscated})")
            elif old in obfuscated:
                obfuscated = obfuscated.replace(old, f"({new})", 1)

    # Compute actual depth using proper nesting measurement
    actual_depth = compute_expr_depth(obfuscated)

    return obfuscated, simplified, actual_depth


def generate_from_identities(target_depth: int) -> Tuple[str, str, int]:
    """Generate pair directly from MBA identity."""
    # Pick random identity
    simplified, obfuscated = random.choice(MBA_IDENTITIES)

    # Substitute random variables
    vars_used = []
    for v in ['x', 'y', 'z']:
        if v in simplified or v in obfuscated:
            new_var = random.choice([v for v in VARIABLES if v not in vars_used])
            vars_used.append(new_var)
            simplified = simplified.replace(v, new_var)
            obfuscated = obfuscated.replace(v, new_var)

    # Optionally nest
    if target_depth > 3 and random.random() < 0.5:
        inner_obf, inner_simp, _ = generate_mba_pair(target_depth - 2)
        # Replace one variable with nested expression
        var_to_replace = random.choice(vars_used) if vars_used else 'x'
        obfuscated = obfuscated.replace(var_to_replace, f"({inner_obf})", 1)
        simplified = simplified.replace(var_to_replace, f"({inner_simp})", 1)

    depth = compute_expr_depth(obfuscated)
    return obfuscated, simplified, depth


def generate_dataset(
    output_path: str,
    num_samples: int,
    min_depth: int = 1,
    max_depth: int = 14,
    seed: int = 42
):
    """Generate full dataset."""
    random.seed(seed)

    # Depth distribution (more shallow expressions for curriculum)
    depth_weights = {d: max_depth - d + 1 for d in range(min_depth, max_depth + 1)}
    total_weight = sum(depth_weights.values())
    depth_probs = {d: w / total_weight for d, w in depth_weights.items()}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_skipped = 0

    with open(output_path, 'w') as f:
        for _ in tqdm(range(num_samples), desc="Generating"):
            # Sample target depth
            target_depth = random.choices(
                list(depth_probs.keys()),
                weights=list(depth_probs.values())
            )[0]

            # Generate pair
            if random.random() < 0.3:
                # From identity directly
                obfuscated, simplified, depth = generate_from_identities(target_depth)
            else:
                # Random generation
                obfuscated, simplified, depth = generate_mba_pair(target_depth)

            # Skip invalid samples
            if not obfuscated or not simplified:
                total_skipped += 1
                continue

            # Write
            item = {
                'obfuscated': obfuscated,
                'simplified': simplified,
                'depth': depth
            }
            f.write(json.dumps(item) + '\n')
            total_generated += 1

    print(f"Generated {total_generated} samples to {output_path}")
    if total_skipped > 0:
        skip_rate = total_skipped / (total_generated + total_skipped) * 100
        print(f"Skipped {total_skipped} invalid samples ({skip_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate MBA deobfuscation dataset')
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--samples', '-n', type=int, default=100000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--min-depth', type=int, default=1,
        help='Minimum expression depth'
    )
    parser.add_argument(
        '--max-depth', type=int, default=14,
        help='Maximum expression depth'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--split', action='store_true',
        help='Generate train/val/test splits'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.1,
        help='Test split ratio'
    )

    args = parser.parse_args()

    if args.split:
        # Generate splits
        base_path = Path(args.output)
        base_name = base_path.stem
        base_dir = base_path.parent

        train_samples = int(args.samples * (1 - args.val_ratio - args.test_ratio))
        val_samples = int(args.samples * args.val_ratio)
        test_samples = int(args.samples * args.test_ratio)

        generate_dataset(
            str(base_dir / f"{base_name}_train.jsonl"),
            train_samples,
            args.min_depth, args.max_depth, args.seed
        )
        generate_dataset(
            str(base_dir / f"{base_name}_val.jsonl"),
            val_samples,
            args.min_depth, args.max_depth, args.seed + 1
        )
        generate_dataset(
            str(base_dir / f"{base_name}_test.jsonl"),
            test_samples,
            args.min_depth, args.max_depth, args.seed + 2
        )
    else:
        generate_dataset(
            args.output,
            args.samples,
            args.min_depth, args.max_depth, args.seed
        )


if __name__ == '__main__':
    main()
