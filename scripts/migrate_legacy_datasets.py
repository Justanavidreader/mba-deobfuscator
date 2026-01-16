#!/usr/bin/env python3
"""
Migrate legacy datasets to current schema.

Converts node type IDs from legacy (pre-2026-01-15) to current format
and adds schema_version field to AST data.

Usage:
    python scripts/migrate_legacy_datasets.py --input old_train.jsonl --output new_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.edge_types import LEGACY_NODE_MAP


def migrate_dataset(input_path: str, output_path: str, verbose: bool = False) -> dict:
    """
    Convert legacy dataset to current schema.

    Args:
        input_path: Path to input JSONL file with legacy schema
        output_path: Path to output JSONL file with current schema
        verbose: Print progress updates

    Returns:
        Dictionary with migration statistics
    """
    stats = {
        'total_lines': 0,
        'migrated_lines': 0,
        'skipped_lines': 0,
        'nodes_converted': 0,
    }

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            stats['total_lines'] += 1

            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Skipping line {line_num} - JSON parse error: {e}")
                stats['skipped_lines'] += 1
                continue

            # Update AST if present
            if 'ast' in item and 'nodes' in item['ast']:
                for node in item['ast']['nodes']:
                    # Handle both 'type_id' and old 'type' numeric field
                    old_id = node.get('type_id')
                    if old_id is None:
                        # Try to get numeric type if 'type' is used as ID
                        type_val = node.get('type')
                        if isinstance(type_val, int):
                            old_id = type_val

                    if old_id is not None and isinstance(old_id, int):
                        if 0 <= old_id <= 9:
                            node['type_id'] = LEGACY_NODE_MAP[old_id]
                            stats['nodes_converted'] += 1
                        else:
                            if verbose:
                                print(f"Warning: Line {line_num} has out-of-range type_id: {old_id}")

                # Add schema version
                item['ast']['schema_version'] = 2

            stats['migrated_lines'] += 1
            f_out.write(json.dumps(item) + '\n')

            if verbose and stats['migrated_lines'] % 10000 == 0:
                print(f"Processed {stats['migrated_lines']} lines...")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Migrate legacy MBA datasets to current schema',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic migration
    python scripts/migrate_legacy_datasets.py --input data/train.jsonl --output data/train_v2.jsonl

    # With progress output
    python scripts/migrate_legacy_datasets.py --input data/train.jsonl --output data/train_v2.jsonl --verbose

Node Type ID Mapping:
    Legacy (pre-2026-01-15)    Current (v2)
    ADD = 0                    ADD = 2
    SUB = 1                    SUB = 3
    MUL = 2                    MUL = 4
    NEG = 3                    NEG = 9
    AND = 4                    AND = 5
    OR  = 5                    OR  = 6
    XOR = 6                    XOR = 7
    NOT = 7                    NOT = 8
    VAR = 8                    VAR = 0
    CONST = 9                  CONST = 1
"""
    )
    parser.add_argument('--input', required=True, help='Input JSONL file (legacy schema)')
    parser.add_argument('--output', required=True, help='Output JSONL file (current schema)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Migrating: {args.input} -> {args.output}")
    stats = migrate_dataset(args.input, args.output, verbose=args.verbose)

    print(f"\nMigration complete:")
    print(f"  Total lines:     {stats['total_lines']}")
    print(f"  Migrated lines:  {stats['migrated_lines']}")
    print(f"  Skipped lines:   {stats['skipped_lines']}")
    print(f"  Nodes converted: {stats['nodes_converted']}")
    print(f"\nOutput written to: {args.output}")


if __name__ == '__main__':
    main()
