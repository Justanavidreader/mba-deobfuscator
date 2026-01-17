#!/usr/bin/env python3
"""
Test script for CustomFormatDataset.

Demonstrates loading data from custom text format with manual fingerprint/AST computation.

Usage:
    python scripts/test_custom_format_dataset.py --data path/to/custom_format.txt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.custom_format_dataset import CustomFormatDataset
from src.data.collate import collate_custom_format
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint


def main():
    parser = argparse.ArgumentParser(description='Test CustomFormatDataset loader')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to custom format text file',
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum expression depth to load',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for DataLoader test',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to display',
    )

    args = parser.parse_args()

    print("="*60)
    print("Testing CustomFormatDataset")
    print("="*60)

    # Initialize tokenizer and fingerprint
    print("\nInitializing tokenizer and fingerprint...")
    tokenizer = MBATokenizer()
    fingerprint = SemanticFingerprint()

    # Load dataset
    print(f"\nLoading dataset from: {args.data}")
    dataset = CustomFormatDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        fingerprint=fingerprint,
        max_depth=args.max_depth,
        skip_invalid=True,
    )

    print(f"\n✓ Loaded {len(dataset)} samples")

    # Print distributions
    print("\nDepth distribution:")
    depth_dist = dataset.get_depth_distribution()
    for depth in sorted(depth_dist.keys()):
        count = depth_dist[depth]
        print(f"  Depth {depth:2d}: {count:4d} samples")

    print("\nSection distribution:")
    section_dist = dataset.get_section_distribution()
    for section, count in sorted(section_dist.items()):
        print(f"  {section}: {count} samples")

    # Display sample data
    print(f"\nFirst {args.num_samples} samples:")
    print("-"*60)
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Section: {sample['section']}")
        print(f"  Depth: {sample['depth']}")
        print(f"  Obfuscated: {sample['obfuscated'][:60]}...")
        print(f"  Simplified: {sample['simplified'][:60]}...")
        if sample['additional']:
            print(f"  Additional: {sample['additional'][:60]}...")
        print(f"  Graph nodes: {sample['graph_data'].num_nodes}")
        print(f"  Graph edges: {sample['graph_data'].num_edges}")
        print(f"  Fingerprint shape: {sample['fingerprint'].shape}")
        print(f"  Obfuscated tokens: {sample['obfuscated_tokens'].shape}")
        print(f"  Simplified tokens: {sample['simplified_tokens'].shape}")

    # Test DataLoader with collate function
    print("\n" + "="*60)
    print(f"Testing DataLoader with batch_size={args.batch_size}")
    print("="*60)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_custom_format,
        num_workers=0,
    )

    # Get first batch
    batch = next(iter(dataloader))

    print(f"\nBatch structure:")
    print(f"  graph_batch: {batch['graph_batch']}")
    print(f"    - Total nodes: {batch['graph_batch'].num_nodes}")
    print(f"    - Total edges: {batch['graph_batch'].num_edges}")
    print(f"  fingerprint: {batch['fingerprint'].shape}")
    print(f"  obfuscated_tokens: {batch['obfuscated_tokens'].shape}")
    print(f"  simplified_tokens: {batch['simplified_tokens'].shape}")
    print(f"  depth: {batch['depth'].shape}")
    print(f"  Number of expressions: {len(batch['obfuscated'])}")
    print(f"  Sections: {batch['section']}")

    print("\n✓ DataLoader test successful!")

    # Test section filtering
    if section_dist:
        first_section = list(section_dist.keys())[0]
        print(f"\nTesting section filtering (section: '{first_section}')...")
        filtered_dataset = dataset.filter_by_section(first_section)
        print(f"✓ Filtered dataset has {len(filtered_dataset)} samples")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == '__main__':
    main()
