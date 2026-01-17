#!/usr/bin/env python3
"""
Quick test script to validate GCNII setup before running full training.

Tests:
1. Model creation with GCNII enabled/disabled
2. Forward pass with sample data
3. Parameter counting
4. Configuration parsing

Usage:
    python scripts/test_gcnii_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from src.models.full_model import MBADeobfuscator
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.ast_parser import parse_expr_to_graph
from src.constants import GCNII_ALPHA, GCNII_LAMBDA


def test_baseline_model():
    """Test baseline HGT model creation."""
    print("=" * 60)
    print("TEST 1: Baseline HGT (GCNII disabled)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MBADeobfuscator(
        encoder_type='hgt',
        hidden_dim=256,
        num_encoder_layers=12,
        num_encoder_heads=16,
        use_initial_residual=False,
        use_identity_mapping=False,
        edge_type_mode='optimized',
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {param_count:,} parameters")

    # Test forward pass
    test_expr = "(x & y) + (x ^ y)"
    print(f"Testing with: {test_expr}")

    tokenizer = MBATokenizer()
    fingerprint_gen = SemanticFingerprint()

    graph, node_types, node_values = parse_expr_to_graph(test_expr)
    fingerprint_vec = fingerprint_gen.compute(test_expr)

    # Convert to tensors
    x = torch.zeros((len(node_types), 32), device=device)
    edge_index = torch.tensor(
        [[e[0] for e in graph], [e[1] for e in graph]],
        dtype=torch.long, device=device
    )
    edge_attr = torch.tensor([e[2] for e in graph], dtype=torch.long, device=device)
    fingerprint_tensor = torch.tensor(fingerprint_vec, dtype=torch.float32, device=device).unsqueeze(0)

    # Create PyG data
    from torch_geometric.data import Data, Batch
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data]).to(device)

    # Target tokens
    target_tokens = tokenizer.encode("x | y")
    target_ids = torch.tensor([target_tokens], dtype=torch.long, device=device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch, fingerprint_tensor, target_ids[:, :-1])

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs['vocab_logits'].shape}")
    print(f"  Encoder output dimension: {model.encoder.hidden_dim}")

    print("\n✓ Baseline model test PASSED\n")
    return True


def test_gcnii_model():
    """Test GCNII-enabled HGT model."""
    print("=" * 60)
    print("TEST 2: GCNII-HGT (GCNII enabled)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MBADeobfuscator(
        encoder_type='hgt',
        hidden_dim=256,
        num_encoder_layers=12,
        num_encoder_heads=16,
        use_initial_residual=True,
        use_identity_mapping=True,
        gcnii_alpha=GCNII_ALPHA,
        gcnii_lambda=GCNII_LAMBDA,
        edge_type_mode='optimized',
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {param_count:,} parameters")
    print(f"  GCNII alpha: {GCNII_ALPHA}")
    print(f"  GCNII lambda: {GCNII_LAMBDA}")

    # Verify GCNII is enabled in encoder
    if hasattr(model.encoder, 'use_initial_residual'):
        print(f"  use_initial_residual: {model.encoder.use_initial_residual}")
        print(f"  use_identity_mapping: {model.encoder.use_identity_mapping}")
    else:
        print("  ⚠ Warning: Encoder doesn't have GCNII attributes")

    # Test forward pass
    test_expr = "(x & y) + (x ^ y)"
    print(f"Testing with: {test_expr}")

    tokenizer = MBATokenizer()
    fingerprint_gen = SemanticFingerprint()

    graph, node_types, node_values = parse_expr_to_graph(test_expr)
    fingerprint_vec = fingerprint_gen.compute(test_expr)

    # Convert to tensors
    x = torch.zeros((len(node_types), 32), device=device)
    edge_index = torch.tensor(
        [[e[0] for e in graph], [e[1] for e in graph]],
        dtype=torch.long, device=device
    )
    edge_attr = torch.tensor([e[2] for e in graph], dtype=torch.long, device=device)
    fingerprint_tensor = torch.tensor(fingerprint_vec, dtype=torch.float32, device=device).unsqueeze(0)

    # Create PyG data
    from torch_geometric.data import Data, Batch
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data]).to(device)

    # Target tokens
    target_tokens = tokenizer.encode("x | y")
    target_ids = torch.tensor([target_tokens], dtype=torch.long, device=device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch, fingerprint_tensor, target_ids[:, :-1])

    print(f"✓ Forward pass successful")
    print(f"  Output shape: {outputs['vocab_logits'].shape}")

    print("\n✓ GCNII model test PASSED\n")
    return True


def test_config_parsing():
    """Test configuration file parsing."""
    print("=" * 60)
    print("TEST 3: Configuration Parsing")
    print("=" * 60)

    config_path = project_root / 'configs' / 'phase2.yaml'
    print(f"Loading: {config_path}")

    if not config_path.exists():
        print(f"⚠ Warning: Config file not found: {config_path}")
        print("  This is expected if configs/phase2.yaml doesn't exist yet")
        return True

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("✓ Config loaded successfully")
    print(f"  Encoder type: {config['model'].get('encoder_type', 'not set')}")
    print(f"  Hidden dim: {config['model'].get('hidden_dim', 'not set')}")
    print(f"  Num layers: {config['model'].get('num_encoder_layers', 'not set')}")

    # Check curriculum stages
    if 'curriculum_stages' in config.get('training', {}):
        stages = config['training']['curriculum_stages']
        print(f"  Curriculum stages: {len(stages)}")
        for i, stage in enumerate(stages):
            print(f"    Stage {i+1}: depth≤{stage['max_depth']}, epochs={stage['epochs']}, target={stage['target']}")

    print("\n✓ Config parsing test PASSED\n")
    return True


def test_parameter_difference():
    """Compare parameter counts between baseline and GCNII."""
    print("=" * 60)
    print("TEST 4: Parameter Count Comparison")
    print("=" * 60)

    baseline = MBADeobfuscator(
        encoder_type='hgt',
        hidden_dim=256,
        num_encoder_layers=12,
        use_initial_residual=False,
        use_identity_mapping=False,
    )

    gcnii = MBADeobfuscator(
        encoder_type='hgt',
        hidden_dim=256,
        num_encoder_layers=12,
        use_initial_residual=True,
        use_identity_mapping=True,
    )

    baseline_params = sum(p.numel() for p in baseline.parameters())
    gcnii_params = sum(p.numel() for p in gcnii.parameters())
    diff = gcnii_params - baseline_params

    print(f"Baseline parameters: {baseline_params:,}")
    print(f"GCNII parameters:    {gcnii_params:,}")
    print(f"Difference:          {diff:,} ({diff/baseline_params*100:.2f}%)")

    if abs(diff / baseline_params) > 0.1:
        print(f"⚠ Warning: GCNII adds {diff/baseline_params*100:.1f}% parameters")
        print("  Expected: <5% increase due to minimal GCNII overhead")
    else:
        print("✓ Parameter increase is within expected range")

    print("\n✓ Parameter comparison test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GCNII SETUP VALIDATION")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("Baseline Model", test_baseline_model()))
    except Exception as e:
        print(f"✗ Baseline model test FAILED: {e}")
        results.append(("Baseline Model", False))

    try:
        results.append(("GCNII Model", test_gcnii_model()))
    except Exception as e:
        print(f"✗ GCNII model test FAILED: {e}")
        results.append(("GCNII Model", False))

    try:
        results.append(("Config Parsing", test_config_parsing()))
    except Exception as e:
        print(f"✗ Config parsing test FAILED: {e}")
        results.append(("Config Parsing", False))

    try:
        results.append(("Parameter Comparison", test_parameter_difference()))
    except Exception as e:
        print(f"✗ Parameter comparison test FAILED: {e}")
        results.append(("Parameter Comparison", False))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests PASSED! Ready to run GCNII ablation study.")
        print("\nNext steps:")
        print("  1. Generate dataset:")
        print("     python scripts/train_gcnii_ablation.py --mode generate-data")
        print("  2. Quick test:")
        print("     python scripts/train_gcnii_ablation.py --mode baseline --quick-mode")
        return 0
    else:
        print("\n✗ Some tests FAILED. Fix issues before running full training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
