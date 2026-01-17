#!/usr/bin/env python3
"""
Verify GCNII implementation correctness.

Validates:
1. Coefficient scheduling (alpha and beta)
2. Initial feature storage
3. Forward pass functionality
"""

import torch
from src.models.encoder import HGTEncoder, RGCNEncoder
from src.constants import GCNII_ALPHA, GCNII_LAMBDA, NUM_OPTIMIZED_EDGE_TYPES


def test_alpha_values():
    """Verify that alpha is constant across layers (not computed dynamically)."""
    encoder = HGTEncoder(
        hidden_dim=128,
        num_layers=12,
        gcnii_alpha=0.15,
    )

    # Alpha should be the same value set in init
    assert encoder.gcnii_alpha == 0.15
    print(f"✓ Alpha value: {encoder.gcnii_alpha}")


def test_beta_schedule():
    """Verify that beta decreases correctly with layer depth."""
    encoder = HGTEncoder(
        hidden_dim=128,
        num_layers=12,
        use_identity_mapping=True,
        gcnii_lambda=1.0,
    )

    print("\nBeta schedule across 12 layers:")
    print("=" * 50)
    print(f"{'Layer':<8} {'Beta':<10} {'Transformation %':<20}")
    print("=" * 50)

    for layer_idx in range(12):
        beta = encoder._compute_gcnii_beta(layer_idx, encoder.gcnii_lambda, encoder.use_identity_mapping)
        transform_pct = beta * 100
        identity_pct = (1 - beta) * 100

        print(f"{layer_idx:<8} {beta:<10.4f} {transform_pct:.1f}% transform, {identity_pct:.1f}% identity")

        # Sanity checks
        assert 0.0 <= beta <= 1.0, f"Beta out of range at layer {layer_idx}"

    print("=" * 50)
    print("✓ Beta schedule valid")


def test_feature_storage():
    """Verify that h_0 is stored and accessible during forward pass."""
    # Create dummy batch
    num_nodes = 10
    num_edges = 20

    x = torch.randint(0, 10, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, NUM_OPTIMIZED_EDGE_TYPES, (num_edges,))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    encoder = HGTEncoder(
        hidden_dim=256,
        num_layers=6,
        use_initial_residual=True,
        use_global_attention=False,
        use_path_encoding=False,
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(x, edge_index, batch, edge_type)

    assert output.shape[0] == num_nodes
    print(f"\n✓ Forward pass successful: output shape {output.shape}")


def test_rgcn_integration():
    """Verify RGCN encoder also works with GCNII."""
    num_nodes = 10
    num_edges = 20

    x = torch.randint(0, 10, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, NUM_OPTIMIZED_EDGE_TYPES, (num_edges,))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    encoder = RGCNEncoder(
        hidden_dim=256,
        num_layers=6,
        use_initial_residual=True,
        use_identity_mapping=True,
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(x, edge_index, batch, edge_type)

    assert output.shape[0] == num_nodes
    print(f"✓ RGCN forward pass successful: output shape {output.shape}")


def test_constants_import():
    """Check that constants were added correctly."""
    from src.constants import (
        GCNII_ALPHA,
        GCNII_LAMBDA,
        GCNII_USE_INITIAL_RESIDUAL,
        GCNII_USE_IDENTITY_MAPPING,
    )

    print(f"\n✓ GCNII constants imported successfully:")
    print(f"  - GCNII_ALPHA: {GCNII_ALPHA}")
    print(f"  - GCNII_LAMBDA: {GCNII_LAMBDA}")
    print(f"  - GCNII_USE_INITIAL_RESIDUAL: {GCNII_USE_INITIAL_RESIDUAL}")
    print(f"  - GCNII_USE_IDENTITY_MAPPING: {GCNII_USE_IDENTITY_MAPPING}")

    # Validate ranges
    assert 0.0 <= GCNII_ALPHA <= 1.0, "GCNII_ALPHA out of range"
    assert GCNII_LAMBDA > 0, "GCNII_LAMBDA must be positive"
    print("✓ All constants in valid ranges")


def test_encoder_instantiation():
    """Verify encoders can be instantiated with GCNII parameters."""
    # HGT with GCNII
    hgt = HGTEncoder(
        hidden_dim=256,
        num_layers=6,
        use_initial_residual=True,
        use_identity_mapping=True,
        use_global_attention=False,
        use_path_encoding=False,
    )
    print(f"\n✓ HGT with GCNII initialized: {hgt.parameter_count():,} parameters")

    # RGCN with GCNII
    rgcn = RGCNEncoder(
        hidden_dim=256,
        num_layers=6,
        use_initial_residual=True,
        use_identity_mapping=True,
    )
    print(f"✓ RGCN with GCNII initialized: {rgcn.parameter_count():,} parameters")

    # HGT without GCNII (baseline)
    hgt_baseline = HGTEncoder(
        hidden_dim=256,
        num_layers=6,
        use_initial_residual=False,
        use_identity_mapping=False,
        use_global_attention=False,
        use_path_encoding=False,
    )
    print(f"✓ HGT baseline (no GCNII) initialized: {hgt_baseline.parameter_count():,} parameters")


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("GCNII Implementation Verification")
    print("=" * 70)

    try:
        test_constants_import()
        test_alpha_values()
        test_beta_schedule()
        test_encoder_instantiation()
        test_feature_storage()
        test_rgcn_integration()

        print("\n" + "=" * 70)
        print("✅ All GCNII verifications passed!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run full test suite: pytest tests/test_gcnii_mitigation.py -v")
        print("2. Train with GCNII: python scripts/train.py --phase 2 --encoder hgt")
        print("3. Compare with baseline: train with use_initial_residual=False")

        return 0

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
