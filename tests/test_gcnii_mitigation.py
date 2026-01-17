"""
Test GCNII-style initial residuals and identity mapping.

Validates that:
1. GCNII parameters can be passed to encoders
2. Beta computation decreases with depth
3. Forward pass works with GCNII enabled/disabled
4. Initial features are preserved through deep layers
5. Checkpoint validation works correctly
"""

import torch
import pytest
from src.models.encoder import HGTEncoder, RGCNEncoder
from src.constants import NUM_OPTIMIZED_EDGE_TYPES, GCNII_ALPHA, GCNII_LAMBDA


def test_hgt_gcnii_parameters():
    """Test that HGT accepts GCNII parameters."""
    encoder = HGTEncoder(
        hidden_dim=256,
        num_layers=6,
        gcnii_alpha=0.2,
        gcnii_lambda=1.5,
        use_initial_residual=True,
        use_identity_mapping=True,
    )

    assert encoder.gcnii_alpha == 0.2
    assert encoder.gcnii_lambda == 1.5
    assert encoder.use_initial_residual == True
    assert encoder.use_identity_mapping == True


def test_rgcn_gcnii_parameters():
    """Test that RGCN accepts GCNII parameters."""
    encoder = RGCNEncoder(
        hidden_dim=256,
        num_layers=6,
        gcnii_alpha=0.15,
        gcnii_lambda=1.0,
        use_initial_residual=True,
        use_identity_mapping=True,
    )

    assert encoder.gcnii_alpha == 0.15
    assert encoder.gcnii_lambda == 1.0
    assert encoder.use_initial_residual == True
    assert encoder.use_identity_mapping == True


def test_beta_computation():
    """Test that beta decreases with layer depth."""
    encoder = HGTEncoder(
        hidden_dim=128,
        num_layers=6,
        use_identity_mapping=True,
        gcnii_lambda=1.0,
    )

    beta_values = [encoder._compute_gcnii_beta(l, encoder.gcnii_lambda, encoder.use_identity_mapping) for l in range(6)]

    # Beta should decrease monotonically
    for i in range(len(beta_values) - 1):
        assert beta_values[i] > beta_values[i+1], \
            f"Beta should decrease: layer {i}: {beta_values[i]}, layer {i+1}: {beta_values[i+1]}"

    # Early layers should have high beta (more transformation)
    assert beta_values[0] > 0.5, f"Early layer beta too low: {beta_values[0]}"

    # Late layers should have low beta (more identity)
    assert beta_values[-1] < 0.3, f"Late layer beta too high: {beta_values[-1]}"


def test_beta_disabled():
    """Test that beta returns 1.0 when identity mapping disabled."""
    encoder = HGTEncoder(
        hidden_dim=128,
        num_layers=6,
        use_identity_mapping=False,
    )

    for layer_idx in range(6):
        beta = encoder._compute_gcnii_beta(layer_idx, encoder.gcnii_lambda, encoder.use_identity_mapping)
        assert beta == 1.0, f"Beta should be 1.0 when disabled, got {beta}"


def test_hgt_forward_with_gcnii():
    """Test HGT forward pass with GCNII enabled."""
    encoder = HGTEncoder(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        use_initial_residual=True,
        use_identity_mapping=True,
        use_global_attention=False,
        use_path_encoding=False,
    )

    # Create dummy graph
    num_nodes = 10
    num_edges = 15

    x = torch.randint(0, 10, (num_nodes,))  # Node types
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, NUM_OPTIMIZED_EDGE_TYPES, (num_edges,))
    batch = torch.cat([
        torch.zeros(num_nodes // 2, dtype=torch.long),
        torch.ones(num_nodes - num_nodes // 2, dtype=torch.long)
    ])

    # Forward pass should not crash
    output = encoder(x, edge_index, batch, edge_type)

    assert output.shape == (num_nodes, 128)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_rgcn_forward_with_gcnii():
    """Test RGCN forward pass with GCNII enabled."""
    encoder = RGCNEncoder(
        hidden_dim=128,
        num_layers=4,
        use_initial_residual=True,
        use_identity_mapping=True,
    )

    # Create dummy graph
    num_nodes = 10
    num_edges = 15

    x = torch.randint(0, 10, (num_nodes,))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, NUM_OPTIMIZED_EDGE_TYPES, (num_edges,))
    batch = torch.cat([
        torch.zeros(num_nodes // 2, dtype=torch.long),
        torch.ones(num_nodes - num_nodes // 2, dtype=torch.long)
    ])

    # Forward pass should not crash
    output = encoder(x, edge_index, batch, edge_type)

    assert output.shape == (num_nodes, 128)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_gcnii_disabled():
    """Test that encoders work with GCNII disabled."""
    hgt = HGTEncoder(
        hidden_dim=64,
        num_layers=2,
        use_initial_residual=False,
        use_identity_mapping=False,
        use_global_attention=False,
        use_path_encoding=False,
    )

    rgcn = RGCNEncoder(
        hidden_dim=64,
        num_layers=2,
        use_initial_residual=False,
        use_identity_mapping=False,
    )

    # Create small test graph
    x = torch.randint(0, 10, (5,))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_type = torch.zeros(4, dtype=torch.long)
    batch = torch.zeros(5, dtype=torch.long)

    # Both should work without GCNII
    hgt_out = hgt(x, edge_index, batch, edge_type)
    rgcn_out = rgcn(x, edge_index, batch, edge_type)

    assert hgt_out.shape == (5, 64)
    assert rgcn_out.shape == (5, 64)


def test_feature_preservation():
    """
    Test that GCNII preserves more initial feature information
    compared to standard residuals.
    """
    # Create encoder with GCNII
    encoder_gcnii = HGTEncoder(
        hidden_dim=64,
        num_layers=6,
        use_initial_residual=True,
        use_identity_mapping=True,
        gcnii_alpha=0.2,  # 20% original features
        use_global_attention=False,
        use_path_encoding=False,
    )

    # Create encoder without GCNII
    encoder_standard = HGTEncoder(
        hidden_dim=64,
        num_layers=6,
        use_initial_residual=False,
        use_identity_mapping=False,
        use_global_attention=False,
        use_path_encoding=False,
    )

    # Same input
    torch.manual_seed(42)
    x = torch.randint(0, 10, (10,))
    edge_index = torch.randint(0, 10, (2, 15))
    edge_type = torch.randint(0, 8, (15,))
    batch = torch.zeros(10, dtype=torch.long)

    # Get initial embeddings
    with torch.no_grad():
        h_0_gcnii = encoder_gcnii.node_type_embed(x)
        h_0_standard = encoder_standard.node_type_embed(x)

        # Forward pass
        h_final_gcnii = encoder_gcnii(x, edge_index, batch, edge_type)
        h_final_standard = encoder_standard(x, edge_index, batch, edge_type)

        # Compute similarity to initial embeddings
        sim_gcnii = torch.nn.functional.cosine_similarity(
            h_0_gcnii, h_final_gcnii, dim=-1
        ).mean()

        sim_standard = torch.nn.functional.cosine_similarity(
            h_0_standard, h_final_standard, dim=-1
        ).mean()

    # GCNII should preserve more similarity to initial features
    assert sim_gcnii > sim_standard, \
        f"GCNII similarity ({sim_gcnii:.4f}) should be higher than standard ({sim_standard:.4f})"

    print(f"✓ GCNII preserves {sim_gcnii:.2%} similarity vs {sim_standard:.2%} for standard")


def test_checkpoint_compatibility_validation():
    """Test that checkpoint validation works correctly."""
    encoder = HGTEncoder(
        hidden_dim=64,
        num_layers=2,
        use_initial_residual=True,
        use_identity_mapping=True,
    )

    # Save state dict with metadata
    state_dict = encoder.state_dict()

    # Should contain metadata
    assert "_metadata" in state_dict
    assert "_gcnii_config" in state_dict["_metadata"]

    # Load into same config - should work
    encoder.load_state_dict(state_dict)

    # Try to load into encoder with different GCNII config - should raise error
    encoder_different = HGTEncoder(
        hidden_dim=64,
        num_layers=2,
        use_initial_residual=False,  # Different!
        use_identity_mapping=True,
    )

    with pytest.raises(ValueError, match="GCNII configuration mismatch"):
        encoder_different.load_state_dict(state_dict)


def test_default_constants_used():
    """Test that default GCNII constants are used when not specified."""
    encoder = HGTEncoder(hidden_dim=128, num_layers=4)

    # Should use values from constants.py
    assert encoder.gcnii_alpha == GCNII_ALPHA
    assert encoder.gcnii_lambda == GCNII_LAMBDA


def test_encoder_registry_integration():
    """Test that encoder registry works with GCNII parameters."""
    from src.models.encoder_registry import get_encoder

    # Should work with custom GCNII parameters
    encoder = get_encoder(
        'hgt',
        hidden_dim=128,
        num_layers=4,
        gcnii_alpha=0.25,
        use_initial_residual=False,
    )

    assert encoder.gcnii_alpha == 0.25
    assert encoder.use_initial_residual == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
