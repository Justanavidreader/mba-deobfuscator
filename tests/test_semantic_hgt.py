"""Tests for Semantic HGT encoder."""

import torch
import pytest
from src.models.semantic_hgt import SemanticHGTEncoder, PropertyAwareReadout
from src.constants import FINGERPRINT_MODE


class TestSemanticHGTEncoder:
    """Tests for Semantic HGT encoder with property detection."""

    def test_initialization_validates_injection_layer(self):
        """Test that invalid property_injection_layer raises ValueError."""
        # Layer out of bounds
        with pytest.raises(ValueError, match="property_injection_layer must be in"):
            SemanticHGTEncoder(
                hidden_dim=256,
                num_layers=12,
                property_injection_layer=12,  # Out of bounds (>= num_layers)
            )

        with pytest.raises(ValueError, match="property_injection_layer must be in"):
            SemanticHGTEncoder(
                hidden_dim=256,
                num_layers=12,
                property_injection_layer=-1,  # Negative
            )

    def test_initialization_with_valid_injection_layer(self):
        """Test initialization with valid injection layer."""
        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=12,
            num_heads=4,
            property_injection_layer=8,
        )
        assert encoder.property_injection_layer == 8

    def test_forward_without_fingerprint(self):
        """Test forward pass without fingerprint (property detection disabled)."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=2,  # Reduced for testing
            num_heads=4,
            enable_property_detection=True,
            property_injection_layer=1,
        )

        # Create minimal test input
        x = torch.tensor([0, 0, 2, 1])  # VAR, VAR, ADD, CONST
        edge_index = torch.tensor([[2, 2], [0, 1]])  # ADD -> VAR edges
        edge_type = torch.tensor([0, 1])  # LEFT_OPERAND, RIGHT_OPERAND
        batch = torch.tensor([0, 0, 0, 0])

        # Without fingerprint, should still work but no properties
        output = encoder(x, edge_index, edge_type, batch, fingerprint=None)

        assert 'embeddings' in output
        assert output['embeddings'].shape == (4, 256)

    def test_forward_with_fingerprint(self):
        """Test forward pass with fingerprint (property detection enabled)."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            enable_property_detection=True,
            property_injection_layer=1,
        )

        x = torch.tensor([0, 0, 2, 1])
        edge_index = torch.tensor([[2, 2], [0, 1]])
        edge_type = torch.tensor([0, 1])
        batch = torch.tensor([0, 0, 0, 0])
        fingerprint = torch.randn(1, 448)

        output = encoder(x, edge_index, edge_type, batch, fingerprint=fingerprint)

        assert 'embeddings' in output
        assert output['embeddings'].shape == (4, 256)
        assert 'var_properties' in output
        assert 'walsh_features' in output

    def test_property_detection_disabled(self):
        """Test encoder with property detection disabled."""
        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            enable_property_detection=False,
        )

        x = torch.tensor([0, 0, 2, 1])
        edge_index = torch.tensor([[2, 2], [0, 1]])
        edge_type = torch.tensor([0, 1])
        batch = torch.tensor([0, 0, 0, 0])
        fingerprint = torch.randn(1, 448)

        output = encoder(x, edge_index, edge_type, batch, fingerprint=fingerprint)

        assert 'embeddings' in output
        # Should not have property outputs when disabled
        assert 'var_properties' not in output or len(output.get('var_properties', [])) == 0

    def test_dict_to_flat_conversion(self):
        """Test heterogeneous dict to flat tensor conversion."""
        encoder = SemanticHGTEncoder(hidden_dim=256, num_layers=2, num_heads=4)

        # Create test dict
        x_dict = {
            '0': torch.randn(2, 256),  # 2 VAR nodes
            '1': torch.randn(1, 256),  # 1 CONST node
            '2': torch.randn(1, 256),  # 1 ADD node
        }
        node_types = torch.tensor([0, 0, 1, 2])

        h_flat = encoder._dict_to_flat(x_dict, node_types)

        assert h_flat.shape == (4, 256)

    def test_flat_to_dict_conversion(self):
        """Test flat tensor to heterogeneous dict conversion."""
        encoder = SemanticHGTEncoder(hidden_dim=256, num_layers=2, num_heads=4)

        h_flat = torch.randn(4, 256)
        node_types = torch.tensor([0, 0, 1, 2])

        x_dict = encoder._flat_to_dict(h_flat, node_types)

        assert '0' in x_dict
        assert '1' in x_dict
        assert '2' in x_dict
        assert x_dict['0'].shape == (2, 256)
        assert x_dict['1'].shape == (1, 256)
        assert x_dict['2'].shape == (1, 256)

    def test_round_trip_conversion(self):
        """Test that dict->flat->dict is consistent."""
        encoder = SemanticHGTEncoder(hidden_dim=256, num_layers=2, num_heads=4)

        original_dict = {
            '0': torch.randn(2, 256),
            '1': torch.randn(1, 256),
        }
        node_types = torch.tensor([0, 0, 1])

        h_flat = encoder._dict_to_flat(original_dict, node_types)
        reconstructed_dict = encoder._flat_to_dict(h_flat, node_types)

        assert set(original_dict.keys()) == set(reconstructed_dict.keys())
        for key in original_dict:
            assert torch.allclose(original_dict[key], reconstructed_dict[key])

    def test_gradient_flow(self):
        """Test gradients flow through Semantic HGT."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            property_injection_layer=1,
        )

        x = torch.tensor([0, 0, 2, 1])
        edge_index = torch.tensor([[2, 2], [0, 1]])
        edge_type = torch.tensor([0, 1])
        batch = torch.tensor([0, 0, 0, 0])
        fingerprint = torch.randn(1, 448)

        output = encoder(x, edge_index, edge_type, batch, fingerprint=fingerprint)
        loss = output['embeddings'].sum()
        loss.backward()

        # Check that parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multiple_batches(self):
        """Test encoder with multiple batch items."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        encoder = SemanticHGTEncoder(
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            property_injection_layer=1,
        )

        # 2 small graphs
        x = torch.tensor([0, 2, 0, 1, 2, 0])
        edge_index = torch.tensor([[1, 1, 4, 4], [0, 2, 3, 5]])
        edge_type = torch.tensor([0, 1, 0, 1])
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        fingerprint = torch.randn(2, 448)

        output = encoder(x, edge_index, edge_type, batch, fingerprint=fingerprint)

        assert output['embeddings'].shape == (6, 256)
        assert output['walsh_features'].shape == (2, 64)


class TestPropertyAwareReadout:
    """Tests for property-aware graph readout."""

    def test_forward_without_properties(self):
        """Test readout without property information."""
        readout = PropertyAwareReadout(hidden_dim=256)
        node_embeds = torch.randn(10, 256)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        output = readout(node_embeds, batch)
        assert output.shape == (2, 256)  # 2 graphs

    def test_forward_with_properties(self):
        """Test readout with property information."""
        readout = PropertyAwareReadout(hidden_dim=256)
        node_embeds = torch.randn(10, 256)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        node_types = torch.tensor([0, 0, 1, 2, 2, 0, 0, 1, 2, 0])
        var_property_probs = torch.rand(5, 8)  # 5 variables, 8 properties

        output = readout(node_embeds, batch, var_property_probs, node_types)
        assert output.shape == (2, 256)

    def test_property_weighting_affects_output(self):
        """Test that property weighting changes the output."""
        readout = PropertyAwareReadout(hidden_dim=256)
        torch.manual_seed(42)  # For reproducibility

        node_embeds = torch.randn(10, 256)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        node_types = torch.tensor([0, 0, 1, 2, 2, 0, 0, 1, 2, 0])

        # Different property probabilities
        var_props_1 = torch.zeros(5, 8)
        var_props_2 = torch.ones(5, 8)

        output_1 = readout(node_embeds, batch, var_props_1, node_types)
        output_2 = readout(node_embeds, batch, var_props_2, node_types)

        # Outputs should be different due to different property weighting
        assert not torch.allclose(output_1, output_2, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through property-aware readout."""
        readout = PropertyAwareReadout(hidden_dim=256)
        node_embeds = torch.randn(10, 256, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        output = readout(node_embeds, batch)
        loss = output.sum()
        loss.backward()

        assert node_embeds.grad is not None
        for param in readout.parameters():
            assert param.grad is not None

    def test_single_graph(self):
        """Test readout with single graph."""
        readout = PropertyAwareReadout(hidden_dim=256)
        node_embeds = torch.randn(5, 256)
        batch = torch.zeros(5, dtype=torch.long)

        output = readout(node_embeds, batch)
        assert output.shape == (1, 256)

    def test_empty_batch_handling(self):
        """Test readout handles edge case of very small batches."""
        readout = PropertyAwareReadout(hidden_dim=256)
        node_embeds = torch.randn(1, 256)
        batch = torch.zeros(1, dtype=torch.long)

        output = readout(node_embeds, batch)
        assert output.shape == (1, 256)
