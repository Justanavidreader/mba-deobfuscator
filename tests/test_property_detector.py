"""Tests for property detection heads."""

import torch
import pytest
from src.models.property_detector import (
    VariablePropertyHead,
    InteractionPropertyHead,
    InvariantDetector,
)
from src.constants import FINGERPRINT_MODE


class TestVariablePropertyHead:
    """Tests for variable property detection head."""

    def test_forward_shape(self):
        """Test output shapes."""
        head = VariablePropertyHead(hidden_dim=256)
        var_embeds = torch.randn(5, 256)  # 5 variables
        output = head(var_embeds)

        assert output['logits'].shape == (5, 8)  # NUM_VAR_PROPERTIES=8
        assert output['probs'].shape == (5, 8)
        assert 'augmented' in output

    def test_probs_in_range(self):
        """Test that probabilities are in [0, 1]."""
        head = VariablePropertyHead(hidden_dim=256)
        var_embeds = torch.randn(5, 256)
        output = head(var_embeds)

        assert (output['probs'] >= 0).all()
        assert (output['probs'] <= 1).all()

    def test_augmented_dimension(self):
        """Test augmented embeddings have correct dimension."""
        head = VariablePropertyHead(hidden_dim=256)
        var_embeds = torch.randn(5, 256)
        output = head(var_embeds)

        # Should concatenate hidden_dim + SEMANTIC_HGT_PROPERTY_DIM (256 + 64)
        assert output['augmented'].shape == (5, 256 + 64)

    def test_without_augmentation(self):
        """Test forward without augmentation."""
        head = VariablePropertyHead(hidden_dim=256)
        var_embeds = torch.randn(5, 256)
        output = head(var_embeds, return_augmented=False)

        assert 'logits' in output
        assert 'probs' in output
        assert 'augmented' not in output

    def test_gradient_flow(self):
        """Test gradients flow through property head."""
        head = VariablePropertyHead(hidden_dim=256)
        var_embeds = torch.randn(5, 256, requires_grad=True)
        output = head(var_embeds, return_augmented=True)

        # Use augmented output which uses the embedding layer
        loss = output['augmented'].sum()
        loss.backward()

        assert var_embeds.grad is not None
        # Check Linear layers have gradients (embedding may not if not used in loss path)
        linear_params_with_grad = []
        for name, param in head.named_parameters():
            if param.requires_grad and 'mlp' in name and isinstance(param, torch.nn.Parameter):
                if 'weight' in name or 'bias' in name:
                    linear_params_with_grad.append((name, param.grad is not None))

        assert any(grad for _, grad in linear_params_with_grad), \
            "At least some Linear layer parameters should have gradients"


class TestInteractionPropertyHead:
    """Tests for interaction property detection head."""

    def test_forward_shape(self):
        """Test output shapes."""
        head = InteractionPropertyHead(hidden_dim=256)
        var_embeds = torch.randn(4, 256)  # 4 variables
        output = head(var_embeds)

        assert output['logits'].shape == (4, 4, 5)  # NUM_INTERACTION_PROPERTIES=5
        assert output['attention'].shape == (4, 4)

    def test_probs_in_range(self):
        """Test that probabilities are in [0, 1]."""
        head = InteractionPropertyHead(hidden_dim=256)
        var_embeds = torch.randn(4, 256)
        output = head(var_embeds)

        assert (output['probs'] >= 0).all()
        assert (output['probs'] <= 1).all()

    def test_attention_is_normalized(self):
        """Test that attention weights sum to 1."""
        head = InteractionPropertyHead(hidden_dim=256)
        var_embeds = torch.randn(4, 256)
        output = head(var_embeds)

        # Each row should sum to approximately 1
        row_sums = output['attention'].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_with_mask(self):
        """Test interaction head with variable mask."""
        head = InteractionPropertyHead(hidden_dim=256)
        var_embeds = torch.randn(4, 256)
        var_mask = torch.tensor([True, True, False, False])  # Only 2 valid variables

        output = head(var_embeds, var_mask=var_mask)

        assert output['logits'].shape == (4, 4, 5)
        # Attention should be masked for invalid variables
        assert not torch.isfinite(output['attention'][2, :]).any()

    def test_symmetric_properties(self):
        """Test that interaction properties are computed for all pairs."""
        head = InteractionPropertyHead(hidden_dim=256)
        var_embeds = torch.randn(3, 256)
        output = head(var_embeds)

        # Should have predictions for all 3x3 pairs (including self-interactions)
        assert output['logits'].shape == (3, 3, 5)


class TestInvariantDetector:
    """Tests for full invariant detector."""

    def test_initialization_validates_fingerprint_mode(self):
        """Test that initialization validates FINGERPRINT_MODE."""
        if FINGERPRINT_MODE != "full":
            with pytest.raises(ValueError, match="requires FINGERPRINT_MODE='full'"):
                InvariantDetector(hidden_dim=256)

    def test_initialization_validates_fingerprint_dim(self):
        """Test that initialization validates fingerprint dimension."""
        if FINGERPRINT_MODE == "full":
            with pytest.raises(ValueError, match="Fingerprint dimension mismatch"):
                InvariantDetector(hidden_dim=256, fingerprint_dim=400)

    def test_forward_with_valid_inputs(self):
        """Test forward pass with valid inputs."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        detector = InvariantDetector(hidden_dim=256, walsh_output_dim=64)

        # Create test inputs
        node_embeddings = torch.randn(10, 256)
        node_types = torch.tensor([0, 0, 0, 1, 2, 2, 5, 5, 1, 0])  # Mixed types
        fingerprint = torch.randn(1, 448)  # Single batch item
        batch = torch.zeros(10, dtype=torch.long)

        output = detector(node_embeddings, node_types, fingerprint, batch)

        assert 'augmented_embeddings' in output
        assert output['augmented_embeddings'].shape == (10, 256)
        assert 'var_properties' in output
        assert 'walsh_features' in output
        assert output['walsh_features'].shape == (1, 64)

    def test_forward_validates_fingerprint_dimension(self):
        """Test that forward validates fingerprint dimension at runtime."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        detector = InvariantDetector(hidden_dim=256)

        node_embeddings = torch.randn(10, 256)
        node_types = torch.zeros(10, dtype=torch.long)
        fingerprint = torch.randn(1, 400)  # Wrong dimension
        batch = torch.zeros(10, dtype=torch.long)

        with pytest.raises(RuntimeError, match="Fingerprint dimension mismatch"):
            detector(node_embeddings, node_types, fingerprint, batch)

    def test_walsh_raw_features(self):
        """Test that walsh_raw dict contains expected keys."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        detector = InvariantDetector(hidden_dim=256)

        # Fixed: node_embeddings and node_types must have matching first dimension
        node_embeddings = torch.randn(4, 256)  # Changed from 10 to 4
        node_types = torch.tensor([0, 0, 1, 2])
        fingerprint = torch.randn(1, 448)
        batch = torch.zeros(4, dtype=torch.long)

        output = detector(node_embeddings, node_types, fingerprint, batch)

        assert 'walsh_raw' in output
        assert 'features' in output['walsh_raw']
        assert 'is_linear' in output['walsh_raw']
        assert 'nonlinearity' in output['walsh_raw']
        assert 'degree_estimate' in output['walsh_raw']

    def test_handles_no_variables(self):
        """Test detector handles graphs with no variable nodes."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        detector = InvariantDetector(hidden_dim=256)

        # No variable nodes (type 0)
        node_embeddings = torch.randn(5, 256)
        node_types = torch.tensor([1, 2, 2, 5, 1])  # No VAR nodes
        fingerprint = torch.randn(1, 448)
        batch = torch.zeros(5, dtype=torch.long)

        output = detector(node_embeddings, node_types, fingerprint, batch)

        # Should still return valid output
        assert output['augmented_embeddings'].shape == (5, 256)
        assert len(output['var_properties']) == 0  # No variables processed

    def test_multiple_batches(self):
        """Test detector with multiple batch items."""
        if FINGERPRINT_MODE != "full":
            pytest.skip("Requires FINGERPRINT_MODE='full'")

        detector = InvariantDetector(hidden_dim=256)

        # 2 batch items with different numbers of nodes
        node_embeddings = torch.randn(8, 256)
        node_types = torch.tensor([0, 0, 1, 2, 0, 1, 2, 0])
        fingerprint = torch.randn(2, 448)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        output = detector(node_embeddings, node_types, fingerprint, batch)

        assert output['augmented_embeddings'].shape == (8, 256)
        assert len(output['var_properties']) == 2  # One per batch
        assert output['walsh_features'].shape == (2, 64)
