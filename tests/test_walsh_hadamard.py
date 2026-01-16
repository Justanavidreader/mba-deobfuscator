"""Tests for Walsh-Hadamard transform and spectral features."""

import torch
import pytest
import numpy as np
from src.data.walsh_hadamard import (
    walsh_hadamard_transform,
    compute_walsh_features,
    compute_extended_walsh_features,
    WalshSpectrumEncoder,
)


class TestWalshHadamardTransform:
    """Tests for Walsh-Hadamard transform correctness."""

    def test_linear_function_sparse_spectrum(self):
        """Linear Boolean functions should have exactly 1 or 2 non-zero Walsh coefficients."""
        # f(x) = x_0 (first variable): truth table is [0,1,0,1,0,1,...] for 64 entries
        # For linear f(x) = x_i, W(0) = 0 (balanced function) and W(2^i) = ±64
        truth_table = torch.zeros(64)
        for i in range(64):
            truth_table[i] = i & 1  # Extract bit 0

        walsh = walsh_hadamard_transform(truth_table)

        # For balanced linear f(x) = x_0: W(0) = 0, W(1) = ±64
        # Count large coefficients (magnitude > 32)
        large_coeffs = (walsh.abs() > 32).sum().item()
        # Balanced linear functions have 1 large coeff, unbalanced have 2
        assert large_coeffs >= 1 and large_coeffs <= 2, \
            f"Linear function should have 1-2 large Walsh coeffs, got {large_coeffs}"

        # Check that W(1) is large (this is the x_0 coefficient)
        assert abs(walsh[1]) > 32, f"W(1) should be large for f(x)=x_0, got {walsh[1]}"

    def test_constant_function(self):
        """Constant function f(x) = 0 should have W(0) = 64, all others = 0."""
        truth_table = torch.zeros(64)
        walsh = walsh_hadamard_transform(truth_table)

        assert walsh[0].item() == 64, "W(0) should be 64 for constant-0 function"
        assert (walsh[1:].abs() < 1e-5).all(), "All other coefficients should be 0"

    def test_constant_one_function(self):
        """Constant function f(x) = 1 should have W(0) = -64, all others = 0."""
        truth_table = torch.ones(64)
        walsh = walsh_hadamard_transform(truth_table)

        assert walsh[0].item() == -64, "W(0) should be -64 for constant-1 function"
        assert (walsh[1:].abs() < 1e-5).all(), "All other coefficients should be 0"

    def test_parseval_identity(self):
        """Parseval's theorem: sum of W^2 = 2^(2n) = 4096 for n=6."""
        # Random Boolean function
        truth_table = (torch.rand(64) > 0.5).float()
        walsh = walsh_hadamard_transform(truth_table)

        sum_sq = (walsh ** 2).sum().item()
        expected = 64 * 64  # 2^12 = 4096
        assert abs(sum_sq - expected) < 1, f"Parseval: expected {expected}, got {sum_sq}"

    def test_batched_transform(self):
        """Test batched Walsh transform."""
        batch_size = 4
        truth_tables = (torch.rand(batch_size, 64) > 0.5).float()

        walsh = walsh_hadamard_transform(truth_tables)
        assert walsh.shape == (batch_size, 64)

        # Verify each batch item satisfies Parseval
        for i in range(batch_size):
            sum_sq = (walsh[i] ** 2).sum().item()
            assert abs(sum_sq - 4096) < 1

    def test_invalid_size_raises_error(self):
        """Test that non-power-of-2 size raises ValueError."""
        truth_table = torch.zeros(63)  # Not power of 2
        with pytest.raises(ValueError, match="must be power of 2"):
            walsh_hadamard_transform(truth_table)

    def test_wrong_size_raises_error(self):
        """Test that size != 64 raises ValueError."""
        truth_table = torch.zeros(32)  # Power of 2 but not 64
        with pytest.raises(ValueError, match="expects 64-entry"):
            walsh_hadamard_transform(truth_table)


class TestWalshFeatures:
    """Tests for Walsh feature extraction."""

    def test_feature_dimensions(self):
        """Test that Walsh features have correct dimensions."""
        truth_table = torch.zeros(64)

        base_features = compute_walsh_features(truth_table)
        assert base_features.shape == (17,), f"Expected 17 features, got {base_features.shape}"

        extended_features = compute_extended_walsh_features(
            truth_table, include_raw_spectrum=True, top_k=16
        )
        assert extended_features.shape == (33,), f"Expected 33 features, got {extended_features.shape}"

    def test_linearity_detection(self):
        """Test that is_linear feature correctly identifies linear functions."""
        # Linear function f(x) = x_0
        linear_tt = torch.tensor([float(i & 1) for i in range(64)])
        linear_features = compute_walsh_features(linear_tt)

        # Non-linear function f(x) = x_0 AND x_1
        nonlinear_tt = torch.tensor([float((i & 1) and (i & 2)) for i in range(64)])
        nonlinear_features = compute_walsh_features(nonlinear_tt)

        # is_linear is at index 13
        assert linear_features[13] > 0.5, "Linear function should be detected as linear"
        assert nonlinear_features[13] < 0.5, "AND function should not be detected as linear"

    def test_batched_features(self):
        """Test batched feature computation."""
        batch_size = 8
        truth_tables = (torch.rand(batch_size, 64) > 0.5).float()

        features = compute_walsh_features(truth_tables)
        assert features.shape == (batch_size, 17)

    def test_extended_features_without_spectrum(self):
        """Test extended features without raw spectrum."""
        truth_table = torch.zeros(64)
        features = compute_extended_walsh_features(truth_table, include_raw_spectrum=False)
        assert features.shape == (17,), "Should return base features only"


class TestWalshSpectrumEncoder:
    """Tests for Walsh spectrum encoder neural network."""

    def test_encoder_shape(self):
        """Test Walsh spectrum encoder output shape."""
        encoder = WalshSpectrumEncoder(output_dim=64)
        truth_table = torch.zeros(4, 64)  # Batch of 4

        output = encoder(truth_table)
        assert output.shape == (4, 64)

    def test_encoder_gradient_flow(self):
        """Test that gradients flow through Walsh encoder."""
        encoder = WalshSpectrumEncoder(output_dim=64)
        truth_table = torch.rand(2, 64, requires_grad=False)  # Truth table doesn't need grad

        output = encoder(truth_table)
        loss = output.sum()
        loss.backward()

        # Check that encoder parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None, "Encoder parameters should have gradients"

    def test_encoder_deterministic(self):
        """Test that encoder produces deterministic outputs."""
        encoder = WalshSpectrumEncoder(output_dim=64)
        truth_table = torch.rand(2, 64)

        # Set to eval mode to disable dropout
        encoder.eval()

        output1 = encoder(truth_table)
        output2 = encoder(truth_table)

        assert torch.allclose(output1, output2), "Encoder should be deterministic in eval mode"

    def test_encoder_with_different_top_k(self):
        """Test encoder with different top_k settings."""
        encoder_8 = WalshSpectrumEncoder(output_dim=64, top_k=8)
        encoder_16 = WalshSpectrumEncoder(output_dim=64, top_k=16)

        truth_table = torch.rand(2, 64)

        output_8 = encoder_8(truth_table)
        output_16 = encoder_16(truth_table)

        # Both should produce same output dim
        assert output_8.shape == output_16.shape == (2, 64)

    def test_encoder_without_raw_spectrum(self):
        """Test encoder without raw spectrum inclusion."""
        encoder = WalshSpectrumEncoder(output_dim=64, include_raw_spectrum=False)
        truth_table = torch.rand(2, 64)

        output = encoder(truth_table)
        assert output.shape == (2, 64)
