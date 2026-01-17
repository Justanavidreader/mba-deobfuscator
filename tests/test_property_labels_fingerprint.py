"""
Unit tests for fingerprint-based property label generation.

Tests individual property detectors, batch processing, and integration with the training pipeline.
"""

import pytest
import torch
import numpy as np
from src.training.property_labels_fingerprint import FingerprintPropertyDetector
from src.data.fingerprint import SemanticFingerprint
from src.constants import MAX_VARS, NUM_VAR_PROPERTIES


class TestPropertyDetection:
    """Test individual property detectors."""

    @pytest.fixture
    def detector(self):
        return FingerprintPropertyDetector(device='cpu')

    @pytest.fixture
    def fingerprint_computer(self):
        return SemanticFingerprint()

    def test_boolean_only_detection(self, detector, fingerprint_computer):
        """Test BOOLEAN_ONLY property on known expressions."""
        # Boolean-only: x & y | z
        fp1 = fingerprint_computer.compute("(x0 & x1) | x2")

        # Mixed: (x & y) + z
        fp2 = fingerprint_computer.compute("(x0 & x1) + x2")

        fp_batch = torch.tensor([fp1, fp2], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # BOOLEAN_ONLY (index 3)
        assert labels[0, 0, 3] > 0.5, "x&y|z should be boolean-only"
        assert labels[1, 0, 3] < 0.5, "(x&y)+z should not be boolean-only"

    def test_arithmetic_only_detection(self, detector, fingerprint_computer):
        """Test ARITHMETIC_ONLY property."""
        # Arithmetic-only: x + y
        fp1 = fingerprint_computer.compute("x0 + x1")

        # Mixed: x + (y & z)
        fp2 = fingerprint_computer.compute("x0 + (x1 & x2)")

        fp_batch = torch.tensor([fp1, fp2], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # ARITHMETIC_ONLY (index 4)
        assert labels[0, 0, 4] > 0.5, "x+y should be arithmetic-only"
        assert labels[1, 0, 4] < 0.5, "x+(y&z) should not be arithmetic-only"

    def test_mixed_domain_detection(self, detector, fingerprint_computer):
        """Test MIXED_DOMAIN property."""
        # Mixed: (x & y) + (x ^ y)
        fp = fingerprint_computer.compute("(x0 & x1) + (x0 ^ x1)")

        fp_batch = torch.tensor([fp], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # MIXED_DOMAIN (index 7)
        assert labels[0, 0, 7] > 0.5, "(x&y)+(x^y) should be mixed domain"

    def test_batch_shape(self, detector):
        """Test output shape for various batch sizes."""
        for B in [1, 16, 32]:
            fp = torch.randn(B, 448)
            labels = detector.detect(fp)
            assert labels.shape == (B, MAX_VARS, NUM_VAR_PROPERTIES), \
                f"Expected shape ({B}, {MAX_VARS}, {NUM_VAR_PROPERTIES}), got {labels.shape}"

    def test_inactive_variables(self, detector, fingerprint_computer):
        """Test that inactive variables get zero labels."""
        # Expression with only x0 (x1-x7 inactive)
        fp = fingerprint_computer.compute("x0 + 1")

        fp_batch = torch.tensor([fp], dtype=torch.float32)
        labels = detector.detect(fp_batch)  # [1, 8, 8]

        # x1-x7 should have all-zero labels
        for var_idx in range(1, MAX_VARS):
            assert labels[0, var_idx].sum() == 0, \
                f"Inactive variable x{var_idx} should have zero labels"

    def test_label_range(self, detector):
        """Test that labels are in valid range [0, 1]."""
        fp = torch.randn(4, 448)
        labels = detector.detect(fp)

        assert torch.all((labels >= 0) & (labels <= 1)), \
            "Labels must be in range [0, 1]"

    def test_non_nan_labels(self, detector):
        """Test that labels contain no NaN values."""
        fp = torch.randn(4, 448)
        labels = detector.detect(fp)

        assert not torch.any(torch.isnan(labels)), \
            "Labels should not contain NaN values"


class TestVectorization:
    """Test vectorization and performance."""

    def test_no_loops_in_detect(self):
        """Verify main detect method is vectorized (no explicit loops over batch)."""
        import inspect
        from src.training.property_labels_fingerprint import FingerprintPropertyDetector

        source = inspect.getsource(FingerprintPropertyDetector.detect)

        # Check that no 'for i in range(B)' patterns exist in detect method
        # Exception: Walsh computation uses loop (documented limitation)
        lines = source.split('\n')
        for line in lines:
            if 'for i in range(B)' in line and '_compute_walsh_batch' not in line:
                pytest.fail(f"Found non-vectorized loop in detect(): {line}")

    def test_batch_consistency(self):
        """Test that batched processing gives same results as sequential."""
        detector = FingerprintPropertyDetector(device='cpu')

        # Create 4 random fingerprints
        fps = torch.randn(4, 448)

        # Batch processing
        labels_batch = detector.detect(fps)

        # Sequential processing
        labels_seq = []
        for i in range(4):
            labels_seq.append(detector.detect(fps[i:i+1]))
        labels_seq = torch.cat(labels_seq, dim=0)

        # Should match
        assert torch.allclose(labels_batch, labels_seq, atol=1e-5), \
            "Batch and sequential processing should give identical results"


class TestIntegration:
    """Test integration with training pipeline."""

    def test_compute_property_labels_integration(self):
        """Test integration with compute_property_labels_from_fingerprint."""
        from src.training.losses import compute_property_labels_from_fingerprint

        # Create mock fingerprint batch
        fingerprint = torch.randn(8, 448)

        # Call function (should use FingerprintPropertyDetector)
        labels = compute_property_labels_from_fingerprint(fingerprint)

        # Verify shape
        assert labels.shape == (8, MAX_VARS, NUM_VAR_PROPERTIES)

        # Verify labels are non-zero (not placeholder)
        assert labels.sum() > 0, "Labels should be non-zero (not placeholder)"

    def test_device_handling(self):
        """Test that detector handles CPU/GPU correctly."""
        from src.training.losses import compute_property_labels_from_fingerprint

        # CPU
        fingerprint_cpu = torch.randn(4, 448, device='cpu')
        labels_cpu = compute_property_labels_from_fingerprint(fingerprint_cpu)
        assert labels_cpu.device.type == 'cpu'

        # GPU (if available)
        if torch.cuda.is_available():
            fingerprint_gpu = torch.randn(4, 448, device='cuda')
            labels_gpu = compute_property_labels_from_fingerprint(fingerprint_gpu)
            assert labels_gpu.device.type == 'cuda'


class TestKnownExpressions:
    """Test against manually verified expressions."""

    @pytest.fixture
    def detector(self):
        return FingerprintPropertyDetector(device='cpu')

    @pytest.fixture
    def fingerprint_computer(self):
        return SemanticFingerprint()

    def test_simple_addition(self, detector, fingerprint_computer):
        """Test: x0 + x1 should be ARITHMETIC_ONLY."""
        fp = fingerprint_computer.compute("x0 + x1")
        fp_batch = torch.tensor([fp], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # ARITHMETIC_ONLY (index 4)
        assert labels[0, 0, 4] > 0.5, "x0+x1 should be arithmetic-only"
        # Not BOOLEAN_ONLY (index 3)
        assert labels[0, 0, 3] < 0.5, "x0+x1 should not be boolean-only"

    def test_simple_and(self, detector, fingerprint_computer):
        """Test: x0 & x1 should be BOOLEAN_ONLY."""
        fp = fingerprint_computer.compute("x0 & x1")
        fp_batch = torch.tensor([fp], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # BOOLEAN_ONLY (index 3)
        assert labels[0, 0, 3] > 0.5, "x0&x1 should be boolean-only"
        # Not ARITHMETIC_ONLY (index 4)
        assert labels[0, 0, 4] < 0.5, "x0&x1 should not be arithmetic-only"

    def test_mba_classic(self, detector, fingerprint_computer):
        """Test: (x & y) + (x ^ y) should be MIXED_DOMAIN."""
        fp = fingerprint_computer.compute("(x0 & x1) + (x0 ^ x1)")
        fp_batch = torch.tensor([fp], dtype=torch.float32)
        labels = detector.detect(fp_batch)

        # MIXED_DOMAIN (index 7)
        assert labels[0, 0, 7] > 0.3, "(x&y)+(x^y) should have some mixed domain signal"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUAcceleration:
    """Test GPU execution (if available)."""

    def test_gpu_execution(self):
        """Test GPU execution speed."""
        detector = FingerprintPropertyDetector(device='cuda')
        fp = torch.randn(64, 448, device='cuda')

        # Warmup
        _ = detector.detect(fp)

        # Time execution
        import time
        start = time.time()
        for _ in range(100):
            labels = detector.detect(fp)
            torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"\nGPU execution time: {avg_time:.2f}ms per batch (B=64)")

        # Target: <0.5ms per batch
        # Relaxed to <2ms for testing (depends on GPU)
        assert avg_time < 2.0, f"GPU execution too slow: {avg_time:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
