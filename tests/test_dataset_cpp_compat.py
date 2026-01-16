"""
Tests for C++ generator format compatibility in dataset classes.

Tests field name normalization, pre-computed fingerprint loading,
and augmentation blocking pre-computed data.
"""

import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import torch

# Import constants directly (no heavy dependencies)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from constants import FINGERPRINT_DIM


# Standalone helper functions - copy logic to avoid import chain
def _normalize_item(item):
    """
    Normalize field names from C++ generator format to internal format.
    (Copied from dataset.py for standalone testing)
    """
    normalized = item.copy()

    if not normalized.get('obfuscated'):
        normalized['obfuscated'] = normalized.get('obfuscated_expr')
    if not normalized.get('simplified'):
        normalized['simplified'] = normalized.get('ground_truth_expr')

    if 'ast' not in normalized and 'ast_v2' in normalized:
        normalized['ast'] = normalized['ast_v2']

    return normalized


def _validate_precomputed_fingerprint(fp):
    """
    Validate pre-computed fingerprint dimension and numeric values.
    (Copied from dataset.py for standalone testing)
    """
    if len(fp) != FINGERPRINT_DIM:
        raise ValueError(
            f"Pre-computed fingerprint has {len(fp)} dims, expected {FINGERPRINT_DIM}"
        )

    fp_array = np.array(fp, dtype=np.float32)
    if not np.all(np.isfinite(fp_array)):
        raise ValueError(
            f"Pre-computed fingerprint contains NaN or inf values"
        )

    return fp_array


class TestFieldNormalization:
    """Tests for _normalize_item() helper function."""

    def test_normalize_item_cpp_format(self):
        """C++ field names normalized to internal format."""
        item = {
            'obfuscated_expr': 'x & y',
            'ground_truth_expr': 'x & y',
            'ast_v2': {'version': 2, 'nodes': [], 'edges': []},
            'fingerprint': {'flat': [0.0] * 448},
        }

        normalized = _normalize_item(item)

        assert normalized['obfuscated'] == 'x & y'
        assert normalized['simplified'] == 'x & y'
        assert normalized['ast'] == item['ast_v2']

    def test_normalize_item_legacy_format(self):
        """Legacy field names preserved."""
        item = {
            'obfuscated': 'x | y',
            'simplified': 'x | y',
            'depth': 1,
        }

        normalized = _normalize_item(item)

        assert normalized['obfuscated'] == 'x | y'
        assert normalized['simplified'] == 'x | y'

    def test_normalize_item_mixed_format(self):
        """Mixed format uses first available (internal takes precedence)."""
        item = {
            'obfuscated': 'a',
            'obfuscated_expr': 'b',
            'simplified': 'a',
        }

        normalized = _normalize_item(item)

        # Internal name takes precedence
        assert normalized['obfuscated'] == 'a'

    def test_normalize_item_empty_string_fallback(self):
        """Empty string falls back to C++ field name."""
        item = {
            'obfuscated': '',
            'obfuscated_expr': 'x & y',
            'simplified': 'x & y',
        }

        normalized = _normalize_item(item)

        # Should use C++ value when internal is empty
        assert normalized['obfuscated'] == 'x & y'


class TestFingerprintValidation:
    """Tests for _validate_precomputed_fingerprint() helper function."""

    def test_valid_fingerprint(self):
        """Valid fingerprint passes validation."""
        fp = [0.5] * FINGERPRINT_DIM
        result = _validate_precomputed_fingerprint(fp)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == FINGERPRINT_DIM
        assert result[0] == pytest.approx(0.5)

    def test_wrong_dimension_raises(self):
        """Wrong dimension raises ValueError."""
        fp = [0.0] * 100

        with pytest.raises(ValueError, match=f"expected {FINGERPRINT_DIM}"):
            _validate_precomputed_fingerprint(fp)

    def test_nan_raises(self):
        """NaN in fingerprint raises ValueError."""
        fp = [0.0] * FINGERPRINT_DIM
        fp[100] = float('nan')

        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_precomputed_fingerprint(fp)

    def test_inf_raises(self):
        """Inf in fingerprint raises ValueError."""
        fp = [0.0] * FINGERPRINT_DIM
        fp[50] = float('inf')

        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_precomputed_fingerprint(fp)

    def test_negative_inf_raises(self):
        """Negative inf in fingerprint raises ValueError."""
        fp = [0.0] * FINGERPRINT_DIM
        fp[50] = float('-inf')

        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_precomputed_fingerprint(fp)


# Integration tests require full dependencies - skip if unavailable
try:
    from src.data.dataset import (
        MBADataset, ContrastiveDataset, ScaledMBADataset,
    )
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Dataset dependencies not available (torch_scatter, etc.)"
)
class TestPrecomputedFingerprint:
    """Tests for pre-computed fingerprint loading in dataset classes."""

    @pytest.fixture
    def cpp_data_file(self, tmp_path):
        """Create test data in C++ generator format."""
        data = [
            {
                'obfuscated_expr': 'x & y',
                'ground_truth_expr': 'x & y',
                'depth': 1,
                'fingerprint': {'flat': [0.5] * 448},
            },
        ]
        path = tmp_path / 'cpp_data.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(path)

    @pytest.fixture
    def mock_tokenizer(self):
        mock = MagicMock()
        mock.encode.return_value = [1, 2, 3]
        mock.get_source_tokens.return_value = [4, 5]
        return mock

    @pytest.fixture
    def mock_fingerprint(self):
        mock = MagicMock()
        mock.compute.return_value = np.zeros(448, dtype=np.float32)
        return mock

    def test_load_precomputed_fingerprint(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """Pre-computed fingerprint loaded instead of computed."""
        dataset = MBADataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        sample = dataset[0]

        # Fingerprint should be 0.5 (from JSON), not 0.0 (from mock)
        assert sample['fingerprint'][0].item() == pytest.approx(0.5)
        mock_fingerprint.compute.assert_not_called()

    def test_fallback_to_computed(self, tmp_path, mock_tokenizer, mock_fingerprint):
        """Fall back to computed fingerprint when not in JSON."""
        data = [{'obfuscated': 'x', 'simplified': 'x', 'depth': 1}]
        path = tmp_path / 'legacy.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        mock_fingerprint.compute.return_value = np.ones(448, dtype=np.float32)

        dataset = MBADataset(
            str(path),
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        sample = dataset[0]

        assert sample['fingerprint'][0].item() == pytest.approx(1.0)
        mock_fingerprint.compute.assert_called_once()

    def test_augmentation_blocks_precomputed(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """Variable augmentation forces fingerprint recomputation."""
        mock_fingerprint.compute.return_value = np.full(448, 0.9, dtype=np.float32)

        dataset = MBADataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=True,
            augment_prob=0.0,
        )

        sample = dataset[0]

        # Even with pre-computed available, augmentation forces compute
        assert sample['fingerprint'][0].item() == pytest.approx(0.9)
        mock_fingerprint.compute.assert_called_once()

    def test_invalid_fingerprint_dimension_raises(self, tmp_path, mock_tokenizer, mock_fingerprint):
        """Wrong fingerprint dimension raises ValueError."""
        data = [{
            'obfuscated_expr': 'x',
            'ground_truth_expr': 'x',
            'fingerprint': {'flat': [0.0] * 100},
        }]
        path = tmp_path / 'bad_fp.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        dataset = MBADataset(
            str(path),
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        with pytest.raises(ValueError, match=f"expected {FINGERPRINT_DIM}"):
            _ = dataset[0]


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Dataset dependencies not available (torch_scatter, etc.)"
)
class TestContrastiveDatasetCppCompat:
    """Tests for ContrastiveDataset C++ compatibility."""

    @pytest.fixture
    def cpp_data_file(self, tmp_path):
        """Create test data in C++ generator format."""
        data = [
            {
                'obfuscated_expr': 'x & y',
                'ground_truth_expr': 'x | y',
                'depth': 1,
                'fingerprint': {'flat': [0.5] * 448},
            },
        ]
        path = tmp_path / 'cpp_contrastive.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(path)

    @pytest.fixture
    def mock_tokenizer(self):
        return MagicMock()

    @pytest.fixture
    def mock_fingerprint(self):
        mock = MagicMock()
        mock.compute.return_value = np.ones(448, dtype=np.float32)
        return mock

    def test_load_cpp_format(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """ContrastiveDataset loads C++ format correctly."""
        dataset = ContrastiveDataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        assert len(dataset) == 1
        sample = dataset[0]

        assert 'obfuscated' in sample
        assert 'simplified' in sample


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Dataset dependencies not available (torch_scatter, etc.)"
)
class TestScaledMBADatasetCppCompat:
    """Tests for ScaledMBADataset C++ compatibility."""

    @pytest.fixture
    def cpp_data_file(self, tmp_path):
        """Create test data in C++ generator format."""
        data = [
            {
                'obfuscated_expr': 'x & y',
                'ground_truth_expr': 'x | y',
                'depth': 1,
                'fingerprint': {'flat': [0.5] * 448},
                'complexity_score': 0.3,
                'boolean_domain_only': True,
            },
        ]
        path = tmp_path / 'cpp_scaled.jsonl'
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(path)

    @pytest.fixture
    def mock_tokenizer(self):
        mock = MagicMock()
        mock.encode.return_value = [1, 2, 3]
        mock.get_source_tokens.return_value = [4, 5]
        return mock

    @pytest.fixture
    def mock_fingerprint(self):
        mock = MagicMock()
        mock.compute.return_value = np.ones(448, dtype=np.float32)
        return mock

    def test_precomputed_with_augmentation_disabled(self, cpp_data_file, mock_tokenizer, mock_fingerprint):
        """Pre-computed fingerprint used when augmentation disabled."""
        dataset = ScaledMBADataset(
            cpp_data_file,
            mock_tokenizer,
            mock_fingerprint,
            node_type_schema='current',
            augment_variables=False,
        )

        sample = dataset[0]

        assert sample['fingerprint'][0].item() == pytest.approx(0.5)
        mock_fingerprint.compute.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
