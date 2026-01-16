"""
Tests for dataset schema version detection and node type conversion.
"""

import pytest
import torch
from src.models.edge_types import (
    convert_legacy_node_types,
    LEGACY_NODE_MAP,
    LEGACY_NODE_ORDER,
)


class TestConvertLegacyNodeTypes:
    """Tests for convert_legacy_node_types function."""

    def test_conversion_correctness(self):
        """Test that legacy node type IDs are correctly mapped to current IDs."""
        # Legacy: ADD=0, SUB=1, MUL=2, NEG=3, AND=4, OR=5, XOR=6, NOT=7, VAR=8, CONST=9
        # Current: VAR=0, CONST=1, ADD=2, SUB=3, MUL=4, AND=5, OR=6, XOR=7, NOT=8, NEG=9
        legacy_ids = torch.tensor([8, 9, 0, 1, 4, 5])  # VAR, CONST, ADD, SUB, AND, OR (legacy)
        current_ids = convert_legacy_node_types(legacy_ids)
        expected = torch.tensor([0, 1, 2, 3, 5, 6])  # VAR, CONST, ADD, SUB, AND, OR (current)
        assert torch.equal(current_ids, expected), f"Expected {expected}, got {current_ids}"

    def test_all_node_types(self):
        """Test conversion for all 10 node types."""
        legacy_ids = torch.tensor(list(range(10)))
        current_ids = convert_legacy_node_types(legacy_ids)

        # Verify each mapping
        for legacy_id in range(10):
            expected_current = LEGACY_NODE_MAP[legacy_id]
            actual_current = current_ids[legacy_id].item()
            assert actual_current == expected_current, \
                f"Legacy {legacy_id} should map to {expected_current}, got {actual_current}"

    def test_empty_tensor(self):
        """Test that empty tensor passes through unchanged."""
        empty = torch.tensor([], dtype=torch.long)
        result = convert_legacy_node_types(empty)
        assert result.numel() == 0

    def test_out_of_range_raises(self):
        """Test that out-of-range IDs raise ValueError."""
        invalid = torch.tensor([10, 11])
        with pytest.raises(ValueError, match="must be in \\[0-9\\] range"):
            convert_legacy_node_types(invalid)

    def test_negative_ids_raise(self):
        """Test that negative IDs raise ValueError."""
        invalid = torch.tensor([-1, 0, 1])
        with pytest.raises(ValueError, match="must be in \\[0-9\\] range"):
            convert_legacy_node_types(invalid)

    def test_preserves_dtype(self):
        """Test that output dtype matches input dtype."""
        legacy_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
        current_ids = convert_legacy_node_types(legacy_ids)
        assert current_ids.dtype == legacy_ids.dtype

    def test_preserves_device(self):
        """Test that output device matches input device."""
        legacy_ids = torch.tensor([0, 1, 2])
        current_ids = convert_legacy_node_types(legacy_ids)
        assert current_ids.device == legacy_ids.device

    def test_vectorized_performance(self):
        """Test that vectorized conversion handles large tensors efficiently."""
        large_tensor = torch.randint(0, 10, (100000,))
        result = convert_legacy_node_types(large_tensor)
        assert result.shape == large_tensor.shape


class TestLegacyNodeMap:
    """Tests for LEGACY_NODE_MAP consistency."""

    def test_map_completeness(self):
        """Test that LEGACY_NODE_MAP covers all 10 node types."""
        assert len(LEGACY_NODE_MAP) == 10

    def test_map_values_in_range(self):
        """Test that all mapped values are in valid range."""
        for legacy_id, current_id in LEGACY_NODE_MAP.items():
            assert 0 <= legacy_id <= 9, f"Invalid legacy ID: {legacy_id}"
            assert 0 <= current_id <= 9, f"Invalid current ID: {current_id}"

    def test_map_is_bijective(self):
        """Test that mapping is one-to-one (bijective)."""
        values = list(LEGACY_NODE_MAP.values())
        assert len(values) == len(set(values)), "Mapping is not bijective"

    def test_legacy_order_consistency(self):
        """Test that LEGACY_NODE_ORDER matches LEGACY_NODE_MAP."""
        assert len(LEGACY_NODE_ORDER) == 10
        for i, name in enumerate(LEGACY_NODE_ORDER):
            assert i in LEGACY_NODE_MAP, f"Missing legacy ID {i} in LEGACY_NODE_MAP"


class TestDatasetSchemaValidation:
    """Tests for dataset schema validation behavior."""

    def test_node_type_schema_required(self):
        """Test that missing node_type_schema raises ValueError."""
        # This test requires actual dataset classes
        pytest.importorskip("src.data.dataset")
        from src.data.dataset import MBADataset
        from src.data.tokenizer import MBATokenizer
        from src.data.fingerprint import SemanticFingerprint

        with pytest.raises(ValueError, match="node_type_schema is REQUIRED"):
            MBADataset(
                data_path="nonexistent.jsonl",
                tokenizer=MBATokenizer(),
                fingerprint=SemanticFingerprint(),
                node_type_schema=None
            )

    def test_invalid_schema_value_raises(self):
        """Test that invalid node_type_schema value raises ValueError."""
        pytest.importorskip("src.data.dataset")
        from src.data.dataset import MBADataset
        from src.data.tokenizer import MBATokenizer
        from src.data.fingerprint import SemanticFingerprint

        with pytest.raises(ValueError, match="must be 'legacy' or 'current'"):
            MBADataset(
                data_path="nonexistent.jsonl",
                tokenizer=MBATokenizer(),
                fingerprint=SemanticFingerprint(),
                node_type_schema="invalid"
            )


class TestConfigValidation:
    """Tests for config validation utilities."""

    def test_validate_config_value_int(self):
        """Test integer value extraction and validation."""
        from src.utils.config import validate_config_value

        config = {'model': {'hidden_dim': 256}}
        assert validate_config_value(config, 'model.hidden_dim', int) == 256

    def test_validate_config_value_bool(self):
        """Test boolean value extraction and validation."""
        from src.utils.config import validate_config_value

        config = {'model': {'use_flag': True}}
        assert validate_config_value(config, 'model.use_flag', bool) is True

    def test_validate_config_value_default(self):
        """Test default value when key missing."""
        from src.utils.config import validate_config_value

        config = {'model': {}}
        assert validate_config_value(config, 'model.missing', int, default=10) == 10

    def test_validate_config_value_type_mismatch(self):
        """Test type mismatch raises ValueError."""
        from src.utils.config import validate_config_value

        config = {'model': {'hidden_dim': "256"}}  # String instead of int
        with pytest.raises(ValueError, match="wrong type"):
            validate_config_value(config, 'model.hidden_dim', int)

    def test_validate_config_value_required_missing(self):
        """Test missing required value raises ValueError."""
        from src.utils.config import validate_config_value

        config = {'model': {}}
        with pytest.raises(ValueError, match="not found"):
            validate_config_value(config, 'model.hidden_dim', int, required=True)
