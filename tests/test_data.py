"""
Tests for data pipeline modules.

Tests AST parser, tokenizer, fingerprint, and dataset classes.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

from src.data.ast_parser import (
    parse_to_ast,
    ast_to_graph,
    expr_to_graph,
    ASTNode,
)
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.dataset import MBADataset, ContrastiveDataset
from src.data.collate import collate_graphs, collate_contrastive
from src.constants import (
    NODE_TYPES,
    EDGE_TYPES,
    FINGERPRINT_DIM,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
)


class TestASTParser:
    """Tests for AST parser."""

    def test_parse_simple_addition(self):
        """Test parsing simple addition."""
        ast = parse_to_ast("x + y")
        assert ast.type == "ADD"
        assert len(ast.children) == 2
        assert ast.children[0].type == "VAR"
        assert ast.children[0].value == "x"
        assert ast.children[1].type == "VAR"
        assert ast.children[1].value == "y"

    def test_parse_with_constants(self):
        """Test parsing with constants."""
        ast = parse_to_ast("x + 5")
        assert ast.type == "ADD"
        assert ast.children[0].type == "VAR"
        assert ast.children[1].type == "CONST"
        assert ast.children[1].value == "5"

    def test_parse_nested_expression(self):
        """Test parsing nested expression."""
        ast = parse_to_ast("(x & y) + (x ^ y)")
        assert ast.type == "ADD"
        assert ast.children[0].type == "AND"
        assert ast.children[1].type == "XOR"

    def test_parse_unary_not(self):
        """Test parsing unary NOT operator."""
        ast = parse_to_ast("~x")
        assert ast.type == "NOT"
        assert len(ast.children) == 1
        assert ast.children[0].type == "VAR"

    def test_parse_operator_precedence(self):
        """Test operator precedence."""
        ast = parse_to_ast("x + y * 2")
        assert ast.type == "ADD"
        assert ast.children[0].type == "VAR"
        assert ast.children[1].type == "MUL"

    def test_parse_with_parentheses(self):
        """Test that parentheses override precedence."""
        ast = parse_to_ast("(x + y) * 2")
        assert ast.type == "MUL"
        assert ast.children[0].type == "ADD"

    def test_ast_to_graph_simple(self):
        """Test AST to graph conversion."""
        ast = parse_to_ast("x + y")
        graph = ast_to_graph(ast)

        assert graph.x.shape[1] == 32  # NODE_DIM
        assert graph.x.shape[0] == 3  # 3 nodes: +, x, y
        assert graph.edge_index.shape[0] == 2
        assert len(graph.edge_type) == graph.edge_index.shape[1]

    def test_edge_types_correctness(self):
        """Test that edge types are correctly assigned."""
        ast = parse_to_ast("x + y")
        graph = ast_to_graph(ast)

        # Should have CHILD_LEFT, CHILD_RIGHT, and PARENT edges
        edge_types = graph.edge_type.tolist()
        assert EDGE_TYPES["CHILD_LEFT"] in edge_types
        assert EDGE_TYPES["CHILD_RIGHT"] in edge_types
        assert EDGE_TYPES["PARENT"] in edge_types

    def test_same_var_edges(self):
        """Test that SAME_VAR edges connect multiple uses of same variable."""
        ast = parse_to_ast("x + x")
        graph = ast_to_graph(ast)

        # Should have SAME_VAR edges connecting the two x nodes
        edge_types = graph.edge_type.tolist()
        assert EDGE_TYPES["SAME_VAR"] in edge_types

    def test_expr_to_graph_convenience(self):
        """Test expr_to_graph convenience function."""
        graph = expr_to_graph("x & y")
        assert graph.x.shape[0] == 3  # 3 nodes


class TestTokenizer:
    """Tests for tokenizer."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = MBATokenizer()
        tokens = tokenizer.tokenize("x + y")
        assert tokens == ["x", "+", "y"]

    def test_tokenize_with_parens(self):
        """Test tokenization with parentheses."""
        tokenizer = MBATokenizer()
        tokens = tokenizer.tokenize("(x & y)")
        assert tokens == ["(", "x", "&", "y", ")"]

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        tokenizer = MBATokenizer()
        expr = "x + y"
        ids = tokenizer.encode(expr, add_special=True)
        decoded = tokenizer.decode(ids, skip_special=True)

        # Normalize spacing for comparison
        assert decoded.replace(" ", "") == expr.replace(" ", "")

    def test_special_tokens(self):
        """Test special tokens in encoding."""
        tokenizer = MBATokenizer()
        ids = tokenizer.encode("x", add_special=True)

        assert ids[0] == SOS_IDX
        assert ids[-1] == EOS_IDX

    def test_no_special_tokens(self):
        """Test encoding without special tokens."""
        tokenizer = MBATokenizer()
        ids = tokenizer.encode("x + y", add_special=False)

        assert SOS_IDX not in ids
        assert EOS_IDX not in ids

    def test_vocabulary_coverage(self):
        """Test that vocabulary covers all necessary tokens."""
        tokenizer = MBATokenizer()

        # Test operators
        for op in ["+", "-", "*", "&", "|", "^", "~"]:
            assert op in tokenizer.token2id

        # Test variables
        for i in range(8):
            assert f"x{i}" in tokenizer.token2id

        # Test constants
        for i in [0, 1, 10, 100, 255]:
            assert str(i) in tokenizer.token2id

    def test_get_source_tokens(self):
        """Test get_source_tokens for copy mechanism."""
        tokenizer = MBATokenizer()
        tokens = tokenizer.get_source_tokens("x + 1")

        assert SOS_IDX not in tokens
        assert EOS_IDX not in tokens

    def test_save_load(self):
        """Test save/load vocabulary."""
        tokenizer = MBATokenizer()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            tokenizer.save(temp_path)

            new_tokenizer = MBATokenizer()
            new_tokenizer.load(temp_path)

            assert tokenizer.token2id == new_tokenizer.token2id
            assert tokenizer.id2token == new_tokenizer.id2token
        finally:
            Path(temp_path).unlink()


class TestSemanticFingerprint:
    """Tests for semantic fingerprint."""

    def test_fingerprint_dimensions(self):
        """Test that fingerprint has correct dimensions."""
        fp = SemanticFingerprint()
        result = fp.compute("x + y")

        assert result.shape == (FINGERPRINT_DIM,)
        assert result.dtype == np.float32

    def test_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic."""
        fp = SemanticFingerprint(seed=42)
        result1 = fp.compute("x + y")

        fp2 = SemanticFingerprint(seed=42)
        result2 = fp2.compute("x + y")

        np.testing.assert_array_equal(result1, result2)

    def test_fingerprint_different_for_different_exprs(self):
        """Test that different expressions have different fingerprints."""
        fp = SemanticFingerprint()
        result1 = fp.compute("x + y")
        result2 = fp.compute("x * y")

        # Should be different
        assert not np.allclose(result1, result2)

    def test_truth_table_correctness(self):
        """Test truth table computation for simple expressions."""
        fp = SemanticFingerprint()

        # Test simple OR operation
        result_or = fp.compute("x | y")
        # Truth table for OR (2 variables): 0|0=0, 0|1=1, 1|0=1, 1|1=1
        # Extract truth table portion (last 64 dims)
        truth_table = result_or[-64:]

        # For x | y with x at bit 0, y at bit 1:
        # index 0: x=0, y=0 -> 0
        # index 1: x=1, y=0 -> 1
        # index 2: x=0, y=1 -> 1
        # index 3: x=1, y=1 -> 1
        assert truth_table[0] == 0.0  # 0 | 0 = 0
        assert truth_table[1] == 1.0  # 1 | 0 = 1
        assert truth_table[2] == 1.0  # 0 | 1 = 1
        assert truth_table[3] == 1.0  # 1 | 1 = 1

    def test_truth_table_and(self):
        """Test truth table for AND operation."""
        fp = SemanticFingerprint()
        result_and = fp.compute("x & y")
        truth_table = result_and[-64:]

        # AND truth table
        assert truth_table[0] == 0.0  # 0 & 0 = 0
        assert truth_table[1] == 0.0  # 1 & 0 = 0
        assert truth_table[2] == 0.0  # 0 & 1 = 0
        assert truth_table[3] == 1.0  # 1 & 1 = 1

    def test_truth_table_xor(self):
        """Test truth table for XOR operation."""
        fp = SemanticFingerprint()
        result_xor = fp.compute("x ^ y")
        truth_table = result_xor[-64:]

        # XOR truth table
        assert truth_table[0] == 0.0  # 0 ^ 0 = 0
        assert truth_table[1] == 1.0  # 1 ^ 0 = 1
        assert truth_table[2] == 1.0  # 0 ^ 1 = 1
        assert truth_table[3] == 0.0  # 1 ^ 1 = 0

    def test_symbolic_features_extraction(self):
        """Test that symbolic features are non-zero for non-trivial expressions."""
        fp = SemanticFingerprint()
        result = fp.compute("(x & y) + (x ^ y)")

        # Symbolic features are first 32 dims
        symbolic = result[:32]

        # Should have non-zero features
        assert np.sum(symbolic > 0) > 0


class TestDataset:
    """Tests for dataset classes."""

    def create_temp_dataset(self, num_samples=10):
        """Create temporary dataset file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".jsonl"
        )

        for i in range(num_samples):
            item = {
                "obfuscated": f"(x{i % 8} & y) + (x{i % 8} ^ y)",
                "simplified": f"x{i % 8} | y",
                "depth": i % 5,
            }
            temp_file.write(json.dumps(item) + "\n")

        temp_file.close()
        return temp_file.name

    def test_dataset_loading(self):
        """Test dataset loading."""
        data_path = self.create_temp_dataset(5)

        try:
            tokenizer = MBATokenizer()
            fingerprint = SemanticFingerprint()
            dataset = MBADataset(data_path, tokenizer, fingerprint)

            assert len(dataset) == 5
        finally:
            Path(data_path).unlink()

    def test_dataset_depth_filter(self):
        """Test depth filtering in dataset."""
        data_path = self.create_temp_dataset(10)

        try:
            tokenizer = MBATokenizer()
            fingerprint = SemanticFingerprint()
            dataset = MBADataset(data_path, tokenizer, fingerprint, max_depth=2)

            # Should filter to only items with depth <= 2
            assert len(dataset) < 10
            for i in range(len(dataset)):
                item = dataset[i]
                assert item["depth"] <= 2
        finally:
            Path(data_path).unlink()

    def test_dataset_item_structure(self):
        """Test dataset item structure."""
        data_path = self.create_temp_dataset(5)

        try:
            tokenizer = MBATokenizer()
            fingerprint = SemanticFingerprint()
            dataset = MBADataset(data_path, tokenizer, fingerprint)

            item = dataset[0]

            # Check all required fields
            assert "graph" in item
            assert "fingerprint" in item
            assert "target_ids" in item
            assert "source_tokens" in item
            assert "depth" in item
            assert "obfuscated" in item
            assert "simplified" in item

            # Check types and shapes
            assert isinstance(item["fingerprint"], torch.Tensor)
            assert item["fingerprint"].shape[0] == FINGERPRINT_DIM
            assert isinstance(item["target_ids"], torch.Tensor)
            assert isinstance(item["source_tokens"], torch.Tensor)
        finally:
            Path(data_path).unlink()

    def test_contrastive_dataset(self):
        """Test contrastive dataset."""
        data_path = self.create_temp_dataset(5)

        try:
            tokenizer = MBATokenizer()
            fingerprint = SemanticFingerprint()
            dataset = ContrastiveDataset(data_path, tokenizer, fingerprint)

            assert len(dataset) == 5

            item = dataset[0]
            assert "obf_graph" in item
            assert "simp_graph" in item
            assert "obf_fingerprint" in item
            assert "simp_fingerprint" in item
            assert "label" in item
        finally:
            Path(data_path).unlink()


class TestCollate:
    """Tests for collate functions."""

    def create_mock_batch(self, batch_size=4):
        """Create mock batch for testing."""
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()

        batch = []
        for i in range(batch_size):
            expr = f"x{i} + y"
            graph = expr_to_graph(expr)
            fp = torch.from_numpy(fingerprint.compute(expr))
            target_ids = torch.tensor(tokenizer.encode(expr, add_special=True))
            source_tokens = torch.tensor(tokenizer.get_source_tokens(expr))

            batch.append(
                {
                    "graph": graph,
                    "fingerprint": fp,
                    "target_ids": target_ids,
                    "source_tokens": source_tokens,
                    "depth": i,
                    "obfuscated": expr,
                    "simplified": expr,
                }
            )

        return batch

    def test_collate_graphs(self):
        """Test collate_graphs function."""
        batch = self.create_mock_batch(4)
        collated = collate_graphs(batch)

        assert "graph_batch" in collated
        assert "fingerprint" in collated
        assert "target_ids" in collated
        assert "target_lengths" in collated

        # Check shapes
        assert collated["fingerprint"].shape[0] == 4
        assert collated["fingerprint"].shape[1] == FINGERPRINT_DIM
        assert collated["target_ids"].shape[0] == 4
        assert collated["target_lengths"].shape[0] == 4

    def test_collate_padding(self):
        """Test that sequences are properly padded."""
        batch = self.create_mock_batch(4)
        collated = collate_graphs(batch)

        target_ids = collated["target_ids"]

        # All sequences should have same length (padded)
        assert target_ids.shape[0] == 4  # batch size

        # Padding should use PAD_IDX
        # Some positions should be padded (unless all sequences have same length)
        max_len = collated["target_lengths"].max().item()
        assert target_ids.shape[1] == max_len

    def test_collate_contrastive(self):
        """Test collate_contrastive function."""
        tokenizer = MBATokenizer()
        fingerprint = SemanticFingerprint()

        batch = []
        for i in range(4):
            obf_expr = f"(x{i} & y) + (x{i} ^ y)"
            simp_expr = f"x{i} | y"

            batch.append(
                {
                    "obf_graph": expr_to_graph(obf_expr),
                    "simp_graph": expr_to_graph(simp_expr),
                    "obf_fingerprint": torch.from_numpy(fingerprint.compute(obf_expr)),
                    "simp_fingerprint": torch.from_numpy(fingerprint.compute(simp_expr)),
                    "label": i,
                    "obfuscated": obf_expr,
                    "simplified": simp_expr,
                }
            )

        collated = collate_contrastive(batch)

        assert "obf_graph_batch" in collated
        assert "simp_graph_batch" in collated
        assert "obf_fingerprint" in collated
        assert "simp_fingerprint" in collated
        assert "labels" in collated

        assert collated["obf_fingerprint"].shape[0] == 4
        assert collated["labels"].shape[0] == 4
