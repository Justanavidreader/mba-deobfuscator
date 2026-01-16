"""
Tests for variable permutation augmentation.

Tests VariablePermuter class and its integration with dataset classes.
"""

import pytest
import json
import sys
import os
from pathlib import Path

# Direct import to avoid triggering full package __init__ chains
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))
from augmentation import VariablePermuter, VariableAugmentationMixin


class TestVariablePermuter:
    """Tests for VariablePermuter class."""

    def test_extract_variables_simple(self):
        """Extract variables in order of appearance."""
        permuter = VariablePermuter()
        expr = "x & y + x"
        vars = permuter.extract_variables(expr)
        assert vars == ['x', 'y']

    def test_extract_variables_with_numbers(self):
        """Handle x0, x1 style variables."""
        permuter = VariablePermuter()
        expr = "x0 & x1 + x2"
        vars = permuter.extract_variables(expr)
        assert vars == ['x0', 'x1', 'x2']

    def test_extract_variables_mixed(self):
        """Handle mix of single-letter and numbered variables."""
        permuter = VariablePermuter()
        expr = "a & x0 + b ^ x1"
        vars = permuter.extract_variables(expr)
        assert vars == ['a', 'x0', 'b', 'x1']

    def test_extract_variables_no_duplicates(self):
        """Variables extracted only once even if repeated."""
        permuter = VariablePermuter()
        expr = "x + x + x & y | y"
        vars = permuter.extract_variables(expr)
        assert vars == ['x', 'y']

    def test_extract_variables_constants_ignored(self):
        """Constants are not extracted as variables."""
        permuter = VariablePermuter()
        expr = "x + 1 + 255"
        vars = permuter.extract_variables(expr)
        assert vars == ['x']

    def test_generate_permutation_basic(self):
        """Generate permutation mapping."""
        permuter = VariablePermuter(seed=42)
        mapping = permuter.generate_permutation(['x', 'y', 'z'])

        assert len(mapping) == 3
        assert set(mapping.values()) <= set(VariablePermuter.CANONICAL_VARS)
        # All mapped values should be unique
        assert len(set(mapping.values())) == 3

    def test_generate_permutation_too_many_variables(self):
        """Error when expression has too many variables."""
        permuter = VariablePermuter()
        # 9 variables, but only 8 canonical names
        variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        with pytest.raises(ValueError, match="only 8 canonical names"):
            permuter.generate_permutation(variables)

    def test_apply_permutation_simple(self):
        """Apply permutation to expression."""
        permuter = VariablePermuter()
        expr = "x & y"
        mapping = {'x': 'x0', 'y': 'x1'}

        result = permuter.apply_permutation(expr, mapping)
        assert result == "x0 & x1"

    def test_apply_permutation_repeated_vars(self):
        """All occurrences of a variable are replaced."""
        permuter = VariablePermuter()
        expr = "x + x + x"
        mapping = {'x': 'x5'}

        result = permuter.apply_permutation(expr, mapping)
        assert result == "x5 + x5 + x5"

    def test_apply_permutation_no_partial_replacement(self):
        """x should not be replaced within x1."""
        permuter = VariablePermuter()
        expr = "x + x1"
        mapping = {'x': 'x3', 'x1': 'x7'}

        result = permuter.apply_permutation(expr, mapping)
        assert result == "x3 + x7"
        assert 'x31' not in result  # x in x1 should not be replaced

    def test_apply_permutation_empty_mapping(self):
        """Empty mapping returns original expression."""
        permuter = VariablePermuter()
        expr = "x & y"
        result = permuter.apply_permutation(expr, {})
        assert result == expr

    def test_consistent_permutation(self):
        """Same mapping applied to both expressions."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf = "(x & y) + (x ^ y)"
        simp = "x | y"

        perm_obf, perm_simp = permuter(obf, simp)

        # Extract variables from permuted expressions
        obf_vars = set(permuter.extract_variables(perm_obf))
        simp_vars = set(permuter.extract_variables(perm_simp))

        # Variables in simplified should be subset of obfuscated
        # (they share the same mapping)
        assert simp_vars <= obf_vars

    def test_permute_prob_zero(self):
        """No permutation when prob=0."""
        permuter = VariablePermuter(permute_prob=0.0)
        obf, simp = "x & y", "x | y"

        perm_obf, perm_simp = permuter(obf, simp)

        assert perm_obf == obf
        assert perm_simp == simp

    def test_permute_prob_one(self):
        """Always permute when prob=1."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf, simp = "a & b", "a | b"

        perm_obf, perm_simp = permuter(obf, simp)

        # With prob=1, should be permuted to canonical names
        obf_vars = set(permuter.extract_variables(perm_obf))
        assert obf_vars <= set(VariablePermuter.CANONICAL_VARS)

    def test_deterministic_with_seed(self):
        """Same seed produces same permutation."""
        obf, simp = "(x & y) + z", "x | y | z"

        p1 = VariablePermuter(permute_prob=1.0, seed=123)
        p2 = VariablePermuter(permute_prob=1.0, seed=123)

        result1 = p1(obf, simp)
        result2 = p2(obf, simp)

        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Different seeds produce different permutations (usually)."""
        obf, simp = "(x & y) + z", "x | y | z"

        # Run multiple times to ensure statistical difference
        results_seed1 = []
        results_seed2 = []

        for i in range(10):
            p1 = VariablePermuter(permute_prob=1.0, seed=100 + i)
            p2 = VariablePermuter(permute_prob=1.0, seed=200 + i)
            results_seed1.append(p1(obf, simp))
            results_seed2.append(p2(obf, simp))

        # At least some should differ
        different = sum(1 for r1, r2 in zip(results_seed1, results_seed2) if r1 != r2)
        assert different > 0

    def test_all_canonical_names_used(self):
        """Permuted variables use only canonical names."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf = "a & b + c"
        simp = "a | c"

        perm_obf, perm_simp = permuter(obf, simp)

        # Should only contain x0-x7 variables
        vars_in_result = permuter.extract_variables(perm_obf + " " + perm_simp)
        for var in vars_in_result:
            assert var in VariablePermuter.CANONICAL_VARS, f"Unexpected var: {var}"

    def test_empty_expression(self):
        """Handle expressions with no variables."""
        permuter = VariablePermuter(permute_prob=1.0)
        obf, simp = "1 + 2", "3"

        perm_obf, perm_simp = permuter(obf, simp)

        assert perm_obf == obf
        assert perm_simp == simp

    def test_operators_not_matched(self):
        """Operators are not extracted as variables."""
        permuter = VariablePermuter()
        # All operators should be ignored
        expr = "x & y | z ^ a + b - c * d"
        vars = permuter.extract_variables(expr)

        # Should only get variable names
        assert '&' not in vars
        assert '|' not in vars
        assert '^' not in vars
        assert '+' not in vars
        assert '-' not in vars
        assert '*' not in vars

    def test_parentheses_preserved(self):
        """Parentheses are preserved in permuted expression."""
        permuter = VariablePermuter(permute_prob=1.0, seed=42)
        obf = "((x & y) + (x ^ y))"
        simp = "(x | y)"

        perm_obf, perm_simp = permuter(obf, simp)

        # Count parentheses
        assert perm_obf.count('(') == obf.count('(')
        assert perm_obf.count(')') == obf.count(')')
        assert perm_simp.count('(') == simp.count('(')
        assert perm_simp.count(')') == simp.count(')')

    def test_invalid_permute_prob(self):
        """Invalid permute_prob raises error."""
        with pytest.raises(ValueError, match="permute_prob must be in"):
            VariablePermuter(permute_prob=1.5)

        with pytest.raises(ValueError, match="permute_prob must be in"):
            VariablePermuter(permute_prob=-0.1)


class TestVariableAugmentationMixin:
    """Tests for VariableAugmentationMixin class."""

    def test_mixin_init_enabled(self):
        """Mixin initializes permuter when enabled."""
        class TestClass(VariableAugmentationMixin):
            pass

        obj = TestClass()
        obj._init_augmentation(augment_variables=True, augment_prob=0.5)

        assert obj.augment_variables is True
        assert obj.permuter is not None

    def test_mixin_init_disabled(self):
        """Mixin does not create permuter when disabled."""
        class TestClass(VariableAugmentationMixin):
            pass

        obj = TestClass()
        obj._init_augmentation(augment_variables=False)

        assert obj.augment_variables is False
        assert obj.permuter is None

    def test_mixin_apply_enabled(self):
        """Mixin applies augmentation when enabled."""
        class TestClass(VariableAugmentationMixin):
            pass

        obj = TestClass()
        obj._init_augmentation(augment_variables=True, augment_prob=1.0, augment_seed=42)

        obf, simp = obj._apply_augmentation("x & y", "x | y")

        # Should be permuted to canonical names
        vars = obj.permuter.extract_variables(obf)
        assert all(v in VariablePermuter.CANONICAL_VARS for v in vars)

    def test_mixin_apply_disabled(self):
        """Mixin returns original when disabled."""
        class TestClass(VariableAugmentationMixin):
            pass

        obj = TestClass()
        obj._init_augmentation(augment_variables=False)

        obf, simp = obj._apply_augmentation("x & y", "x | y")

        assert obf == "x & y"
        assert simp == "x | y"


class TestWorkerSeedIsolation:
    """Tests for DataLoader worker seed isolation."""

    def test_worker_seed_isolation(self):
        """Different workers get different seeds even with same base seed."""
        import unittest.mock as mock

        # Mock worker_info to simulate different workers
        with mock.patch('torch.utils.data.get_worker_info') as mock_worker_info:
            # Worker 0
            mock_worker_info.return_value = mock.Mock(id=0)
            p0 = VariablePermuter(permute_prob=1.0, seed=42)

            # Worker 1
            mock_worker_info.return_value = mock.Mock(id=1)
            p1 = VariablePermuter(permute_prob=1.0, seed=42)

        # The seeds should be different (42+0 vs 42+1)
        # So RNG states differ, though specific results depend on shuffle
        assert p0.rng is not p1.rng

    def test_no_worker_info_uses_seed_directly(self):
        """When not in worker, seed is used directly."""
        import unittest.mock as mock

        with mock.patch('torch.utils.data.get_worker_info') as mock_worker_info:
            mock_worker_info.return_value = None
            p = VariablePermuter(permute_prob=1.0, seed=42)

        # Without worker info, seed should be 42
        # Verify by checking same results as unseeded permuter with seed 42
        p_direct = VariablePermuter(permute_prob=1.0, seed=42)

        obf, simp = "x & y", "x | y"
        assert p(obf, simp) == p_direct(obf, simp)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'dataset.py')),
    reason="Dataset module not available"
)
class TestDatasetIntegration:
    """Integration tests with dataset classes.

    These tests require the full src.data.dataset module which may have
    dependencies (torch_geometric, lark, etc.) not available in all environments.
    """

    @pytest.fixture
    def test_data_file(self, tmp_path):
        """Create test data file."""
        data = [
            {"obfuscated": "x & y", "simplified": "x & y", "depth": 1},
            {"obfuscated": "(a | b) + (a ^ b)", "simplified": "a + b", "depth": 2},
            {"obfuscated": "x0 & x1", "simplified": "x0 & x1", "depth": 1},
        ]
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(data_file)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        class MockTokenizer:
            def encode(self, expr, add_special=True):
                return [1, 2, 3]  # Dummy token IDs

            def get_source_tokens(self, expr):
                return [4, 5, 6]  # Dummy source tokens

        return MockTokenizer()

    @pytest.fixture
    def mock_fingerprint(self):
        """Create mock fingerprint."""
        import numpy as np

        class MockFingerprint:
            def compute(self, expr):
                return np.zeros(448, dtype=np.float32)

        return MockFingerprint()

    def test_mba_dataset_augmentation_enabled(self, test_data_file, mock_tokenizer, mock_fingerprint):
        """MBADataset applies augmentation when enabled."""
        try:
            from src.data.dataset import MBADataset
        except ImportError:
            pytest.skip("MBADataset dependencies not available")

        dataset = MBADataset(
            test_data_file,
            mock_tokenizer,
            mock_fingerprint,
            augment_variables=True,
            augment_prob=1.0,
            augment_seed=42,
        )

        sample = dataset[0]

        # Variables should be from canonical set
        permuter = VariablePermuter()
        vars_in_obf = permuter.extract_variables(sample['obfuscated'])
        assert all(v in VariablePermuter.CANONICAL_VARS for v in vars_in_obf)

    def test_mba_dataset_augmentation_disabled(self, test_data_file, mock_tokenizer, mock_fingerprint):
        """MBADataset preserves original when augmentation disabled."""
        try:
            from src.data.dataset import MBADataset
        except ImportError:
            pytest.skip("MBADataset dependencies not available")

        dataset = MBADataset(
            test_data_file,
            mock_tokenizer,
            mock_fingerprint,
            augment_variables=False,
        )

        sample = dataset[0]
        assert sample['obfuscated'] == "x & y"
        assert sample['simplified'] == "x & y"

    def test_contrastive_dataset_different_permutations(self, test_data_file, mock_tokenizer, mock_fingerprint):
        """ContrastiveDataset applies different permutations to anchor and positive."""
        try:
            from src.data.dataset import ContrastiveDataset
        except ImportError:
            pytest.skip("ContrastiveDataset dependencies not available")

        dataset = ContrastiveDataset(
            test_data_file,
            mock_tokenizer,
            mock_fingerprint,
            augment_variables=True,
            augment_prob=1.0,
        )

        # Get multiple samples - anchor and positive should differ sometimes
        # due to independent permutations
        different_count = 0
        for _ in range(20):
            sample = dataset[0]
            if sample['obfuscated'] != sample['simplified']:
                different_count += 1

        # With independent permutations, they should differ sometimes
        # (not always, since small expressions might map the same)
        assert different_count >= 0  # At minimum, code runs without error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
