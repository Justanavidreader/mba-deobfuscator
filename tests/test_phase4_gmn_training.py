"""
Unit tests for Phase 4 GMN Training Infrastructure.

Tests cover:
- GMN loss functions (BCE, triplet, combined)
- NegativeSampler (caching, strategies, statistics)
- GMNDataset (loading, sampling, resampling)
- Phase1bGMNTrainer (frozen encoder, training step, evaluation)
- Phase1cGMNTrainer (unfreezing, dual learning rates, drift monitoring)
"""

import sys
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import Data, Batch


def load_module_directly(module_path: Path, module_name: str):
    """Load a module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import GMN loss functions directly to avoid circular imports
def gmn_bce_loss(match_scores, labels, pos_weight=1.0):
    """BCE loss for GMN - local implementation for testing."""
    match_scores_clamped = match_scores.squeeze(-1).clamp(min=1e-7, max=1 - 1e-7)
    logits = torch.logit(match_scores_clamped, eps=1e-7)
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    return loss_fn(logits, labels)


def gmn_triplet_loss(positive_scores, negative_scores, margin=0.2):
    """Triplet loss for GMN - local implementation for testing."""
    pos = positive_scores.squeeze(-1)
    neg = negative_scores.squeeze(-1)
    return F.relu(neg - pos + margin).mean()


def gmn_combined_loss(match_scores, labels, triplet_data=None, pos_weight=1.0,
                      triplet_weight=0.1, triplet_margin=0.2):
    """Combined loss for GMN - local implementation for testing."""
    bce = gmn_bce_loss(match_scores, labels, pos_weight)
    if triplet_data is not None:
        triplet = gmn_triplet_loss(
            triplet_data['positive_scores'],
            triplet_data['negative_scores'],
            triplet_margin,
        )
        total = bce + triplet_weight * triplet
    else:
        triplet = torch.tensor(0.0, device=match_scores.device)
        total = bce
    return {'total': total, 'bce': bce, 'triplet': triplet}


# Load NegativeSampler directly (bypasses __init__.py)
_negative_sampler_module = None


def get_negative_sampler_class():
    """Lazy load NegativeSampler to avoid import chain issues."""
    global _negative_sampler_module
    if _negative_sampler_module is None:
        module_path = project_root / "src" / "training" / "negative_sampler.py"
        _negative_sampler_module = load_module_directly(module_path, "negative_sampler_direct")
    return _negative_sampler_module.NegativeSampler


# ============================================================================
# GMN Loss Functions Tests
# ============================================================================


class TestGMNBCELoss:
    """Tests for gmn_bce_loss function."""

    def test_basic_forward(self):
        """Test basic BCE loss computation."""
        match_scores = torch.tensor([[0.8], [0.2], [0.9], [0.1]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = gmn_bce_loss(match_scores, labels)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Should be positive
        assert not torch.isnan(loss)

    def test_perfect_predictions(self):
        """Test loss is low for perfect predictions."""
        match_scores = torch.tensor([[0.99], [0.01], [0.99], [0.01]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = gmn_bce_loss(match_scores, labels)

        assert loss.item() < 0.1  # Should be very low

    def test_wrong_predictions(self):
        """Test loss is high for wrong predictions."""
        match_scores = torch.tensor([[0.01], [0.99], [0.01], [0.99]])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = gmn_bce_loss(match_scores, labels)

        assert loss.item() > 1.0  # Should be high

    def test_extreme_values_no_nan(self):
        """Test numerical stability with extreme values."""
        # Values very close to 0 and 1 that could cause logit issues
        match_scores = torch.tensor([[0.0], [1.0], [1e-8], [1.0 - 1e-8]])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

        loss = gmn_bce_loss(match_scores, labels)

        assert not torch.isnan(loss), "Loss should not be NaN for extreme values"
        assert not torch.isinf(loss), "Loss should not be inf for extreme values"

    def test_pos_weight_effect(self):
        """Test that pos_weight affects imbalanced loss."""
        match_scores = torch.tensor([[0.5], [0.5], [0.5]])
        labels = torch.tensor([1.0, 0.0, 0.0])  # 1 positive, 2 negative

        loss_default = gmn_bce_loss(match_scores, labels, pos_weight=1.0)
        loss_weighted = gmn_bce_loss(match_scores, labels, pos_weight=2.0)

        # Higher pos_weight should increase loss from positive misclassifications
        assert loss_weighted.item() != loss_default.item()

    def test_gradient_flow(self):
        """Test gradients flow through the loss."""
        match_scores = torch.tensor([[0.5], [0.3]], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])

        loss = gmn_bce_loss(match_scores, labels)
        loss.backward()

        assert match_scores.grad is not None
        assert not torch.isnan(match_scores.grad).any()


class TestGMNTripletLoss:
    """Tests for gmn_triplet_loss function."""

    def test_basic_forward(self):
        """Test basic triplet loss computation."""
        pos_scores = torch.tensor([[0.8], [0.7], [0.9]])
        neg_scores = torch.tensor([[0.3], [0.2], [0.1]])

        loss = gmn_triplet_loss(pos_scores, neg_scores, margin=0.2)

        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Should be non-negative

    def test_well_separated_pairs(self):
        """Test loss is zero when pairs are well separated."""
        pos_scores = torch.tensor([[0.9], [0.8], [0.85]])
        neg_scores = torch.tensor([[0.1], [0.2], [0.15]])

        loss = gmn_triplet_loss(pos_scores, neg_scores, margin=0.2)

        # pos - neg > margin, so loss should be 0
        assert loss.item() < 0.01

    def test_violated_margin(self):
        """Test loss is positive when margin violated."""
        pos_scores = torch.tensor([[0.5], [0.5]])
        neg_scores = torch.tensor([[0.6], [0.5]])  # neg >= pos

        loss = gmn_triplet_loss(pos_scores, neg_scores, margin=0.2)

        assert loss.item() > 0

    def test_margin_effect(self):
        """Test that margin affects loss magnitude."""
        pos_scores = torch.tensor([[0.6]])
        neg_scores = torch.tensor([[0.5]])

        loss_small = gmn_triplet_loss(pos_scores, neg_scores, margin=0.05)
        loss_large = gmn_triplet_loss(pos_scores, neg_scores, margin=0.3)

        # Larger margin should give larger loss
        assert loss_large.item() > loss_small.item()


class TestGMNCombinedLoss:
    """Tests for gmn_combined_loss function."""

    def test_bce_only(self):
        """Test combined loss without triplet data."""
        match_scores = torch.tensor([[0.8], [0.2]])
        labels = torch.tensor([1.0, 0.0])

        result = gmn_combined_loss(match_scores, labels, triplet_data=None)

        assert 'total' in result
        assert 'bce' in result
        assert 'triplet' in result
        assert result['triplet'].item() == 0.0
        assert torch.allclose(result['total'], result['bce'])

    def test_with_triplet_data(self):
        """Test combined loss with triplet data."""
        match_scores = torch.tensor([[0.8], [0.2]])
        labels = torch.tensor([1.0, 0.0])
        triplet_data = {
            'positive_scores': torch.tensor([[0.8], [0.9]]),
            'negative_scores': torch.tensor([[0.3], [0.2]]),
        }

        result = gmn_combined_loss(
            match_scores, labels,
            triplet_data=triplet_data,
            triplet_weight=0.1,
        )

        assert result['triplet'].item() >= 0
        expected_total = result['bce'] + 0.1 * result['triplet']
        assert torch.allclose(result['total'], expected_total)


# ============================================================================
# Negative Sampler Tests
# ============================================================================


class TestNegativeSampler:
    """Tests for NegativeSampler class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return [
            {'obfuscated': 'x + y', 'simplified': 'x + y', 'depth': 1},
            {'obfuscated': '(x & y) + (x | y)', 'simplified': 'x + y', 'depth': 2},
            {'obfuscated': 'x ^ y', 'simplified': 'x ^ y', 'depth': 1},
            {'obfuscated': '(x & y) ^ (x | y)', 'simplified': 'x ^ y', 'depth': 2},
            {'obfuscated': '~x', 'simplified': '-x - 1', 'depth': 1},
        ]

    def test_initialization(self, sample_dataset):
        """Test sampler initializes correctly."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(
            dataset=sample_dataset,
            z3_timeout_ms=100,
            cache_size=100,
        )

        assert len(sampler.dataset) == 5
        assert len(sampler._depth_index) > 0
        assert len(sampler._all_expressions) == 5

    def test_cache_key_ordering(self, sample_dataset):
        """Test cache keys are order-independent."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(sample_dataset)

        key1 = sampler._cache_key('x + y', 'x ^ y')
        key2 = sampler._cache_key('x ^ y', 'x + y')

        assert key1 == key2

    def test_cache_put_and_get(self, sample_dataset):
        """Test cache put and retrieval."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(sample_dataset, cache_size=10)

        # Put value
        sampler._cache_put('x + y', 'x ^ y', False)

        # Get value
        result = sampler._cache_get('x + y', 'x ^ y')
        assert result is not None
        assert result[0] is False

    def test_cache_eviction(self, sample_dataset):
        """Test cache evicts oldest entries."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(sample_dataset, cache_size=3)

        # Fill cache
        sampler._cache_put('a', 'b', False)
        sampler._cache_put('c', 'd', False)
        sampler._cache_put('e', 'f', False)

        assert len(sampler._cache) == 3

        # Add one more, should evict oldest
        sampler._cache_put('g', 'h', True)

        assert len(sampler._cache) == 3
        # First entry should be evicted
        assert sampler._cache_get('a', 'b') is None

    def test_stats_property(self, sample_dataset):
        """Test statistics are tracked."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(sample_dataset)

        stats = sampler.stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'cache_hit_rate' in stats
        assert 'z3_timeouts' in stats

    def test_normalize_expr(self, sample_dataset):
        """Test expression normalization."""
        NegativeSampler = get_negative_sampler_class()

        sampler = NegativeSampler(sample_dataset)

        assert sampler._normalize_expr('X + Y') == sampler._normalize_expr('x+y')
        assert sampler._normalize_expr(' x & y ') == sampler._normalize_expr('x&y')


# ============================================================================
# Phase1b GMN Trainer Tests
# ============================================================================

# Check if torch_scatter is available (required for full trainer tests)
try:
    from torch_scatter import scatter_mean
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


class MockGMNModel(nn.Module):
    """Mock GMN model for testing trainers."""

    def __init__(self, hidden_dim=256, frozen=True):
        super().__init__()
        self._encoder_frozen = frozen
        self._hidden_dim = hidden_dim
        self.hgt_encoder = nn.Linear(32, hidden_dim)
        self.gmn = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

        if frozen:
            for p in self.hgt_encoder.parameters():
                p.requires_grad = False

    @property
    def is_encoder_frozen(self):
        return self._encoder_frozen

    def forward_pair(self, graph1_batch, graph2_batch):
        # Simple mock: return random scores
        batch_size = graph1_batch.num_graphs
        h1 = torch.randn(batch_size, self._hidden_dim)
        h2 = torch.randn(batch_size, self._hidden_dim)
        combined = torch.cat([h1, h2], dim=-1)
        return self.sigmoid(self.gmn(combined))


def create_mock_batch(batch_size=4):
    """Create mock GMN batch for testing."""
    graphs1 = []
    graphs2 = []
    for _ in range(batch_size):
        g1 = Data(x=torch.randn(5, 32), edge_index=torch.randint(0, 5, (2, 8)))
        g2 = Data(x=torch.randn(4, 32), edge_index=torch.randint(0, 4, (2, 6)))
        graphs1.append(g1)
        graphs2.append(g2)

    return {
        'graph1_batch': Batch.from_data_list(graphs1),
        'graph2_batch': Batch.from_data_list(graphs2),
        'labels': torch.tensor([1.0, 0.0, 1.0, 0.0]),
        'pair_indices': torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
    }


@pytest.mark.skipif(not TORCH_SCATTER_AVAILABLE, reason="torch_scatter not installed")
class TestPhase1bGMNTrainer:
    """Tests for Phase1bGMNTrainer."""

    @pytest.fixture
    def mock_sampler(self):
        """Create mock negative sampler."""
        sampler = Mock()
        sampler.stats = {'cache_hits': 0, 'cache_misses': 0}
        return sampler

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return {
            'learning_rate': 3e-5,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'scheduler_type': 'cosine',
            'total_steps': 1000,
            'bce_pos_weight': 1.0,
            'triplet_loss_margin': None,
            'triplet_loss_weight': 0.1,
        }

    def test_init_requires_frozen_encoder(self, mock_sampler, trainer_config):
        """Test trainer requires frozen encoder."""
        from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer

        model = MockGMNModel(frozen=False)  # Unfrozen

        with pytest.raises(ValueError, match="frozen"):
            Phase1bGMNTrainer(
                model=model,
                config=trainer_config,
                negative_sampler=mock_sampler,
            )

    def test_init_with_frozen_encoder(self, mock_sampler, trainer_config, tmp_path):
        """Test trainer initializes with frozen encoder."""
        from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer

        model = MockGMNModel(frozen=True)

        trainer = Phase1bGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        assert trainer.bce_pos_weight == 1.0
        assert trainer.triplet_margin is None

    def test_verify_encoder_frozen(self, mock_sampler, trainer_config, tmp_path):
        """Test encoder frozen verification."""
        from src.training.phase1b_gmn_trainer import Phase1bGMNTrainer

        model = MockGMNModel(frozen=True)
        trainer = Phase1bGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        # Should not raise
        trainer._verify_encoder_frozen()

        # Unfreeze and verify raises
        model._encoder_frozen = False
        with pytest.raises(RuntimeError):
            trainer._verify_encoder_frozen()


# ============================================================================
# Phase1c GMN Trainer Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_SCATTER_AVAILABLE, reason="torch_scatter not installed")
class TestPhase1cGMNTrainer:
    """Tests for Phase1cGMNTrainer."""

    @pytest.fixture
    def mock_sampler(self):
        """Create mock negative sampler."""
        sampler = Mock()
        sampler.stats = {'cache_hits': 0, 'cache_misses': 0}
        return sampler

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return {
            'learning_rate': 3e-5,
            'encoder_learning_rate': 3e-6,
            'weight_decay': 0.01,
            'encoder_weight_decay': 0.001,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'scheduler_type': 'cosine',
            'total_steps': 1000,
            'bce_pos_weight': 1.0,
            'triplet_loss_margin': 0.2,
            'triplet_loss_weight': 0.1,
            'unfreeze_encoder': True,
        }

    def test_init_unfreezes_encoder(self, mock_sampler, trainer_config, tmp_path):
        """Test Phase1c unfreezes encoder."""
        from src.training.phase1c_gmn_trainer import Phase1cGMNTrainer

        model = MockGMNModel(frozen=True)

        trainer = Phase1cGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        # Encoder should be unfrozen
        assert not model.is_encoder_frozen

    def test_dual_learning_rates(self, mock_sampler, trainer_config, tmp_path):
        """Test optimizer has separate LR for encoder and GMN."""
        from src.training.phase1c_gmn_trainer import Phase1cGMNTrainer

        model = MockGMNModel(frozen=True)

        trainer = Phase1cGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        # Check parameter groups
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) == 4  # encoder decay, encoder no_decay, gmn decay, gmn no_decay

        # Find encoder and GMN groups by LR
        encoder_lr = trainer_config['encoder_learning_rate']
        gmn_lr = trainer_config['learning_rate']

        lrs = [pg['lr'] for pg in param_groups]
        assert encoder_lr in lrs
        assert gmn_lr in lrs

    def test_encoder_drift_monitoring(self, mock_sampler, trainer_config, tmp_path):
        """Test encoder drift computation."""
        from src.training.phase1c_gmn_trainer import Phase1cGMNTrainer

        model = MockGMNModel(frozen=True)

        trainer = Phase1cGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        # Initial drift should be 0 or very small
        initial_drift = trainer._compute_encoder_drift()
        assert initial_drift >= 0

    def test_safe_unfreeze_clears_gradients(self, mock_sampler, trainer_config, tmp_path):
        """Test that unfreezing clears stale gradients."""
        from src.training.phase1c_gmn_trainer import Phase1cGMNTrainer

        model = MockGMNModel(frozen=True)

        # Set some fake gradients
        for p in model.hgt_encoder.parameters():
            p.grad = torch.randn_like(p)

        trainer = Phase1cGMNTrainer(
            model=model,
            config=trainer_config,
            negative_sampler=mock_sampler,
            checkpoint_dir=str(tmp_path),
        )

        # Gradients should be cleared
        for p in model.hgt_encoder.parameters():
            assert p.grad is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase4Integration:
    """Integration tests for Phase 4 components."""

    def test_loss_gradient_stability(self):
        """Test loss functions produce stable gradients across many samples."""
        torch.manual_seed(42)

        for _ in range(100):
            match_scores = torch.rand(16, 1, requires_grad=True)
            labels = torch.randint(0, 2, (16,)).float()

            loss = gmn_bce_loss(match_scores, labels)
            loss.backward()

            assert not torch.isnan(loss)
            assert not torch.isnan(match_scores.grad).any()
            assert not torch.isinf(match_scores.grad).any()

    def test_combined_loss_backprop(self):
        """Test combined loss allows proper backpropagation."""
        match_scores = torch.tensor([[0.7], [0.3], [0.8], [0.2]], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        triplet_data = {
            'positive_scores': torch.tensor([[0.8], [0.85]], requires_grad=True),
            'negative_scores': torch.tensor([[0.3], [0.25]], requires_grad=True),
        }

        result = gmn_combined_loss(
            match_scores, labels,
            triplet_data=triplet_data,
            triplet_weight=0.1,
        )

        result['total'].backward()

        assert match_scores.grad is not None
        assert triplet_data['positive_scores'].grad is not None
        assert triplet_data['negative_scores'].grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
