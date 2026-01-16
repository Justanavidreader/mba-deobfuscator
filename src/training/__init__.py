"""Training modules for MBA Deobfuscator."""

from src.training.losses import (
    infonce_loss,
    masklm_loss,
    copy_loss,
    complexity_loss,
    phase2_loss,
    ppo_policy_loss,
    ppo_value_loss,
    ppo_combined_loss,
    compute_reward,
    compute_batch_rewards,
    compute_advantages,
)
from src.training.base_trainer import BaseTrainer
from src.training.phase1_trainer import Phase1Trainer
from src.training.phase2_trainer import Phase2Trainer
from src.training.phase3_trainer import Phase3Trainer

__all__ = [
    # Loss functions
    'infonce_loss',
    'masklm_loss',
    'copy_loss',
    'complexity_loss',
    'phase2_loss',
    'ppo_policy_loss',
    'ppo_value_loss',
    'ppo_combined_loss',
    'compute_reward',
    'compute_batch_rewards',
    'compute_advantages',
    # Trainers
    'BaseTrainer',
    'Phase1Trainer',
    'Phase2Trainer',
    'Phase3Trainer',
]
