"""
Phase 3 Trainer: Reinforcement Learning with PPO.

Fine-tunes the model using:
1. PPO policy loss: Learn to generate better simplifications
2. PPO value loss: Learn to predict expected rewards
3. Entropy bonus: Encourage exploration

Rewards:
- Equivalence preservation (verified via execution/Z3)
- Length/depth reduction
- Anti-identity penalty (don't just return input)
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.losses import (
    ppo_combined_loss,
    compute_reward,
    compute_batch_rewards,
    compute_advantages,
)
from src.data.tokenizer import MBATokenizer
from src.data.fingerprint import SemanticFingerprint
from src.data.ast_parser import expr_to_ast_depth
from src.inference.verify import ThreeTierVerifier
from src.constants import (
    PPO_EPSILON, PPO_VALUE_COEF, PPO_ENTROPY_COEF,
    SOS_IDX, EOS_IDX, PAD_IDX,
    MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


class Phase3Trainer(BaseTrainer):
    """
    PPO-based RL fine-tuning trainer.

    Uses the pre-trained model as a policy and fine-tunes it
    using rewards based on equivalence verification and simplification.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: MBATokenizer,
        fingerprint: SemanticFingerprint,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Phase 3 trainer.

        Args:
            model: MBADeobfuscator model (must have value_head)
            tokenizer: MBATokenizer for encoding/decoding
            fingerprint: SemanticFingerprint for computing fingerprints
            config: Training configuration with:
                - ppo_epsilon: float (default: 0.2)
                - value_coef: float (default: 0.5)
                - entropy_coef: float (default: 0.01)
                - ppo_epochs: int (default: 4)
                - verify_exec_only: bool (default: True, skip Z3 for speed)
            device: Training device
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(
            model=model,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase3_rl",
        )

        self.tokenizer = tokenizer
        self.fingerprint = fingerprint

        # PPO hyperparameters
        self.ppo_epsilon = config.get("ppo_epsilon", PPO_EPSILON)
        self.value_coef = config.get("value_coef", PPO_VALUE_COEF)
        self.entropy_coef = config.get("entropy_coef", PPO_ENTROPY_COEF)
        self.ppo_epochs = config.get("ppo_epochs", 4)

        # Verification settings
        self.verify_exec_only = config.get("verify_exec_only", True)
        self.verifier = ThreeTierVerifier(tokenizer)

        # Sampling temperature
        self.sample_temperature = config.get("sample_temperature", 1.0)

        logger.info(
            f"Phase3Trainer initialized: "
            f"epsilon={self.ppo_epsilon}, value_coef={self.value_coef}, "
            f"entropy_coef={self.entropy_coef}, ppo_epochs={self.ppo_epochs}"
        )

    def collect_rollout(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout data by sampling from policy.

        Args:
            batch: Batch from MBADataset

        Returns:
            Rollout data with:
                - sampled_outputs: [batch, seq_len] sampled token IDs
                - log_probs: [batch, seq_len] log probabilities
                - values: [batch] value predictions
                - rewards: [batch] computed rewards
        """
        self.model.eval()

        # Move to device
        graph_batch = batch['graph_batch'].to(self.device)
        fingerprint = batch['fingerprint'].to(self.device)

        with torch.no_grad():
            # Encode
            memory = self.model.encode(graph_batch, fingerprint)
            batch_size = memory.shape[0]

            # Sample sequences
            sampled_outputs, log_probs = self._sample_sequences(memory)

            # Get value predictions
            # Use mean-pooled memory as input to value head
            memory_pooled = memory.mean(dim=1)  # [batch, d_model]
            values = self.model.value_head(memory_pooled).squeeze(-1)  # [batch]

        # Compute rewards
        inputs = batch['obfuscated']
        outputs = self._decode_outputs(sampled_outputs)

        rewards = self._compute_rewards(inputs, outputs)

        return {
            'graph_batch': graph_batch,
            'fingerprint': fingerprint,
            'sampled_outputs': sampled_outputs,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'inputs': inputs,
            'outputs': outputs,
        }

    def _sample_sequences(
        self,
        memory: torch.Tensor,
        max_len: int = MAX_SEQ_LEN
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences from policy with temperature.

        Args:
            memory: [batch, mem_len, d_model] encoder output
            max_len: Maximum sequence length

        Returns:
            sampled_outputs: [batch, seq_len] token IDs
            log_probs: [batch, seq_len] log probabilities of sampled tokens
        """
        batch_size = memory.shape[0]

        # Start with SOS
        output = torch.full(
            (batch_size, 1), SOS_IDX, dtype=torch.long, device=self.device
        )
        all_log_probs = []

        for _ in range(max_len - 1):
            # Decode
            decoder_out = self.model.decode(output, memory)

            # Get logits
            if isinstance(decoder_out, dict):
                logits = decoder_out['vocab_logits'][:, -1, :]
            else:
                logits = self.model.vocab_head(decoder_out[:, -1, :])

            # Apply temperature
            logits = logits / self.sample_temperature

            # Sample
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)

            output = torch.cat([output, next_token.unsqueeze(1)], dim=1)
            all_log_probs.append(log_prob)

            # Check if all sequences have EOS
            if (next_token == EOS_IDX).all():
                break

        # Pad log_probs to same length as output
        log_probs = torch.stack(all_log_probs, dim=1)  # [batch, seq_len-1]

        # Pad with zeros for SOS position
        log_probs = torch.cat([
            torch.zeros(batch_size, 1, device=self.device),
            log_probs
        ], dim=1)

        return output, log_probs

    def _decode_outputs(self, sampled_outputs: torch.Tensor) -> List[str]:
        """Decode sampled outputs to strings."""
        outputs = []
        for i in range(sampled_outputs.shape[0]):
            tokens = sampled_outputs[i].tolist()
            output_str = self.tokenizer.decode(tokens)
            outputs.append(output_str)
        return outputs

    def _compute_rewards(
        self,
        inputs: List[str],
        outputs: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for (input, output) pairs.

        Uses verification to check equivalence.
        """
        batch_size = len(inputs)

        # Verify outputs
        equivalence_results = []
        syntax_results = []
        input_lengths = []
        output_lengths = []
        input_depths = []
        output_depths = []

        for inp, out in zip(inputs, outputs):
            # Check syntax
            try:
                out_tokens = self.tokenizer.encode(out)
                syntax_valid = len(out_tokens) > 2
            except Exception:
                syntax_valid = False

            syntax_results.append(syntax_valid)

            # Check equivalence (execution only for speed)
            # verify_batch can return None or empty list on failure
            if syntax_valid and self.verify_exec_only:
                result = self.verifier.verify_batch(inp, [out])
                if result and len(result) > 0:
                    equiv = result[0].exec_valid
                else:
                    equiv = False
            elif syntax_valid:
                result = self.verifier.verify_batch(inp, [out])
                if result and len(result) > 0:
                    equiv = result[0].z3_verified or result[0].exec_valid
                else:
                    equiv = False
            else:
                equiv = False

            equivalence_results.append(equiv)

            # Compute lengths
            inp_tokens = self.tokenizer.encode(inp)
            input_lengths.append(len(inp_tokens))
            output_lengths.append(len(out_tokens) if syntax_valid else len(inp_tokens) + 10)

            # Compute depths
            try:
                input_depths.append(expr_to_ast_depth(inp))
            except Exception:
                input_depths.append(5)

            try:
                output_depths.append(expr_to_ast_depth(out) if syntax_valid else 10)
            except Exception:
                output_depths.append(10)

        # Compute rewards
        rewards = compute_batch_rewards(
            inputs, outputs,
            equivalence_results, syntax_results,
            input_lengths, output_lengths,
            input_depths, output_depths
        )

        return rewards.to(self.device)

    def _ppo_update_step(
        self,
        rollout: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single PPO update epoch.

        Returns dict with policy, value, entropy losses.
        """
        # Re-encode (gradients needed)
        memory = self.model.encode(rollout['graph_batch'], rollout['fingerprint'])

        # Re-compute log probs and values
        new_log_probs = self._compute_log_probs(memory, rollout['sampled_outputs'])
        memory_pooled = memory.mean(dim=1)
        new_values = self.model.value_head(memory_pooled).squeeze(-1)

        # PPO loss
        loss_dict = ppo_combined_loss(
            log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            value_pred=new_values,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            epsilon=self.ppo_epsilon,
        )

        # Backward
        self.backward(loss_dict['total'], update=True)

        return {
            'policy': loss_dict['policy'].item(),
            'value': loss_dict['value'].item(),
            'entropy': loss_dict['entropy'].item(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single PPO training step.

        1. Collect rollout (sample from policy)
        2. Compute advantages
        3. Update policy with PPO loss (multiple epochs)

        Args:
            batch: Batch from MBADataset

        Returns:
            Dict with loss values and reward statistics
        """
        # Collect rollout
        rollout = self.collect_rollout(batch)

        # Compute advantages
        advantages, returns = compute_advantages(rollout['rewards'], rollout['values'])

        # Store old log probs and values (detached)
        old_log_probs = rollout['log_probs'].detach()
        old_values = rollout['values'].detach()

        # PPO update (multiple epochs on same batch)
        self.model.train()
        losses = [
            self._ppo_update_step(rollout, old_log_probs, old_values, advantages, returns)
            for _ in range(self.ppo_epochs)
        ]

        # Average losses over PPO epochs
        avg_policy_loss = sum(l['policy'] for l in losses) / self.ppo_epochs
        avg_value_loss = sum(l['value'] for l in losses) / self.ppo_epochs
        avg_entropy_loss = sum(l['entropy'] for l in losses) / self.ppo_epochs

        # Reward statistics
        rewards = rollout['rewards']

        return {
            'total': avg_policy_loss + self.value_coef * avg_value_loss,
            'policy': avg_policy_loss,
            'value': avg_value_loss,
            'entropy': avg_entropy_loss,
            'reward': rewards.mean().item(),
            'equiv_rate': (rewards > 0).float().mean().item(),
        }

    def _compute_log_probs(
        self,
        memory: torch.Tensor,
        sampled_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Re-compute log probabilities for sampled outputs.

        Used during PPO update to get gradients.
        """
        batch_size, seq_len = sampled_outputs.shape
        all_log_probs = []

        # Forward pass through decoder
        for t in range(1, seq_len):
            # Input: tokens up to t-1
            decoder_input = sampled_outputs[:, :t]

            decoder_out = self.model.decode(decoder_input, memory)

            if isinstance(decoder_out, dict):
                logits = decoder_out['vocab_logits'][:, -1, :]
            else:
                logits = self.model.vocab_head(decoder_out[:, -1, :])

            logits = logits / self.sample_temperature

            # Get log prob of actual token at position t
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1,
                index=sampled_outputs[:, t:t+1]
            ).squeeze(-1)

            all_log_probs.append(token_log_probs)

        # Stack and add zero for first position
        log_probs = torch.stack(all_log_probs, dim=1)
        log_probs = torch.cat([
            torch.zeros(batch_size, 1, device=self.device),
            log_probs
        ], dim=1)

        return log_probs

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate RL model.

        Args:
            dataloader: Validation dataloader

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        total_reward = 0.0
        total_equiv = 0
        total_syntax_valid = 0
        total_samples = 0

        for batch in dataloader:
            # Collect rollout (no gradient)
            rollout = self.collect_rollout(batch)

            batch_size = len(rollout['inputs'])
            total_samples += batch_size

            # Reward statistics
            rewards = rollout['rewards']
            total_reward += rewards.sum().item()
            total_equiv += (rewards > 0).sum().item()

            # Check syntax validity
            for out in rollout['outputs']:
                try:
                    tokens = self.tokenizer.encode(out)
                    if len(tokens) > 2:
                        total_syntax_valid += 1
                except Exception:
                    pass

        return {
            'avg_reward': total_reward / max(total_samples, 1),
            'equiv_rate': total_equiv / max(total_samples, 1),
            'syntax_valid': total_syntax_valid / max(total_samples, 1),
        }

    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save Phase 3 checkpoint."""
        if filename is None:
            filename = f"phase3_checkpoint_{self.global_step}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved Phase 3 checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "phase3_best.pt"
            torch.save(checkpoint, best_path)

        return str(checkpoint_path)
