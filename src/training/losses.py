"""
Loss functions for MBA Deobfuscator training pipeline.

Phase 1: InfoNCE + MaskLM (contrastive pretraining)
Phase 2: CE + Copy + Complexity (supervised)
Phase 3: PPO Policy + Value (reinforcement learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.constants import (
    PAD_IDX, EOS_IDX,
    INFONCE_TEMPERATURE,
    CE_WEIGHT, COMPLEXITY_WEIGHT, COPY_WEIGHT,
    PPO_EPSILON, PPO_VALUE_COEF, PPO_ENTROPY_COEF,
    REWARD_EQUIV_BONUS, REWARD_LEN_PENALTY, REWARD_DEPTH_PENALTY,
    REWARD_IDENTITY_PENALTY, REWARD_SYNTAX_ERROR_PENALTY,
    REWARD_SIMPLIFICATION_BONUS, REWARD_IDENTITY_THRESHOLD,
    NUM_NODE_TYPES,
)


# =============================================================================
# PHASE 1: CONTRASTIVE LOSSES
# =============================================================================

def infonce_loss(
    obf_embeddings: torch.Tensor,
    simp_embeddings: torch.Tensor,
    temperature: float = INFONCE_TEMPERATURE
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for learning expression equivalence.

    Pulls together embeddings of semantically equivalent expressions (obfuscated
    and simplified pairs) while pushing apart non-equivalent expressions.

    Args:
        obf_embeddings: [batch, hidden_dim] embeddings of obfuscated expressions
        simp_embeddings: [batch, hidden_dim] embeddings of simplified expressions
        temperature: Softmax temperature (default: 0.07)

    Returns:
        Scalar loss value

    Algorithm:
        For each obfuscated expression i:
        - Positive pair: (obf[i], simp[i]) - semantically equivalent
        - Negative pairs: (obf[i], simp[j]) for j != i - not equivalent

        Loss = -log(exp(sim(obf[i], simp[i])/τ) / Σ_j exp(sim(obf[i], simp[j])/τ))
    """
    # L2 normalize embeddings for cosine similarity
    # Without normalization, similarity degenerates to dot product causing
    # numerical instability (exp(large_value/τ) → Inf → NaN gradients)
    obf_norm = F.normalize(obf_embeddings, p=2, dim=-1)
    simp_norm = F.normalize(simp_embeddings, p=2, dim=-1)

    # Compute cosine similarity matrix: [batch, batch]
    # sim[i,j] = obf[i] · simp[j] (after normalization, this is cosine similarity)
    similarity = torch.matmul(obf_norm, simp_norm.T)

    # Scale by temperature
    logits = similarity / temperature

    # Labels: diagonal entries are positive pairs
    batch_size = obf_embeddings.shape[0]
    labels = torch.arange(batch_size, device=obf_embeddings.device)

    # Cross-entropy loss (log-softmax + NLL)
    loss = F.cross_entropy(logits, labels)

    return loss


def masklm_loss(
    node_embeddings: torch.Tensor,
    original_types: torch.Tensor,
    mask_indices: torch.Tensor,
    prediction_head: nn.Module
) -> torch.Tensor:
    """
    Masked language modeling loss for expression structure learning.

    Masks random nodes in AST graph and predicts their types. Forces encoder
    to learn structural patterns in MBA expressions.

    Args:
        node_embeddings: [total_nodes, hidden_dim] encoder outputs
        original_types: [num_masked] original node type IDs for masked nodes
        mask_indices: [num_masked] indices of masked nodes
        prediction_head: Linear layer mapping hidden_dim → NUM_NODE_TYPES

    Returns:
        Scalar cross-entropy loss
    """
    if mask_indices.numel() == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    # Extract embeddings at masked positions
    masked_embeddings = node_embeddings[mask_indices]  # [num_masked, hidden_dim]

    # Predict node types
    logits = prediction_head(masked_embeddings)  # [num_masked, NUM_NODE_TYPES]

    # Cross-entropy with original types
    loss = F.cross_entropy(logits, original_types)

    return loss


# =============================================================================
# PHASE 1B/1C: GMN LOSSES (Graph Matching Network)
# =============================================================================

def gmn_bce_loss(
    match_scores: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for GMN graph pair matching.

    This is the ONLY function that should handle sigmoid→logit conversion.
    All other code should call this function rather than reimplementing.

    Args:
        match_scores: [batch_size, 1] predicted match scores (sigmoid-activated, 0-1)
        labels: [batch_size] ground truth labels (1.0=equivalent, 0.0=not)
        pos_weight: Positive class weight (default: 1.0, increase if class imbalance)

    Returns:
        Scalar BCE loss
    """
    # CRITICAL: Clamp match_scores BEFORE logit conversion to prevent numerical issues
    # Sigmoid outputs can saturate to exact 0.0 or 1.0, causing logit to fail
    match_scores_clamped = match_scores.squeeze(-1).clamp(min=1e-7, max=1 - 1e-7)

    # Convert to logits (now safe because inputs are in (1e-7, 1-1e-7))
    logits = torch.logit(match_scores_clamped, eps=1e-7)

    # Weighted BCE with device-aware pos_weight
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    loss = loss_fn(logits, labels)

    return loss


def gmn_triplet_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet loss for GMN training (supplementary to BCE).

    Encourages: score(anchor, positive) > score(anchor, negative) + margin

    Args:
        positive_scores: [batch_size, 1] scores for (anchor, positive) pairs
        negative_scores: [batch_size, 1] scores for (anchor, negative) pairs
        margin: Triplet margin (default: 0.2)

    Returns:
        Scalar triplet loss
    """
    # Triplet loss: max(0, score(A,N) - score(A,P) + margin)
    # GMN outputs are similarity scores in [0, 1], so higher is more similar
    pos = positive_scores.squeeze(-1)
    neg = negative_scores.squeeze(-1)
    loss = F.relu(neg - pos + margin).mean()

    return loss


def gmn_combined_loss(
    match_scores: torch.Tensor,
    labels: torch.Tensor,
    triplet_data: Optional[Dict[str, torch.Tensor]] = None,
    pos_weight: float = 1.0,
    triplet_weight: float = 0.1,
    triplet_margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Combined GMN loss: BCE + optional triplet loss.

    Args:
        match_scores: [batch_size, 1] predicted match scores
        labels: [batch_size] ground truth labels
        triplet_data: Optional dict with keys:
            - 'positive_scores': [batch_size, 1]
            - 'negative_scores': [batch_size, 1]
        pos_weight: BCE positive weight
        triplet_weight: Weight for triplet loss term
        triplet_margin: Triplet margin

    Returns:
        Dict with keys:
            - 'total': Total loss
            - 'bce': BCE loss component
            - 'triplet': Triplet loss component (0 if not enabled)
    """
    # Use centralized BCE loss function
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


# =============================================================================
# PHASE 2: SUPERVISED LOSSES
# =============================================================================

def copy_loss(
    vocab_logits: torch.Tensor,
    copy_attn: torch.Tensor,
    p_gen: torch.Tensor,
    target_ids: torch.Tensor,
    source_tokens: torch.Tensor,
    pad_idx: int = PAD_IDX
) -> torch.Tensor:
    """
    Copy mechanism loss combining generation and copying.

    Learns when to generate from vocabulary vs. copy from source. Critical for
    preserving variable names in simplified expressions.

    Args:
        vocab_logits: [batch, tgt_len, vocab_size] vocabulary distribution logits
        copy_attn: [batch, tgt_len, src_len] attention over source tokens
        p_gen: [batch, tgt_len, 1] probability of generating vs. copying
        target_ids: [batch, tgt_len] target token IDs
        source_tokens: [batch, src_len] source token IDs
        pad_idx: Padding token ID to ignore

    Returns:
        Scalar loss value

    Algorithm:
        P(token) = p_gen * P_vocab(token) + (1 - p_gen) * Σ_j copy_attn[j] * 1[src[j] == token]
        Loss = -log P(target_token)
    """
    batch_size, tgt_len, vocab_size = vocab_logits.shape
    _, src_len = source_tokens.shape

    # Compute vocabulary distribution
    vocab_dist = F.softmax(vocab_logits, dim=-1)  # [batch, tgt_len, vocab_size]

    # Compute copy distribution using scatter_add
    # scatter_add accumulates probabilities for duplicate source tokens
    # scatter() would overwrite, producing incorrect distributions
    copy_dist = torch.zeros_like(vocab_dist)  # [batch, tgt_len, vocab_size]

    # Expand source tokens: [batch, src_len] -> [batch, tgt_len, src_len]
    source_expanded = source_tokens.unsqueeze(1).expand(-1, tgt_len, -1)

    # Scatter copy attention to vocabulary positions
    copy_dist.scatter_add_(dim=-1, index=source_expanded, src=copy_attn)

    # Squeeze p_gen if needed: [batch, tgt_len, 1] -> [batch, tgt_len]
    if p_gen.dim() == 3:
        p_gen = p_gen.squeeze(-1)

    # Interpolate distributions
    # final_dist = p_gen * vocab + (1 - p_gen) * copy
    p_gen_expanded = p_gen.unsqueeze(-1)  # [batch, tgt_len, 1]
    final_dist = p_gen_expanded * vocab_dist + (1 - p_gen_expanded) * copy_dist

    # Clamp to avoid log(0)
    final_dist = final_dist.clamp(min=1e-10)

    # Gather probabilities for target tokens
    target_probs = final_dist.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # Create mask for non-pad tokens
    mask = (target_ids != pad_idx).float()

    # Negative log likelihood
    nll = -torch.log(target_probs) * mask

    # Average over non-pad tokens
    loss = nll.sum() / mask.sum().clamp(min=1)

    return loss


def complexity_loss(
    length_pred: torch.Tensor,
    depth_pred: torch.Tensor,
    target_ids: torch.Tensor,
    depth_labels: torch.Tensor,
    length_weight: float = 0.5,
    depth_weight: float = 0.5,
    pad_idx: int = PAD_IDX,
    eos_idx: int = EOS_IDX
) -> torch.Tensor:
    """
    Loss for predicting output complexity (length and depth).

    Helps model learn to produce appropriately-sized simplified expressions.

    Args:
        length_pred: [batch, MAX_OUTPUT_LENGTH] length prediction logits
        depth_pred: [batch, MAX_OUTPUT_DEPTH] depth prediction logits
        target_ids: [batch, tgt_len] target token IDs (for computing actual length)
        depth_labels: [batch] target depth labels
        length_weight: Weight for length loss
        depth_weight: Weight for depth loss
        pad_idx: Padding token ID
        eos_idx: End-of-sequence token ID

    Returns:
        Scalar loss value
    """
    batch_size = target_ids.shape[0]
    device = target_ids.device

    # Compute actual target lengths (position of first EOS or pad)
    # Find first EOS token position for each sequence
    eos_mask = (target_ids == eos_idx)
    # If no EOS found, length is the full sequence length
    has_eos = eos_mask.any(dim=1)

    # Get position of first EOS (or sequence length if none)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        if has_eos[i]:
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                lengths[i] = eos_positions[0].item()
            else:
                # Fallback: count non-pad tokens
                lengths[i] = (target_ids[i] != pad_idx).sum().item()
        else:
            # Count non-pad tokens
            lengths[i] = (target_ids[i] != pad_idx).sum().item()

    # Clamp lengths to valid range for classification
    max_length_classes = length_pred.shape[1]
    lengths = lengths.clamp(0, max_length_classes - 1)

    # Length loss: cross-entropy
    length_loss = F.cross_entropy(length_pred, lengths)

    # Depth loss: cross-entropy
    max_depth_classes = depth_pred.shape[1]
    depth_labels = depth_labels.clamp(0, max_depth_classes - 1)
    depth_loss = F.cross_entropy(depth_pred, depth_labels)

    # Weighted sum
    total_loss = length_weight * length_loss + depth_weight * depth_loss

    return total_loss


def phase2_loss(
    vocab_logits: torch.Tensor,
    copy_attn: torch.Tensor,
    p_gen: torch.Tensor,
    length_pred: torch.Tensor,
    depth_pred: torch.Tensor,
    target_ids: torch.Tensor,
    source_tokens: torch.Tensor,
    depth_labels: torch.Tensor,
    ce_weight: float = CE_WEIGHT,
    complexity_weight: float = COMPLEXITY_WEIGHT,
    copy_weight: float = COPY_WEIGHT
) -> Dict[str, torch.Tensor]:
    """
    Combined loss for Phase 2 supervised training.

    Combines cross-entropy, copy mechanism, and complexity prediction losses.

    Args:
        vocab_logits: [batch, tgt_len, vocab_size] vocabulary logits
        copy_attn: [batch, tgt_len, src_len] copy attention weights
        p_gen: [batch, tgt_len, 1] generation probability
        length_pred: [batch, MAX_OUTPUT_LENGTH] length prediction
        depth_pred: [batch, MAX_OUTPUT_DEPTH] depth prediction
        target_ids: [batch, tgt_len] target token IDs
        source_tokens: [batch, src_len] source tokens for copy
        depth_labels: [batch] target depths
        ce_weight: Weight for cross-entropy loss
        complexity_weight: Weight for complexity loss
        copy_weight: Weight for copy mechanism loss

    Returns:
        Dict with 'total', 'ce', 'complexity', 'copy' losses
    """
    # Standard cross-entropy loss (without copy mechanism)
    # Shift logits and targets for autoregressive prediction
    shift_logits = vocab_logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_targets.view(-1),
        ignore_index=PAD_IDX
    )

    # Copy mechanism loss
    copy_loss_val = copy_loss(
        vocab_logits[:, :-1, :],
        copy_attn[:, :-1, :] if copy_attn is not None else None,
        p_gen[:, :-1, :] if p_gen is not None else None,
        shift_targets,
        source_tokens
    ) if copy_attn is not None and p_gen is not None else torch.tensor(0.0, device=vocab_logits.device)

    # Complexity loss
    complexity_loss_val = complexity_loss(
        length_pred, depth_pred, target_ids, depth_labels
    )

    # Weighted combination
    total = ce_weight * ce_loss + complexity_weight * complexity_loss_val + copy_weight * copy_loss_val

    return {
        'total': total,
        'ce': ce_loss,
        'complexity': complexity_loss_val,
        'copy': copy_loss_val,
    }


# =============================================================================
# PHASE 3: PPO LOSSES
# =============================================================================

def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = PPO_EPSILON
) -> torch.Tensor:
    """
    PPO clipped policy loss.

    Prevents destructively large policy updates by clipping the probability ratio.

    Args:
        log_probs: [batch, seq_len] current policy log probabilities
        old_log_probs: [batch, seq_len] old policy log probabilities (detached)
        advantages: [batch] advantage estimates
        epsilon: Clipping parameter (default: 0.2)

    Returns:
        Scalar policy loss (negated for gradient ascent)
    """
    # Compute probability ratio
    ratio = torch.exp(log_probs.sum(dim=-1) - old_log_probs.sum(dim=-1))

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # PPO objective: min(ratio * A, clip(ratio) * A)
    obj1 = ratio * advantages
    obj2 = clipped_ratio * advantages
    policy_loss = -torch.min(obj1, obj2).mean()

    return policy_loss


def ppo_value_loss(
    value_pred: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_value: bool = True,
    epsilon: float = PPO_EPSILON
) -> torch.Tensor:
    """
    PPO value function loss with optional clipping.

    Args:
        value_pred: [batch] predicted values
        returns: [batch] actual returns (rewards-to-go)
        old_values: [batch] old value predictions (for clipping)
        clip_value: Whether to clip value loss
        epsilon: Clipping parameter

    Returns:
        Scalar value loss
    """
    if clip_value and old_values is not None:
        # Clipped value loss
        value_clipped = old_values + torch.clamp(
            value_pred - old_values, -epsilon, epsilon
        )
        loss_unclipped = (value_pred - returns) ** 2
        loss_clipped = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
    else:
        # Simple MSE
        value_loss = 0.5 * F.mse_loss(value_pred, returns)

    return value_loss


def entropy_loss(log_probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Entropy bonus for encouraging exploration.

    Args:
        log_probs: [batch, seq_len, vocab_size] log probabilities
        mask: [batch, seq_len] mask for valid positions. Optional - when provided,
              entropy is computed only over masked positions. Currently not used
              by ppo_combined_loss but available for future extensions.

    Returns:
        Negative entropy (to be minimized, adding to total loss)
    """
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)

    if mask is not None:
        entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)
    else:
        entropy = entropy.mean()

    # Return negative because we want to maximize entropy
    return -entropy


def ppo_combined_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    value_pred: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    action_log_probs: Optional[torch.Tensor] = None,
    value_coef: float = PPO_VALUE_COEF,
    entropy_coef: float = PPO_ENTROPY_COEF,
    epsilon: float = PPO_EPSILON
) -> Dict[str, torch.Tensor]:
    """
    Combined PPO loss: policy + value + entropy.

    Args:
        log_probs: Current policy log probs
        old_log_probs: Old policy log probs (detached)
        value_pred: Value function prediction
        old_values: Old value predictions
        advantages: Advantage estimates
        returns: Computed returns
        action_log_probs: Full action log probs for entropy (optional)
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        epsilon: PPO clipping parameter

    Returns:
        Dict with 'total', 'policy', 'value', 'entropy' losses
    """
    policy_loss = ppo_policy_loss(log_probs, old_log_probs, advantages, epsilon)
    value_loss = ppo_value_loss(value_pred, returns, old_values, clip_value=True, epsilon=epsilon)

    # Entropy bonus (if action_log_probs provided)
    if action_log_probs is not None:
        ent_loss = entropy_loss(action_log_probs)
    else:
        ent_loss = torch.tensor(0.0, device=log_probs.device)

    total = policy_loss + value_coef * value_loss + entropy_coef * ent_loss

    return {
        'total': total,
        'policy': policy_loss,
        'value': value_loss,
        'entropy': ent_loss,
    }


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_reward(
    input_expr: str,
    output_expr: str,
    is_equivalent: bool,
    is_syntax_valid: bool,
    input_length: int,
    output_length: int,
    input_depth: int,
    output_depth: int
) -> float:
    """
    Compute reward for RL training.

    Rewards equivalence-preserving simplifications and penalizes:
    - Identity outputs (returning input unchanged)
    - Syntax errors
    - Longer/deeper outputs

    Args:
        input_expr: Original obfuscated expression
        output_expr: Model-generated simplified expression
        is_equivalent: Whether output is semantically equivalent to input
        is_syntax_valid: Whether output has valid syntax
        input_length: Token length of input
        output_length: Token length of output
        input_depth: AST depth of input
        output_depth: AST depth of output

    Returns:
        Scalar reward value
    """
    reward = 0.0

    # Syntax error penalty
    if not is_syntax_valid:
        return -REWARD_SYNTAX_ERROR_PENALTY

    # Equivalence bonus (major reward)
    if is_equivalent:
        reward += REWARD_EQUIV_BONUS

        # Length reduction bonus
        if output_length < input_length:
            length_reduction = (input_length - output_length) / input_length
            reward += REWARD_SIMPLIFICATION_BONUS * length_reduction

        # Depth reduction bonus
        if output_depth < input_depth:
            depth_reduction = (input_depth - output_depth) / input_depth
            reward += REWARD_SIMPLIFICATION_BONUS * depth_reduction

        # Length penalty (prefer shorter outputs)
        reward -= REWARD_LEN_PENALTY * output_length

        # Depth penalty (prefer shallower outputs)
        reward -= REWARD_DEPTH_PENALTY * output_depth

    else:
        # Non-equivalent output: small negative reward
        reward = -1.0

    # Identity penalty: penalize returning input unchanged
    # Use similarity threshold to catch near-identity outputs
    input_normalized = input_expr.replace(' ', '').lower()
    output_normalized = output_expr.replace(' ', '').lower()

    if input_normalized == output_normalized:
        reward -= REWARD_IDENTITY_PENALTY

    return reward


def compute_batch_rewards(
    inputs: list,
    outputs: list,
    equivalence_results: list,
    syntax_results: list,
    input_lengths: list,
    output_lengths: list,
    input_depths: list,
    output_depths: list
) -> torch.Tensor:
    """
    Compute rewards for a batch of (input, output) pairs.

    Args:
        inputs: List of input expressions
        outputs: List of output expressions
        equivalence_results: List of bool for semantic equivalence
        syntax_results: List of bool for syntax validity
        input_lengths: List of input token lengths
        output_lengths: List of output token lengths
        input_depths: List of input AST depths
        output_depths: List of output AST depths

    Returns:
        [batch] tensor of rewards
    """
    rewards = []
    for i in range(len(inputs)):
        r = compute_reward(
            inputs[i], outputs[i],
            equivalence_results[i], syntax_results[i],
            input_lengths[i], output_lengths[i],
            input_depths[i], output_depths[i]
        )
        rewards.append(r)

    return torch.tensor(rewards, dtype=torch.float32)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    For single-step episodes (typical in MBA simplification), this simplifies to:
    advantage = reward - value

    Args:
        rewards: [batch] rewards
        values: [batch] value predictions
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    # For single-step episodes, advantage = reward - value
    # Returns = reward (no future rewards in single-step)
    advantages = rewards - values.detach()
    returns = rewards.clone()

    # Normalize advantages for stability
    # Only normalize if there's meaningful variance to prevent division issues
    if advantages.numel() > 1:
        std = advantages.std()
        if std > 1e-6:
            advantages = (advantages - advantages.mean()) / std
        # else: keep unnormalized advantages (uniform rewards case)

    return advantages, returns
